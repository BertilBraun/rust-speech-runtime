use bytes::Bytes;
use tokio::sync::{mpsc, oneshot};
use tokio_util::sync::CancellationToken;

use crate::{
    protocol::{ErrorCode, SessionEvent, SessionId, SessionRecord, TurnId},
    session::manager::ManagerCommand,
    worker::{Command, WorkerHandle},
};

use super::RuntimeError;

/// An admitted conversation, permanently bound to one worker.
///
/// For each turn: begin, append ordered audio, optionally prepare, then commit and
/// consume events. Command success acknowledges worker-side validation, not completed
/// inference. Consume events throughout the session: bounded output stops slow consumers.
/// Close explicitly to retrieve the archive record; dropping the handle only cancels.
pub struct SessionHandle {
    pub(crate) key: String,
    pub(crate) worker_id: usize,
    pub(crate) worker: WorkerHandle,
    pub(crate) events: mpsc::Receiver<SessionEvent>,
    pub(crate) cancellation: CancellationToken,
    pub(crate) manager: mpsc::Sender<ManagerCommand>,
    pub(crate) session_id: SessionId,
}

impl SessionHandle {
    /// Returns the worker selected at admission; it does not change across turns.
    pub fn worker_id(&self) -> usize {
        self.worker_id
    }

    /// Reserves compute capacity and begins capture using a session-unique turn ID.
    ///
    /// A successfully admitted new turn interrupts an active response while retaining
    /// already accepted text. Rejection leaves that response intact.
    pub async fn begin_turn(&self, turn_id: TurnId) -> Result<(), RuntimeError> {
        self.request(|reply| Command::Begin {
            key: self.key.clone(),
            turn_id,
            reply,
        })
        .await
    }

    /// Appends nonempty mono PCM16 little-endian audio sampled at 16 kHz.
    ///
    /// Chunk indices start at zero and must be contiguous within the capturing turn.
    /// More audio invalidates any prepared candidate. Byte lengths must contain whole
    /// samples; the final packet may be shorter than the usual 100 ms.
    pub async fn audio(
        &self,
        turn_id: TurnId,
        chunk_index: u32,
        audio: Bytes,
    ) -> Result<(), RuntimeError> {
        self.request(|reply| Command::Audio {
            key: self.key.clone(),
            turn_id,
            chunk_index,
            audio,
            reply,
        })
        .await
    }

    /// Prepares the current candidate while end-of-turn confirmation continues.
    ///
    /// Counts must match all accepted audio. Preparation holds its cache and first token
    /// privately; it does not end capture or emit text. Commit need not wait for the
    /// [`SessionEvent::Prepared`] notification. Repeating the same candidate is idempotent.
    pub async fn prepare(
        &self,
        turn_id: TurnId,
        chunk_count: u32,
        sample_count: usize,
    ) -> Result<(), RuntimeError> {
        self.request(|reply| Command::Prepare {
            key: self.key.clone(),
            turn_id,
            chunk_count,
            sample_count,
            reply,
        })
        .await
    }

    /// Confirms a complete utterance and permits response generation.
    ///
    /// Counts must match all accepted audio. A valid prepared candidate is activated;
    /// otherwise ordinary prefill runs. Repeating an unchanged commit is idempotent
    /// and does not restart a completed response.
    pub async fn commit(
        &self,
        turn_id: TurnId,
        final_chunk_count: u32,
        sample_count: usize,
    ) -> Result<(), RuntimeError> {
        self.request(|reply| Command::Commit {
            key: self.key.clone(),
            turn_id,
            final_chunk_count,
            sample_count,
            reply,
        })
        .await
    }

    /// Stops the current turn logically, preserving received audio and accepted output.
    ///
    /// An in-flight GPU operation may finish; its later proposal is fenced. Cancellation
    /// does not rewind committed model state or erase events already accepted for delivery.
    pub async fn cancel(&self, turn_id: TurnId) -> Result<(), RuntimeError> {
        self.request(|reply| Command::Cancel {
            key: self.key.clone(),
            turn_id,
            reply,
        })
        .await
    }

    /// Reads the next event, draining already accepted output before cancellation ends it.
    ///
    /// Returns `None` after the event stream closes or cancellation has no queued events.
    pub async fn next_event(&mut self) -> Option<SessionEvent> {
        if let Ok(event) = self.events.try_recv() {
            return Some(event);
        }

        // Accepted events must remain observable when cancellation is also ready.
        tokio::select! {
            biased;

            event = self.events.recv() => event,
            _ = self.cancellation.cancelled() => None,
        }
    }

    /// Releases the backend cache and returns the complete record for archiving.
    ///
    /// Unlike ordinary commands, close waits for mailbox space so queue saturation cannot
    /// drop a record. After success, drain remaining events if they must reach the client.
    pub async fn close(&mut self) -> Result<SessionRecord, RuntimeError> {
        let (reply, response) = oneshot::channel();
        self.worker.close(self.key.clone(), reply).await?;
        let record = response
            .await
            .map_err(|_| RuntimeError::new(ErrorCode::Shutdown, "worker stopped"))??;

        self.cancellation.cancel();
        let _ = self.manager.try_send(ManagerCommand::Release {
            session_id: self.session_id.clone(),
            key: self.key.clone(),
        });
        Ok(record)
    }

    async fn request(
        &self,
        command: impl FnOnce(oneshot::Sender<Result<(), RuntimeError>>) -> Command,
    ) -> Result<(), RuntimeError> {
        let (reply, response) = oneshot::channel();
        self.worker.send(command(reply))?;
        response
            .await
            .map_err(|_| RuntimeError::new(ErrorCode::Shutdown, "worker stopped"))?
    }
}

impl Drop for SessionHandle {
    fn drop(&mut self) {
        self.cancellation.cancel();
    }
}
