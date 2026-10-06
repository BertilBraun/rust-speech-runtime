mod actor;
mod backend;
mod batch;
mod commands;
mod completion;
mod output;

use crate::{
    config::{RuntimeConfig, WorkerConfig},
    metrics::Metrics,
    protocol::{ErrorCode, SessionEvent, SessionId, SessionRecord, TurnId},
    runtime::RuntimeError,
};
use bytes::Bytes;
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
};
use tokio::{
    sync::{mpsc, oneshot},
    task::JoinHandle,
};
use tokio_util::sync::CancellationToken;

pub(crate) enum Command {
    Open {
        key: String,
        session_id: SessionId,
        events: mpsc::Sender<SessionEvent>,
        cancellation: CancellationToken,
        reply: oneshot::Sender<Result<(), RuntimeError>>,
    },
    Begin {
        key: String,
        turn_id: TurnId,
        reply: oneshot::Sender<Result<(), RuntimeError>>,
    },
    Audio {
        key: String,
        turn_id: TurnId,
        chunk_index: u32,
        audio: Bytes,
        reply: oneshot::Sender<Result<(), RuntimeError>>,
    },
    Commit {
        key: String,
        turn_id: TurnId,
        final_chunk_count: u32,
        sample_count: usize,
        reply: oneshot::Sender<Result<(), RuntimeError>>,
    },
    Cancel {
        key: String,
        turn_id: TurnId,
        reply: oneshot::Sender<Result<(), RuntimeError>>,
    },
    Close {
        key: String,
        reply: Option<oneshot::Sender<Result<SessionRecord, RuntimeError>>>,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bounded_worker_mailbox_rejects_without_allocating_more_queue_slots() {
        let (sender, _receiver) = mpsc::channel(1);
        let metrics = Arc::new(Metrics::default());
        let worker = WorkerHandle {
            id: 0,
            sender,
            load: Arc::new(AtomicUsize::new(0)),
            available: Arc::new(AtomicBool::new(true)),
            metrics: metrics.clone(),
        };
        worker
            .send(Command::Close {
                key: "first".into(),
                reply: None,
            })
            .unwrap();
        assert_eq!(
            worker
                .send(Command::Close {
                    key: "second".into(),
                    reply: None
                })
                .unwrap_err()
                .code(),
            ErrorCode::ChannelSaturated
        );
        assert_eq!(metrics.snapshot().channel_saturation_events, 1);
    }
}
#[derive(Clone)]
pub(crate) struct WorkerHandle {
    pub id: usize,
    sender: mpsc::Sender<Command>,
    load: Arc<AtomicUsize>,
    available: Arc<AtomicBool>,
    metrics: Arc<Metrics>,
}
impl WorkerHandle {
    pub async fn start(
        id: usize,
        worker_config: &WorkerConfig,
        config: RuntimeConfig,
        metrics: Arc<Metrics>,
        clock: Arc<AtomicU64>,
        cancellation: CancellationToken,
    ) -> Result<(Self, JoinHandle<()>), RuntimeError> {
        let (connection, ready) = tokio::time::timeout(
            std::time::Duration::from_millis(config.backend_timeout_ms),
            backend::BackendConnection::connect(worker_config.endpoint),
        )
        .await
        .map_err(|_| {
            RuntimeError::new(ErrorCode::BackendUnavailable, "worker startup timeout")
        })??;
        let (sender, receiver) = mpsc::channel(config.mailbox_capacity);
        let load = Arc::new(AtomicUsize::new(0));
        let available = Arc::new(AtomicBool::new(true));
        let actor = actor::WorkerActor::new(
            id,
            config,
            ready,
            metrics.clone(),
            clock,
            load.clone(),
            available.clone(),
        );
        let task = tokio::spawn(actor.run(receiver, connection, cancellation));
        Ok((
            Self {
                id,
                sender,
                load,
                available,
                metrics,
            },
            task,
        ))
    }
    pub fn load(&self) -> usize {
        self.load.load(Ordering::Relaxed)
    }
    pub fn available(&self) -> bool {
        self.available.load(Ordering::Acquire)
    }
    pub fn send(&self, command: Command) -> Result<(), RuntimeError> {
        self.sender.try_send(command).map_err(|error| {
            self.metrics.saturation.fetch_add(1, Ordering::Relaxed);
            RuntimeError::new(
                match error {
                    mpsc::error::TrySendError::Closed(_) => ErrorCode::BackendUnavailable,
                    mpsc::error::TrySendError::Full(_) => ErrorCode::ChannelSaturated,
                },
                "worker mailbox unavailable",
            )
        })
    }
}
