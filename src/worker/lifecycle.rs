//! Session allocation, closure and record transfer stay with the owning actor.

use super::actor::WorkerActor;
use crate::{
    protocol::{ErrorCode, SessionEvent, SessionId, SessionRecord},
    runtime::RuntimeError,
    session::state::SessionState,
};
use std::sync::atomic::Ordering;
use tokio::sync::{mpsc, oneshot};
use tokio_util::sync::CancellationToken;

impl WorkerActor {
    pub(super) fn open_session(
        &mut self,
        key: String,
        session_id: SessionId,
        events: mpsc::Sender<SessionEvent>,
        cancellation: CancellationToken,
        reply: oneshot::Sender<Result<(), RuntimeError>>,
    ) {
        if !self.available.load(Ordering::Acquire)
            || self.sessions.len() >= self.config.max_sessions_per_worker
        {
            let _ = reply.send(Err(RuntimeError::new(
                ErrorCode::CapacityExceeded,
                "worker session capacity reached",
            )));
            return;
        }

        let mut session = SessionState::new(
            session_id,
            self.id,
            self.ready.model_id.clone(),
            events,
            cancellation,
        );
        session.open_reply = Some(reply);
        self.sessions.insert(key, session);
        self.load.store(self.sessions.len(), Ordering::Relaxed);
        self.metrics.active_sessions.fetch_add(1, Ordering::Relaxed);
    }

    pub(super) fn close_session(
        &mut self,
        key: &str,
        reply: Option<oneshot::Sender<Result<SessionRecord, RuntimeError>>>,
    ) {
        let generation = self.generation();
        let Some(session) = self.sessions.get_mut(key) else {
            if let Some(reply) = reply {
                let _ = reply.send(Err(RuntimeError::new(
                    ErrorCode::SessionNotFound,
                    "session unavailable",
                )));
            }
            return;
        };

        if !session.closing {
            session.interrupt(generation);
            session.closing = true;
            self.metrics.active_sessions.fetch_sub(1, Ordering::Relaxed);
        }
        session.close_reply = reply;
        if (!session.opened || session.backend_closed) && !session.in_flight {
            self.remove_closed(key);
        }
    }

    pub(super) fn session(&mut self, key: &str) -> Result<&mut SessionState, RuntimeError> {
        if !self.available.load(Ordering::Acquire) {
            return Err(RuntimeError::new(
                ErrorCode::BackendUnavailable,
                "worker connection failed",
            ));
        }
        self.sessions
            .get_mut(key)
            .filter(|session| !session.closing)
            .ok_or_else(|| RuntimeError::new(ErrorCode::SessionNotFound, "session unavailable"))
    }

    pub fn remove_closed(&mut self, key: &str) {
        if let Some(mut session) = self.sessions.remove(key) {
            let _ = session.events.try_send(SessionEvent::Closed {
                session_id: session.record.session_id.clone(),
            });
            if let Some(reply) = session.close_reply.take() {
                let _ = reply.send(Ok(session.record));
            }
            self.load.store(self.sessions.len(), Ordering::Relaxed);
        }
    }
}
