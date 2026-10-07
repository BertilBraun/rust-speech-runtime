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
        session_key: String,
        session_id: SessionId,
        events: mpsc::Sender<SessionEvent>,
        cancellation: CancellationToken,
        reply: oneshot::Sender<Result<(), RuntimeError>>,
    ) {
        if !self.backend_available.load(Ordering::Acquire)
            || self.sessions.len() >= self.configuration.max_sessions_per_worker
        {
            let _ = reply.send(Err(RuntimeError::new(
                ErrorCode::CapacityExceeded,
                "worker session capacity reached",
            )));
            return;
        }

        let mut session = SessionState::new(
            session_id,
            self.worker_id,
            self.backend_capabilities.model_id.clone(),
            events,
            cancellation,
        );
        session.open_reply = Some(reply);
        self.sessions.insert(session_key, session);
        self.session_count
            .store(self.sessions.len(), Ordering::Relaxed);
        self.metrics.active_sessions.fetch_add(1, Ordering::Relaxed);
    }

    pub(super) fn close_session(
        &mut self,
        session_key: &str,
        reply: Option<oneshot::Sender<Result<SessionRecord, RuntimeError>>>,
    ) {
        let generation = self.allocate_generation_id();
        let Some(session) = self.sessions.get_mut(session_key) else {
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
            self.remove_closed(session_key);
        }
    }

    pub(super) fn session_mut(
        &mut self,
        session_key: &str,
    ) -> Result<&mut SessionState, RuntimeError> {
        if !self.backend_available.load(Ordering::Acquire) {
            return Err(RuntimeError::new(
                ErrorCode::BackendUnavailable,
                "worker connection failed",
            ));
        }
        self.sessions
            .get_mut(session_key)
            .filter(|session| !session.closing)
            .ok_or_else(|| RuntimeError::new(ErrorCode::SessionNotFound, "session unavailable"))
    }

    pub(super) fn remove_closed(&mut self, session_key: &str) {
        if let Some(mut session) = self.sessions.remove(session_key) {
            let _ = session.events.try_send(SessionEvent::Closed {
                session_id: session.record.session_id.clone(),
            });
            if let Some(reply) = session.close_reply.take() {
                let _ = reply.send(Ok(session.record));
            }
            self.session_count
                .store(self.sessions.len(), Ordering::Relaxed);
        }
    }

    pub(super) fn stop_sessions(&mut self) {
        self.backend_available.store(false, Ordering::Release);
        self.metrics.active_sessions.fetch_sub(
            self.sessions
                .values()
                .filter(|session| !session.closing)
                .count() as u64,
            Ordering::Relaxed,
        );
        self.session_count.store(0, Ordering::Relaxed);
        for mut session in self.sessions.drain().map(|(_, session)| session) {
            session.cancellation.cancel();
            if let Some(reply) = session.close_reply.take() {
                let _ = reply.send(Err(RuntimeError::new(
                    ErrorCode::BackendUnavailable,
                    "worker stopped",
                )));
            }
        }
    }

    pub(super) fn cleanup_disconnected_sessions(&mut self) {
        let generation = self.allocate_generation_id();
        for session in self.sessions.values_mut() {
            if !session.closing
                && (session.cancellation.is_cancelled() || session.events.is_closed())
            {
                session.interrupt(generation);
                session.closing = true;
                self.metrics.active_sessions.fetch_sub(1, Ordering::Relaxed);
            }
        }
        self.sessions
            .retain(|_, session| !session.can_remove_after_disconnect());
        self.session_count
            .store(self.sessions.len(), Ordering::Relaxed);
    }
}
