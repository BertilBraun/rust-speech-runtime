//! Ends affected generations when the backend can no longer preserve cache state.

use super::actor::WorkerActor;
use crate::{
    protocol::{FinishReason, SessionEvent},
    runtime::RuntimeError,
};
use std::sync::atomic::Ordering;

impl WorkerActor {
    pub(super) fn handle_backend_failure(&mut self, error: RuntimeError) {
        self.backend_available.store(false, Ordering::Release);
        self.in_flight_batch = None;
        self.metrics
            .backend_failures
            .fetch_add(1, Ordering::Relaxed);
        for session in self.sessions.values_mut() {
            session.opened = false;
            session.backend_closed = true;
            session.in_flight = false;
            if session.has_active_turn() {
                session.finish_turn(FinishReason::BackendFailed);
            }
            if let Some(reply) = session.open_reply.take() {
                let _ = reply.send(Err(error.clone()));
            }
            let _ = session.events.try_send(SessionEvent::Failed {
                turn_id: session.current_turn().map(|turn| turn.turn_id),
                code: error.code(),
                message: error.to_string(),
            });
        }
        let closing = self
            .sessions
            .iter()
            .filter(|(_, session)| session.closing && session.close_reply.is_some())
            .map(|(session_key, _)| session_key.clone())
            .collect::<Vec<_>>();
        for session_key in closing {
            self.remove_closed(&session_key);
        }
    }
}
