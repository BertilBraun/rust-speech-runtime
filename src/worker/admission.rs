//! Validates a new turn before interruption can change accepted conversation state.

use super::actor::WorkerActor;
use crate::{
    protocol::{ErrorCode, TurnId},
    runtime::RuntimeError,
    session::state::{SessionState, Stage},
};
use std::sync::atomic::Ordering;

impl WorkerActor {
    pub(super) fn validate_turn_start(
        &self,
        key: &str,
        turn_id: TurnId,
    ) -> Result<(), RuntimeError> {
        if !self.available.load(Ordering::Acquire) {
            return Err(RuntimeError::new(
                ErrorCode::BackendUnavailable,
                "worker connection failed",
            ));
        }
        let previous = self
            .sessions
            .get(key)
            .ok_or_else(|| RuntimeError::new(ErrorCode::SessionNotFound, "session unavailable"))?;
        if previous
            .record
            .turns
            .iter()
            .any(|turn| turn.turn_id == turn_id)
        {
            return Err(RuntimeError::new(
                ErrorCode::InvalidInput,
                "turn ID must be unique within session",
            ));
        }
        self.validate_compute_reservation(previous)?;
        self.validate_history_capacity(previous)
    }

    fn validate_compute_reservation(&self, previous: &SessionState) -> Result<(), RuntimeError> {
        let active = self
            .sessions
            .values()
            .filter(|session| session.active())
            .count();
        let reserved = active + usize::from(!previous.active());
        let reserve_prefill = self
            .sessions
            .values()
            .any(|session| session.active() && matches!(session.stage, Stage::Generating { .. }));
        // Reserve using the longest active context and room for audio prefill alongside decode.
        let context = self
            .sessions
            .values()
            .filter(|session| session.active())
            .map(|session| session.context_tokens)
            .chain(std::iter::once(previous.context_tokens))
            .max()
            .unwrap_or(0);
        if reserved > self.config.max_active_turns_per_worker
            || (!previous.active()
                && !self.costs.admits(
                    reserved,
                    self.config.max_batch_size.min(self.ready.max_batch_size),
                    context,
                    &self.config,
                    reserve_prefill,
                ))
        {
            return Err(RuntimeError::new(
                ErrorCode::CapacityExceeded,
                "worker cannot reserve another active turn at target token rate",
            ));
        }
        Ok(())
    }

    fn validate_history_capacity(&self, previous: &SessionState) -> Result<(), RuntimeError> {
        if previous.history_bytes.saturating_add(1024) > self.config.max_history_bytes {
            return Err(RuntimeError::new(
                ErrorCode::HistoryLimit,
                "session record limit reached",
            ));
        }
        if previous.context_tokens.saturating_add(32)
            >= self
                .config
                .max_context_tokens
                .min(self.ready.max_context_tokens)
        {
            return Err(RuntimeError::new(
                ErrorCode::ContextLimit,
                "conversation context exhausted",
            ));
        }
        Ok(())
    }
}
