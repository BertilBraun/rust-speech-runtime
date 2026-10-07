//! Validates a new turn before interruption can change accepted conversation state.

use super::actor::WorkerActor;
use crate::{
    protocol::{ErrorCode, TurnId},
    runtime::RuntimeError,
    session::state::{SessionState, Stage},
};
use std::sync::atomic::Ordering;

const MINIMUM_TURN_RECORD_HEADROOM_BYTES: usize = 1024;
const MINIMUM_TURN_CONTEXT_HEADROOM_TOKENS: usize = 32;

impl WorkerActor {
    pub(super) fn validate_turn_start(
        &self,
        session_key: &str,
        turn_id: TurnId,
    ) -> Result<(), RuntimeError> {
        if !self.backend_available.load(Ordering::Acquire) {
            return Err(RuntimeError::new(
                ErrorCode::BackendUnavailable,
                "worker connection failed",
            ));
        }
        let session = self
            .sessions
            .get(session_key)
            .ok_or_else(|| RuntimeError::new(ErrorCode::SessionNotFound, "session unavailable"))?;
        if session
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
        self.validate_compute_reservation(session)?;
        self.validate_history_capacity(session)
    }

    fn validate_compute_reservation(&self, session: &SessionState) -> Result<(), RuntimeError> {
        let active_turn_count = self
            .sessions
            .values()
            .filter(|session| session.has_active_turn())
            .count();
        let reserved_turn_count = active_turn_count + usize::from(!session.has_active_turn());
        if reserved_turn_count > self.configuration.max_active_turns_per_worker {
            return Err(turn_capacity_error());
        }
        // Interrupting an existing turn reuses its reservation instead of adding another.
        if session.has_active_turn() {
            return Ok(());
        }

        let has_generating_turn = self.sessions.values().any(|session| {
            session.has_active_turn() && matches!(session.stage, Stage::Generating { .. })
        });
        // Reserve using the longest active context and room for audio prefill alongside decode.
        let longest_context_tokens = self
            .sessions
            .values()
            .filter(|session| session.has_active_turn())
            .map(|session| session.context_tokens)
            .max()
            .unwrap_or(0)
            .max(session.context_tokens);
        let fits_compute_budget = self.forward_costs.admits(
            reserved_turn_count,
            self.batch_size_limit(),
            longest_context_tokens,
            &self.configuration,
            has_generating_turn,
        );
        if !fits_compute_budget {
            return Err(turn_capacity_error());
        }
        Ok(())
    }

    fn validate_history_capacity(&self, session: &SessionState) -> Result<(), RuntimeError> {
        if session
            .history_bytes
            .saturating_add(MINIMUM_TURN_RECORD_HEADROOM_BYTES)
            > self.configuration.max_history_bytes
        {
            return Err(RuntimeError::new(
                ErrorCode::HistoryLimit,
                "session record limit reached",
            ));
        }
        if session
            .context_tokens
            .saturating_add(MINIMUM_TURN_CONTEXT_HEADROOM_TOKENS)
            >= self.context_token_limit()
        {
            return Err(RuntimeError::new(
                ErrorCode::ContextLimit,
                "conversation context exhausted",
            ));
        }
        Ok(())
    }
}

fn turn_capacity_error() -> RuntimeError {
    RuntimeError::new(
        ErrorCode::CapacityExceeded,
        "worker cannot reserve another active turn at target token rate",
    )
}
