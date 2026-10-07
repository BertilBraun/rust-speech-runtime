//! Owns private endpoint candidates, their invalidation and safe fallback to ordinary prefill.

use super::{actor::WorkerActor, turn::check_turn};
use crate::{
    metrics::Metrics,
    protocol::{
        ErrorCode, FinishReason, SessionEvent, TurnId,
        backend::{OperationResult, Outcome},
    },
    runtime::RuntimeError,
    session::state::{CaptureSnapshot, Preparation, PreparationRequest, SessionState, Stage},
};
use std::sync::atomic::Ordering;
use tokio::time::Instant;

impl WorkerActor {
    pub(super) fn prepare_turn(
        &mut self,
        session_key: &str,
        turn_id: TurnId,
        chunk_count: u32,
        sample_count: usize,
    ) -> Result<(), RuntimeError> {
        let context_limit = self.context_token_limit();
        let session = self.session_mut(session_key)?;
        check_turn(session, turn_id)?;
        validate_candidate_capture(session, chunk_count, sample_count, context_limit)?;
        let snapshot = CaptureSnapshot {
            turn_id,
            generation: session.generation,
            chunk_count,
            sample_count,
        };
        if session.preparation.snapshot() == Some(snapshot) {
            return Ok(());
        }
        session.preparation.queue(PreparationRequest {
            snapshot,
            queued_at: Instant::now(),
        });
        Ok(())
    }

    pub(super) fn complete_preparation(
        &mut self,
        session_key: &str,
        turn_id: u64,
        generation: u64,
        result: OperationResult,
    ) -> Result<(), RuntimeError> {
        if result.turn_id != Some(turn_id) || result.generation != Some(generation) {
            return Err(RuntimeError::new(
                ErrorCode::BackendFailed,
                "prepared turn/generation mismatch",
            ));
        }
        let session = self
            .sessions
            .get_mut(session_key)
            .expect("preparing session");
        let snapshot = current_preparation_snapshot(session, turn_id, generation);
        match result.outcome {
            Outcome::Token { .. } => publish_prepared_candidate(session, snapshot, &self.metrics),
            Outcome::Failed { .. } => {
                if snapshot.is_some() {
                    session.preparation = Preparation::DiscardPending { next: None };
                }
                self.metrics
                    .preparation_fallbacks
                    .fetch_add(1, Ordering::Relaxed);
            }
            _ => {
                return Err(RuntimeError::new(
                    ErrorCode::BackendFailed,
                    "unexpected preparation result",
                ));
            }
        }
        Ok(())
    }

    pub(super) fn complete_discard(
        &mut self,
        session_key: &str,
        outcome: Outcome,
    ) -> Result<(), RuntimeError> {
        if !matches!(outcome, Outcome::Discarded) {
            return Err(RuntimeError::new(
                ErrorCode::BackendFailed,
                "unexpected discard result",
            ));
        }
        let session = self
            .sessions
            .get_mut(session_key)
            .expect("discarding session");
        let Preparation::Discarding { next } = session.preparation else {
            return Err(RuntimeError::new(
                ErrorCode::BackendFailed,
                "discard completed in unexpected state",
            ));
        };
        session.preparation = next.map_or(Preparation::None, Preparation::Queued);
        self.metrics
            .preparations_discarded
            .fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
}

fn validate_candidate_capture(
    session: &SessionState,
    chunk_count: u32,
    sample_count: usize,
    context_limit: usize,
) -> Result<(), RuntimeError> {
    if !matches!(session.stage, Stage::Capturing { .. }) {
        return Err(RuntimeError::new(
            ErrorCode::InvalidState,
            "preparation requires a capturing turn",
        ));
    }
    let captured_samples = session
        .current_turn()
        .expect("capturing turn")
        .audio_pcm16
        .len()
        / 2;
    if sample_count == 0 || session.chunk_count != chunk_count || captured_samples != sample_count {
        return Err(RuntimeError::new(
            ErrorCode::InvalidInput,
            "prepare chunk/sample counts mismatch",
        ));
    }
    let estimated_context_tokens = session
        .context_tokens
        .saturating_add(sample_count.div_ceil(1600) + 32);
    if estimated_context_tokens >= context_limit {
        return Err(RuntimeError::new(
            ErrorCode::ContextLimit,
            "utterance exceeds remaining context",
        ));
    }
    Ok(())
}

fn current_preparation_snapshot(
    session: &SessionState,
    turn_id: u64,
    generation: u64,
) -> Option<CaptureSnapshot> {
    let Preparation::Running(snapshot) = session.preparation else {
        return None;
    };
    if session.closing || !session.has_active_turn() || session.generation != generation {
        return None;
    }
    if snapshot.generation != generation || snapshot.turn_id.0 != turn_id {
        return None;
    }
    Some(snapshot)
}

fn publish_prepared_candidate(
    session: &mut SessionState,
    snapshot: Option<CaptureSnapshot>,
    metrics: &Metrics,
) {
    let Some(snapshot) = snapshot else {
        metrics.stale_results.fetch_add(1, Ordering::Relaxed);
        return;
    };
    session.preparation = Preparation::Ready(snapshot);
    let event = SessionEvent::Prepared {
        turn_id: snapshot.turn_id,
        chunk_count: snapshot.chunk_count,
        sample_count: snapshot.sample_count,
    };
    if session.events.try_send(event).is_ok() {
        return;
    }
    session.preparation.invalidate();
    session.finish_turn(FinishReason::SlowConsumer);
    session.cancellation.cancel();
    session.closing = true;
    metrics.active_sessions.fetch_sub(1, Ordering::Relaxed);
    metrics.saturation.fetch_add(1, Ordering::Relaxed);
}
