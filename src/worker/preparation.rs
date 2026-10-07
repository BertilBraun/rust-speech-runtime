//! Owns private endpoint candidates, their invalidation and safe fallback to ordinary prefill.

use super::{actor::WorkerActor, turn::check_turn};
use crate::{
    protocol::{
        ErrorCode, SessionEvent, TurnId,
        backend::{OperationResult, Outcome},
    },
    runtime::RuntimeError,
    session::state::{CaptureSnapshot, Preparation, PreparationRequest, Stage},
};
use std::sync::atomic::Ordering;
use tokio::time::Instant;

impl WorkerActor {
    pub(super) fn prepare(
        &mut self,
        key: &str,
        turn_id: TurnId,
        chunk_count: u32,
        sample_count: usize,
    ) -> Result<(), RuntimeError> {
        let context_limit = self
            .config
            .max_context_tokens
            .min(self.ready.max_context_tokens);
        let session = self.session(key)?;
        check_turn(session, turn_id)?;
        if !matches!(session.stage, Stage::Capturing { .. }) {
            return Err(RuntimeError::new(
                ErrorCode::InvalidState,
                "preparation requires a capturing turn",
            ));
        }
        if sample_count == 0
            || session.chunk_count != chunk_count
            || session.turn().expect("capturing turn").audio_pcm16.len() / 2 != sample_count
        {
            return Err(RuntimeError::new(
                ErrorCode::InvalidInput,
                "prepare chunk/sample counts mismatch",
            ));
        }
        if session
            .context_tokens
            .saturating_add(sample_count.div_ceil(1600) + 32)
            >= context_limit
        {
            return Err(RuntimeError::new(
                ErrorCode::ContextLimit,
                "utterance exceeds remaining context",
            ));
        }
        let snapshot = CaptureSnapshot {
            turn_id,
            generation: session.generation(),
            chunk_count,
            sample_count,
        };
        if session.preparation.snapshot() == Some(snapshot) {
            return Ok(());
        }
        let request = PreparationRequest {
            snapshot,
            queued_at: Instant::now(),
        };
        session.preparation = match session.preparation {
            Preparation::None | Preparation::Queued(_) => Preparation::Queued(request),
            Preparation::Discarding { .. } => Preparation::Discarding {
                next: Some(request),
            },
            Preparation::Running(_)
            | Preparation::Ready(_)
            | Preparation::DiscardPending { .. } => Preparation::DiscardPending {
                next: Some(request),
            },
        };
        Ok(())
    }

    pub(super) fn complete_preparation(
        &mut self,
        key: &str,
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
        let session = self.sessions.get_mut(key).expect("preparing session");
        let snapshot = match session.preparation {
            Preparation::Running(snapshot)
                if snapshot.generation == generation
                    && snapshot.turn_id.0 == turn_id
                    && session.generation() == generation
                    && !session.closing
                    && session.active() =>
            {
                Some(snapshot)
            }
            _ => None,
        };
        match result.outcome {
            Outcome::Token { .. } => {
                if let Some(snapshot) = snapshot {
                    session.preparation = Preparation::Ready(snapshot);
                    if session
                        .events
                        .try_send(SessionEvent::Prepared {
                            turn_id: snapshot.turn_id,
                            chunk_count: snapshot.chunk_count,
                            sample_count: snapshot.sample_count,
                        })
                        .is_err()
                    {
                        session.preparation.invalidate();
                        session.finish(crate::protocol::FinishReason::SlowConsumer);
                        session.cancellation.cancel();
                        session.closing = true;
                        self.metrics.active_sessions.fetch_sub(1, Ordering::Relaxed);
                        self.metrics.saturation.fetch_add(1, Ordering::Relaxed);
                    }
                } else {
                    self.metrics.stale_results.fetch_add(1, Ordering::Relaxed);
                }
            }
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
        key: &str,
        outcome: Outcome,
    ) -> Result<(), RuntimeError> {
        if !matches!(outcome, Outcome::Discarded) {
            return Err(RuntimeError::new(
                ErrorCode::BackendFailed,
                "unexpected discard result",
            ));
        }
        let session = self.sessions.get_mut(key).expect("discarding session");
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
