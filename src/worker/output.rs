//! Accepts token proposals into the bounded event stream before recording cache acknowledgements.

use super::actor::WorkerActor;
use crate::{
    metrics::Metrics,
    protocol::{ErrorCode, FinishReason, SessionEvent, TokenRecord, backend::AcceptedToken},
    runtime::RuntimeError,
    session::state::{SessionState, Stage},
};
use std::sync::atomic::Ordering;
use tokio::time::Instant;

impl WorkerActor {
    pub(super) fn accept_token(
        &mut self,
        key: &str,
        token_id: u32,
        text: String,
        eos: bool,
        context_tokens: usize,
    ) -> Result<(), RuntimeError> {
        let interval = SessionState::token_interval(self.config.target_tokens_per_second);
        let limit = self
            .config
            .max_context_tokens
            .min(self.ready.max_context_tokens);
        let session = self.sessions.get_mut(key).expect("completion session");
        session.context_tokens = context_tokens;
        if !enforce_token_limits(session, limit, self.config.max_history_bytes, text.len()) {
            return Ok(());
        }

        let now = Instant::now();
        observe_token(session, now, &self.metrics)?;
        if !publish_token(session, token_id, text, now, &self.metrics) {
            return Ok(());
        }

        session.stage = Stage::Generating {
            last_token_at: now,
            deadline: now + interval,
        };
        if eos {
            session.finish(FinishReason::Eos);
        } else if session.turn().expect("inference turn").tokens.len()
            >= self.config.max_output_tokens
        {
            session.finish(FinishReason::TokenLimit);
        }
        Ok(())
    }
}

fn enforce_token_limits(
    session: &mut SessionState,
    context_limit: usize,
    history_limit: usize,
    text_bytes: usize,
) -> bool {
    if session.context_tokens >= context_limit {
        session.finish(FinishReason::ContextLimit);
        return false;
    }
    if session.history_bytes.saturating_add(text_bytes + 64) > history_limit {
        session.finish(FinishReason::ContextLimit);
        let _ = session.events.try_send(SessionEvent::Failed {
            turn_id: session.turn().map(|turn| turn.turn_id),
            code: ErrorCode::HistoryLimit,
            message: "session record limit exceeded".into(),
        });
        return false;
    }
    true
}

fn observe_token(
    session: &SessionState,
    now: Instant,
    metrics: &Metrics,
) -> Result<(), RuntimeError> {
    let committed_at = session.committed_at.expect("committed inference turn");
    match session.stage {
        Stage::Prefill { .. } => metrics
            .ttft
            .record((now - committed_at).as_secs_f64() * 1000.0),
        Stage::Generating {
            last_token_at,
            deadline,
            ..
        } => {
            metrics
                .token_gap
                .record((now - last_token_at).as_secs_f64() * 1000.0);
            if now > deadline {
                metrics.deadline_misses.fetch_add(1, Ordering::Relaxed);
            }
        }
        _ => {
            return Err(RuntimeError::new(
                ErrorCode::BackendFailed,
                "token completion outside inference state",
            ));
        }
    }
    Ok(())
}

fn publish_token(
    session: &mut SessionState,
    token_id: u32,
    text: String,
    now: Instant,
    metrics: &Metrics,
) -> bool {
    let committed_at = session.committed_at.expect("committed inference turn");
    let generation = session.generation();
    let turn = session.turn_mut().expect("inference turn");
    let index = turn.tokens.len() as u64;
    let turn_id = turn.turn_id;
    let token = TokenRecord {
        token_id,
        text: text.clone(),
        index,
        elapsed_ms: (now - committed_at).as_secs_f64() * 1000.0,
    };
    let event = SessionEvent::TextDelta {
        turn_id,
        generation,
        sequence: index,
        token_id,
        text,
    };
    // Only successfully queued proposals become recorded output and future cache acknowledgements.
    if session.events.try_send(event).is_err() {
        metrics.saturation.fetch_add(1, Ordering::Relaxed);
        session.finish(FinishReason::SlowConsumer);
        session.cancellation.cancel();
        return false;
    }
    session.history_bytes += token.text.len() + 64;
    session
        .turn_mut()
        .expect("inference turn")
        .tokens
        .push(token);
    session.pending_token = Some(AcceptedToken {
        turn_id: turn_id.0,
        index,
        token_id,
    });
    metrics.generated_tokens.fetch_add(1, Ordering::Relaxed);
    true
}
