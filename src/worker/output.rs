use super::actor::WorkerActor;
use crate::{
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
        if context_tokens >= limit {
            session.finish(FinishReason::ContextLimit);
            return Ok(());
        }
        if session.history_bytes.saturating_add(text.len() + 64) > self.config.max_history_bytes {
            session.finish(FinishReason::ContextLimit);
            let _ = session.events.try_send(SessionEvent::Failed {
                turn_id: session.turn().map(|turn| turn.turn_id),
                code: ErrorCode::HistoryLimit,
                message: "session record limit exceeded".into(),
            });
            return Ok(());
        }
        let now = Instant::now();
        let committed_at = session.committed_at.expect("committed inference turn");
        match session.stage {
            Stage::Prefill { .. } => self
                .metrics
                .ttft
                .record((now - committed_at).as_secs_f64() * 1000.0),
            Stage::Generating {
                last_token_at,
                deadline,
                ..
            } => {
                self.metrics
                    .token_gap
                    .record((now - last_token_at).as_secs_f64() * 1000.0);
                if now > deadline {
                    self.metrics.deadline_misses.fetch_add(1, Ordering::Relaxed);
                }
            }
            _ => {
                return Err(RuntimeError::new(
                    ErrorCode::BackendFailed,
                    "token completion outside inference state",
                ));
            }
        }
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
        if session.events.try_send(event).is_err() {
            self.metrics.saturation.fetch_add(1, Ordering::Relaxed);
            session.finish(FinishReason::SlowConsumer);
            session.cancellation.cancel();
            return Ok(());
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
        self.metrics
            .generated_tokens
            .fetch_add(1, Ordering::Relaxed);
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
