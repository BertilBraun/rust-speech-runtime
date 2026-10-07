use super::actor::{Completion, WorkerActor};
use crate::{
    protocol::{
        ErrorCode, FinishReason, SessionEvent,
        backend::{BatchResponse, Operation, OperationResult, Outcome},
    },
    runtime::RuntimeError,
    scheduler::BatchKind,
};
use std::sync::atomic::Ordering;

impl WorkerActor {
    pub fn complete(&mut self, completion: Completion) -> Result<(), RuntimeError> {
        let active = self.active.take().expect("active batch");
        let response = completion.response?;
        self.observe_batch(
            &active.request.operations,
            active.kind,
            completion.elapsed_ms,
            &response,
        );
        for (operation, result) in active.request.operations.into_iter().zip(response.results) {
            let key = operation.session_id();
            self.sessions
                .get_mut(key)
                .ok_or_else(|| protocol_error("completion for unknown session"))?
                .in_flight = false;
            match operation {
                Operation::Open { session_id, .. } => {
                    self.complete_open(&session_id, result.outcome)?
                }
                Operation::Close { session_id, .. } => {
                    self.complete_close(&session_id, result.outcome)?
                }
                Operation::DiscardPrepared { session_id, .. } => {
                    self.complete_discard(&session_id, result.outcome)?
                }
                Operation::Prepare {
                    session_id,
                    turn_id,
                    generation,
                    ..
                } => self.complete_preparation(&session_id, turn_id, generation, result)?,
                Operation::Activate {
                    session_id,
                    turn_id,
                    generation,
                    ..
                } => self.complete_activation(&session_id, turn_id, generation, result)?,
                Operation::Prefill {
                    session_id,
                    turn_id,
                    generation,
                    ..
                }
                | Operation::Decode {
                    session_id,
                    turn_id,
                    generation,
                    ..
                } => self.complete_inference(&session_id, turn_id, generation, result)?,
            }
        }
        Ok(())
    }
    fn observe_batch(
        &mut self,
        operations: &[Operation],
        kind: BatchKind,
        elapsed_ms: f64,
        response: &BatchResponse,
    ) {
        self.metrics.worker_observation(
            self.id,
            elapsed_ms,
            response.memory.allocated_bytes,
            response.memory.reserved_bytes,
        );
        for (duration, distribution) in [
            (response.timing.encode_ms, &self.metrics.encode),
            (response.timing.prefill_ms, &self.metrics.prefill),
            (response.timing.decode_ms, &self.metrics.decode),
        ] {
            if duration > 0.0 {
                distribution.record(duration);
            }
        }
        self.metrics.inference.record(elapsed_ms);
        let context = operations
            .iter()
            .filter_map(|operation| {
                self.sessions
                    .get(operation.session_id())
                    .map(|session| (operation, session))
            })
            .map(|(operation, session)| {
                let appended_tokens = match operation {
                    Operation::Prefill { audio_bytes, .. }
                    | Operation::Prepare { audio_bytes, .. } => audio_bytes.div_ceil(3200) + 32,
                    _ => 0,
                };
                // One in-flight operation keeps this cache length stable until completion.
                session.context_tokens.saturating_add(appended_tokens)
            })
            .max()
            .unwrap_or(0);
        self.costs
            .observe(kind, elapsed_ms, operations.len(), context);
    }
    fn complete_open(&mut self, key: &str, outcome: Outcome) -> Result<(), RuntimeError> {
        let session = self.sessions.get_mut(key).expect("opening session");
        match outcome {
            Outcome::Opened => {
                session.opened = true;
                self.metrics
                    .admitted_sessions
                    .fetch_add(1, Ordering::Relaxed);
                let _ = session.events.try_send(SessionEvent::Opened {
                    session_id: session.record.session_id.clone(),
                    worker_id: self.id,
                });
                if let Some(reply) = session.open_reply.take() {
                    let _ = reply.send(Ok(()));
                }
            }
            Outcome::Failed { code, message } => {
                if let Some(reply) = session.open_reply.take() {
                    let _ = reply.send(Err(RuntimeError::new(code, message)));
                }
                if !session.closing {
                    self.metrics.active_sessions.fetch_sub(1, Ordering::Relaxed);
                }
                self.sessions.remove(key);
                self.load.store(self.sessions.len(), Ordering::Relaxed);
            }
            _ => return Err(protocol_error("unexpected open result")),
        }
        Ok(())
    }
    fn complete_close(&mut self, key: &str, outcome: Outcome) -> Result<(), RuntimeError> {
        let session = self.sessions.get_mut(key).expect("closing session");
        session.backend_closed = true;
        let failure = match outcome {
            Outcome::Closed => None,
            Outcome::Failed { code, message } => Some(RuntimeError::new(code, message)),
            _ => Some(protocol_error("unexpected close result")),
        };
        if session.close_reply.is_some() || session.events.is_closed() {
            self.remove_closed(key);
        }
        failure.map_or(Ok(()), Err)
    }
    fn complete_inference(
        &mut self,
        key: &str,
        turn_id: u64,
        generation: u64,
        result: OperationResult,
    ) -> Result<(), RuntimeError> {
        if result.turn_id != Some(turn_id) || result.generation != Some(generation) {
            return Err(protocol_error("turn/generation mismatch in backend result"));
        }
        let session = self.sessions.get_mut(key).expect("inference session");
        match result.outcome {
            Outcome::Failed { code, message } => {
                if !session.closing {
                    session.finish(if code == ErrorCode::ContextLimit {
                        FinishReason::ContextLimit
                    } else {
                        FinishReason::BackendFailed
                    });
                    session.closing = true;
                    self.metrics.active_sessions.fetch_sub(1, Ordering::Relaxed);
                }
                let _ = session.events.try_send(SessionEvent::Failed {
                    turn_id: session.turn().map(|turn| turn.turn_id),
                    code,
                    message,
                });
                session.cancellation.cancel();
                self.metrics
                    .backend_failures
                    .fetch_add(1, Ordering::Relaxed);
            }
            Outcome::Token {
                token_id,
                text_delta,
                eos,
                context_tokens,
            } => {
                session.context_tokens = context_tokens;
                if session.generation() != generation || session.closing || !session.active() {
                    self.metrics.stale_results.fetch_add(1, Ordering::Relaxed);
                } else {
                    self.accept_token(key, token_id, text_delta, eos, context_tokens)?;
                }
            }
            _ => return Err(protocol_error("unexpected inference result")),
        }
        Ok(())
    }
    fn complete_activation(
        &mut self,
        key: &str,
        turn_id: u64,
        generation: u64,
        result: OperationResult,
    ) -> Result<(), RuntimeError> {
        if result.turn_id != Some(turn_id) || result.generation != Some(generation) {
            return Err(protocol_error("activation turn/generation mismatch"));
        }
        if matches!(result.outcome, Outcome::Failed { .. }) {
            let session = self.sessions.get_mut(key).expect("activating session");
            if matches!(
                session.preparation,
                crate::session::state::Preparation::Ready(_)
            ) {
                session.preparation.invalidate();
            }
            self.metrics
                .preparation_fallbacks
                .fetch_add(1, Ordering::Relaxed);
            return Ok(());
        }
        let session = self.sessions.get_mut(key).expect("activating session");
        session.pending_token = None;
        session.preparation = match session.preparation {
            crate::session::state::Preparation::DiscardPending { next } => next.map_or(
                crate::session::state::Preparation::None,
                crate::session::state::Preparation::Queued,
            ),
            _ => crate::session::state::Preparation::None,
        };
        self.metrics
            .preparations_activated
            .fetch_add(1, Ordering::Relaxed);
        self.complete_inference(key, turn_id, generation, result)
    }
}
fn protocol_error(message: &str) -> RuntimeError {
    RuntimeError::new(ErrorCode::BackendFailed, message)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        config::RuntimeConfig,
        metrics::Metrics,
        protocol::{
            SessionId, TurnId, TurnRecord,
            backend::{Memory, Ready, Timing},
        },
        session::state::{SessionState, Stage},
        worker::Command,
    };
    use std::sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, AtomicUsize},
    };
    use tokio::{
        sync::{mpsc, oneshot},
        time::Instant,
    };
    use tokio_util::sync::CancellationToken;

    #[tokio::test]
    async fn interrupted_prefill_cost_uses_original_audio_and_cache_prefix() {
        let configuration = RuntimeConfig::default();
        let ready = Ready {
            r#type: "ready".into(),
            protocol_version: 1,
            body_bytes: 0,
            model_id: "test-model".into(),
            max_context_tokens: configuration.max_context_tokens,
            max_batch_size: configuration.max_batch_size,
            max_audio_samples: configuration.max_audio_samples,
        };
        let metrics = Arc::new(Metrics::new(1));
        let mut worker = WorkerActor::new(
            0,
            configuration,
            ready,
            metrics.clone(),
            Arc::new(AtomicU64::new(1)),
            Arc::new(AtomicUsize::new(1)),
            Arc::new(AtomicBool::new(true)),
        );
        let (events, _receiver) = mpsc::channel(8);
        let mut session = SessionState::new(
            SessionId("interrupted-prefill".into()),
            0,
            "test-model".into(),
            events,
            CancellationToken::new(),
        );
        session.opened = true;
        session.context_tokens = 100;
        session.record.turns.push(TurnRecord {
            turn_id: TurnId(1),
            audio_pcm16: vec![0; 30 * 16_000 * 2],
            tokens: Vec::new(),
            finish_reason: None,
            committed: true,
        });
        session.stage = Stage::Prefill {
            queued_at: Instant::now(),
        };
        worker.sessions.insert("session-key".into(), session);
        let job = worker.build_batch(BatchKind::Prefill, vec!["session-key".into()]);
        let (reply, response) = oneshot::channel();
        worker.command(Command::Begin {
            key: "session-key".into(),
            turn_id: TurnId(2),
            reply,
        });
        response.await.unwrap().unwrap();
        worker
            .complete(Completion {
                elapsed_ms: 300.0,
                response: Ok(BatchResponse {
                    request_id: job.request.request_id,
                    body_bytes: 0,
                    results: vec![OperationResult {
                        operation_id: job.request.operations[0].operation_id(),
                        session_id: "session-key".into(),
                        turn_id: Some(1),
                        generation: Some(0),
                        outcome: Outcome::Token {
                            token_id: 100,
                            text_delta: "stale".into(),
                            eos: false,
                            context_tokens: 412,
                        },
                    }],
                    timing: Timing::default(),
                    memory: Memory::default(),
                }),
            })
            .unwrap();
        let session = &worker.sessions["session-key"];
        assert_eq!(session.context_tokens, 412);
        assert!(session.turn().unwrap().audio_pcm16.is_empty());
        assert!(session.turn().unwrap().tokens.is_empty());
        assert_eq!(metrics.snapshot().stale_results_discarded, 1);
        assert_eq!(worker.costs.estimate(BatchKind::Prefill, 1, 432), 330.0);
    }
}
