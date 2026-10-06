use super::actor::{ActiveBatch, Completion, Job, WorkerActor};
use crate::{
    protocol::{
        ErrorCode, FinishReason, SessionEvent, TokenRecord,
        backend::{AcceptedToken, BatchRequest, Operation, Outcome},
    },
    runtime::RuntimeError,
    scheduler::BatchKind,
    session::state::{SessionState, Stage},
};
use std::sync::atomic::Ordering;
use tokio::time::Instant;

impl WorkerActor {
    pub fn build_batch(&mut self, kind: BatchKind, keys: Vec<String>) -> Job {
        let mut operations = Vec::with_capacity(keys.len());
        let mut audio = Vec::new();
        for key in keys {
            self.next_operation += 1;
            let operation_id = self.next_operation;
            let session = self.sessions.get_mut(&key).expect("selected session");
            session.in_flight = true;
            let operation = match kind {
                BatchKind::Open => Operation::Open {
                    operation_id,
                    session_id: key,
                },
                BatchKind::Close => Operation::Close {
                    operation_id,
                    session_id: key,
                },
                BatchKind::Prefill => {
                    let turn = session.turn().expect("prefill turn");
                    let turn_id = turn.turn_id.0;
                    let offset = audio.len();
                    let bytes = turn.audio_pcm16.len();
                    audio.extend_from_slice(&turn.audio_pcm16);
                    if let Stage::Prefill { queued_at } = session.stage {
                        self.metrics
                            .queue_delay
                            .record(queued_at.elapsed().as_secs_f64() * 1000.0);
                    }
                    Operation::Prefill {
                        operation_id,
                        session_id: key,
                        turn_id,
                        generation: session.generation(),
                        audio_offset: offset,
                        audio_bytes: bytes,
                        accepted: session.pending_token.take(),
                    }
                }
                BatchKind::Decode => Operation::Decode {
                    operation_id,
                    session_id: key,
                    turn_id: session.turn().expect("decode turn").turn_id.0,
                    generation: session.generation(),
                    accepted: session
                        .pending_token
                        .take()
                        .expect("eligible accepted token"),
                },
            };
            operations.push(operation);
        }
        let request = BatchRequest {
            request_id: self.next_operation,
            body_bytes: audio.len(),
            operations,
        };
        let job_request = BatchRequest {
            request_id: request.request_id,
            body_bytes: request.body_bytes,
            operations: request.operations.clone(),
        };
        self.metrics.batches.fetch_add(1, Ordering::Relaxed);
        self.metrics
            .batch_items
            .fetch_add(request.operations.len() as u64, Ordering::Relaxed);
        if kind == BatchKind::Decode {
            self.metrics.decode_batches.fetch_add(1, Ordering::Relaxed);
            self.metrics
                .decode_items
                .fetch_add(request.operations.len() as u64, Ordering::Relaxed);
            self.metrics.decode_slots.fetch_add(
                self.config.max_batch_size.min(self.ready.max_batch_size) as u64,
                Ordering::Relaxed,
            );
        }
        self.active = Some(ActiveBatch { request, kind });
        if kind == BatchKind::Decode {
            self.consecutive_decode_batches += 1;
        } else {
            self.consecutive_decode_batches = 0;
        }
        Job {
            request: job_request,
            audio,
        }
    }
    pub fn complete(&mut self, completion: Completion) -> Result<(), RuntimeError> {
        let active = self.active.take().expect("active batch");
        let response = completion.response?;
        self.metrics.worker_observation(
            self.id,
            completion.elapsed_ms,
            response.memory.allocated_bytes,
            response.memory.reserved_bytes,
        );
        if response.timing.encode_ms > 0.0 {
            self.metrics.encode.record(response.timing.encode_ms);
        }
        if response.timing.prefill_ms > 0.0 {
            self.metrics.prefill.record(response.timing.prefill_ms);
        }
        if response.timing.decode_ms > 0.0 {
            self.metrics.decode.record(response.timing.decode_ms);
        }
        let context = active
            .request
            .operations
            .iter()
            .filter_map(|operation| {
                self.sessions
                    .get(operation.session_id())
                    .map(|session| session.context_tokens)
            })
            .max()
            .unwrap_or(0);
        self.costs.observe(
            active.kind,
            completion.elapsed_ms,
            active.request.operations.len(),
            context,
        );
        self.metrics.inference.record(completion.elapsed_ms);
        for (operation, result) in active.request.operations.into_iter().zip(response.results) {
            let key = operation.session_id().to_owned();
            let Some(session) = self.sessions.get_mut(&key) else {
                return Err(RuntimeError::new(
                    ErrorCode::BackendFailed,
                    "completion for unknown session",
                ));
            };
            session.in_flight = false;
            match operation {
                Operation::Open { .. } => match result.outcome {
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
                        self.metrics.active_sessions.fetch_sub(1, Ordering::Relaxed);
                        self.sessions.remove(&key);
                        self.load.store(self.sessions.len(), Ordering::Relaxed);
                    }
                    _ => {
                        return Err(RuntimeError::new(
                            ErrorCode::BackendFailed,
                            "unexpected open result",
                        ));
                    }
                },
                Operation::Close { .. } => match result.outcome {
                    Outcome::Closed => {
                        session.backend_closed = true;
                        if session.close_reply.is_some() || session.events.is_closed() {
                            self.remove_closed(&key);
                        }
                    }
                    Outcome::Failed { message, .. } => {
                        self.remove_closed(&key);
                        return Err(RuntimeError::new(ErrorCode::BackendFailed, message));
                    }
                    _ => {
                        return Err(RuntimeError::new(
                            ErrorCode::BackendFailed,
                            "unexpected close result",
                        ));
                    }
                },
                Operation::Prefill {
                    turn_id,
                    generation,
                    ..
                }
                | Operation::Decode {
                    turn_id,
                    generation,
                    ..
                } => {
                    if result.turn_id != Some(turn_id) || result.generation != Some(generation) {
                        return Err(RuntimeError::new(
                            ErrorCode::BackendFailed,
                            "turn/generation mismatch in backend result",
                        ));
                    }
                    if let Outcome::Token { context_tokens, .. } = &result.outcome {
                        session.context_tokens = *context_tokens;
                    }
                    if session.generation() != generation || session.closing || !session.active() {
                        self.metrics.stale_results.fetch_add(1, Ordering::Relaxed);
                        continue;
                    }
                    match result.outcome {
                        Outcome::Token {
                            token_id,
                            text_delta,
                            eos,
                            context_tokens,
                        } => self.accept_token(&key, token_id, text_delta, eos, context_tokens)?,
                        Outcome::Failed { code, message } => {
                            session.finish(if code == ErrorCode::ContextLimit {
                                FinishReason::ContextLimit
                            } else {
                                FinishReason::BackendFailed
                            });
                            let _ = session.events.try_send(SessionEvent::Failed {
                                turn_id: session.turn().map(|turn| turn.turn_id),
                                code,
                                message,
                            });
                            self.metrics
                                .backend_failures
                                .fetch_add(1, Ordering::Relaxed);
                        }
                        _ => {
                            return Err(RuntimeError::new(
                                ErrorCode::BackendFailed,
                                "unexpected inference result",
                            ));
                        }
                    }
                }
            }
        }
        Ok(())
    }
    fn accept_token(
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
