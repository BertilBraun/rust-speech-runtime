//! Correlates backend results, observes cost and fences stale inference proposals.

use super::{actor::WorkerActor, execution::BatchCompletion};
use crate::{
    protocol::{
        ErrorCode, FinishReason, SessionEvent,
        backend::{BatchResponse, Operation, OperationResult, Outcome},
    },
    runtime::RuntimeError,
    scheduler::BatchKind,
    session::state::Preparation,
};
use std::sync::atomic::Ordering;

impl WorkerActor {
    pub(super) fn apply_completion(
        &mut self,
        completion: BatchCompletion,
    ) -> Result<(), RuntimeError> {
        let in_flight_batch = self
            .in_flight_batch
            .take()
            .expect("batch awaiting completion");
        let response = completion.response?;
        self.observe_batch(
            &in_flight_batch.request.operations,
            in_flight_batch.kind,
            completion.elapsed_ms,
            &response,
        );
        for (operation, result) in in_flight_batch
            .request
            .operations
            .into_iter()
            .zip(response.results)
        {
            let session_key = operation.session_id();
            self.sessions
                .get_mut(session_key)
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
            self.worker_id,
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
        self.metrics
            .runtime_timing
            .backend_rpc_overhead
            .record((elapsed_ms - response.timing.elapsed_ms).max(0.0));
        let input_context_tokens = operations
            .iter()
            .filter_map(|operation| self.operation_input_context(operation))
            .max()
            .unwrap_or(0);
        self.forward_costs
            .observe(kind, elapsed_ms, operations.len(), input_context_tokens);
    }

    fn operation_input_context(&self, operation: &Operation) -> Option<usize> {
        let session = self.sessions.get(operation.session_id())?;
        let appended_tokens = match operation {
            Operation::Prefill { audio_bytes, .. } | Operation::Prepare { audio_bytes, .. } => {
                audio_bytes.div_ceil(3200) + 32
            }
            _ => 0,
        };
        // One in-flight operation keeps this cache length stable until completion.
        Some(session.context_tokens.saturating_add(appended_tokens))
    }

    fn complete_open(&mut self, session_key: &str, outcome: Outcome) -> Result<(), RuntimeError> {
        let session = self.sessions.get_mut(session_key).expect("opening session");
        match outcome {
            Outcome::Opened => {
                session.opened = true;
                self.metrics
                    .admitted_sessions
                    .fetch_add(1, Ordering::Relaxed);
                let _ = session.events.try_send(SessionEvent::Opened {
                    session_id: session.record.session_id.clone(),
                    worker_id: self.worker_id,
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
                self.sessions.remove(session_key);
                self.session_count
                    .store(self.sessions.len(), Ordering::Relaxed);
            }
            _ => return Err(protocol_error("unexpected open result")),
        }
        Ok(())
    }

    fn complete_close(&mut self, session_key: &str, outcome: Outcome) -> Result<(), RuntimeError> {
        let session = self.sessions.get_mut(session_key).expect("closing session");
        session.backend_closed = true;
        let failure = match outcome {
            Outcome::Closed => None,
            Outcome::Failed { code, message } => Some(RuntimeError::new(code, message)),
            _ => Some(protocol_error("unexpected close result")),
        };
        if session.close_reply.is_some() || session.events.is_closed() {
            self.remove_closed(session_key);
        }
        failure.map_or(Ok(()), Err)
    }

    fn complete_inference(
        &mut self,
        session_key: &str,
        turn_id: u64,
        generation: u64,
        result: OperationResult,
    ) -> Result<(), RuntimeError> {
        if result.turn_id != Some(turn_id) || result.generation != Some(generation) {
            return Err(protocol_error("turn/generation mismatch in backend result"));
        }
        let session = self
            .sessions
            .get_mut(session_key)
            .expect("inference session");
        match result.outcome {
            Outcome::Failed { code, message } => {
                if !session.closing {
                    session.finish_turn(if code == ErrorCode::ContextLimit {
                        FinishReason::ContextLimit
                    } else {
                        FinishReason::BackendFailed
                    });
                    session.closing = true;
                    self.metrics.active_sessions.fetch_sub(1, Ordering::Relaxed);
                }
                let _ = session.events.try_send(SessionEvent::Failed {
                    turn_id: session.current_turn().map(|turn| turn.turn_id),
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
                if session.generation != generation || session.closing || !session.has_active_turn()
                {
                    self.metrics.stale_results.fetch_add(1, Ordering::Relaxed);
                } else {
                    self.accept_token(session_key, token_id, text_delta, eos, context_tokens)?;
                }
            }
            _ => return Err(protocol_error("unexpected inference result")),
        }
        Ok(())
    }

    fn complete_activation(
        &mut self,
        session_key: &str,
        turn_id: u64,
        generation: u64,
        result: OperationResult,
    ) -> Result<(), RuntimeError> {
        if result.turn_id != Some(turn_id) || result.generation != Some(generation) {
            return Err(protocol_error("activation turn/generation mismatch"));
        }
        if matches!(result.outcome, Outcome::Failed { .. }) {
            let session = self
                .sessions
                .get_mut(session_key)
                .expect("activating session");
            if matches!(session.preparation, Preparation::Ready(_)) {
                session.preparation.invalidate();
            }
            self.metrics
                .preparation_fallbacks
                .fetch_add(1, Ordering::Relaxed);
            return Ok(());
        }
        let session = self
            .sessions
            .get_mut(session_key)
            .expect("activating session");
        session.pending_token = None;
        session.preparation = match session.preparation {
            Preparation::DiscardPending { next } => {
                next.map_or(Preparation::None, Preparation::Queued)
            }
            _ => Preparation::None,
        };
        self.metrics
            .preparations_activated
            .fetch_add(1, Ordering::Relaxed);
        self.complete_inference(session_key, turn_id, generation, result)
    }
}

fn protocol_error(message: &str) -> RuntimeError {
    RuntimeError::new(ErrorCode::BackendFailed, message)
}

#[cfg(test)]
mod tests;
