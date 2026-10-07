//! Builds homogeneous batches and packs audio bodies for the execution task.

use super::{
    actor::{InFlightBatch, WorkerActor},
    execution::BatchJob,
};
use crate::{
    metrics::Metrics,
    protocol::backend::{BatchRequest, Operation},
    scheduler::BatchKind,
    session::state::{Preparation, SessionState, Stage},
};
use std::sync::atomic::Ordering;

impl WorkerActor {
    pub(super) fn build_batch(&mut self, kind: BatchKind, session_keys: Vec<String>) -> BatchJob {
        let mut operations = Vec::with_capacity(session_keys.len());
        let mut audio = Vec::new();
        for session_key in session_keys {
            operations.push(self.build_operation(kind, session_key, &mut audio));
        }

        let request = BatchRequest {
            request_id: self.next_operation_id,
            body_bytes: audio.len(),
            operations,
        };
        // The actor retains identities until completion; the execution task owns its request.
        let job_request = request.clone();
        self.observe_dispatch(kind, request.operations.len());
        self.in_flight_batch = Some(InFlightBatch { request, kind });
        if kind == BatchKind::Decode {
            self.consecutive_decode_batches += 1;
        } else {
            self.consecutive_decode_batches = 0;
        }
        BatchJob {
            request: job_request,
            audio,
            prepared_at: tokio::time::Instant::now(),
        }
    }

    fn build_operation(
        &mut self,
        kind: BatchKind,
        session_key: String,
        audio: &mut Vec<u8>,
    ) -> Operation {
        self.next_operation_id += 1;
        let operation_id = self.next_operation_id;
        let session = self
            .sessions
            .get_mut(&session_key)
            .expect("selected session");
        session.in_flight = true;
        match kind {
            BatchKind::Open => Operation::Open {
                operation_id,
                session_id: session_key,
            },
            BatchKind::Close => Operation::Close {
                operation_id,
                session_id: session_key,
            },
            BatchKind::Discard => {
                let Preparation::DiscardPending { next } = session.preparation else {
                    unreachable!("selected discard")
                };
                session.preparation = Preparation::Discarding { next };
                Operation::DiscardPrepared {
                    operation_id,
                    session_id: session_key,
                }
            }
            BatchKind::Activate => Operation::Activate {
                operation_id,
                session_id: session_key,
                turn_id: session.current_turn().expect("activation turn").turn_id.0,
                generation: session.generation,
            },
            BatchKind::Prefill => {
                prefill_operation(operation_id, session_key, session, audio, &self.metrics)
            }
            BatchKind::Decode => Operation::Decode {
                operation_id,
                session_id: session_key,
                turn_id: session.current_turn().expect("decode turn").turn_id.0,
                generation: session.generation,
                accepted: session
                    .pending_token
                    .take()
                    .expect("eligible accepted token"),
            },
        }
    }

    fn observe_dispatch(&self, kind: BatchKind, item_count: usize) {
        self.metrics.batches.fetch_add(1, Ordering::Relaxed);
        self.metrics
            .batch_items
            .fetch_add(item_count as u64, Ordering::Relaxed);
        if kind == BatchKind::Decode {
            self.metrics.decode_batches.fetch_add(1, Ordering::Relaxed);
            self.metrics
                .decode_items
                .fetch_add(item_count as u64, Ordering::Relaxed);
            self.metrics.decode_slots.fetch_add(
                self.configuration
                    .max_batch_size
                    .min(self.backend_capabilities.max_batch_size) as u64,
                Ordering::Relaxed,
            );
        }
    }
}

fn prefill_operation(
    operation_id: u64,
    session_key: String,
    session: &mut SessionState,
    audio: &mut Vec<u8>,
    metrics: &Metrics,
) -> Operation {
    let turn = session.current_turn().expect("prefill turn");
    let turn_id = turn.turn_id.0;
    let offset = audio.len();
    let bytes = turn.audio_pcm16.len();
    audio.extend_from_slice(&turn.audio_pcm16);
    if let Stage::Prefill { queued_at } = session.stage {
        metrics
            .queue_delay
            .record(queued_at.elapsed().as_secs_f64() * 1000.0);
    }
    if let Preparation::Queued(request) = session.preparation {
        metrics.preparations_started.fetch_add(1, Ordering::Relaxed);
        session.preparation = Preparation::Running(request.snapshot);
        Operation::Prepare {
            operation_id,
            session_id: session_key,
            turn_id,
            generation: request.snapshot.generation,
            audio_offset: offset,
            audio_bytes: bytes,
            accepted: session.pending_token.clone(),
        }
    } else {
        Operation::Prefill {
            operation_id,
            session_id: session_key,
            turn_id,
            generation: session.generation,
            audio_offset: offset,
            audio_bytes: bytes,
            accepted: session.pending_token.take(),
        }
    }
}
