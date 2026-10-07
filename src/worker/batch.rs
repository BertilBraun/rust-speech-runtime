use super::actor::{ActiveBatch, Job, WorkerActor};
use crate::{
    protocol::backend::{BatchRequest, Operation},
    scheduler::BatchKind,
    session::state::{Preparation, Stage},
};
use std::sync::atomic::Ordering;

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
                BatchKind::Discard => {
                    let Preparation::DiscardPending { next } = session.preparation else {
                        unreachable!("selected discard")
                    };
                    session.preparation = Preparation::Discarding { next };
                    Operation::DiscardPrepared {
                        operation_id,
                        session_id: key,
                    }
                }
                BatchKind::Activate => Operation::Activate {
                    operation_id,
                    session_id: key,
                    turn_id: session.turn().expect("activation turn").turn_id.0,
                    generation: session.generation(),
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
                    if let Preparation::Queued(request) = session.preparation {
                        self.metrics
                            .preparations_started
                            .fetch_add(1, Ordering::Relaxed);
                        session.preparation = Preparation::Running(request.snapshot);
                        Operation::Prepare {
                            operation_id,
                            session_id: key,
                            turn_id,
                            generation: request.snapshot.generation,
                            audio_offset: offset,
                            audio_bytes: bytes,
                            accepted: session.pending_token.clone(),
                        }
                    } else {
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
}
