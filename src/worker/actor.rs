use std::{collections::HashMap, sync::Arc};

use tokio::{
    sync::{mpsc, oneshot},
    task::JoinHandle,
    time::Instant,
};
use tokio_util::sync::CancellationToken;

use crate::config::RuntimeConfig;
use crate::metrics::WorkerMeasurements;
use crate::protocol::{Assignment, Generation, InferenceOutput, InputFrame, SessionId, WorkerId};
use crate::scheduler::deadline::{ReadySession, construct_batch};

use super::cache::{CacheHandle, CachePool};
use super::mock_gpu::{self, Batch, BatchResult, WorkItem};

pub(crate) enum WorkerCommand {
    AddSession {
        session_id: SessionId,
        assignment: Assignment,
        deadline: Instant,
        reply: oneshot::Sender<bool>,
    },
    InputReady {
        session_id: SessionId,
        generation: Generation,
        input: InputFrame,
    },
    RemoveSession {
        session_id: SessionId,
        generation: Generation,
        reply: oneshot::Sender<()>,
    },
}

struct WorkerSession {
    assignment: Assignment,
    next_deadline: Instant,
    cache: CacheHandle,
    pending_input: Option<InputFrame>,
    in_flight: bool,
}

struct Worker {
    configuration: Arc<RuntimeConfig>,
    sessions: HashMap<SessionId, WorkerSession>,
    cache: CachePool,
    measurements: WorkerMeasurements,
    outputs: mpsc::Sender<InferenceOutput>,
    active_batch: bool,
    epoch: Instant,
}

pub(crate) fn spawn_worker(
    worker_id: WorkerId,
    configuration: Arc<RuntimeConfig>,
    outputs: mpsc::Sender<InferenceOutput>,
    cancellation: CancellationToken,
    epoch: Instant,
) -> (mpsc::Sender<WorkerCommand>, JoinHandle<WorkerMeasurements>) {
    let (commands, mailbox) = mpsc::channel(configuration.worker_channel_capacity);
    let worker = Worker {
        cache: CachePool::new(worker_id, configuration.cache_slots_per_worker),
        configuration,
        sessions: HashMap::new(),
        measurements: WorkerMeasurements::new(worker_id),
        outputs,
        active_batch: false,
        epoch,
    };
    (commands, tokio::spawn(worker.run(mailbox, cancellation)))
}

impl Worker {
    async fn run(
        mut self,
        mut commands: mpsc::Receiver<WorkerCommand>,
        cancellation: CancellationToken,
    ) -> WorkerMeasurements {
        let (batches, batch_mailbox) = mpsc::channel(1);
        let (device_results, mut results) = mpsc::channel(1);
        let device = tokio::spawn(mock_gpu::run(
            batch_mailbox,
            device_results,
            self.configuration.inference_latency,
            self.configuration.slowdown.clone(),
            self.epoch,
        ));
        loop {
            let wakeup = self.next_wakeup();
            tokio::select! {
                biased;
                _ = cancellation.cancelled() => break,
                result = results.recv(), if self.active_batch => {
                    self.complete(result.expect("device returns every active batch"));
                }
                _ = tokio::time::sleep_until(wakeup), if !self.active_batch => {
                    if let Some(batch) = self.build_batch() {
                        self.active_batch = true;
                        batches.send(batch).await.expect("device lives until scheduler shuts down");
                    }
                }
                command = commands.recv() => {
                    let Some(command) = command else { break; };
                    if !self.configuration.worker_control_delay.is_zero() {
                        tokio::select! {
                            _ = cancellation.cancelled() => break,
                            _ = tokio::time::sleep(self.configuration.worker_control_delay) => {}
                        }
                    }
                    self.handle(command);
                }
            }
        }
        self.sessions.clear();
        if self.active_batch {
            self.complete(results.recv().await.expect("drain running device batch"));
        }
        drop(batches);
        device.await.expect("mock device does not panic");
        self.measurements.elapsed = Instant::now().duration_since(self.epoch);
        self.measurements
    }

    fn handle(&mut self, command: WorkerCommand) {
        match command {
            WorkerCommand::AddSession {
                session_id,
                assignment,
                deadline,
                reply,
            } => {
                assert_eq!(assignment.worker_id, self.measurements.worker_id);
                assert!(!self.sessions.contains_key(&session_id));
                let Some(cache) = self.cache.allocate() else {
                    let _ = reply.send(false);
                    return;
                };
                self.sessions.insert(
                    session_id,
                    WorkerSession {
                        assignment,
                        next_deadline: deadline,
                        cache,
                        pending_input: None,
                        in_flight: false,
                    },
                );
                self.measurements.peak_sessions =
                    self.measurements.peak_sessions.max(self.sessions.len());
                let _ = reply.send(true);
            }
            WorkerCommand::InputReady {
                session_id,
                generation,
                input,
            } => {
                if Instant::now().duration_since(input.timestamp) > self.configuration.max_input_age
                {
                    self.measurements.stale_inputs += 1;
                    return;
                }
                if let Some(session) = self.sessions.get_mut(&session_id)
                    && session.assignment.generation == generation
                    && session.pending_input.replace(input).is_some()
                {
                    self.measurements.coalesced_inputs += 1;
                }
            }
            WorkerCommand::RemoveSession {
                session_id,
                generation,
                reply,
            } => {
                if self
                    .sessions
                    .get(&session_id)
                    .is_some_and(|session| session.assignment.generation == generation)
                {
                    let session = self.sessions.remove(&session_id).expect("session exists");
                    self.cache.free(session.cache);
                }
                let _ = reply.send(());
            }
        }
    }

    fn next_wakeup(&self) -> Instant {
        let latency = self.configuration.inference_latency + self.configuration.scheduling_margin;
        let ready: Vec<&WorkerSession> = self
            .sessions
            .values()
            .filter(|session| session.pending_input.is_some() && !session.in_flight)
            .collect();
        if ready.len() >= self.configuration.batch_size {
            return Instant::now();
        }
        ready
            .into_iter()
            .map(|session| session.next_deadline - latency)
            .min()
            .unwrap_or_else(|| Instant::now() + self.configuration.session_timeout)
    }

    fn build_batch(&mut self) -> Option<Batch> {
        let now = Instant::now();
        for session in self.sessions.values_mut() {
            if session.pending_input.as_ref().is_some_and(|input| {
                now.duration_since(input.timestamp) > self.configuration.max_input_age
            }) {
                session.pending_input = None;
                self.measurements.stale_inputs += 1;
            }
        }
        let ready = self
            .sessions
            .iter()
            .filter(|(_, session)| session.pending_input.is_some() && !session.in_flight)
            .map(|(session_id, session)| ReadySession {
                session_id: *session_id,
                deadline: session.next_deadline,
            })
            .collect();
        let selected = construct_batch(ready, self.configuration.batch_size);
        if selected.is_empty() {
            return None;
        }
        let items = selected
            .into_iter()
            .map(|selected| {
                let session = self
                    .sessions
                    .get_mut(&selected.session_id)
                    .expect("selected session exists");
                self.cache.assert_owned(session.cache);
                session.in_flight = true;
                WorkItem {
                    session_id: selected.session_id,
                    assignment: session.assignment,
                    input: session.pending_input.take().expect("selected input exists"),
                    deadline: session.next_deadline,
                }
            })
            .collect();
        Some(Batch { items })
    }

    fn complete(&mut self, result: BatchResult) {
        self.active_batch = false;
        let inference_latency = result.completed_at.duration_since(result.started_at);
        self.measurements.batches += 1;
        self.measurements.processed_frames += result.batch.items.len() as u64;
        self.measurements
            .batch_sizes
            .record(result.batch.items.len() as u64)
            .expect("batch size is representable");
        self.measurements.busy_time += inference_latency;
        self.measurements
            .inference_latency
            .record(inference_latency);
        for item in result.batch.items {
            let Some(session) = self
                .sessions
                .get_mut(&item.session_id)
                .filter(|session| session.assignment == item.assignment)
            else {
                self.measurements.stale_results += 1;
                continue;
            };
            session.in_flight = false;
            session.next_deadline += self.configuration.tick_interval;
            self.measurements.valid_results += 1;
            self.measurements
                .queue_delay
                .record(result.started_at.duration_since(item.input.timestamp));
            self.measurements
                .end_to_end_latency
                .record(result.completed_at.duration_since(item.input.timestamp));
            if result.completed_at > item.deadline {
                self.measurements.deadline_misses += 1;
                self.measurements
                    .deadline_lateness
                    .record(result.completed_at.duration_since(item.deadline));
            }
            let output = InferenceOutput {
                session_id: item.session_id,
                assignment: item.assignment,
                input_timestamp: item.input.timestamp,
                completed_at: result.completed_at,
            };
            if self.outputs.try_send(output).is_err() {
                self.measurements.result_channel_saturation += 1;
            }
        }
    }
}
