use super::{
    cache::{CacheHandle, CachePool},
    mock_gpu::{self, DeviceJob, DeviceResult, DeviceWork, SessionCancellation, WorkItem},
};
use crate::{
    config::RuntimeConfig,
    metrics::WorkerMeasurements,
    metrics::profile::{PacketTimings, SlowWorkerPacket},
    protocol::{
        Assignment, AudioContext, AudioResult, CacheOutcome, FrameRejection, Generation,
        InferenceOutput, InputFrame, InputOutcome, PrefixState, SessionId, SessionLease, WorkerId,
    },
    scheduler::{
        admission::{ServiceEstimator, WorkerStatus, session_limit},
        deadline::{ReadySession, construct_batch},
    },
};
use std::{
    collections::{HashMap, HashSet, VecDeque},
    sync::Arc,
    time::Duration,
};
use tokio::{
    sync::{mpsc, oneshot, watch},
    task::JoinHandle,
    time::Instant,
};
use tokio_util::sync::CancellationToken;

pub(crate) enum WorkerCommand {
    AddSession {
        session_id: SessionId,
        assignment: Assignment,
        reply: oneshot::Sender<bool>,
    },
    InputReady {
        session_id: SessionId,
        generation: Generation,
        input: InputFrame,
        reply: oneshot::Sender<InputOutcome>,
        routed_at: Instant,
    },
    RemoveSession {
        session_id: SessionId,
        generation: Generation,
        reply: oneshot::Sender<()>,
    },
    EvictCache {
        session_id: SessionId,
        generation: Generation,
        reply: oneshot::Sender<bool>,
    },
}
pub(crate) struct WorkerHandle {
    pub commands: mpsc::Sender<WorkerCommand>,
    pub status: watch::Receiver<WorkerStatus>,
    pub task: JoinHandle<WorkerMeasurements>,
}
struct PendingFrame {
    item: WorkItem,
    queued_at: Instant,
}
struct WorkerSession {
    assignment: Assignment,
    cache: CacheHandle,
    prefix: PrefixState,
    pending: Option<PendingFrame>,
    in_flight: bool,
    cancellation: SessionCancellation,
}
struct SubmittedBatch {
    submitted_at: Instant,
    latency: Duration,
    expected_completion: Instant,
}
struct Worker {
    configuration: Arc<RuntimeConfig>,
    sessions: HashMap<SessionId, WorkerSession>,
    cache: CachePool,
    retired_cache: HashMap<SessionLease, CacheHandle>,
    measurements: WorkerMeasurements,
    estimator: ServiceEstimator,
    status: watch::Sender<WorkerStatus>,
    submitted: VecDeque<SubmittedBatch>,
    prepared: Vec<ReadySession>,
    session_limit: usize,
    epoch: Instant,
}
pub(crate) fn spawn_worker(
    worker_id: WorkerId,
    configuration: Arc<RuntimeConfig>,
    cancellation: CancellationToken,
) -> (WorkerHandle, oneshot::Receiver<()>) {
    let (commands, mailbox) = mpsc::channel(configuration.worker_channel_capacity);
    let (status, updates) = watch::channel(WorkerStatus {
        worker_id,
        sessions: HashSet::new(),
        session_limit: 0,
        service_time: configuration.inference_latency,
    });
    let (ready, response) = oneshot::channel();
    let worker = Worker {
        cache: CachePool::new(worker_id, configuration.cache_slots_per_worker),
        estimator: ServiceEstimator::new(&configuration),
        configuration,
        sessions: HashMap::new(),
        retired_cache: HashMap::new(),
        measurements: WorkerMeasurements::new(worker_id),
        status,
        submitted: VecDeque::new(),
        prepared: Vec::new(),
        session_limit: 0,
        epoch: Instant::now(),
    };
    (
        WorkerHandle {
            commands,
            status: updates,
            task: tokio::spawn(worker.run(mailbox, cancellation, ready)),
        },
        response,
    )
}
impl Worker {
    async fn run(
        mut self,
        mut commands: mpsc::Receiver<WorkerCommand>,
        cancellation: CancellationToken,
        ready: oneshot::Sender<()>,
    ) -> WorkerMeasurements {
        let (jobs, job_mailbox) = mpsc::channel(self.configuration.device_queue_capacity);
        let (device_results, mut results) =
            mpsc::channel(self.configuration.device_queue_capacity + 1);
        let wait = self.configuration.device_wait;
        let device =
            tokio::task::spawn_blocking(move || mock_gpu::run(job_mailbox, device_results, wait));
        for _ in 0..self.configuration.calibration_samples {
            jobs.send(DeviceJob {
                work: DeviceWork::Probe,
                latency: self.configuration.inference_latency,
                submitted_at: Instant::now(),
            })
            .await
            .expect("device is running");
            let result = results.recv().await.expect("probe result");
            let elapsed = result.observed_at.duration_since(result.started_at);
            self.estimator
                .observe(result.completed_at - result.started_at);
            self.estimator
                .observe_host_delay(result.observed_at - result.completed_at);
            self.measurements.calibration_latency.record(elapsed);
        }
        self.epoch = Instant::now();
        self.refresh_capacity();
        self.measurements.initial_session_limit = self.session_limit;
        let _ = ready.send(());
        let mut next_probe = self.epoch + self.configuration.probe_interval;
        loop {
            let wakeup = self.next_wakeup();
            let can_dispatch = !self.prepared.is_empty()
                && self.submitted.len() < self.configuration.device_queue_capacity + 1
                && wakeup <= Instant::now();
            tokio::select! {
                biased;
                _ = cancellation.cancelled() => break,
                result = results.recv(), if !self.submitted.is_empty() => {
                    self.complete(result.expect("device returns active work"));
                }
                permit = jobs.reserve(), if can_dispatch => self.dispatch_batch(permit.expect("device is running")),
                command = commands.recv() => {
                    let Some(command) = command else { break; };
                    if matches!(&command, WorkerCommand::InputReady { .. }) && !self.configuration.worker_input_delay.is_zero() {
                        tokio::select! {
                            _ = cancellation.cancelled() => break,
                            _ = tokio::time::sleep(self.configuration.worker_input_delay) => {}
                        }
                    }
                    self.handle(command).await;
                }
                _ = tokio::time::sleep_until(wakeup), if !can_dispatch => {},
                _ = tokio::time::sleep_until(next_probe), if self.submitted.is_empty() && self.sessions.is_empty() => {
                    let submitted_at = Instant::now();
                    let latency = self.base_latency();
                    jobs.send(DeviceJob { work: DeviceWork::Probe, latency, submitted_at }).await.expect("device is running");
                    self.track_submission(submitted_at, latency);
                    next_probe = Instant::now() + self.configuration.probe_interval;
                }
            }
        }
        let sessions: Vec<SessionId> = self.sessions.keys().copied().collect();
        for session_id in sessions {
            self.terminate(session_id, FrameRejection::Cancelled);
        }
        while !self.submitted.is_empty() {
            self.complete(results.recv().await.expect("drain physical device work"));
        }
        commands.close();
        while let Ok(command) = commands.try_recv() {
            match command {
                WorkerCommand::InputReady { reply, .. } => {
                    let _ = reply.send(InputOutcome::Rejected(FrameRejection::Cancelled));
                }
                WorkerCommand::AddSession { reply, .. } => {
                    let _ = reply.send(false);
                }
                WorkerCommand::RemoveSession { reply, .. } => {
                    let _ = reply.send(());
                }
                WorkerCommand::EvictCache { reply, .. } => {
                    let _ = reply.send(false);
                }
            }
        }
        drop(jobs);
        self.measurements.device_cpu = device.await.expect("mock device does not panic");
        self.measurements.elapsed = Instant::now().duration_since(self.epoch);
        self.measurements.final_session_limit = self.session_limit;
        self.measurements.service_time = self.estimator.service_time();
        self.measurements.host_delay = self.estimator.host_delay();
        self.measurements
    }
    fn base_latency(&self) -> Duration {
        match &self.configuration.slowdown {
            Some(slowdown) if Instant::now().duration_since(self.epoch) >= slowdown.after => {
                slowdown.inference_latency
            }
            _ => self.configuration.inference_latency,
        }
    }
    fn publish(&self) {
        let sessions: HashSet<SessionId> = self.sessions.keys().copied().collect();
        self.status.send_replace(WorkerStatus {
            worker_id: self.measurements.worker_id,
            sessions,
            session_limit: self.admission_limit(),
            service_time: self.estimator.service_time(),
        });
    }
    fn admission_limit(&self) -> usize {
        let queued = self
            .sessions
            .values()
            .filter_map(|session| session.pending.as_ref())
            .map(|pending| Instant::now().duration_since(pending.queued_at))
            .max()
            .unwrap_or(Duration::ZERO);
        self.estimator.available_limit(
            &self.configuration,
            self.session_limit,
            self.sessions.len(),
            queued,
        )
    }
    fn refresh_capacity(&mut self) {
        self.session_limit = session_limit(&self.configuration, self.estimator.service_time());
        let mut sessions: Vec<SessionId> = self.sessions.keys().copied().collect();
        sessions.sort_unstable();
        while self.sessions.len() > self.session_limit {
            self.measurements.counters.capacity_terminations += 1;
            self.terminate(
                sessions.pop().expect("excess sessions exist"),
                FrameRejection::WorkerCapacityLost,
            );
        }
        self.publish();
    }
    fn terminate(&mut self, session_id: SessionId, reason: FrameRejection) {
        if let Some(session) = self.sessions.remove(&session_id) {
            session.cancellation.cancel();
            if let Some(pending) = session.pending {
                self.measurements.counters.rejected_frames += 1;
                let _ = pending.item.reply.send(InputOutcome::Rejected(reason));
            }
            if session.in_flight {
                let prior = self.retired_cache.insert(
                    SessionLease {
                        session_id,
                        generation: session.assignment.generation,
                    },
                    session.cache,
                );
                assert!(prior.is_none(), "each cache lease is retired once");
                self.measurements.peak_retired_cache_slots = self
                    .measurements
                    .peak_retired_cache_slots
                    .max(self.retired_cache.len());
            } else {
                self.cache.free(session.cache);
            }
            self.refresh_prepared();
        }
    }
    async fn handle(&mut self, command: WorkerCommand) {
        match command {
            WorkerCommand::AddSession {
                session_id,
                assignment,
                reply,
            } => {
                assert_eq!(assignment.worker_id, self.measurements.worker_id);
                assert!(!self.sessions.contains_key(&session_id));
                if self.sessions.len() >= self.admission_limit() {
                    let _ = reply.send(false);
                    return;
                }
                let Some(cache) = self.cache.allocate() else {
                    let _ = reply.send(false);
                    return;
                };
                self.sessions.insert(
                    session_id,
                    WorkerSession {
                        assignment,
                        cache,
                        prefix: PrefixState::default(),
                        pending: None,
                        in_flight: false,
                        cancellation: SessionCancellation::default(),
                    },
                );
                self.measurements.peak_sessions =
                    self.measurements.peak_sessions.max(self.sessions.len());
                self.publish();
                let _ = reply.send(true);
            }
            WorkerCommand::InputReady {
                session_id,
                generation,
                input,
                reply,
                routed_at,
            } => {
                self.queue_input(session_id, generation, input, reply, routed_at)
                    .await
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
                    self.terminate(session_id, FrameRejection::Cancelled);
                    self.publish();
                }
                let _ = reply.send(());
            }
            WorkerCommand::EvictCache {
                session_id,
                generation,
                reply,
            } => {
                let removed = match self.sessions.get(&session_id) {
                    Some(session) if session.assignment.generation == generation => {
                        self.cache.evict(session.cache);
                        self.measurements.counters.cache_evictions += 1;
                        true
                    }
                    _ => false,
                };
                let _ = reply.send(removed);
            }
        }
    }
    fn reject(
        &mut self,
        session_id: SessionId,
        reply: oneshot::Sender<InputOutcome>,
        reason: FrameRejection,
    ) {
        self.measurements.counters.rejected_frames += 1;
        match reason {
            FrameRejection::Overloaded => self.measurements.counters.busy_rejections += 1,
            FrameRejection::InvalidSequence
            | FrameRejection::InvalidPrefix
            | FrameRejection::PrefixCapacity
            | FrameRejection::ReplayTooExpensive => {
                self.measurements.counters.prefix_rejections += 1
            }
            _ => {}
        }
        self.terminate(session_id, reason);
        self.publish();
        let _ = reply.send(InputOutcome::Rejected(reason));
    }
    async fn queue_input(
        &mut self,
        session_id: SessionId,
        generation: Generation,
        input: InputFrame,
        reply: oneshot::Sender<InputOutcome>,
        routed_at: Instant,
    ) {
        let received_at = Instant::now();
        self.measurements
            .profile
            .mailbox
            .record(received_at - routed_at);
        let Some(session) = self.sessions.get(&session_id) else {
            let _ = reply.send(InputOutcome::Rejected(FrameRejection::UnknownSession));
            return;
        };
        if session.assignment.generation != generation {
            let _ = reply.send(InputOutcome::Rejected(FrameRejection::Cancelled));
            return;
        }
        if session.in_flight || session.pending.is_some() {
            self.reject(session_id, reply, FrameRejection::Overloaded);
            return;
        }
        if input.deadline <= Instant::now() {
            self.reject(session_id, reply, FrameRejection::DeadlineExceeded);
            return;
        }
        if input.packet.sequence.0 != session.prefix.packets {
            self.reject(session_id, reply, FrameRejection::InvalidSequence);
            return;
        }
        let prior_prefix = session.prefix;
        if prior_prefix.packets >= self.configuration.audio_limits.max_prefix_packets as u64
            || prior_prefix.bytes + input.packet.payload.len() as u64
                > self.configuration.audio_limits.max_prefix_bytes as u64
        {
            self.reject(session_id, reply, FrameRejection::PrefixCapacity);
            return;
        }
        let cache = match &input.packet.context {
            AudioContext::Cached(expected) => {
                if *expected != prior_prefix {
                    self.reject(session_id, reply, FrameRejection::InvalidPrefix);
                    return;
                }
                if self.cache.lookup(session.cache) != Some(prior_prefix) {
                    self.measurements.counters.cache_misses += 1;
                    let _ = reply.send(InputOutcome::CacheMiss);
                    return;
                }
                CacheOutcome::Hit
            }
            AudioContext::Replay(prefix) => {
                if prefix.0.len() > self.configuration.audio_limits.max_prefix_packets
                    || prefix.byte_len() > self.configuration.audio_limits.max_prefix_bytes
                    || prefix
                        .0
                        .iter()
                        .any(|frame| frame.len() > self.configuration.audio_limits.max_frame_bytes)
                {
                    self.reject(session_id, reply, FrameRejection::PrefixCapacity);
                    return;
                }
                let prefix = prefix.clone();
                let state = tokio::task::spawn_blocking(move || prefix.state())
                    .await
                    .expect("prefix hashing does not panic");
                if state != prior_prefix {
                    self.reject(session_id, reply, FrameRejection::InvalidPrefix);
                    return;
                }
                CacheOutcome::Replayed {
                    packets: state.packets,
                    bytes: state.bytes,
                }
            }
        };
        let replay_duration = match cache {
            CacheOutcome::Hit => Duration::ZERO,
            CacheOutcome::Replayed { packets, .. } => self
                .configuration
                .replay_latency_per_packet
                .mul_f64(packets as f64),
        };
        if Instant::now()
            + self.estimator.device_time()
            + replay_duration
            + self.configuration.scheduling_margin
            > input.deadline
        {
            self.reject(
                session_id,
                reply,
                if replay_duration.is_zero() {
                    FrameRejection::DeadlineExceeded
                } else {
                    FrameRejection::ReplayTooExpensive
                },
            );
            return;
        }
        let prefix = prior_prefix.append(&input.packet.payload);
        let queued_at = Instant::now();
        self.measurements
            .profile
            .validation
            .record(queued_at - received_at);
        let session = self
            .sessions
            .get_mut(&session_id)
            .expect("session remains owned during validation");
        session.pending = Some(PendingFrame {
            item: WorkItem {
                session_id,
                assignment: session.assignment,
                input,
                prefix,
                cache,
                replay_duration,
                reply,
                routed_at,
                received_at,
                queued_at,
                cancellation: session.cancellation.clone(),
            },
            queued_at: Instant::now(),
        });
        self.refresh_prepared();
        self.publish();
    }
    fn refresh_prepared(&mut self) {
        let started = Instant::now();
        self.prepared = construct_batch(
            self.sessions
                .iter()
                .filter_map(|(session_id, session)| {
                    session.pending.as_ref().map(|pending| ReadySession {
                        session_id: *session_id,
                        deadline: pending.item.input.deadline,
                    })
                })
                .collect(),
            self.configuration.batch_size,
        );
        if !self.submitted.is_empty() && !self.prepared.is_empty() {
            self.measurements.counters.prepared_while_running += 1;
        }
        self.measurements
            .profile
            .batch_preparation
            .record(started.elapsed());
    }
    fn track_submission(&mut self, submitted_at: Instant, latency: Duration) {
        let start = self.submitted.back().map_or(submitted_at, |batch| {
            batch.expected_completion.max(submitted_at)
        });
        self.submitted.push_back(SubmittedBatch {
            submitted_at,
            latency,
            expected_completion: start + latency,
        });
        self.measurements.peak_device_jobs =
            self.measurements.peak_device_jobs.max(self.submitted.len());
    }
    fn next_wakeup(&self) -> Instant {
        let now = Instant::now();
        if self.prepared.is_empty()
            || self.submitted.len() > self.configuration.device_queue_capacity
        {
            return now + self.configuration.session_timeout;
        }
        if self.prepared.len() == self.configuration.batch_size {
            return now;
        }
        if let Some(previous) = self.submitted.back() {
            return previous.expected_completion - self.configuration.launch_ahead;
        }
        let mut count = 0;
        let mut wakeup = now + self.configuration.session_timeout;
        for pending in self
            .sessions
            .values()
            .filter_map(|session| session.pending.as_ref())
        {
            count += 1;
            let latest = pending.item.input.deadline
                - self.estimator.device_time()
                - pending.item.replay_duration
                - self.configuration.scheduling_margin;
            wakeup =
                wakeup.min((pending.queued_at + self.configuration.max_batch_wait).min(latest));
        }
        if count >= self.configuration.batch_size || (count > 0 && count == self.sessions.len()) {
            now
        } else {
            wakeup
        }
    }
    fn dispatch_batch(&mut self, permit: mpsc::Permit<'_, DeviceJob>) {
        let assembly_started = Instant::now();
        let selected = std::mem::take(&mut self.prepared);
        let predicted_start = self.submitted.back().map_or(Instant::now(), |batch| {
            batch.expected_completion.max(Instant::now())
        });
        let mut items = Vec::new();
        let mut replay = Duration::ZERO;
        let mut earliest_deadline = None;
        for selected in selected {
            let session = self
                .sessions
                .get_mut(&selected.session_id)
                .expect("selected session exists");
            let pending = session.pending.take().expect("pending frame exists");
            let projected = self.estimator.device_time()
                + replay
                + pending.item.replay_duration
                + self.configuration.scheduling_margin;
            let batch_deadline = earliest_deadline
                .unwrap_or(pending.item.input.deadline)
                .min(pending.item.input.deadline);
            if predicted_start + projected > batch_deadline {
                self.measurements
                    .profile
                    .rejected_queue_delay
                    .record(pending.item.input.timestamp.elapsed());
                self.reject(
                    selected.session_id,
                    pending.item.reply,
                    if pending.item.replay_duration.is_zero() {
                        FrameRejection::DeadlineExceeded
                    } else {
                        FrameRejection::ReplayTooExpensive
                    },
                );
                continue;
            }
            replay += pending.item.replay_duration;
            earliest_deadline = Some(batch_deadline);
            self.sessions
                .get_mut(&selected.session_id)
                .expect("session exists")
                .in_flight = true;
            items.push(pending.item);
        }
        if items.is_empty() {
            self.refresh_prepared();
            return;
        }
        if self
            .submitted
            .back()
            .is_some_and(|previous| previous.expected_completion > Instant::now())
        {
            self.measurements.counters.queued_batch_launches += 1;
        }
        let submitted_at = Instant::now();
        let latency = self.base_latency() + replay;
        self.measurements
            .profile
            .batch_assembly
            .record(assembly_started.elapsed());
        permit.send(DeviceJob {
            latency,
            work: DeviceWork::Inference(items),
            submitted_at,
        });
        self.track_submission(submitted_at, latency);
        self.refresh_prepared();
    }
    fn complete(&mut self, result: DeviceResult) {
        self.submitted
            .pop_front()
            .expect("completion matches submitted work");
        let mut completion = result.completed_at;
        for batch in &mut self.submitted {
            completion = completion.max(batch.submitted_at) + batch.latency;
            batch.expected_completion = completion;
        }
        let inference_latency = result.completed_at.duration_since(result.started_at);
        let host_wait = result.observed_at.duration_since(result.completed_at);
        self.estimator.observe_host_delay(host_wait);
        self.measurements
            .profile
            .host_completion_delay
            .record(host_wait);
        self.measurements.profile.host_device_wakeup.record(
            result
                .host_started_at
                .saturating_duration_since(result.started_at),
        );
        match result.work {
            DeviceWork::Probe => {
                self.estimator.observe(inference_latency);
                self.refresh_capacity();
            }
            DeviceWork::Inference(items) => {
                if result.processed_frames > 0 {
                    self.measurements.counters.batches += 1;
                    self.measurements.counters.processed_frames += result.processed_frames as u64;
                    self.measurements
                        .batch_sizes
                        .record(result.processed_frames as u64)
                        .expect("bounded batch size");
                }
                self.measurements.busy_time += inference_latency;
                let occupied_start = self
                    .measurements
                    .occupied_until
                    .map_or(result.started_at, |previous| {
                        previous.max(result.started_at)
                    });
                self.measurements.occupied_time +=
                    result.observed_at.saturating_duration_since(occupied_start);
                self.measurements.occupied_until = Some(result.observed_at);
                if result.processed_frames > 0 {
                    self.measurements
                        .inference_latency
                        .record(inference_latency);
                }
                if result.processed_frames > 0 {
                    self.estimator
                        .observe(inference_latency - result.replay_duration);
                }
                for item in items {
                    if let Some(cache) = self.retired_cache.remove(&SessionLease {
                        session_id: item.session_id,
                        generation: item.assignment.generation,
                    }) {
                        self.cache.free(cache);
                    }
                    let handled_at = Instant::now();
                    let timings = PacketTimings {
                        ingress: item.routed_at - item.input.timestamp,
                        worker_mailbox: item.received_at - item.routed_at,
                        validation: item.queued_at - item.received_at,
                        scheduler_queue: result.submitted_at - item.queued_at,
                        device_queue: result.started_at - result.submitted_at,
                        device_execution: inference_latency,
                        host_completion_delay: host_wait,
                        result_delivery: handled_at - result.observed_at,
                        gateway_return: Duration::ZERO,
                    };
                    self.measurements.profile.record(SlowWorkerPacket {
                        session_id: item.session_id,
                        sequence: item.input.packet.sequence,
                        elapsed_secs: handled_at.duration_since(self.epoch).as_secs_f64(),
                        deadline_exceeded: handled_at > item.input.deadline,
                        timings,
                    });
                    let Some(session) = self
                        .sessions
                        .get_mut(&item.session_id)
                        .filter(|session| session.assignment == item.assignment)
                    else {
                        self.measurements.counters.stale_results += 1;
                        self.measurements.counters.rejected_frames += 1;
                        let _ = item
                            .reply
                            .send(InputOutcome::Rejected(FrameRejection::Cancelled));
                        continue;
                    };
                    session.in_flight = false;
                    self.measurements
                        .queue_delay
                        .record(result.started_at.duration_since(item.input.timestamp));
                    self.measurements
                        .end_to_end_latency
                        .record(Instant::now().duration_since(item.input.timestamp));
                    if Instant::now() > item.input.deadline {
                        self.measurements.counters.deadline_misses += 1;
                        self.measurements
                            .deadline_lateness
                            .record(Instant::now().duration_since(item.input.deadline));
                        self.reject(
                            item.session_id,
                            item.reply,
                            FrameRejection::DeadlineExceeded,
                        );
                        continue;
                    }
                    session.prefix = item.prefix;
                    self.cache.update(session.cache, item.prefix);
                    match item.cache {
                        CacheOutcome::Hit => self.measurements.counters.cache_hits += 1,
                        CacheOutcome::Replayed { packets, bytes } => {
                            self.measurements.counters.replayed_packets += packets;
                            self.measurements.counters.replayed_bytes += bytes;
                        }
                    }
                    let output = InferenceOutput {
                        session_id: item.session_id,
                        input_timestamp: item.input.timestamp,
                        completed_at: result.observed_at,
                        audio: AudioResult {
                            assignment: item.assignment,
                            sequence: item.input.packet.sequence,
                            payload: item.input.packet.payload,
                            prefix: item.prefix,
                            cache: item.cache,
                            timings: Box::new(timings),
                        },
                    };
                    if item.reply.send(InputOutcome::Processed(output)).is_ok() {
                        self.measurements.counters.delivered_frames += 1;
                    } else {
                        self.terminate(item.session_id, FrameRejection::Cancelled);
                    }
                }
                self.refresh_capacity();
            }
        }
    }
}
