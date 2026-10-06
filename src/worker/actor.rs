use super::{
    cache::{CacheHandle, CachePool},
    mailbox::{WorkerCommand, WorkerHandle},
    mock_gpu::{self, DeviceJob, DeviceResult, DeviceWork, SessionCancellation},
    session::{SessionWork, WorkerSession},
};
use crate::{
    config::RuntimeConfig,
    metrics::WorkerMeasurements,
    protocol::{FrameRejection, InputOutcome, PrefixState, SessionId, SessionLease, WorkerId},
    scheduler::{
        admission::{ServiceEstimator, WorkerStatus},
        deadline::ReadySession,
    },
};
use std::{
    collections::{HashMap, HashSet, VecDeque},
    sync::Arc,
    time::Duration,
};
use tokio::{
    sync::{mpsc, oneshot, watch},
    time::Instant,
};
use tokio_util::sync::CancellationToken;

mod admission;
mod input;
mod scheduling;

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
        let device = mock_gpu::spawn(job_mailbox, device_results, wait);
        self.calibrate(&jobs, &mut results).await;
        self.epoch = Instant::now();
        self.refresh_capacity();
        self.measurements.initial_session_limit = self.admission_limit();
        let _ = ready.send(());
        let mut next_probe = self.epoch + self.configuration.probe_interval;
        let mut slowdown_time = self
            .configuration
            .slowdown
            .as_ref()
            .map(|slowdown| self.epoch + slowdown.after);
        loop {
            let wakeup = self.next_wakeup();
            let slowdown_wakeup = slowdown_time.unwrap_or(next_probe);
            let can_dispatch = !self.prepared.is_empty()
                && self.submitted.len() < self.configuration.device_queue_capacity + 1
                && (self.prepared.len() == self.configuration.batch_size
                    || !self.submitted.is_empty()
                    || commands.is_empty()
                    || self.idle_collection_expired())
                && wakeup <= Instant::now();
            tokio::select! {
                biased;
                _ = cancellation.cancelled() => break,
                _ = tokio::time::sleep_until(slowdown_wakeup), if slowdown_time.is_some() => {
                    slowdown_time = None;
                    self.refresh_capacity();
                }
                result = results.recv(), if !self.submitted.is_empty() => {
                    self.complete(result.expect("device returns active work"));
                }
                permit = jobs.reserve(), if can_dispatch => {
                    self.dispatch_batch(permit.expect("device is running"));
                }
                command = commands.recv() => {
                    let Some(command) = command else {
                        break;
                    };
                    if matches!(&command, WorkerCommand::InputReady { .. })
                        && !self.configuration.worker_input_delay.is_zero()
                    {
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
                    jobs.send(DeviceJob {
                        work: DeviceWork::Probe,
                        latency,
                        submitted_at,
                    })
                    .await
                    .expect("device is running");
                    self.track_submission(submitted_at, latency);
                    next_probe = Instant::now() + self.configuration.probe_interval;
                }
            }
        }
        self.drain(&mut commands, &mut results).await;
        drop(jobs);
        self.measurements.device_cpu = device.await.expect("mock device does not panic");
        self.measurements.elapsed = Instant::now().duration_since(self.epoch);
        self.measurements.final_session_limit = self.admission_limit();
        self.measurements.service_time = self.projected_service_time();
        self.measurements.host_delay = self.estimator.host_delay();
        self.measurements.host_reserve = self.estimator.host_reserve();
        self.measurements
    }
    async fn calibrate(
        &mut self,
        jobs: &mpsc::Sender<DeviceJob>,
        results: &mut mpsc::Receiver<DeviceResult>,
    ) {
        for _ in 0..self.configuration.calibration_samples {
            jobs.send(DeviceJob {
                work: DeviceWork::Probe,
                latency: self.configuration.inference_latency,
                submitted_at: Instant::now(),
            })
            .await
            .expect("device is running");
            let result = results.recv().await.expect("probe result");
            let received_at = Instant::now();
            let elapsed = received_at.duration_since(result.started_at);
            self.estimator
                .observe(result.completed_at - result.started_at);
            self.estimator
                .observe_host_delay(received_at - result.completed_at);
            self.measurements.calibration_latency.record(elapsed);
        }
    }

    async fn drain(
        &mut self,
        commands: &mut mpsc::Receiver<WorkerCommand>,
        results: &mut mpsc::Receiver<DeviceResult>,
    ) {
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
    }
    fn terminate(&mut self, session_id: SessionId, reason: FrameRejection) {
        if let Some(session) = self.sessions.remove(&session_id) {
            session.cancellation.cancel();
            let submitted = matches!(session.work, SessionWork::Submitted);
            if let SessionWork::Ready(pending) = session.work {
                self.measurements.counters.rejected_frames += 1;
                let _ = pending.reply.send(InputOutcome::Rejected(reason));
            }
            if submitted {
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
                        work: SessionWork::Idle,
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
                    Some(session)
                        if session.assignment.generation == generation
                            && matches!(session.work, SessionWork::Idle) =>
                    {
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
}
