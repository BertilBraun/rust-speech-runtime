use super::state::SessionState;
use crate::{
    config::RuntimeConfig,
    metrics::profile::monitor_runtime,
    metrics::{ManagerMeasurements, Report},
    protocol::{
        Assignment, CreateOutcome, CreateRejection, FrameRejection, Generation, InputFrame,
        InputOutcome, SessionAdmission, SessionId, SessionTarget, WorkerId,
    },
    runtime::{Command, IngressMeasurements, RuntimeError},
    scheduler::{
        admission::WorkerStatus,
        placement::{LeastLoaded, PlacementPolicy},
    },
    worker::{WorkerCommand, WorkerHandle, spawn_worker},
};
use std::{
    collections::HashMap,
    sync::{Arc, atomic::Ordering},
};
use tokio::{
    sync::{mpsc, oneshot},
    time::{Instant, MissedTickBehavior},
};
use tokio_util::sync::CancellationToken;

pub(crate) struct SessionManager {
    configuration: Arc<RuntimeConfig>,
    sessions: HashMap<SessionId, SessionState>,
    workers: Vec<WorkerHandle>,
    worker_cancellation: CancellationToken,
    measurements: ManagerMeasurements,
    ingress_measurements: Arc<IngressMeasurements>,
    next_generation: u64,
    epoch: Instant,
}
impl SessionManager {
    pub(crate) async fn new(
        configuration: Arc<RuntimeConfig>,
        ingress_measurements: Arc<IngressMeasurements>,
    ) -> Result<Self, RuntimeError> {
        let worker_cancellation = CancellationToken::new();
        let mut workers = Vec::new();
        let mut ready = Vec::new();
        for index in 0..configuration.workers {
            let (worker, response) = spawn_worker(
                WorkerId(index),
                configuration.clone(),
                worker_cancellation.clone(),
            );
            workers.push(worker);
            ready.push(response);
        }
        for response in ready {
            response.await.map_err(|_| RuntimeError::Stopped)?;
        }
        Ok(Self {
            configuration,
            sessions: HashMap::new(),
            workers,
            worker_cancellation,
            measurements: ManagerMeasurements::default(),
            ingress_measurements,
            next_generation: 0,
            epoch: Instant::now(),
        })
    }
    pub(crate) async fn run(
        mut self,
        mut commands: mpsc::Receiver<Command>,
        cancellation: CancellationToken,
    ) -> Result<Report, RuntimeError> {
        let monitor_cancellation = CancellationToken::new();
        let monitor = monitor_runtime(monitor_cancellation.clone());
        let mut maintenance = tokio::time::interval(
            self.configuration
                .minimum_packet_interval
                .min(self.configuration.session_timeout),
        );
        maintenance.set_missed_tick_behavior(MissedTickBehavior::Skip);
        let outcome: Result<(), RuntimeError> = async {
            loop {
                tokio::select! {
                    _ = cancellation.cancelled() => break,
                    command = commands.recv() => {
                        let Some(command) = command else { break; };
                        tokio::select! {
                            _ = cancellation.cancelled() => break,
                            result = self.handle(command) => result?,
                        }
                    }
                    _ = maintenance.tick() => {
                        self.reconcile();
                        tokio::select! {
                            _ = cancellation.cancelled() => break,
                            result = self.expire_sessions() => result?,
                        }
                    }
                }
            }
            Ok(())
        }
        .await;
        commands.close();
        self.measurements.active_sessions = self.sessions.len();
        self.worker_cancellation.cancel();
        let mut workers = Vec::new();
        for worker in self.workers.drain(..) {
            workers.push(worker.task.await?);
        }
        monitor_cancellation.cancel();
        let runtime_lag = monitor.await?;
        outcome?;
        Ok(Report::assemble(
            self.measurements,
            workers,
            self.ingress_measurements
                .channel_saturation
                .load(Ordering::Relaxed),
            self.ingress_measurements
                .inputs_overloaded
                .load(Ordering::Relaxed),
            Instant::now().duration_since(self.epoch),
            self.configuration.batch_size,
            runtime_lag,
        ))
    }
    fn statuses(&self) -> Vec<WorkerStatus> {
        self.workers
            .iter()
            .map(|worker| worker.status.borrow().clone())
            .collect()
    }
    fn reconcile(&mut self) {
        let workers = &self.workers;
        let previous = self.sessions.len();
        self.sessions.retain(|session_id, session| {
            workers[session.assignment.worker_id.0]
                .status
                .borrow()
                .sessions
                .contains(session_id)
        });
        self.measurements.terminated_sessions += (previous - self.sessions.len()) as u64;
        self.measurements.active_sessions = self.sessions.len();
    }
    async fn handle(&mut self, command: Command) -> Result<(), RuntimeError> {
        let control = !matches!(command, Command::InputFrame { .. });
        let started = Instant::now();
        match command {
            Command::CreateSession { session_id, reply } => {
                let _ = reply.send(self.create(session_id).await?);
            }
            Command::InputFrame {
                target,
                input,
                reply,
            } => self.input(target, input, reply).await?,
            Command::CloseSession { target, reply } => {
                let _ = reply.send(self.close(target).await?);
            }
            Command::EvictCache { target, reply } => {
                let session_id = target.session_id();
                if let Some(session) = self
                    .sessions
                    .get(&session_id)
                    .filter(|session| target.matches_generation(session.assignment.generation))
                {
                    self.send_control(
                        session.assignment.worker_id,
                        WorkerCommand::EvictCache {
                            session_id,
                            generation: session.assignment.generation,
                            reply,
                        },
                    )
                    .await?;
                } else {
                    let _ = reply.send(false);
                }
            }
        }
        if control {
            self.measurements.control_duration.record(started.elapsed());
        }
        Ok(())
    }
    async fn create(&mut self, session_id: SessionId) -> Result<CreateOutcome, RuntimeError> {
        self.reconcile();
        if self.sessions.contains_key(&session_id) {
            return Ok(CreateOutcome::Rejected(CreateRejection::AlreadyExists));
        }
        let mut workers = self.statuses();
        while let Some(worker_id) = LeastLoaded.select_worker(&workers) {
            self.next_generation = self
                .next_generation
                .checked_add(1)
                .expect("generation space exhausted");
            let assignment = Assignment {
                worker_id,
                generation: Generation(self.next_generation),
            };
            let (reply, response) = oneshot::channel();
            self.send_control(
                worker_id,
                WorkerCommand::AddSession {
                    session_id,
                    assignment,
                    reply,
                },
            )
            .await?;
            if response.await.map_err(|_| RuntimeError::Stopped)? {
                self.sessions.insert(
                    session_id,
                    SessionState {
                        assignment,
                        last_input_at: Instant::now(),
                    },
                );
                self.measurements.admitted_sessions += 1;
                self.measurements.active_sessions = self.sessions.len();
                self.measurements.peak_active_sessions = self
                    .measurements
                    .peak_active_sessions
                    .max(self.sessions.len());
                return Ok(CreateOutcome::Admitted(SessionAdmission {
                    assignment,
                    audio_limits: self.configuration.audio_limits,
                    packet_deadline: self.configuration.packet_deadline,
                }));
            }
            workers[worker_id.0].session_limit = 0;
        }
        self.measurements.rejected_sessions += 1;
        Ok(CreateOutcome::Rejected(CreateRejection::Capacity))
    }
    async fn input(
        &mut self,
        target: SessionTarget,
        input: InputFrame,
        reply: oneshot::Sender<InputOutcome>,
    ) -> Result<(), RuntimeError> {
        let session_id = target.session_id();
        let Some(session) = self.sessions.get_mut(&session_id) else {
            self.measurements.rejected_frames += 1;
            let _ = reply.send(InputOutcome::Rejected(FrameRejection::UnknownSession));
            return Ok(());
        };
        if !target.matches_generation(session.assignment.generation) {
            self.measurements.rejected_frames += 1;
            let _ = reply.send(InputOutcome::Rejected(FrameRejection::Cancelled));
            return Ok(());
        }
        session.last_input_at = Instant::now();
        self.measurements
            .ingress_delay
            .record(input.timestamp.elapsed());
        let command = WorkerCommand::InputReady {
            session_id,
            generation: session.assignment.generation,
            input,
            reply,
            routed_at: Instant::now(),
        };
        match self.workers[session.assignment.worker_id.0]
            .commands
            .try_send(command)
        {
            Ok(()) => Ok(()),
            Err(mpsc::error::TrySendError::Full(WorkerCommand::InputReady { reply, .. })) => {
                self.measurements.worker_channel_saturation += 1;
                self.measurements.rejected_frames += 1;
                let _ = reply.send(InputOutcome::Rejected(FrameRejection::Overloaded));
                self.close(target).await?;
                Ok(())
            }
            Err(mpsc::error::TrySendError::Closed(_)) => Err(RuntimeError::Stopped),
            Err(mpsc::error::TrySendError::Full(_)) => unreachable!("only input was sent"),
        }
    }
    async fn close(&mut self, target: SessionTarget) -> Result<bool, RuntimeError> {
        self.reconcile();
        let session_id = target.session_id();
        if !self
            .sessions
            .get(&session_id)
            .is_some_and(|session| target.matches_generation(session.assignment.generation))
        {
            return Ok(false);
        }
        let session = self
            .sessions
            .remove(&session_id)
            .expect("validated session target exists");
        let (reply, response) = oneshot::channel();
        self.send_control(
            session.assignment.worker_id,
            WorkerCommand::RemoveSession {
                session_id,
                generation: session.assignment.generation,
                reply,
            },
        )
        .await?;
        response.await.map_err(|_| RuntimeError::Stopped)?;
        self.measurements.closed_sessions += 1;
        self.measurements.active_sessions = self.sessions.len();
        Ok(true)
    }
    async fn expire_sessions(&mut self) -> Result<(), RuntimeError> {
        let now = Instant::now();
        let expired: Vec<SessionId> = self
            .sessions
            .iter()
            .filter(|(_, session)| {
                now.duration_since(session.last_input_at) >= self.configuration.session_timeout
            })
            .map(|(session_id, _)| *session_id)
            .collect();
        for session_id in expired {
            self.close(session_id.into()).await?;
            self.measurements.timed_out_sessions += 1;
        }
        Ok(())
    }
    async fn send_control(
        &mut self,
        worker_id: WorkerId,
        command: WorkerCommand,
    ) -> Result<(), RuntimeError> {
        match self.workers[worker_id.0].commands.try_send(command) {
            Ok(()) => Ok(()),
            Err(mpsc::error::TrySendError::Full(command)) => {
                self.measurements.worker_channel_saturation += 1;
                self.workers[worker_id.0]
                    .commands
                    .send(command)
                    .await
                    .map_err(|_| RuntimeError::Stopped)
            }
            Err(mpsc::error::TrySendError::Closed(_)) => Err(RuntimeError::Stopped),
        }
    }
}
