use std::{
    collections::HashMap,
    sync::{Arc, atomic::Ordering},
};

use tokio::{
    sync::{mpsc, oneshot},
    task::JoinHandle,
    time::{Instant, MissedTickBehavior},
};
use tokio_util::sync::CancellationToken;

use crate::config::RuntimeConfig;
use crate::metrics::{ManagerMeasurements, Report, WorkerMeasurements};
use crate::protocol::{
    Assignment, CreateOutcome, Generation, InferenceOutput, InputFrame, InputOutcome, SessionId,
    WorkerId,
};
use crate::runtime::{Command, IngressMeasurements, RuntimeError};
use crate::scheduler::placement::PlacementPolicy;
use crate::worker::{WorkerCommand, spawn_worker};

use super::state::SessionState;

pub(crate) struct SessionManager {
    configuration: Arc<RuntimeConfig>,
    placement: Box<dyn PlacementPolicy>,
    sessions: HashMap<SessionId, SessionState>,
    loads: Vec<usize>,
    worker_commands: Vec<mpsc::Sender<WorkerCommand>>,
    worker_tasks: Vec<JoinHandle<WorkerMeasurements>>,
    worker_cancellation: CancellationToken,
    measurements: ManagerMeasurements,
    ingress_measurements: Arc<IngressMeasurements>,
    outputs: mpsc::Sender<InferenceOutput>,
    completions: mpsc::Receiver<InferenceOutput>,
    next_generation: u64,
    epoch: Instant,
}

impl SessionManager {
    pub(crate) fn new(
        configuration: Arc<RuntimeConfig>,
        placement: Box<dyn PlacementPolicy>,
        outputs: mpsc::Sender<InferenceOutput>,
        ingress_measurements: Arc<IngressMeasurements>,
    ) -> Self {
        let epoch = Instant::now();
        let worker_cancellation = CancellationToken::new();
        let (results, completions) = mpsc::channel(configuration.result_channel_capacity);
        let mut worker_commands = Vec::with_capacity(configuration.workers);
        let mut worker_tasks = Vec::with_capacity(configuration.workers);
        for index in 0..configuration.workers {
            let (commands, task) = spawn_worker(
                WorkerId(index),
                configuration.clone(),
                results.clone(),
                worker_cancellation.clone(),
                epoch,
            );
            worker_commands.push(commands);
            worker_tasks.push(task);
        }
        Self {
            loads: vec![0; configuration.workers],
            configuration,
            placement,
            sessions: HashMap::new(),
            worker_commands,
            worker_tasks,
            worker_cancellation,
            measurements: ManagerMeasurements::default(),
            ingress_measurements,
            outputs,
            completions,
            next_generation: 0,
            epoch,
        }
    }

    pub(crate) async fn run(
        mut self,
        mut commands: mpsc::Receiver<Command>,
        cancellation: CancellationToken,
    ) -> Result<Report, RuntimeError> {
        let mut maintenance = tokio::time::interval(
            self.configuration
                .tick_interval
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
                            outcome = self.handle(command) => outcome?,
                        }
                    }
                    Some(output) = self.completions.recv() => self.dispatch(output),
                    _ = maintenance.tick() => {
                        tokio::select! {
                            _ = cancellation.cancelled() => break,
                            outcome = self.expire_sessions() => outcome?,
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
        let mut workers = Vec::with_capacity(self.worker_tasks.len());
        for task in self.worker_tasks.drain(..) {
            workers.push(task.await?);
        }
        while let Ok(output) = self.completions.try_recv() {
            self.dispatch(output);
        }
        outcome?;
        self.measurements.inputs_overloaded += self
            .ingress_measurements
            .inputs_overloaded
            .load(Ordering::Relaxed);
        Ok(Report::assemble(
            self.measurements,
            workers,
            self.ingress_measurements
                .channel_saturation
                .load(Ordering::Relaxed),
            Instant::now().duration_since(self.epoch),
            self.configuration.batch_size,
        ))
    }

    async fn handle(&mut self, command: Command) -> Result<(), RuntimeError> {
        match command {
            Command::CreateSession { session_id, reply } => {
                let result = self.create(session_id).await?;
                let _ = reply.send(result);
            }
            Command::InputFrame {
                session_id,
                input,
                reply,
            } => {
                let result = self.input(session_id, input);
                let _ = reply.send(result);
            }
            Command::CloseSession { session_id, reply } => {
                let removed = self.close(session_id).await?;
                let _ = reply.send(removed);
            }
        }
        Ok(())
    }

    async fn create(&mut self, session_id: SessionId) -> Result<CreateOutcome, RuntimeError> {
        if self.sessions.contains_key(&session_id) {
            return Ok(CreateOutcome::AlreadyExists);
        }
        let Some(worker_id) = self
            .placement
            .select_worker(&self.loads, self.configuration.admission_limit())
        else {
            self.measurements.rejected_sessions += 1;
            return Ok(CreateOutcome::RejectedCapacity);
        };
        self.next_generation = self
            .next_generation
            .checked_add(1)
            .expect("generation space exhausted");
        let assignment = Assignment {
            worker_id,
            generation: Generation(self.next_generation),
        };
        let now = Instant::now();
        let first_tick = align_start(now, self.epoch, self.configuration.phase_bucket);
        let (reply, response) = oneshot::channel();
        self.send_control(
            worker_id,
            WorkerCommand::AddSession {
                session_id,
                assignment,
                deadline: first_tick + self.configuration.tick_interval,
                reply,
            },
        )
        .await?;
        if !response.await.map_err(|_| RuntimeError::Stopped)? {
            self.measurements.rejected_sessions += 1;
            return Ok(CreateOutcome::RejectedCapacity);
        }
        self.sessions.insert(
            session_id,
            SessionState {
                assignment,
                last_input_at: now,
            },
        );
        self.loads[worker_id.0] += 1;
        self.measurements.admitted_sessions += 1;
        self.measurements.active_sessions = self.sessions.len();
        self.measurements.peak_active_sessions = self
            .measurements
            .peak_active_sessions
            .max(self.sessions.len());
        self.measurements
            .phase_added_latency
            .record(first_tick.duration_since(now));
        Ok(CreateOutcome::Admitted(assignment))
    }

    fn input(&mut self, session_id: SessionId, input: InputFrame) -> InputOutcome {
        let Some(session) = self.sessions.get_mut(&session_id) else {
            self.measurements.unknown_session_inputs += 1;
            return InputOutcome::UnknownSession;
        };
        let now = Instant::now();
        if now.duration_since(input.timestamp) > self.configuration.max_input_age {
            self.measurements.inputs_stale += 1;
            return InputOutcome::Stale;
        }
        let command = WorkerCommand::InputReady {
            session_id,
            generation: session.assignment.generation,
            input,
        };
        match self.worker_commands[session.assignment.worker_id.0].try_send(command) {
            Ok(()) => {
                session.last_input_at = now;
                self.measurements.inputs_accepted += 1;
                InputOutcome::Accepted
            }
            Err(mpsc::error::TrySendError::Full(_)) => {
                self.measurements.worker_channel_saturation += 1;
                self.measurements.inputs_overloaded += 1;
                InputOutcome::Overloaded
            }
            Err(mpsc::error::TrySendError::Closed(_)) => InputOutcome::Overloaded,
        }
    }

    async fn close(&mut self, session_id: SessionId) -> Result<bool, RuntimeError> {
        let Some(session) = self.sessions.remove(&session_id) else {
            return Ok(false);
        };
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
        self.loads[session.assignment.worker_id.0] -= 1;
        self.measurements.active_sessions = self.sessions.len();
        self.measurements.closed_sessions += 1;
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
            self.close(session_id).await?;
            self.measurements.timed_out_sessions += 1;
        }
        Ok(())
    }

    fn dispatch(&mut self, output: InferenceOutput) {
        if !self
            .sessions
            .get(&output.session_id)
            .is_some_and(|session| session.assignment == output.assignment)
        {
            self.measurements.stale_results += 1;
            return;
        }
        match self.outputs.try_send(output) {
            Ok(()) => self.measurements.delivered_results += 1,
            Err(mpsc::error::TrySendError::Full(_)) => {
                self.measurements.output_channel_saturation += 1;
                self.measurements.dropped_results += 1;
            }
            Err(mpsc::error::TrySendError::Closed(_)) => self.measurements.dropped_results += 1,
        }
    }

    async fn send_control(
        &mut self,
        worker_id: WorkerId,
        command: WorkerCommand,
    ) -> Result<(), RuntimeError> {
        match self.worker_commands[worker_id.0].try_send(command) {
            Ok(()) => Ok(()),
            Err(mpsc::error::TrySendError::Full(command)) => {
                self.measurements.worker_channel_saturation += 1;
                self.worker_commands[worker_id.0]
                    .send(command)
                    .await
                    .map_err(|_| RuntimeError::Stopped)
            }
            Err(mpsc::error::TrySendError::Closed(_)) => Err(RuntimeError::Stopped),
        }
    }
}

fn align_start(now: Instant, epoch: Instant, bucket: Option<std::time::Duration>) -> Instant {
    let Some(bucket) = bucket else {
        return now;
    };
    let elapsed = now.duration_since(epoch).as_nanos();
    let remainder = elapsed % bucket.as_nanos();
    if remainder == 0 {
        now
    } else {
        now + std::time::Duration::from_nanos((bucket.as_nanos() - remainder) as u64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn phase_buckets_only_add_bounded_delay() {
        let epoch = Instant::now();
        let arrival = epoch + Duration::from_millis(23);
        assert_eq!(
            align_start(arrival, epoch, Some(Duration::from_millis(5))),
            epoch + Duration::from_millis(25)
        );
        assert_eq!(align_start(arrival, epoch, None), arrival);
        assert_eq!(
            align_start(epoch, epoch, Some(Duration::from_millis(5))),
            epoch
        );
    }
}
