use std::{cmp::Reverse, collections::BinaryHeap, time::Duration};

use bytes::Bytes;
use rand::{Rng, SeedableRng, rngs::SmallRng};
use serde::Serialize;
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;

use crate::{
    Node, RuntimeError,
    config::RuntimeConfig,
    metrics::{LatencyDistribution, LatencyHistogram, Report},
    protocol::{CreateOutcome, InputFrame, InputOutcome, SessionId},
};

#[derive(Clone, Copy, Debug, Default, clap::ValueEnum, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ArrivalPhase {
    #[default]
    Aligned,
    Random,
}

#[derive(Clone, Debug, Serialize)]
pub struct ChurnConfig {
    pub min_duration: Duration,
    pub max_duration: Duration,
}

#[derive(Clone, Debug, Serialize)]
pub struct SimulationConfig {
    pub sessions: usize,
    pub duration: Duration,
    pub jitter: Duration,
    pub arrival_phase: ArrivalPhase,
    pub churn: Option<ChurnConfig>,
    pub seed: u64,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            sessions: 400,
            duration: Duration::from_secs(60),
            jitter: Duration::ZERO,
            arrival_phase: ArrivalPhase::Aligned,
            churn: None,
            seed: 42,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum SimulationError {
    #[error("invalid workload: {0}")]
    InvalidConfiguration(&'static str),
    #[error(transparent)]
    Runtime(#[from] RuntimeError),
    #[error(transparent)]
    Task(#[from] tokio::task::JoinError),
}

impl SimulationConfig {
    pub fn validate(&self, tick_interval: Duration) -> Result<(), SimulationError> {
        if self.sessions == 0 {
            return Err(SimulationError::InvalidConfiguration(
                "sessions must be positive",
            ));
        }
        if self.duration.is_zero() || self.duration > Duration::from_secs(86400) {
            return Err(SimulationError::InvalidConfiguration(
                "duration must be in (0, 24 hours]",
            ));
        }
        if self.jitter >= tick_interval / 2 {
            return Err(SimulationError::InvalidConfiguration(
                "jitter must be less than half the tick interval",
            ));
        }
        if let Some(churn) = &self.churn
            && (churn.min_duration < Duration::from_micros(1)
                || churn.max_duration < churn.min_duration
                || churn.max_duration > Duration::from_secs(86400))
        {
            return Err(SimulationError::InvalidConfiguration(
                "churn requires 0 < min_duration <= max_duration <= 24 hours",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Serialize)]
pub struct WorkloadReport {
    pub create_attempts: u64,
    pub admitted: u64,
    pub rejected: u64,
    pub closed: u64,
    pub offered_frames: u64,
    pub accepted_frames: u64,
    pub overloaded_frames: u64,
    pub stale_frames: u64,
    pub unknown_session_frames: u64,
    pub skipped_generator_frames: u64,
    pub received_outputs: u64,
    pub peak_scheduled_events: usize,
    pub generator_lag: LatencyDistribution,
    pub interrupted: bool,
}

#[derive(Debug, Serialize)]
pub struct BenchmarkResult {
    pub runtime_configuration: RuntimeConfig,
    pub simulation_configuration: SimulationConfig,
    pub workload: WorkloadReport,
    pub runtime: Report,
}

#[derive(Clone, Copy, Eq, Ord, PartialEq, PartialOrd)]
enum EventKind {
    Create,
    Close,
    Input { tick_at: Instant },
}

#[derive(Clone, Copy, Eq, Ord, PartialEq, PartialOrd)]
struct ScheduledEvent {
    at: Instant,
    kind: EventKind,
    slot: usize,
}

struct Workload {
    configuration: SimulationConfig,
    tick_interval: Duration,
    pending: BinaryHeap<Reverse<ScheduledEvent>>,
    random: SmallRng,
    end: Instant,
    report: WorkloadReport,
    generator_lag: LatencyHistogram,
}

pub async fn run_benchmark(
    runtime_configuration: RuntimeConfig,
    simulation_configuration: SimulationConfig,
    cancellation: CancellationToken,
) -> Result<BenchmarkResult, SimulationError> {
    simulation_configuration.validate(runtime_configuration.tick_interval)?;
    let mut node = Node::start(runtime_configuration.clone())?;
    let mut outputs = node
        .take_outputs()
        .expect("output receiver has not been taken");
    let output_consumer = tokio::spawn(async move {
        let mut count: u64 = 0;
        while outputs.recv().await.is_some() {
            count += 1;
        }
        count
    });
    let workload = Workload::new(
        simulation_configuration.clone(),
        runtime_configuration.tick_interval,
    );
    let outcome = workload.run(&node, cancellation).await;
    let runtime = node.shutdown().await?;
    let received_outputs = output_consumer.await?;
    let mut workload = outcome?;
    workload.received_outputs = received_outputs;
    Ok(BenchmarkResult {
        runtime_configuration,
        simulation_configuration,
        workload,
        runtime,
    })
}

impl Workload {
    fn new(configuration: SimulationConfig, tick_interval: Duration) -> Self {
        let epoch = Instant::now();
        let mut random = SmallRng::seed_from_u64(configuration.seed);
        let pending = (0..configuration.sessions)
            .map(|slot| {
                let phase = match configuration.arrival_phase {
                    ArrivalPhase::Aligned => Duration::ZERO,
                    ArrivalPhase::Random => Duration::from_nanos(
                        random.random_range(0..tick_interval.as_nanos() as u64),
                    ),
                };
                Reverse(ScheduledEvent {
                    at: epoch + phase,
                    slot,
                    kind: EventKind::Create,
                })
            })
            .collect();
        Self {
            end: epoch + configuration.duration,
            report: WorkloadReport {
                create_attempts: 0,
                admitted: 0,
                rejected: 0,
                closed: 0,
                offered_frames: 0,
                accepted_frames: 0,
                overloaded_frames: 0,
                stale_frames: 0,
                unknown_session_frames: 0,
                skipped_generator_frames: 0,
                received_outputs: 0,
                peak_scheduled_events: configuration.sessions,
                generator_lag: LatencyHistogram::default().summary(),
                interrupted: false,
            },
            configuration,
            tick_interval,
            pending,
            random,
            generator_lag: LatencyHistogram::default(),
        }
    }

    async fn run(
        mut self,
        node: &Node,
        cancellation: CancellationToken,
    ) -> Result<WorkloadReport, SimulationError> {
        while let Some(Reverse(event)) = self.pending.pop() {
            if event.at >= self.end {
                break;
            }
            tokio::select! {
                _ = cancellation.cancelled() => { self.report.interrupted = true; break; }
                _ = tokio::time::sleep_until(self.end) => break,
                _ = tokio::time::sleep_until(event.at) => {}
            }
            tokio::select! {
                _ = cancellation.cancelled() => { self.report.interrupted = true; break; }
                _ = tokio::time::sleep_until(self.end) => break,
                outcome = self.handle(node, event) => outcome?,
            }
        }
        if !self.report.interrupted && Instant::now() < self.end {
            tokio::select! {
                _ = cancellation.cancelled() => self.report.interrupted = true,
                _ = tokio::time::sleep_until(self.end) => {}
            }
        }
        self.report.generator_lag = self.generator_lag.summary();
        Ok(self.report)
    }

    async fn handle(&mut self, node: &Node, event: ScheduledEvent) -> Result<(), SimulationError> {
        let session_id = SessionId(event.slot as u64);
        match event.kind {
            EventKind::Create => {
                self.report.create_attempts += 1;
                match node.ingress.create_session(session_id).await? {
                    CreateOutcome::Admitted(_) => {
                        self.report.admitted += 1;
                        self.schedule(ScheduledEvent {
                            kind: EventKind::Input { tick_at: event.at },
                            ..event
                        });
                        if self.configuration.churn.is_some() {
                            let duration = self.session_duration();
                            self.schedule(ScheduledEvent {
                                at: Instant::now() + duration,
                                kind: EventKind::Close,
                                ..event
                            });
                        }
                    }
                    CreateOutcome::RejectedCapacity => {
                        self.report.rejected += 1;
                        if self.configuration.churn.is_some() {
                            let duration = self.session_duration();
                            self.schedule(ScheduledEvent {
                                at: Instant::now() + duration,
                                ..event
                            });
                        }
                    }
                    CreateOutcome::AlreadyExists => {
                        unreachable!("simulator creates each incarnation once")
                    }
                }
            }
            EventKind::Input { tick_at } => {
                self.generator_lag
                    .record(Instant::now().duration_since(event.at));
                self.report.offered_frames += 1;
                let input = InputFrame {
                    timestamp: event.at,
                    payload: Bytes::from_static(b"simulated-audio-frame"),
                };
                match node.ingress.input_frame(session_id, input).await? {
                    InputOutcome::Accepted => self.report.accepted_frames += 1,
                    InputOutcome::Overloaded => self.report.overloaded_frames += 1,
                    InputOutcome::Stale => self.report.stale_frames += 1,
                    InputOutcome::UnknownSession => self.report.unknown_session_frames += 1,
                }
                let mut next_tick = tick_at + self.tick_interval;
                let now = Instant::now();
                if next_tick + self.configuration.jitter < now {
                    let skipped = now.duration_since(next_tick).as_nanos()
                        / self.tick_interval.as_nanos()
                        + 1;
                    self.report.skipped_generator_frames += skipped as u64;
                    next_tick +=
                        Duration::from_nanos((skipped * self.tick_interval.as_nanos()) as u64);
                }
                let at = self.jittered(next_tick);
                self.schedule(ScheduledEvent {
                    at,
                    kind: EventKind::Input { tick_at: next_tick },
                    ..event
                });
            }
            EventKind::Close => {
                if node.ingress.close_session(session_id).await? {
                    self.report.closed += 1;
                }
                self.pending
                    .retain(|Reverse(pending)| pending.slot != event.slot);
                self.schedule(ScheduledEvent {
                    at: Instant::now(),
                    kind: EventKind::Create,
                    ..event
                });
            }
        }
        Ok(())
    }

    fn session_duration(&mut self) -> Duration {
        let churn = self.configuration.churn.as_ref().expect("churn configured");
        Duration::from_micros(self.random.random_range(
            churn.min_duration.as_micros() as u64..=churn.max_duration.as_micros() as u64,
        ))
    }

    fn jittered(&mut self, tick_at: Instant) -> Instant {
        let jitter = self.configuration.jitter.as_nanos() as i64;
        let offset = self.random.random_range(-jitter..=jitter);
        let duration = Duration::from_nanos(offset.unsigned_abs());
        if offset < 0 {
            tick_at - duration
        } else {
            tick_at + duration
        }
    }

    fn schedule(&mut self, event: ScheduledEvent) {
        self.pending.push(Reverse(event));
        self.report.peak_scheduled_events =
            self.report.peak_scheduled_events.max(self.pending.len());
    }
}
