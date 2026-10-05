use std::{cmp::Reverse, collections::BinaryHeap, net::SocketAddr, time::Duration};

use bytes::Bytes;
use rand::{Rng, SeedableRng, rngs::SmallRng};
use serde::{Deserialize, Serialize};
use tokio::{sync::mpsc, task::JoinSet, time::Instant};
use tokio_util::sync::CancellationToken;

use crate::{
    metrics::profile::{PacketTimings, RETAINED_TRACES, RuntimeLagReport, monitor_runtime},
    metrics::{LatencyDistribution, LatencyHistogram},
    protocol::{CacheOutcome, CreateRejection, FrameRejection, PacketSequence, SessionId},
    transport::{AudioSession, ClientError, ConnectOutcome, connect_session},
};

#[derive(Clone, Copy, Debug, Serialize, Deserialize, clap::ValueEnum)]
pub enum ArrivalPhase {
    Aligned,
    Random,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SimulationConfig {
    pub sessions: usize,
    pub duration: Duration,
    pub minimum_interval: Duration,
    pub maximum_interval: Duration,
    pub payload_bytes: usize,
    pub phase: ArrivalPhase,
    pub seed: u64,
    pub evict_every: Option<u64>,
    pub churn_after: Option<Duration>,
    pub io_timeout: Duration,
}
impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            sessions: 400,
            duration: Duration::from_secs(10),
            minimum_interval: Duration::from_millis(48),
            maximum_interval: Duration::from_millis(55),
            payload_bytes: 1600,
            phase: ArrivalPhase::Random,
            seed: 42,
            evict_every: None,
            churn_after: None,
            io_timeout: Duration::from_secs(5),
        }
    }
}
impl SimulationConfig {
    pub fn validate(&self) -> Result<(), SimulationError> {
        if self.sessions == 0
            || self.sessions > 10_000
            || self.duration.is_zero()
            || self.duration > Duration::from_secs(86400)
            || self.minimum_interval < Duration::from_millis(1)
            || self.maximum_interval < self.minimum_interval
            || self.maximum_interval > Duration::from_secs(1)
            || self.payload_bytes == 0
            || self.payload_bytes > 4096
            || self.io_timeout.is_zero()
            || self.io_timeout > Duration::from_secs(86400)
            || self.evict_every == Some(0)
            || self
                .churn_after
                .is_some_and(|duration| duration.is_zero() || duration > Duration::from_secs(86400))
        {
            return Err(SimulationError::Configuration(
                "invalid bounded workload configuration",
            ));
        }
        Ok(())
    }
}
#[derive(Debug, thiserror::Error)]
pub enum SimulationError {
    #[error("invalid simulation configuration: {0}")]
    Configuration(&'static str),
    #[error(transparent)]
    Task(#[from] tokio::task::JoinError),
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum FailureReason {
    DeadlineExceeded,
    ServerRejected(FrameRejection),
    Connection,
    PrefixCapacity,
    InvalidEcho,
    Protocol,
    GeneratorOverrun,
}
#[derive(Debug, Serialize, Deserialize)]
pub struct FailureCount {
    pub reason: FailureReason,
    pub count: u64,
}
#[derive(Default, Debug, Serialize, Deserialize)]
pub struct ClientCounters {
    pub admitted_sessions: u64,
    pub rejected_capacity: u64,
    pub rejected_duplicate: u64,
    pub closed_sessions: u64,
    pub failed_sessions: u64,
    pub attempted_frames: u64,
    pub echoed_frames: u64,
    pub cache_hits: u64,
    pub replayed_frames: u64,
    pub replayed_packets: u64,
    pub replayed_bytes: u64,
    pub failures: Vec<FailureCount>,
}
impl ClientCounters {
    fn failure(&mut self, reason: FailureReason) {
        self.failed_sessions += 1;
        if let Some(entry) = self
            .failures
            .iter_mut()
            .find(|entry| entry.reason == reason)
        {
            entry.count += 1;
        } else {
            self.failures.push(FailureCount { reason, count: 1 });
        }
    }
    fn merge(&mut self, other: Self) {
        self.admitted_sessions += other.admitted_sessions;
        self.rejected_capacity += other.rejected_capacity;
        self.rejected_duplicate += other.rejected_duplicate;
        self.closed_sessions += other.closed_sessions;
        self.failed_sessions += other.failed_sessions;
        self.attempted_frames += other.attempted_frames;
        self.echoed_frames += other.echoed_frames;
        self.cache_hits += other.cache_hits;
        self.replayed_frames += other.replayed_frames;
        self.replayed_packets += other.replayed_packets;
        self.replayed_bytes += other.replayed_bytes;
        for failure in other.failures {
            if let Some(entry) = self
                .failures
                .iter_mut()
                .find(|entry| entry.reason == failure.reason)
            {
                entry.count += failure.count;
            } else {
                self.failures.push(failure);
            }
        }
    }
    fn admission(&mut self, reason: CreateRejection) {
        match reason {
            CreateRejection::Capacity => self.rejected_capacity += 1,
            CreateRejection::AlreadyExists => self.rejected_duplicate += 1,
        }
    }
}
#[derive(Debug, Serialize, Deserialize)]
pub struct SimulationReport {
    pub configuration: SimulationConfig,
    pub elapsed_secs: f64,
    pub admission_secs: f64,
    pub counters: ClientCounters,
    pub throughput_frames_per_sec: f64,
    pub round_trip_latency: LatencyDistribution,
    pub failed_round_trip_latency: LatencyDistribution,
    pub generator_delay: LatencyDistribution,
    pub generated_intervals: LatencyDistribution,
    pub generator_overruns: u64,
    pub client_start_delay: LatencyDistribution,
    pub outside_server_latency: LatencyDistribution,
    pub runtime_lag: RuntimeLagReport,
    pub slowest_packets: Vec<ClientPacketTrace>,
}
#[derive(Debug, Serialize, Deserialize)]
pub enum ClientPacketTrace {
    Echo {
        session_id: SessionId,
        sequence: PacketSequence,
        round_trip: Duration,
        client_start_delay: Duration,
        outside_server: Duration,
        server: PacketTimings,
    },
    Failure {
        session_id: SessionId,
        sequence: PacketSequence,
        round_trip: Duration,
        client_start_delay: Duration,
        reason: FailureReason,
    },
}
impl ClientPacketTrace {
    fn round_trip(&self) -> Duration {
        match self {
            Self::Echo { round_trip, .. } | Self::Failure { round_trip, .. } => *round_trip,
        }
    }
}
#[derive(Default)]
struct ClientMeasurements {
    counters: ClientCounters,
    round_trip: LatencyHistogram,
    failed_round_trip: LatencyHistogram,
    client_start_delay: LatencyHistogram,
    outside_server: LatencyHistogram,
    slowest_packets: Vec<ClientPacketTrace>,
}
impl ClientMeasurements {
    fn trace(&mut self, packet: ClientPacketTrace) {
        self.slowest_packets.push(packet);
        self.slowest_packets
            .sort_unstable_by_key(|packet| Reverse(packet.round_trip()));
        self.slowest_packets.truncate(RETAINED_TRACES);
    }
}
struct PacerSlot {
    sender: mpsc::Sender<Instant>,
    cancellation: CancellationToken,
}
#[derive(Default)]
struct PacerMeasurements {
    delay: LatencyHistogram,
    intervals: LatencyHistogram,
    overruns: u64,
}

fn pace(
    slots: Vec<PacerSlot>,
    configuration: SimulationConfig,
    cancellation: CancellationToken,
) -> PacerMeasurements {
    let mut random = SmallRng::seed_from_u64(configuration.seed);
    let mut events = BinaryHeap::new();
    let start = std::time::Instant::now();
    let stop = start + configuration.duration;
    for index in 0..slots.len() {
        let offset = match configuration.phase {
            ArrivalPhase::Aligned => Duration::ZERO,
            ArrivalPhase::Random => Duration::from_micros(
                random.random_range(0..configuration.maximum_interval.as_micros() as u64),
            ),
        };
        events.push(Reverse((start + offset, index)));
    }
    let mut measurements = PacerMeasurements::default();
    let mut last_captures = vec![None; slots.len()];
    while let Some(Reverse((scheduled, index))) = events.pop() {
        if scheduled >= stop || cancellation.is_cancelled() {
            break;
        }
        std::thread::sleep(scheduled.saturating_duration_since(std::time::Instant::now()));
        let captured = std::time::Instant::now();
        if captured >= stop {
            break;
        }
        let slot = &slots[index];
        if slot.cancellation.is_cancelled() {
            continue;
        }
        measurements
            .delay
            .record(captured.saturating_duration_since(scheduled));
        match slot.sender.try_send(Instant::from_std(captured)) {
            Ok(()) => {
                if let Some(previous) = last_captures[index] {
                    measurements
                        .intervals
                        .record(captured.duration_since(previous));
                }
                last_captures[index] = Some(captured);
            }
            Err(mpsc::error::TrySendError::Closed(_)) => continue,
            Err(mpsc::error::TrySendError::Full(_)) => {
                measurements.overruns += 1;
                slot.cancellation.cancel();
                continue;
            }
        }
        let interval = Duration::from_micros(random.random_range(
            configuration.minimum_interval.as_micros() as u64
                ..=configuration.maximum_interval.as_micros() as u64,
        ));
        events.push(Reverse((captured + interval, index)));
    }
    measurements
}

fn classify(error: ClientError) -> FailureReason {
    match error {
        ClientError::DeadlineExceeded => FailureReason::DeadlineExceeded,
        ClientError::Rejected(reason) => FailureReason::ServerRejected(reason),
        ClientError::Wire(_) => FailureReason::Connection,
        ClientError::Protocol(_) => FailureReason::Protocol,
        ClientError::PrefixCapacity => FailureReason::PrefixCapacity,
        ClientError::EchoMismatch => FailureReason::InvalidEcho,
    }
}
async fn run_session(
    mut session: Box<AudioSession>,
    session_id: SessionId,
    address: SocketAddr,
    configuration: SimulationConfig,
    mut ticks: mpsc::Receiver<Instant>,
    cancellation: CancellationToken,
) -> ClientMeasurements {
    let mut measurements = ClientMeasurements::default();
    measurements.counters.admitted_sessions = 1;
    let payload = Bytes::from(vec![42; configuration.payload_bytes]);
    let mut random = SmallRng::seed_from_u64(configuration.seed.wrapping_add(session_id.0));
    let mut created = Instant::now();
    let mut lifetime = configuration
        .churn_after
        .map(|duration| duration.mul_f64(random.random_range(0.5..=1.0)));
    loop {
        let captured_at = tokio::select! {
            _ = cancellation.cancelled() => {
                measurements.counters.failure(FailureReason::GeneratorOverrun);
                return measurements;
            }
            tick = ticks.recv() => match tick { Some(tick) => tick, None => break },
        };
        measurements.counters.attempted_frames += 1;
        let sequence = session.next_sequence();
        let client_start_delay = captured_at.elapsed();
        measurements.client_start_delay.record(client_start_delay);
        match session.infer_audio(payload.clone(), captured_at).await {
            Ok(audio) => {
                let round_trip = captured_at.elapsed();
                let outside_server =
                    round_trip.saturating_sub(client_start_delay + audio.timings.total());
                measurements.round_trip.record(round_trip);
                measurements.outside_server.record(outside_server);
                measurements.trace(ClientPacketTrace::Echo {
                    session_id,
                    sequence: audio.sequence,
                    round_trip,
                    client_start_delay,
                    outside_server,
                    server: *audio.timings,
                });
                measurements.counters.echoed_frames += 1;
                match audio.cache {
                    CacheOutcome::Hit => measurements.counters.cache_hits += 1,
                    CacheOutcome::Replayed { packets, bytes } => {
                        measurements.counters.replayed_frames += 1;
                        measurements.counters.replayed_packets += packets;
                        measurements.counters.replayed_bytes += bytes;
                    }
                }
                if configuration
                    .evict_every
                    .is_some_and(|period| (audio.sequence.0 + 1) % period == 0)
                {
                    match tokio::time::timeout(configuration.io_timeout, session.evict_cache())
                        .await
                    {
                        Ok(Ok(_)) => {}
                        Ok(Err(error)) => {
                            measurements.counters.failure(classify(error));
                            return measurements;
                        }
                        Err(_) => {
                            measurements.counters.failure(FailureReason::Connection);
                            return measurements;
                        }
                    }
                }
            }
            Err(error) => {
                let round_trip = captured_at.elapsed();
                let reason = classify(error);
                measurements.failed_round_trip.record(round_trip);
                measurements.trace(ClientPacketTrace::Failure {
                    session_id,
                    sequence,
                    round_trip,
                    client_start_delay,
                    reason,
                });
                measurements.counters.failure(reason);
                return measurements;
            }
        }
        if lifetime.is_some_and(|duration| created.elapsed() >= duration) {
            match tokio::time::timeout(configuration.io_timeout, session.close()).await {
                Ok(Ok(_)) => measurements.counters.closed_sessions += 1,
                _ => {
                    measurements.counters.failure(FailureReason::Connection);
                    return measurements;
                }
            }
            match connect_session(address, session_id, configuration.io_timeout).await {
                Ok(ConnectOutcome::Admitted(new_session)) => {
                    session = new_session;
                    measurements.counters.admitted_sessions += 1;
                }
                Ok(ConnectOutcome::Rejected(reason)) => {
                    measurements.counters.admission(reason);
                    return measurements;
                }
                Err(error) => {
                    measurements.counters.failure(classify(error));
                    return measurements;
                }
            }
            created = Instant::now();
            lifetime = configuration
                .churn_after
                .map(|duration| duration.mul_f64(random.random_range(0.5..=1.0)));
        }
    }
    match tokio::time::timeout(configuration.io_timeout, session.close()).await {
        Ok(Ok(_)) => measurements.counters.closed_sessions += 1,
        _ => measurements.counters.failure(FailureReason::Connection),
    }
    measurements
}

pub async fn run(
    address: SocketAddr,
    configuration: SimulationConfig,
) -> Result<SimulationReport, SimulationError> {
    configuration.validate()?;
    let admission_start = Instant::now();
    let mut connections = JoinSet::new();
    for index in 0..configuration.sessions {
        let timeout = configuration.io_timeout;
        connections.spawn(async move {
            (
                SessionId(index as u64),
                connect_session(address, SessionId(index as u64), timeout).await,
            )
        });
    }
    let mut admitted = Vec::new();
    let mut totals = ClientMeasurements::default();
    while let Some(result) = connections.join_next().await {
        let (session_id, outcome) = result?;
        match outcome {
            Ok(ConnectOutcome::Admitted(session)) => admitted.push((session_id, session)),
            Ok(ConnectOutcome::Rejected(reason)) => totals.counters.admission(reason),
            Err(error) => totals.counters.failure(classify(error)),
        }
    }
    let admission_secs = admission_start.elapsed().as_secs_f64();
    let started = Instant::now();
    let monitor_cancellation = CancellationToken::new();
    let monitor = monitor_runtime(monitor_cancellation.clone());
    let mut tasks = JoinSet::new();
    let mut slots = Vec::new();
    for (session_id, session) in admitted {
        let (sender, receiver) = mpsc::channel(1);
        let cancellation = CancellationToken::new();
        slots.push(PacerSlot {
            sender,
            cancellation: cancellation.clone(),
        });
        tasks.spawn(run_session(
            session,
            session_id,
            address,
            configuration.clone(),
            receiver,
            cancellation,
        ));
    }
    let pacing_configuration = configuration.clone();
    let pacer = tokio::task::spawn_blocking(move || {
        pace(slots, pacing_configuration, CancellationToken::new())
    });
    while let Some(result) = tasks.join_next().await {
        let measurements = result?;
        totals.round_trip.merge(&measurements.round_trip);
        totals
            .failed_round_trip
            .merge(&measurements.failed_round_trip);
        totals.counters.merge(measurements.counters);
        totals
            .client_start_delay
            .merge(&measurements.client_start_delay);
        totals.outside_server.merge(&measurements.outside_server);
        for packet in measurements.slowest_packets {
            totals.trace(packet);
        }
    }
    let pacing = pacer.await?;
    let elapsed_secs = started.elapsed().as_secs_f64();
    monitor_cancellation.cancel();
    let runtime_lag = monitor.await?;
    Ok(SimulationReport {
        throughput_frames_per_sec: totals.counters.echoed_frames as f64
            / elapsed_secs.max(f64::EPSILON),
        configuration,
        elapsed_secs,
        admission_secs,
        counters: totals.counters,
        round_trip_latency: totals.round_trip.summary(),
        failed_round_trip_latency: totals.failed_round_trip.summary(),
        generator_delay: pacing.delay.summary(),
        generated_intervals: pacing.intervals.summary(),
        generator_overruns: pacing.overruns,
        client_start_delay: totals.client_start_delay.summary(),
        outside_server_latency: totals.outside_server.summary(),
        slowest_packets: totals.slowest_packets,
        runtime_lag,
    })
}
