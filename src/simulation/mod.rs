use crate::{
    metrics::profile::monitor_runtime,
    protocol::SessionId,
    transport::{ConnectOutcome, connect_session},
};
use std::net::SocketAddr;
use tokio::{sync::mpsc, task::JoinSet, time::Instant};
use tokio_util::sync::CancellationToken;

mod config;
mod measurements;
mod pacing;
mod quality;
mod report;
mod session;

pub use config::{ArrivalPhase, SimulationConfig};
use measurements::PacketMeasurements;
pub use measurements::{ClientEchoTrace, ClientPacketTrace};
use pacing::{PacerSlot, pace};
pub use quality::PacketQualityPolicy;
pub use report::{ClientCounters, FailureCount, FailureReason, SimulationReport};
use session::{SessionInput, SimulatedSession, classify};

const CAPTURE_QUEUE_CAPACITY: usize = 4;

#[derive(Debug, thiserror::Error)]
pub enum SimulationError {
    #[error("session task exited before the workload was ready: {0}")]
    Startup(#[from] tokio::sync::oneshot::error::RecvError),
    #[error("CPU measurement failed: {0}")]
    CpuClock(#[from] std::io::Error),
    #[error("invalid simulation configuration: {0}")]
    Configuration(&'static str),
    #[error(transparent)]
    Task(#[from] tokio::task::JoinError),
}
pub async fn run(
    address: SocketAddr,
    configuration: SimulationConfig,
) -> Result<SimulationReport, SimulationError> {
    configuration.validate()?;
    let cancellation = CancellationToken::new();
    let _workload_cleanup = cancellation.clone().drop_guard();
    let cpu = crate::metrics::cpu::ProcessCpuMeasurement::start()?;
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
    let mut counters = ClientCounters::default();
    while let Some(result) = connections.join_next().await {
        let (session_id, outcome) = result?;
        match outcome {
            Ok(ConnectOutcome::Admitted(session)) => admitted.push((session_id, session)),
            Ok(ConnectOutcome::Rejected(reason)) => counters.admission(reason),
            Err(error) => counters.failure(classify(error)),
        }
    }
    let admission_secs = admission_start.elapsed().as_secs_f64();
    let setup_start = Instant::now();
    let mut tasks = JoinSet::new();
    let mut slots = Vec::new();
    let mut readiness = Vec::new();
    let (metrics, packet_mailbox) = mpsc::channel(configuration.metric_channel_capacity);
    let collector = tokio::spawn(PacketMeasurements::default().collect(packet_mailbox));
    for (session_id, session) in admitted {
        let (sender, receiver) = mpsc::channel(CAPTURE_QUEUE_CAPACITY);
        let session_cancellation = cancellation.child_token();
        let (ready, initialized) = tokio::sync::oneshot::channel();
        readiness.push(initialized);
        slots.push(PacerSlot {
            sender,
            cancellation: session_cancellation.clone(),
        });
        let workload = SimulatedSession::new(
            session,
            session_id,
            address,
            configuration.clone(),
            metrics.clone(),
        );
        tasks.spawn(workload.run(SessionInput {
            ticks: receiver,
            cancellation: session_cancellation,
            ready,
        }));
    }
    drop(metrics);
    for initialized in readiness {
        initialized.await?;
    }
    let setup_secs = setup_start.elapsed().as_secs_f64();
    let started = Instant::now();
    let monitor_cancellation = cancellation.child_token();
    let monitor = monitor_runtime(monitor_cancellation.clone());
    let pacing_configuration = configuration.clone();
    let pacer =
        tokio::task::spawn_blocking(move || pace(slots, pacing_configuration, cancellation));
    while let Some(result) = tasks.join_next().await {
        counters.merge(result?);
    }
    let pacing = pacer.await?;
    let elapsed_secs = started.elapsed().as_secs_f64();
    monitor_cancellation.cancel();
    let runtime_lag = monitor.await?;
    let totals = collector.await?;
    Ok(SimulationReport {
        process_cpu: cpu.finish()?,
        throughput_frames_per_sec: counters.echoed_frames as f64 / elapsed_secs.max(f64::EPSILON),
        configuration,
        elapsed_secs,
        admission_secs,
        setup_secs,
        counters,
        round_trip_latency: totals.round_trip.summary(),
        failed_round_trip_latency: totals.failed_round_trip.summary(),
        deadline_lateness: totals.deadline_lateness.summary(),
        generator_delay: pacing.delay.summary(),
        generated_intervals: pacing.intervals.summary(),
        generator_overruns: pacing.overruns,
        client_start_delay: totals.client_start_delay.summary(),
        outside_server_latency: totals.outside_server.summary(),
        slowest_packets: totals.slowest_packets,
        runtime_lag,
    })
}
