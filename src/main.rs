use std::{
    net::SocketAddr,
    path::PathBuf,
    process::{ExitStatus, Stdio},
    time::Duration,
};

use clap::Parser;
use serde::Serialize;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio_util::sync::CancellationToken;
use voice_scheduler::{
    config::{RuntimeConfig, WorkerSlowdown},
    simulation::{self, ArrivalPhase, SimulationConfig, SimulationError, SimulationReport},
    transport::{Gateway, GatewayConfig, GatewayError, GatewayReport},
};

mod cli;
use cli::{BenchmarkArguments, Cli, Command};

#[derive(Debug, thiserror::Error)]
enum ApplicationError {
    #[error(transparent)]
    Gateway(#[from] GatewayError),
    #[error(transparent)]
    Simulation(#[from] SimulationError),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Report(#[from] serde_json::Error),
    #[error(transparent)]
    Task(#[from] tokio::task::JoinError),
    #[error("external simulator exited with {0}")]
    SimulatorExited(ExitStatus),
    #[error("calibration found no realtime capacity")]
    NoCapacity,
}

type ApplicationResult<T> = Result<T, ApplicationError>;

#[derive(Serialize)]
struct BenchmarkReport {
    client: SimulationReport,
    gateway: GatewayReport,
}
#[derive(Serialize)]
struct ScenarioReport {
    scenario: Scenario,
    result: BenchmarkReport,
}
#[derive(Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
enum Scenario {
    LowLoad,
    HalfCapacity,
    EightyPercentCapacity,
    NinetyFivePercentCapacity,
    Overload,
    Aligned,
    Random,
    Jitter,
    Churn,
    CacheReplay,
    Slowdown,
    ChannelSaturation,
}

async fn benchmark(
    runtime: RuntimeConfig,
    workload: SimulationConfig,
) -> ApplicationResult<BenchmarkReport> {
    workload.validate()?;
    let gateway = Gateway::bind(
        runtime,
        GatewayConfig {
            listen_address: SocketAddr::from(([127, 0, 0, 1], 0)),
            ..GatewayConfig::default()
        },
    )
    .await?;
    let address = gateway.local_address()?;
    let cancellation = CancellationToken::new();
    let _server_cleanup = cancellation.clone().drop_guard();
    let signal = cancellation.clone();
    let server = tokio::spawn(gateway.serve(signal));
    let client = run_external_simulator(address, &workload).await;
    cancellation.cancel();
    let gateway = server.await??;
    let client = client?;
    print_benchmark(&client, &gateway);
    Ok(BenchmarkReport { client, gateway })
}

async fn run_external_simulator(
    address: SocketAddr,
    workload: &SimulationConfig,
) -> ApplicationResult<SimulationReport> {
    let mut child = tokio::process::Command::new(std::env::current_exe()?)
        .args(["simulate-stdin", "--address", &address.to_string()])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .kill_on_drop(true)
        .spawn()?;
    let mut input = child.stdin.take().expect("piped stdin");
    input.write_all(&serde_json::to_vec(workload)?).await?;
    drop(input);
    let output = child.wait_with_output().await?;
    if !output.status.success() {
        return Err(ApplicationError::SimulatorExited(output.status));
    }
    Ok(serde_json::from_slice(&output.stdout)?)
}

fn print_benchmark(client: &SimulationReport, gateway: &GatewayReport) {
    eprintln!(
        "admitted={} rejected={} failed={} echoed={} late={} discarded={} bursts={} RTT p50/p95/p99/max={:.1}/{:.1}/{:.1}/{:.1}ms batch fill={:.1}%",
        client.counters.admitted_sessions,
        client.counters.rejected_capacity,
        client.counters.failed_sessions,
        client.counters.echoed_frames,
        client.counters.late_frames,
        client.counters.discarded_output_frames,
        client.counters.quality_failed_sessions,
        client.round_trip_latency.p50_ms,
        client.round_trip_latency.p95_ms,
        client.round_trip_latency.p99_ms,
        client.round_trip_latency.max_ms,
        gateway.runtime.batch_fill_ratio * 100.0
    );
}

async fn suite(arguments: BenchmarkArguments) -> ApplicationResult<Vec<ScenarioReport>> {
    let runtime = arguments.runtime.configuration();
    let workload = arguments.workload.configuration();
    let mut reports = Vec::new();
    let mut low = workload.clone();
    low.sessions = runtime.workers;
    let baseline = benchmark(runtime.clone(), low).await?;
    let capacity = baseline
        .gateway
        .runtime
        .workers
        .iter()
        .map(|worker| worker.initial_session_limit)
        .sum::<usize>();
    if capacity == 0 {
        return Err(ApplicationError::NoCapacity);
    }
    reports.push(ScenarioReport {
        scenario: Scenario::LowLoad,
        result: baseline,
    });
    for (name, sessions) in [
        (Scenario::HalfCapacity, capacity / 2),
        (Scenario::EightyPercentCapacity, capacity * 8 / 10),
        (Scenario::NinetyFivePercentCapacity, capacity * 95 / 100),
        (Scenario::Overload, capacity * 2),
    ] {
        let mut scenario = workload.clone();
        scenario.sessions = sessions.max(1);
        reports.push(ScenarioReport {
            scenario: name,
            result: benchmark(runtime.clone(), scenario).await?,
        });
    }
    for name in [
        Scenario::Aligned,
        Scenario::Random,
        Scenario::Jitter,
        Scenario::Churn,
        Scenario::CacheReplay,
        Scenario::Slowdown,
        Scenario::ChannelSaturation,
    ] {
        let mut scenario = workload.clone();
        scenario.sessions = (capacity / 2).max(1);
        let mut node = runtime.clone();
        match name {
            Scenario::Aligned => {
                scenario.phase = ArrivalPhase::Aligned;
                scenario.minimum_interval = Duration::from_millis(50);
                scenario.maximum_interval = scenario.minimum_interval;
            }
            Scenario::Random => {
                scenario.phase = ArrivalPhase::Random;
                scenario.minimum_interval = Duration::from_millis(50);
                scenario.maximum_interval = scenario.minimum_interval;
            }
            Scenario::Jitter => {
                scenario.minimum_interval = Duration::from_millis(48);
                scenario.maximum_interval = Duration::from_millis(55);
            }
            Scenario::Churn => scenario.churn_after = Some(Duration::from_secs(1)),
            Scenario::CacheReplay => scenario.evict_every = Some(10),
            Scenario::Slowdown => {
                node.slowdown = Some(WorkerSlowdown {
                    after: workload.duration / 2,
                    inference_latency: Duration::from_millis(30),
                })
            }
            Scenario::ChannelSaturation => {
                node.worker_channel_capacity = 1;
                node.worker_input_delay = Duration::from_millis(20);
            }
            _ => unreachable!("fixed scenario list"),
        }
        reports.push(ScenarioReport {
            scenario: name,
            result: benchmark(node, scenario).await?,
        });
    }
    Ok(reports)
}

async fn write_report<T: Serialize>(report: &T, output: Option<PathBuf>) -> ApplicationResult<()> {
    let data = serde_json::to_string_pretty(report)?;
    if let Some(path) = output {
        tokio::fs::write(path, data).await?;
    } else {
        println!("{data}");
    }
    Ok(())
}

async fn serve_gateway(gateway: Gateway) -> ApplicationResult<GatewayReport> {
    let cancellation = CancellationToken::new();
    let serving = gateway.serve(cancellation.clone());
    tokio::pin!(serving);
    tokio::select! {
        report = &mut serving => Ok(report?),
        interrupt = tokio::signal::ctrl_c() => {
            cancellation.cancel();
            let report = serving.await?;
            interrupt?;
            Ok(report)
        }
    }
}

async fn read_workload() -> ApplicationResult<SimulationConfig> {
    let mut data = Vec::new();
    tokio::io::stdin()
        .take(1024 * 1024)
        .read_to_end(&mut data)
        .await?;
    Ok(serde_json::from_slice(&data)?)
}

#[tokio::main]
async fn main() -> ApplicationResult<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Serve { listen, runtime } => {
            let gateway = Gateway::bind(
                runtime.configuration(),
                GatewayConfig {
                    listen_address: listen,
                    ..GatewayConfig::default()
                },
            )
            .await?;
            eprintln!(
                "ready: {} (Ctrl+C to stop and write metrics)",
                gateway.local_address()?
            );
            let report = serve_gateway(gateway).await?;
            write_report(&report, cli.output).await
        }
        Command::Simulate { address, workload } => {
            let configuration = workload.configuration();
            let report = simulation::run(address, configuration).await?;
            write_report(&report, cli.output).await
        }
        Command::SimulateFromStdin { address } => {
            let report = simulation::run(address, read_workload().await?).await?;
            write_report(&report, cli.output).await
        }
        Command::Benchmark(arguments) => {
            write_report(
                &benchmark(
                    arguments.runtime.configuration(),
                    arguments.workload.configuration(),
                )
                .await?,
                cli.output,
            )
            .await
        }
        Command::Suite(arguments) => write_report(&suite(arguments).await?, cli.output).await,
    }
}
