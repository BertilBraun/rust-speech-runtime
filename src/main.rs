use std::{error::Error, net::SocketAddr, path::PathBuf, process::Stdio, time::Duration};

use clap::{Args, Parser, Subcommand};
use serde::Serialize;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio_util::sync::CancellationToken;
use voice_scheduler::{
    config::{DeviceWait, RuntimeConfig, WorkerSlowdown},
    simulation::{self, ArrivalPhase, SimulationConfig, SimulationReport},
    transport::{Gateway, GatewayConfig, GatewayReport},
};

type ApplicationResult<T> = Result<T, Box<dyn Error + Send + Sync>>;

#[derive(Parser)]
#[command(about = "Realtime audio echo gateway and external TCP benchmark")]
struct Cli {
    #[arg(long, global = true)]
    output: Option<PathBuf>,
    #[command(subcommand)]
    command: Command,
}
#[derive(Subcommand)]
enum Command {
    Serve {
        #[arg(long, default_value = "127.0.0.1:9000")]
        listen: SocketAddr,
        #[command(flatten)]
        runtime: RuntimeArguments,
    },
    Simulate {
        #[arg(long, default_value = "127.0.0.1:9000")]
        address: SocketAddr,
        #[arg(long, hide = true)]
        config_stdin: bool,
        #[command(flatten)]
        workload: WorkloadArguments,
    },
    Benchmark(BenchmarkArguments),
    Suite(BenchmarkArguments),
}
#[derive(Clone, Default, Args)]
struct RuntimeArguments {
    #[arg(long)]
    workers: Option<usize>,
    #[arg(long)]
    batch_size: Option<usize>,
    #[arg(long)]
    inference_ms: Option<u64>,
    #[arg(long)]
    device_wait: Option<DeviceWait>,
    #[arg(long)]
    device_queue_capacity: Option<usize>,
    #[arg(long)]
    launch_ahead_us: Option<u64>,
    #[arg(long)]
    deadline_ms: Option<u64>,
    #[arg(long)]
    lateness_grace_ms: Option<u64>,
    #[arg(long)]
    batch_wait_ms: Option<u64>,
    #[arg(long)]
    max_sessions_per_worker: Option<usize>,
    #[arg(long)]
    admission_headroom: Option<f64>,
    #[arg(long)]
    batch_fill_reserve: Option<f64>,
    #[arg(long)]
    latency_safety_factor: Option<f64>,
    #[arg(long)]
    worker_channel_capacity: Option<usize>,
    #[arg(long)]
    worker_input_delay_ms: Option<u64>,
    #[arg(long, requires = "slowdown_ms")]
    slowdown_after_secs: Option<u64>,
    #[arg(long, requires = "slowdown_after_secs")]
    slowdown_ms: Option<u64>,
}
impl RuntimeArguments {
    fn configuration(&self) -> RuntimeConfig {
        let mut configuration = RuntimeConfig::default();
        if let Some(value) = self.workers {
            configuration.workers = value;
        }
        if let Some(value) = self.batch_size {
            configuration.batch_size = value;
        }
        if let Some(value) = self.inference_ms {
            configuration.inference_latency = Duration::from_millis(value);
        }
        if let Some(value) = self.device_wait {
            configuration.device_wait = value;
        }
        if let Some(value) = self.device_queue_capacity {
            configuration.device_queue_capacity = value;
        }
        if let Some(value) = self.launch_ahead_us {
            configuration.launch_ahead = Duration::from_micros(value);
        }
        if let Some(value) = self.deadline_ms {
            configuration.packet_deadline = Duration::from_millis(value);
        }
        if let Some(value) = self.lateness_grace_ms {
            configuration.packet_lateness_grace = Duration::from_millis(value);
        }
        if let Some(value) = self.batch_wait_ms {
            configuration.max_batch_wait = Duration::from_millis(value);
        }
        if let Some(value) = self.max_sessions_per_worker {
            configuration.max_sessions_per_worker = value;
        }
        if let Some(value) = self.admission_headroom {
            configuration.admission_headroom = value;
        }
        if let Some(value) = self.batch_fill_reserve {
            configuration.batch_fill_reserve = value;
        }
        if let Some(value) = self.latency_safety_factor {
            configuration.latency_safety_factor = value;
        }
        if let Some(value) = self.worker_channel_capacity {
            configuration.worker_channel_capacity = value;
        }
        if let Some(value) = self.worker_input_delay_ms {
            configuration.worker_input_delay = Duration::from_millis(value);
        }
        if let (Some(after), Some(latency)) = (self.slowdown_after_secs, self.slowdown_ms) {
            configuration.slowdown = Some(WorkerSlowdown {
                after: Duration::from_secs(after),
                inference_latency: Duration::from_millis(latency),
            });
        }
        configuration
    }
}
#[derive(Clone, Default, Args)]
struct WorkloadArguments {
    #[arg(long)]
    sessions: Option<usize>,
    #[arg(long)]
    duration_secs: Option<u64>,
    #[arg(long)]
    min_interval_ms: Option<u64>,
    #[arg(long)]
    max_interval_ms: Option<u64>,
    #[arg(long)]
    payload_bytes: Option<usize>,
    #[arg(long, value_enum)]
    phase: Option<ArrivalPhase>,
    #[arg(long)]
    seed: Option<u64>,
    #[arg(long)]
    evict_every: Option<u64>,
    #[arg(long)]
    churn_secs: Option<u64>,
    #[arg(long)]
    quality_window_packets: Option<usize>,
    #[arg(long)]
    quality_miss_limit: Option<usize>,
}
impl WorkloadArguments {
    fn configuration(&self) -> SimulationConfig {
        let mut configuration = SimulationConfig::default();
        if let Some(value) = self.sessions {
            configuration.sessions = value;
        }
        if let Some(value) = self.duration_secs {
            configuration.duration = Duration::from_secs(value);
        }
        if let Some(value) = self.min_interval_ms {
            configuration.minimum_interval = Duration::from_millis(value);
        }
        if let Some(value) = self.max_interval_ms {
            configuration.maximum_interval = Duration::from_millis(value);
        }
        if let Some(value) = self.payload_bytes {
            configuration.payload_bytes = value;
        }
        if let Some(value) = self.phase {
            configuration.phase = value;
        }
        if let Some(value) = self.seed {
            configuration.seed = value;
        }
        configuration.evict_every = self.evict_every;
        configuration.churn_after = self.churn_secs.map(Duration::from_secs);
        if let Some(value) = self.quality_window_packets {
            configuration.quality.window_packets = value;
        }
        if let Some(value) = self.quality_miss_limit {
            configuration.quality.miss_limit = value;
        }
        configuration
    }
}
#[derive(Clone, Args)]
struct BenchmarkArguments {
    #[command(flatten)]
    runtime: RuntimeArguments,
    #[command(flatten)]
    workload: WorkloadArguments,
}
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
            listen_address: "127.0.0.1:0".parse()?,
            ..GatewayConfig::default()
        },
    )
    .await?;
    let address = gateway.local_address()?;
    let cancellation = CancellationToken::new();
    let signal = cancellation.clone();
    let server = tokio::spawn(gateway.serve(signal));
    let client = async {
        let mut child = tokio::process::Command::new(std::env::current_exe()?)
            .args([
                "simulate",
                "--address",
                &address.to_string(),
                "--config-stdin",
            ])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .kill_on_drop(true)
            .spawn()?;
        let mut input = child.stdin.take().expect("piped stdin");
        input.write_all(&serde_json::to_vec(&workload)?).await?;
        drop(input);
        let output = child.wait_with_output().await?;
        if !output.status.success() {
            return Err(format!("external simulator exited with {}", output.status).into());
        }
        Ok::<SimulationReport, Box<dyn Error + Send + Sync>>(serde_json::from_slice(
            &output.stdout,
        )?)
    }
    .await;
    cancellation.cancel();
    let gateway = server.await??;
    let client = client?;
    eprintln!(
        "admitted={} rejected={} failed={} echoed={} late={} bursts={} RTT p50/p95/p99={:.1}/{:.1}/{:.1}ms batch fill={:.1}%",
        client.counters.admitted_sessions,
        client.counters.rejected_capacity,
        client.counters.failed_sessions,
        client.counters.echoed_frames,
        client.counters.late_frames,
        client.counters.quality_failed_sessions,
        client.round_trip_latency.p50_ms,
        client.round_trip_latency.p95_ms,
        client.round_trip_latency.p99_ms,
        gateway.runtime.batch_fill_ratio * 100.0
    );
    Ok(BenchmarkReport { client, gateway })
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
        return Err("calibration found no realtime capacity".into());
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
            let cancellation = CancellationToken::new();
            let signal = cancellation.clone();
            let interrupt = tokio::spawn(async move {
                if tokio::signal::ctrl_c().await.is_ok() {
                    signal.cancel();
                }
            });
            let report = gateway.serve(cancellation).await?;
            interrupt.abort();
            write_report(&report, cli.output).await
        }
        Command::Simulate {
            address,
            config_stdin,
            workload,
        } => {
            let configuration = if config_stdin {
                let mut data = Vec::new();
                tokio::io::stdin()
                    .take(1024 * 1024)
                    .read_to_end(&mut data)
                    .await?;
                serde_json::from_slice::<SimulationConfig>(&data)?
            } else {
                workload.configuration()
            };
            let report = simulation::run(address, configuration).await?;
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
