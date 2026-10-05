use std::{path::PathBuf, time::Duration};

use clap::{Args, Parser, Subcommand};
use serde::Serialize;
use tokio_util::sync::CancellationToken;
use voice_scheduler::{
    config::{RuntimeConfig, WorkerSlowdown},
    metrics::{LatencyDistribution, Report},
    simulation::{
        ArrivalPhase, BenchmarkResult, ChurnConfig, SimulationConfig, SimulationError,
        benchmark_suite, run_benchmark,
    },
};

#[derive(Parser)]
#[command(
    version,
    about = "Benchmark a bounded realtime voice inference runtime"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Run one seeded workload.
    Benchmark(BenchmarkArguments),
    /// Compare load, phase, jitter, churn, saturation, and slowdown scenarios.
    Suite(SuiteArguments),
}

#[derive(Args)]
struct RuntimeArguments {
    #[arg(long, default_value_t = RuntimeConfig::default().workers)]
    workers: usize,
    #[arg(long, default_value_t = RuntimeConfig::default().batch_size)]
    batch_size: usize,
    #[arg(long, default_value_t = RuntimeConfig::default().tick_interval.as_millis() as u64)]
    tick_ms: u64,
    #[arg(long, default_value_t = RuntimeConfig::default().inference_latency.as_millis() as u64)]
    inference_ms: u64,
    #[arg(long, default_value_t = RuntimeConfig::default().max_sessions_per_worker)]
    max_sessions_per_worker: usize,
    #[arg(long, default_value_t = RuntimeConfig::default().cache_slots_per_worker)]
    cache_slots: usize,
    #[arg(long, default_value_t = RuntimeConfig::default().ingress_capacity)]
    ingress_capacity: usize,
    #[arg(long, default_value_t = RuntimeConfig::default().worker_channel_capacity)]
    worker_capacity: usize,
    #[arg(long, default_value_t = RuntimeConfig::default().result_channel_capacity)]
    result_capacity: usize,
    #[arg(long, default_value_t = RuntimeConfig::default().output_channel_capacity)]
    output_capacity: usize,
    #[arg(long, default_value_t = RuntimeConfig::default().max_frame_bytes)]
    max_frame_bytes: usize,
    #[arg(long, default_value_t = RuntimeConfig::default().max_input_age.as_millis() as u64)]
    max_input_age_ms: u64,
    #[arg(long, default_value_t = RuntimeConfig::default().session_timeout.as_millis() as u64)]
    session_timeout_ms: u64,
    #[arg(long, default_value_t = 0)]
    phase_bucket_ms: u64,
    #[arg(long, default_value_t = RuntimeConfig::default().scheduling_margin.as_millis() as u64)]
    scheduling_margin_ms: u64,
    /// Artificial input-consumer delay for saturation experiments.
    #[arg(long, default_value_t = RuntimeConfig::default().worker_input_delay.as_millis() as u64)]
    worker_input_delay_ms: u64,
    #[arg(long, requires = "slowdown_inference_ms")]
    slowdown_after_ms: Option<u64>,
    #[arg(long, requires = "slowdown_after_ms")]
    slowdown_inference_ms: Option<u64>,
}

impl RuntimeArguments {
    fn into_configuration(self) -> RuntimeConfig {
        let slowdown = self.slowdown_after_ms.map(|after| WorkerSlowdown {
            after: Duration::from_millis(after),
            inference_latency: Duration::from_millis(
                self.slowdown_inference_ms
                    .expect("clap requires slowdown latency"),
            ),
        });
        RuntimeConfig {
            workers: self.workers,
            tick_interval: Duration::from_millis(self.tick_ms),
            batch_size: self.batch_size,
            inference_latency: Duration::from_millis(self.inference_ms),
            max_sessions_per_worker: self.max_sessions_per_worker,
            cache_slots_per_worker: self.cache_slots,
            ingress_capacity: self.ingress_capacity,
            worker_channel_capacity: self.worker_capacity,
            result_channel_capacity: self.result_capacity,
            output_channel_capacity: self.output_capacity,
            max_frame_bytes: self.max_frame_bytes,
            max_input_age: Duration::from_millis(self.max_input_age_ms),
            session_timeout: Duration::from_millis(self.session_timeout_ms),
            phase_bucket: (self.phase_bucket_ms > 0)
                .then(|| Duration::from_millis(self.phase_bucket_ms)),
            scheduling_margin: Duration::from_millis(self.scheduling_margin_ms),
            worker_input_delay: Duration::from_millis(self.worker_input_delay_ms),
            slowdown,
        }
    }
}

#[derive(Args)]
struct OutputArguments {
    /// Emit structured JSON to stdout instead of the text summary.
    #[arg(long)]
    json: bool,
    /// Also save the complete JSON report to this file.
    #[arg(long)]
    output: Option<PathBuf>,
}

#[derive(Args)]
struct BenchmarkArguments {
    #[command(flatten)]
    runtime: RuntimeArguments,
    #[arg(long, default_value_t = SimulationConfig::default().sessions)]
    sessions: usize,
    #[arg(long, default_value_t = SimulationConfig::default().duration.as_secs())]
    duration_secs: u64,
    #[arg(long, default_value_t = SimulationConfig::default().jitter.as_millis() as u64)]
    jitter_ms: u64,
    #[arg(long, value_enum, default_value_t = SimulationConfig::default().arrival_phase)]
    arrival_phase: ArrivalPhase,
    #[arg(long, requires = "churn_max_ms")]
    churn_min_ms: Option<u64>,
    #[arg(long, requires = "churn_min_ms")]
    churn_max_ms: Option<u64>,
    #[arg(long, default_value_t = SimulationConfig::default().seed)]
    seed: u64,
    #[command(flatten)]
    output: OutputArguments,
}

#[derive(Args)]
struct SuiteArguments {
    #[command(flatten)]
    runtime: RuntimeArguments,
    /// Duration of each scenario; the suite has 14 scenarios by default.
    #[arg(long, default_value_t = 10)]
    duration_secs: u64,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[command(flatten)]
    output: OutputArguments,
}

#[derive(Serialize)]
struct ScenarioResult {
    name: &'static str,
    result: BenchmarkResult,
}

#[derive(Debug, thiserror::Error)]
enum CliError {
    #[error(transparent)]
    Simulation(#[from] SimulationError),
    #[error(transparent)]
    Configuration(#[from] voice_scheduler::config::ConfigError),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

#[tokio::main]
async fn main() -> Result<(), CliError> {
    let cancellation = CancellationToken::new();
    let work = execute(Cli::parse().command, cancellation.clone());
    tokio::pin!(work);
    tokio::select! {
        result = &mut work => result,
        signal = tokio::signal::ctrl_c() => {
            signal?;
            cancellation.cancel();
            work.await
        }
    }
}

async fn execute(command: Command, cancellation: CancellationToken) -> Result<(), CliError> {
    match command {
        Command::Benchmark(arguments) => {
            let churn = arguments.churn_min_ms.map(|minimum| ChurnConfig {
                min_duration: Duration::from_millis(minimum),
                max_duration: Duration::from_millis(
                    arguments
                        .churn_max_ms
                        .expect("clap requires maximum churn duration"),
                ),
            });
            let simulation = SimulationConfig {
                sessions: arguments.sessions,
                duration: Duration::from_secs(arguments.duration_secs),
                jitter: Duration::from_millis(arguments.jitter_ms),
                arrival_phase: arguments.arrival_phase,
                churn,
                seed: arguments.seed,
            };
            let result = run_benchmark(
                arguments.runtime.into_configuration(),
                simulation,
                cancellation,
            )
            .await?;
            let json = serde_json::to_string_pretty(&result)?;
            if arguments.output.json {
                println!("{json}");
            } else {
                print_report(&result.runtime);
            }
            save_report(arguments.output.output, json).await?;
        }
        Command::Suite(arguments) => {
            let runtime = arguments.runtime.into_configuration();
            runtime.validate()?;
            let mut results = Vec::new();
            for scenario in benchmark_suite(
                &runtime,
                Duration::from_secs(arguments.duration_secs),
                arguments.seed,
            ) {
                eprintln!(
                    "Running {} ({} sessions)",
                    scenario.name, scenario.simulation.sessions
                );
                let result =
                    run_benchmark(scenario.runtime, scenario.simulation, cancellation.clone())
                        .await?;
                let interrupted = result.workload.interrupted;
                results.push(ScenarioResult {
                    name: scenario.name,
                    result,
                });
                if interrupted {
                    break;
                }
            }
            let json = serde_json::to_string_pretty(&results)?;
            if arguments.output.json {
                println!("{json}");
            } else {
                print_suite(&results);
            }
            save_report(arguments.output.output, json).await?;
        }
    }
    Ok(())
}

async fn save_report(path: Option<PathBuf>, json: String) -> Result<(), std::io::Error> {
    if let Some(path) = path {
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            tokio::fs::create_dir_all(parent).await?;
        }
        tokio::fs::write(path, json).await?;
    }
    Ok(())
}

fn print_report(report: &Report) {
    println!(
        "Admitted: {}  Rejected: {}  Peak active: {}",
        report.admitted_sessions, report.rejected_sessions, report.peak_active_sessions
    );
    println!(
        "Delivered: {}  Throughput: {:.1} frames/s  Elapsed: {:.2}s",
        report.delivered_results, report.throughput_frames_per_sec, report.elapsed_secs
    );
    println!(
        "Batches: {}  Mean size: {:.2}  Fill: {:.1}%  Device utilization: {:.1}%",
        report.batches,
        report.mean_batch_size,
        report.batch_fill_ratio * 100.0,
        report.worker_utilization * 100.0
    );
    println!(
        "Deadline misses: {} ({:.2}%)  Stale results: {}",
        report.deadline_misses,
        report.deadline_miss_ratio * 100.0,
        report.stale_results_discarded
    );
    println!(
        "Overloaded inputs: {}  Coalesced inputs: {}  Stale inputs: {}  Channel saturation: {}",
        report.inputs_overloaded,
        report.coalesced_inputs,
        report.inputs_stale,
        report.channel_saturation_events
    );
    println!("Latency (ms)         p50       p95       p99       max");
    print_latency("Queue", &report.queue_delay);
    print_latency("Inference", &report.inference_latency);
    print_latency("End to end", &report.end_to_end_latency);
    print_latency("Deadline lateness", &report.deadline_lateness);
    print_latency("Phase added", &report.phase_added_latency);
}

fn print_latency(name: &str, latency: &LatencyDistribution) {
    println!(
        "{name:<18} {:>8.2}  {:>8.2}  {:>8.2}  {:>8.2}",
        latency.p50_ms, latency.p95_ms, latency.p99_ms, latency.max_ms
    );
}

fn print_suite(results: &[ScenarioResult]) {
    println!(
        "Scenario              Admit Reject    FPS   Fill%   Util%   Miss%  E2E p99  Saturation"
    );
    for scenario in results {
        let report = &scenario.result.runtime;
        println!(
            "{:<22} {:>5} {:>6} {:>6.0} {:>7.1} {:>7.1} {:>7.2} {:>8.2} {:>11}",
            scenario.name,
            report.admitted_sessions,
            report.rejected_sessions,
            report.throughput_frames_per_sec,
            report.batch_fill_ratio * 100.0,
            report.worker_utilization * 100.0,
            report.deadline_miss_ratio * 100.0,
            report.end_to_end_latency.p99_ms,
            report.channel_saturation_events
        );
    }
}
