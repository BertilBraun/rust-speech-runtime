use std::{path::Path, process::ExitCode};

use clap::Parser;
use serde::Serialize;
use tokio_util::sync::CancellationToken;
use voice_scheduler::{
    config::RuntimeConfig,
    simulation::{self, DEFAULT_AUDIO_PACKET_MS, SimulationError, SimulationReport},
    transport::{Gateway, GatewayError},
};

mod cli;
use cli::{BenchmarkArguments, Cli, Command, ServeArguments, SuiteArguments};

#[derive(Debug, thiserror::Error)]
enum ApplicationError {
    #[error(transparent)]
    Gateway(#[from] GatewayError),
    #[error(transparent)]
    Simulation(#[from] SimulationError),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Task(#[from] tokio::task::JoinError),
    #[error("configuration file must be at most 64 KiB")]
    ConfigurationTooLarge,
}

#[tokio::main]
async fn main() -> ExitCode {
    match run(Cli::parse()).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}

async fn run(cli: Cli) -> Result<(), ApplicationError> {
    match cli.command {
        Command::Serve(arguments) => serve(arguments).await,
        Command::Benchmark(arguments) => benchmark(arguments).await,
        Command::Suite(arguments) => suite(arguments).await,
    }
}

async fn serve(arguments: ServeArguments) -> Result<(), ApplicationError> {
    let metadata = tokio::fs::metadata(&arguments.runtime_config).await?;
    if metadata.len() > 64 * 1024 {
        return Err(ApplicationError::ConfigurationTooLarge);
    }
    let runtime: RuntimeConfig =
        serde_json::from_slice(&tokio::fs::read(&arguments.runtime_config).await?)?;
    let gateway = Gateway::bind(runtime, arguments.gateway_config()).await?;
    println!(
        "WebSocket gateway listening on ws://{}/v1",
        gateway.local_address()?
    );
    let cancellation = CancellationToken::new();
    let _cleanup = cancellation.clone().drop_guard();
    let mut server = tokio::spawn(gateway.serve(cancellation.clone()));
    let report = tokio::select! {
        result = &mut server => result??,
        signal = tokio::signal::ctrl_c() => {
            signal?;
            cancellation.cancel();
            server.await??
        }
    };
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

async fn benchmark(arguments: BenchmarkArguments) -> Result<(), ApplicationError> {
    let report = simulation::run(&arguments.url, arguments.workload()).await?;
    print_summary(&report);
    if let Some(path) = arguments.report {
        save_report(&path, &report).await?;
    }
    Ok(())
}

#[derive(Serialize)]
struct ScenarioReport {
    scenario: &'static str,
    result: SimulationReport,
}

async fn suite(arguments: SuiteArguments) -> Result<(), ApplicationError> {
    if arguments.benchmark.measurement_secs.is_some() {
        return Err(
            SimulationError::Configuration("timed cohorts use benchmark, not suite").into(),
        );
    }
    let mut reports = Vec::new();
    for (name, fraction) in [
        ("low_load", 0.10),
        ("half_capacity", 0.50),
        ("eighty_percent", 0.80),
        ("ninety_five_percent", 0.95),
        ("overload", 1.20),
    ] {
        let mut workload = arguments.benchmark.workload();
        workload.sessions = (arguments.session_budget as f64 * fraction).ceil().max(1.0) as usize;
        let result = simulation::run(&arguments.benchmark.url, workload).await?;
        println!("scenario: {name}");
        print_summary(&result);
        reports.push(ScenarioReport {
            scenario: name,
            result,
        });
    }
    for name in ["aligned", "jitter", "churn", "interruption"] {
        let mut workload = arguments.benchmark.workload();
        match name {
            "aligned" => {
                workload.start_spread_ms = 0;
                workload.minimum_packet_ms = DEFAULT_AUDIO_PACKET_MS;
                workload.maximum_packet_ms = DEFAULT_AUDIO_PACKET_MS;
            }
            "jitter" => {
                workload.minimum_packet_ms = DEFAULT_AUDIO_PACKET_MS;
                workload.maximum_packet_ms = DEFAULT_AUDIO_PACKET_MS + 10;
            }
            "churn" => workload.churn_rounds = 3,
            "interruption" => workload.interrupt_after_tokens = Some(2),
            _ => unreachable!("scenario list is fixed"),
        }
        let result = simulation::run(&arguments.benchmark.url, workload).await?;
        println!("scenario: {name}");
        print_summary(&result);
        reports.push(ScenarioReport {
            scenario: name,
            result,
        });
    }
    if let Some(path) = arguments.benchmark.report {
        save_report(&path, &reports).await?;
    }
    Ok(())
}

fn print_summary(report: &SimulationReport) {
    if let Some(measured) = &report.steady_state {
        println!(
            "steady measurement: {:.1} s ramp-up, {:.1} s measured; {}/{} sessions produced tokens; {:.1} tokens/s; TTFT p50/p95/p99/max {:.1}/{:.1}/{:.1}/{:.1} ms",
            measured.ramp_up_seconds,
            measured.measurement_seconds,
            measured.sessions_with_tokens,
            report.offered_sessions,
            measured.aggregate_tokens_per_second,
            measured.traffic.time_to_first_token_ms.p50_ms,
            measured.traffic.time_to_first_token_ms.p95_ms,
            measured.traffic.time_to_first_token_ms.p99_ms,
            measured.traffic.time_to_first_token_ms.max_ms
        );
        println!("lifetime totals below include ramp-up and draining:");
    }
    println!(
        "sessions: {} admitted, {} rejected, {} failed; {} tokens ({:.1} aggregate tokens/s)",
        report.admitted_sessions,
        report.rejected_sessions,
        report.failed_sessions,
        report.received_tokens,
        report.aggregate_tokens_per_second
    );
    println!(
        "turns: {} admitted, {} completed, {} token-limited, {} interrupted, {} rejected",
        report.admitted_turns,
        report.completed_turns,
        report.token_limited_turns,
        report.interrupted_turns,
        report.rejected_turns
    );
    println!(
        "TTFT p50/p95/p99/max: {:.1}/{:.1}/{:.1}/{:.1} ms",
        report.time_to_first_token_ms.p50_ms,
        report.time_to_first_token_ms.p95_ms,
        report.time_to_first_token_ms.p99_ms,
        report.time_to_first_token_ms.max_ms
    );
    println!(
        "endpoint confirmation: {} ms configured, preparation {}; last audio to first token p50/p95/p99/max: {:.1}/{:.1}/{:.1}/{:.1} ms",
        report.endpointing_ms,
        report.prepare_before_commit,
        report.end_of_audio_to_first_token_ms.p50_ms,
        report.end_of_audio_to_first_token_ms.p95_ms,
        report.end_of_audio_to_first_token_ms.p99_ms,
        report.end_of_audio_to_first_token_ms.max_ms,
    );
    println!(
        "token gap p50/p95/p99/max: {:.1}/{:.1}/{:.1}/{:.1} ms; {} gaps above target; {} sessions below rolling rate",
        report.token_gap_ms.p50_ms,
        report.token_gap_ms.p95_ms,
        report.token_gap_ms.p99_ms,
        report.token_gap_ms.max_ms,
        report.token_gaps_over_target,
        report.sessions_with_rolling_rate_violations
    );
}

async fn save_report<T: Serialize + Sync>(path: &Path, report: &T) -> Result<(), ApplicationError> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        tokio::fs::create_dir_all(parent).await?;
    }
    tokio::fs::write(path, serde_json::to_vec_pretty(report)?).await?;
    Ok(())
}
