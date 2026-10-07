use std::{net::SocketAddr, path::PathBuf, time::Duration};

use clap::{Args, Parser, Subcommand};
use voice_scheduler::{
    simulation::{DEFAULT_AUDIO_PACKET_MS, SimulationConfig},
    transport::GatewayConfig,
};

#[derive(Parser)]
#[command(
    version,
    about = "Turn-based speech gateway with sticky GPU workers and bounded scheduling"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand)]
pub enum Command {
    Serve(ServeArguments),
    Benchmark(BenchmarkArguments),
    Suite(SuiteArguments),
}

#[derive(Args)]
pub struct ServeArguments {
    /// Canonical RuntimeConfig JSON with an arbitrary list of GPU worker endpoints.
    #[arg(long)]
    pub runtime_config: PathBuf,
    #[arg(long, default_value = "127.0.0.1:8080")]
    pub listen: SocketAddr,
    #[arg(long, default_value = "session-archives")]
    pub archive_directory: PathBuf,
    #[arg(long)]
    pub no_archive: bool,
    #[arg(long, default_value_t = 4096)]
    pub max_connections: usize,
}

impl ServeArguments {
    pub fn gateway_config(&self) -> GatewayConfig {
        GatewayConfig {
            listen_address: self.listen,
            archive_directory: (!self.no_archive).then(|| self.archive_directory.clone()),
            max_connections: self.max_connections,
            ..GatewayConfig::default()
        }
    }
}

#[derive(Args, Clone)]
pub struct BenchmarkArguments {
    #[arg(long, default_value = "ws://127.0.0.1:8080/v1")]
    pub url: String,
    #[arg(long, default_value_t = 8)]
    pub sessions: usize,
    #[arg(long, default_value_t = 2)]
    pub turns: usize,
    #[arg(long, default_value_t = 1000)]
    pub utterance_ms: u64,
    #[arg(long, default_value_t = DEFAULT_AUDIO_PACKET_MS)]
    pub minimum_packet_ms: u64,
    #[arg(long, default_value_t = DEFAULT_AUDIO_PACKET_MS)]
    pub maximum_packet_ms: u64,
    #[arg(long, default_value_t = 500)]
    pub start_spread_ms: u64,
    #[arg(long, default_value_t = 250)]
    pub think_ms: u64,
    /// Delay between the last audio packet and confirmed end of turn.
    #[arg(long, default_value_t = 0)]
    pub endpointing_ms: u64,
    /// Prepare inference provisionally during the endpoint confirmation delay.
    #[arg(long)]
    pub prepare_before_commit: bool,
    #[arg(long, default_value_t = 4.0)]
    pub target_tokens_per_second: f64,
    #[arg(long, default_value_t = 2000)]
    pub throughput_window_ms: u64,
    #[arg(long, default_value_t = 120)]
    pub timeout_secs: u64,
    #[arg(long)]
    pub interrupt_after_tokens: Option<usize>,
    #[arg(long, default_value_t = 1)]
    pub churn_rounds: usize,
    /// Raw signed little-endian PCM16, mono, 16 kHz; silence is used when omitted.
    #[arg(long)]
    pub audio_file: Option<PathBuf>,
    #[arg(long, default_value_t = 7)]
    pub seed: u64,
    #[arg(long)]
    pub report: Option<PathBuf>,
}

impl BenchmarkArguments {
    pub fn workload(&self) -> SimulationConfig {
        SimulationConfig {
            sessions: self.sessions,
            turns_per_session: self.turns,
            utterance_ms: self.utterance_ms,
            minimum_packet_ms: self.minimum_packet_ms,
            maximum_packet_ms: self.maximum_packet_ms,
            start_spread_ms: self.start_spread_ms,
            think_ms: self.think_ms,
            endpointing_ms: self.endpointing_ms,
            prepare_before_commit: self.prepare_before_commit,
            target_tokens_per_second: self.target_tokens_per_second,
            throughput_window_ms: self.throughput_window_ms,
            response_timeout: Duration::from_secs(self.timeout_secs),
            interrupt_after_tokens: self.interrupt_after_tokens,
            churn_rounds: self.churn_rounds,
            audio_file: self.audio_file.clone(),
            seed: self.seed,
        }
    }
}

#[derive(Args)]
pub struct SuiteArguments {
    #[command(flatten)]
    pub benchmark: BenchmarkArguments,
    /// Offered concurrency used as the reference capacity for the load sweep.
    #[arg(long)]
    pub session_budget: usize,
}
