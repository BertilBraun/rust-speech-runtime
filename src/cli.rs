use clap::{Args, Parser, Subcommand};
use std::{net::SocketAddr, path::PathBuf, time::Duration};
use voice_scheduler::{
    config::{DeviceWait, RuntimeConfig, WorkerSlowdown},
    simulation::{ArrivalPhase, SimulationConfig},
};

#[derive(Parser)]
#[command(about = "Realtime audio echo gateway and external TCP benchmark")]
pub(super) struct Cli {
    #[arg(long, global = true)]
    pub(super) output: Option<PathBuf>,
    #[command(subcommand)]
    pub(super) command: Command,
}
#[derive(Subcommand)]
pub(super) enum Command {
    Serve {
        #[arg(long, default_value = "127.0.0.1:9000")]
        listen: SocketAddr,
        #[command(flatten)]
        runtime: RuntimeArguments,
    },
    Simulate {
        #[arg(long, default_value = "127.0.0.1:9000")]
        address: SocketAddr,
        #[command(flatten)]
        workload: WorkloadArguments,
    },
    #[command(name = "simulate-stdin", hide = true)]
    SimulateFromStdin {
        #[arg(long)]
        address: SocketAddr,
    },
    Benchmark(BenchmarkArguments),
    Suite(BenchmarkArguments),
}
#[derive(Clone, Default, Args)]
pub(super) struct RuntimeArguments {
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
    scheduling_margin_ms: Option<u64>,
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
    pub(super) fn configuration(&self) -> RuntimeConfig {
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
        if let Some(value) = self.scheduling_margin_ms {
            configuration.scheduling_margin = Duration::from_millis(value);
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
pub(super) struct WorkloadArguments {
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
    pub(super) fn configuration(&self) -> SimulationConfig {
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
pub(super) struct BenchmarkArguments {
    #[command(flatten)]
    pub(super) runtime: RuntimeArguments,
    #[command(flatten)]
    pub(super) workload: WorkloadArguments,
}
