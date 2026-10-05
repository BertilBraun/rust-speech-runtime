use std::time::Duration;

use crate::config::{RuntimeConfig, WorkerSlowdown};

use super::{ArrivalPhase, ChurnConfig, SimulationConfig};

pub struct Scenario {
    pub name: &'static str,
    pub runtime: RuntimeConfig,
    pub simulation: SimulationConfig,
}

pub fn benchmark_suite(runtime: &RuntimeConfig, duration: Duration, seed: u64) -> Vec<Scenario> {
    let capacity = runtime.workers * runtime.admission_limit();
    let baseline = SimulationConfig {
        sessions: (capacity * 80 / 100).max(1),
        duration,
        seed,
        ..SimulationConfig::default()
    };
    let mut scenarios = Vec::new();
    for (name, percent) in [
        ("low-load", 10),
        ("50-percent", 50),
        ("80-percent", 80),
        ("95-percent", 95),
        ("overload", 120),
    ] {
        scenarios.push(Scenario {
            name,
            runtime: runtime.clone(),
            simulation: SimulationConfig {
                sessions: (capacity * percent / 100).max(1),
                ..baseline.clone()
            },
        });
    }
    scenarios.push(Scenario {
        name: "random-phase",
        runtime: runtime.clone(),
        simulation: SimulationConfig {
            arrival_phase: ArrivalPhase::Random,
            ..baseline.clone()
        },
    });
    for (name, jitter_ms) in [("jitter-2ms", 2), ("jitter-5ms", 5), ("jitter-10ms", 10)] {
        let jitter = Duration::from_millis(jitter_ms);
        if jitter < runtime.tick_interval / 2 {
            scenarios.push(Scenario {
                name,
                runtime: runtime.clone(),
                simulation: SimulationConfig {
                    jitter,
                    ..baseline.clone()
                },
            });
        }
    }
    for (name, bucket_ms) in [("phase-bucket-5ms", 5), ("phase-bucket-10ms", 10)] {
        let bucket = Duration::from_millis(bucket_ms);
        if bucket <= runtime.tick_interval {
            scenarios.push(Scenario {
                name,
                runtime: RuntimeConfig {
                    phase_bucket: Some(bucket),
                    ..runtime.clone()
                },
                simulation: SimulationConfig {
                    arrival_phase: ArrivalPhase::Random,
                    ..baseline.clone()
                },
            });
        }
    }
    scenarios.push(Scenario {
        name: "session-churn",
        runtime: runtime.clone(),
        simulation: SimulationConfig {
            churn: Some(ChurnConfig {
                min_duration: duration / 10,
                max_duration: duration / 2,
            }),
            ..baseline.clone()
        },
    });
    scenarios.push(Scenario {
        name: "channel-saturation",
        runtime: RuntimeConfig {
            worker_channel_capacity: 1,
            worker_input_delay: Duration::from_millis(2),
            ..runtime.clone()
        },
        simulation: baseline.clone(),
    });
    scenarios.push(Scenario {
        name: "worker-slowdown",
        runtime: RuntimeConfig {
            slowdown: Some(WorkerSlowdown {
                after: duration / 3,
                inference_latency: runtime.inference_latency.mul_f64(20.0 / 12.0),
            }),
            ..runtime.clone()
        },
        simulation: baseline,
    });
    scenarios
}
