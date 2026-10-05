use std::time::Duration;

use tokio_util::sync::CancellationToken;
use voice_scheduler::{
    config::RuntimeConfig,
    simulation::{ArrivalPhase, ChurnConfig, SimulationConfig, benchmark_suite, run_benchmark},
};

fn runtime_configuration() -> RuntimeConfig {
    RuntimeConfig {
        workers: 2,
        max_sessions_per_worker: 8,
        cache_slots_per_worker: 8,
        batch_size: 4,
        ..RuntimeConfig::default()
    }
}

fn simulation_configuration() -> SimulationConfig {
    SimulationConfig {
        sessions: 12,
        duration: Duration::from_millis(500),
        ..SimulationConfig::default()
    }
}

#[tokio::test(start_paused = true)]
async fn overload_rejects_admission_and_keeps_event_heap_bounded() {
    let result = run_benchmark(
        runtime_configuration(),
        SimulationConfig {
            sessions: 20,
            ..simulation_configuration()
        },
        CancellationToken::new(),
    )
    .await
    .unwrap();
    assert_eq!(result.workload.admitted, 16);
    assert_eq!(result.workload.rejected, 4);
    assert_eq!(result.runtime.peak_active_sessions, 16);
    assert_eq!(
        result.workload.received_outputs,
        result.runtime.delivered_results
    );
    assert!(result.workload.peak_scheduled_events <= 20);
    assert_eq!(result.workload.overloaded_frames, 0);
}

#[tokio::test(start_paused = true)]
async fn fast_churn_reuses_capacity_without_accumulating_stale_events() {
    let configuration = SimulationConfig {
        churn: Some(ChurnConfig {
            min_duration: Duration::from_millis(3),
            max_duration: Duration::from_millis(10),
        }),
        ..simulation_configuration()
    };
    let result = run_benchmark(
        runtime_configuration(),
        configuration,
        CancellationToken::new(),
    )
    .await
    .unwrap();
    assert!(result.workload.admitted > 100);
    assert!(result.workload.closed > 100);
    assert_eq!(result.workload.rejected, 0);
    assert!(result.workload.peak_scheduled_events <= 24);
    assert!(result.runtime.stale_results_discarded > 0);
    assert!(result.runtime.peak_active_sessions <= 12);
}

#[tokio::test(start_paused = true)]
async fn random_phase_jitter_and_quantization_record_latency_distributions() {
    let result = run_benchmark(
        RuntimeConfig {
            phase_bucket: Some(Duration::from_millis(10)),
            ..runtime_configuration()
        },
        SimulationConfig {
            arrival_phase: ArrivalPhase::Random,
            jitter: Duration::from_millis(5),
            ..simulation_configuration()
        },
        CancellationToken::new(),
    )
    .await
    .unwrap();
    assert!(result.runtime.phase_added_latency.mean_ms > 0.0);
    assert!(result.runtime.phase_added_latency.max_ms <= 10.1);
    assert!(result.runtime.queue_delay.samples > 0);
    assert!(result.runtime.end_to_end_latency.p99_ms >= result.runtime.end_to_end_latency.p50_ms);
    assert!(result.workload.generator_lag.samples > 0);
    assert_eq!(result.workload.unknown_session_frames, 0);
}

#[tokio::test(start_paused = true)]
async fn delayed_input_consumer_saturates_bounded_mailboxes() {
    let result = run_benchmark(
        RuntimeConfig {
            worker_channel_capacity: 1,
            worker_input_delay: Duration::from_millis(20),
            ..runtime_configuration()
        },
        simulation_configuration(),
        CancellationToken::new(),
    )
    .await
    .unwrap();
    assert_eq!(result.workload.admitted, 12);
    assert!(result.workload.overloaded_frames > 0);
    assert!(result.runtime.worker_channel_saturation > 0);
    assert_eq!(
        result.runtime.inputs_overloaded,
        result.workload.overloaded_frames
    );
}

#[tokio::test(start_paused = true)]
async fn cancellation_returns_a_partial_report_and_drains_device_tasks() {
    let cancellation = CancellationToken::new();
    let signal = cancellation.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(100)).await;
        signal.cancel();
    });
    let result = run_benchmark(
        runtime_configuration(),
        simulation_configuration(),
        cancellation,
    )
    .await
    .unwrap();
    assert!(result.workload.interrupted);
    assert!(result.runtime.elapsed_secs < 0.2);
}

#[test]
fn suite_covers_required_experiments() {
    let scenarios = benchmark_suite(&RuntimeConfig::default(), Duration::from_secs(2), 42);
    assert_eq!(scenarios.len(), 14);
    for name in [
        "low-load",
        "50-percent",
        "80-percent",
        "95-percent",
        "overload",
        "random-phase",
        "session-churn",
        "channel-saturation",
        "worker-slowdown",
        "phase-bucket-5ms",
        "phase-bucket-10ms",
    ] {
        assert!(scenarios.iter().any(|scenario| scenario.name == name));
    }
}

#[test]
fn invalid_workload_parameters_fail_clearly() {
    let tick = Duration::from_millis(50);
    assert!(
        SimulationConfig {
            jitter: Duration::from_millis(25),
            ..simulation_configuration()
        }
        .validate(tick)
        .is_err()
    );
    assert!(
        SimulationConfig {
            duration: Duration::ZERO,
            ..simulation_configuration()
        }
        .validate(tick)
        .is_err()
    );
    assert!(
        SimulationConfig {
            churn: Some(ChurnConfig {
                min_duration: Duration::from_millis(20),
                max_duration: Duration::from_millis(10)
            }),
            ..simulation_configuration()
        }
        .validate(tick)
        .is_err()
    );
}
