use std::{net::SocketAddr, time::Duration};

use bytes::Bytes;
use tokio::{io::AsyncWriteExt, net::TcpStream, task::JoinHandle, time::Instant};
use tokio_util::sync::CancellationToken;
use voice_scheduler::{
    config::RuntimeConfig,
    protocol::{CacheOutcome, CreateRejection, PacketSequence, SessionId},
    simulation::{self, SimulationConfig},
    transport::{
        AudioSession, ClientError, ConnectOutcome, Gateway, GatewayConfig, GatewayError,
        GatewayReport, connect_session,
    },
};

fn configuration() -> RuntimeConfig {
    RuntimeConfig {
        workers: 2,
        batch_size: 4,
        inference_latency: Duration::from_millis(3),
        packet_deadline: Duration::from_millis(250),
        minimum_packet_interval: Duration::from_millis(250),
        max_sessions_per_worker: 4,
        cache_slots_per_worker: 4,
        max_batch_wait: Duration::ZERO,
        calibration_samples: 2,
        latency_window: 8,
        ..RuntimeConfig::default()
    }
}
async fn start(
    runtime: RuntimeConfig,
) -> (
    SocketAddr,
    CancellationToken,
    JoinHandle<Result<GatewayReport, GatewayError>>,
) {
    let gateway = Gateway::bind(
        runtime,
        GatewayConfig {
            listen_address: "127.0.0.1:0".parse().unwrap(),
            ..GatewayConfig::default()
        },
    )
    .await
    .unwrap();
    let address = gateway.local_address().unwrap();
    let cancellation = CancellationToken::new();
    let task = tokio::spawn(gateway.serve(cancellation.clone()));
    (address, cancellation, task)
}
async fn connect(address: SocketAddr, id: u64) -> Box<AudioSession> {
    match connect_session(address, SessionId(id), Duration::from_secs(2))
        .await
        .unwrap()
    {
        ConnectOutcome::Admitted(session) => session,
        ConnectOutcome::Rejected(reason) => panic!("unexpected rejection {reason:?}"),
    }
}

#[tokio::test]
async fn tcp_echo_is_sticky_and_replays_complete_prefix_on_cache_miss() {
    let (address, signal, task) = start(configuration()).await;
    let mut session = connect(address, 1).await;
    let assignment = session.assignment();
    let payload = Bytes::from(vec![21; 1600]);
    for sequence in 0..3 {
        assert_eq!(session.next_sequence(), PacketSequence(sequence));
        let captured_at = Instant::now();
        let audio = session
            .infer_audio(payload.clone(), captured_at)
            .await
            .unwrap();
        assert!(audio.timings.total() <= captured_at.elapsed());
        assert!(audio.timings.device_execution >= Duration::from_millis(3));
        assert_eq!(audio.assignment, assignment);
        assert_eq!(audio.sequence.0, sequence);
        assert_eq!(audio.payload, payload);
        assert_eq!(audio.cache, CacheOutcome::Hit);
    }
    assert!(session.evict_cache().await.unwrap());
    let audio = session.infer_audio(payload, Instant::now()).await.unwrap();
    assert_eq!(audio.prefix.packets, 4);
    assert_eq!(
        audio.cache,
        CacheOutcome::Replayed {
            packets: 3,
            bytes: 4800
        }
    );
    assert!(session.close().await.unwrap());
    signal.cancel();
    let report = task.await.unwrap().unwrap();
    assert_eq!(report.runtime.inference.delivered_frames, 4);
    assert_eq!(report.runtime.inference.replayed_packets, 3);
    assert_eq!(report.runtime.inference.cache_misses, 1);
    assert_eq!(report.runtime.profile.scheduler_queue.samples, 4);
    assert_eq!(report.runtime.active_sessions_at_shutdown, 0);
}

#[tokio::test]
async fn late_audio_recovers_on_the_same_session_and_retains_its_entire_prefix() {
    let runtime = RuntimeConfig {
        packet_lateness_grace: Duration::from_millis(100),
        ..configuration()
    };
    let (address, signal, task) = start(runtime).await;
    let mut session = connect(address, 1).await;
    let assignment = session.assignment();
    let payload = Bytes::from_static(b"voice");
    for sequence in 0..3 {
        let error = session
            .infer_audio(payload.clone(), Instant::now() - Duration::from_millis(260))
            .await
            .unwrap_err();
        let ClientError::LateEcho(audio) = error else {
            panic!("expected recoverable late echo, got {error:?}");
        };
        assert_eq!(audio.assignment, assignment);
        assert_eq!(audio.sequence.0, sequence);
        assert_eq!(audio.cache, CacheOutcome::Hit);
        assert_eq!(session.next_sequence().0, sequence + 1);
    }
    assert!(session.evict_cache().await.unwrap());
    let audio = session.infer_audio(payload, Instant::now()).await.unwrap();
    assert_eq!(audio.assignment, assignment);
    assert_eq!(audio.prefix.packets, 4);
    assert_eq!(
        audio.cache,
        CacheOutcome::Replayed {
            packets: 3,
            bytes: 15
        }
    );
    session.close().await.unwrap();
    signal.cancel();
    let report = task.await.unwrap().unwrap();
    assert_eq!(report.runtime.inference.deadline_misses, 3);
    assert_eq!(report.runtime.inference.delivered_frames, 4);
    assert_eq!(report.runtime.active_sessions_at_shutdown, 0);
}

#[tokio::test]
async fn discarded_output_keeps_the_input_cache_and_complete_replay_history() {
    let (address, signal, task) = start(configuration()).await;
    let mut session = connect(address, 1).await;
    let assignment = session.assignment();
    let error = session
        .infer_audio(
            Bytes::from_static(b"voice"),
            Instant::now() - Duration::from_millis(300),
        )
        .await
        .unwrap_err();
    let ClientError::ExpiredEcho(audio) = error else {
        panic!("expected discarded output, got {error:?}");
    };
    assert_eq!(audio.assignment, assignment);
    assert_eq!(session.next_sequence().0, 1);
    let audio = session
        .infer_audio(Bytes::from_static(b"next"), Instant::now())
        .await
        .unwrap();
    assert_eq!(audio.assignment, assignment);
    assert_eq!(audio.cache, CacheOutcome::Hit);
    assert_eq!(audio.prefix.packets, 2);
    assert!(session.evict_cache().await.unwrap());
    let audio = session
        .infer_audio(Bytes::from_static(b"last"), Instant::now())
        .await
        .unwrap();
    assert_eq!(audio.prefix.packets, 3);
    assert_eq!(
        audio.cache,
        CacheOutcome::Replayed {
            packets: 2,
            bytes: 9
        }
    );
    session.close().await.unwrap();
    signal.cancel();
    let report = task.await.unwrap().unwrap();
    assert_eq!(report.runtime.inference.deadline_misses, 1);
    assert_eq!(report.runtime.inference.delivered_frames, 3);
    assert_eq!(report.runtime.active_sessions_at_shutdown, 0);
}

#[tokio::test]
async fn gateway_rejects_capacity_and_reclaims_disconnected_sessions() {
    let (address, signal, task) = start(configuration()).await;
    let mut sessions = Vec::new();
    for id in 0..8 {
        sessions.push(connect(address, id).await);
    }
    assert!(matches!(
        connect_session(address, SessionId(9), Duration::from_secs(2))
            .await
            .unwrap(),
        ConnectOutcome::Rejected(CreateRejection::Capacity)
    ));
    assert!(matches!(
        connect_session(address, SessionId(0), Duration::from_secs(2))
            .await
            .unwrap(),
        ConnectOutcome::Rejected(CreateRejection::AlreadyExists)
    ));
    let old = sessions.pop().unwrap();
    let assignment = old.assignment();
    old.close().await.unwrap();
    let replacement = connect(address, 7).await;
    assert!(replacement.assignment().generation.0 > assignment.generation.0);
    assert_eq!(replacement.assignment().worker_id, assignment.worker_id);
    replacement.close().await.unwrap();
    drop(sessions);
    signal.cancel();
    let report = task.await.unwrap().unwrap();
    assert_eq!(report.runtime.admitted_sessions, 9);
    assert_eq!(report.runtime.rejected_sessions, 1);
    assert_eq!(report.runtime.active_sessions_at_shutdown, 0);
}

#[tokio::test]
async fn disconnect_during_inference_discards_running_result() {
    let (address, signal, task) = start(RuntimeConfig {
        inference_latency: Duration::from_millis(80),
        ..configuration()
    })
    .await;
    let mut session = connect(address, 1).await;
    let inference = tokio::spawn(async move {
        session
            .infer_audio(Bytes::from_static(b"audio"), Instant::now())
            .await
    });
    tokio::time::sleep(Duration::from_millis(25)).await;
    inference.abort();
    let _ = inference.await;
    tokio::time::sleep(Duration::from_millis(100)).await;
    signal.cancel();
    let report = task.await.unwrap().unwrap();
    assert_eq!(report.runtime.inference.stale_results, 1);
    assert_eq!(report.runtime.inference.delivered_frames, 0);
    assert_eq!(report.runtime.active_sessions_at_shutdown, 0);
}

#[tokio::test]
async fn oversized_network_frame_is_rejected_before_allocation() {
    let (address, signal, task) = start(configuration()).await;
    let mut connection = TcpStream::connect(address).await.unwrap();
    connection.write_all(&u32::MAX.to_be_bytes()).await.unwrap();
    tokio::time::sleep(Duration::from_millis(25)).await;
    drop(connection);
    signal.cancel();
    let report = task.await.unwrap().unwrap();
    assert_eq!(report.connections.failed_connections, 1);
    assert_eq!(report.runtime.admitted_sessions, 0);
}

#[tokio::test]
async fn network_simulation_reports_admission_echo_replay_and_churn() {
    let (address, signal, task) = start(configuration()).await;
    let report = simulation::run(
        address,
        SimulationConfig {
            sessions: 12,
            duration: Duration::from_millis(600),
            evict_every: Some(2),
            churn_after: Some(Duration::from_millis(200)),
            ..SimulationConfig::default()
        },
    )
    .await
    .unwrap();
    assert_eq!(report.counters.rejected_capacity, 4);
    assert!(report.counters.admitted_sessions > 8);
    assert!(report.counters.echoed_frames > 20);
    assert!(report.counters.replayed_packets > 0);
    assert_eq!(report.counters.failed_sessions, 0);
    assert_eq!(report.generator_overruns, 0);
    assert_eq!(report.counters.metric_samples_dropped, 0);
    assert!(report.generated_intervals.min_ms >= 47.9);
    assert_eq!(
        report.round_trip_latency.samples,
        report.counters.echoed_frames
    );
    signal.cancel();
    let gateway = task.await.unwrap().unwrap();
    assert_eq!(
        gateway.runtime.inference.delivered_frames,
        report.counters.echoed_frames
    );
}

#[tokio::test]
async fn bounded_client_metrics_preserve_audio_and_account_for_all_samples() {
    let (address, signal, task) = start(configuration()).await;
    let report = simulation::run(
        address,
        SimulationConfig {
            sessions: 8,
            duration: Duration::from_millis(300),
            phase: voice_scheduler::simulation::ArrivalPhase::Aligned,
            metric_channel_capacity: 1,
            ..SimulationConfig::default()
        },
    )
    .await
    .unwrap();
    assert_eq!(report.counters.failed_sessions, 0);
    assert!(report.counters.echoed_frames > 8);
    assert_eq!(
        report.round_trip_latency.samples
            + report.failed_round_trip_latency.samples
            + report.counters.metric_samples_dropped,
        report.counters.attempted_frames
    );
    assert!(report.slowest_packets.len() <= 8);
    signal.cancel();
    task.await.unwrap().unwrap();
}
