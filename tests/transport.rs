mod support;

use std::time::Duration;

use bytes::Bytes;
use futures_util::{SinkExt, StreamExt};
use tokio::{net::TcpStream, task::JoinHandle, time::timeout};
use tokio_tungstenite::{connect_async, tungstenite::Message};
use tokio_util::sync::CancellationToken;
use voice_scheduler::{
    config::RuntimeConfig,
    protocol::{ErrorCode, SessionEvent, SessionId, TurnId},
    simulation::{self, SimulationConfig},
    transport::{Gateway, GatewayConfig, GatewayReport, VoiceClient},
};

use support::Fixture;

struct TestGateway {
    url: String,
    cancellation: CancellationToken,
    task: JoinHandle<GatewayReport>,
}

impl TestGateway {
    async fn start(runtime: RuntimeConfig, configuration: GatewayConfig) -> Self {
        let gateway = Gateway::bind(runtime, configuration)
            .await
            .expect("gateway starts");
        let url = format!(
            "ws://{}/v1",
            gateway.local_address().expect("listener address")
        );
        let cancellation = CancellationToken::new();
        let signal = cancellation.clone();
        let task =
            tokio::spawn(async move { gateway.serve(signal).await.expect("gateway shutdown") });
        Self {
            url,
            cancellation,
            task,
        }
    }

    async fn shutdown(self) -> GatewayReport {
        self.cancellation.cancel();
        timeout(Duration::from_secs(10), self.task)
            .await
            .expect("bounded shutdown")
            .expect("gateway task")
    }
}

fn gateway_config() -> GatewayConfig {
    GatewayConfig {
        listen_address: "127.0.0.1:0".parse().expect("literal"),
        archive_directory: None,
        ..GatewayConfig::default()
    }
}

async fn connected(url: &str, id: &str) -> VoiceClient {
    let mut client = VoiceClient::connect(url, Duration::from_secs(5))
        .await
        .expect("WebSocket connected");
    client
        .open(SessionId(id.into()))
        .await
        .expect("session opened");
    client
}

async fn commit(client: &mut VoiceClient, turn_id: u64) {
    client
        .begin_turn(TurnId(turn_id))
        .await
        .expect("turn accepted");
    client
        .audio(TurnId(turn_id), 0, Bytes::from(vec![1; 1600]))
        .await
        .expect("audio delivered");
    client
        .commit(TurnId(turn_id), 1, 800)
        .await
        .expect("turn committed");
}

async fn finish(client: &mut VoiceClient, turn_id: u64) -> Vec<u32> {
    let mut tokens = Vec::new();
    loop {
        match client.next_event().await.expect("session event") {
            SessionEvent::TextDelta {
                turn_id: observed,
                token_id,
                ..
            } if observed == TurnId(turn_id) => tokens.push(token_id),
            SessionEvent::Finished {
                turn_id: observed, ..
            } if observed == TurnId(turn_id) => return tokens,
            SessionEvent::Failed { message, .. } => panic!("{message}"),
            _ => {}
        }
    }
}

#[tokio::test]
async fn websocket_multi_turn_preserves_assignment_and_exact_archive() {
    let backend = Fixture::start(2).await;
    let directory = std::env::temp_dir().join(format!(
        "voice-archive-test-{}-{}",
        std::process::id(),
        rand::random::<u64>()
    ));
    let gateway = TestGateway::start(
        backend.config(),
        GatewayConfig {
            archive_directory: Some(directory.clone()),
            ..gateway_config()
        },
    )
    .await;
    let mut client = connected(&gateway.url, "conversation").await;
    commit(&mut client, 1).await;
    let first = finish(&mut client, 1).await;
    commit(&mut client, 2).await;
    let second = finish(&mut client, 2).await;
    assert_eq!(first.len(), 5);
    assert_eq!(second.len(), 5);
    client.close().await.expect("client closes cleanly");
    let report = gateway.shutdown().await;
    assert_eq!(report.archives.saved, 1);
    assert_eq!(report.runtime.generated_tokens, 10);
    let mut files = tokio::fs::read_dir(&directory)
        .await
        .expect("archive directory");
    let archive = files
        .next_entry()
        .await
        .expect("read entry")
        .expect("archive file");
    let contents = tokio::fs::read(archive.path())
        .await
        .expect("archive bytes");
    #[derive(serde::Deserialize)]
    struct Archive {
        conversation: voice_scheduler::protocol::SessionRecord,
    }
    let record: Archive = serde_json::from_slice(&contents).expect("canonical archive record");
    assert_eq!(record.conversation.turns[0].audio_pcm16, vec![1; 1600]);
    assert_eq!(
        record.conversation.turns[1]
            .tokens
            .iter()
            .map(|token| token.token_id)
            .collect::<Vec<_>>(),
        second
    );
    assert!(files.next_entry().await.expect("read entry").is_none());
    std::fs::remove_file(archive.path()).expect("remove test archive");
    std::fs::remove_dir(directory).expect("remove empty test directory");
}

#[tokio::test]
async fn application_close_waits_only_for_bounded_peer_acknowledgement() {
    let backend = Fixture::start(1).await;
    let gateway = TestGateway::start(
        backend.config(),
        GatewayConfig {
            write_timeout: Duration::from_millis(100),
            ..gateway_config()
        },
    )
    .await;
    let (mut websocket, _) = connect_async(&gateway.url).await.expect("connected");
    websocket
        .send(Message::Text(
            serde_json::to_string(&voice_scheduler::transport::wire::ClientControl::Open {
                session_id: SessionId("unacknowledged-close".into()),
            })
            .expect("open JSON")
            .into(),
        ))
        .await
        .expect("open sent");
    let opened = websocket
        .next()
        .await
        .expect("opened frame")
        .expect("opened message");
    assert!(matches!(opened, Message::Text(_)));
    websocket
        .send(Message::Text(
            serde_json::to_string(&voice_scheduler::transport::wire::ClientControl::Close)
                .expect("close JSON")
                .into(),
        ))
        .await
        .expect("close sent");
    let closed = websocket
        .next()
        .await
        .expect("closed frame")
        .expect("closed message");
    assert!(matches!(closed, Message::Text(_)));
    tokio::time::sleep(Duration::from_millis(250)).await;
    let report = gateway.shutdown().await;
    assert_eq!(report.failed_connections, 1);
    assert_eq!(report.runtime.active_sessions, 0);
}

#[tokio::test]
async fn peer_initiated_websocket_close_finishes_without_second_handshake() {
    let backend = Fixture::start(1).await;
    let gateway = TestGateway::start(backend.config(), gateway_config()).await;
    let (mut websocket, _) = connect_async(&gateway.url).await.expect("connected");
    websocket.close(None).await.expect("close sent");
    timeout(Duration::from_secs(1), websocket.next())
        .await
        .expect("peer acknowledgement bounded");
    let report = gateway.shutdown().await;
    assert_eq!(report.failed_connections, 0);
}

#[tokio::test]
async fn rejected_open_completes_websocket_close_without_connection_failure() {
    let backend = Fixture::start(1).await;
    let gateway = TestGateway::start(
        RuntimeConfig {
            max_sessions_per_worker: 1,
            max_active_turns_per_worker: 1,
            ..backend.config()
        },
        gateway_config(),
    )
    .await;
    let first = connected(&gateway.url, "admitted").await;
    let mut rejected = VoiceClient::connect(&gateway.url, Duration::from_secs(5))
        .await
        .expect("connected");
    assert!(matches!(
        rejected.open(SessionId("rejected".into())).await,
        Err(voice_scheduler::transport::GatewayError::Rejected(
            ErrorCode::CapacityExceeded,
            _
        ))
    ));
    first.close().await.expect("admitted client closed");
    let report = gateway.shutdown().await;
    assert_eq!(report.runtime.rejected_sessions, 1);
    assert_eq!(report.failed_connections, 0);
}

#[tokio::test]
async fn websocket_rejects_capacity_and_releases_disconnected_session() {
    let backend = Fixture::start(2).await;
    let runtime = RuntimeConfig {
        max_sessions_per_worker: 1,
        max_active_turns_per_worker: 1,
        ..backend.config()
    };
    let gateway = TestGateway::start(runtime, gateway_config()).await;
    let first = connected(&gateway.url, "reused-id").await;
    let mut rejected = VoiceClient::connect(&gateway.url, Duration::from_secs(5))
        .await
        .expect("connect second");
    assert!(matches!(
        rejected.open(SessionId("other".into())).await,
        Err(voice_scheduler::transport::GatewayError::Rejected(
            ErrorCode::CapacityExceeded,
            _
        ))
    ));
    drop(first);
    tokio::time::sleep(Duration::from_millis(150)).await;
    let mut replacement = connected(&gateway.url, "reused-id").await;
    commit(&mut replacement, 1).await;
    assert_eq!(finish(&mut replacement, 1).await.len(), 5);
    replacement.close().await.expect("replacement closed");
    drop(rejected);
    assert_eq!(gateway.shutdown().await.runtime.active_sessions, 0);
}

#[tokio::test]
async fn public_packet_order_and_commit_counts_are_validated() {
    let backend = Fixture::start(1).await;
    let gateway = TestGateway::start(backend.config(), gateway_config()).await;
    let mut client = connected(&gateway.url, "invalid-chunk").await;
    client.begin_turn(TurnId(1)).await.expect("turn accepted");
    client
        .audio(TurnId(1), 1, Bytes::from_static(&[0, 0]))
        .await
        .expect("send invalid sequence");
    assert!(matches!(
        client.next_event().await.expect("failed event"),
        SessionEvent::Failed {
            code: ErrorCode::InvalidInput,
            ..
        }
    ));
    client
        .audio(TurnId(1), 0, Bytes::from_static(&[0, 0]))
        .await
        .expect("send first chunk");
    client
        .commit(TurnId(1), 2, 1)
        .await
        .expect("invalid commit sent");
    assert!(matches!(
        client.next_event().await.expect("failed event"),
        SessionEvent::Failed {
            code: ErrorCode::InvalidInput,
            ..
        }
    ));
    client
        .commit(TurnId(1), 1, 1)
        .await
        .expect("valid commit sent");
    client
        .commit(TurnId(1), 1, 1)
        .await
        .expect("duplicate commit sent");
    assert_eq!(finish(&mut client, 1).await.len(), 5);
    client.close().await.expect("client closed");
    assert_eq!(gateway.shutdown().await.runtime.admitted_turns, 1);
}

#[tokio::test]
async fn new_audio_turn_interrupts_generation_over_real_websocket() {
    let backend = Fixture::start(40).await;
    let gateway = TestGateway::start(backend.config(), gateway_config()).await;
    let mut client = connected(&gateway.url, "interrupt").await;
    commit(&mut client, 1).await;
    loop {
        if matches!(
            client.next_event().await.expect("token"),
            SessionEvent::TextDelta {
                turn_id: TurnId(1),
                ..
            }
        ) {
            break;
        }
    }
    commit(&mut client, 2).await;
    assert_eq!(finish(&mut client, 2).await.len(), 5);
    client.close().await.expect("client closed");
    assert!(gateway.shutdown().await.runtime.stale_results_discarded > 0);
}

#[tokio::test]
async fn network_benchmark_exercises_multi_turn_churn_and_latency_distributions() {
    let backend = Fixture::start(2).await;
    let gateway = TestGateway::start(backend.config(), gateway_config()).await;
    let configuration = SimulationConfig {
        sessions: 4,
        turns_per_session: 2,
        utterance_ms: 50,
        start_spread_ms: 10,
        think_ms: 0,
        churn_rounds: 2,
        ..SimulationConfig::default()
    };
    let report = simulation::run(&gateway.url, configuration)
        .await
        .expect("benchmark completed");
    assert_eq!(report.failed_sessions, 0, "{:?}", report.sessions);
    assert_eq!(report.admitted_sessions, 8);
    assert_eq!(report.completed_turns, 16);
    assert_eq!(report.received_tokens, 80);
    assert_eq!(report.time_to_first_token_ms.count, 16);
    assert_eq!(report.token_gap_ms.count, 64);
    gateway.shutdown().await;
}

#[tokio::test]
async fn bounded_handshakes_and_connection_limit_do_not_block_healthy_clients() {
    let backend = Fixture::start(1).await;
    let gateway = TestGateway::start(
        backend.config(),
        GatewayConfig {
            max_connections: 1,
            handshake_timeout: Duration::from_millis(80),
            ..gateway_config()
        },
    )
    .await;
    let address = gateway
        .url
        .trim_start_matches("ws://")
        .trim_end_matches("/v1");
    let half_open = TcpStream::connect(address).await.expect("TCP connects");
    tokio::time::sleep(Duration::from_millis(20)).await;
    assert!(
        VoiceClient::connect(&gateway.url, Duration::from_millis(200))
            .await
            .is_err()
    );
    tokio::time::sleep(Duration::from_millis(100)).await;
    let client = connected(&gateway.url, "healthy").await;
    client.close().await.expect("healthy closed");
    drop(half_open);
    assert_eq!(gateway.shutdown().await.rejected_connections, 1);
}

#[tokio::test]
async fn websocket_message_limit_closes_oversized_sender() {
    let backend = Fixture::start(1).await;
    let gateway = TestGateway::start(
        backend.config(),
        GatewayConfig {
            max_message_bytes: 128,
            ..gateway_config()
        },
    )
    .await;
    let (mut websocket, _) = connect_async(&gateway.url).await.expect("handshake");
    websocket
        .send(Message::Binary(Bytes::from(vec![0; 129])))
        .await
        .expect("oversized packet sent");
    let received = timeout(Duration::from_secs(2), websocket.next())
        .await
        .expect("bounded rejection");
    assert!(!matches!(received, Some(Ok(Message::Text(_)))));
    assert_eq!(gateway.shutdown().await.failed_connections, 1);
}

#[tokio::test]
#[ignore = "requires uv and installed backend Python environment; run explicitly before hardware deployment"]
async fn python_worker_process_to_websocket_client_full_pipeline() {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("reserve fixture port");
    let address = listener.local_addr().expect("fixture address");
    drop(listener);
    let mut worker = tokio::process::Command::new("uv")
        .current_dir(concat!(env!("CARGO_MANIFEST_DIR"), "/backend"))
        .args([
            "run",
            "python",
            "tests/fixture_worker.py",
            "--port",
            &address.port().to_string(),
            "--response-tokens",
            "12",
        ])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::inherit())
        .kill_on_drop(true)
        .spawn()
        .expect("Python fixture process starts");
    use tokio::io::{AsyncBufReadExt, BufReader};
    let mut lines = BufReader::new(worker.stdout.take().expect("piped stdout")).lines();
    let line = timeout(Duration::from_secs(60), lines.next_line())
        .await
        .expect("fixture startup timeout")
        .expect("fixture output")
        .expect("ready line");
    assert!(line.contains("ready"), "{line}");
    let runtime = RuntimeConfig {
        workers: vec![voice_scheduler::config::WorkerConfig { endpoint: address }],
        initial_forward_estimate_ms: 20.0,
        ..RuntimeConfig::default()
    };
    let gateway = TestGateway::start(runtime, gateway_config()).await;
    let report = simulation::run(
        &gateway.url,
        SimulationConfig {
            sessions: 4,
            turns_per_session: 2,
            utterance_ms: 50,
            start_spread_ms: 10,
            think_ms: 0,
            ..SimulationConfig::default()
        },
    )
    .await
    .expect("cross-language benchmark");
    assert_eq!(report.failed_sessions, 0, "{:?}", report.sessions);
    assert_eq!(report.received_tokens, 104);
    assert_eq!(gateway.shutdown().await.runtime.backend_failures, 0);
    worker.kill().await.expect("fixture stops");
    worker.wait().await.expect("fixture reaped");
}

#[tokio::test]
async fn output_progress_keeps_generation_alive_beyond_idle_timeout() {
    let backend = Fixture::start(150).await;
    let gateway = TestGateway::start(
        backend.config(),
        GatewayConfig {
            idle_timeout: Duration::from_millis(500),
            ..gateway_config()
        },
    )
    .await;
    let mut client = connected(&gateway.url, "active-generator").await;
    commit(&mut client, 1).await;
    assert_eq!(finish(&mut client, 1).await.len(), 5);
    tokio::time::sleep(Duration::from_millis(750)).await;
    assert!(matches!(
        client.next_event().await.expect("idle cleanup event"),
        SessionEvent::Closed { .. }
    ));
    assert_eq!(gateway.shutdown().await.runtime.active_sessions, 0);
}

#[tokio::test]
async fn benchmark_cancellation_drains_accepted_tokens_and_finish_before_next_turn() {
    let backend = Fixture::start(10).await;
    let gateway = TestGateway::start(backend.config(), gateway_config()).await;
    let report = simulation::run(
        &gateway.url,
        SimulationConfig {
            sessions: 2,
            turns_per_session: 2,
            utterance_ms: 50,
            start_spread_ms: 0,
            think_ms: 0,
            interrupt_after_tokens: Some(2),
            ..SimulationConfig::default()
        },
    )
    .await
    .expect("interruption workload");
    assert_eq!(report.failed_sessions, 0, "{:?}", report.sessions);
    assert_eq!(report.interrupted_turns, 4);
    assert_eq!(report.completed_turns, 0);
    assert!(report.received_tokens >= 8);
    let gateway_report = gateway.shutdown().await;
    assert_eq!(
        gateway_report.runtime.generated_tokens as usize,
        report.received_tokens
    );
}

#[tokio::test]
async fn malformed_controls_report_stable_failure_and_endpoint_is_versioned() {
    let backend = Fixture::start(1).await;
    let gateway = TestGateway::start(backend.config(), gateway_config()).await;
    assert!(
        connect_async(gateway.url.trim_end_matches("/v1"))
            .await
            .is_err()
    );
    let (mut websocket, _) = connect_async(&gateway.url).await.expect("handshake");
    websocket
        .send(Message::Text(
            r#"{"type":"open","session_id":"s","unknown":true}"#.into(),
        ))
        .await
        .expect("malformed control sent");
    let event = websocket
        .next()
        .await
        .expect("response")
        .expect("response text");
    let Message::Text(text) = event else {
        panic!("JSON failure expected")
    };
    assert!(matches!(
        serde_json::from_str::<SessionEvent>(&text).expect("typed failure"),
        SessionEvent::Failed {
            code: ErrorCode::InvalidInput,
            ..
        }
    ));
    websocket.close(None).await.expect("close peer");
    gateway.shutdown().await;
}

#[tokio::test]
async fn benchmark_reports_terminal_backend_failure_instead_of_completed_turn() {
    let backend = Fixture::failing_decode(1).await;
    let gateway = TestGateway::start(backend.config(), gateway_config()).await;
    let report = simulation::run(
        &gateway.url,
        SimulationConfig {
            sessions: 1,
            turns_per_session: 1,
            utterance_ms: 50,
            start_spread_ms: 0,
            ..SimulationConfig::default()
        },
    )
    .await
    .expect("failure report");
    assert_eq!(report.failed_sessions, 1);
    assert_eq!(report.completed_turns, 0);
    assert_eq!(report.received_tokens, 1);
    assert!(
        report.sessions[0]
            .error
            .as_ref()
            .expect("failure reason")
            .contains("BackendFailed")
    );
    gateway.shutdown().await;
}

#[tokio::test]
async fn gateway_shutdown_cancels_pending_backend_open() {
    let backend = Fixture::start(5000).await;
    let gateway = TestGateway::start(backend.config(), gateway_config()).await;
    let url = gateway.url.clone();
    let opening = tokio::spawn(async move {
        let mut client = VoiceClient::connect(&url, Duration::from_secs(10))
            .await
            .expect("WebSocket connects");
        client.open(SessionId("slow-open".into())).await
    });
    tokio::time::sleep(Duration::from_millis(100)).await;
    timeout(Duration::from_secs(1), gateway.shutdown())
        .await
        .expect("shutdown does not await backend open latency");
    assert!(opening.await.expect("opening client task").is_err());
}

#[tokio::test]
async fn benchmark_distinguishes_session_admission_from_active_turn_rejection() {
    let backend = Fixture::start(1).await;
    let gateway = TestGateway::start(
        RuntimeConfig {
            max_sessions_per_worker: 8,
            max_active_turns_per_worker: 1,
            ..backend.config()
        },
        gateway_config(),
    )
    .await;
    let report = simulation::run(
        &gateway.url,
        SimulationConfig {
            sessions: 8,
            turns_per_session: 1,
            utterance_ms: 100,
            start_spread_ms: 0,
            ..SimulationConfig::default()
        },
    )
    .await
    .expect("capacity report");
    assert_eq!(report.admitted_sessions, 8);
    assert_eq!(report.rejected_sessions, 0);
    assert_eq!(report.rejected_turns, 7);
    assert_eq!(report.admitted_turns, 1);
    assert_eq!(report.completed_turns, 1);
    assert_eq!(report.failed_sessions, 0);
    gateway.shutdown().await;
}
