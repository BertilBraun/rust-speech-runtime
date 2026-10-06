use bytes::Bytes;
use std::{
    collections::HashMap,
    net::SocketAddr,
    sync::{Arc, Mutex},
    time::Duration,
};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream},
    task::JoinHandle,
};
use voice_scheduler::{
    Node, SessionHandle,
    config::{RuntimeConfig, WorkerConfig},
    protocol::{
        ErrorCode, FinishReason, SessionEvent, SessionId, TurnId,
        backend::{
            BatchRequest, BatchResponse, Memory, Operation, OperationResult, Outcome, Ready, Timing,
        },
    },
};

struct Fixture {
    endpoint: SocketAddr,
    operations: Arc<Mutex<Vec<Operation>>>,
    task: JoinHandle<()>,
}
impl Fixture {
    async fn start(delay_ms: u64) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = listener.local_addr().unwrap();
        let operations = Arc::new(Mutex::new(Vec::new()));
        let recorded = operations.clone();
        let task = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            stream.set_nodelay(true).unwrap();
            write_json(
                &mut stream,
                &Ready {
                    r#type: "ready".into(),
                    protocol_version: 1,
                    body_bytes: 0,
                    model_id: "test-model-v1".into(),
                    max_context_tokens: 16384,
                    max_batch_size: 16,
                    max_audio_samples: 480000,
                },
            )
            .await;
            let mut contexts = HashMap::<String, usize>::new();
            while let Ok(length) = stream.read_u32().await {
                let mut metadata = vec![0; length as usize];
                if stream.read_exact(&mut metadata).await.is_err() {
                    break;
                }
                let request: BatchRequest = serde_json::from_slice(&metadata).unwrap();
                let mut audio = vec![0; request.body_bytes];
                stream.read_exact(&mut audio).await.unwrap();
                let mut results = Vec::new();
                recorded.lock().unwrap().extend(request.operations.clone());
                if delay_ms > 0 {
                    tokio::time::sleep(Duration::from_millis(delay_ms)).await;
                }
                for operation in request.operations {
                    let operation_id = operation.operation_id();
                    let session_id = operation.session_id().to_string();
                    let (turn_id, generation, outcome) = match operation {
                        Operation::Open { .. } => {
                            contexts.insert(session_id.clone(), 0);
                            (None, None, Outcome::Opened)
                        }
                        Operation::Close { .. } => {
                            contexts.remove(&session_id);
                            (None, None, Outcome::Closed)
                        }
                        Operation::Prefill {
                            turn_id,
                            generation,
                            audio_bytes,
                            accepted,
                            ..
                        } => {
                            let context = contexts.get_mut(&session_id).unwrap();
                            *context +=
                                audio_bytes.div_ceil(3200) + 12 + usize::from(accepted.is_some());
                            (
                                Some(turn_id),
                                Some(generation),
                                Outcome::Token {
                                    token_id: 100,
                                    text_delta: "a".into(),
                                    eos: false,
                                    context_tokens: *context,
                                },
                            )
                        }
                        Operation::Decode {
                            turn_id,
                            generation,
                            accepted,
                            ..
                        } => {
                            let context = contexts.get_mut(&session_id).unwrap();
                            *context += 1;
                            (
                                Some(turn_id),
                                Some(generation),
                                Outcome::Token {
                                    token_id: 101 + accepted.index as u32,
                                    text_delta: if accepted.index >= 3 {
                                        String::new()
                                    } else {
                                        "b".into()
                                    },
                                    eos: accepted.index >= 3,
                                    context_tokens: *context,
                                },
                            )
                        }
                    };
                    results.push(OperationResult {
                        operation_id,
                        session_id,
                        turn_id,
                        generation,
                        outcome,
                    });
                }
                write_json(
                    &mut stream,
                    &BatchResponse {
                        request_id: request.request_id,
                        body_bytes: 0,
                        results,
                        timing: Timing {
                            elapsed_ms: delay_ms as f64,
                            decode_ms: delay_ms as f64,
                            ..Timing::default()
                        },
                        memory: Memory::default(),
                    },
                )
                .await;
            }
        });
        Self {
            endpoint,
            operations,
            task,
        }
    }
    fn config(&self) -> RuntimeConfig {
        RuntimeConfig {
            workers: vec![WorkerConfig {
                endpoint: self.endpoint,
            }],
            initial_forward_estimate_ms: 1.0,
            ..RuntimeConfig::default()
        }
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        self.task.abort();
    }
}
async fn write_json<T: serde::Serialize>(stream: &mut TcpStream, value: &T) {
    let metadata = serde_json::to_vec(value).unwrap();
    if stream.write_u32(metadata.len() as u32).await.is_err() {
        return;
    }
    let _ = stream.write_all(&metadata).await;
}
async fn start_turn(session: &SessionHandle, turn_id: u64) {
    session.begin_turn(TurnId(turn_id)).await.unwrap();
    session
        .audio(TurnId(turn_id), 0, Bytes::from(vec![0; 1600]))
        .await
        .unwrap();
    session.commit(TurnId(turn_id), 1, 800).await.unwrap();
}
async fn finish_turn(session: &mut SessionHandle) -> usize {
    let mut tokens = 0;
    loop {
        let event = tokio::time::timeout(Duration::from_secs(5), session.next_event())
            .await
            .unwrap()
            .unwrap();
        match event {
            SessionEvent::TextDelta { .. } => tokens += 1,
            SessionEvent::Finished {
                reason: FinishReason::Eos,
                ..
            } => return tokens,
            SessionEvent::Failed { message, .. } => panic!("{message}"),
            _ => {}
        }
    }
}

#[tokio::test]
async fn multi_turn_history_remains_sticky_and_cache_reconciles_last_token() {
    let backend = Fixture::start(2).await;
    let node = Node::start(backend.config()).await.unwrap();
    let mut session = node
        .ingress()
        .open_session(SessionId("conversation".into()))
        .await
        .unwrap();
    assert_eq!(session.worker_id(), 0);
    start_turn(&session, 1).await;
    assert_eq!(finish_turn(&mut session).await, 5);
    start_turn(&session, 2).await;
    assert_eq!(finish_turn(&mut session).await, 5);
    let record = session.close().await.unwrap();
    assert_eq!(record.turns.len(), 2);
    assert_eq!(record.turns[0].audio_pcm16.len(), 1600);
    assert_eq!(record.turns[1].tokens.len(), 5);
    assert_eq!(node.metrics().generated_tokens, 10);
    assert_eq!(node.metrics().ttft.count, 2);
    node.shutdown().await.unwrap();
}

#[tokio::test]
async fn interruption_discards_inflight_proposals_and_preserves_committed_history() {
    let backend = Fixture::start(40).await;
    let node = Node::start(backend.config()).await.unwrap();
    let mut session = node
        .ingress()
        .open_session(SessionId("interrupt".into()))
        .await
        .unwrap();
    start_turn(&session, 1).await;
    loop {
        if matches!(
            session.next_event().await,
            Some(SessionEvent::TextDelta { .. })
        ) {
            break;
        }
    }
    start_turn(&session, 2).await;
    assert_eq!(finish_turn(&mut session).await, 5);
    let record = session.close().await.unwrap();
    assert_eq!(record.turns[0].finish_reason, Some(FinishReason::Cancelled));
    assert_eq!(record.turns[0].tokens.len(), 1);
    assert_eq!(record.turns[1].tokens.len(), 5);
    assert!(node.metrics().stale_results_discarded >= 1);
    {
        let operations = backend.operations.lock().unwrap();
        assert!(
            operations
                .iter()
                .filter(|operation| matches!(operation, Operation::Decode { turn_id: 1, .. }))
                .count()
                <= 1
        );
    }
    node.shutdown().await.unwrap();
}

#[tokio::test]
async fn admission_rejects_excess_sessions_and_releases_closed_slot() {
    let backend = Fixture::start(1).await;
    let config = RuntimeConfig {
        max_sessions_per_worker: 1,
        max_active_turns_per_worker: 1,
        ..backend.config()
    };
    let node = Node::start(config).await.unwrap();
    let mut session = node
        .ingress()
        .open_session(SessionId("one".into()))
        .await
        .unwrap();
    let error = node
        .ingress()
        .open_session(SessionId("two".into()))
        .await
        .err()
        .unwrap();
    assert_eq!(error.code(), ErrorCode::CapacityExceeded);
    session.close().await.unwrap();
    let mut next = node
        .ingress()
        .open_session(SessionId("one".into()))
        .await
        .unwrap();
    next.close().await.unwrap();
    assert_eq!(node.metrics().active_sessions, 0);
    node.shutdown().await.unwrap();
}

#[tokio::test]
async fn chunk_validation_commit_idempotence_and_record_bounds_are_explicit() {
    let backend = Fixture::start(1).await;
    let node = Node::start(backend.config()).await.unwrap();
    let mut session = node
        .ingress()
        .open_session(SessionId("validation".into()))
        .await
        .unwrap();
    session.begin_turn(TurnId(1)).await.unwrap();
    assert_eq!(
        session
            .audio(TurnId(1), 1, Bytes::from_static(&[0, 0]))
            .await
            .unwrap_err()
            .code(),
        ErrorCode::InvalidInput
    );
    assert_eq!(
        session
            .audio(TurnId(1), 0, Bytes::from_static(&[0]))
            .await
            .unwrap_err()
            .code(),
        ErrorCode::InvalidInput
    );
    session
        .audio(TurnId(1), 0, Bytes::from_static(&[0, 0]))
        .await
        .unwrap();
    assert_eq!(
        session.commit(TurnId(1), 2, 1).await.unwrap_err().code(),
        ErrorCode::InvalidInput
    );
    session.commit(TurnId(1), 1, 1).await.unwrap();
    session.commit(TurnId(1), 1, 1).await.unwrap();
    finish_turn(&mut session).await;
    session.close().await.unwrap();
    node.shutdown().await.unwrap();
}

#[tokio::test]
async fn many_sessions_use_bounded_actor_state_and_dynamic_batches() {
    let backend = Fixture::start(0).await;
    let config = RuntimeConfig {
        max_sessions_per_worker: 1000,
        max_active_turns_per_worker: 64,
        initial_forward_estimate_ms: 0.01,
        ..backend.config()
    };
    let node = Node::start(config).await.unwrap();
    let mut sessions = Vec::new();
    for index in 0..1000 {
        sessions.push(
            node.ingress()
                .open_session(SessionId(format!("session-{index}")))
                .await
                .unwrap(),
        );
    }
    assert_eq!(node.metrics().active_sessions, 1000);
    for session in sessions.iter().take(16) {
        start_turn(session, 1).await;
    }
    for session in sessions.iter_mut().take(16) {
        finish_turn(session).await;
    }
    for mut session in sessions {
        session.close().await.unwrap();
    }
    assert_eq!(node.metrics().active_sessions, 0);
    assert!(node.metrics().batch_items > node.metrics().batches);
    node.shutdown().await.unwrap();
}

#[tokio::test]
async fn dropped_handle_reaps_cache_and_slow_output_cannot_block_other_sessions() {
    let backend = Fixture::start(2).await;
    let config = RuntimeConfig {
        event_capacity: 2,
        ..backend.config()
    };
    let node = Node::start(config).await.unwrap();
    let session = node
        .ingress()
        .open_session(SessionId("slow".into()))
        .await
        .unwrap();
    start_turn(&session, 1).await;
    drop(session);
    tokio::time::sleep(Duration::from_millis(100)).await;
    let mut healthy = node
        .ingress()
        .open_session(SessionId("healthy".into()))
        .await
        .unwrap();
    healthy.next_event().await;
    healthy.begin_turn(TurnId(1)).await.unwrap();
    healthy.next_event().await;
    healthy
        .audio(TurnId(1), 0, Bytes::from_static(&[0, 0]))
        .await
        .unwrap();
    healthy.commit(TurnId(1), 1, 1).await.unwrap();
    assert_eq!(finish_turn(&mut healthy).await, 5);
    healthy.close().await.unwrap();
    node.shutdown().await.unwrap();
}

#[tokio::test]
async fn opening_a_slow_worker_does_not_block_admission_to_another_worker() {
    let slow = Fixture::start(150).await;
    let fast = Fixture::start(0).await;
    let node = Node::start(RuntimeConfig {
        workers: vec![
            WorkerConfig {
                endpoint: slow.endpoint,
            },
            WorkerConfig {
                endpoint: fast.endpoint,
            },
        ],
        ..fast.config()
    })
    .await
    .unwrap();
    let first_ingress = node.ingress();
    let first = tokio::spawn(async move {
        first_ingress
            .open_session(SessionId("slow-open".into()))
            .await
            .unwrap()
    });
    tokio::time::sleep(Duration::from_millis(10)).await;
    let mut second = tokio::time::timeout(
        Duration::from_millis(100),
        node.ingress().open_session(SessionId("fast-open".into())),
    )
    .await
    .unwrap()
    .unwrap();
    assert_eq!(second.worker_id(), 1);
    let mut first = first.await.unwrap();
    first.close().await.unwrap();
    second.close().await.unwrap();
    node.shutdown().await.unwrap();
}

#[tokio::test]
async fn accepted_output_is_preserved_in_order_across_interrupt_and_close() {
    let backend = Fixture::start(0).await;
    let node = Node::start(backend.config()).await.unwrap();
    let mut session = node
        .ingress()
        .open_session(SessionId("ordered".into()))
        .await
        .unwrap();
    start_turn(&session, 1).await;
    while node.metrics().generated_tokens < 5 {
        tokio::task::yield_now().await;
    }
    session.begin_turn(TurnId(2)).await.unwrap();
    let record = session.close().await.unwrap();
    let mut events = Vec::new();
    while let Some(event) = session.next_event().await {
        events.push(event);
    }
    let accepted_new = events
        .iter()
        .position(|event| matches!(event, SessionEvent::Accepted { turn_id: TurnId(2) }))
        .unwrap();
    let old_tokens = events
        .iter()
        .enumerate()
        .filter(|(_, event)| {
            matches!(
                event,
                SessionEvent::TextDelta {
                    turn_id: TurnId(1),
                    ..
                }
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(old_tokens.len(), 5);
    assert!(old_tokens.iter().all(|(index, _)| *index < accepted_new));
    assert_eq!(record.turns[0].tokens.len(), 5);
    node.shutdown().await.unwrap();
}

#[tokio::test]
async fn stalled_consumer_stops_generation_and_retains_record_for_explicit_archive() {
    let backend = Fixture::start(0).await;
    let node = Node::start(RuntimeConfig {
        event_capacity: 2,
        ..backend.config()
    })
    .await
    .unwrap();
    let mut session = node
        .ingress()
        .open_session(SessionId("stalled".into()))
        .await
        .unwrap();
    start_turn(&session, 1).await;
    tokio::time::sleep(Duration::from_millis(150)).await;
    let record = session.close().await.unwrap();
    assert_eq!(
        record.turns[0].finish_reason,
        Some(FinishReason::SlowConsumer)
    );
    assert!(record.turns[0].tokens.is_empty());
    assert!(node.metrics().channel_saturation_events > 0);
    node.shutdown().await.unwrap();
}
