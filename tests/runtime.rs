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
struct FixtureConfiguration {
    open_delay_ms: u64,
    prefill_delay_ms: u64,
    decode_delay_ms: u64,
    response_tokens: u64,
    maximum_sessions: usize,
    fail_decode: bool,
    fail_prepare: bool,
    fail_activate: bool,
    activate_delay_ms: u64,
}
struct PreparedCache {
    turn_id: u64,
    generation: u64,
    context: usize,
}

impl Fixture {
    async fn start(delay_ms: u64) -> Self {
        Self::limited(delay_ms, usize::MAX).await
    }

    async fn limited(delay_ms: u64, maximum_sessions: usize) -> Self {
        Self::configured(delay_ms, maximum_sessions, false).await
    }

    async fn configured(delay_ms: u64, maximum_sessions: usize, fail_decode: bool) -> Self {
        Self::with_configuration(FixtureConfiguration {
            open_delay_ms: delay_ms,
            prefill_delay_ms: delay_ms,
            decode_delay_ms: delay_ms,
            response_tokens: 4,
            maximum_sessions,
            fail_decode,
            fail_prepare: false,
            fail_activate: false,
            activate_delay_ms: 0,
        })
        .await
    }

    async fn with_configuration(configuration: FixtureConfiguration) -> Self {
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
            let mut prepared = HashMap::<String, PreparedCache>::new();
            while let Ok(length) = stream.read_u32().await {
                let mut metadata = vec![0; length as usize];
                if stream.read_exact(&mut metadata).await.is_err() {
                    break;
                }
                let request: BatchRequest = serde_json::from_slice(&metadata).unwrap();
                assert!(
                    request
                        .operations
                        .iter()
                        .all(|operation| std::mem::discriminant(operation)
                            == std::mem::discriminant(&request.operations[0]))
                );
                let mut audio = vec![0; request.body_bytes];
                stream.read_exact(&mut audio).await.unwrap();
                let mut results = Vec::new();
                recorded.lock().unwrap().extend(request.operations.clone());
                let delay_ms = match request.operations[0] {
                    Operation::Prefill { .. } | Operation::Prepare { .. } => {
                        configuration.prefill_delay_ms
                    }
                    Operation::Decode { .. } => configuration.decode_delay_ms,
                    Operation::Activate { .. } => configuration.activate_delay_ms,
                    Operation::DiscardPrepared { .. } => 0,
                    _ => configuration.open_delay_ms,
                };
                if delay_ms > 0 {
                    tokio::time::sleep(Duration::from_millis(delay_ms)).await;
                }
                for operation in request.operations {
                    let operation_id = operation.operation_id();
                    let session_id = operation.session_id().to_string();
                    let (turn_id, generation, outcome) = match operation {
                        Operation::Open { .. } => {
                            if contexts.len() >= configuration.maximum_sessions {
                                (
                                    None,
                                    None,
                                    Outcome::Failed {
                                        code: ErrorCode::CapacityExceeded,
                                        message: "fixture cache full".into(),
                                    },
                                )
                            } else {
                                contexts.insert(session_id.clone(), 0);
                                (None, None, Outcome::Opened)
                            }
                        }
                        Operation::Close { .. } => {
                            contexts.remove(&session_id);
                            prepared.remove(&session_id);
                            (None, None, Outcome::Closed)
                        }
                        Operation::Prepare {
                            turn_id,
                            generation,
                            audio_bytes,
                            accepted,
                            ..
                        } => {
                            if configuration.fail_prepare {
                                results.push(OperationResult {
                                    operation_id,
                                    session_id,
                                    turn_id: Some(turn_id),
                                    generation: Some(generation),
                                    outcome: Outcome::Failed {
                                        code: ErrorCode::CapacityExceeded,
                                        message: "preparation capacity unavailable".into(),
                                    },
                                });
                                continue;
                            }
                            let context = contexts[&session_id]
                                + audio_bytes.div_ceil(3200)
                                + 12
                                + usize::from(accepted.is_some());
                            prepared.insert(
                                session_id.clone(),
                                PreparedCache {
                                    turn_id,
                                    generation,
                                    context,
                                },
                            );
                            (
                                Some(turn_id),
                                Some(generation),
                                Outcome::Token {
                                    token_id: 100,
                                    text_delta: "a".into(),
                                    eos: false,
                                    context_tokens: context,
                                },
                            )
                        }
                        Operation::Activate {
                            turn_id,
                            generation,
                            ..
                        } => {
                            if configuration.fail_activate {
                                results.push(OperationResult {
                                    operation_id,
                                    session_id,
                                    turn_id: Some(turn_id),
                                    generation: Some(generation),
                                    outcome: Outcome::Failed {
                                        code: ErrorCode::InvalidState,
                                        message: "preparation unavailable".into(),
                                    },
                                });
                                continue;
                            }
                            let cache = prepared.remove(&session_id).expect("prepared cache");
                            assert_eq!((cache.turn_id, cache.generation), (turn_id, generation));
                            contexts.insert(session_id.clone(), cache.context);
                            (
                                Some(turn_id),
                                Some(generation),
                                Outcome::Token {
                                    token_id: 100,
                                    text_delta: "a".into(),
                                    eos: false,
                                    context_tokens: cache.context,
                                },
                            )
                        }
                        Operation::DiscardPrepared { .. } => {
                            prepared.remove(&session_id);
                            (None, None, Outcome::Discarded)
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
                            if configuration.fail_decode {
                                (
                                    Some(turn_id),
                                    Some(generation),
                                    Outcome::Failed {
                                        code: ErrorCode::CapacityExceeded,
                                        message: "fixture rejected decode before consuming token"
                                            .into(),
                                    },
                                )
                            } else {
                                let context = contexts.get_mut(&session_id).unwrap();
                                *context += 1;
                                (
                                    Some(turn_id),
                                    Some(generation),
                                    Outcome::Token {
                                        token_id: 101 + accepted.index as u32,
                                        text_delta: if accepted.index
                                            >= configuration.response_tokens - 1
                                        {
                                            String::new()
                                        } else {
                                            "b".into()
                                        },
                                        eos: accepted.index >= configuration.response_tokens - 1,
                                        context_tokens: *context,
                                    },
                                )
                            }
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

#[tokio::test]
async fn backend_cache_rejection_isolated_from_existing_session() {
    let backend = Fixture::limited(0, 1).await;
    let node = Node::start(backend.config()).await.unwrap();
    let mut existing = node
        .ingress()
        .open_session(SessionId("existing".into()))
        .await
        .unwrap();
    let rejected = node
        .ingress()
        .open_session(SessionId("rejected".into()))
        .await
        .err()
        .unwrap();
    assert_eq!(rejected.code(), ErrorCode::CapacityExceeded);
    assert_eq!(node.metrics().active_sessions, 1);
    start_turn(&existing, 1).await;
    assert_eq!(finish_turn(&mut existing).await, 5);
    existing.close().await.unwrap();
    node.shutdown().await.unwrap();
}

#[tokio::test]
async fn backend_disconnect_retains_accepted_history_for_archive() {
    let backend = Fixture::start(25).await;
    let node = Node::start(backend.config()).await.unwrap();
    let mut session = node
        .ingress()
        .open_session(SessionId("disconnect".into()))
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
    backend.task.abort();
    loop {
        if matches!(
            tokio::time::timeout(Duration::from_secs(2), session.next_event())
                .await
                .unwrap(),
            Some(SessionEvent::Failed { .. })
        ) {
            break;
        }
    }
    assert_eq!(
        session.begin_turn(TurnId(2)).await.unwrap_err().code(),
        ErrorCode::BackendUnavailable
    );
    let record = session.close().await.unwrap();
    assert!(!record.turns[0].tokens.is_empty());
    assert_eq!(
        record.turns[0].finish_reason,
        Some(FinishReason::BackendFailed)
    );
    node.shutdown().await.unwrap();
}

#[tokio::test]
async fn slowdown_exposes_token_gaps_and_rejects_new_turn_reservations() {
    let backend = Fixture::start(300).await;
    let node = Node::start(backend.config()).await.unwrap();
    let mut generating = node
        .ingress()
        .open_session(SessionId("generating".into()))
        .await
        .unwrap();
    let mut waiting = node
        .ingress()
        .open_session(SessionId("waiting".into()))
        .await
        .unwrap();
    start_turn(&generating, 1).await;
    assert_eq!(finish_turn(&mut generating).await, 5);
    assert!(node.metrics().token_deadline_misses >= 1);
    assert!(node.metrics().token_gap.p50_ms > 250.0);
    assert_eq!(
        waiting.begin_turn(TurnId(1)).await.unwrap_err().code(),
        ErrorCode::CapacityExceeded
    );
    generating.close().await.unwrap();
    waiting.close().await.unwrap();
    node.shutdown().await.unwrap();
}

#[tokio::test]
async fn closed_handle_rejects_future_commands_and_duplicate_commit_metadata_changes() {
    let backend = Fixture::start(0).await;
    let node = Node::start(backend.config()).await.unwrap();
    let mut session = node
        .ingress()
        .open_session(SessionId("closed".into()))
        .await
        .unwrap();
    start_turn(&session, 1).await;
    assert_eq!(
        session.commit(TurnId(1), 2, 800).await.unwrap_err().code(),
        ErrorCode::InvalidInput
    );
    session.cancel(TurnId(1)).await.unwrap();
    let record = session.close().await.unwrap();
    assert_eq!(record.turns[0].finish_reason, Some(FinishReason::Cancelled));
    assert_eq!(
        session.close().await.unwrap_err().code(),
        ErrorCode::SessionNotFound
    );
    assert_eq!(
        session.begin_turn(TurnId(2)).await.unwrap_err().code(),
        ErrorCode::SessionNotFound
    );
    node.shutdown().await.unwrap();
}

#[tokio::test]
async fn failed_decode_makes_cache_terminal_without_losing_accepted_record() {
    let backend = Fixture::configured(0, 64, true).await;
    let node = Node::start(backend.config()).await.unwrap();
    let mut session = node
        .ingress()
        .open_session(SessionId("failed-decode".into()))
        .await
        .unwrap();
    start_turn(&session, 1).await;
    loop {
        if matches!(
            session.next_event().await,
            Some(SessionEvent::Failed {
                code: ErrorCode::CapacityExceeded,
                ..
            })
        ) {
            break;
        }
    }
    assert_eq!(
        session.begin_turn(TurnId(2)).await.unwrap_err().code(),
        ErrorCode::SessionNotFound
    );
    let record = session.close().await.unwrap();
    assert_eq!(record.turns[0].tokens.len(), 1);
    assert_eq!(
        record.turns[0].finish_reason,
        Some(FinishReason::BackendFailed)
    );
    node.shutdown().await.unwrap();
}

#[tokio::test]
async fn slow_prefill_with_fast_decode_still_allows_sequential_turns() {
    let backend = Fixture::with_configuration(FixtureConfiguration {
        open_delay_ms: 0,
        prefill_delay_ms: 300,
        decode_delay_ms: 20,
        response_tokens: 4,
        maximum_sessions: 64,
        fail_decode: false,
        fail_prepare: false,
        fail_activate: false,
        activate_delay_ms: 0,
    })
    .await;
    let node = Node::start(backend.config()).await.unwrap();
    let mut session = node
        .ingress()
        .open_session(SessionId("solo-turns".into()))
        .await
        .unwrap();
    start_turn(&session, 1).await;
    assert_eq!(finish_turn(&mut session).await, 5);
    start_turn(&session, 2).await;
    assert_eq!(finish_turn(&mut session).await, 5);
    let record = session.close().await.unwrap();
    assert_eq!(record.turns.len(), 2);
    assert_eq!(node.metrics().token_deadline_misses, 0);
    node.shutdown().await.unwrap();
}

#[tokio::test]
async fn admitted_prefill_has_bounded_wait_even_when_it_exceeds_token_gap_budget() {
    let backend = Fixture::with_configuration(FixtureConfiguration {
        open_delay_ms: 0,
        prefill_delay_ms: 300,
        decode_delay_ms: 20,
        response_tokens: 100,
        maximum_sessions: 64,
        fail_decode: false,
        fail_prepare: false,
        fail_activate: false,
        activate_delay_ms: 0,
    })
    .await;
    let node = Node::start(RuntimeConfig {
        max_prefill_wait_ms: 100,
        ..backend.config()
    })
    .await
    .unwrap();
    let mut first = node
        .ingress()
        .open_session(SessionId("first-generator".into()))
        .await
        .unwrap();
    let mut second = node
        .ingress()
        .open_session(SessionId("queued-prefill".into()))
        .await
        .unwrap();
    first.begin_turn(TurnId(1)).await.unwrap();
    second.begin_turn(TurnId(1)).await.unwrap();
    first
        .audio(TurnId(1), 0, Bytes::from_static(&[0, 0]))
        .await
        .unwrap();
    second
        .audio(TurnId(1), 0, Bytes::from_static(&[0, 0]))
        .await
        .unwrap();
    first.commit(TurnId(1), 1, 1).await.unwrap();
    second.commit(TurnId(1), 1, 1).await.unwrap();
    let mut second_tokens = 0;
    loop {
        match tokio::time::timeout(Duration::from_secs(2), second.next_event())
            .await
            .unwrap()
            .unwrap()
        {
            SessionEvent::TextDelta { .. } => {
                second_tokens += 1;
                break;
            }
            SessionEvent::Failed { message, .. } => panic!("{message}"),
            _ => {}
        }
    }
    assert_eq!(second_tokens, 1);
    assert!(node.metrics().ttft.max_ms < 1500.0);
    first.cancel(TurnId(1)).await.unwrap();
    second.cancel(TurnId(1)).await.unwrap();
    first.close().await.unwrap();
    second.close().await.unwrap();
    node.shutdown().await.unwrap();
}

#[tokio::test]
async fn rejected_accepted_event_does_not_leave_capture_reservation_alive() {
    let backend = Fixture::start(0).await;
    let node = Node::start(RuntimeConfig {
        event_capacity: 1,
        ..backend.config()
    })
    .await
    .unwrap();
    let mut session = node
        .ingress()
        .open_session(SessionId("accepted-full".into()))
        .await
        .unwrap();
    assert_eq!(
        session.begin_turn(TurnId(1)).await.unwrap_err().code(),
        ErrorCode::SlowConsumer
    );
    let record = session.close().await.unwrap();
    assert_eq!(
        record.turns[0].finish_reason,
        Some(FinishReason::SlowConsumer)
    );
    assert_eq!(node.metrics().active_sessions, 0);
    node.shutdown().await.unwrap();
}

async fn capture(session: &SessionHandle, turn_id: u64) {
    session.begin_turn(TurnId(turn_id)).await.unwrap();
    session
        .audio(TurnId(turn_id), 0, Bytes::from(vec![0; 1600]))
        .await
        .unwrap();
}

async fn wait_for_operation(backend: &Fixture, predicate: impl Fn(&Operation) -> bool) {
    tokio::time::timeout(Duration::from_secs(2), async {
        let mut interval = tokio::time::interval(Duration::from_millis(1));
        loop {
            if backend.operations.lock().unwrap().iter().any(&predicate) {
                return;
            }
            interval.tick().await;
        }
    })
    .await
    .unwrap();
}

async fn wait_prepared(
    session: &mut SessionHandle,
    turn_id: u64,
    chunk_count: u32,
    sample_count: usize,
) {
    loop {
        match tokio::time::timeout(Duration::from_secs(2), session.next_event())
            .await
            .unwrap()
            .unwrap()
        {
            SessionEvent::Prepared {
                turn_id: prepared_turn,
                chunk_count: prepared_chunks,
                sample_count: prepared_samples,
            } => {
                assert_eq!(
                    (prepared_turn.0, prepared_chunks, prepared_samples),
                    (turn_id, chunk_count, sample_count)
                );
                return;
            }
            SessionEvent::TextDelta { .. } => panic!("provisional output must remain private"),
            SessionEvent::Failed { message, .. } => panic!("{message}"),
            _ => {}
        }
    }
}

#[tokio::test]
async fn preparation_is_private_idempotent_and_activation_avoids_another_forward() {
    let backend = Fixture::configured(80, 64, false).await;
    let node = Node::start(backend.config()).await.unwrap();
    let mut session = node
        .ingress()
        .open_session(SessionId("prepared".into()))
        .await
        .unwrap();
    capture(&session, 1).await;
    assert_eq!(
        session.prepare(TurnId(1), 2, 800).await.unwrap_err().code(),
        ErrorCode::InvalidInput
    );
    session.prepare(TurnId(1), 1, 800).await.unwrap();
    session.prepare(TurnId(1), 1, 800).await.unwrap();
    wait_prepared(&mut session, 1, 1, 800).await;
    session.prepare(TurnId(1), 1, 800).await.unwrap();
    assert_eq!(node.metrics().generated_tokens, 0);
    assert!(
        tokio::time::timeout(Duration::from_millis(10), session.next_event())
            .await
            .is_err()
    );
    session.commit(TurnId(1), 1, 800).await.unwrap();
    assert_eq!(finish_turn(&mut session).await, 5);
    assert!(node.metrics().ttft.max_ms < 70.0);
    let record = session.close().await.unwrap();
    assert_eq!(
        record.turns[0]
            .tokens
            .iter()
            .map(|token| token.index)
            .collect::<Vec<_>>(),
        vec![0, 1, 2, 3, 4]
    );
    assert_eq!(node.metrics().preparations_started, 1);
    assert_eq!(node.metrics().preparations_activated, 1);
    assert!(
        !backend
            .operations
            .lock()
            .unwrap()
            .iter()
            .any(|operation| matches!(operation, Operation::Prefill { .. }))
    );
    node.shutdown().await.unwrap();
}

#[tokio::test]
async fn commit_during_running_preparation_waits_then_activates_once() {
    let backend = Fixture::start(50).await;
    let node = Node::start(backend.config()).await.unwrap();
    let mut session = node
        .ingress()
        .open_session(SessionId("commit-running".into()))
        .await
        .unwrap();
    capture(&session, 1).await;
    session.prepare(TurnId(1), 1, 800).await.unwrap();
    wait_for_operation(&backend, |operation| {
        matches!(operation, Operation::Prepare { .. })
    })
    .await;
    session.commit(TurnId(1), 1, 800).await.unwrap();
    session.commit(TurnId(1), 1, 800).await.unwrap();
    assert_eq!(finish_turn(&mut session).await, 5);
    assert_eq!(node.metrics().preparations_activated, 1);
    assert!(
        !backend
            .operations
            .lock()
            .unwrap()
            .iter()
            .any(|operation| matches!(operation, Operation::Prefill { .. }))
    );
    session.close().await.unwrap();
    node.shutdown().await.unwrap();
}

#[tokio::test]
async fn resumed_audio_invalidates_running_preparation_and_can_prepare_the_new_snapshot() {
    let backend = Fixture::start(60).await;
    let node = Node::start(backend.config()).await.unwrap();
    let mut session = node
        .ingress()
        .open_session(SessionId("resumed".into()))
        .await
        .unwrap();
    capture(&session, 1).await;
    session.prepare(TurnId(1), 1, 800).await.unwrap();
    wait_for_operation(&backend, |operation| {
        matches!(operation, Operation::Prepare { .. })
    })
    .await;
    session
        .audio(TurnId(1), 1, Bytes::from(vec![0; 1600]))
        .await
        .unwrap();
    session.prepare(TurnId(1), 2, 1600).await.unwrap();
    session.commit(TurnId(1), 2, 1600).await.unwrap();
    wait_prepared(&mut session, 1, 2, 1600).await;
    assert_eq!(finish_turn(&mut session).await, 5);
    let metrics = node.metrics();
    assert_eq!(metrics.preparations_started, 2);
    assert_eq!(metrics.preparations_discarded, 1);
    assert_eq!(metrics.preparations_activated, 1);
    assert_eq!(metrics.stale_results_discarded, 1);
    let record = session.close().await.unwrap();
    assert_eq!(record.turns[0].audio_pcm16.len(), 3200);
    assert_eq!(record.turns[0].tokens.len(), 5);
    node.shutdown().await.unwrap();
}

#[tokio::test]
async fn cancelled_preparation_does_not_change_the_next_turn_history() {
    let backend = Fixture::start(30).await;
    let node = Node::start(backend.config()).await.unwrap();
    let mut session = node
        .ingress()
        .open_session(SessionId("cancelled-preparation".into()))
        .await
        .unwrap();
    capture(&session, 1).await;
    session.prepare(TurnId(1), 1, 800).await.unwrap();
    wait_for_operation(&backend, |operation| {
        matches!(operation, Operation::Prepare { .. })
    })
    .await;
    session.cancel(TurnId(1)).await.unwrap();
    capture(&session, 2).await;
    session.prepare(TurnId(2), 1, 800).await.unwrap();
    session.commit(TurnId(2), 1, 800).await.unwrap();
    assert_eq!(finish_turn(&mut session).await, 5);
    let record = session.close().await.unwrap();
    assert!(!record.turns[0].committed);
    assert!(record.turns[0].tokens.is_empty());
    assert_eq!(record.turns[0].finish_reason, Some(FinishReason::Cancelled));
    assert_eq!(record.turns[1].tokens.len(), 5);
    assert_eq!(node.metrics().preparations_discarded, 1);
    node.shutdown().await.unwrap();
}

#[tokio::test]
async fn resumed_audio_without_new_preparation_falls_back_to_full_prefill() {
    let backend = Fixture::start(20).await;
    let node = Node::start(backend.config()).await.unwrap();
    let mut session = node
        .ingress()
        .open_session(SessionId("fallback-audio".into()))
        .await
        .unwrap();
    capture(&session, 1).await;
    session.prepare(TurnId(1), 1, 800).await.unwrap();
    wait_prepared(&mut session, 1, 1, 800).await;
    session
        .audio(TurnId(1), 1, Bytes::from(vec![0; 1600]))
        .await
        .unwrap();
    session.commit(TurnId(1), 2, 1600).await.unwrap();
    assert_eq!(finish_turn(&mut session).await, 5);
    assert_eq!(node.metrics().preparations_activated, 0);
    assert!(
        backend
            .operations
            .lock()
            .unwrap()
            .iter()
            .any(|operation| matches!(
                operation,
                Operation::Prefill {
                    audio_bytes: 3200,
                    ..
                }
            ))
    );
    session.close().await.unwrap();
    node.shutdown().await.unwrap();
}

#[tokio::test]
async fn failed_preparation_and_activation_preserve_previous_accepted_token_for_fallback() {
    for fail_prepare in [true, false] {
        let backend = Fixture::with_configuration(FixtureConfiguration {
            open_delay_ms: 0,
            prefill_delay_ms: 20,
            decode_delay_ms: 5,
            response_tokens: 4,
            maximum_sessions: 64,
            fail_decode: false,
            fail_prepare,
            fail_activate: !fail_prepare,
            activate_delay_ms: 0,
        })
        .await;
        let node = Node::start(backend.config()).await.unwrap();
        let mut session = node
            .ingress()
            .open_session(SessionId("failed-speculation".into()))
            .await
            .unwrap();
        start_turn(&session, 1).await;
        assert_eq!(finish_turn(&mut session).await, 5);
        capture(&session, 2).await;
        session.prepare(TurnId(2), 1, 800).await.unwrap();
        session.commit(TurnId(2), 1, 800).await.unwrap();
        assert_eq!(finish_turn(&mut session).await, 5);
        let record = session.close().await.unwrap();
        assert_eq!(record.turns[0].tokens.len(), 5);
        assert_eq!(record.turns[1].tokens.len(), 5);
        assert_eq!(node.metrics().preparation_fallbacks, 1);
        assert_eq!(node.metrics().backend_failures, 0);
        assert!(backend.operations.lock().unwrap().iter().any(|operation| matches!(operation, Operation::Prefill {turn_id: 2, accepted: Some(accepted), .. } if accepted.turn_id == 1 && accepted.index == 4)));
        node.shutdown().await.unwrap();
    }
}

#[tokio::test]
async fn stale_failed_preparation_does_not_terminate_a_new_turn() {
    let backend = Fixture::with_configuration(FixtureConfiguration {
        open_delay_ms: 0,
        prefill_delay_ms: 40,
        decode_delay_ms: 2,
        response_tokens: 4,
        maximum_sessions: 64,
        fail_decode: false,
        fail_prepare: true,
        fail_activate: false,
        activate_delay_ms: 0,
    })
    .await;
    let node = Node::start(backend.config()).await.unwrap();
    let mut session = node
        .ingress()
        .open_session(SessionId("stale-failure".into()))
        .await
        .unwrap();
    capture(&session, 1).await;
    session.prepare(TurnId(1), 1, 800).await.unwrap();
    wait_for_operation(&backend, |operation| {
        matches!(operation, Operation::Prepare { .. })
    })
    .await;
    start_turn(&session, 2).await;
    assert_eq!(finish_turn(&mut session).await, 5);
    let record = session.close().await.unwrap();
    assert!(record.turns[0].tokens.is_empty());
    assert_eq!(record.turns[1].tokens.len(), 5);
    assert_eq!(node.metrics().backend_failures, 0);
    node.shutdown().await.unwrap();
}

#[tokio::test]
async fn interrupted_activation_does_not_resend_previously_reconciled_token() {
    let backend = Fixture::with_configuration(FixtureConfiguration {
        open_delay_ms: 0,
        prefill_delay_ms: 5,
        decode_delay_ms: 2,
        response_tokens: 4,
        maximum_sessions: 64,
        fail_decode: false,
        fail_prepare: false,
        fail_activate: false,
        activate_delay_ms: 60,
    })
    .await;
    let node = Node::start(backend.config()).await.unwrap();
    let mut session = node
        .ingress()
        .open_session(SessionId("interrupted-activation".into()))
        .await
        .unwrap();
    start_turn(&session, 1).await;
    assert_eq!(finish_turn(&mut session).await, 5);
    capture(&session, 2).await;
    session.prepare(TurnId(2), 1, 800).await.unwrap();
    wait_prepared(&mut session, 2, 1, 800).await;
    session.commit(TurnId(2), 1, 800).await.unwrap();
    wait_for_operation(&backend, |operation| {
        matches!(operation, Operation::Activate { turn_id: 2, .. })
    })
    .await;
    capture(&session, 3).await;
    session.prepare(TurnId(3), 1, 800).await.unwrap();
    session.commit(TurnId(3), 1, 800).await.unwrap();
    assert_eq!(finish_turn(&mut session).await, 5);
    let record = session.close().await.unwrap();
    assert!(record.turns[1].tokens.is_empty());
    assert_eq!(record.turns[2].tokens.len(), 5);
    assert!(
        backend
            .operations
            .lock()
            .unwrap()
            .iter()
            .any(|operation| matches!(
                operation,
                Operation::Prepare {
                    turn_id: 3,
                    accepted: None,
                    ..
                }
            ))
    );
    node.shutdown().await.unwrap();
}
