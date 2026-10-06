use bytes::Bytes;
use std::time::Duration;
use tokio::time::Instant;
use voice_scheduler::{
    Node, RuntimeError,
    config::{AudioLimits, RuntimeConfig, WorkerSlowdown},
    protocol::{
        Assignment, AudioContext, AudioPacket, AudioPrefix, CacheOutcome, CreateOutcome,
        CreateRejection, FrameRejection, InputFrame, InputOutcome, PacketSequence, PrefixState,
        SessionId, SessionLease,
    },
};

fn configuration() -> RuntimeConfig {
    RuntimeConfig {
        workers: 1,
        inference_latency: Duration::from_millis(3),
        packet_deadline: Duration::from_millis(250),
        minimum_packet_interval: Duration::from_millis(250),
        max_batch_wait: Duration::ZERO,
        max_sessions_per_worker: 4,
        cache_slots_per_worker: 4,
        batch_size: 4,
        calibration_samples: 2,
        latency_window: 8,
        ..RuntimeConfig::default()
    }
}
fn frame(sequence: u64, prefix: PrefixState, payload: Bytes, budget: Duration) -> InputFrame {
    let timestamp = Instant::now();
    InputFrame {
        timestamp,
        deadline: timestamp + budget,
        packet: AudioPacket {
            sequence: PacketSequence(sequence),
            payload,
            context: AudioContext::Cached(prefix),
        },
    }
}
async fn admit(node: &Node, session_id: u64) -> Assignment {
    match node
        .ingress
        .create_session(SessionId(session_id))
        .await
        .unwrap()
    {
        CreateOutcome::Admitted(admission) => admission.assignment,
        outcome => panic!("expected admission: {outcome:?}"),
    }
}
async fn first_packet(node: &Node, session_id: u64) -> InputOutcome {
    node.ingress
        .input_frame(
            SessionId(session_id),
            frame(
                0,
                PrefixState::default(),
                Bytes::from_static(b"audio"),
                Duration::from_millis(200),
            ),
        )
        .await
        .unwrap()
}

#[tokio::test]
async fn calibration_rejects_sessions_that_cannot_fit_realtime_budget() {
    let node = Node::start(RuntimeConfig {
        minimum_packet_interval: Duration::from_millis(48),
        inference_latency: Duration::from_millis(60),
        ..configuration()
    })
    .await
    .unwrap();
    assert_eq!(
        node.ingress.create_session(SessionId(1)).await.unwrap(),
        CreateOutcome::Rejected(CreateRejection::Capacity)
    );
    let report = node.shutdown().await.unwrap();
    assert_eq!(report.workers[0].initial_session_limit, 0);
    assert!(report.workers[0].calibration_latency.samples > 0);
}
#[tokio::test(start_paused = true)]
async fn long_queued_device_work_pauses_new_admission_until_it_completes() {
    let node = Node::start(RuntimeConfig {
        packet_deadline: Duration::from_millis(500),
        replay_latency_per_packet: Duration::from_millis(2),
        ..configuration()
    })
    .await
    .unwrap();
    admit(&node, 1).await;
    let mut prefix = PrefixState::default();
    let mut history = Vec::new();
    for sequence in 0..120 {
        let payload = Bytes::from(vec![sequence as u8]);
        let InputOutcome::Processed(output) = node
            .ingress
            .input_frame(
                SessionId(1),
                frame(
                    sequence,
                    prefix,
                    payload.clone(),
                    Duration::from_millis(450),
                ),
            )
            .await
            .unwrap()
        else {
            panic!("expected prefix warmup");
        };
        prefix = output.audio.prefix;
        history.push(payload);
    }
    assert!(node.ingress.evict_cache(SessionId(1)).await.unwrap());
    let ingress = node.ingress.clone();
    let replay = tokio::spawn(async move {
        let mut input = frame(
            120,
            prefix,
            Bytes::from_static(b"next"),
            Duration::from_millis(450),
        );
        input.packet.context = AudioContext::Replay(AudioPrefix(history));
        ingress.input_frame(SessionId(1), input).await.unwrap()
    });
    tokio::time::sleep(Duration::from_millis(20)).await;
    assert_eq!(
        node.ingress.create_session(SessionId(2)).await.unwrap(),
        CreateOutcome::Rejected(CreateRejection::Capacity)
    );
    assert!(matches!(replay.await.unwrap(), InputOutcome::Processed(_)));
    admit(&node, 2).await;
    let report = node.shutdown().await.unwrap();
    assert_eq!(report.rejected_sessions, 1);
    assert_eq!(report.inference.capacity_terminations, 0);
}

#[tokio::test]
async fn capacity_and_cache_slots_are_reclaimed() {
    let node = Node::start(RuntimeConfig {
        cache_slots_per_worker: 2,
        ..configuration()
    })
    .await
    .unwrap();
    let original = admit(&node, 1).await;
    admit(&node, 2).await;
    assert_eq!(
        node.ingress.create_session(SessionId(3)).await.unwrap(),
        CreateOutcome::Rejected(CreateRejection::Capacity)
    );
    assert_eq!(
        node.ingress.create_session(SessionId(1)).await.unwrap(),
        CreateOutcome::Rejected(CreateRejection::AlreadyExists)
    );
    assert!(node.ingress.close_session(SessionId(1)).await.unwrap());
    let replacement = admit(&node, 1).await;
    assert!(replacement.generation.0 > original.generation.0);
    assert_eq!(original.worker_id, replacement.worker_id);
    let report = node.shutdown().await.unwrap();
    assert_eq!(report.peak_active_sessions, 2);
    assert_eq!(report.rejected_sessions, 1);
}
#[tokio::test]
async fn audio_echo_preserves_sequence_payload_and_entire_prefix() {
    let node = Node::start(configuration()).await.unwrap();
    let assignment = admit(&node, 1).await;
    let mut prefix = PrefixState::default();
    for sequence in 0..5 {
        let payload = Bytes::from(vec![sequence as u8; 1600]);
        let expected = prefix.append(&payload);
        let InputOutcome::Processed(output) = node
            .ingress
            .input_frame(
                SessionId(1),
                frame(
                    sequence,
                    prefix,
                    payload.clone(),
                    Duration::from_millis(200),
                ),
            )
            .await
            .unwrap()
        else {
            panic!("expected echo");
        };
        assert_eq!(output.audio.assignment, assignment);
        assert_eq!(output.audio.sequence, PacketSequence(sequence));
        assert_eq!(output.audio.payload, payload);
        assert_eq!(output.audio.prefix, expected);
        assert_eq!(output.audio.cache, CacheOutcome::Hit);
        prefix = expected;
    }
    let report = node.shutdown().await.unwrap();
    assert_eq!(report.inference.delivered_frames, 5);
    assert_eq!(report.inference.cache_hits, 5);
}

#[tokio::test]
async fn packet_profile_separates_ingress_mailbox_batching_and_device_time() {
    let node = Node::start(RuntimeConfig {
        worker_input_delay: Duration::from_millis(5),
        max_batch_wait: Duration::from_millis(10),
        ..configuration()
    })
    .await
    .unwrap();
    admit(&node, 1).await;
    admit(&node, 2).await;
    let input = frame(
        0,
        PrefixState::default(),
        Bytes::from_static(b"profile"),
        Duration::from_millis(200),
    );
    tokio::time::sleep(Duration::from_millis(5)).await;
    let InputOutcome::Processed(output) =
        node.ingress.input_frame(SessionId(1), input).await.unwrap()
    else {
        panic!("expected profiled audio");
    };
    let timings = &output.audio.timings;
    assert!(timings.ingress >= Duration::from_millis(5));
    assert!(timings.worker_mailbox >= Duration::from_millis(5));
    assert!(timings.scheduler_queue >= Duration::from_millis(10));
    assert!(timings.device_execution >= Duration::from_millis(3));
    assert!(timings.total() >= output.completed_at - output.input_timestamp);
    assert!(timings.total() <= output.input_timestamp.elapsed());
    let report = node.shutdown().await.unwrap();
    assert_eq!(report.ingress_delay.samples, 1);
    let worker = &report.workers[0];
    assert_eq!(worker.profile.worker_mailbox.samples, 1);
    assert_eq!(worker.profile.scheduler_queue.samples, 1);
    assert_eq!(worker.slowest_packets.len(), 1);
}

#[tokio::test]
async fn simultaneous_inputs_form_one_full_batch_before_the_idle_device_starts() {
    let node = Node::start(RuntimeConfig {
        max_batch_wait: Duration::from_millis(1),
        ..configuration()
    })
    .await
    .unwrap();
    for session_id in 0..4 {
        admit(&node, session_id).await;
    }
    let outcomes = tokio::join!(
        first_packet(&node, 0),
        first_packet(&node, 1),
        first_packet(&node, 2),
        first_packet(&node, 3),
    );
    for outcome in [outcomes.0, outcomes.1, outcomes.2, outcomes.3] {
        assert!(matches!(outcome, InputOutcome::Processed(_)));
    }
    let report = node.shutdown().await.unwrap();
    assert_eq!(report.inference.processed_frames, 4);
    assert_eq!(report.inference.batches, 1);
    assert_eq!(report.batch_fill_ratio, 1.0);
}

#[tokio::test(start_paused = true)]
async fn final_partial_batch_is_prequeued_when_every_session_has_outstanding_input() {
    let node = Node::start(RuntimeConfig {
        inference_latency: Duration::from_millis(50),
        max_batch_wait: Duration::from_millis(200),
        max_sessions_per_worker: 6,
        cache_slots_per_worker: 6,
        ..configuration()
    })
    .await
    .unwrap();
    for session_id in 0..6 {
        admit(&node, session_id).await;
    }
    let first_ingress = node.ingress.clone();
    let first = tokio::spawn(async move {
        let requests = (0..4).map(|session_id| {
            first_ingress.input_frame(
                SessionId(session_id),
                frame(
                    0,
                    PrefixState::default(),
                    Bytes::from_static(b"burst"),
                    Duration::from_millis(200),
                ),
            )
        });
        futures_util::future::join_all(requests).await
    });
    tokio::time::sleep(Duration::from_millis(5)).await;
    let remaining = tokio::join!(first_packet(&node, 4), first_packet(&node, 5));
    for outcome in [remaining.0, remaining.1] {
        let InputOutcome::Processed(output) = outcome else {
            panic!("expected partial successor");
        };
        assert!(output.audio.timings.device_queue >= Duration::from_millis(20));
    }
    for outcome in first.await.unwrap() {
        assert!(matches!(outcome.unwrap(), InputOutcome::Processed(_)));
    }
    let report = node.shutdown().await.unwrap();
    assert_eq!(report.inference.batches, 2);
    assert_eq!(report.inference.queued_batch_launches, 1);
}

#[tokio::test]
async fn batches_are_prepared_and_submitted_while_the_device_is_running() {
    let node = Node::start(RuntimeConfig {
        batch_size: 1,
        inference_latency: Duration::from_millis(20),
        device_queue_capacity: 2,
        admission_headroom: 1.0,
        batch_fill_reserve: 1.0,
        latency_safety_factor: 1.0,
        ..configuration()
    })
    .await
    .unwrap();
    for session_id in 0..3 {
        admit(&node, session_id).await;
    }
    let mut requests = Vec::new();
    for session_id in 0..3 {
        let ingress = node.ingress.clone();
        requests.push(tokio::spawn(async move {
            ingress
                .input_frame(
                    SessionId(session_id),
                    frame(
                        0,
                        PrefixState::default(),
                        Bytes::from_static(b"queued"),
                        Duration::from_millis(200),
                    ),
                )
                .await
                .unwrap()
        }));
    }
    for request in requests {
        assert!(matches!(request.await.unwrap(), InputOutcome::Processed(_)));
    }
    let report = node.shutdown().await.unwrap();
    assert!(report.inference.prepared_while_running > 0);
    assert_eq!(report.inference.queued_batch_launches, 2);
    assert_eq!(report.workers[0].peak_device_jobs, 3);
    assert_eq!(report.inference.processed_frames, 3);
    assert!((report.inference_latency.mean_ms - 20.0).abs() < 0.02);
    assert!(report.profile.device_queue.max_ms >= 39.0);
}

#[tokio::test]
async fn cancellation_invalidates_work_submitted_while_the_device_is_running() {
    let node = Node::start(RuntimeConfig {
        batch_size: 1,
        inference_latency: Duration::from_millis(30),
        device_queue_capacity: 2,
        admission_headroom: 1.0,
        batch_fill_reserve: 1.0,
        latency_safety_factor: 1.0,
        ..configuration()
    })
    .await
    .unwrap();
    admit(&node, 1).await;
    admit(&node, 2).await;
    let first_ingress = node.ingress.clone();
    let first = tokio::spawn(async move {
        first_ingress
            .input_frame(
                SessionId(1),
                frame(
                    0,
                    PrefixState::default(),
                    Bytes::from_static(b"first"),
                    Duration::from_millis(200),
                ),
            )
            .await
            .unwrap()
    });
    tokio::time::sleep(Duration::from_millis(5)).await;
    let second_ingress = node.ingress.clone();
    let second = tokio::spawn(async move {
        second_ingress
            .input_frame(
                SessionId(2),
                frame(
                    0,
                    PrefixState::default(),
                    Bytes::from_static(b"cancelled"),
                    Duration::from_millis(200),
                ),
            )
            .await
            .unwrap()
    });
    tokio::time::sleep(Duration::from_millis(5)).await;
    node.ingress.close_session(SessionId(2)).await.unwrap();
    assert!(matches!(first.await.unwrap(), InputOutcome::Processed(_)));
    assert!(matches!(
        second.await.unwrap(),
        InputOutcome::Rejected(FrameRejection::Cancelled)
    ));
    let report = node.shutdown().await.unwrap();
    assert!(report.inference.processed_frames <= 2);
    assert_eq!(report.inference.stale_results, 1);
    assert_eq!(report.inference.delivered_frames, 1);
}

#[tokio::test]
async fn cancelled_device_work_retains_cache_until_physical_completion() {
    let node = Node::start(RuntimeConfig {
        cache_slots_per_worker: 1,
        inference_latency: Duration::from_millis(40),
        ..configuration()
    })
    .await
    .unwrap();
    admit(&node, 1).await;
    let ingress = node.ingress.clone();
    let pending = tokio::spawn(async move {
        ingress
            .input_frame(
                SessionId(1),
                frame(
                    0,
                    PrefixState::default(),
                    Bytes::from_static(b"running"),
                    Duration::from_millis(200),
                ),
            )
            .await
            .unwrap()
    });
    tokio::time::sleep(Duration::from_millis(5)).await;
    node.ingress.close_session(SessionId(1)).await.unwrap();
    assert_eq!(
        node.ingress.create_session(SessionId(2)).await.unwrap(),
        CreateOutcome::Rejected(CreateRejection::Capacity)
    );
    assert!(matches!(
        pending.await.unwrap(),
        InputOutcome::Rejected(FrameRejection::Cancelled)
    ));
    admit(&node, 2).await;
    assert!(matches!(
        first_packet(&node, 2).await,
        InputOutcome::Processed(_)
    ));
    let report = node.shutdown().await.unwrap();
    assert_eq!(report.workers[0].peak_retired_cache_slots, 1);
    assert_eq!(report.inference.stale_results, 1);
}

#[tokio::test]
async fn cache_replay_that_cannot_join_an_urgent_batch_is_deferred_without_rejection() {
    let node = Node::start(RuntimeConfig {
        batch_size: 2,
        inference_latency: Duration::from_millis(12),
        replay_latency_per_packet: Duration::from_millis(100),
        max_batch_wait: Duration::from_millis(20),
        ..configuration()
    })
    .await
    .unwrap();
    admit(&node, 1).await;
    admit(&node, 2).await;
    let prefix_audio = Bytes::from_static(b"prefix");
    let first = node
        .ingress
        .input_frame(
            SessionId(2),
            frame(
                0,
                PrefixState::default(),
                prefix_audio.clone(),
                Duration::from_millis(200),
            ),
        )
        .await
        .unwrap();
    assert!(matches!(first, InputOutcome::Processed(_)));
    node.ingress.evict_cache(SessionId(2)).await.unwrap();
    let captured = Instant::now();
    let urgent = node.ingress.input_frame(
        SessionId(1),
        frame(
            0,
            PrefixState::default(),
            Bytes::from_static(b"urgent"),
            Duration::from_millis(35),
        ),
    );
    let replay = node.ingress.input_frame(
        SessionId(2),
        InputFrame {
            timestamp: captured,
            deadline: captured + Duration::from_millis(200),
            packet: AudioPacket {
                sequence: PacketSequence(1),
                payload: Bytes::from_static(b"next"),
                context: AudioContext::Replay(voice_scheduler::protocol::AudioPrefix(vec![
                    prefix_audio,
                ])),
            },
        },
    );
    let (urgent, replay) = tokio::join!(urgent, replay);
    assert!(matches!(urgent.unwrap(), InputOutcome::Processed(_)));
    assert!(matches!(replay.unwrap(), InputOutcome::Processed(_)));
    let report = node.shutdown().await.unwrap();
    assert_eq!(report.inference.rejected_frames, 0);
    assert_eq!(report.inference.replayed_packets, 1);
    assert_eq!(report.inference.delivered_frames, 3);
    assert!(report.inference_latency.max_ms >= 111.9);
}

#[tokio::test]
async fn cache_cannot_be_evicted_while_device_work_still_references_it() {
    let node = Node::start(RuntimeConfig {
        inference_latency: Duration::from_millis(40),
        ..configuration()
    })
    .await
    .unwrap();
    admit(&node, 1).await;
    let ingress = node.ingress.clone();
    let work = tokio::spawn(async move {
        ingress
            .input_frame(
                SessionId(1),
                frame(
                    0,
                    PrefixState::default(),
                    Bytes::from_static(b"audio"),
                    Duration::from_millis(200),
                ),
            )
            .await
            .unwrap()
    });
    tokio::time::sleep(Duration::from_millis(5)).await;
    assert!(!node.ingress.evict_cache(SessionId(1)).await.unwrap());
    let InputOutcome::Processed(output) = work.await.unwrap() else {
        panic!("expected audio");
    };
    assert!(node.ingress.evict_cache(SessionId(1)).await.unwrap());
    assert!(matches!(
        node.ingress
            .input_frame(
                SessionId(1),
                frame(
                    1,
                    output.audio.prefix,
                    Bytes::from_static(b"next"),
                    Duration::from_millis(200)
                ),
            )
            .await
            .unwrap(),
        InputOutcome::CacheMiss
    ));
    let report = node.shutdown().await.unwrap();
    assert_eq!(report.inference.cache_evictions, 1);
}
#[tokio::test]
async fn cache_eviction_requires_replaying_all_preceding_audio() {
    let node = Node::start(RuntimeConfig {
        replay_latency_per_packet: Duration::from_millis(5),
        ..configuration()
    })
    .await
    .unwrap();
    admit(&node, 1).await;
    let InputOutcome::Processed(first) = first_packet(&node, 1).await else {
        panic!("expected audio");
    };
    assert!(node.ingress.evict_cache(SessionId(1)).await.unwrap());
    let mut next = frame(
        1,
        first.audio.prefix,
        Bytes::from_static(b"next"),
        Duration::from_millis(200),
    );
    assert!(matches!(
        node.ingress
            .input_frame(
                SessionId(1),
                frame(
                    1,
                    first.audio.prefix,
                    Bytes::from_static(b"next"),
                    Duration::from_millis(200)
                )
            )
            .await
            .unwrap(),
        InputOutcome::CacheMiss
    ));
    next.packet.context = AudioContext::Replay(AudioPrefix(vec![Bytes::from_static(b"audio")]));
    let InputOutcome::Processed(output) =
        node.ingress.input_frame(SessionId(1), next).await.unwrap()
    else {
        panic!("expected replay echo");
    };
    assert_eq!(output.audio.prefix, first.audio.prefix.append(b"next"));
    assert_eq!(
        output.audio.cache,
        CacheOutcome::Replayed {
            packets: 1,
            bytes: 5
        }
    );
    let report = node.shutdown().await.unwrap();
    assert_eq!(report.inference.cache_misses, 1);
    assert_eq!(report.inference.replayed_packets, 1);
    assert_eq!(report.inference.replayed_bytes, 5);
    assert!(report.inference_latency.max_ms >= 7.0);
}
#[tokio::test]
async fn impossible_cache_recovery_is_rejected_before_requesting_prefix_upload() {
    let node = Node::start(RuntimeConfig {
        replay_latency_per_packet: Duration::from_millis(200),
        ..configuration()
    })
    .await
    .unwrap();
    admit(&node, 1).await;
    let InputOutcome::Processed(first) = first_packet(&node, 1).await else {
        panic!("expected prefix warmup");
    };
    assert!(node.ingress.evict_cache(SessionId(1)).await.unwrap());
    assert!(matches!(
        node.ingress
            .input_frame(
                SessionId(1),
                frame(
                    1,
                    first.audio.prefix,
                    Bytes::from_static(b"next"),
                    Duration::from_millis(50)
                ),
            )
            .await
            .unwrap(),
        InputOutcome::Rejected(FrameRejection::ReplayTooExpensive)
    ));
    let report = node.shutdown().await.unwrap();
    assert_eq!(report.inference.cache_misses, 1);
    assert_eq!(report.inference.batches, 1);
    assert_eq!(report.inference.replayed_packets, 0);
    assert_eq!(report.inference.deadline_misses, 0);
    assert_eq!(report.active_sessions_at_shutdown, 0);
    assert_eq!(report.terminated_sessions, 1);
}

#[tokio::test]
async fn expensive_prefix_replay_fails_instead_of_missing_realtime() {
    let node = Node::start(RuntimeConfig {
        replay_latency_per_packet: Duration::from_millis(200),
        ..configuration()
    })
    .await
    .unwrap();
    admit(&node, 1).await;
    first_packet(&node, 1).await;
    node.ingress.evict_cache(SessionId(1)).await.unwrap();
    let mut replay = frame(
        1,
        PrefixState::default(),
        Bytes::from_static(b"next"),
        Duration::from_millis(50),
    );
    replay.packet.context = AudioContext::Replay(AudioPrefix(vec![Bytes::from_static(b"audio")]));
    assert!(matches!(
        node.ingress
            .input_frame(SessionId(1), replay)
            .await
            .unwrap(),
        InputOutcome::Rejected(FrameRejection::ReplayTooExpensive)
    ));
    assert!(matches!(
        first_packet(&node, 1).await,
        InputOutcome::Rejected(FrameRejection::UnknownSession)
    ));
    let report = node.shutdown().await.unwrap();
    assert_eq!(report.inference.deadline_misses, 0);
}
#[tokio::test]
async fn missing_or_corrupt_audio_prefix_terminates_session() {
    let node = Node::start(configuration()).await.unwrap();
    admit(&node, 1).await;
    assert!(matches!(
        node.ingress
            .input_frame(
                SessionId(1),
                frame(
                    1,
                    PrefixState::default(),
                    Bytes::new(),
                    Duration::from_millis(100)
                )
            )
            .await
            .unwrap(),
        InputOutcome::Rejected(FrameRejection::InvalidSequence)
    ));
    admit(&node, 1).await;
    let invalid = PrefixState {
        digest: [1; 32],
        ..PrefixState::default()
    };
    assert!(matches!(
        node.ingress
            .input_frame(
                SessionId(1),
                frame(0, invalid, Bytes::new(), Duration::from_millis(100))
            )
            .await
            .unwrap(),
        InputOutcome::Rejected(FrameRejection::InvalidPrefix)
    ));
    let report = node.shutdown().await.unwrap();
    assert_eq!(report.inference.prefix_rejections, 2);
}
#[tokio::test]
async fn cancelled_inflight_results_cannot_reach_recreated_session() {
    let node = Node::start(RuntimeConfig {
        minimum_packet_interval: Duration::from_millis(300),
        packet_deadline: Duration::from_millis(300),
        inference_latency: Duration::from_millis(50),
        ..configuration()
    })
    .await
    .unwrap();
    let old = admit(&node, 1).await;
    let ingress = node.ingress.clone();
    let work = tokio::spawn(async move {
        ingress
            .input_frame(
                SessionId(1),
                frame(
                    0,
                    PrefixState::default(),
                    Bytes::from_static(b"old"),
                    Duration::from_millis(250),
                ),
            )
            .await
            .unwrap()
    });
    tokio::time::sleep(Duration::from_millis(5)).await;
    node.ingress.close_session(SessionId(1)).await.unwrap();
    let replacement = admit(&node, 1).await;
    assert_ne!(old.generation, replacement.generation);
    assert!(matches!(
        work.await.unwrap(),
        InputOutcome::Rejected(FrameRejection::Cancelled)
    ));
    let InputOutcome::Processed(output) = first_packet(&node, 1).await else {
        panic!("expected replacement audio");
    };
    assert_eq!(output.audio.assignment, replacement);
    assert_eq!(node.shutdown().await.unwrap().inference.stale_results, 1);
}
#[tokio::test]
async fn oversending_terminates_session_without_coalescing_audio() {
    let node = Node::start(RuntimeConfig {
        minimum_packet_interval: Duration::from_millis(300),
        packet_deadline: Duration::from_millis(300),
        inference_latency: Duration::from_millis(50),
        ..configuration()
    })
    .await
    .unwrap();
    admit(&node, 1).await;
    let ingress = node.ingress.clone();
    let first = tokio::spawn(async move {
        ingress
            .input_frame(
                SessionId(1),
                frame(
                    0,
                    PrefixState::default(),
                    Bytes::from_static(b"first"),
                    Duration::from_millis(250),
                ),
            )
            .await
            .unwrap()
    });
    tokio::time::sleep(Duration::from_millis(5)).await;
    assert!(matches!(
        first_packet(&node, 1).await,
        InputOutcome::Rejected(FrameRejection::Overloaded)
    ));
    assert!(matches!(
        first.await.unwrap(),
        InputOutcome::Rejected(FrameRejection::Cancelled)
    ));
    let report = node.shutdown().await.unwrap();
    assert_eq!(report.inference.busy_rejections, 1);
    assert_eq!(report.inference.delivered_frames, 0);
}
#[tokio::test]
async fn configured_slow_work_is_rejected_before_launch_if_it_cannot_meet_the_packet_deadline() {
    let node = Node::start(RuntimeConfig {
        slowdown: Some(WorkerSlowdown {
            after: Duration::from_millis(30),
            inference_latency: Duration::from_millis(100),
        }),
        ..configuration()
    })
    .await
    .unwrap();
    admit(&node, 1).await;
    tokio::time::sleep(Duration::from_millis(40)).await;
    let outcome = node
        .ingress
        .input_frame(
            SessionId(1),
            frame(
                0,
                PrefixState::default(),
                Bytes::from_static(b"too slow"),
                Duration::from_millis(50),
            ),
        )
        .await
        .unwrap();
    assert!(matches!(
        outcome,
        InputOutcome::Rejected(FrameRejection::DeadlineExceeded)
    ));
    let report = node.shutdown().await.unwrap();
    assert_eq!(report.inference.processed_frames, 0);
    assert_eq!(report.inference.deadline_misses, 0);
    assert_eq!(report.inference.rejected_frames, 1);
}

#[tokio::test]
async fn observed_slowdown_reduces_admission_and_sheds_unsustainable_sessions() {
    let node = Node::start(RuntimeConfig {
        minimum_packet_interval: Duration::from_millis(100),
        packet_deadline: Duration::from_millis(250),
        max_sessions_per_worker: 6,
        cache_slots_per_worker: 6,
        slowdown: Some(WorkerSlowdown {
            after: Duration::from_millis(30),
            inference_latency: Duration::from_millis(100),
        }),
        ..configuration()
    })
    .await
    .unwrap();
    for session_id in 0..6 {
        admit(&node, session_id).await;
    }
    tokio::time::sleep(Duration::from_millis(40)).await;
    first_packet(&node, 0).await;
    assert_eq!(
        node.ingress.create_session(SessionId(99)).await.unwrap(),
        CreateOutcome::Rejected(CreateRejection::Capacity)
    );
    let report = node.shutdown().await.unwrap();
    assert_eq!(report.workers[0].final_session_limit, 0);
    assert_eq!(report.inference.capacity_terminations, 6);
}
#[tokio::test]
async fn bounded_mailboxes_reject_frames_and_close_their_sessions() {
    let node = Node::start(RuntimeConfig {
        worker_channel_capacity: 1,
        ingress_capacity: 2,
        worker_input_delay: Duration::from_millis(50),
        minimum_packet_interval: Duration::from_millis(300),
        packet_deadline: Duration::from_millis(300),
        ..configuration()
    })
    .await
    .unwrap();
    admit(&node, 1).await;
    let mut producers = Vec::new();
    for _ in 0..20 {
        let ingress = node.ingress.clone();
        producers.push(tokio::spawn(async move {
            ingress
                .input_frame(
                    SessionId(1),
                    frame(
                        0,
                        PrefixState::default(),
                        Bytes::new(),
                        Duration::from_millis(250),
                    ),
                )
                .await
                .unwrap()
        }));
    }
    let mut rejected = 0;
    for producer in producers {
        if matches!(producer.await.unwrap(), InputOutcome::Rejected(_)) {
            rejected += 1;
        }
    }
    assert!(rejected > 0);
    let report = node.shutdown().await.unwrap();
    assert!(report.channel_saturation_events > 0);
    assert_eq!(report.active_sessions_at_shutdown, 0);
}
#[tokio::test]
async fn prefix_memory_limits_fail_explicitly_and_release_cache_slots() {
    let node = Node::start(RuntimeConfig {
        audio_limits: AudioLimits {
            max_prefix_packets: 1,
            ..AudioLimits::default()
        },
        ..configuration()
    })
    .await
    .unwrap();
    admit(&node, 1).await;
    let InputOutcome::Processed(output) = first_packet(&node, 1).await else {
        panic!("expected first");
    };
    assert!(matches!(
        node.ingress
            .input_frame(
                SessionId(1),
                frame(
                    1,
                    output.audio.prefix,
                    Bytes::new(),
                    Duration::from_millis(100)
                )
            )
            .await
            .unwrap(),
        InputOutcome::Rejected(FrameRejection::PrefixCapacity)
    ));
    admit(&node, 2).await;
    node.shutdown().await.unwrap();
}
#[tokio::test]
async fn idle_timeout_releases_session_and_cache() {
    let node = Node::start(RuntimeConfig {
        session_timeout: Duration::from_millis(20),
        ..configuration()
    })
    .await
    .unwrap();
    admit(&node, 1).await;
    tokio::time::sleep(Duration::from_millis(80)).await;
    assert!(matches!(
        first_packet(&node, 1).await,
        InputOutcome::Rejected(FrameRejection::UnknownSession)
    ));
    assert_eq!(node.shutdown().await.unwrap().timed_out_sessions, 1);
}
#[tokio::test]
async fn two_thousand_sessions_use_eight_device_workers_with_sticky_echoes() {
    let node = Node::start(RuntimeConfig {
        workers: 8,
        minimum_packet_interval: Duration::from_secs(2),
        packet_deadline: Duration::from_secs(2),
        max_sessions_per_worker: 256,
        cache_slots_per_worker: 256,
        batch_size: 32,
        ingress_capacity: 4096,
        worker_channel_capacity: 1024,
        inference_latency: Duration::from_millis(1),
        max_batch_wait: Duration::from_millis(10),
        ..configuration()
    })
    .await
    .unwrap();
    let mut assignments = Vec::new();
    for session_id in 0..2000 {
        assignments.push(admit(&node, session_id).await);
    }
    let mut clients = Vec::new();
    for session_id in 0..2000 {
        let ingress = node.ingress.clone();
        clients.push(tokio::spawn(async move {
            ingress
                .input_frame(
                    SessionId(session_id),
                    frame(
                        0,
                        PrefixState::default(),
                        Bytes::from_static(b"audio"),
                        Duration::from_secs(1),
                    ),
                )
                .await
                .unwrap()
        }));
    }
    for (index, client) in clients.into_iter().enumerate() {
        let InputOutcome::Processed(output) = client.await.unwrap() else {
            panic!("expected stress echo");
        };
        assert_eq!(output.audio.assignment, assignments[index]);
    }
    let report = node.shutdown().await.unwrap();
    assert_eq!(report.peak_active_sessions, 2000);
    assert_eq!(report.inference.delivered_frames, 2000);
    assert_eq!(report.workers.len(), 8);
    assert!(report.mean_batch_size > 1.0);
    assert!(report.batch_fill_ratio <= 1.0);
}
#[tokio::test]
async fn old_lease_cannot_send_close_or_evict_a_recreated_session() {
    let node = Node::start(configuration()).await.unwrap();
    let old = admit(&node, 1).await;
    let lease = SessionLease {
        session_id: SessionId(1),
        generation: old.generation,
    };
    node.ingress.close_session(lease).await.unwrap();
    let current = admit(&node, 1).await;
    assert!(matches!(
        node.ingress
            .input_frame(
                lease,
                frame(
                    0,
                    PrefixState::default(),
                    Bytes::from_static(b"stale"),
                    Duration::from_millis(200)
                )
            )
            .await
            .unwrap(),
        InputOutcome::Rejected(FrameRejection::Cancelled)
    ));
    assert!(!node.ingress.close_session(lease).await.unwrap());
    assert!(!node.ingress.evict_cache(lease).await.unwrap());
    let InputOutcome::Processed(output) = first_packet(&node, 1).await else {
        panic!("replacement must remain active");
    };
    assert_eq!(output.audio.assignment, current);
    assert_eq!(output.audio.cache, CacheOutcome::Hit);
    node.shutdown().await.unwrap();
}
#[tokio::test]
async fn partial_batch_runs_immediately_when_all_assigned_sessions_are_ready() {
    let node = Node::start(RuntimeConfig {
        max_sessions_per_worker: 1,
        cache_slots_per_worker: 1,
        batch_size: 32,
        max_batch_wait: Duration::from_millis(200),
        packet_deadline: Duration::from_millis(500),
        ..configuration()
    })
    .await
    .unwrap();
    admit(&node, 1).await;
    assert!(matches!(
        first_packet(&node, 1).await,
        InputOutcome::Processed(_)
    ));
    let report = node.shutdown().await.unwrap();
    assert_eq!(report.mean_batch_size, 1.0);
    assert!(report.queue_delay.max_ms < 150.0);
}
#[tokio::test]
async fn input_and_configuration_boundaries_fail_clearly() {
    assert!(
        Node::start(RuntimeConfig {
            workers: 0,
            ..configuration()
        })
        .await
        .is_err()
    );
    let node = Node::start(configuration()).await.unwrap();
    admit(&node, 1).await;
    let invalid = frame(
        0,
        PrefixState::default(),
        Bytes::from(vec![0; 4097]),
        Duration::from_millis(100),
    );
    assert!(matches!(
        node.ingress.input_frame(SessionId(1), invalid).await,
        Err(RuntimeError::InvalidFrame(_))
    ));
    let now = Instant::now();
    let invalid = InputFrame {
        timestamp: now + Duration::from_secs(1),
        deadline: now + Duration::from_secs(1),
        packet: AudioPacket {
            sequence: PacketSequence(0),
            payload: Bytes::new(),
            context: AudioContext::Cached(PrefixState::default()),
        },
    };
    assert!(matches!(
        node.ingress.input_frame(SessionId(1), invalid).await,
        Err(RuntimeError::InvalidFrame(_))
    ));
    node.shutdown().await.unwrap();
}
