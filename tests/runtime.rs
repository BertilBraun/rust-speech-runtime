use std::time::Duration;

use bytes::Bytes;
use tokio::time::Instant;
use voice_scheduler::{
    Node, RuntimeError,
    config::{RuntimeConfig, WorkerSlowdown},
    protocol::{Assignment, CreateOutcome, InputFrame, InputOutcome, SessionId},
};

fn configuration() -> RuntimeConfig {
    RuntimeConfig {
        workers: 1,
        max_sessions_per_worker: 4,
        cache_slots_per_worker: 4,
        batch_size: 4,
        ..RuntimeConfig::default()
    }
}

fn frame() -> InputFrame {
    InputFrame {
        timestamp: Instant::now(),
        payload: Bytes::from_static(b"audio"),
    }
}

async fn admit(node: &Node, session_id: u64) -> Assignment {
    match node
        .ingress
        .create_session(SessionId(session_id))
        .await
        .unwrap()
    {
        CreateOutcome::Admitted(assignment) => assignment,
        outcome => panic!("expected admission, got {outcome:?}"),
    }
}

#[tokio::test(start_paused = true)]
async fn admission_and_cache_capacity_are_enforced_and_reclaimed() {
    let node = Node::start(RuntimeConfig {
        cache_slots_per_worker: 2,
        ..configuration()
    })
    .unwrap();
    let first = admit(&node, 1).await;
    admit(&node, 2).await;
    assert_eq!(
        node.ingress.create_session(SessionId(3)).await.unwrap(),
        CreateOutcome::RejectedCapacity
    );
    assert_eq!(
        node.ingress.create_session(SessionId(1)).await.unwrap(),
        CreateOutcome::AlreadyExists
    );
    assert!(node.ingress.close_session(SessionId(1)).await.unwrap());
    let replacement = admit(&node, 1).await;
    assert_eq!(first.worker_id, replacement.worker_id);
    assert!(replacement.generation.0 > first.generation.0);
    assert!(!node.ingress.close_session(SessionId(99)).await.unwrap());
    let report = node.shutdown().await.unwrap();
    assert_eq!(report.admitted_sessions, 3);
    assert_eq!(report.rejected_sessions, 1);
    assert_eq!(report.peak_active_sessions, 2);
}

#[tokio::test(start_paused = true)]
async fn two_thousand_sessions_remain_sticky_and_form_dynamic_batches() {
    let mut node = Node::start(RuntimeConfig {
        workers: 40,
        max_sessions_per_worker: 52,
        cache_slots_per_worker: 52,
        output_channel_capacity: 4096,
        result_channel_capacity: 4096,
        ..RuntimeConfig::default()
    })
    .unwrap();
    let mut outputs = node.take_outputs().unwrap();
    let mut assignments = Vec::new();
    for session_id in 0..2000 {
        assignments.push(admit(&node, session_id).await);
    }
    for session_id in 0..2000 {
        assert_eq!(
            node.ingress
                .input_frame(SessionId(session_id), frame())
                .await
                .unwrap(),
            InputOutcome::Accepted
        );
    }
    tokio::time::sleep(Duration::from_millis(60)).await;
    let mut completed = 0;
    while let Ok(output) = outputs.try_recv() {
        assert_eq!(output.assignment, assignments[output.session_id.0 as usize]);
        completed += 1;
    }
    assert_eq!(completed, 2000);
    for session_id in 0..40 {
        node.ingress
            .close_session(SessionId(session_id))
            .await
            .unwrap();
    }
    for session_id in 2000..2040 {
        admit(&node, session_id).await;
        node.ingress
            .input_frame(SessionId(session_id), frame())
            .await
            .unwrap();
    }
    tokio::time::sleep(Duration::from_millis(60)).await;
    let report = node.shutdown().await.unwrap();
    assert_eq!(report.peak_active_sessions, 2000);
    assert!(report.batches > 40);
    assert!(report.batch_fill_ratio > 0.5);
    assert!(
        report
            .workers
            .iter()
            .all(|worker| worker.peak_sessions <= 50)
    );
}

#[tokio::test(start_paused = true)]
async fn disconnect_during_inference_discards_old_generation_after_recreation() {
    let mut node = Node::start(RuntimeConfig {
        batch_size: 1,
        ..configuration()
    })
    .unwrap();
    let mut outputs = node.take_outputs().unwrap();
    let old = admit(&node, 1).await;
    node.ingress
        .input_frame(SessionId(1), frame())
        .await
        .unwrap();
    tokio::time::sleep(Duration::from_millis(1)).await;
    node.ingress.close_session(SessionId(1)).await.unwrap();
    let current = admit(&node, 1).await;
    assert_ne!(old.generation, current.generation);
    node.ingress
        .input_frame(SessionId(1), frame())
        .await
        .unwrap();
    tokio::time::sleep(Duration::from_millis(30)).await;
    let output = outputs.try_recv().unwrap();
    assert_eq!(output.assignment, current);
    assert!(outputs.try_recv().is_err());
    let report = node.shutdown().await.unwrap();
    assert_eq!(report.stale_results_discarded, 1);
    assert_eq!(report.delivered_results, 1);
}

#[tokio::test(start_paused = true)]
async fn bounded_worker_channel_exposes_overload_without_losing_close() {
    let node = Node::start(RuntimeConfig {
        worker_channel_capacity: 1,
        worker_control_delay: Duration::from_millis(20),
        max_input_age: Duration::from_secs(2),
        ..configuration()
    })
    .unwrap();
    admit(&node, 1).await;
    let mut overloaded = 0;
    for _ in 0..20 {
        if node
            .ingress
            .input_frame(SessionId(1), frame())
            .await
            .unwrap()
            == InputOutcome::Overloaded
        {
            overloaded += 1;
        }
    }
    assert!(overloaded > 0);
    assert!(node.ingress.close_session(SessionId(1)).await.unwrap());
    let report = node.shutdown().await.unwrap();
    assert!(report.worker_channel_saturation > 0);
    assert!(report.inputs_overloaded > 0);
    assert_eq!(report.active_sessions_at_shutdown, 0);
}

#[tokio::test(start_paused = true)]
async fn unconsumed_outputs_are_dropped_in_a_bounded_mailbox() {
    let node = Node::start(RuntimeConfig {
        output_channel_capacity: 1,
        batch_size: 1,
        ..configuration()
    })
    .unwrap();
    admit(&node, 1).await;
    for _ in 0..3 {
        node.ingress
            .input_frame(SessionId(1), frame())
            .await
            .unwrap();
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    let report = node.shutdown().await.unwrap();
    assert_eq!(report.delivered_results, 1);
    assert_eq!(report.output_channel_saturation, 2);
}

#[tokio::test(start_paused = true)]
async fn worker_slowdown_increases_deadline_misses() {
    async fn run(slowdown: Option<WorkerSlowdown>) -> voice_scheduler::metrics::Report {
        let node = Node::start(RuntimeConfig {
            workers: 1,
            max_sessions_per_worker: 64,
            cache_slots_per_worker: 64,
            slowdown,
            ..RuntimeConfig::default()
        })
        .unwrap();
        for session_id in 0..64 {
            admit(&node, session_id).await;
        }
        for _ in 0..5 {
            for session_id in 0..64 {
                node.ingress
                    .input_frame(SessionId(session_id), frame())
                    .await
                    .unwrap();
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        node.shutdown().await.unwrap()
    }
    let normal = run(None).await;
    let slow = run(Some(WorkerSlowdown {
        after: Duration::from_millis(50),
        inference_latency: Duration::from_millis(20),
    }))
    .await;
    assert!(slow.deadline_misses > normal.deadline_misses);
    assert!(slow.inference_latency.p95_ms > normal.inference_latency.p95_ms);
    assert!(slow.coalesced_inputs > 0);
}

#[tokio::test(start_paused = true)]
async fn idle_sessions_expire_and_capacity_is_reusable() {
    let node = Node::start(RuntimeConfig {
        session_timeout: Duration::from_millis(100),
        ..configuration()
    })
    .unwrap();
    admit(&node, 1).await;
    tokio::time::sleep(Duration::from_millis(160)).await;
    assert_eq!(
        node.ingress
            .input_frame(SessionId(1), frame())
            .await
            .unwrap(),
        InputOutcome::UnknownSession
    );
    admit(&node, 1).await;
    let report = node.shutdown().await.unwrap();
    assert_eq!(report.timed_out_sessions, 1);
}

#[tokio::test(start_paused = true)]
async fn input_boundaries_reject_invalid_and_stale_frames() {
    let node = Node::start(configuration()).unwrap();
    admit(&node, 1).await;
    let timestamp = Instant::now();
    tokio::time::sleep(Duration::from_millis(110)).await;
    assert_eq!(
        node.ingress
            .input_frame(
                SessionId(1),
                InputFrame {
                    timestamp,
                    payload: Bytes::new()
                }
            )
            .await
            .unwrap(),
        InputOutcome::Stale
    );
    assert!(matches!(
        node.ingress
            .input_frame(
                SessionId(1),
                InputFrame {
                    timestamp: Instant::now(),
                    payload: Bytes::from(vec![0; 4097])
                }
            )
            .await,
        Err(RuntimeError::InvalidFrame(_))
    ));
    assert!(matches!(
        node.ingress
            .input_frame(
                SessionId(1),
                InputFrame {
                    timestamp: Instant::now() + Duration::from_secs(1),
                    payload: Bytes::new()
                }
            )
            .await,
        Err(RuntimeError::InvalidFrame(_))
    ));
    assert_eq!(node.shutdown().await.unwrap().inputs_stale, 1);
}

#[test]
fn invalid_configuration_fails_before_spawning_tasks() {
    assert!(
        Node::start(RuntimeConfig {
            workers: 0,
            ..configuration()
        })
        .is_err()
    );
    assert!(
        Node::start(RuntimeConfig {
            phase_bucket: Some(Duration::ZERO),
            ..configuration()
        })
        .is_err()
    );
}
