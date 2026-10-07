use super::*;
use crate::{
    config::RuntimeConfig,
    protocol::{SessionId, TurnId, TurnRecord, backend::AcceptedToken},
    session::state::{CaptureSnapshot, Preparation, PreparationRequest, SessionState, Stage},
};
use std::collections::HashMap;
use tokio::{sync::mpsc, time::Instant};
use tokio_util::sync::CancellationToken;

fn session(key: &str, queued_at: Instant, preparing: bool) -> SessionState {
    let (events, _) = mpsc::channel(4);
    let mut session = SessionState::new(
        SessionId(key.into()),
        0,
        "test".into(),
        events,
        CancellationToken::new(),
    );
    session.opened = true;
    session.record.turns.push(TurnRecord {
        turn_id: TurnId(1),
        audio_pcm16: vec![0; 3200],
        tokens: Vec::new(),
        finish_reason: None,
        committed: !preparing,
    });
    if preparing {
        session.stage = Stage::Capturing { chunks: 1 };
        session.preparation = Preparation::Queued(PreparationRequest {
            snapshot: CaptureSnapshot {
                turn_id: TurnId(1),
                generation: 0,
                chunk_count: 1,
                sample_count: 1600,
            },
            queued_at,
        });
    } else {
        session.stage = Stage::Prefill { queued_at };
    }
    session
}

#[test]
fn overlapping_preparations_and_commits_select_homogeneous_oldest_class() {
    for oldest_preparing in [true, false] {
        let now = Instant::now();
        let mut sessions = HashMap::from([
            ("oldest".into(), session("oldest", now, oldest_preparing)),
            (
                "other-class".into(),
                session(
                    "other-class",
                    now + std::time::Duration::from_millis(1),
                    !oldest_preparing,
                ),
            ),
            (
                "same-class".into(),
                session(
                    "same-class",
                    now + std::time::Duration::from_millis(2),
                    oldest_preparing,
                ),
            ),
        ]);
        let costs = CostModel::new(100.0);
        let selected = select_batch(
            &sessions,
            16,
            4,
            0,
            &costs,
            std::time::Duration::from_millis(100),
        )
        .unwrap();
        assert_eq!(
            selected,
            (
                BatchKind::Prefill,
                vec!["oldest".into(), "same-class".into()]
            )
        );
        sessions.remove("oldest");
        sessions.remove("same-class");
        assert_eq!(
            select_batch(
                &sessions,
                16,
                4,
                0,
                &costs,
                std::time::Duration::from_millis(100)
            )
            .unwrap(),
            (BatchKind::Prefill, vec!["other-class".into()])
        );
    }
}

#[test]
fn prefill_batch_size_uses_measured_shape_and_decode_deadline_budget() {
    let now = Instant::now();
    let mut generator = session("generator", now, false);
    generator.stage = Stage::Generating {
        last_token_at: now,
        deadline: now + std::time::Duration::from_millis(120),
    };
    generator.pending_token = Some(AcceptedToken {
        turn_id: 1,
        index: 0,
        token_id: 100,
    });
    let sessions = HashMap::from([
        ("first".into(), session("first", now, false)),
        ("second".into(), session("second", now, false)),
        ("generator".into(), generator),
    ]);
    let mut costs = CostModel::new(100.0);
    costs.observe(BatchKind::Prefill, 20.0, 1, 33);
    costs.observe(BatchKind::Prefill, 300.0, 2, 33);
    costs.observe(BatchKind::Decode, 30.0, 1, 0);
    let selected = select_batch(
        &sessions,
        16,
        4,
        0,
        &costs,
        std::time::Duration::from_millis(100),
    )
    .unwrap();
    assert_eq!(selected.0, BatchKind::Prefill);
    assert_eq!(selected.1.len(), 1);
}

#[test]
fn prepared_activation_runs_before_another_model_forward() {
    let now = Instant::now();
    let mut ready = session("ready", now, true);
    let snapshot = ready.preparation.snapshot().unwrap();
    ready.preparation = Preparation::Ready(snapshot);
    ready.stage = Stage::Prefill { queued_at: now };
    let sessions = HashMap::from([
        ("ready".into(), ready),
        ("ordinary".into(), session("ordinary", now, false)),
    ]);
    assert_eq!(
        select_batch(
            &sessions,
            16,
            4,
            0,
            &CostModel::new(100.0),
            std::time::Duration::from_millis(100)
        )
        .unwrap(),
        (BatchKind::Activate, vec!["ready".into()])
    );
}

#[test]
fn unknown_shape_does_not_assume_linear_batch_scaling() {
    let mut costs = CostModel::new(100.0);
    let config = RuntimeConfig::default();
    assert!(costs.admits(1, 16, 32, &config, true));
    assert!(!costs.admits(2, 16, 32, &config, true));
    costs.observe(BatchKind::Decode, 5.0, 1, 32);
    costs.observe(BatchKind::Prefill, 10.0, 1, 32);
    assert_eq!(costs.estimate(BatchKind::Decode, 8, 32), 100.0);
    assert!(costs.admits(16, 16, 32, &config, true));
    costs.observe(BatchKind::Decode, 120.0, 8, 32);
    assert!(!costs.admits(16, 16, 32, &config, true));
}

#[test]
fn context_costs_and_slow_recent_batches_remain_conservative() {
    let mut costs = CostModel::new(100.0);
    costs.observe(BatchKind::Decode, 10.0, 4, 32);
    costs.observe(BatchKind::Decode, 50.0, 4, 1024);
    costs.observe(BatchKind::Decode, 1.0, 4, 32);
    assert_eq!(costs.estimate(BatchKind::Decode, 4, 32), 11.0);
    assert_eq!(
        costs.estimate(BatchKind::Decode, 4, 1024),
        55.00000000000001
    );
}

#[test]
fn slow_prefill_does_not_permanently_reject_solo_turns() {
    let mut costs = CostModel::new(100.0);
    let config = RuntimeConfig::default();
    costs.observe(BatchKind::Prefill, 300.0, 1, 32);
    costs.observe(BatchKind::Decode, 20.0, 1, 32);
    assert!(costs.admits(1, 16, 32, &config, false));
    assert!(!costs.admits(2, 16, 32, &config, true));
}

#[test]
fn admission_uses_only_batch_shapes_that_fit_the_active_turn_count() {
    let mut costs = CostModel::new(100.0);
    let config = RuntimeConfig::default();
    costs.observe(BatchKind::Decode, 20.0, 1, 32);
    costs.observe(BatchKind::Decode, 300.0, 16, 32);
    costs.observe(BatchKind::Prefill, 10.0, 1, 32);
    assert!(costs.admits(1, 16, 32, &config, false));
    assert!(costs.admits(2, 16, 32, &config, true));
    assert!(!costs.admits(16, 16, 32, &config, false));
    assert!(!costs.admits(16, 8, 32, &config, false));
    assert_eq!(costs.estimate(BatchKind::Decode, 16, 32), 330.0);
}
