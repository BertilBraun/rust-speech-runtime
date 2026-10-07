use super::*;
use crate::{
    config::RuntimeConfig,
    metrics::Metrics,
    protocol::{
        SessionId, TurnId, TurnRecord,
        backend::{Memory, Ready, Timing},
    },
    session::state::{SessionState, Stage},
    worker::Command,
};
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicU64, AtomicUsize},
};
use tokio::{
    sync::{mpsc, oneshot},
    time::Instant,
};
use tokio_util::sync::CancellationToken;

#[tokio::test]
async fn interrupted_prefill_cost_uses_original_audio_and_cache_prefix() {
    let configuration = RuntimeConfig::default();
    let ready = Ready {
        r#type: "ready".into(),
        protocol_version: 1,
        body_bytes: 0,
        model_id: "test-model".into(),
        max_context_tokens: configuration.max_context_tokens,
        max_batch_size: configuration.max_batch_size,
        max_audio_samples: configuration.max_audio_samples,
    };
    let metrics = Arc::new(Metrics::new(1));
    let mut worker = WorkerActor::new(
        0,
        configuration,
        ready,
        metrics.clone(),
        Arc::new(AtomicU64::new(1)),
        Arc::new(AtomicUsize::new(1)),
        Arc::new(AtomicBool::new(true)),
    );
    let (events, _receiver) = mpsc::channel(8);
    let mut session = SessionState::new(
        SessionId("interrupted-prefill".into()),
        0,
        "test-model".into(),
        events,
        CancellationToken::new(),
    );
    session.opened = true;
    session.context_tokens = 100;
    session.record.turns.push(TurnRecord {
        turn_id: TurnId(1),
        audio_pcm16: vec![0; 30 * 16_000 * 2],
        tokens: Vec::new(),
        finish_reason: None,
        committed: true,
    });
    session.stage = Stage::Prefill {
        queued_at: Instant::now(),
    };
    worker
        .sessions
        .insert("session-session_key".into(), session);
    let job = worker.build_batch(BatchKind::Prefill, vec!["session-session_key".into()]);
    let (reply, response) = oneshot::channel();
    worker.handle_command(Command::Begin {
        session_key: "session-session_key".into(),
        turn_id: TurnId(2),
        reply,
    });
    response.await.unwrap().unwrap();
    worker
        .apply_completion(BatchCompletion {
            elapsed_ms: 300.0,
            response: Ok(BatchResponse {
                request_id: job.request.request_id,
                body_bytes: 0,
                results: vec![OperationResult {
                    operation_id: job.request.operations[0].operation_id(),
                    session_id: "session-session_key".into(),
                    turn_id: Some(1),
                    generation: Some(0),
                    outcome: Outcome::Token {
                        token_id: 100,
                        text_delta: "stale".into(),
                        eos: false,
                        context_tokens: 412,
                    },
                }],
                timing: Timing::default(),
                memory: Memory::default(),
            }),
        })
        .unwrap();
    let session = &worker.sessions["session-session_key"];
    assert_eq!(session.context_tokens, 412);
    assert!(session.current_turn().unwrap().audio_pcm16.is_empty());
    assert!(session.current_turn().unwrap().tokens.is_empty());
    assert_eq!(metrics.snapshot().stale_results_discarded, 1);
    assert_eq!(
        worker.forward_costs.estimate(BatchKind::Prefill, 1, 432),
        330.0
    );
}
