use super::*;

pub(super) async fn prepare(client: &mut VoiceClient, turn_id: u64) {
    client
        .begin_turn(TurnId(turn_id))
        .await
        .expect("turn accepted");
    client
        .audio(TurnId(turn_id), 0, Bytes::from(vec![1; 1600]))
        .await
        .expect("audio delivered");
    client
        .prepare(TurnId(turn_id), 1, 800)
        .await
        .expect("preparation requested");
}

pub(super) async fn prepared(
    client: &mut VoiceClient,
    turn_id: u64,
    chunk_count: u32,
    sample_count: usize,
) {
    assert!(matches!(
        client.next_event().await.expect("preparation readiness"),
        SessionEvent::Prepared {
            turn_id: observed,
            chunk_count: observed_chunks,
            sample_count: observed_samples,
        } if observed == TurnId(turn_id)
            && observed_chunks == chunk_count
            && observed_samples == sample_count
    ));
}

#[tokio::test]
async fn prepared_turn_has_no_text_until_explicit_commit() {
    let backend = Fixture::start(2).await;
    let gateway = TestGateway::start(backend.config(), gateway_config()).await;
    let mut client = connected(&gateway.url, "provisional").await;
    prepare(&mut client, 1).await;
    prepared(&mut client, 1, 1, 800).await;
    assert!(
        timeout(Duration::from_millis(30), client.next_event())
            .await
            .is_err()
    );
    client.commit(TurnId(1), 1, 800).await.expect("commit sent");
    assert_eq!(finish(&mut client, 1).await.len(), 5);
    client.close().await.expect("clean close");
    let report = gateway.shutdown().await;
    assert_eq!(report.failed_connections, 0);
    assert_eq!(report.runtime.generated_tokens, 5);
}

async fn resumed_audio(reprepare: bool) {
    let backend = Fixture::start(2).await;
    let gateway = TestGateway::start(backend.config(), gateway_config()).await;
    let mut client = connected(&gateway.url, "resumed").await;
    prepare(&mut client, 1).await;
    prepared(&mut client, 1, 1, 800).await;
    client
        .audio(TurnId(1), 1, Bytes::from(vec![2; 1600]))
        .await
        .expect("speech resumes");
    if reprepare {
        client
            .prepare(TurnId(1), 2, 1600)
            .await
            .expect("new candidate");
        prepared(&mut client, 1, 2, 1600).await;
    }
    client
        .commit(TurnId(1), 2, 1600)
        .await
        .expect("updated commit");
    assert_eq!(finish(&mut client, 1).await.len(), 5);
    commit(&mut client, 2).await;
    assert_eq!(finish(&mut client, 2).await.len(), 5);
    client.close().await.expect("clean close");
    let report = gateway.shutdown().await;
    assert_eq!(report.failed_connections, 0);
    assert_eq!(report.runtime.generated_tokens, 10);
}

#[tokio::test]
async fn resumed_audio_uses_full_prefill_when_candidate_is_invalidated() {
    resumed_audio(false).await;
}

#[tokio::test]
async fn resumed_audio_can_prepare_a_new_candidate_before_confirmation() {
    resumed_audio(true).await;
}

#[tokio::test]
async fn immediate_commit_after_prepare_returns_exactly_one_response() {
    let backend = Fixture::start(30).await;
    let gateway = TestGateway::start(backend.config(), gateway_config()).await;
    let mut client = connected(&gateway.url, "commit-race").await;
    prepare(&mut client, 1).await;
    client
        .commit(TurnId(1), 1, 800)
        .await
        .expect("commit races preparation");
    assert_eq!(finish(&mut client, 1).await.len(), 5);
    client.close().await.expect("clean close");
    assert_eq!(gateway.shutdown().await.runtime.generated_tokens, 5);
}

#[tokio::test]
async fn cancelled_candidate_never_becomes_conversation_output() {
    let backend = Fixture::start(2).await;
    let gateway = TestGateway::start(backend.config(), gateway_config()).await;
    let mut client = connected(&gateway.url, "cancel-candidate").await;
    prepare(&mut client, 1).await;
    prepared(&mut client, 1, 1, 800).await;
    client.cancel(TurnId(1)).await.expect("cancel candidate");
    assert!(finish(&mut client, 1).await.is_empty());
    prepare(&mut client, 2).await;
    prepared(&mut client, 2, 1, 800).await;
    client
        .commit(TurnId(2), 1, 800)
        .await
        .expect("confirm next turn");
    assert_eq!(finish(&mut client, 2).await.len(), 5);
    client.close().await.expect("clean close");
    assert_eq!(gateway.shutdown().await.runtime.generated_tokens, 5);
}

#[tokio::test]
async fn prepare_counts_are_validated_without_poisoning_turn() {
    let backend = Fixture::start(2).await;
    let gateway = TestGateway::start(backend.config(), gateway_config()).await;
    let mut client = connected(&gateway.url, "invalid-prepare").await;
    client.begin_turn(TurnId(1)).await.expect("turn accepted");
    client
        .audio(TurnId(1), 0, Bytes::from(vec![0; 1600]))
        .await
        .expect("audio");
    client
        .prepare(TurnId(1), 2, 800)
        .await
        .expect("invalid counts sent");
    assert!(matches!(
        client.next_event().await.expect("failure"),
        SessionEvent::Failed {
            code: ErrorCode::InvalidInput,
            ..
        }
    ));
    client
        .prepare(TurnId(1), 1, 800)
        .await
        .expect("corrected counts sent");
    prepared(&mut client, 1, 1, 800).await;
    client.commit(TurnId(1), 1, 800).await.expect("commit sent");
    assert_eq!(finish(&mut client, 1).await.len(), 5);
    client.close().await.expect("clean close");
    gateway.shutdown().await;
}

#[tokio::test]
async fn benchmark_endpoint_confirmation_preserves_commit_based_latency() {
    let backend = Fixture::start(2).await;
    let gateway = TestGateway::start(backend.config(), gateway_config()).await;
    for prepare_before_commit in [false, true] {
        let report = simulation::run(
            &gateway.url,
            SimulationConfig {
                sessions: 1,
                length: voice_scheduler::simulation::WorkloadLength::Turns(2),
                utterance_ms: 50,
                start_spread_ms: 0,
                think_ms: 0,
                endpointing_ms: 100,
                prepare_before_commit,
                ..SimulationConfig::default()
            },
        )
        .await
        .expect("endpointing benchmark");
        assert_eq!(report.failed_sessions, 0, "{:?}", report.sessions);
        assert_eq!(report.completed_turns, 2);
        assert_eq!(report.time_to_first_token_ms.count, 2);
        assert_eq!(report.end_of_audio_to_first_token_ms.count, 2);
        assert_eq!(report.endpoint_confirmation_ms.count, 2);
        assert_eq!(report.endpointing_ms, 100);
        assert_eq!(report.prepare_before_commit, prepare_before_commit);
        assert!(report.endpoint_confirmation_ms.p50_ms >= 99.0);
        assert!(
            report.end_of_audio_to_first_token_ms.p50_ms
                >= report.time_to_first_token_ms.p50_ms + 99.0
        );
    }
    gateway.shutdown().await;
}
