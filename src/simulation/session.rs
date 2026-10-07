use std::{sync::Arc, time::Duration};

use bytes::Bytes;
use rand::{Rng, SeedableRng, rngs::SmallRng};
use tokio::{
    sync::{oneshot, watch},
    time::Instant,
};

use crate::{
    protocol::{ErrorCode, FinishReason, SessionEvent, SessionId, TurnId},
    transport::{GatewayError, VoiceClient},
};

use super::{
    SimulationConfig, WorkloadLength,
    measurements::{Measurements, TurnTiming},
    steady::MeasurementWindow,
};

pub(crate) async fn run(
    url: Arc<str>,
    configuration: Arc<SimulationConfig>,
    audio: Bytes,
    index: usize,
    round: usize,
    opened: oneshot::Sender<()>,
    measurement_window: watch::Receiver<Option<MeasurementWindow>>,
) -> Measurements {
    let session_id = format!("benchmark-{round}-{index}");
    let mut measurements = Measurements::new(session_id.clone());
    measurements.measurement_window = Some(measurement_window);
    let mut random = SmallRng::seed_from_u64(
        configuration
            .seed
            .wrapping_add((round * configuration.sessions + index) as u64),
    );
    let delay = random.random_range(0..=configuration.start_spread_ms);
    tokio::time::sleep(Duration::from_millis(delay)).await;
    if let Err(error) = conversation(
        &url,
        &configuration,
        audio,
        session_id,
        &mut random,
        &mut measurements,
        opened,
    )
    .await
    {
        let capacity_rejection = matches!(
            error,
            GatewayError::Rejected(ErrorCode::CapacityExceeded, _)
        );
        if capacity_rejection && measurements.summary.admitted {
            measurements.summary.rejected_turns += 1;
        } else {
            measurements.summary.rejected = capacity_rejection;
        }
        measurements.summary.failed = !capacity_rejection;
        measurements.summary.error = Some(error.to_string());
    }
    measurements
}

async fn conversation(
    url: &str,
    configuration: &SimulationConfig,
    audio: Bytes,
    session_id: String,
    random: &mut SmallRng,
    measurements: &mut Measurements,
    opened: oneshot::Sender<()>,
) -> Result<(), GatewayError> {
    let mut client = VoiceClient::connect(url, configuration.response_timeout).await?;
    let worker_id = client.open(SessionId(session_id)).await?;
    measurements.summary.worker_id = Some(worker_id);
    measurements.summary.admitted = true;
    let _ = opened.send(());
    let result = turns(&mut client, configuration, audio, random, measurements).await;
    let cleanup = client.close().await;
    result?;
    cleanup
}

async fn turns(
    client: &mut VoiceClient,
    configuration: &SimulationConfig,
    audio: Bytes,
    random: &mut SmallRng,
    measurements: &mut Measurements,
) -> Result<(), GatewayError> {
    let mut index = 0;
    while should_start_turn(configuration, index, measurements.window()) {
        let turn_id = TurnId(index as u64 + 1);
        let rejected = match client.begin_turn(turn_id).await {
            Ok(()) => {
                measurements.summary.admitted_turns += 1;
                run_turn(
                    client,
                    turn_id,
                    configuration,
                    audio.clone(),
                    random,
                    measurements,
                )
                .await?;
                false
            }
            Err(GatewayError::Rejected(ErrorCode::CapacityExceeded, _))
                if matches!(configuration.length, WorkloadLength::SteadyState { .. }) =>
            {
                measurements.reject_turn(Instant::now());
                true
            }
            Err(error) => return Err(error),
        };
        index += 1;
        if should_start_turn(configuration, index, measurements.window()) {
            let think_ms = if rejected {
                configuration.think_ms.max(100)
            } else {
                configuration.think_ms
            };
            tokio::time::sleep(Duration::from_millis(think_ms)).await;
        }
    }
    Ok(())
}

fn should_start_turn(
    configuration: &SimulationConfig,
    index: usize,
    window: Option<MeasurementWindow>,
) -> bool {
    match configuration.length {
        WorkloadLength::Turns(count) => index < count,
        WorkloadLength::SteadyState { .. } => {
            window.is_none_or(|window| Instant::now() < window.ends_at)
        }
    }
}

async fn run_turn(
    client: &mut VoiceClient,
    turn_id: TurnId,
    configuration: &SimulationConfig,
    audio: Bytes,
    random: &mut SmallRng,
    measurements: &mut Measurements,
) -> Result<(), GatewayError> {
    let chunks = send_audio(client, turn_id, audio.clone(), configuration, random).await?;
    let audio_finished = Instant::now();
    if configuration.prepare_before_commit {
        client.prepare(turn_id, chunks, audio.len() / 2).await?;
    }
    tokio::time::sleep_until(audio_finished + Duration::from_millis(configuration.endpointing_ms))
        .await;
    let committed = Instant::now();
    client.commit(turn_id, chunks, audio.len() / 2).await?;
    generate(
        client,
        turn_id,
        TurnTiming::new(audio_finished, committed),
        configuration,
        measurements,
    )
    .await?;
    Ok(())
}

async fn send_audio(
    client: &mut VoiceClient,
    turn_id: TurnId,
    audio: Bytes,
    configuration: &SimulationConfig,
    random: &mut SmallRng,
) -> Result<u32, GatewayError> {
    let mut offset = 0;
    let mut index = 0;
    while offset < audio.len() {
        let interval =
            random.random_range(configuration.minimum_packet_ms..=configuration.maximum_packet_ms);
        let packet_bytes = (interval as usize * 16 * 2).min(audio.len() - offset);
        tokio::time::sleep(Duration::from_millis(packet_bytes as u64 / 32)).await;
        client
            .audio(turn_id, index, audio.slice(offset..offset + packet_bytes))
            .await?;
        offset += packet_bytes;
        index += 1;
    }
    Ok(index)
}

async fn generate(
    client: &mut VoiceClient,
    turn_id: TurnId,
    mut timing: TurnTiming,
    configuration: &SimulationConfig,
    measurements: &mut Measurements,
) -> Result<(), GatewayError> {
    let deadline = timing.committed() + configuration.response_timeout;
    let mut received = 0;
    let mut previous_sequence = None;
    let mut interrupted = false;
    let mut observations = tokio::time::interval(Duration::from_millis(100));
    observations.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    loop {
        let event = tokio::select! {
            _ = tokio::time::sleep_until(deadline) => return Err(GatewayError::Timeout),
            _ = observations.tick() => {
                timing.observe_rate(
                    Instant::now(),
                    measurements,
                    configuration.target_tokens_per_second,
                    Duration::from_millis(configuration.throughput_window_ms),
                );
                continue;
            }
            event = client.next_event() => event?,
        };
        match event {
            SessionEvent::TextDelta {
                turn_id: observed,
                sequence,
                ..
            } if observed == turn_id => {
                if previous_sequence.is_some_and(|previous| sequence != previous + 1) {
                    return Err(GatewayError::Protocol("token sequence has a gap"));
                }
                previous_sequence = Some(sequence);
                received += 1;
                timing.token(
                    Instant::now(),
                    measurements,
                    configuration.target_tokens_per_second,
                );
                if !interrupted && configuration.interrupt_after_tokens == Some(received) {
                    client.cancel(turn_id).await?;
                    interrupted = true;
                }
            }
            SessionEvent::Finished {
                turn_id: observed,
                reason,
                ..
            } if observed == turn_id => {
                timing.finish(measurements, reason);
                match reason {
                    FinishReason::Eos => measurements.summary.completed_turns += 1,
                    FinishReason::TokenLimit => {
                        measurements.summary.completed_turns += 1;
                        measurements.summary.token_limited_turns += 1;
                    }
                    FinishReason::Cancelled if interrupted => {
                        measurements.summary.interrupted_turns += 1
                    }
                    failure => {
                        return Err(GatewayError::Rejected(
                            finish_error(failure),
                            format!("generation finished with {failure:?}"),
                        ));
                    }
                }
                return Ok(());
            }
            SessionEvent::Failed { code, message, .. } => {
                return Err(GatewayError::Rejected(code, message));
            }
            _ => {}
        }
    }
}

fn finish_error(reason: FinishReason) -> ErrorCode {
    match reason {
        FinishReason::ContextLimit => ErrorCode::ContextLimit,
        FinishReason::SlowConsumer => ErrorCode::SlowConsumer,
        FinishReason::BackendFailed => ErrorCode::BackendFailed,
        FinishReason::Cancelled => ErrorCode::InvalidState,
        FinishReason::Eos | FinishReason::TokenLimit => {
            unreachable!("successful finish handled by generation loop")
        }
    }
}
