//! Ordinary WebSocket clients exercising capture, multi-turn generation and churn.

mod config;
mod measurements;
mod report;
mod session;

use std::sync::Arc;

use bytes::Bytes;
use tokio::{task::JoinSet, time::Instant};

pub use config::SimulationConfig;
pub use report::{SessionSummary, SimulationReport};

#[derive(Debug, thiserror::Error)]
pub enum SimulationError {
    #[error("invalid workload configuration: {0}")]
    Configuration(&'static str),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Task(#[from] tokio::task::JoinError),
}

pub async fn run(
    url: &str,
    configuration: SimulationConfig,
) -> Result<SimulationReport, SimulationError> {
    configuration.validate()?;
    let audio = load_audio(&configuration).await?;
    let configuration = Arc::new(configuration);
    let url: Arc<str> = Arc::from(url);
    let started = Instant::now();
    let mut ttft = measurements::histogram();
    let mut gaps = measurements::histogram();
    let mut sessions = Vec::with_capacity(configuration.sessions * configuration.churn_rounds);
    for round in 0..configuration.churn_rounds {
        let mut clients = JoinSet::new();
        for index in 0..configuration.sessions {
            clients.spawn(session::run(
                url.clone(),
                configuration.clone(),
                audio.clone(),
                index,
                round,
            ));
        }
        while let Some(result) = clients.join_next().await {
            let measurement = result?;
            ttft.add(&measurement.ttft)
                .expect("matching histogram precision");
            gaps.add(&measurement.token_gaps)
                .expect("matching histogram precision");
            sessions.push(measurement.summary);
        }
    }
    sessions.sort_by(|left, right| left.session_id.cmp(&right.session_id));
    let elapsed_seconds = started.elapsed().as_secs_f64();
    let received_tokens = sessions.iter().map(|session| session.received_tokens).sum();
    Ok(SimulationReport {
        elapsed_seconds,
        offered_sessions: sessions.len(),
        admitted_sessions: sessions.iter().filter(|session| session.admitted).count(),
        rejected_sessions: sessions.iter().filter(|session| session.rejected).count(),
        failed_sessions: sessions.iter().filter(|session| session.failed).count(),
        admitted_turns: sessions.iter().map(|session| session.admitted_turns).sum(),
        completed_turns: sessions.iter().map(|session| session.completed_turns).sum(),
        token_limited_turns: sessions
            .iter()
            .map(|session| session.token_limited_turns)
            .sum(),
        rejected_turns: sessions.iter().map(|session| session.rejected_turns).sum(),
        interrupted_turns: sessions
            .iter()
            .map(|session| session.interrupted_turns)
            .sum(),
        received_tokens,
        aggregate_tokens_per_second: received_tokens as f64 / elapsed_seconds,
        sessions_with_rolling_rate_violations: sessions
            .iter()
            .filter(|session| session.rolling_rate_violations > 0)
            .count(),
        token_gaps_over_target: sessions
            .iter()
            .map(|session| session.token_gaps_over_target)
            .sum(),
        time_to_first_token_ms: measurements::distribution(&ttft),
        token_gap_ms: measurements::distribution(&gaps),
        sessions,
    })
}

async fn load_audio(configuration: &SimulationConfig) -> Result<Bytes, SimulationError> {
    if let Some(path) = &configuration.audio_file {
        let metadata = tokio::fs::metadata(path).await?;
        if metadata.len() == 0 || metadata.len() > 960_000 || metadata.len() % 2 != 0 {
            return Err(SimulationError::Configuration(
                "audio file must contain at most 30 s of nonempty mono PCM16 at 16 kHz",
            ));
        }
        return Ok(Bytes::from(tokio::fs::read(path).await?));
    }
    Ok(Bytes::from(vec![
        0;
        configuration.utterance_ms as usize * 32
    ]))
}
