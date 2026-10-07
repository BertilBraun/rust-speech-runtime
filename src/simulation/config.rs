use std::{path::PathBuf, time::Duration};

use super::SimulationError;

pub const DEFAULT_AUDIO_PACKET_MS: u64 = 100;

/// Fixed-turn tests or a timed measurement after every session-open attempt has finished.
#[derive(Clone, Copy, Debug)]
pub enum WorkloadLength {
    Turns(usize),
    SteadyState { measurement: Duration },
}

#[derive(Clone, Debug)]
pub struct SimulationConfig {
    pub sessions: usize,
    pub length: WorkloadLength,
    pub utterance_ms: u64,
    pub minimum_packet_ms: u64,
    pub maximum_packet_ms: u64,
    pub start_spread_ms: u64,
    pub think_ms: u64,
    pub endpointing_ms: u64,
    pub prepare_before_commit: bool,
    pub target_tokens_per_second: f64,
    pub throughput_window_ms: u64,
    pub response_timeout: Duration,
    pub interrupt_after_tokens: Option<usize>,
    pub churn_rounds: usize,
    pub audio_file: Option<PathBuf>,
    pub seed: u64,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            sessions: 8,
            length: WorkloadLength::Turns(2),
            utterance_ms: 1000,
            minimum_packet_ms: DEFAULT_AUDIO_PACKET_MS,
            maximum_packet_ms: DEFAULT_AUDIO_PACKET_MS,
            start_spread_ms: 10_000,
            think_ms: 250,
            endpointing_ms: 0,
            prepare_before_commit: false,
            target_tokens_per_second: 4.0,
            throughput_window_ms: 2000,
            response_timeout: Duration::from_secs(120),
            interrupt_after_tokens: None,
            churn_rounds: 1,
            audio_file: None,
            seed: 7,
        }
    }
}

impl SimulationConfig {
    pub fn validate(&self) -> Result<(), SimulationError> {
        if self.sessions == 0
            || self.sessions > 10_000
            || matches!(self.length, WorkloadLength::Turns(0))
            || self.churn_rounds == 0
        {
            return Err(SimulationError::Configuration(
                "sessions (1..10000), turns and rounds must be positive",
            ));
        }
        match self.length {
            WorkloadLength::SteadyState { measurement }
                if measurement.is_zero()
                    || measurement > Duration::from_secs(3600)
                    || self.churn_rounds != 1 =>
            {
                return Err(SimulationError::Configuration(
                    "steady measurement must be positive, at most 3600 seconds, and use one cohort",
                ));
            }
            _ => {}
        }
        if self
            .sessions
            .checked_mul(self.churn_rounds)
            .is_none_or(|total| total > 100_000)
        {
            return Err(SimulationError::Configuration(
                "total offered sessions must not exceed 100000",
            ));
        }
        if self.minimum_packet_ms == 0
            || self.maximum_packet_ms < self.minimum_packet_ms
            || self.maximum_packet_ms > 250
        {
            return Err(SimulationError::Configuration(
                "packet interval must be ordered and between 1 and 250 ms",
            ));
        }
        if self.utterance_ms == 0
            || self.utterance_ms > 30_000
            || self.throughput_window_ms == 0
            || self.response_timeout.is_zero()
        {
            return Err(SimulationError::Configuration(
                "utterance must be 1..30000 ms; window and timeout must be positive",
            ));
        }
        if !self.target_tokens_per_second.is_finite()
            || self.target_tokens_per_second <= 0.0
            || self.interrupt_after_tokens == Some(0)
        {
            return Err(SimulationError::Configuration(
                "throughput target and interruption token count must be positive",
            ));
        }
        Ok(())
    }
}
