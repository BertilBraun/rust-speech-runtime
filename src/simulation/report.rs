use serde::Serialize;

use super::steady::{MeasuredTraffic, SteadyStateReport};
use crate::metrics::LatencyDistribution;

#[derive(Debug, Serialize)]
pub struct SessionSummary {
    pub session_id: String,
    pub worker_id: Option<usize>,
    pub admitted: bool,
    pub rejected: bool,
    pub failed: bool,
    pub error: Option<String>,
    pub admitted_turns: usize,
    pub completed_turns: usize,
    pub token_limited_turns: usize,
    pub rejected_turns: usize,
    pub interrupted_turns: usize,
    pub received_tokens: usize,
    pub token_gaps_over_target: usize,
    pub rolling_rate_violations: usize,
    pub minimum_rolling_tokens_per_second: Option<f64>,
    pub minimum_turn_tokens_per_second: Option<f64>,
    pub measured: Option<MeasuredTraffic>,
}

impl SessionSummary {
    pub(crate) fn new(session_id: String) -> Self {
        Self {
            session_id,
            worker_id: None,
            admitted: false,
            rejected: false,
            failed: false,
            error: None,
            admitted_turns: 0,
            completed_turns: 0,
            token_limited_turns: 0,
            rejected_turns: 0,
            interrupted_turns: 0,
            received_tokens: 0,
            token_gaps_over_target: 0,
            rolling_rate_violations: 0,
            minimum_rolling_tokens_per_second: None,
            minimum_turn_tokens_per_second: None,
            measured: None,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct SimulationReport {
    pub prepare_before_commit: bool,
    pub endpointing_ms: u64,
    pub elapsed_seconds: f64,
    pub offered_sessions: usize,
    pub admitted_sessions: usize,
    pub rejected_sessions: usize,
    pub failed_sessions: usize,
    pub admitted_turns: usize,
    pub completed_turns: usize,
    pub token_limited_turns: usize,
    pub rejected_turns: usize,
    pub interrupted_turns: usize,
    pub received_tokens: usize,
    pub aggregate_tokens_per_second: f64,
    pub sessions_with_rolling_rate_violations: usize,
    pub token_gaps_over_target: usize,
    pub time_to_first_token_ms: LatencyDistribution,
    pub end_of_audio_to_first_token_ms: LatencyDistribution,
    pub endpoint_confirmation_ms: LatencyDistribution,
    pub token_gap_ms: LatencyDistribution,
    pub sessions: Vec<SessionSummary>,
    pub steady_state: Option<SteadyStateReport>,
}
