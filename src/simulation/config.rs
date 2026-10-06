use super::{PacketQualityPolicy, SimulationError};
use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(Clone, Copy, Debug, Serialize, Deserialize, clap::ValueEnum)]
pub enum ArrivalPhase {
    Aligned,
    Random,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SimulationConfig {
    pub sessions: usize,
    pub duration: Duration,
    pub minimum_interval: Duration,
    pub maximum_interval: Duration,
    pub payload_bytes: usize,
    pub phase: ArrivalPhase,
    pub seed: u64,
    pub evict_every: Option<u64>,
    pub churn_after: Option<Duration>,
    pub io_timeout: Duration,
    pub metric_channel_capacity: usize,
    pub quality: PacketQualityPolicy,
}
impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            sessions: 400,
            duration: Duration::from_secs(10),
            minimum_interval: Duration::from_millis(48),
            maximum_interval: Duration::from_millis(55),
            payload_bytes: 1600,
            phase: ArrivalPhase::Random,
            seed: 42,
            evict_every: None,
            churn_after: None,
            io_timeout: Duration::from_secs(5),
            metric_channel_capacity: 4096,
            quality: PacketQualityPolicy::default(),
        }
    }
}
impl SimulationConfig {
    pub fn validate(&self) -> Result<(), SimulationError> {
        if self.sessions == 0
            || self.sessions > 10_000
            || self.duration.is_zero()
            || self.duration > Duration::from_secs(86400)
            || self.minimum_interval < Duration::from_millis(1)
            || self.maximum_interval < self.minimum_interval
            || self.maximum_interval > Duration::from_secs(1)
            || self.payload_bytes == 0
            || self.payload_bytes > 4096
            || self.io_timeout.is_zero()
            || self.io_timeout > Duration::from_secs(86400)
            || self.evict_every == Some(0)
            || self.metric_channel_capacity == 0
            || self.metric_channel_capacity > 1_000_000
            || self.quality.window_packets == 0
            || self.quality.window_packets > 10_000
            || self.quality.miss_limit == 0
            || self.quality.miss_limit > self.quality.window_packets
            || self
                .churn_after
                .is_some_and(|duration| duration.is_zero() || duration > Duration::from_secs(86400))
        {
            return Err(SimulationError::Configuration(
                "invalid bounded workload configuration",
            ));
        }
        Ok(())
    }
}
