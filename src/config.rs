use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use thiserror::Error;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerConfig {
    pub endpoint: SocketAddr,
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn invalid_token_intervals_fail_at_configuration_boundary() {
        for rate in [f64::NAN, f64::INFINITY, 0.0, -1.0, 1e-300, 1e300] {
            let configuration = RuntimeConfig {
                target_tokens_per_second: rate,
                ..RuntimeConfig::default()
            };
            assert!(configuration.validate().is_err(), "rate {rate}");
        }
        assert!(RuntimeConfig::default().validate().is_ok());
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeConfig {
    pub workers: Vec<WorkerConfig>,
    pub max_sessions_per_worker: usize,
    pub max_active_turns_per_worker: usize,
    pub max_audio_samples: usize,
    pub max_history_bytes: usize,
    pub max_output_tokens: usize,
    pub max_context_tokens: usize,
    pub mailbox_capacity: usize,
    pub event_capacity: usize,
    pub max_batch_size: usize,
    pub target_tokens_per_second: f64,
    pub backend_timeout_ms: u64,
    pub admission_headroom: f64,
    pub initial_forward_estimate_ms: f64,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            workers: vec![WorkerConfig {
                endpoint: "127.0.0.1:9100".parse().expect("literal address"),
            }],
            max_sessions_per_worker: 64,
            max_active_turns_per_worker: 16,
            max_audio_samples: 480_000,
            max_history_bytes: 32 * 1024 * 1024,
            max_output_tokens: 512,
            max_context_tokens: 16_384,
            mailbox_capacity: 128,
            event_capacity: 128,
            max_batch_size: 16,
            target_tokens_per_second: 4.0,
            backend_timeout_ms: 120_000,
            admission_headroom: 0.8,
            initial_forward_estimate_ms: 100.0,
        }
    }
}

#[derive(Debug, Error)]
#[error("invalid runtime configuration: {0}")]
pub struct ConfigError(pub String);

impl RuntimeConfig {
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.workers.is_empty() {
            return Err(ConfigError(
                "at least one worker endpoint is required".into(),
            ));
        }
        for (name, value) in [
            ("max_sessions_per_worker", self.max_sessions_per_worker),
            (
                "max_active_turns_per_worker",
                self.max_active_turns_per_worker,
            ),
            ("max_audio_samples", self.max_audio_samples),
            ("max_history_bytes", self.max_history_bytes),
            ("max_output_tokens", self.max_output_tokens),
            ("max_context_tokens", self.max_context_tokens),
            ("mailbox_capacity", self.mailbox_capacity),
            ("event_capacity", self.event_capacity),
            ("max_batch_size", self.max_batch_size),
        ] {
            if value == 0 {
                return Err(ConfigError(format!("{name} must be positive")));
            }
        }
        if !self.target_tokens_per_second.is_finite()
            || self.target_tokens_per_second <= 0.0
            || !self.initial_forward_estimate_ms.is_finite()
            || self.initial_forward_estimate_ms <= 0.0
            || !self.admission_headroom.is_finite()
            || !(0.0..=1.0).contains(&self.admission_headroom)
            || self.admission_headroom == 0.0
            || self.backend_timeout_ms == 0
        {
            return Err(ConfigError("invalid throughput, timing or headroom".into()));
        }
        let interval = std::time::Duration::try_from_secs_f64(1.0 / self.target_tokens_per_second)
            .map_err(|_| ConfigError("token interval exceeds representable duration".into()))?;
        if interval.is_zero() {
            return Err(ConfigError(
                "token interval must be a nonzero representable duration".into(),
            ));
        }
        if self.max_active_turns_per_worker > self.max_sessions_per_worker {
            return Err(ConfigError(
                "active turn capacity exceeds session capacity".into(),
            ));
        }
        if self.max_audio_samples > usize::MAX / 2
            || self
                .workers
                .len()
                .checked_mul(self.max_sessions_per_worker)
                .is_none()
        {
            return Err(ConfigError("configured capacity overflows".into()));
        }
        let mut endpoints = self
            .workers
            .iter()
            .map(|worker| worker.endpoint)
            .collect::<Vec<_>>();
        endpoints.sort();
        endpoints.dedup();
        if endpoints.len() != self.workers.len() {
            return Err(ConfigError("worker endpoints must be unique".into()));
        }
        Ok(())
    }
}
