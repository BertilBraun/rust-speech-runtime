//! Canonical scheduling limits, validated before any worker is started.

use serde::{Deserialize, Serialize};
use std::{net::SocketAddr, time::Duration};
use thiserror::Error;

/// Endpoint of one persistent, warmed model process with an exclusive device cache.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerConfig {
    pub endpoint: SocketAddr,
}

/// Node-wide scheduling and bounded-memory limits shared by the manager and workers.
///
/// Defaults support local experiments. Hardware capacity must be measured; session
/// slots and active-turn compute reservations are separate limits. Backend readiness
/// may lower model-facing batch, audio and context limits. JSON requires every field.
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
    pub max_prefill_batch_size: usize,
    /// Minimum desired generation rate after the first token, not an audio packet deadline.
    pub target_tokens_per_second: f64,
    pub backend_timeout_ms: u64,
    pub max_prefill_wait_ms: u64,
    /// Fraction of the estimated generation budget available to admitted work, in (0, 1].
    pub admission_headroom: f64,
    /// Conservative duration for unobserved batch/context shapes; never scaled linearly by fill.
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
            max_prefill_batch_size: 4,
            target_tokens_per_second: 4.0,
            backend_timeout_ms: 120_000,
            max_prefill_wait_ms: 100,
            admission_headroom: 0.8,
            initial_forward_estimate_ms: 100.0,
        }
    }
}

/// Configuration failure reported before task startup or resource allocation.
#[derive(Debug, Error)]
#[error("invalid runtime configuration: {0}")]
pub struct ConfigError(pub String);

impl RuntimeConfig {
    /// Maximum desired interval between accepted tokens, after validated startup.
    pub(crate) fn token_interval(&self) -> Duration {
        Duration::from_secs_f64(1.0 / self.target_tokens_per_second)
    }

    /// Checks limits, timer ranges and unique endpoints before allocating node resources.
    pub fn validate(&self) -> Result<(), ConfigError> {
        self.validate_positive_limits()?;
        self.validate_timing()?;
        self.validate_capacity_bounds()?;
        self.validate_endpoints()
    }

    fn validate_positive_limits(&self) -> Result<(), ConfigError> {
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
            ("max_prefill_batch_size", self.max_prefill_batch_size),
        ] {
            if value == 0 {
                return Err(ConfigError(format!("{name} must be positive")));
            }
        }
        Ok(())
    }

    fn validate_timing(&self) -> Result<(), ConfigError> {
        if !self.target_tokens_per_second.is_finite()
            || self.target_tokens_per_second <= 0.0
            || !self.initial_forward_estimate_ms.is_finite()
            || self.initial_forward_estimate_ms <= 0.0
            || !self.admission_headroom.is_finite()
            || !(0.0..=1.0).contains(&self.admission_headroom)
            || self.admission_headroom == 0.0
            || self.backend_timeout_ms == 0
            || self.max_prefill_wait_ms == 0
        {
            return Err(ConfigError("invalid throughput, timing or headroom".into()));
        }
        let interval = std::time::Duration::try_from_secs_f64(1.0 / self.target_tokens_per_second)
            .map_err(|_| ConfigError("token interval exceeds representable duration".into()))?;
        if interval.is_zero() || std::time::Instant::now().checked_add(interval).is_none() {
            return Err(ConfigError(
                "token interval must be a nonzero representable duration".into(),
            ));
        }
        if std::time::Instant::now()
            .checked_add(std::time::Duration::from_millis(self.backend_timeout_ms))
            .is_none()
        {
            return Err(ConfigError(
                "backend timeout exceeds platform timer range".into(),
            ));
        }
        if std::time::Instant::now()
            .checked_add(std::time::Duration::from_millis(self.max_prefill_wait_ms))
            .is_none()
        {
            return Err(ConfigError(
                "prefill queue wait exceeds platform timer range".into(),
            ));
        }
        Ok(())
    }

    fn validate_capacity_bounds(&self) -> Result<(), ConfigError> {
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
        Ok(())
    }

    fn validate_endpoints(&self) -> Result<(), ConfigError> {
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
