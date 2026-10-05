use std::time::Duration;

use thiserror::Error;

#[derive(Clone, Debug)]
pub struct RuntimeConfig {
    pub workers: usize,
    pub tick_interval: Duration,
    pub batch_size: usize,
    pub inference_latency: Duration,
    pub max_sessions_per_worker: usize,
    pub cache_slots_per_worker: usize,
    pub ingress_capacity: usize,
    pub worker_channel_capacity: usize,
    pub result_channel_capacity: usize,
    pub output_channel_capacity: usize,
    pub max_frame_bytes: usize,
    pub max_input_age: Duration,
    pub session_timeout: Duration,
    pub phase_bucket: Option<Duration>,
    pub scheduling_margin: Duration,
    pub worker_control_delay: Duration,
    pub slowdown: Option<WorkerSlowdown>,
}

#[derive(Clone, Debug)]
pub struct WorkerSlowdown {
    pub after: Duration,
    pub inference_latency: Duration,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            workers: 8,
            tick_interval: Duration::from_millis(50),
            batch_size: 16,
            inference_latency: Duration::from_millis(12),
            max_sessions_per_worker: 52,
            cache_slots_per_worker: 64,
            ingress_capacity: 1024,
            worker_channel_capacity: 256,
            result_channel_capacity: 256,
            output_channel_capacity: 1024,
            max_frame_bytes: 4096,
            max_input_age: Duration::from_millis(100),
            session_timeout: Duration::from_secs(30),
            phase_bucket: None,
            scheduling_margin: Duration::from_millis(1),
            worker_control_delay: Duration::ZERO,
            slowdown: None,
        }
    }
}

#[derive(Debug, Error)]
#[error("invalid configuration: {0}")]
pub struct ConfigError(pub String);

impl RuntimeConfig {
    pub fn validate(&self) -> Result<(), ConfigError> {
        let sizes = [
            ("workers", self.workers),
            ("batch_size", self.batch_size),
            ("max_sessions_per_worker", self.max_sessions_per_worker),
            ("cache_slots_per_worker", self.cache_slots_per_worker),
            ("ingress_capacity", self.ingress_capacity),
            ("worker_channel_capacity", self.worker_channel_capacity),
            ("result_channel_capacity", self.result_channel_capacity),
            ("output_channel_capacity", self.output_channel_capacity),
            ("max_frame_bytes", self.max_frame_bytes),
        ];
        for (name, value) in sizes {
            if value == 0 {
                return Err(ConfigError(format!("{name} must be positive")));
            }
        }
        for (name, value) in [
            ("tick_interval", self.tick_interval),
            ("inference_latency", self.inference_latency),
            ("max_input_age", self.max_input_age),
            ("session_timeout", self.session_timeout),
        ] {
            if value.is_zero() || value > Duration::from_secs(86400) {
                return Err(ConfigError(format!("{name} must be in (0, 24 hours]")));
            }
        }
        if let Some(bucket) = self.phase_bucket
            && (bucket.is_zero() || bucket > self.tick_interval)
        {
            return Err(ConfigError(
                "phase_bucket must be in (0, tick_interval]".into(),
            ));
        }
        if let Some(slowdown) = &self.slowdown
            && (slowdown.inference_latency.is_zero()
                || slowdown.inference_latency > Duration::from_secs(86400))
        {
            return Err(ConfigError(
                "slowdown latency must be in (0, 24 hours]".into(),
            ));
        }
        if self.scheduling_margin >= self.tick_interval {
            return Err(ConfigError(
                "scheduling_margin must be less than tick_interval".into(),
            ));
        }
        if self.cache_slots_per_worker > u32::MAX as usize {
            return Err(ConfigError("cache slots exceed handle range".into()));
        }
        if self
            .workers
            .checked_mul(self.max_sessions_per_worker)
            .is_none()
        {
            return Err(ConfigError("total session capacity overflows".into()));
        }
        Ok(())
    }

    pub fn admission_limit(&self) -> usize {
        self.max_sessions_per_worker
            .min(self.cache_slots_per_worker)
    }
}
