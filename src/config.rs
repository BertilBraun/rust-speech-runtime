use serde::{Deserialize, Serialize};
use std::{str::FromStr, time::Duration};
use thiserror::Error;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AudioLimits {
    pub max_frame_bytes: usize,
    pub max_prefix_bytes: usize,
    pub max_prefix_packets: usize,
}
impl Default for AudioLimits {
    fn default() -> Self {
        Self {
            max_frame_bytes: 4096,
            max_prefix_bytes: 16 * 1024 * 1024,
            max_prefix_packets: 10_000,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct RuntimeConfig {
    pub workers: usize,
    pub minimum_packet_interval: Duration,
    pub packet_deadline: Duration,
    pub packet_lateness_grace: Duration,
    pub batch_size: usize,
    pub inference_latency: Duration,
    pub device_wait: DeviceWait,
    pub device_queue_capacity: usize,
    pub launch_ahead: Duration,
    pub max_batch_wait: Duration,
    pub max_sessions_per_worker: usize,
    pub cache_slots_per_worker: usize,
    pub ingress_capacity: usize,
    pub worker_channel_capacity: usize,
    pub audio_limits: AudioLimits,
    pub replay_latency_per_packet: Duration,
    pub session_timeout: Duration,
    pub scheduling_margin: Duration,
    pub admission_headroom: f64,
    pub batch_fill_reserve: f64,
    pub latency_safety_factor: f64,
    pub calibration_samples: usize,
    pub latency_window: usize,
    pub probe_interval: Duration,
    pub worker_input_delay: Duration,
    pub slowdown: Option<WorkerSlowdown>,
}
#[derive(Clone, Debug, Serialize)]
pub struct WorkerSlowdown {
    pub after: Duration,
    pub inference_latency: Duration,
}
impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            workers: 8,
            minimum_packet_interval: Duration::from_millis(48),
            packet_deadline: Duration::from_millis(50),
            packet_lateness_grace: Duration::from_millis(10),
            batch_size: 16,
            inference_latency: Duration::from_millis(12),
            device_wait: DeviceWait::Tokio,
            device_queue_capacity: 2,
            launch_ahead: Duration::from_millis(2),
            max_batch_wait: Duration::from_millis(5),
            max_sessions_per_worker: 32,
            cache_slots_per_worker: 64,
            ingress_capacity: 1024,
            worker_channel_capacity: 256,
            audio_limits: AudioLimits::default(),
            replay_latency_per_packet: Duration::from_micros(100),
            session_timeout: Duration::from_secs(30),
            scheduling_margin: Duration::from_millis(5),
            admission_headroom: 0.9,
            batch_fill_reserve: 1.0,
            latency_safety_factor: 1.0,
            calibration_samples: 16,
            latency_window: 64,
            probe_interval: Duration::from_secs(1),
            worker_input_delay: Duration::ZERO,
            slowdown: None,
        }
    }
}
#[derive(Debug, Error)]
#[error("invalid configuration: {0}")]
pub struct ConfigError(pub String);

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub enum DeviceWait {
    #[default]
    Tokio,
    Thread(ThreadWait),
}
#[derive(Clone, Copy, Debug, Serialize)]
pub enum ThreadWait {
    Sleep,
    Hybrid { spin_tail: Duration },
    Poll { sleep_interval: Duration },
}
impl FromStr for DeviceWait {
    type Err = ConfigError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        if value == "tokio" {
            return Ok(Self::Tokio);
        }
        if value == "sleep" {
            return Ok(Self::Thread(ThreadWait::Sleep));
        }
        let (kind, interval) = value.split_once(':').ok_or_else(|| {
            ConfigError("device wait must be tokio, sleep, hybrid:200us or poll:500ns".into())
        })?;
        let duration = if let Some(amount) = interval.strip_suffix("ns") {
            Duration::from_nanos(
                amount
                    .parse()
                    .map_err(|_| ConfigError("invalid nanoseconds".into()))?,
            )
        } else if let Some(amount) = interval.strip_suffix("us") {
            Duration::from_micros(
                amount
                    .parse()
                    .map_err(|_| ConfigError("invalid microseconds".into()))?,
            )
        } else {
            return Err(ConfigError("device wait interval requires ns or us".into()));
        };
        match kind {
            "hybrid" if !duration.is_zero() && duration <= Duration::from_millis(1) => {
                Ok(Self::Thread(ThreadWait::Hybrid {
                    spin_tail: duration,
                }))
            }
            "poll" if !duration.is_zero() && duration <= Duration::from_millis(1) => {
                Ok(Self::Thread(ThreadWait::Poll {
                    sleep_interval: duration,
                }))
            }
            _ => Err(ConfigError(
                "device wait interval must be in (0, 1 ms]".into(),
            )),
        }
    }
}
impl RuntimeConfig {
    pub fn packet_completion_budget(&self) -> Duration {
        self.packet_deadline + self.packet_lateness_grace
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.packet_lateness_grace > self.minimum_packet_interval {
            return Err(ConfigError(
                "packet_lateness_grace must not exceed minimum_packet_interval".into(),
            ));
        }
        match self.device_wait {
            DeviceWait::Tokio | DeviceWait::Thread(ThreadWait::Sleep) => {}
            DeviceWait::Thread(
                ThreadWait::Hybrid {
                    spin_tail: interval,
                }
                | ThreadWait::Poll {
                    sleep_interval: interval,
                },
            ) => {
                if interval.is_zero() || interval > Duration::from_millis(1) {
                    return Err(ConfigError(
                        "device wait interval must be in (0, 1 ms]".into(),
                    ));
                }
            }
        }
        for (name, size) in [
            ("workers", self.workers),
            ("batch_size", self.batch_size),
            ("max_sessions_per_worker", self.max_sessions_per_worker),
            ("cache_slots_per_worker", self.cache_slots_per_worker),
            ("ingress_capacity", self.ingress_capacity),
            ("worker_channel_capacity", self.worker_channel_capacity),
            ("device_queue_capacity", self.device_queue_capacity),
            ("max_frame_bytes", self.audio_limits.max_frame_bytes),
            ("max_prefix_bytes", self.audio_limits.max_prefix_bytes),
            ("max_prefix_packets", self.audio_limits.max_prefix_packets),
            ("calibration_samples", self.calibration_samples),
            ("latency_window", self.latency_window),
        ] {
            if size == 0 {
                return Err(ConfigError(format!("{name} must be positive")));
            }
        }
        for (name, duration) in [
            ("minimum_packet_interval", self.minimum_packet_interval),
            ("packet_deadline", self.packet_deadline),
            ("inference_latency", self.inference_latency),
            ("session_timeout", self.session_timeout),
            ("probe_interval", self.probe_interval),
        ] {
            if duration.is_zero() || duration > Duration::from_secs(86400) {
                return Err(ConfigError(format!("{name} must be in (0, 24 hours]")));
            }
        }
        if !self.admission_headroom.is_finite()
            || !(0.0..=1.0).contains(&self.admission_headroom)
            || self.admission_headroom == 0.0
        {
            return Err(ConfigError("admission_headroom must be in (0, 1]".into()));
        }
        if !self.latency_safety_factor.is_finite()
            || !(1.0..=10.0).contains(&self.latency_safety_factor)
        {
            return Err(ConfigError(
                "latency_safety_factor must be in [1, 10]".into(),
            ));
        }
        if !self.batch_fill_reserve.is_finite()
            || !(0.0..=1.0).contains(&self.batch_fill_reserve)
            || self.batch_fill_reserve == 0.0
        {
            return Err(ConfigError("batch_fill_reserve must be in (0, 1]".into()));
        }
        if self.scheduling_margin >= self.packet_deadline.min(self.minimum_packet_interval) {
            return Err(ConfigError(
                "scheduling_margin must be less than the deadline and minimum interval".into(),
            ));
        }
        if self.max_batch_wait >= self.packet_deadline {
            return Err(ConfigError(
                "max_batch_wait must be below packet_deadline".into(),
            ));
        }
        if self.latency_window < self.calibration_samples {
            return Err(ConfigError(
                "latency_window must hold calibration_samples".into(),
            ));
        }
        if self.device_queue_capacity > 16 {
            return Err(ConfigError(
                "device_queue_capacity must be in 1..=16".into(),
            ));
        }
        if self.launch_ahead > self.packet_deadline {
            return Err(ConfigError(
                "launch_ahead must not exceed packet_deadline".into(),
            ));
        }
        if self.cache_slots_per_worker > u32::MAX as usize
            || self
                .workers
                .checked_mul(self.max_sessions_per_worker)
                .is_none()
        {
            return Err(ConfigError("worker or cache capacity overflows".into()));
        }
        if let Some(slowdown) = &self.slowdown
            && (slowdown.inference_latency.is_zero()
                || slowdown.inference_latency > Duration::from_secs(86400))
        {
            return Err(ConfigError(
                "slowdown latency must be in (0, 24 hours]".into(),
            ));
        }
        Ok(())
    }
    pub fn compute_budget(&self) -> Duration {
        (self.packet_deadline.min(self.minimum_packet_interval) - self.scheduling_margin)
            .mul_f64(self.admission_headroom)
    }
}
