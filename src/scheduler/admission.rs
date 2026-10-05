use crate::{
    config::RuntimeConfig,
    protocol::{SessionId, WorkerId},
};
use std::{
    collections::{HashSet, VecDeque},
    time::Duration,
};

#[derive(Clone, Debug)]
pub(crate) struct WorkerStatus {
    pub worker_id: WorkerId,
    pub sessions: HashSet<SessionId>,
    pub session_limit: usize,
    pub service_time: Duration,
}
pub(crate) struct ServiceEstimator {
    samples: VecDeque<Duration>,
    host_samples: VecDeque<Duration>,
    window: usize,
    safety_factor: f64,
}
impl ServiceEstimator {
    pub(crate) fn new(configuration: &RuntimeConfig) -> Self {
        Self {
            samples: VecDeque::with_capacity(configuration.latency_window),
            host_samples: VecDeque::with_capacity(configuration.latency_window),
            window: configuration.latency_window,
            safety_factor: configuration.latency_safety_factor,
        }
    }
    pub(crate) fn observe(&mut self, duration: Duration) {
        if self.samples.len() == self.window {
            self.samples.pop_front();
        }
        self.samples.push_back(duration);
    }
    pub(crate) fn service_time(&self) -> Duration {
        self.device_time().mul_f64(self.safety_factor)
    }
    pub(crate) fn observe_host_delay(&mut self, duration: Duration) {
        if self.host_samples.len() == self.window {
            self.host_samples.pop_front();
        }
        self.host_samples.push_back(duration);
    }
    pub(crate) fn host_delay(&self) -> Duration {
        assert!(
            !self.host_samples.is_empty(),
            "admission waits for host calibration"
        );
        let mut samples: Vec<Duration> = self.host_samples.iter().copied().collect();
        samples.sort_unstable();
        samples[(samples.len() * 95).div_ceil(100) - 1]
    }
    pub(crate) fn available_limit(
        &self,
        configuration: &RuntimeConfig,
        capacity: usize,
        active_sessions: usize,
        queue_delay: Duration,
    ) -> usize {
        let host_delay = self.host_delay();
        if host_delay > configuration.scheduling_margin
            || queue_delay + self.device_time() + host_delay > configuration.compute_budget()
        {
            capacity.min(active_sessions)
        } else {
            capacity
        }
    }
    pub(crate) fn device_time(&self) -> Duration {
        assert!(!self.samples.is_empty(), "admission waits for calibration");
        let mut samples: Vec<Duration> = self.samples.iter().copied().collect();
        samples.sort_unstable();
        samples[(samples.len() * 95).div_ceil(100) - 1]
    }
}
pub(crate) fn session_limit(configuration: &RuntimeConfig, service_time: Duration) -> usize {
    let batches = configuration.compute_budget().as_nanos() / service_time.as_nanos();
    (((batches as usize).saturating_mul(configuration.batch_size) as f64
        * configuration.batch_fill_reserve)
        .floor() as usize)
        .min(configuration.max_sessions_per_worker)
        .min(configuration.cache_slots_per_worker)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn isolated_spike_does_not_hide_sustained_slowdown() {
        let mut estimator = ServiceEstimator::new(&RuntimeConfig {
            latency_safety_factor: 1.0,
            ..RuntimeConfig::default()
        });
        for _ in 0..64 {
            estimator.observe(Duration::from_millis(12));
        }
        estimator.observe(Duration::from_millis(40));
        assert_eq!(estimator.service_time(), Duration::from_millis(12));
        for _ in 0..3 {
            estimator.observe(Duration::from_millis(40));
        }
        assert_eq!(estimator.service_time(), Duration::from_millis(40));
    }
    #[test]
    fn capacity_reserves_whole_batches_and_headroom() {
        let configuration = RuntimeConfig::default();
        assert_eq!(session_limit(&configuration, Duration::from_millis(12)), 48);
        assert_eq!(session_limit(&configuration, Duration::from_millis(20)), 16);
        assert_eq!(session_limit(&configuration, Duration::from_millis(40)), 0);
    }
    #[test]
    fn rolling_tail_recovers_after_slow_samples_age_out() {
        let mut estimator = ServiceEstimator::new(&RuntimeConfig {
            calibration_samples: 1,
            latency_window: 2,
            latency_safety_factor: 1.0,
            ..RuntimeConfig::default()
        });
        estimator.observe(Duration::from_millis(30));
        estimator.observe(Duration::from_millis(12));
        assert_eq!(estimator.service_time(), Duration::from_millis(30));
        estimator.observe(Duration::from_millis(12));
        assert_eq!(estimator.service_time(), Duration::from_millis(12));
    }

    #[test]
    fn sustained_host_delay_pauses_admission_without_changing_device_capacity() {
        let configuration = RuntimeConfig::default();
        let mut estimator = ServiceEstimator::new(&configuration);
        estimator.observe(Duration::from_millis(12));
        for _ in 0..64 {
            estimator.observe_host_delay(Duration::from_millis(1));
        }
        estimator.observe_host_delay(Duration::from_millis(20));
        assert_eq!(
            estimator.available_limit(&configuration, 8, 4, Duration::ZERO),
            8
        );
        for _ in 0..3 {
            estimator.observe_host_delay(Duration::from_millis(20));
        }
        assert_eq!(
            estimator.available_limit(&configuration, 8, 4, Duration::ZERO),
            4
        );
        assert_eq!(estimator.device_time(), Duration::from_millis(12));
        for _ in 0..64 {
            estimator.observe_host_delay(Duration::from_millis(1));
        }
        assert_eq!(
            estimator.available_limit(&configuration, 8, 4, Duration::ZERO),
            8
        );
        assert_eq!(
            estimator.available_limit(&configuration, 8, 4, Duration::from_millis(40)),
            4
        );
    }
}
