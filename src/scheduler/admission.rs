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
    window: usize,
    safety_factor: f64,
}
impl ServiceEstimator {
    pub(crate) fn new(configuration: &RuntimeConfig) -> Self {
        Self {
            samples: VecDeque::with_capacity(configuration.latency_window),
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
        assert!(!self.samples.is_empty(), "admission waits for calibration");
        let mut samples: Vec<Duration> = self.samples.iter().copied().collect();
        samples.sort_unstable();
        samples[(samples.len() * 95).div_ceil(100) - 1].mul_f64(self.safety_factor)
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
        assert_eq!(session_limit(&configuration, Duration::from_millis(12)), 16);
        assert_eq!(session_limit(&configuration, Duration::from_millis(20)), 8);
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
}
