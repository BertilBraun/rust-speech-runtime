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
    pub(crate) fn host_reserve(&self) -> Duration {
        self.host_samples
            .iter()
            .copied()
            .max()
            .expect("admission waits for host calibration")
    }
    pub(crate) fn available_limit(
        &self,
        configuration: &RuntimeConfig,
        capacity: usize,
        active_sessions: usize,
        queue_delay: Duration,
        known_device_time: Duration,
    ) -> usize {
        let host_delay = self.host_reserve();
        let device_time = self.device_time().max(known_device_time);
        let available = capacity.min(session_limit_with_host_delay(
            configuration,
            device_time.mul_f64(self.safety_factor),
            host_delay,
        ));
        if queue_delay + device_time + host_delay
            > configuration.compute_budget() + configuration.packet_lateness_grace
        {
            available.min(active_sessions)
        } else {
            available
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
    session_limit_with_host_delay(configuration, service_time, Duration::ZERO)
}
fn session_limit_with_host_delay(
    configuration: &RuntimeConfig,
    service_time: Duration,
    host_delay: Duration,
) -> usize {
    let budget = configuration
        .compute_budget()
        .saturating_sub(host_delay.saturating_sub(configuration.packet_lateness_grace));
    if service_time > budget {
        return 0;
    }
    let throughput_budget = budget.mul_f64(configuration.batch_fill_reserve);
    let throughput_sessions =
        throughput_budget.as_nanos() * configuration.batch_size as u128 / service_time.as_nanos();
    let burst_budget = configuration
        .packet_completion_budget()
        .saturating_sub(configuration.scheduling_margin + host_delay);
    // A partial final batch consumes a full device step during a simultaneous burst.
    let burst_sessions =
        burst_budget.as_nanos() / service_time.as_nanos() * configuration.batch_size as u128;
    throughput_sessions
        .min(burst_sessions)
        .min(configuration.max_sessions_per_worker as u128)
        .min(configuration.cache_slots_per_worker as u128) as usize
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
    fn streaming_capacity_retains_fractional_cycles_and_whole_batch_burst_cost() {
        let configuration = RuntimeConfig {
            max_sessions_per_worker: 64,
            ..RuntimeConfig::default()
        };
        assert_eq!(session_limit(&configuration, Duration::from_millis(12)), 51);
        assert_eq!(session_limit(&configuration, Duration::from_millis(20)), 30);
        assert_eq!(session_limit(&configuration, Duration::from_millis(40)), 0);
        let strict = RuntimeConfig {
            packet_lateness_grace: Duration::ZERO,
            ..configuration
        };
        assert_eq!(session_limit(&strict, Duration::from_millis(12)), 48);
        let fourth_batch = Duration::from_millis(12) * 4;
        assert!(fourth_batch + strict.scheduling_margin > strict.packet_deadline);
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
    fn host_spikes_pause_admission_without_changing_device_capacity() {
        let configuration = RuntimeConfig {
            max_sessions_per_worker: 48,
            packet_lateness_grace: Duration::ZERO,
            ..RuntimeConfig::default()
        };
        let mut estimator = ServiceEstimator::new(&configuration);
        estimator.observe(Duration::from_millis(12));
        for _ in 0..64 {
            estimator.observe_host_delay(Duration::from_millis(1));
        }
        estimator.observe_host_delay(Duration::from_millis(20));
        assert_eq!(
            estimator.available_limit(
                &configuration,
                48,
                32,
                Duration::ZERO,
                Duration::from_millis(12)
            ),
            24
        );
        for _ in 0..3 {
            estimator.observe_host_delay(Duration::from_millis(20));
        }
        assert_eq!(
            estimator.available_limit(
                &configuration,
                48,
                32,
                Duration::ZERO,
                Duration::from_millis(12)
            ),
            24
        );
        assert_eq!(estimator.device_time(), Duration::from_millis(12));
        for _ in 0..64 {
            estimator.observe_host_delay(Duration::from_millis(1));
        }
        assert_eq!(
            estimator.available_limit(
                &configuration,
                48,
                32,
                Duration::ZERO,
                Duration::from_millis(12)
            ),
            48
        );
        assert_eq!(
            estimator.available_limit(
                &configuration,
                48,
                32,
                Duration::from_millis(40),
                Duration::from_millis(12)
            ),
            32
        );
    }

    #[test]
    fn host_delay_uses_available_deadline_budget_instead_of_a_fixed_margin_cutoff() {
        let configuration = RuntimeConfig {
            packet_deadline: Duration::from_millis(250),
            minimum_packet_interval: Duration::from_millis(250),
            batch_size: 4,
            max_sessions_per_worker: 4,
            cache_slots_per_worker: 4,
            ..RuntimeConfig::default()
        };
        let mut estimator = ServiceEstimator::new(&configuration);
        estimator.observe(Duration::from_millis(3));
        estimator.observe_host_delay(Duration::from_millis(10));
        assert_eq!(
            estimator.available_limit(
                &configuration,
                4,
                0,
                Duration::ZERO,
                Duration::from_millis(3)
            ),
            4
        );
    }

    #[test]
    fn known_device_cost_reserves_host_slack_before_slow_results_are_observed() {
        let configuration = RuntimeConfig {
            max_sessions_per_worker: 48,
            packet_lateness_grace: Duration::ZERO,
            ..RuntimeConfig::default()
        };
        let mut estimator = ServiceEstimator::new(&configuration);
        estimator.observe(Duration::from_millis(12));
        estimator.observe_host_delay(Duration::from_millis(5));
        assert_eq!(
            estimator.available_limit(
                &configuration,
                48,
                0,
                Duration::ZERO,
                Duration::from_millis(12),
            ),
            44
        );
        assert_eq!(
            estimator.available_limit(
                &configuration,
                48,
                0,
                Duration::ZERO,
                Duration::from_millis(18),
            ),
            29
        );
        assert_eq!(estimator.device_time(), Duration::from_millis(12));
    }

    #[test]
    fn recovery_grace_absorbs_host_reserve_and_checks_the_burst_budget() {
        let mut configuration = RuntimeConfig {
            max_sessions_per_worker: 64,
            packet_lateness_grace: Duration::ZERO,
            ..RuntimeConfig::default()
        };
        let mut estimator = ServiceEstimator::new(&configuration);
        estimator.observe(Duration::from_millis(12));
        estimator.observe_host_delay(Duration::from_millis(5));
        let capacity = session_limit(&configuration, Duration::from_millis(12));
        assert_eq!(capacity, 48);
        assert_eq!(
            estimator.available_limit(
                &configuration,
                capacity,
                0,
                Duration::ZERO,
                Duration::from_millis(12)
            ),
            44
        );
        configuration.packet_lateness_grace = Duration::from_millis(10);
        assert_eq!(session_limit(&configuration, Duration::from_millis(12)), 51);
        assert_eq!(
            estimator.available_limit(
                &configuration,
                51,
                0,
                Duration::ZERO,
                Duration::from_millis(12)
            ),
            51
        );
        estimator.observe_host_delay(Duration::from_millis(20));
        assert_eq!(
            estimator.available_limit(
                &configuration,
                51,
                0,
                Duration::ZERO,
                Duration::from_millis(12)
            ),
            32
        );
    }
}
