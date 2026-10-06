use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PacketQualityPolicy {
    pub window_packets: usize,
    pub miss_limit: usize,
}

impl Default for PacketQualityPolicy {
    fn default() -> Self {
        Self {
            window_packets: 10,
            miss_limit: 4,
        }
    }
}

pub(super) struct PacketQuality {
    policy: PacketQualityPolicy,
    window: VecDeque<bool>,
    pub consecutive_misses: u64,
    pub window_misses: u64,
}

impl PacketQuality {
    pub fn new(policy: PacketQualityPolicy) -> Self {
        Self {
            policy,
            window: VecDeque::with_capacity(policy.window_packets),
            consecutive_misses: 0,
            window_misses: 0,
        }
    }

    pub fn observe(&mut self, late: bool) -> bool {
        if self.window.len() == self.policy.window_packets {
            self.window_misses -= u64::from(self.window.pop_front().expect("full window"));
        }
        self.window.push_back(late);
        self.window_misses += u64::from(late);
        self.consecutive_misses = if late { self.consecutive_misses + 1 } else { 0 };
        self.window_misses >= self.policy.miss_limit as u64
    }
}

#[cfg(test)]
mod tests {
    use super::{PacketQuality, PacketQualityPolicy};

    #[test]
    fn isolated_misses_age_out_without_failing_the_session() {
        let mut quality = PacketQuality::new(PacketQualityPolicy::default());
        for index in 0..100 {
            assert!(!quality.observe(index % 5 == 0));
            assert!(quality.window.len() <= 10);
            assert!(quality.window_misses <= 2);
        }
    }

    #[test]
    fn four_nearby_misses_fail_even_when_not_consecutive() {
        let mut quality = PacketQuality::new(PacketQualityPolicy::default());
        for late in [true, false, true, false, true, false] {
            assert!(!quality.observe(late));
        }
        assert!(quality.observe(true));
        assert_eq!(quality.consecutive_misses, 1);
        assert_eq!(quality.window_misses, 4);
    }

    #[test]
    fn four_consecutive_misses_fail() {
        let mut quality = PacketQuality::new(PacketQualityPolicy::default());
        for _ in 0..3 {
            assert!(!quality.observe(true));
        }
        assert!(quality.observe(true));
        assert_eq!(quality.consecutive_misses, 4);
    }
}
