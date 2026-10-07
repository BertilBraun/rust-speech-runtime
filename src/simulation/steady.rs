//! A fixed measurement interval excludes ramp-up and in-flight turn draining.

use hdrhistogram::Histogram;
use serde::Serialize;
use tokio::time::Instant;

use super::measurements::{TurnTiming, distribution, histogram, record};
use super::report::SessionSummary;
use crate::metrics::LatencyDistribution;

#[derive(Clone, Copy, Debug)]
pub(super) struct MeasurementWindow {
    pub starts_at: Instant,
    pub ends_at: Instant,
}

impl MeasurementWindow {
    pub fn contains(self, now: Instant) -> bool {
        now >= self.starts_at && now < self.ends_at
    }
}

/// Observations in one measurement interval, shared by cohort and per-session reports.
#[derive(Debug, Serialize)]
pub struct MeasuredTraffic {
    pub received_tokens: usize,
    pub completed_turns: usize,
    pub rejected_turns: usize,
    pub token_gaps_over_target: usize,
    pub rolling_rate_observations: usize,
    pub rolling_rate_violations: usize,
    pub minimum_rolling_tokens_per_second: Option<f64>,
    pub time_to_first_token_ms: LatencyDistribution,
    pub end_of_audio_to_first_token_ms: LatencyDistribution,
    pub token_gap_ms: LatencyDistribution,
}

/// Fixed-window throughput; lifetime session accounting remains in the outer report.
#[derive(Debug, Serialize)]
pub struct SteadyStateReport {
    pub ramp_up_seconds: f64,
    pub measurement_seconds: f64,
    pub sessions_with_tokens: usize,
    pub sessions_with_rolling_rate_observations: usize,
    pub sessions_with_rolling_rate_violations: usize,
    pub aggregate_tokens_per_second: f64,
    pub traffic: MeasuredTraffic,
}

pub(super) struct SteadyMeasurements {
    tokens: usize,
    completed_turns: usize,
    rejected_turns: usize,
    gaps_over_target: usize,
    rate_observations: usize,
    rate_violations: usize,
    minimum_rate: Option<f64>,
    ttft: Histogram<u64>,
    end_of_audio: Histogram<u64>,
    gaps: Histogram<u64>,
}

impl Default for SteadyMeasurements {
    fn default() -> Self {
        Self {
            tokens: 0,
            completed_turns: 0,
            rejected_turns: 0,
            gaps_over_target: 0,
            rate_observations: 0,
            rate_violations: 0,
            minimum_rate: None,
            ttft: histogram(),
            end_of_audio: histogram(),
            gaps: histogram(),
        }
    }
}

impl SteadyMeasurements {
    pub fn token(
        &mut self,
        now: Instant,
        timing: &TurnTiming,
        window: MeasurementWindow,
        target: f64,
    ) {
        if timing.previous_token.is_none() && window.contains(timing.committed) {
            // Retain late first tokens during draining instead of censoring latency tails.
            record(&mut self.ttft, now - timing.committed);
            record(&mut self.end_of_audio, now - timing.audio_finished);
        }
        if !window.contains(now) {
            return;
        }
        self.tokens += 1;
        match timing.previous_token {
            Some(previous) if window.contains(previous) => {
                let gap = now - previous;
                record(&mut self.gaps, gap);
                self.gaps_over_target += usize::from(gap.as_secs_f64() > 1.0 / target);
            }
            _ => {}
        }
    }

    pub fn observe_rate(&mut self, rate: f64, target: f64) {
        self.rate_observations += 1;
        self.rate_violations += usize::from(rate < target);
        self.minimum_rate = Some(
            self.minimum_rate
                .map_or(rate, |previous| previous.min(rate)),
        );
    }

    pub fn finish_turn(&mut self) {
        self.completed_turns += 1;
    }

    pub fn reject_turn(&mut self) {
        self.rejected_turns += 1;
    }

    pub fn merge(&mut self, other: &Self) {
        self.tokens += other.tokens;
        self.completed_turns += other.completed_turns;
        self.rejected_turns += other.rejected_turns;
        self.gaps_over_target += other.gaps_over_target;
        self.rate_observations += other.rate_observations;
        self.rate_violations += other.rate_violations;
        if let Some(rate) = other.minimum_rate {
            self.minimum_rate = Some(
                self.minimum_rate
                    .map_or(rate, |previous| previous.min(rate)),
            );
        }
        for (destination, source) in [
            (&mut self.ttft, &other.ttft),
            (&mut self.end_of_audio, &other.end_of_audio),
            (&mut self.gaps, &other.gaps),
        ] {
            destination
                .add(source)
                .expect("matching histogram precision");
        }
    }

    pub fn report(&self) -> MeasuredTraffic {
        MeasuredTraffic {
            received_tokens: self.tokens,
            completed_turns: self.completed_turns,
            rejected_turns: self.rejected_turns,
            token_gaps_over_target: self.gaps_over_target,
            rolling_rate_observations: self.rate_observations,
            rolling_rate_violations: self.rate_violations,
            minimum_rolling_tokens_per_second: self.minimum_rate,
            time_to_first_token_ms: distribution(&self.ttft),
            end_of_audio_to_first_token_ms: distribution(&self.end_of_audio),
            token_gap_ms: distribution(&self.gaps),
        }
    }

    pub fn cohort_report(
        &self,
        window: MeasurementWindow,
        started: Instant,
        sessions: &[SessionSummary],
    ) -> SteadyStateReport {
        let mut sessions_with_tokens = 0;
        let mut sessions_with_rolling_rate_observations = 0;
        let mut sessions_with_rolling_rate_violations = 0;
        for session in sessions {
            let Some(traffic) = &session.measured else {
                continue;
            };
            sessions_with_tokens += usize::from(traffic.received_tokens > 0);
            sessions_with_rolling_rate_observations +=
                usize::from(traffic.rolling_rate_observations > 0);
            sessions_with_rolling_rate_violations +=
                usize::from(traffic.rolling_rate_violations > 0);
        }
        let measurement_seconds = (window.ends_at - window.starts_at).as_secs_f64();
        let traffic = self.report();
        SteadyStateReport {
            ramp_up_seconds: (window.starts_at - started).as_secs_f64(),
            measurement_seconds,
            sessions_with_tokens,
            sessions_with_rolling_rate_observations,
            sessions_with_rolling_rate_violations,
            aggregate_tokens_per_second: traffic.received_tokens as f64 / measurement_seconds,
            traffic,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;
    use tokio::sync::watch;

    use super::*;
    use crate::simulation::measurements::Measurements;

    #[test]
    fn ramp_up_and_drain_do_not_enter_the_measurement_interval() {
        let started = Instant::now();
        let window = MeasurementWindow {
            starts_at: started + Duration::from_secs(10),
            ends_at: started + Duration::from_secs(20),
        };
        let (_sender, receiver) = watch::channel(Some(window));
        let mut measurements = Measurements::new("window".into());
        measurements.measurement_window = Some(receiver);
        let mut timing = TurnTiming::new(started, started);
        for seconds in [9, 10, 11, 20, 21] {
            timing.token(
                started + Duration::from_secs(seconds),
                &mut measurements,
                4.0,
            );
        }
        let observed = measurements.steady.report();
        assert_eq!(measurements.summary.received_tokens, 5);
        assert_eq!(observed.received_tokens, 2);
        assert_eq!(observed.time_to_first_token_ms.count, 0);
        assert_eq!(observed.token_gap_ms.count, 1);
    }

    #[test]
    fn rolling_rate_needs_a_full_window_after_ramp_up() {
        let started = Instant::now();
        let (_sender, receiver) = watch::channel(Some(MeasurementWindow {
            starts_at: started + Duration::from_secs(10),
            ends_at: started + Duration::from_secs(20),
        }));
        let mut measurements = Measurements::new("rate".into());
        measurements.measurement_window = Some(receiver);
        let mut timing = TurnTiming::new(started, started);
        timing.token(started + Duration::from_secs(8), &mut measurements, 4.0);
        for seconds in [11, 13, 20] {
            timing.observe_rate(
                started + Duration::from_secs(seconds),
                &mut measurements,
                4.0,
                Duration::from_secs(2),
            );
        }
        let observed = measurements.steady.report();
        assert_eq!(observed.rolling_rate_observations, 1);
        assert_eq!(observed.rolling_rate_violations, 1);
        assert_eq!(observed.minimum_rolling_tokens_per_second, Some(0.0));
    }

    #[test]
    fn first_tokens_in_draining_preserve_committed_turn_latency() {
        let started = Instant::now();
        let window = MeasurementWindow {
            starts_at: started,
            ends_at: started + Duration::from_secs(10),
        };
        let (_sender, receiver) = watch::channel(Some(window));
        let mut measurements = Measurements::new("late-first-token".into());
        measurements.measurement_window = Some(receiver);
        let mut timing = TurnTiming::new(started, started);
        timing.token(
            window.ends_at + Duration::from_secs(1),
            &mut measurements,
            4.0,
        );
        let observed = measurements.steady.report();
        assert_eq!(observed.received_tokens, 0);
        assert_eq!(observed.time_to_first_token_ms.count, 1);
    }
}
