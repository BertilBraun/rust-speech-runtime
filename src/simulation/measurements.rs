use std::{collections::VecDeque, time::Duration};

use hdrhistogram::Histogram;
use tokio::time::Instant;

use super::report::SessionSummary;
use crate::metrics::LatencyDistribution;

pub(crate) struct Measurements {
    pub summary: SessionSummary,
    pub ttft: Histogram<u64>,
    pub end_of_audio_to_first_token: Histogram<u64>,
    pub endpoint_confirmation: Histogram<u64>,
    pub token_gaps: Histogram<u64>,
}

impl Measurements {
    pub fn new(session_id: String) -> Self {
        Self {
            summary: SessionSummary::new(session_id),
            ttft: histogram(),
            end_of_audio_to_first_token: histogram(),
            endpoint_confirmation: histogram(),
            token_gaps: histogram(),
        }
    }
}

pub(crate) fn histogram() -> Histogram<u64> {
    Histogram::new(3).expect("valid histogram precision")
}

pub(crate) fn record(histogram: &mut Histogram<u64>, duration: Duration) {
    histogram
        .record(duration.as_micros().clamp(1, 3_600_000_000) as u64)
        .expect("duration clamped to histogram bounds");
}

pub(crate) fn distribution(histogram: &Histogram<u64>) -> LatencyDistribution {
    LatencyDistribution {
        count: histogram.len(),
        p50_ms: histogram.value_at_quantile(0.50) as f64 / 1000.0,
        p95_ms: histogram.value_at_quantile(0.95) as f64 / 1000.0,
        p99_ms: histogram.value_at_quantile(0.99) as f64 / 1000.0,
        max_ms: histogram.max() as f64 / 1000.0,
    }
}

pub(crate) struct TurnTiming {
    audio_finished: Instant,
    committed: Instant,
    first_token: Option<Instant>,
    previous_token: Option<Instant>,
    arrivals: VecDeque<Instant>,
    tokens: usize,
}

impl TurnTiming {
    pub fn new(audio_finished: Instant, committed: Instant) -> Self {
        Self {
            audio_finished,
            committed,
            first_token: None,
            previous_token: None,
            arrivals: VecDeque::new(),
            tokens: 0,
        }
    }

    pub fn committed(&self) -> Instant {
        self.committed
    }

    pub fn token(&mut self, now: Instant, measurements: &mut Measurements, target: f64) {
        if let Some(previous) = self.previous_token {
            let gap = now - previous;
            record(&mut measurements.token_gaps, gap);
            if gap.as_secs_f64() > 1.0 / target {
                measurements.summary.token_gaps_over_target += 1;
            }
        } else {
            record(&mut measurements.ttft, now - self.committed);
            record(
                &mut measurements.end_of_audio_to_first_token,
                now - self.audio_finished,
            );
            record(
                &mut measurements.endpoint_confirmation,
                self.committed - self.audio_finished,
            );
            self.first_token = Some(now);
        }
        self.previous_token = Some(now);
        self.tokens += 1;
        measurements.summary.received_tokens += 1;
        self.arrivals.push_back(now);
    }

    pub fn observe_rate(
        &mut self,
        now: Instant,
        measurements: &mut Measurements,
        target: f64,
        window: Duration,
    ) {
        while self
            .arrivals
            .front()
            .is_some_and(|arrival| now - *arrival >= window)
        {
            self.arrivals.pop_front();
        }
        if self.first_token.is_some_and(|first| now - first >= window) {
            let rate = self.arrivals.len() as f64 / window.as_secs_f64();
            if rate < target {
                measurements.summary.rolling_rate_violations += 1;
            }
            measurements.summary.minimum_rolling_tokens_per_second = Some(
                measurements
                    .summary
                    .minimum_rolling_tokens_per_second
                    .map_or(rate, |previous| previous.min(rate)),
            );
        }
    }

    pub fn finish(&self, measurements: &mut Measurements) {
        if self.tokens > 1 {
            let duration = self.previous_token.expect("tokens observed")
                - self.first_token.expect("tokens observed");
            let rate = (self.tokens - 1) as f64 / duration.as_secs_f64().max(f64::EPSILON);
            measurements.summary.minimum_turn_tokens_per_second = Some(
                measurements
                    .summary
                    .minimum_turn_tokens_per_second
                    .map_or(rate, |previous| previous.min(rate)),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoint_confirmation_is_separate_from_commit_to_first_token() {
        let audio_finished = Instant::now();
        let committed = audio_finished + Duration::from_millis(300);
        let mut timing = TurnTiming::new(audio_finished, committed);
        let mut measurements = Measurements::new("endpointing".into());
        timing.token(
            committed + Duration::from_millis(12),
            &mut measurements,
            4.0,
        );
        assert_eq!(distribution(&measurements.ttft).p50_ms.round(), 12.0);
        assert_eq!(
            distribution(&measurements.end_of_audio_to_first_token)
                .p50_ms
                .round(),
            312.0,
        );
        assert_eq!(
            distribution(&measurements.endpoint_confirmation)
                .p50_ms
                .round(),
            300.0,
        );
    }

    #[test]
    fn stalled_generation_has_zero_rolling_throughput() {
        let started = Instant::now();
        let mut timing = TurnTiming::new(started, started);
        let mut measurements = Measurements::new("stalled".into());
        timing.token(started, &mut measurements, 4.0);
        timing.observe_rate(
            started + Duration::from_secs(3),
            &mut measurements,
            4.0,
            Duration::from_secs(2),
        );
        assert_eq!(measurements.summary.rolling_rate_violations, 1);
        assert_eq!(
            measurements.summary.minimum_rolling_tokens_per_second,
            Some(0.0)
        );
    }
}
