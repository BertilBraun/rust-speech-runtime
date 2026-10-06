use super::{ArrivalPhase, SimulationConfig};
use crate::metrics::LatencyHistogram;
use rand::{Rng, SeedableRng, rngs::SmallRng};
use std::{cmp::Reverse, collections::BinaryHeap, time::Duration};
use tokio::{sync::mpsc, time::Instant};
use tokio_util::sync::CancellationToken;

pub(super) struct PacerSlot {
    pub(super) sender: mpsc::Sender<Instant>,
    pub(super) cancellation: CancellationToken,
}
#[derive(Default)]
pub(super) struct PacerMeasurements {
    pub(super) delay: LatencyHistogram,
    pub(super) intervals: LatencyHistogram,
    pub(super) overruns: u64,
}

pub(super) fn pace(
    slots: Vec<PacerSlot>,
    configuration: SimulationConfig,
    cancellation: CancellationToken,
) -> PacerMeasurements {
    let mut random = SmallRng::seed_from_u64(configuration.seed);
    let mut events = BinaryHeap::new();
    let start = std::time::Instant::now();
    let stop = start + configuration.duration;
    for index in 0..slots.len() {
        let offset = match configuration.phase {
            ArrivalPhase::Aligned => Duration::ZERO,
            ArrivalPhase::Random => Duration::from_micros(
                random.random_range(0..configuration.maximum_interval.as_micros() as u64),
            ),
        };
        events.push(Reverse((start + offset, index)));
    }
    let mut measurements = PacerMeasurements::default();
    let mut last_captures = vec![None; slots.len()];
    let mut capture_group = Vec::with_capacity(slots.len());
    while let Some(Reverse((scheduled, index))) = events.pop() {
        if scheduled >= stop || cancellation.is_cancelled() {
            break;
        }
        capture_group.clear();
        capture_group.push(index);
        while events
            .peek()
            .is_some_and(|Reverse((time, _))| *time == scheduled)
        {
            let Reverse((_, index)) = events.pop().expect("matching capture exists");
            capture_group.push(index);
        }
        let remaining = scheduled.saturating_duration_since(std::time::Instant::now());
        if !remaining.is_zero() {
            std::thread::sleep(remaining);
        }
        let captured = std::time::Instant::now();
        if captured >= stop {
            break;
        }
        for &index in &capture_group {
            let slot = &slots[index];
            if slot.cancellation.is_cancelled() {
                continue;
            }
            measurements
                .delay
                .record(captured.saturating_duration_since(scheduled));
            match slot.sender.try_send(Instant::from_std(captured)) {
                Ok(()) => {
                    if let Some(previous) = last_captures[index] {
                        measurements
                            .intervals
                            .record(captured.duration_since(previous));
                    }
                    last_captures[index] = Some(captured);
                }
                Err(mpsc::error::TrySendError::Closed(_)) => continue,
                Err(mpsc::error::TrySendError::Full(_)) => {
                    measurements.overruns += 1;
                    slot.cancellation.cancel();
                    continue;
                }
            }
            let interval = Duration::from_micros(random.random_range(
                configuration.minimum_interval.as_micros() as u64
                    ..=configuration.maximum_interval.as_micros() as u64,
            ));
            events.push(Reverse((captured + interval, index)));
        }
    }
    measurements
}

#[cfg(test)]
mod tests {
    use super::{ArrivalPhase, PacerSlot, SimulationConfig, pace};
    use std::time::Duration;
    use tokio::sync::mpsc;
    use tokio_util::sync::CancellationToken;

    #[tokio::test]
    async fn aligned_capture_waves_keep_identical_timestamps_across_sessions() {
        let mut slots = Vec::new();
        let mut captures = Vec::new();
        for _ in 0..4 {
            let (sender, receiver) = mpsc::channel(16);
            slots.push(PacerSlot {
                sender,
                cancellation: CancellationToken::new(),
            });
            captures.push(receiver);
        }
        let measurements = tokio::task::spawn_blocking(move || {
            pace(
                slots,
                SimulationConfig {
                    phase: ArrivalPhase::Aligned,
                    minimum_interval: Duration::from_millis(20),
                    maximum_interval: Duration::from_millis(20),
                    duration: Duration::from_millis(100),
                    ..SimulationConfig::default()
                },
                CancellationToken::new(),
            )
        })
        .await
        .unwrap();
        let waves: Vec<Vec<_>> = captures
            .iter_mut()
            .map(|receiver| std::iter::from_fn(|| receiver.try_recv().ok()).collect())
            .collect();
        assert!(waves[0].len() >= 2);
        assert!(waves.iter().all(|wave| *wave == waves[0]));
        assert_eq!(measurements.overruns, 0);
    }
}
