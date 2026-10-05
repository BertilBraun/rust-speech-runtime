use std::time::Duration;

use hdrhistogram::Histogram;
use serde::Serialize;

use crate::protocol::WorkerId;

pub(crate) struct LatencyHistogram(Histogram<u64>);

impl Default for LatencyHistogram {
    fn default() -> Self {
        Self(Histogram::new_with_bounds(1, 86_400_000_000, 3).expect("valid histogram bounds"))
    }
}

impl LatencyHistogram {
    pub(crate) fn record(&mut self, duration: Duration) {
        let micros = duration.as_micros().min(86_400_000_000) as u64;
        self.0.record(micros).expect("sample is within bounds");
    }

    fn merge(&mut self, other: &Self) {
        self.0.add(&other.0).expect("histograms share bounds");
    }

    pub(crate) fn summary(&self) -> LatencyDistribution {
        LatencyDistribution {
            samples: self.0.len(),
            mean_ms: self.0.mean() / 1000.0,
            p50_ms: self.0.value_at_quantile(0.50) as f64 / 1000.0,
            p95_ms: self.0.value_at_quantile(0.95) as f64 / 1000.0,
            p99_ms: self.0.value_at_quantile(0.99) as f64 / 1000.0,
            max_ms: self.0.max() as f64 / 1000.0,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct LatencyDistribution {
    pub samples: u64,
    pub mean_ms: f64,
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
    pub max_ms: f64,
}

#[derive(Default)]
pub(crate) struct ManagerMeasurements {
    pub active_sessions: usize,
    pub peak_active_sessions: usize,
    pub admitted_sessions: u64,
    pub rejected_sessions: u64,
    pub closed_sessions: u64,
    pub timed_out_sessions: u64,
    pub inputs_accepted: u64,
    pub inputs_overloaded: u64,
    pub inputs_stale: u64,
    pub unknown_session_inputs: u64,
    pub worker_channel_saturation: u64,
    pub output_channel_saturation: u64,
    pub dropped_results: u64,
    pub stale_results: u64,
    pub delivered_results: u64,
    pub phase_added_latency: LatencyHistogram,
}

pub(crate) struct WorkerMeasurements {
    pub worker_id: WorkerId,
    pub peak_sessions: usize,
    pub batches: u64,
    pub processed_frames: u64,
    pub valid_results: u64,
    pub deadline_misses: u64,
    pub stale_results: u64,
    pub stale_inputs: u64,
    pub coalesced_inputs: u64,
    pub skipped_inference_ticks: u64,
    pub result_channel_saturation: u64,
    pub dropped_results: u64,
    pub busy_time: Duration,
    pub elapsed: Duration,
    pub batch_sizes: Histogram<u64>,
    pub queue_delay: LatencyHistogram,
    pub inference_latency: LatencyHistogram,
    pub end_to_end_latency: LatencyHistogram,
    pub deadline_lateness: LatencyHistogram,
}

impl WorkerMeasurements {
    pub(crate) fn new(worker_id: WorkerId) -> Self {
        Self {
            worker_id,
            peak_sessions: 0,
            batches: 0,
            processed_frames: 0,
            valid_results: 0,
            deadline_misses: 0,
            stale_results: 0,
            stale_inputs: 0,
            coalesced_inputs: 0,
            skipped_inference_ticks: 0,
            result_channel_saturation: 0,
            dropped_results: 0,
            busy_time: Duration::ZERO,
            elapsed: Duration::ZERO,
            batch_sizes: Histogram::new(3).expect("valid precision"),
            queue_delay: LatencyHistogram::default(),
            inference_latency: LatencyHistogram::default(),
            end_to_end_latency: LatencyHistogram::default(),
            deadline_lateness: LatencyHistogram::default(),
        }
    }
}

#[derive(Debug, Serialize)]
pub struct WorkerReport {
    pub worker_id: WorkerId,
    pub peak_sessions: usize,
    pub batches: u64,
    pub processed_frames: u64,
    pub valid_results: u64,
    pub deadline_misses: u64,
    pub stale_results_discarded: u64,
    pub mean_batch_size: f64,
    pub batch_size_p50: u64,
    pub batch_size_p95: u64,
    pub batch_size_max: u64,
    pub batch_fill_ratio: f64,
    pub utilization: f64,
    pub queue_delay: LatencyDistribution,
    pub inference_latency: LatencyDistribution,
    pub end_to_end_latency: LatencyDistribution,
}

#[derive(Debug, Serialize)]
pub struct Report {
    pub elapsed_secs: f64,
    pub active_sessions_at_shutdown: usize,
    pub peak_active_sessions: usize,
    pub admitted_sessions: u64,
    pub rejected_sessions: u64,
    pub closed_sessions: u64,
    pub timed_out_sessions: u64,
    pub inputs_accepted: u64,
    pub inputs_overloaded: u64,
    pub inputs_stale: u64,
    pub unknown_session_inputs: u64,
    pub coalesced_inputs: u64,
    pub skipped_inference_ticks: u64,
    pub processed_frames: u64,
    pub delivered_results: u64,
    pub throughput_frames_per_sec: f64,
    pub batches: u64,
    pub mean_batch_size: f64,
    pub batch_fill_ratio: f64,
    pub worker_utilization: f64,
    pub deadline_misses: u64,
    pub deadline_miss_ratio: f64,
    pub stale_results_discarded: u64,
    pub dropped_results: u64,
    pub channel_saturation_events: u64,
    pub ingress_channel_saturation: u64,
    pub worker_channel_saturation: u64,
    pub result_channel_saturation: u64,
    pub output_channel_saturation: u64,
    pub queue_delay: LatencyDistribution,
    pub inference_latency: LatencyDistribution,
    pub end_to_end_latency: LatencyDistribution,
    pub deadline_lateness: LatencyDistribution,
    pub phase_added_latency: LatencyDistribution,
    pub workers: Vec<WorkerReport>,
}

impl Report {
    pub(crate) fn assemble(
        manager: ManagerMeasurements,
        workers: Vec<WorkerMeasurements>,
        ingress_saturation: u64,
        elapsed: Duration,
        batch_size: usize,
    ) -> Self {
        let mut queue_delay = LatencyHistogram::default();
        let mut inference_latency = LatencyHistogram::default();
        let mut end_to_end_latency = LatencyHistogram::default();
        let mut deadline_lateness = LatencyHistogram::default();
        for worker in &workers {
            queue_delay.merge(&worker.queue_delay);
            inference_latency.merge(&worker.inference_latency);
            end_to_end_latency.merge(&worker.end_to_end_latency);
            deadline_lateness.merge(&worker.deadline_lateness);
        }
        let batches: u64 = workers.iter().map(|worker| worker.batches).sum();
        let processed_frames: u64 = workers.iter().map(|worker| worker.processed_frames).sum();
        let valid_results: u64 = workers.iter().map(|worker| worker.valid_results).sum();
        let deadline_misses: u64 = workers.iter().map(|worker| worker.deadline_misses).sum();
        let result_saturation: u64 = workers
            .iter()
            .map(|worker| worker.result_channel_saturation)
            .sum();
        let busy_secs: f64 = workers
            .iter()
            .map(|worker| worker.busy_time.as_secs_f64())
            .sum();
        let worker_secs: f64 = workers
            .iter()
            .map(|worker| worker.elapsed.as_secs_f64())
            .sum();
        Self {
            elapsed_secs: elapsed.as_secs_f64(),
            active_sessions_at_shutdown: manager.active_sessions,
            peak_active_sessions: manager.peak_active_sessions,
            admitted_sessions: manager.admitted_sessions,
            rejected_sessions: manager.rejected_sessions,
            closed_sessions: manager.closed_sessions,
            timed_out_sessions: manager.timed_out_sessions,
            inputs_accepted: manager.inputs_accepted,
            inputs_overloaded: manager.inputs_overloaded,
            inputs_stale: manager.inputs_stale
                + workers
                    .iter()
                    .map(|worker| worker.stale_inputs)
                    .sum::<u64>(),
            unknown_session_inputs: manager.unknown_session_inputs,
            coalesced_inputs: workers.iter().map(|worker| worker.coalesced_inputs).sum(),
            skipped_inference_ticks: workers
                .iter()
                .map(|worker| worker.skipped_inference_ticks)
                .sum(),
            processed_frames,
            delivered_results: manager.delivered_results,
            throughput_frames_per_sec: ratio(
                manager.delivered_results as f64,
                elapsed.as_secs_f64(),
            ),
            batches,
            mean_batch_size: ratio(processed_frames as f64, batches as f64),
            batch_fill_ratio: ratio(processed_frames as f64, batches as f64 * batch_size as f64),
            worker_utilization: ratio(busy_secs, worker_secs),
            deadline_misses,
            deadline_miss_ratio: ratio(deadline_misses as f64, valid_results as f64),
            stale_results_discarded: manager.stale_results
                + workers
                    .iter()
                    .map(|worker| worker.stale_results)
                    .sum::<u64>(),
            dropped_results: manager.dropped_results
                + workers
                    .iter()
                    .map(|worker| worker.dropped_results)
                    .sum::<u64>(),
            channel_saturation_events: ingress_saturation
                + manager.worker_channel_saturation
                + result_saturation
                + manager.output_channel_saturation,
            ingress_channel_saturation: ingress_saturation,
            worker_channel_saturation: manager.worker_channel_saturation,
            result_channel_saturation: result_saturation,
            output_channel_saturation: manager.output_channel_saturation,
            queue_delay: queue_delay.summary(),
            inference_latency: inference_latency.summary(),
            end_to_end_latency: end_to_end_latency.summary(),
            deadline_lateness: deadline_lateness.summary(),
            phase_added_latency: manager.phase_added_latency.summary(),
            workers: workers
                .into_iter()
                .map(|worker| WorkerReport {
                    worker_id: worker.worker_id,
                    peak_sessions: worker.peak_sessions,
                    batches: worker.batches,
                    processed_frames: worker.processed_frames,
                    valid_results: worker.valid_results,
                    deadline_misses: worker.deadline_misses,
                    stale_results_discarded: worker.stale_results,
                    mean_batch_size: worker.batch_sizes.mean(),
                    batch_size_p50: worker.batch_sizes.value_at_quantile(0.5),
                    batch_size_p95: worker.batch_sizes.value_at_quantile(0.95),
                    batch_size_max: worker.batch_sizes.max(),
                    batch_fill_ratio: ratio(
                        worker.processed_frames as f64,
                        worker.batches as f64 * batch_size as f64,
                    ),
                    utilization: ratio(
                        worker.busy_time.as_secs_f64(),
                        worker.elapsed.as_secs_f64(),
                    ),
                    queue_delay: worker.queue_delay.summary(),
                    inference_latency: worker.inference_latency.summary(),
                    end_to_end_latency: worker.end_to_end_latency.summary(),
                })
                .collect(),
        }
    }
}

fn ratio(numerator: f64, denominator: f64) -> f64 {
    if denominator == 0.0 {
        0.0
    } else {
        numerator / denominator
    }
}
