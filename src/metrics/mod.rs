use crate::protocol::WorkerId;
use hdrhistogram::Histogram;
use serde::{Deserialize, Serialize};
use std::time::Duration;

pub mod profile;
use profile::{RuntimeLagReport, SlowWorkerPacket, WorkerProfile, WorkerProfileReport};

pub(crate) struct LatencyHistogram(Histogram<u64>);
impl Default for LatencyHistogram {
    fn default() -> Self {
        Self(Histogram::new_with_bounds(1, 86_400_000_000, 3).expect("valid bounds"))
    }
}
impl LatencyHistogram {
    pub(crate) fn record(&mut self, duration: Duration) {
        self.0
            .record(duration.as_micros().min(86_400_000_000) as u64)
            .expect("bounded sample");
    }
    pub(crate) fn merge(&mut self, other: &Self) {
        self.0.add(&other.0).expect("same bounds");
    }
    pub(crate) fn summary(&self) -> LatencyDistribution {
        LatencyDistribution {
            samples: self.0.len(),
            mean_ms: self.0.mean() / 1000.0,
            min_ms: self.0.min() as f64 / 1000.0,
            p50_ms: self.0.value_at_quantile(0.5) as f64 / 1000.0,
            p95_ms: self.0.value_at_quantile(0.95) as f64 / 1000.0,
            p99_ms: self.0.value_at_quantile(0.99) as f64 / 1000.0,
            max_ms: self.0.max() as f64 / 1000.0,
        }
    }
}
#[derive(Debug, Serialize, Deserialize)]
pub struct LatencyDistribution {
    pub samples: u64,
    pub mean_ms: f64,
    pub min_ms: f64,
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
    pub max_ms: f64,
}
#[derive(Clone, Debug, Default, Serialize)]
pub struct InferenceCounters {
    pub batches: u64,
    pub processed_frames: u64,
    pub delivered_frames: u64,
    pub rejected_frames: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub cache_evictions: u64,
    pub replayed_packets: u64,
    pub replayed_bytes: u64,
    pub deadline_misses: u64,
    pub stale_results: u64,
    pub capacity_terminations: u64,
    pub busy_rejections: u64,
    pub prefix_rejections: u64,
}
impl InferenceCounters {
    fn merge(&mut self, other: &Self) {
        self.batches += other.batches;
        self.processed_frames += other.processed_frames;
        self.delivered_frames += other.delivered_frames;
        self.rejected_frames += other.rejected_frames;
        self.cache_hits += other.cache_hits;
        self.cache_misses += other.cache_misses;
        self.cache_evictions += other.cache_evictions;
        self.replayed_packets += other.replayed_packets;
        self.replayed_bytes += other.replayed_bytes;
        self.deadline_misses += other.deadline_misses;
        self.stale_results += other.stale_results;
        self.capacity_terminations += other.capacity_terminations;
        self.busy_rejections += other.busy_rejections;
        self.prefix_rejections += other.prefix_rejections;
    }
}
#[derive(Default)]
pub(crate) struct ManagerMeasurements {
    pub active_sessions: usize,
    pub peak_active_sessions: usize,
    pub admitted_sessions: u64,
    pub rejected_sessions: u64,
    pub closed_sessions: u64,
    pub timed_out_sessions: u64,
    pub terminated_sessions: u64,
    pub worker_channel_saturation: u64,
    pub rejected_frames: u64,
    pub ingress_delay: LatencyHistogram,
    pub control_duration: LatencyHistogram,
}
pub(crate) struct WorkerMeasurements {
    pub worker_id: WorkerId,
    pub initial_session_limit: usize,
    pub final_session_limit: usize,
    pub service_time: Duration,
    pub peak_sessions: usize,
    pub counters: InferenceCounters,
    pub busy_time: Duration,
    pub elapsed: Duration,
    pub batch_sizes: Histogram<u64>,
    pub queue_delay: LatencyHistogram,
    pub inference_latency: LatencyHistogram,
    pub end_to_end_latency: LatencyHistogram,
    pub deadline_lateness: LatencyHistogram,
    pub calibration_latency: LatencyHistogram,
    pub profile: WorkerProfile,
}
impl WorkerMeasurements {
    pub(crate) fn new(worker_id: WorkerId) -> Self {
        Self {
            worker_id,
            initial_session_limit: 0,
            final_session_limit: 0,
            service_time: Duration::ZERO,
            peak_sessions: 0,
            counters: InferenceCounters::default(),
            busy_time: Duration::ZERO,
            elapsed: Duration::ZERO,
            batch_sizes: Histogram::new(3).expect("valid precision"),
            queue_delay: LatencyHistogram::default(),
            inference_latency: LatencyHistogram::default(),
            end_to_end_latency: LatencyHistogram::default(),
            deadline_lateness: LatencyHistogram::default(),
            calibration_latency: LatencyHistogram::default(),
            profile: WorkerProfile::default(),
        }
    }
}
#[derive(Debug, Serialize)]
pub struct WorkerReport {
    pub worker_id: WorkerId,
    pub initial_session_limit: usize,
    pub final_session_limit: usize,
    pub service_time_ms: f64,
    pub peak_sessions: usize,
    pub inference: InferenceCounters,
    pub mean_batch_size: f64,
    pub batch_size_p95: u64,
    pub batch_fill_ratio: f64,
    pub utilization: f64,
    pub queue_delay: LatencyDistribution,
    pub inference_latency: LatencyDistribution,
    pub calibration_latency: LatencyDistribution,
    pub profile: WorkerProfileReport,
    pub slowest_packets: Vec<SlowWorkerPacket>,
}
#[derive(Debug, Serialize)]
pub struct Report {
    pub runtime_lag: RuntimeLagReport,
    pub elapsed_secs: f64,
    pub active_sessions_at_shutdown: usize,
    pub peak_active_sessions: usize,
    pub admitted_sessions: u64,
    pub rejected_sessions: u64,
    pub closed_sessions: u64,
    pub timed_out_sessions: u64,
    pub terminated_sessions: u64,
    pub inference: InferenceCounters,
    pub throughput_frames_per_sec: f64,
    pub mean_batch_size: f64,
    pub batch_fill_ratio: f64,
    pub worker_utilization: f64,
    pub deadline_miss_ratio: f64,
    pub channel_saturation_events: u64,
    pub queue_delay: LatencyDistribution,
    pub inference_latency: LatencyDistribution,
    pub end_to_end_latency: LatencyDistribution,
    pub deadline_lateness: LatencyDistribution,
    pub ingress_delay: LatencyDistribution,
    pub control_duration: LatencyDistribution,
    pub workers: Vec<WorkerReport>,
}
impl Report {
    pub(crate) fn assemble(
        manager: ManagerMeasurements,
        workers: Vec<WorkerMeasurements>,
        ingress_saturation: u64,
        ingress_rejections: u64,
        elapsed: Duration,
        batch_size: usize,
        runtime_lag: RuntimeLagReport,
    ) -> Self {
        let mut counters = InferenceCounters::default();
        let mut queue_delay = LatencyHistogram::default();
        let mut inference_latency = LatencyHistogram::default();
        let mut end_to_end_latency = LatencyHistogram::default();
        let mut deadline_lateness = LatencyHistogram::default();
        for worker in &workers {
            counters.merge(&worker.counters);
            queue_delay.merge(&worker.queue_delay);
            inference_latency.merge(&worker.inference_latency);
            end_to_end_latency.merge(&worker.end_to_end_latency);
            deadline_lateness.merge(&worker.deadline_lateness);
        }
        counters.rejected_frames += manager.rejected_frames + ingress_rejections;
        let busy_secs: f64 = workers
            .iter()
            .map(|worker| worker.busy_time.as_secs_f64())
            .sum();
        let worker_secs: f64 = workers
            .iter()
            .map(|worker| worker.elapsed.as_secs_f64())
            .sum();
        Self {
            runtime_lag,
            elapsed_secs: elapsed.as_secs_f64(),
            active_sessions_at_shutdown: manager.active_sessions,
            peak_active_sessions: manager.peak_active_sessions,
            admitted_sessions: manager.admitted_sessions,
            rejected_sessions: manager.rejected_sessions,
            closed_sessions: manager.closed_sessions,
            timed_out_sessions: manager.timed_out_sessions,
            terminated_sessions: manager.terminated_sessions,
            throughput_frames_per_sec: ratio(
                counters.delivered_frames as f64,
                elapsed.as_secs_f64(),
            ),
            mean_batch_size: ratio(counters.processed_frames as f64, counters.batches as f64),
            batch_fill_ratio: ratio(
                counters.processed_frames as f64,
                counters.batches as f64 * batch_size as f64,
            ),
            deadline_miss_ratio: ratio(
                counters.deadline_misses as f64,
                counters.processed_frames as f64,
            ),
            inference: counters,
            worker_utilization: ratio(busy_secs, worker_secs),
            channel_saturation_events: ingress_saturation + manager.worker_channel_saturation,
            queue_delay: queue_delay.summary(),
            inference_latency: inference_latency.summary(),
            end_to_end_latency: end_to_end_latency.summary(),
            deadline_lateness: deadline_lateness.summary(),
            ingress_delay: manager.ingress_delay.summary(),
            control_duration: manager.control_duration.summary(),
            workers: workers
                .into_iter()
                .map(|worker| WorkerReport {
                    worker_id: worker.worker_id,
                    initial_session_limit: worker.initial_session_limit,
                    final_session_limit: worker.final_session_limit,
                    service_time_ms: worker.service_time.as_secs_f64() * 1000.0,
                    peak_sessions: worker.peak_sessions,
                    mean_batch_size: worker.batch_sizes.mean(),
                    batch_size_p95: worker.batch_sizes.value_at_quantile(0.95),
                    batch_fill_ratio: ratio(
                        worker.counters.processed_frames as f64,
                        worker.counters.batches as f64 * batch_size as f64,
                    ),
                    utilization: ratio(
                        worker.busy_time.as_secs_f64(),
                        worker.elapsed.as_secs_f64(),
                    ),
                    inference: worker.counters,
                    queue_delay: worker.queue_delay.summary(),
                    inference_latency: worker.inference_latency.summary(),
                    calibration_latency: worker.calibration_latency.summary(),
                    profile: worker.profile.report(),
                    slowest_packets: worker.profile.slowest_packets,
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
