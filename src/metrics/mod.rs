//! Cumulative counters and latency observations, separate from mutable scheduling state.
//! Histogram locks cover only synchronous record/snapshot operations and never span an await.

mod runtime_timing;

use runtime_timing::RuntimeTiming;
pub use runtime_timing::RuntimeTimingSnapshot;

use hdrhistogram::Histogram;
use serde::{Deserialize, Serialize};
use std::sync::{
    Mutex,
    atomic::{AtomicU64, Ordering},
};

/// Histogram sample count and latency quantiles in milliseconds.
/// An empty histogram reports count zero and zero-valued quantiles.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct LatencyDistribution {
    pub count: u64,
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
    pub max_ms: f64,
}

/// Cumulative observations since node startup, read independently of the worker actors.
/// Counter and histogram reads are individually synchronized, not one atomic transaction.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct MetricsSnapshot {
    pub active_sessions: u64,
    pub admitted_sessions: u64,
    pub rejected_sessions: u64,
    pub admitted_turns: u64,
    pub rejected_turns: u64,
    pub generated_tokens: u64,
    pub stale_results_discarded: u64,
    pub preparations_started: u64,
    pub preparations_activated: u64,
    pub preparations_discarded: u64,
    pub preparation_fallbacks: u64,
    pub channel_saturation_events: u64,
    pub token_deadline_misses: u64,
    pub backend_failures: u64,
    pub batches: u64,
    pub batch_items: u64,
    /// Worker-side time from definitive commit to its first accepted token.
    pub ttft: LatencyDistribution,
    pub token_gap: LatencyDistribution,
    pub queue_delay: LatencyDistribution,
    /// Full backend round-trip durations, including IPC; stage times come from Python.
    pub inference: LatencyDistribution,
    pub encode: LatencyDistribution,
    pub prefill: LatencyDistribution,
    pub decode: LatencyDistribution,
    pub runtime_timing: RuntimeTimingSnapshot,
    pub decode_batches: u64,
    pub mean_decode_batch_size: f64,
    pub decode_batch_fill_ratio: f64,
    pub workers: Vec<WorkerMetricsSnapshot>,
}

/// Per-worker backend occupancy and latest reported device-allocator memory in bytes.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct WorkerMetricsSnapshot {
    pub worker_id: usize,
    pub busy_ms: f64,
    pub observed_ms: f64,
    /// Backend request wall time divided by node lifetime, capped at one; not GPU SM utilization.
    pub utilization: f64,
    pub allocated_bytes: u64,
    pub reserved_bytes: u64,
}

struct WorkerMetrics {
    busy_microseconds: AtomicU64,
    allocated_bytes: AtomicU64,
    reserved_bytes: AtomicU64,
}

pub(crate) struct Metrics {
    pub active_sessions: AtomicU64,
    pub admitted_sessions: AtomicU64,
    pub rejected_sessions: AtomicU64,
    pub admitted_turns: AtomicU64,
    pub rejected_turns: AtomicU64,
    pub generated_tokens: AtomicU64,
    pub stale_results: AtomicU64,
    pub preparations_started: AtomicU64,
    pub preparations_activated: AtomicU64,
    pub preparations_discarded: AtomicU64,
    pub preparation_fallbacks: AtomicU64,
    pub saturation: AtomicU64,
    pub deadline_misses: AtomicU64,
    pub backend_failures: AtomicU64,
    pub batches: AtomicU64,
    pub batch_items: AtomicU64,
    pub ttft: Distribution,
    pub token_gap: Distribution,
    pub queue_delay: Distribution,
    pub inference: Distribution,
    pub encode: Distribution,
    pub prefill: Distribution,
    pub decode: Distribution,
    pub runtime_timing: RuntimeTiming,
    pub decode_batches: AtomicU64,
    pub decode_items: AtomicU64,
    pub decode_slots: AtomicU64,
    workers: Vec<WorkerMetrics>,
    started: std::time::Instant,
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new(0)
    }
}

impl Metrics {
    pub fn new(worker_count: usize) -> Self {
        Self {
            active_sessions: AtomicU64::new(0),
            admitted_sessions: AtomicU64::new(0),
            rejected_sessions: AtomicU64::new(0),
            admitted_turns: AtomicU64::new(0),
            rejected_turns: AtomicU64::new(0),
            generated_tokens: AtomicU64::new(0),
            stale_results: AtomicU64::new(0),
            preparations_started: AtomicU64::new(0),
            preparations_activated: AtomicU64::new(0),
            preparations_discarded: AtomicU64::new(0),
            preparation_fallbacks: AtomicU64::new(0),
            saturation: AtomicU64::new(0),
            deadline_misses: AtomicU64::new(0),
            backend_failures: AtomicU64::new(0),
            batches: AtomicU64::new(0),
            batch_items: AtomicU64::new(0),
            ttft: Distribution::default(),
            token_gap: Distribution::default(),
            queue_delay: Distribution::default(),
            inference: Distribution::default(),
            encode: Distribution::default(),
            prefill: Distribution::default(),
            decode: Distribution::default(),
            runtime_timing: RuntimeTiming::default(),
            decode_batches: AtomicU64::new(0),
            decode_items: AtomicU64::new(0),
            decode_slots: AtomicU64::new(0),
            workers: (0..worker_count)
                .map(|_| WorkerMetrics {
                    busy_microseconds: AtomicU64::new(0),
                    allocated_bytes: AtomicU64::new(0),
                    reserved_bytes: AtomicU64::new(0),
                })
                .collect(),
            started: std::time::Instant::now(),
        }
    }

    pub fn worker_observation(
        &self,
        worker_id: usize,
        elapsed_ms: f64,
        allocated_bytes: u64,
        reserved_bytes: u64,
    ) {
        let worker = &self.workers[worker_id];
        worker
            .busy_microseconds
            .fetch_add((elapsed_ms * 1000.0) as u64, Ordering::Relaxed);
        worker
            .allocated_bytes
            .store(allocated_bytes, Ordering::Relaxed);
        worker
            .reserved_bytes
            .store(reserved_bytes, Ordering::Relaxed);
    }

    pub fn snapshot(&self) -> MetricsSnapshot {
        let read = |counter: &AtomicU64| counter.load(Ordering::Relaxed);
        let batches = read(&self.decode_batches);
        let items = read(&self.decode_items);
        let slots = read(&self.decode_slots);
        let observed_ms = self.started.elapsed().as_secs_f64() * 1000.0;
        MetricsSnapshot {
            active_sessions: read(&self.active_sessions),
            admitted_sessions: read(&self.admitted_sessions),
            rejected_sessions: read(&self.rejected_sessions),
            admitted_turns: read(&self.admitted_turns),
            rejected_turns: read(&self.rejected_turns),
            generated_tokens: read(&self.generated_tokens),
            stale_results_discarded: read(&self.stale_results),
            preparations_started: read(&self.preparations_started),
            preparations_activated: read(&self.preparations_activated),
            preparations_discarded: read(&self.preparations_discarded),
            preparation_fallbacks: read(&self.preparation_fallbacks),
            channel_saturation_events: read(&self.saturation),
            token_deadline_misses: read(&self.deadline_misses),
            backend_failures: read(&self.backend_failures),
            batches: read(&self.batches),
            batch_items: read(&self.batch_items),
            ttft: self.ttft.snapshot(),
            token_gap: self.token_gap.snapshot(),
            queue_delay: self.queue_delay.snapshot(),
            inference: self.inference.snapshot(),
            encode: self.encode.snapshot(),
            prefill: self.prefill.snapshot(),
            decode: self.decode.snapshot(),
            runtime_timing: self.runtime_timing.snapshot(),
            decode_batches: batches,
            mean_decode_batch_size: if batches > 0 {
                items as f64 / batches as f64
            } else {
                0.0
            },
            decode_batch_fill_ratio: if slots > 0 {
                items as f64 / slots as f64
            } else {
                0.0
            },
            workers: self
                .workers
                .iter()
                .enumerate()
                .map(|(worker_id, worker)| {
                    let busy_ms = read(&worker.busy_microseconds) as f64 / 1000.0;
                    WorkerMetricsSnapshot {
                        worker_id,
                        busy_ms,
                        observed_ms,
                        utilization: if observed_ms > 0.0 {
                            (busy_ms / observed_ms).min(1.0)
                        } else {
                            0.0
                        },
                        allocated_bytes: read(&worker.allocated_bytes),
                        reserved_bytes: read(&worker.reserved_bytes),
                    }
                })
                .collect(),
        }
    }
}

pub(crate) struct Distribution(Mutex<Histogram<u64>>);
impl Default for Distribution {
    fn default() -> Self {
        Self(Mutex::new(
            Histogram::new_with_bounds(1, 86_400_000_000, 3).expect("valid histogram"),
        ))
    }
}

impl Distribution {
    pub fn record(&self, milliseconds: f64) {
        let microseconds = (milliseconds * 1000.0).clamp(1.0, 86_400_000_000.0) as u64;
        let _ = self
            .0
            .lock()
            .expect("metrics lock poisoned")
            .record(microseconds);
    }

    fn snapshot(&self) -> LatencyDistribution {
        let histogram = self.0.lock().expect("metrics lock poisoned");
        if histogram.is_empty() {
            return LatencyDistribution::default();
        }
        LatencyDistribution {
            count: histogram.len(),
            p50_ms: histogram.value_at_quantile(0.50) as f64 / 1000.0,
            p95_ms: histogram.value_at_quantile(0.95) as f64 / 1000.0,
            p99_ms: histogram.value_at_quantile(0.99) as f64 / 1000.0,
            max_ms: histogram.max() as f64 / 1000.0,
        }
    }
}
