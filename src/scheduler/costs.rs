//! Conservative recent forward costs indexed by workload, batch size and context bucket.

use super::BatchKind;
use crate::config::RuntimeConfig;
use std::collections::{HashMap, VecDeque};

#[derive(Clone, Copy, Eq, Hash, PartialEq)]
struct ForwardShape {
    kind: BatchKind,
    batch_size: usize,
    context_bucket: u32,
}

/// Keeps recent forward measurements; unseen shapes never assume proportional batch cost.
pub(crate) struct CostModel {
    initial_forward_ms: f64,
    duration_samples: HashMap<ForwardShape, VecDeque<f64>>,
}

impl CostModel {
    pub fn new(initial_forward_ms: f64) -> Self {
        Self {
            initial_forward_ms,
            duration_samples: HashMap::new(),
        }
    }

    pub fn observe(
        &mut self,
        kind: BatchKind,
        elapsed_ms: f64,
        batch_size: usize,
        context_tokens: usize,
    ) {
        let samples = self
            .duration_samples
            .entry(ForwardShape {
                kind,
                batch_size,
                context_bucket: context_bucket(context_tokens),
            })
            .or_default();
        if samples.len() == 32 {
            samples.pop_front();
        }
        samples.push_back(elapsed_ms.max(0.01));
    }

    pub fn estimate(&self, kind: BatchKind, batch_size: usize, context_tokens: usize) -> f64 {
        let shape = ForwardShape {
            kind,
            batch_size,
            context_bucket: context_bucket(context_tokens),
        };
        if let Some(samples) = self.duration_samples.get(&shape) {
            return conservative_duration_ms(samples);
        }
        let known_batch_duration_ms = self
            .duration_samples
            .iter()
            .filter(|(shape, _)| shape.kind == kind && shape.batch_size == batch_size)
            .map(|(_, samples)| conservative_duration_ms(samples))
            .reduce(f64::max);
        if let Some(duration_ms) = known_batch_duration_ms {
            // An unseen context bucket needs headroom even when batch size has been measured.
            return duration_ms * 2.0;
        }
        // Unknown batch sizes inherit a conservative duration, never a proportional fraction.
        self.duration_samples
            .iter()
            .filter(|(shape, _)| shape.kind == kind)
            .map(|(_, samples)| conservative_duration_ms(samples))
            .fold(self.initial_forward_ms, f64::max)
    }

    pub fn admits(
        &self,
        active_turn_count: usize,
        max_batch_size: usize,
        context_tokens: usize,
        configuration: &RuntimeConfig,
        reserve_prefill: bool,
    ) -> bool {
        let measured_decode_batch_size = self
            .duration_samples
            .keys()
            .filter(|shape| {
                shape.kind == BatchKind::Decode
                    && shape.batch_size <= max_batch_size
                    && shape.batch_size <= active_turn_count
            })
            .map(|shape| shape.batch_size)
            .max()
            .unwrap_or(1);
        let decode_batches_per_round = active_turn_count.div_ceil(measured_decode_batch_size);
        let decode_round_ms = self.estimate(
            BatchKind::Decode,
            measured_decode_batch_size,
            context_tokens,
        ) * decode_batches_per_round as f64;
        let prefill_allowance_ms = if reserve_prefill {
            self.estimate(BatchKind::Prefill, 1, context_tokens)
        } else {
            0.0
        };
        let token_interval_ms = 1000.0 / configuration.target_tokens_per_second;
        let compute_budget_ms = token_interval_ms * configuration.admission_headroom;
        decode_round_ms + prefill_allowance_ms <= compute_budget_ms
    }
}

fn context_bucket(context_tokens: usize) -> u32 {
    // Fourfold context ranges share sparse observations without claiming exact-cost equivalence.
    context_tokens.max(1).ilog2() / 2
}

fn conservative_duration_ms(samples: &VecDeque<f64>) -> f64 {
    samples.iter().copied().fold(0.0, f64::max) * 1.1
}
