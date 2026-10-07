use super::BatchKind;
use crate::config::RuntimeConfig;
use std::collections::{HashMap, VecDeque};

#[derive(Clone, Copy, Eq, Hash, PartialEq)]
struct Shape {
    kind: BatchKind,
    items: usize,
    context_bucket: u32,
}
pub(crate) struct CostModel {
    initial_ms: f64,
    samples: HashMap<Shape, VecDeque<f64>>,
}
impl CostModel {
    pub fn new(initial_ms: f64) -> Self {
        Self {
            initial_ms,
            samples: HashMap::new(),
        }
    }
    pub fn observe(&mut self, kind: BatchKind, elapsed_ms: f64, items: usize, context: usize) {
        let samples = self
            .samples
            .entry(Shape {
                kind,
                items,
                context_bucket: bucket(context),
            })
            .or_default();
        if samples.len() == 32 {
            samples.pop_front();
        }
        samples.push_back(elapsed_ms.max(0.01));
    }
    pub fn estimate(&self, kind: BatchKind, items: usize, context: usize) -> f64 {
        let shape = Shape {
            kind,
            items,
            context_bucket: bucket(context),
        };
        if let Some(samples) = self.samples.get(&shape) {
            return conservative(samples);
        }
        let same_batch = self
            .samples
            .iter()
            .filter(|(shape, _)| shape.kind == kind && shape.items == items)
            .map(|(_, samples)| conservative(samples))
            .reduce(f64::max);
        if let Some(measured) = same_batch {
            return measured * 2.0;
        }
        self.samples
            .iter()
            .filter(|(shape, _)| shape.kind == kind)
            .map(|(_, samples)| conservative(samples))
            .fold(self.initial_ms, f64::max)
    }
    pub fn admits(
        &self,
        active: usize,
        maximum_batch: usize,
        context: usize,
        config: &RuntimeConfig,
        reserve_prefill: bool,
    ) -> bool {
        let measured = self
            .samples
            .keys()
            .filter(|shape| shape.kind == BatchKind::Decode && shape.items <= maximum_batch)
            .map(|shape| shape.items)
            .max()
            .unwrap_or(1);
        let batches = active.div_ceil(measured);
        let decode = self.estimate(BatchKind::Decode, measured, context) * batches as f64;
        let prefill = if reserve_prefill {
            self.estimate(BatchKind::Prefill, 1, context)
        } else {
            0.0
        };
        decode + prefill <= (1000.0 / config.target_tokens_per_second) * config.admission_headroom
    }
}
fn bucket(context: usize) -> u32 {
    context.max(1).ilog2() / 2
}
fn conservative(samples: &VecDeque<f64>) -> f64 {
    samples.iter().copied().fold(0.0, f64::max) * 1.1
}
