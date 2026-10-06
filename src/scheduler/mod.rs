use crate::session::state::{SessionState, Stage};
use std::collections::{HashMap, VecDeque};
use tokio::time::Instant;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum BatchKind {
    Open,
    Close,
    Prefill,
    Decode,
}
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
        interval_ms: f64,
        headroom: f64,
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
        let prefill = self.estimate(BatchKind::Prefill, 1, context);
        decode + prefill <= interval_ms * headroom
    }
}
fn bucket(context: usize) -> u32 {
    context.max(1).ilog2() / 2
}
fn conservative(samples: &VecDeque<f64>) -> f64 {
    samples.iter().copied().fold(0.0, f64::max) * 1.1
}

pub(crate) fn select_batch(
    sessions: &HashMap<String, SessionState>,
    maximum: usize,
    consecutive_decode_batches: usize,
    costs: &CostModel,
) -> Option<(BatchKind, Vec<String>)> {
    for kind in [BatchKind::Close, BatchKind::Open] {
        let keys = sessions
            .iter()
            .filter(|(_, session)| {
                !session.in_flight
                    && match kind {
                        BatchKind::Close => {
                            session.closing && session.opened && !session.backend_closed
                        }
                        BatchKind::Open => !session.closing && !session.opened,
                        _ => false,
                    }
            })
            .map(|(key, _)| key.clone())
            .take(maximum)
            .collect::<Vec<_>>();
        if !keys.is_empty() {
            return Some((kind, keys));
        }
    }
    let mut decode = sessions
        .iter()
        .filter_map(|(key, session)| {
            if !session.opened || session.closing || session.in_flight {
                return None;
            }
            match session.stage {
                Stage::Generating { deadline, .. } if session.pending_token.is_some() => {
                    Some((deadline, key.clone()))
                }
                _ => None,
            }
        })
        .collect::<Vec<_>>();
    let mut prefill = sessions
        .iter()
        .filter_map(|(key, session)| {
            if !session.opened || session.closing || session.in_flight {
                return None;
            }
            match session.stage {
                Stage::Prefill { queued_at } => Some((queued_at, key.clone())),
                _ => None,
            }
        })
        .collect::<Vec<_>>();
    decode.sort();
    prefill.sort();
    let choose_prefill = if let Some((_, key)) = prefill.first() {
        let session = &sessions[key];
        let audio_tokens = session
            .turn()
            .expect("prefill turn")
            .audio_pcm16
            .len()
            .div_ceil(3200);
        let context = session.context_tokens.saturating_add(audio_tokens + 32);
        let predicted_prefill = costs.estimate(BatchKind::Prefill, 1, context);
        let decode_context = decode
            .iter()
            .take(maximum)
            .map(|(_, key)| sessions[key].context_tokens)
            .max()
            .unwrap_or(0);
        let predicted_decode = costs.estimate(
            BatchKind::Decode,
            decode.len().min(maximum).max(1),
            decode_context,
        );
        let available = decode
            .first()
            .map(|(deadline, _)| {
                deadline
                    .saturating_duration_since(Instant::now())
                    .as_secs_f64()
                    * 1000.0
            })
            .unwrap_or(f64::INFINITY);
        decode.is_empty()
            || predicted_prefill + predicted_decode < available
            || (consecutive_decode_batches >= 4 && predicted_prefill < available)
    } else {
        false
    };
    let (kind, selected) = if choose_prefill {
        (BatchKind::Prefill, prefill)
    } else {
        (BatchKind::Decode, decode)
    };
    if selected.is_empty() {
        None
    } else {
        Some((
            kind,
            selected
                .into_iter()
                .take(if kind == BatchKind::Prefill {
                    1
                } else {
                    maximum
                })
                .map(|(_, key)| key)
                .collect(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn unknown_shape_does_not_assume_linear_batch_scaling() {
        let mut costs = CostModel::new(100.0);
        assert!(costs.admits(1, 16, 32, 250.0, 0.8));
        assert!(!costs.admits(2, 16, 32, 250.0, 0.8));
        costs.observe(BatchKind::Decode, 5.0, 1, 32);
        costs.observe(BatchKind::Prefill, 10.0, 1, 32);
        assert_eq!(costs.estimate(BatchKind::Decode, 8, 32), 100.0);
        assert!(costs.admits(16, 16, 32, 250.0, 0.8));
        costs.observe(BatchKind::Decode, 120.0, 8, 32);
        assert!(!costs.admits(16, 16, 32, 250.0, 0.8));
    }
    #[test]
    fn context_costs_and_slow_recent_batches_remain_conservative() {
        let mut costs = CostModel::new(100.0);
        costs.observe(BatchKind::Decode, 10.0, 4, 32);
        costs.observe(BatchKind::Decode, 50.0, 4, 1024);
        costs.observe(BatchKind::Decode, 1.0, 4, 32);
        assert_eq!(costs.estimate(BatchKind::Decode, 4, 32), 11.0);
        assert_eq!(
            costs.estimate(BatchKind::Decode, 4, 1024),
            55.00000000000001
        );
    }
}
