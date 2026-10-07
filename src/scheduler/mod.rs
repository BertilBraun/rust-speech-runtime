use crate::{
    config::RuntimeConfig,
    session::state::{Preparation, SessionState, Stage},
};
use std::collections::{HashMap, VecDeque};
use tokio::time::Instant;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum BatchKind {
    Open,
    Close,
    Discard,
    Activate,
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

pub(crate) fn select_batch(
    sessions: &HashMap<String, SessionState>,
    maximum: usize,
    maximum_prefill: usize,
    consecutive_decode_batches: usize,
    costs: &CostModel,
    maximum_prefill_wait: std::time::Duration,
) -> Option<(BatchKind, Vec<String>)> {
    for kind in [
        BatchKind::Close,
        BatchKind::Discard,
        BatchKind::Open,
        BatchKind::Activate,
    ] {
        let keys = sessions
            .iter()
            .filter(|(_, session)| {
                !session.in_flight
                    && match kind {
                        BatchKind::Close => {
                            session.closing && session.opened && !session.backend_closed
                        }
                        BatchKind::Open => !session.closing && !session.opened,
                        BatchKind::Discard => {
                            !session.closing
                                && session.opened
                                && matches!(session.preparation, Preparation::DiscardPending { .. })
                        }
                        BatchKind::Activate => {
                            !session.closing
                                && session.opened
                                && matches!(session.stage, Stage::Prefill { .. })
                                && matches!(session.preparation, Preparation::Ready(_))
                        }
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
            match (&session.stage, &session.preparation) {
                (Stage::Prefill { queued_at }, Preparation::None) => {
                    Some((*queued_at, key.clone()))
                }
                (Stage::Capturing { .. } | Stage::Prefill { .. }, Preparation::Queued(request)) => {
                    Some((request.queued_at, key.clone()))
                }
                _ => None,
            }
        })
        .collect::<Vec<_>>();
    decode.sort();
    prefill.sort();
    if let Some((_, oldest)) = prefill.first() {
        let preparing = matches!(sessions[oldest].preparation, Preparation::Queued(_));
        prefill.retain(|(_, key)| {
            matches!(sessions[key].preparation, Preparation::Queued(_)) == preparing
        });
    }
    let prefill_size = if let Some((queued_at, _)) = prefill.first() {
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
        (1..=prefill.len().min(maximum_prefill).min(maximum))
            .rev()
            .find(|size| {
                let context = prefill
                    .iter()
                    .take(*size)
                    .map(|(_, key)| {
                        let session = &sessions[key];
                        let audio_tokens = session
                            .turn()
                            .expect("prefill turn")
                            .audio_pcm16
                            .len()
                            .div_ceil(3200);
                        session.context_tokens.saturating_add(audio_tokens + 32)
                    })
                    .max()
                    .unwrap_or(0);
                let predicted_prefill = costs.estimate(BatchKind::Prefill, *size, context);
                decode.is_empty()
                    || queued_at.elapsed() >= maximum_prefill_wait
                    || predicted_prefill + predicted_decode < available
                    || (consecutive_decode_batches >= 4 && predicted_prefill < available)
            })
    } else {
        None
    };
    let (kind, selected, size) = if let Some(size) = prefill_size {
        (BatchKind::Prefill, prefill, size)
    } else {
        (BatchKind::Decode, decode, maximum)
    };
    if selected.is_empty() {
        None
    } else {
        Some((
            kind,
            selected
                .into_iter()
                .take(size)
                .map(|(_, key)| key)
                .collect(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        protocol::{SessionId, TurnId, TurnRecord, backend::AcceptedToken},
        session::state::{CaptureSnapshot, PreparationRequest},
    };
    use tokio::sync::mpsc;
    use tokio_util::sync::CancellationToken;

    fn session(key: &str, queued_at: Instant, preparing: bool) -> SessionState {
        let (events, _) = mpsc::channel(4);
        let mut session = SessionState::new(
            SessionId(key.into()),
            0,
            "test".into(),
            events,
            CancellationToken::new(),
        );
        session.opened = true;
        session.record.turns.push(TurnRecord {
            turn_id: TurnId(1),
            audio_pcm16: vec![0; 3200],
            tokens: Vec::new(),
            finish_reason: None,
            committed: !preparing,
        });
        if preparing {
            session.stage = Stage::Capturing { chunks: 1 };
            session.preparation = Preparation::Queued(PreparationRequest {
                snapshot: CaptureSnapshot {
                    turn_id: TurnId(1),
                    generation: 0,
                    chunk_count: 1,
                    sample_count: 1600,
                },
                queued_at,
            });
        } else {
            session.stage = Stage::Prefill { queued_at };
        }
        session
    }

    #[test]
    fn overlapping_preparations_and_commits_select_homogeneous_oldest_class() {
        for oldest_preparing in [true, false] {
            let now = Instant::now();
            let mut sessions = HashMap::from([
                ("oldest".into(), session("oldest", now, oldest_preparing)),
                (
                    "other-class".into(),
                    session(
                        "other-class",
                        now + std::time::Duration::from_millis(1),
                        !oldest_preparing,
                    ),
                ),
                (
                    "same-class".into(),
                    session(
                        "same-class",
                        now + std::time::Duration::from_millis(2),
                        oldest_preparing,
                    ),
                ),
            ]);
            let costs = CostModel::new(100.0);
            let selected = select_batch(
                &sessions,
                16,
                4,
                0,
                &costs,
                std::time::Duration::from_millis(100),
            )
            .unwrap();
            assert_eq!(
                selected,
                (
                    BatchKind::Prefill,
                    vec!["oldest".into(), "same-class".into()]
                )
            );
            sessions.remove("oldest");
            sessions.remove("same-class");
            assert_eq!(
                select_batch(
                    &sessions,
                    16,
                    4,
                    0,
                    &costs,
                    std::time::Duration::from_millis(100)
                )
                .unwrap(),
                (BatchKind::Prefill, vec!["other-class".into()])
            );
        }
    }

    #[test]
    fn prefill_batch_size_uses_measured_shape_and_decode_deadline_budget() {
        let now = Instant::now();
        let mut generator = session("generator", now, false);
        generator.stage = Stage::Generating {
            last_token_at: now,
            deadline: now + std::time::Duration::from_millis(120),
        };
        generator.pending_token = Some(AcceptedToken {
            turn_id: 1,
            index: 0,
            token_id: 100,
        });
        let sessions = HashMap::from([
            ("first".into(), session("first", now, false)),
            ("second".into(), session("second", now, false)),
            ("generator".into(), generator),
        ]);
        let mut costs = CostModel::new(100.0);
        costs.observe(BatchKind::Prefill, 20.0, 1, 33);
        costs.observe(BatchKind::Prefill, 300.0, 2, 33);
        costs.observe(BatchKind::Decode, 30.0, 1, 0);
        let selected = select_batch(
            &sessions,
            16,
            4,
            0,
            &costs,
            std::time::Duration::from_millis(100),
        )
        .unwrap();
        assert_eq!(selected.0, BatchKind::Prefill);
        assert_eq!(selected.1.len(), 1);
    }

    #[test]
    fn prepared_activation_runs_before_another_model_forward() {
        let now = Instant::now();
        let mut ready = session("ready", now, true);
        let snapshot = ready.preparation.snapshot().unwrap();
        ready.preparation = Preparation::Ready(snapshot);
        ready.stage = Stage::Prefill { queued_at: now };
        let sessions = HashMap::from([
            ("ready".into(), ready),
            ("ordinary".into(), session("ordinary", now, false)),
        ]);
        assert_eq!(
            select_batch(
                &sessions,
                16,
                4,
                0,
                &CostModel::new(100.0),
                std::time::Duration::from_millis(100)
            )
            .unwrap(),
            (BatchKind::Activate, vec!["ready".into()])
        );
    }

    #[test]
    fn unknown_shape_does_not_assume_linear_batch_scaling() {
        let mut costs = CostModel::new(100.0);
        let config = RuntimeConfig::default();
        assert!(costs.admits(1, 16, 32, &config, true));
        assert!(!costs.admits(2, 16, 32, &config, true));
        costs.observe(BatchKind::Decode, 5.0, 1, 32);
        costs.observe(BatchKind::Prefill, 10.0, 1, 32);
        assert_eq!(costs.estimate(BatchKind::Decode, 8, 32), 100.0);
        assert!(costs.admits(16, 16, 32, &config, true));
        costs.observe(BatchKind::Decode, 120.0, 8, 32);
        assert!(!costs.admits(16, 16, 32, &config, true));
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
    #[test]
    fn slow_prefill_does_not_permanently_reject_solo_turns() {
        let mut costs = CostModel::new(100.0);
        let config = RuntimeConfig::default();
        costs.observe(BatchKind::Prefill, 300.0, 1, 32);
        costs.observe(BatchKind::Decode, 20.0, 1, 32);
        assert!(costs.admits(1, 16, 32, &config, false));
        assert!(!costs.admits(2, 16, 32, &config, true));
    }
}
