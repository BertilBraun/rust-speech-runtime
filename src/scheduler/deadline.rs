use super::{BatchKind, CostModel};
use crate::session::state::{Preparation, SessionState, Stage};
use std::collections::HashMap;
use tokio::time::Instant;

pub(crate) fn select_batch(
    sessions: &HashMap<String, SessionState>,
    maximum: usize,
    maximum_prefill: usize,
    consecutive_decode_batches: usize,
    costs: &CostModel,
    maximum_prefill_wait: std::time::Duration,
) -> Option<(BatchKind, Vec<String>)> {
    if let Some(batch) = control_batch(sessions, maximum) {
        return Some(batch);
    }
    let decode = ordered_candidates(sessions, |session| match session.stage {
        Stage::Generating { deadline, .. } if session.pending_token.is_some() => Some(deadline),
        _ => None,
    });
    let mut prefill = ordered_candidates(sessions, |session| {
        match (&session.stage, &session.preparation) {
            (Stage::Prefill { queued_at }, Preparation::None) => Some(*queued_at),
            (Stage::Capturing { .. } | Stage::Prefill { .. }, Preparation::Queued(request)) => {
                Some(request.queued_at)
            }
            _ => None,
        }
    });
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
                    .map(|(_, key)| prefill_context(&sessions[key]))
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

fn control_batch(
    sessions: &HashMap<String, SessionState>,
    maximum: usize,
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
    None
}

fn ordered_candidates(
    sessions: &HashMap<String, SessionState>,
    ready_at: fn(&SessionState) -> Option<Instant>,
) -> Vec<(Instant, String)> {
    let mut candidates = sessions
        .iter()
        .filter_map(|(key, session)| {
            if !session.opened || session.closing || session.in_flight {
                return None;
            }
            ready_at(session).map(|instant| (instant, key.clone()))
        })
        .collect::<Vec<_>>();
    candidates.sort();
    candidates
}

fn prefill_context(session: &SessionState) -> usize {
    let audio_tokens = session
        .turn()
        .expect("prefill turn")
        .audio_pcm16
        .len()
        .div_ceil(3200);
    session.context_tokens.saturating_add(audio_tokens + 32)
}
