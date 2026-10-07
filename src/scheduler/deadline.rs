//! Selects cache control work first, then balances decode deadlines with prefill waiting time.

use super::{BatchKind, CostModel};
use crate::session::state::{Preparation, SessionState, Stage};
use std::{collections::HashMap, time::Duration};
use tokio::time::Instant;

/// The ordering time is a decode deadline or a prefill enqueue time.
#[derive(Eq, PartialEq, Ord, PartialOrd)]
struct Candidate {
    priority_at: Instant,
    session_key: String,
}

struct DecodeBudget {
    deadline_slack_ms: f64,
    forward_duration_ms: f64,
}

pub(crate) fn select_batch(
    sessions: &HashMap<String, SessionState>,
    max_batch_size: usize,
    max_prefill_batch_size: usize,
    consecutive_decode_batches: usize,
    forward_costs: &CostModel,
    maximum_prefill_wait: Duration,
) -> Option<(BatchKind, Vec<String>)> {
    if let Some(batch) = control_batch(sessions, max_batch_size) {
        return Some(batch);
    }

    let decode_candidates = ordered_candidates(sessions, decode_deadline);
    let prefill_candidates = compatible_prefill_candidates(sessions);
    let decode_budget = decode_budget(sessions, &decode_candidates, max_batch_size, forward_costs);
    let prefill_size = select_prefill_size(
        sessions,
        &prefill_candidates,
        decode_budget,
        max_prefill_batch_size.min(max_batch_size),
        consecutive_decode_batches,
        forward_costs,
        maximum_prefill_wait,
    );

    match prefill_size {
        Some(size) => selected_batch(BatchKind::Prefill, prefill_candidates, size),
        None => selected_batch(BatchKind::Decode, decode_candidates, max_batch_size),
    }
}

fn compatible_prefill_candidates(sessions: &HashMap<String, SessionState>) -> Vec<Candidate> {
    let mut candidates = ordered_candidates(sessions, prefill_enqueue_time);
    let Some(oldest) = candidates.first() else {
        return candidates;
    };
    // A forward batches provisional candidates or committed utterances, never both.
    let oldest_is_preparation = matches!(
        sessions[&oldest.session_key].preparation,
        Preparation::Queued(_)
    );
    candidates.retain(|candidate| {
        let is_preparation = matches!(
            sessions[&candidate.session_key].preparation,
            Preparation::Queued(_)
        );
        is_preparation == oldest_is_preparation
    });
    candidates
}

fn decode_budget(
    sessions: &HashMap<String, SessionState>,
    candidates: &[Candidate],
    max_batch_size: usize,
    forward_costs: &CostModel,
) -> Option<DecodeBudget> {
    let earliest = candidates.first()?;
    let batch_size = candidates.len().min(max_batch_size);
    let context_tokens = candidates
        .iter()
        .take(batch_size)
        .map(|candidate| sessions[&candidate.session_key].context_tokens)
        .max()
        .unwrap_or(0);
    Some(DecodeBudget {
        forward_duration_ms: forward_costs.estimate(BatchKind::Decode, batch_size, context_tokens),
        deadline_slack_ms: earliest
            .priority_at
            .saturating_duration_since(Instant::now())
            .as_secs_f64()
            * 1000.0,
    })
}

fn select_prefill_size(
    sessions: &HashMap<String, SessionState>,
    candidates: &[Candidate],
    decode_budget: Option<DecodeBudget>,
    max_batch_size: usize,
    consecutive_decode_batches: usize,
    forward_costs: &CostModel,
    maximum_wait: Duration,
) -> Option<usize> {
    let oldest = candidates.first()?;
    for batch_size in (1..=candidates.len().min(max_batch_size)).rev() {
        let context_tokens = candidates
            .iter()
            .take(batch_size)
            .map(|candidate| prefill_context(&sessions[&candidate.session_key]))
            .max()
            .unwrap_or(0);
        let prefill_duration_ms =
            forward_costs.estimate(BatchKind::Prefill, batch_size, context_tokens);
        let Some(budget) = &decode_budget else {
            return Some(batch_size);
        };
        // Admitted prefills eventually run even when their forward exceeds decode slack.
        if oldest.priority_at.elapsed() >= maximum_wait {
            return Some(batch_size);
        }
        let both_forwards_fit =
            prefill_duration_ms + budget.forward_duration_ms < budget.deadline_slack_ms;
        let decode_should_yield =
            consecutive_decode_batches >= 4 && prefill_duration_ms < budget.deadline_slack_ms;
        if both_forwards_fit || decode_should_yield {
            return Some(batch_size);
        }
    }
    None
}

fn selected_batch(
    kind: BatchKind,
    candidates: Vec<Candidate>,
    max_batch_size: usize,
) -> Option<(BatchKind, Vec<String>)> {
    if candidates.is_empty() {
        return None;
    }
    let session_keys = candidates
        .into_iter()
        .take(max_batch_size)
        .map(|candidate| candidate.session_key)
        .collect();
    Some((kind, session_keys))
}

fn control_batch(
    sessions: &HashMap<String, SessionState>,
    max_batch_size: usize,
) -> Option<(BatchKind, Vec<String>)> {
    for kind in [
        BatchKind::Close,
        BatchKind::Discard,
        BatchKind::Open,
        BatchKind::Activate,
    ] {
        let session_keys: Vec<String> = sessions
            .iter()
            .filter(|(_, session)| control_operation_is_ready(kind, session))
            .take(max_batch_size)
            .map(|(session_key, _)| session_key.clone())
            .collect();
        if !session_keys.is_empty() {
            return Some((kind, session_keys));
        }
    }
    None
}

fn control_operation_is_ready(kind: BatchKind, session: &SessionState) -> bool {
    if session.in_flight {
        return false;
    }
    match kind {
        BatchKind::Close => session.closing && session.opened && !session.backend_closed,
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
        BatchKind::Prefill | BatchKind::Decode => false,
    }
}

fn ordered_candidates(
    sessions: &HashMap<String, SessionState>,
    priority_at: fn(&SessionState) -> Option<Instant>,
) -> Vec<Candidate> {
    let mut candidates = Vec::new();
    for (session_key, session) in sessions {
        if !session.opened || session.closing || session.in_flight {
            continue;
        }
        if let Some(priority_at) = priority_at(session) {
            candidates.push(Candidate {
                priority_at,
                session_key: session_key.clone(),
            });
        }
    }
    candidates.sort();
    candidates
}

fn decode_deadline(session: &SessionState) -> Option<Instant> {
    match session.stage {
        Stage::Generating { deadline, .. } if session.pending_token.is_some() => Some(deadline),
        _ => None,
    }
}

fn prefill_enqueue_time(session: &SessionState) -> Option<Instant> {
    match (&session.stage, &session.preparation) {
        (Stage::Prefill { queued_at }, Preparation::None) => Some(*queued_at),
        (Stage::Capturing { .. } | Stage::Prefill { .. }, Preparation::Queued(request)) => {
            Some(request.queued_at)
        }
        _ => None,
    }
}

fn prefill_context(session: &SessionState) -> usize {
    let audio_tokens = session
        .current_turn()
        .expect("prefill turn")
        .audio_pcm16
        .len()
        .div_ceil(3200);
    session.context_tokens.saturating_add(audio_tokens + 32)
}
