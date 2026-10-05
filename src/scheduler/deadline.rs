use std::time::Duration;
use tokio::time::Instant;

use crate::protocol::SessionId;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReadySession {
    pub session_id: SessionId,
    pub deadline: Instant,
}

pub fn construct_batch(mut ready: Vec<ReadySession>, batch_size: usize) -> Vec<ReadySession> {
    ready.sort_unstable_by_key(|session| (session.deadline, session.session_id));
    ready.truncate(batch_size);
    ready
}

pub(crate) fn advance_deadline(
    deadline: Instant,
    input_timestamp: Instant,
    tick: Duration,
) -> (Instant, u64) {
    let phase_start = deadline - tick;
    let rounded_elapsed = input_timestamp.saturating_duration_since(phase_start) + tick / 2;
    let remainder = rounded_elapsed.as_nanos() % tick.as_nanos();
    let advance = rounded_elapsed - Duration::from_nanos(remainder as u64);
    (
        deadline + advance,
        (advance.as_nanos() / tick.as_nanos()) as u64,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn batches_earliest_deadlines_with_deterministic_ties() {
        let now = Instant::now();
        let ready = vec![
            ReadySession {
                session_id: SessionId(3),
                deadline: now + Duration::from_millis(2),
            },
            ReadySession {
                session_id: SessionId(2),
                deadline: now,
            },
            ReadySession {
                session_id: SessionId(1),
                deadline: now,
            },
        ];
        let batch = construct_batch(ready, 2);
        assert_eq!(
            batch.iter().map(|item| item.session_id).collect::<Vec<_>>(),
            vec![SessionId(1), SessionId(2)]
        );
        assert!(construct_batch(Vec::new(), 16).is_empty());
    }

    #[test]
    fn fresh_input_skips_obsolete_ticks_without_drifting_phase() {
        let origin = Instant::now();
        let tick = Duration::from_millis(50);
        for (deadline_ms, input_ms, expected_ms, skipped) in [
            (50, 0, 50, 0),
            (50, 48, 100, 1),
            (50, 52, 100, 1),
            (50, 148, 200, 3),
            (100, 48, 100, 0),
        ] {
            assert_eq!(
                advance_deadline(
                    origin + Duration::from_millis(deadline_ms),
                    origin + Duration::from_millis(input_ms),
                    tick
                ),
                (origin + Duration::from_millis(expected_ms), skipped)
            );
        }
    }
}
