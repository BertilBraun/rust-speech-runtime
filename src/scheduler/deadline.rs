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
}
