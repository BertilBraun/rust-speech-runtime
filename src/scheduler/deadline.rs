use crate::protocol::SessionId;
use tokio::time::Instant;
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ReadySession {
    pub session_id: SessionId,
    pub deadline: Instant,
}
pub(crate) fn construct_batch(
    mut ready: Vec<ReadySession>,
    batch_size: usize,
) -> Vec<ReadySession> {
    ready.sort_unstable_by_key(|session| (session.deadline, session.session_id));
    ready.truncate(batch_size);
    ready
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    #[test]
    fn edf_batches_are_reconstructed_with_stable_ties() {
        let now = Instant::now();
        let selected = construct_batch(
            vec![
                ReadySession {
                    session_id: SessionId(3),
                    deadline: now + Duration::from_millis(1),
                },
                ReadySession {
                    session_id: SessionId(2),
                    deadline: now,
                },
                ReadySession {
                    session_id: SessionId(1),
                    deadline: now,
                },
            ],
            2,
        );
        assert_eq!(
            selected
                .iter()
                .map(|item| item.session_id)
                .collect::<Vec<_>>(),
            vec![SessionId(1), SessionId(2)]
        );
    }
}
