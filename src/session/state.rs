use tokio::time::Instant;

use crate::protocol::Assignment;

pub(super) struct SessionState {
    pub assignment: Assignment,
    pub last_input_at: Instant,
}
