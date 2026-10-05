use bytes::Bytes;
use serde::Serialize;
use tokio::time::Instant;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
pub struct SessionId(pub u64);

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct WorkerId(pub usize);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Generation(pub u64);

#[derive(Debug)]
pub struct InputFrame {
    pub timestamp: Instant,
    pub payload: Bytes,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Assignment {
    pub worker_id: WorkerId,
    pub generation: Generation,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CreateOutcome {
    Admitted(Assignment),
    RejectedCapacity,
    AlreadyExists,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InputOutcome {
    Accepted,
    UnknownSession,
    Overloaded,
    Stale,
}

#[derive(Debug)]
pub struct InferenceOutput {
    pub session_id: SessionId,
    pub assignment: Assignment,
    pub input_timestamp: Instant,
    pub completed_at: Instant,
}
