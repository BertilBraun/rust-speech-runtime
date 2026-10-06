use super::{
    cache::CacheHandle,
    mock_gpu::{SessionCancellation, WorkItem},
};
use crate::protocol::{Assignment, PrefixState};

// One inline frame per bounded session avoids allocating a box on every audio packet.
#[allow(clippy::large_enum_variant)]
pub(super) enum SessionWork {
    Idle,
    Ready(WorkItem),
    Submitted,
}
impl SessionWork {
    pub(super) fn ready(&self) -> Option<&WorkItem> {
        match self {
            Self::Ready(frame) => Some(frame),
            Self::Idle | Self::Submitted => None,
        }
    }
    pub(super) fn take_ready(&mut self) -> WorkItem {
        let Self::Ready(frame) = std::mem::replace(self, Self::Idle) else {
            panic!("only selected ready frames are consumed");
        };
        frame
    }
}

pub(super) struct WorkerSession {
    pub assignment: Assignment,
    pub cache: CacheHandle,
    pub prefix: PrefixState,
    pub work: SessionWork,
    pub cancellation: SessionCancellation,
}
