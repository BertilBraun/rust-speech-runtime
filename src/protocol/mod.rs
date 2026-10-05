use crate::config::AudioLimits;
use crate::metrics::profile::PacketTimings;
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tokio::time::Instant;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct SessionId(pub u64);
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct WorkerId(pub usize);
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct Generation(pub u64);
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PacketSequence(pub u64);

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PrefixState {
    pub packets: u64,
    pub bytes: u64,
    pub digest: [u8; 32],
}
impl PrefixState {
    pub fn append(self, payload: &[u8]) -> Self {
        let mut digest = blake3::Hasher::new();
        digest.update(&self.digest);
        digest.update(&(payload.len() as u64).to_le_bytes());
        digest.update(payload);
        Self {
            packets: self.packets + 1,
            bytes: self.bytes + payload.len() as u64,
            digest: *digest.finalize().as_bytes(),
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(transparent)]
pub struct AudioPrefix(pub Vec<Bytes>);
impl AudioPrefix {
    pub fn state(&self) -> PrefixState {
        self.0
            .iter()
            .fold(PrefixState::default(), |prefix, frame| prefix.append(frame))
    }
    pub fn byte_len(&self) -> usize {
        self.0.iter().map(Bytes::len).sum()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum AudioContext {
    Cached(PrefixState),
    Replay(AudioPrefix),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AudioPacket {
    pub sequence: PacketSequence,
    pub payload: Bytes,
    pub context: AudioContext,
}
#[derive(Debug)]
pub struct InputFrame {
    pub timestamp: Instant,
    pub deadline: Instant,
    pub packet: AudioPacket,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Assignment {
    pub worker_id: WorkerId,
    pub generation: Generation,
}
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct SessionLease {
    pub session_id: SessionId,
    pub generation: Generation,
}
#[derive(Clone, Copy, Debug)]
pub enum SessionTarget {
    Current(SessionId),
    Owned(SessionLease),
}
impl SessionTarget {
    pub fn session_id(self) -> SessionId {
        match self {
            Self::Current(id) => id,
            Self::Owned(lease) => lease.session_id,
        }
    }
    pub fn matches_generation(self, generation: Generation) -> bool {
        match self {
            Self::Current(_) => true,
            Self::Owned(lease) => lease.generation == generation,
        }
    }
}
impl From<SessionId> for SessionTarget {
    fn from(id: SessionId) -> Self {
        Self::Current(id)
    }
}
impl From<SessionLease> for SessionTarget {
    fn from(lease: SessionLease) -> Self {
        Self::Owned(lease)
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum CreateOutcome {
    Admitted(SessionAdmission),
    Rejected(CreateRejection),
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum CreateRejection {
    Capacity,
    AlreadyExists,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SessionAdmission {
    pub assignment: Assignment,
    pub audio_limits: AudioLimits,
    pub packet_deadline: Duration,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum FrameRejection {
    UnknownSession,
    Overloaded,
    DeadlineExceeded,
    InvalidSequence,
    InvalidPrefix,
    PrefixCapacity,
    ReplayTooExpensive,
    Cancelled,
    WorkerCapacityLost,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum CacheOutcome {
    Hit,
    Replayed { packets: u64, bytes: u64 },
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AudioResult {
    pub assignment: Assignment,
    pub sequence: PacketSequence,
    pub payload: Bytes,
    pub prefix: PrefixState,
    pub cache: CacheOutcome,
    pub timings: Box<PacketTimings>,
}
#[derive(Debug)]
pub struct InferenceOutput {
    pub session_id: SessionId,
    pub audio: AudioResult,
    pub input_timestamp: Instant,
    pub completed_at: Instant,
}
#[derive(Debug)]
pub enum InputOutcome {
    Processed(InferenceOutput),
    CacheMiss,
    Rejected(FrameRejection),
}
