//! Canonical session commands, events and archive records.
//!
//! Events describe actor acceptance, not acknowledgement of network delivery.
//! IDs, token order and generations are retained in archives to audit interruptions.

pub mod backend;
use serde::{Deserialize, Serialize};

/// Caller-chosen conversation identity, unique among live sessions on this node.
/// Admission requires a nonempty string of at most 128 bytes.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SessionId(pub String);

/// Caller-chosen turn identity, unique for the lifetime of its session.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct TurnId(pub u64);

/// Stable failure categories shared by admission, transport and backend responses.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorCode {
    InvalidInput,
    SessionExists,
    SessionNotFound,
    TurnNotFound,
    InvalidState,
    CapacityExceeded,
    ContextLimit,
    HistoryLimit,
    ChannelSaturated,
    BackendUnavailable,
    BackendFailed,
    SlowConsumer,
    Shutdown,
}

/// Why a turn stopped producing accepted output; cancellation keeps its accepted prefix.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Eos,
    TokenLimit,
    Cancelled,
    ContextLimit,
    BackendFailed,
    SlowConsumer,
}

/// FIFO notifications emitted by the assigned worker.
///
/// Continue consuming them during capture and generation. Saturating the bounded
/// event queue stops the session rather than buffering output indefinitely.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum SessionEvent {
    /// Backend state exists and the worker assignment is fixed.
    Opened {
        session_id: SessionId,
        worker_id: usize,
    },
    /// Backend state has been released; already accepted output precedes this event.
    Closed { session_id: SessionId },
    /// Compute capacity has been reserved for capture of this turn.
    Accepted { turn_id: TurnId },
    /// This audio snapshot is ready privately; no text is visible until commit.
    Prepared {
        turn_id: TurnId,
        chunk_count: u32,
        sample_count: usize,
    },
    /// One accepted model token, including EOS and tokens with empty decoded text.
    /// Sequence is zero-based within the turn; generation fences interrupted work.
    TextDelta {
        turn_id: TurnId,
        generation: u64,
        sequence: u64,
        token_id: u32,
        text: String,
    },
    /// The turn stopped; generated_tokens counts every accepted model token.
    Finished {
        turn_id: TurnId,
        generation: u64,
        reason: FinishReason,
        generated_tokens: usize,
    },
    /// Rejected input or terminal runtime failure, identified by a stable code.
    Failed {
        turn_id: Option<TurnId>,
        code: ErrorCode,
        message: String,
    },
}

/// An accepted token and its elapsed milliseconds since definitive turn commit.
/// Stored even if the client disconnects before receiving its text delta.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TokenRecord {
    pub token_id: u32,
    pub text: String,
    pub index: u64,
    pub elapsed_ms: f64,
}

/// Original PCM16 input and accepted output for one user/assistant pair.
/// Partial or cancelled turns remain present; committed records definitive end-of-turn.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TurnRecord {
    pub turn_id: TurnId,
    pub audio_pcm16: Vec<u8>,
    pub tokens: Vec<TokenRecord>,
    pub finish_reason: Option<FinishReason>,
    pub committed: bool,
}

/// Bounded in-memory conversation audit returned by explicit session close.
/// This contains original audio and accepted tokens, not the GPU conversation cache.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SessionRecord {
    pub session_id: SessionId,
    pub worker_id: usize,
    pub model_id: String,
    pub turns: Vec<TurnRecord>,
}
