pub mod backend;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SessionId(pub String);
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct TurnId(pub u64);

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

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum SessionEvent {
    Opened {
        session_id: SessionId,
        worker_id: usize,
    },
    Closed {
        session_id: SessionId,
    },
    Accepted {
        turn_id: TurnId,
    },
    TextDelta {
        turn_id: TurnId,
        generation: u64,
        sequence: u64,
        token_id: u32,
        text: String,
    },
    Finished {
        turn_id: TurnId,
        generation: u64,
        reason: FinishReason,
        generated_tokens: usize,
    },
    Failed {
        turn_id: Option<TurnId>,
        code: ErrorCode,
        message: String,
    },
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TokenRecord {
    pub token_id: u32,
    pub text: String,
    pub index: u64,
    pub elapsed_ms: f64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TurnRecord {
    pub turn_id: TurnId,
    pub audio_pcm16: Vec<u8>,
    pub tokens: Vec<TokenRecord>,
    pub finish_reason: Option<FinishReason>,
    pub committed: bool,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SessionRecord {
    pub session_id: SessionId,
    pub worker_id: usize,
    pub model_id: String,
    pub turns: Vec<TurnRecord>,
}
