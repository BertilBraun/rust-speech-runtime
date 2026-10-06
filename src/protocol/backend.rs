use super::ErrorCode;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Ready {
    pub r#type: String,
    pub protocol_version: u32,
    pub body_bytes: usize,
    pub model_id: String,
    pub max_context_tokens: usize,
    pub max_batch_size: usize,
    pub max_audio_samples: usize,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AcceptedToken {
    pub turn_id: u64,
    pub index: u64,
    pub token_id: u32,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum Operation {
    Open {
        operation_id: u64,
        session_id: String,
    },
    Prefill {
        operation_id: u64,
        session_id: String,
        turn_id: u64,
        generation: u64,
        audio_offset: usize,
        audio_bytes: usize,
        accepted: Option<AcceptedToken>,
    },
    Decode {
        operation_id: u64,
        session_id: String,
        turn_id: u64,
        generation: u64,
        accepted: AcceptedToken,
    },
    Close {
        operation_id: u64,
        session_id: String,
    },
}
impl Operation {
    pub fn operation_id(&self) -> u64 {
        match self {
            Self::Open { operation_id, .. }
            | Self::Prefill { operation_id, .. }
            | Self::Decode { operation_id, .. }
            | Self::Close { operation_id, .. } => *operation_id,
        }
    }
    pub fn session_id(&self) -> &str {
        match self {
            Self::Open { session_id, .. }
            | Self::Prefill { session_id, .. }
            | Self::Decode { session_id, .. }
            | Self::Close { session_id, .. } => session_id,
        }
    }
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BatchRequest {
    pub request_id: u64,
    pub body_bytes: usize,
    pub operations: Vec<Operation>,
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum Outcome {
    Opened,
    Closed,
    Token {
        token_id: u32,
        text_delta: String,
        eos: bool,
        context_tokens: usize,
    },
    Failed {
        code: ErrorCode,
        message: String,
    },
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OperationResult {
    pub operation_id: u64,
    pub session_id: String,
    pub turn_id: Option<u64>,
    pub generation: Option<u64>,
    pub outcome: Outcome,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Timing {
    pub elapsed_ms: f64,
    pub encode_ms: f64,
    pub prefill_ms: f64,
    pub decode_ms: f64,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Memory {
    pub allocated_bytes: u64,
    pub reserved_bytes: u64,
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BatchResponse {
    pub request_id: u64,
    pub body_bytes: usize,
    pub results: Vec<OperationResult>,
    pub timing: Timing,
    pub memory: Memory,
}
