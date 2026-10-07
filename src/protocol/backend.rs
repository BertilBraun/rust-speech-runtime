//! Version-one Rust/Python worker protocol over a persistent TCP connection.
//!
//! A big-endian u32 metadata length precedes strict JSON, then body_bytes raw PCM16
//! bytes. Audio offsets address that body. Each batch response preserves operation
//! order and identity. Backend session IDs include a node-assigned incarnation so
//! reusing a public session ID cannot reuse an earlier cache.

use super::ErrorCode;
use serde::{Deserialize, Serialize};

/// Startup capabilities advertised after the backend loads and warms its model.
/// Runtime limits are capped by these values before scheduling model work.
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

/// Exact proposal accepted by Rust, acknowledged on the next backend operation.
/// The turn/index/token tuple prevents rejected or stale proposals entering history.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AcceptedToken {
    pub turn_id: u64,
    pub index: u64,
    pub token_id: u32,
}

/// Work for one backend-owned conversation cache.
/// Only the assigned worker may issue operations for that session incarnation.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum Operation {
    /// Allocate an empty conversation cache.
    Open {
        operation_id: u64,
        session_id: String,
    },
    /// Encode a complete committed utterance and propose its first response token.
    Prefill {
        operation_id: u64,
        session_id: String,
        turn_id: u64,
        generation: u64,
        audio_offset: usize,
        audio_bytes: usize,
        accepted: Option<AcceptedToken>,
    },
    /// Run whole-candidate inference on an isolated cache without publishing text.
    /// Whisper is bidirectional, so this uses the candidate rather than causal packets.
    Prepare {
        operation_id: u64,
        session_id: String,
        turn_id: u64,
        generation: u64,
        audio_offset: usize,
        audio_bytes: usize,
        accepted: Option<AcceptedToken>,
    },
    /// Promote matching prepared state and its first token without another forward.
    Activate {
        operation_id: u64,
        session_id: String,
        turn_id: u64,
        generation: u64,
    },
    /// Release provisional state while retaining the accepted conversation cache.
    DiscardPrepared {
        operation_id: u64,
        session_id: String,
    },
    /// Reconcile the accepted previous token and propose one autoregressive token.
    Decode {
        operation_id: u64,
        session_id: String,
        turn_id: u64,
        generation: u64,
        accepted: AcceptedToken,
    },
    /// Release accepted and provisional cache state for this session.
    Close {
        operation_id: u64,
        session_id: String,
    },
}

impl Operation {
    /// Correlation ID that must be echoed unchanged by the result.
    pub fn operation_id(&self) -> u64 {
        match self {
            Self::Open { operation_id, .. }
            | Self::Prefill { operation_id, .. }
            | Self::Prepare { operation_id, .. }
            | Self::Activate { operation_id, .. }
            | Self::DiscardPrepared { operation_id, .. }
            | Self::Decode { operation_id, .. }
            | Self::Close { operation_id, .. } => *operation_id,
        }
    }

    /// Backend cache incarnation, distinct from the public conversation ID.
    pub fn session_id(&self) -> &str {
        match self {
            Self::Open { session_id, .. }
            | Self::Prefill { session_id, .. }
            | Self::Prepare { session_id, .. }
            | Self::Activate { session_id, .. }
            | Self::DiscardPrepared { session_id, .. }
            | Self::Decode { session_id, .. }
            | Self::Close { session_id, .. } => session_id,
        }
    }
}

/// Ordered operations whose audio ranges reference the accompanying binary body.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BatchRequest {
    pub request_id: u64,
    pub body_bytes: usize,
    pub operations: Vec<Operation>,
}

/// Per-operation result; a token remains a proposal until the actor accepts it.
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum Outcome {
    Opened,
    Closed,
    Discarded,
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

/// Correlated result, including a turn/generation fence for inference operations.
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OperationResult {
    pub operation_id: u64,
    pub session_id: String,
    pub turn_id: Option<u64>,
    pub generation: Option<u64>,
    pub outcome: Outcome,
}

/// Backend-observed wall times in milliseconds for this batch and its model stages.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Timing {
    pub elapsed_ms: f64,
    pub encode_ms: f64,
    pub prefill_ms: f64,
    pub decode_ms: f64,
}

/// PyTorch device allocator observations in bytes, distinct from host RAM usage.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Memory {
    pub allocated_bytes: u64,
    pub reserved_bytes: u64,
}

/// Ordered results for one request, with shared timing and memory observations.
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BatchResponse {
    pub request_id: u64,
    pub body_bytes: usize,
    pub results: Vec<OperationResult>,
    pub timing: Timing,
    pub memory: Memory,
}
