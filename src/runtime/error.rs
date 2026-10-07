use thiserror::Error;

use crate::protocol::ErrorCode;

/// A runtime failure with a stable code suitable for a client response.
///
/// Use [`Self::code`] for decisions; the display message provides diagnostic context.
#[derive(Clone, Debug, Error)]
#[error("{code:?}: {message}")]
pub struct RuntimeError {
    code: ErrorCode,
    message: String,
}

impl RuntimeError {
    /// Associates a protocol error code with an explanation of the failed operation.
    pub fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }

    /// Returns the machine-readable reason for this failure.
    pub fn code(&self) -> ErrorCode {
        self.code
    }
}
