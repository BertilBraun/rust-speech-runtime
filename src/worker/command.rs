//! Commands acknowledge actor-side validation; inference results use a separate channel.

use crate::{
    protocol::{SessionEvent, SessionId, SessionRecord, TurnId},
    runtime::RuntimeError,
};
use bytes::Bytes;
use tokio::sync::{mpsc, oneshot};
use tokio_util::sync::CancellationToken;

pub(crate) enum Command {
    Open {
        session_key: String,
        session_id: SessionId,
        events: mpsc::Sender<SessionEvent>,
        cancellation: CancellationToken,
        reply: oneshot::Sender<Result<(), RuntimeError>>,
    },
    Begin {
        session_key: String,
        turn_id: TurnId,
        reply: oneshot::Sender<Result<(), RuntimeError>>,
    },
    Audio {
        session_key: String,
        turn_id: TurnId,
        chunk_index: u32,
        audio: Bytes,
        reply: oneshot::Sender<Result<(), RuntimeError>>,
    },
    Prepare {
        session_key: String,
        turn_id: TurnId,
        chunk_count: u32,
        sample_count: usize,
        reply: oneshot::Sender<Result<(), RuntimeError>>,
    },
    Commit {
        session_key: String,
        turn_id: TurnId,
        final_chunk_count: u32,
        sample_count: usize,
        reply: oneshot::Sender<Result<(), RuntimeError>>,
    },
    Cancel {
        session_key: String,
        turn_id: TurnId,
        reply: oneshot::Sender<Result<(), RuntimeError>>,
    },
    Close {
        session_key: String,
        reply: Option<oneshot::Sender<Result<SessionRecord, RuntimeError>>>,
    },
}
