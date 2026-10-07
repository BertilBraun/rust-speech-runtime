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
        key: String,
        session_id: SessionId,
        events: mpsc::Sender<SessionEvent>,
        cancellation: CancellationToken,
        reply: oneshot::Sender<Result<(), RuntimeError>>,
    },
    Begin {
        key: String,
        turn_id: TurnId,
        reply: oneshot::Sender<Result<(), RuntimeError>>,
    },
    Audio {
        key: String,
        turn_id: TurnId,
        chunk_index: u32,
        audio: Bytes,
        reply: oneshot::Sender<Result<(), RuntimeError>>,
    },
    Prepare {
        key: String,
        turn_id: TurnId,
        chunk_count: u32,
        sample_count: usize,
        reply: oneshot::Sender<Result<(), RuntimeError>>,
    },
    Commit {
        key: String,
        turn_id: TurnId,
        final_chunk_count: u32,
        sample_count: usize,
        reply: oneshot::Sender<Result<(), RuntimeError>>,
    },
    Cancel {
        key: String,
        turn_id: TurnId,
        reply: oneshot::Sender<Result<(), RuntimeError>>,
    },
    Close {
        key: String,
        reply: Option<oneshot::Sender<Result<SessionRecord, RuntimeError>>>,
    },
}
