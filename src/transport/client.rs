//! Ordinary WebSocket client used by examples and benchmark workloads.

use std::time::Duration;

use bytes::Bytes;
use futures_util::{SinkExt, StreamExt};
use tokio::{
    net::TcpStream,
    time::{Instant, timeout, timeout_at},
};
use tokio_tungstenite::{
    MaybeTlsStream, WebSocketStream, connect_async_with_config,
    tungstenite::{Error, Message, protocol::WebSocketConfig},
};

use crate::protocol::{SessionEvent, SessionId, TurnId};

use super::{
    GatewayError,
    wire::{AudioChunk, ClientControl},
};

/// One conversation connection to the versioned `/v1` speech endpoint.
///
/// Sends controls and PCM16 audio over the same ordered WebSocket. Except for open
/// and begin_turn, command methods acknowledge socket writes; inspect server events
/// for validation failures or completion. The client owns no model/cache state.
pub struct VoiceClient {
    websocket: WebSocketStream<MaybeTlsStream<TcpStream>>,
    timeout: Duration,
}

impl VoiceClient {
    /// Opens a WebSocket, bounding connection and later reads/writes by operation_timeout.
    pub async fn connect(url: &str, operation_timeout: Duration) -> Result<Self, GatewayError> {
        let configuration = WebSocketConfig::default()
            .max_message_size(Some(1024 * 1024))
            .max_frame_size(Some(1024 * 1024))
            .write_buffer_size(0)
            .max_write_buffer_size(2 * 1024 * 1024);
        let (websocket, _) = timeout(
            operation_timeout,
            connect_async_with_config(url, Some(configuration), true),
        )
        .await
        .map_err(|_| GatewayError::Timeout)??;
        Ok(Self {
            websocket,
            timeout: operation_timeout,
        })
    }

    /// Requests admission and waits for Opened, returning the sticky worker ID.
    /// A rejected open completes the close handshake before returning the failure.
    pub async fn open(&mut self, session_id: SessionId) -> Result<usize, GatewayError> {
        self.send_control(ClientControl::Open { session_id })
            .await?;
        match self.next_event().await? {
            SessionEvent::Opened { worker_id, .. } => Ok(worker_id),
            SessionEvent::Failed { code, message, .. } => {
                self.close_websocket().await?;
                Err(GatewayError::Rejected(code, message))
            }
            _ => Err(GatewayError::Protocol("expected opened event")),
        }
    }

    /// Starts capture and waits for admission; discards older queued events while waiting.
    /// Read the old turn first if its output is needed before an interruption.
    pub async fn begin_turn(&mut self, turn_id: TurnId) -> Result<(), GatewayError> {
        self.send_control(ClientControl::StartTurn { turn_id })
            .await?;
        loop {
            match self.next_event().await? {
                SessionEvent::Accepted { turn_id: accepted } if accepted == turn_id => {
                    return Ok(());
                }
                SessionEvent::Failed { code, message, .. } => {
                    return Err(GatewayError::Rejected(code, message));
                }
                _ => {}
            }
        }
    }

    /// Sends a contiguous, zero-based packet of mono 16 kHz little-endian PCM16.
    /// Use 100 ms packets (1,600 samples), retaining a shorter final packet.
    pub async fn audio(
        &mut self,
        turn_id: TurnId,
        chunk_index: u32,
        pcm16: Bytes,
    ) -> Result<(), GatewayError> {
        self.send(Message::Binary(
            AudioChunk {
                turn_id,
                chunk_index,
                pcm16,
            }
            .encode(),
        ))
        .await
    }

    /// Requests private candidate inference during endpoint confirmation.
    /// Counts must describe all audio sent so far; text remains hidden until commit.
    pub async fn prepare(
        &mut self,
        turn_id: TurnId,
        chunk_count: u32,
        sample_count: usize,
    ) -> Result<(), GatewayError> {
        self.send_control(ClientControl::Prepare {
            turn_id,
            chunk_count,
            sample_count,
        })
        .await
    }

    /// Confirms end-of-turn with exact packet/sample totals and enables text generation.
    pub async fn commit(
        &mut self,
        turn_id: TurnId,
        chunk_count: u32,
        sample_count: usize,
    ) -> Result<(), GatewayError> {
        self.send_control(ClientControl::Commit {
            turn_id,
            chunk_count,
            sample_count,
        })
        .await
    }

    /// Requests logical interruption; read remaining accepted events in their FIFO order.
    pub async fn cancel(&mut self, turn_id: TurnId) -> Result<(), GatewayError> {
        self.send_control(ClientControl::Cancel { turn_id }).await
    }

    /// Reads one JSON event, skipping transport ping/pong frames.
    /// Each read is bounded; an early peer close is an error.
    pub async fn next_event(&mut self) -> Result<SessionEvent, GatewayError> {
        loop {
            let incoming = timeout(self.timeout, self.websocket.next())
                .await
                .map_err(|_| GatewayError::Timeout)?;
            match incoming {
                Some(Ok(Message::Text(text))) => return Ok(serde_json::from_str(&text)?),
                Some(Ok(Message::Ping(_) | Message::Pong(_))) => {}
                None | Some(Ok(Message::Close(_))) => {
                    return Err(GatewayError::Protocol(
                        "connection closed before expected event",
                    ));
                }
                Some(Ok(_)) => return Err(GatewayError::Protocol("expected JSON server event")),
                Some(Err(error)) => return Err(error.into()),
            }
        }
    }

    /// Closes the session, consumes events through Closed, then finishes socket shutdown.
    /// Drain output beforehand if the caller needs to retain those events.
    pub async fn close(mut self) -> Result<(), GatewayError> {
        self.send_control(ClientControl::Close).await?;
        loop {
            if matches!(self.next_event().await?, SessionEvent::Closed { .. }) {
                break;
            }
        }
        self.close_websocket().await
    }

    async fn close_websocket(&mut self) -> Result<(), GatewayError> {
        let deadline = Instant::now() + self.timeout;
        timeout_at(deadline, self.websocket.close(None))
            .await
            .map_err(|_| GatewayError::Timeout)??;
        loop {
            match timeout_at(deadline, self.websocket.next())
                .await
                .map_err(|_| GatewayError::Timeout)?
            {
                None | Some(Ok(Message::Close(_))) | Some(Err(Error::ConnectionClosed)) => {
                    return Ok(());
                }
                Some(Err(error)) => return Err(error.into()),
                Some(Ok(_)) => {}
            }
        }
    }

    /// Sends a typed control without consuming its server-side acknowledgement.
    /// Use this when the caller needs to manage every event, including interruptions.
    pub async fn send_control(&mut self, control: ClientControl) -> Result<(), GatewayError> {
        self.send(Message::Text(serde_json::to_string(&control)?.into()))
            .await
    }

    async fn send(&mut self, message: Message) -> Result<(), GatewayError> {
        timeout(self.timeout, self.websocket.send(message))
            .await
            .map_err(|_| GatewayError::Timeout)??;
        Ok(())
    }
}
