//! Validates wire messages and dispatches them to the admitted session handle.

use bytes::Bytes;
use tokio_tungstenite::tungstenite::Message;

use crate::{
    SessionHandle,
    protocol::{ErrorCode, SessionEvent},
    transport::{
        GatewayError,
        wire::{AudioChunk, ClientControl},
    },
};

use super::Connection;

impl Connection {
    pub(super) async fn message(&mut self, message: Message) -> Result<bool, GatewayError> {
        let result = match message {
            Message::Close(_) => return Ok(false),
            Message::Text(text) => match serde_json::from_str::<ClientControl>(&text) {
                Ok(ClientControl::Close) => {
                    self.application_close_requested = true;
                    return Ok(false);
                }
                Ok(control) => self.control(control).await,
                Err(error) => Err(error.into()),
            },
            Message::Binary(bytes) => self.audio(bytes).await,
            Message::Ping(_) | Message::Pong(_) => Ok(()),
            Message::Frame(_) => Err(GatewayError::Protocol("unexpected raw WebSocket frame")),
        };
        if let Err(error) = result {
            self.failure(error)?;
        }
        Ok(true)
    }

    async fn audio(&self, bytes: Bytes) -> Result<(), GatewayError> {
        let chunk = AudioChunk::decode(bytes)?;
        self.require_session()?
            .audio(chunk.turn_id, chunk.chunk_index, chunk.pcm16)
            .await?;
        Ok(())
    }

    async fn control(&mut self, control: ClientControl) -> Result<(), GatewayError> {
        match control {
            ClientControl::Open { session_id } => {
                if self.session.is_some() {
                    return Err(GatewayError::Protocol("connection already owns a session"));
                }
                self.session = Some(tokio::select! {
                    _ = self.cancellation.cancelled() => return Ok(()),
                    session = self.ingress.open_session(session_id) => session?,
                });
            }
            ClientControl::StartTurn { turn_id } => {
                self.require_session()?.begin_turn(turn_id).await?
            }
            ClientControl::Prepare {
                turn_id,
                chunk_count,
                sample_count,
            } => {
                self.require_session()?
                    .prepare(turn_id, chunk_count, sample_count)
                    .await?
            }
            ClientControl::Commit {
                turn_id,
                chunk_count,
                sample_count,
            } => {
                self.require_session()?
                    .commit(turn_id, chunk_count, sample_count)
                    .await?
            }
            ClientControl::Cancel { turn_id } => self.require_session()?.cancel(turn_id).await?,
            ClientControl::Close => unreachable!("close handled by connection loop"),
        }
        Ok(())
    }

    fn require_session(&self) -> Result<&SessionHandle, GatewayError> {
        self.session
            .as_ref()
            .ok_or(GatewayError::Protocol("open a session first"))
    }

    fn failure(&self, error: GatewayError) -> Result<(), GatewayError> {
        let code = match &error {
            GatewayError::Runtime(error) => error.code(),
            _ => ErrorCode::InvalidInput,
        };
        self.emit(SessionEvent::Failed {
            turn_id: None,
            code,
            message: error.to_string(),
        })
    }

    pub(super) fn emit(&self, event: SessionEvent) -> Result<(), GatewayError> {
        self.outbound
            .try_send(event)
            .map_err(|_| GatewayError::SlowConsumer)
    }
}
