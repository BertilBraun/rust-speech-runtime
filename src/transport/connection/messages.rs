//! Validates wire messages and dispatches them to the admitted session handle.

use bytes::Bytes;
use tokio_tungstenite::tungstenite::Message;

use crate::{
    SessionHandle,
    protocol::{ErrorCode, SessionEvent, SessionId},
    transport::{
        GatewayError,
        wire::{AudioChunk, ClientControl},
    },
};

use super::Connection;

pub(super) enum ConnectionAction {
    Continue,
    Close,
}

impl Connection {
    pub(super) async fn handle_message(
        &mut self,
        message: Message,
    ) -> Result<ConnectionAction, GatewayError> {
        let result = match message {
            Message::Close(_) => return Ok(ConnectionAction::Close),
            Message::Text(text) => self.handle_text_message(&text).await,
            Message::Binary(bytes) => self
                .handle_audio_chunk(bytes)
                .await
                .map(|()| ConnectionAction::Continue),
            Message::Ping(_) | Message::Pong(_) => Ok(ConnectionAction::Continue),
            Message::Frame(_) => Err(GatewayError::Protocol("unexpected raw WebSocket frame")),
        };
        match result {
            Ok(action) => Ok(action),
            Err(error) => {
                self.emit_failure(error)?;
                Ok(ConnectionAction::Continue)
            }
        }
    }

    async fn handle_text_message(&mut self, text: &str) -> Result<ConnectionAction, GatewayError> {
        let control = serde_json::from_str::<ClientControl>(text)?;
        if matches!(control, ClientControl::Close) {
            self.application_close_requested = true;
            return Ok(ConnectionAction::Close);
        }
        self.handle_control(control).await?;
        Ok(ConnectionAction::Continue)
    }

    async fn handle_audio_chunk(&self, bytes: Bytes) -> Result<(), GatewayError> {
        let chunk = AudioChunk::decode(bytes)?;
        self.require_session()?
            .audio(chunk.turn_id, chunk.chunk_index, chunk.pcm16)
            .await?;
        Ok(())
    }

    async fn handle_control(&mut self, control: ClientControl) -> Result<(), GatewayError> {
        match control {
            ClientControl::Open { session_id } => self.open_session(session_id).await?,
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

    async fn open_session(&mut self, session_id: SessionId) -> Result<(), GatewayError> {
        if self.session.is_some() {
            return Err(GatewayError::Protocol("connection already owns a session"));
        }
        let session = tokio::select! {
            _ = self.cancellation.cancelled() => return Ok(()),
            session = self.ingress.open_session(session_id) => session?,
        };
        self.session = Some(session);
        Ok(())
    }

    fn require_session(&self) -> Result<&SessionHandle, GatewayError> {
        self.session
            .as_ref()
            .ok_or(GatewayError::Protocol("open a session first"))
    }

    fn emit_failure(&self, error: GatewayError) -> Result<(), GatewayError> {
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
