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

pub struct VoiceClient {
    websocket: WebSocketStream<MaybeTlsStream<TcpStream>>,
    timeout: Duration,
}

impl VoiceClient {
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

    pub async fn cancel(&mut self, turn_id: TurnId) -> Result<(), GatewayError> {
        self.send_control(ClientControl::Cancel { turn_id }).await
    }

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
