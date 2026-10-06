use std::sync::Arc;

use futures_util::{StreamExt, stream::SplitStream};
use tokio::{
    net::TcpStream,
    sync::mpsc,
    time::{Instant, timeout},
};
use tokio_tungstenite::{
    WebSocketStream, accept_hdr_async_with_config,
    tungstenite::{
        Message,
        handshake::server::{ErrorResponse, Request, Response},
        protocol::WebSocketConfig,
    },
};
use tokio_util::sync::CancellationToken;

use crate::{
    Ingress, SessionHandle,
    protocol::{ErrorCode, SessionEvent},
};

use super::{
    GatewayConfig, GatewayError,
    archive::{ArchiveRecord, unix_milliseconds},
    gateway::ArchiveSender,
    wire::{AudioChunk, ClientControl},
    writer::start_writer,
};

#[derive(Default)]
pub(crate) struct ConnectionReport {
    pub failed: bool,
    pub archive_rejected: bool,
}

struct Connection {
    reader: SplitStream<WebSocketStream<TcpStream>>,
    outbound: mpsc::Sender<SessionEvent>,
    session: Option<SessionHandle>,
    ingress: Ingress,
    configuration: Arc<GatewayConfig>,
    cancellation: CancellationToken,
    created_unix_ms: u128,
}

pub(crate) async fn run(
    stream: TcpStream,
    ingress: Ingress,
    configuration: Arc<GatewayConfig>,
    archives: Option<ArchiveSender>,
    cancellation: CancellationToken,
) -> ConnectionReport {
    let websocket_configuration = WebSocketConfig::default()
        .max_message_size(Some(configuration.max_message_bytes))
        .max_frame_size(Some(configuration.max_message_bytes))
        .write_buffer_size(0)
        .max_write_buffer_size(configuration.max_message_bytes * 2);
    let handshake = tokio::select! {
        _ = cancellation.cancelled() => return ConnectionReport::default(),
        result = timeout(configuration.handshake_timeout, accept_hdr_async_with_config(stream, validate_upgrade, Some(websocket_configuration))) => result,
    };
    let websocket = match handshake {
        Ok(Ok(websocket)) => websocket,
        failure => {
            eprintln!("WebSocket handshake failed: {failure:?}");
            return ConnectionReport {
                failed: true,
                archive_rejected: false,
            };
        }
    };
    let (writer, reader) = websocket.split();
    let (outbound, mailbox) = mpsc::channel(configuration.outbound_capacity);
    let writer_task = start_writer(
        writer,
        mailbox,
        configuration.write_timeout,
        cancellation.clone(),
    );
    let mut connection = Connection {
        reader,
        outbound,
        session: None,
        ingress,
        configuration,
        cancellation,
        created_unix_ms: unix_milliseconds(),
    };
    let result = connection.serve().await;
    let mut report = ConnectionReport {
        failed: result.is_err(),
        archive_rejected: false,
    };
    if let Err(error) = result {
        eprintln!("WebSocket connection closed: {error}");
    }
    if let Some(mut session) = connection.session.take() {
        match session.close().await {
            Ok(conversation) => {
                while let Some(event) = session.next_event().await {
                    if connection.outbound.try_send(event).is_err() {
                        report.failed = true;
                        break;
                    }
                }
                if let Some(sender) = archives {
                    let record = ArchiveRecord {
                        created_unix_ms: connection.created_unix_ms,
                        closed_unix_ms: unix_milliseconds(),
                        conversation,
                    };
                    if sender.try_send(record).is_err() {
                        report.archive_rejected = true;
                        eprintln!(
                            "session archive rejected: bounded archive queue is full or stopped"
                        );
                    }
                }
            }
            Err(error) => {
                report.failed = true;
                eprintln!("session cleanup failed: {error}");
            }
        }
    }
    drop(connection.outbound);
    match writer_task.await {
        Ok(Ok(())) => {}
        failure => {
            report.failed = true;
            eprintln!("WebSocket writer failed: {failure:?}");
        }
    }
    report
}

impl Connection {
    async fn serve(&mut self) -> Result<(), GatewayError> {
        let mut last_input = Instant::now();
        loop {
            tokio::select! {
                _ = self.cancellation.cancelled() => return Ok(()),
                _ = tokio::time::sleep_until(last_input + self.configuration.idle_timeout) => return Err(GatewayError::Timeout),
                message = self.reader.next() => {
                    last_input = Instant::now();
                    match message {
                        None => return Ok(()),
                        Some(Ok(message)) => if !self.message(message).await? { return Ok(()); },
                        Some(Err(error)) => return Err(error.into()),
                    }
                }
                event = next_event(&mut self.session), if self.session.is_some() => {
                    let Some(event) = event else { return Ok(()); };
                    last_input = Instant::now();
                    self.emit(event)?;
                }
            }
        }
    }

    async fn message(&mut self, message: Message) -> Result<bool, GatewayError> {
        let result = match message {
            Message::Close(_) => return Ok(false),
            Message::Text(text) => match serde_json::from_str::<ClientControl>(&text) {
                Ok(ClientControl::Close) => return Ok(false),
                Ok(control) => self.control(control).await,
                Err(error) => Err(error.into()),
            },
            Message::Binary(bytes) => match AudioChunk::decode(bytes) {
                Ok(chunk) => match self.require_session() {
                    Ok(session) => session
                        .audio(chunk.turn_id, chunk.chunk_index, chunk.pcm16)
                        .await
                        .map_err(GatewayError::from),
                    Err(error) => Err(error),
                },
                Err(error) => Err(error),
            },
            Message::Ping(_) | Message::Pong(_) => Ok(()),
            Message::Frame(_) => Err(GatewayError::Protocol("unexpected raw WebSocket frame")),
        };
        if let Err(error) = result {
            self.failure(error)?;
        }
        Ok(true)
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

    fn emit(&self, event: SessionEvent) -> Result<(), GatewayError> {
        self.outbound
            .try_send(event)
            .map_err(|_| GatewayError::SlowConsumer)
    }
}

#[allow(
    clippy::result_large_err,
    reason = "Tungstenite's upgrade callback requires its concrete HTTP error response"
)]
fn validate_upgrade(request: &Request, response: Response) -> Result<Response, ErrorResponse> {
    if request.uri().path() != "/v1" {
        return Err(Response::builder()
            .status(404)
            .body(Some("use the /v1 speech WebSocket endpoint".into()))
            .expect("literal HTTP response"));
    }
    Ok(response)
}

async fn next_event(session: &mut Option<SessionHandle>) -> Option<SessionEvent> {
    session
        .as_mut()
        .expect("event branch requires session")
        .next_event()
        .await
}
