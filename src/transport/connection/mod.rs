//! Coordinates one bounded WebSocket connection through handshake, serving and cleanup.

mod cleanup;
mod handshake;
mod messages;

use cleanup::serve_and_close;
use handshake::accept;

use std::sync::Arc;

use futures_util::{StreamExt, stream::SplitStream};
use tokio::{net::TcpStream, sync::mpsc, time::Instant};
use tokio_tungstenite::WebSocketStream;
use tokio_util::sync::CancellationToken;

use crate::{Ingress, SessionHandle, protocol::SessionEvent};

use super::{
    GatewayConfig, GatewayError, archive::unix_milliseconds, gateway::ArchiveSender,
    writer::start_writer,
};

#[derive(Default)]
pub(crate) struct ConnectionReport {
    pub failed: bool,
    pub archive_rejected: bool,
    pub archive_backpressured: bool,
}

struct Connection {
    reader: SplitStream<WebSocketStream<TcpStream>>,
    outbound: mpsc::Sender<SessionEvent>,
    session: Option<SessionHandle>,
    ingress: Ingress,
    configuration: Arc<GatewayConfig>,
    cancellation: CancellationToken,
    created_unix_ms: u128,
    application_close_requested: bool,
}

pub(crate) async fn run(
    stream: TcpStream,
    ingress: Ingress,
    configuration: Arc<GatewayConfig>,
    archives: Option<ArchiveSender>,
    cancellation: CancellationToken,
) -> ConnectionReport {
    let websocket = match accept(stream, &configuration, &cancellation).await {
        Ok(Some(websocket)) => websocket,
        Ok(None) => return ConnectionReport::default(),
        Err(error) => {
            eprintln!("WebSocket handshake failed: {error}");
            return ConnectionReport {
                failed: true,
                ..ConnectionReport::default()
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
    let connection = Connection {
        reader,
        outbound,
        session: None,
        ingress,
        configuration,
        cancellation,
        created_unix_ms: unix_milliseconds(),
        application_close_requested: false,
    };
    serve_and_close(connection, archives, writer_task).await
}

impl Connection {
    async fn serve(&mut self) -> Result<(), GatewayError> {
        let mut last_activity = Instant::now();
        loop {
            tokio::select! {
                _ = self.cancellation.cancelled() => {
                    return Ok(());
                }
                _ = tokio::time::sleep_until(last_activity + self.configuration.idle_timeout) => {
                    return Err(GatewayError::Timeout);
                }
                message = self.reader.next() => {
                    let Some(message) = message else {
                        return Ok(());
                    };
                    last_activity = Instant::now();
                    if !self.message(message?).await? {
                        return Ok(());
                    }
                }
                event = next_event(&mut self.session), if self.session.is_some() => {
                    let Some(event) = event else {
                        return Ok(());
                    };
                    // A generating session is active even while its client sends no audio.
                    last_activity = Instant::now();
                    self.emit(event)?;
                }
            }
        }
    }
}

async fn next_event(session: &mut Option<SessionHandle>) -> Option<SessionEvent> {
    session
        .as_mut()
        .expect("event branch requires session")
        .next_event()
        .await
}
