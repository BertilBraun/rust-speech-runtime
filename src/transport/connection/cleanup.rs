//! Releases model state, drains accepted events and transfers the final archive record.

use futures_util::{StreamExt, stream::SplitStream};
use tokio::{net::TcpStream, task::JoinHandle, time::Instant};
use tokio_tungstenite::{WebSocketStream, tungstenite::Message};

use crate::transport::{
    GatewayError,
    archive::{self, ArchiveRecord, EnqueueOutcome, unix_milliseconds},
    gateway::ArchiveSender,
};

use super::{Connection, ConnectionReport};

pub(super) async fn serve_and_close(
    mut connection: Connection,
    archives: Option<ArchiveSender>,
    writer_task: JoinHandle<Result<(), GatewayError>>,
) -> ConnectionReport {
    let result = connection.serve().await;
    let mut report = ConnectionReport {
        failed: result.is_err(),
        ..ConnectionReport::default()
    };
    if let Err(error) = result {
        eprintln!("WebSocket connection closed: {error}");
    }

    close_session_and_archive(&mut connection, archives, &mut report).await;
    drop(connection.outbound);
    let writer_result = match writer_task.await {
        Ok(result) => result,
        Err(error) => Err(error.into()),
    };
    if let Err(error) = writer_result {
        report.failed = true;
        eprintln!("WebSocket writer failed: {error}");
    }

    if connection.application_close_requested
        && let Err(error) = finish_close_handshake(
            &mut connection.reader,
            connection.configuration.write_timeout,
        )
        .await
    {
        report.failed = true;
        eprintln!("WebSocket close handshake failed: {error}");
    }
    report
}

async fn close_session_and_archive(
    connection: &mut Connection,
    archives: Option<ArchiveSender>,
    report: &mut ConnectionReport,
) {
    let Some(mut session) = connection.session.take() else {
        return;
    };
    let conversation = match session.close().await {
        Ok(conversation) => conversation,
        Err(error) => {
            report.failed = true;
            eprintln!("session cleanup failed: {error}");
            return;
        }
    };
    while let Some(event) = session.next_event().await {
        if connection.outbound.try_send(event).is_err() {
            report.failed = true;
            break;
        }
    }

    let Some(sender) = archives else {
        return;
    };
    let record = ArchiveRecord {
        created_unix_ms: connection.created_unix_ms,
        closed_unix_ms: unix_milliseconds(),
        conversation,
    };
    match archive::enqueue(&sender, record).await {
        Ok(outcome) => {
            report.archive_backpressured = matches!(outcome, EnqueueOutcome::Backpressured)
        }
        Err(error) => {
            report.archive_rejected = true;
            eprintln!("session archive enqueue failed: {error}");
        }
    }
}

async fn finish_close_handshake(
    reader: &mut SplitStream<WebSocketStream<TcpStream>>,
    write_timeout: std::time::Duration,
) -> Result<(), GatewayError> {
    let deadline = Instant::now() + write_timeout;
    loop {
        let incoming = tokio::time::timeout_at(deadline, reader.next())
            .await
            .map_err(|_| GatewayError::Timeout)?;
        match incoming {
            None | Some(Ok(Message::Close(_))) => return Ok(()),
            Some(Err(error)) => return Err(error.into()),
            Some(Ok(_)) => continue,
        }
    }
}
