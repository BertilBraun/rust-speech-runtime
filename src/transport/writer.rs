//! Owns socket writes so a slow client cannot block the connection's input loop.

use std::{sync::Arc, time::Duration};

use futures_util::{SinkExt, stream::SplitSink};
use tokio::{
    io::{AsyncRead, AsyncWrite},
    sync::mpsc,
    task::JoinHandle,
    time::timeout,
};
use tokio_tungstenite::{WebSocketStream, tungstenite::Message};
use tokio_util::sync::CancellationToken;

use crate::{metrics::Metrics, protocol::SessionEvent};

use super::GatewayError;

pub(crate) fn start_writer<S: AsyncRead + AsyncWrite + Unpin + Send + 'static>(
    mut writer: SplitSink<WebSocketStream<S>, Message>,
    mut mailbox: mpsc::Receiver<SessionEvent>,
    write_timeout: Duration,
    cancellation: CancellationToken,
    metrics: Arc<Metrics>,
) -> JoinHandle<Result<(), GatewayError>> {
    tokio::spawn(async move {
        let _writer_cleanup = cancellation.clone().drop_guard();
        loop {
            let event = tokio::select! {
                _ = cancellation.cancelled() => return Ok(()),
                event = mailbox.recv() => {
                    let Some(event) = event else {
                        break;
                    };
                    event
                }
            };
            let started = tokio::time::Instant::now();
            let message = Message::Text(serde_json::to_string(&event)?.into());
            metrics
                .runtime_timing
                .websocket_serialization
                .record(started.elapsed().as_secs_f64() * 1000.0);
            let started = tokio::time::Instant::now();
            let result = tokio::select! {
                _ = cancellation.cancelled() => return Ok(()),
                result = timeout(write_timeout, writer.send(message)) => {
                    result
                }
            };
            metrics
                .runtime_timing
                .websocket_write
                .record(started.elapsed().as_secs_f64() * 1000.0);
            result.map_err(|_| GatewayError::Timeout)??;
        }
        timeout(write_timeout, writer.close())
            .await
            .map_err(|_| GatewayError::Timeout)??;
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::SessionId;
    use futures_util::StreamExt;
    use tokio_tungstenite::tungstenite::protocol::Role;

    #[tokio::test]
    async fn blocked_writer_times_out_and_cancels_only_its_connection() {
        let (stream, _unread_peer) = tokio::io::duplex(8);
        let websocket = WebSocketStream::from_raw_socket(stream, Role::Server, None).await;
        let (writer, _reader) = websocket.split();
        let (sender, mailbox) = mpsc::channel(1);
        let node = CancellationToken::new();
        let connection = node.child_token();
        let task = start_writer(
            writer,
            mailbox,
            Duration::from_millis(20),
            connection.clone(),
            Arc::new(Metrics::default()),
        );
        sender
            .send(SessionEvent::Opened {
                session_id: SessionId("backpressured".into()),
                worker_id: 0,
            })
            .await
            .expect("bounded event queued");
        assert!(matches!(
            task.await.expect("writer task"),
            Err(GatewayError::Timeout)
        ));
        assert!(connection.is_cancelled());
        assert!(!node.is_cancelled());
    }
}
