//! Validates the public versioned WebSocket endpoint.

use tokio::{net::TcpStream, time::timeout};
use tokio_tungstenite::{
    WebSocketStream, accept_hdr_async_with_config,
    tungstenite::{
        handshake::server::{ErrorResponse, Request, Response},
        protocol::WebSocketConfig,
    },
};
use tokio_util::sync::CancellationToken;

use crate::transport::{GatewayConfig, GatewayError};

pub(super) async fn accept(
    stream: TcpStream,
    configuration: &GatewayConfig,
    cancellation: &CancellationToken,
) -> Result<Option<WebSocketStream<TcpStream>>, GatewayError> {
    let websocket_configuration = WebSocketConfig::default()
        .max_message_size(Some(configuration.max_message_bytes))
        .max_frame_size(Some(configuration.max_message_bytes))
        .write_buffer_size(0)
        .max_write_buffer_size(configuration.max_message_bytes * 2);
    let upgrade =
        accept_hdr_async_with_config(stream, validate_upgrade, Some(websocket_configuration));

    tokio::select! {
        _ = cancellation.cancelled() => Ok(None),
        result = timeout(configuration.handshake_timeout, upgrade) => {
            let websocket = result.map_err(|_| GatewayError::Timeout)??;
            Ok(Some(websocket))
        }
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
