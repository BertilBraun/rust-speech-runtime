//! Exclusive framed TCP connection to one model process.
//! Requests and replies are serial; losing framing makes this worker unavailable.

use crate::{
    protocol::{
        ErrorCode,
        backend::{BatchRequest, BatchResponse, Ready},
    },
    runtime::RuntimeError,
};
use std::net::SocketAddr;
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpStream,
};

const MAX_METADATA_BYTES: usize = 1024 * 1024;

pub(crate) struct BackendConnection {
    stream: TcpStream,
}

impl BackendConnection {
    pub async fn connect(endpoint: SocketAddr) -> Result<(Self, Ready), RuntimeError> {
        let stream = TcpStream::connect(endpoint).await.map_err(network_error)?;
        stream.set_nodelay(true).map_err(network_error)?;
        let mut connection = Self { stream };
        let ready = connection.read_json::<Ready>().await?;
        if ready.r#type != "ready"
            || ready.protocol_version != 1
            || ready.body_bytes != 0
            || ready.max_batch_size == 0
            || ready.max_context_tokens == 0
            || ready.max_audio_samples == 0
        {
            return Err(RuntimeError::new(
                ErrorCode::BackendFailed,
                "invalid worker readiness capabilities",
            ));
        }
        Ok((connection, ready))
    }

    pub async fn execute(
        &mut self,
        request: &BatchRequest,
        audio: &[u8],
    ) -> Result<BatchResponse, RuntimeError> {
        self.write_request(request, audio).await?;
        let response = self.read_json::<BatchResponse>().await?;
        validate_response(request, &response)?;
        Ok(response)
    }

    async fn write_request(
        &mut self,
        request: &BatchRequest,
        audio: &[u8],
    ) -> Result<(), RuntimeError> {
        let metadata = serde_json::to_vec(request).map_err(protocol_error)?;
        if metadata.len() > MAX_METADATA_BYTES {
            return Err(RuntimeError::new(
                ErrorCode::InvalidInput,
                "worker metadata limit exceeded",
            ));
        }
        self.stream
            .write_u32(metadata.len() as u32)
            .await
            .map_err(network_error)?;
        self.stream
            .write_all(&metadata)
            .await
            .map_err(network_error)?;
        self.stream.write_all(audio).await.map_err(network_error)?;
        self.stream.flush().await.map_err(network_error)?;
        Ok(())
    }

    async fn read_json<T: serde::de::DeserializeOwned>(&mut self) -> Result<T, RuntimeError> {
        let length = self.stream.read_u32().await.map_err(network_error)? as usize;
        if length == 0 || length > MAX_METADATA_BYTES {
            return Err(RuntimeError::new(
                ErrorCode::BackendFailed,
                "worker metadata frame exceeds bound",
            ));
        }
        let mut metadata = vec![0u8; length];
        self.stream
            .read_exact(&mut metadata)
            .await
            .map_err(network_error)?;
        serde_json::from_slice(&metadata).map_err(protocol_error)
    }
}

fn validate_response(request: &BatchRequest, response: &BatchResponse) -> Result<(), RuntimeError> {
    if response.request_id != request.request_id
        || response.body_bytes != 0
        || response.results.len() != request.operations.len()
    {
        return Err(RuntimeError::new(
            ErrorCode::BackendFailed,
            "worker response framing or batch identity mismatch",
        ));
    }
    for (operation, result) in request.operations.iter().zip(&response.results) {
        if operation.operation_id() != result.operation_id
            || operation.session_id() != result.session_id
        {
            return Err(RuntimeError::new(
                ErrorCode::BackendFailed,
                "worker result identity mismatch",
            ));
        }
    }
    Ok(())
}

fn network_error(error: std::io::Error) -> RuntimeError {
    RuntimeError::new(ErrorCode::BackendUnavailable, error.to_string())
}

fn protocol_error(error: serde_json::Error) -> RuntimeError {
    RuntimeError::new(ErrorCode::BackendFailed, error.to_string())
}
