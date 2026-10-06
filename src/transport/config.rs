use std::{net::SocketAddr, path::PathBuf, time::Duration};

use super::GatewayError;

#[derive(Clone, Debug)]
pub struct GatewayConfig {
    pub listen_address: SocketAddr,
    pub max_connections: usize,
    pub max_message_bytes: usize,
    pub outbound_capacity: usize,
    pub handshake_timeout: Duration,
    pub write_timeout: Duration,
    pub idle_timeout: Duration,
    pub archive_directory: Option<PathBuf>,
    pub archive_capacity: usize,
}

impl Default for GatewayConfig {
    fn default() -> Self {
        Self {
            listen_address: SocketAddr::from(([127, 0, 0, 1], 8080)),
            max_connections: 4096,
            max_message_bytes: 16 * 1024,
            outbound_capacity: 64,
            handshake_timeout: Duration::from_secs(5),
            write_timeout: Duration::from_secs(2),
            idle_timeout: Duration::from_secs(120),
            archive_directory: Some(PathBuf::from("session-archives")),
            archive_capacity: 8,
        }
    }
}

impl GatewayConfig {
    pub fn validate(&self) -> Result<(), GatewayError> {
        if self.max_connections == 0
            || self.outbound_capacity == 0
            || self.archive_capacity == 0
            || self.max_message_bytes < 16
            || self.handshake_timeout.is_zero()
            || self.write_timeout.is_zero()
            || self.idle_timeout.is_zero()
        {
            return Err(GatewayError::Configuration(
                "limits and timeouts must be positive",
            ));
        }
        if self.max_message_bytes > 1024 * 1024 {
            return Err(GatewayError::Configuration(
                "WebSocket messages must not exceed 1 MiB",
            ));
        }
        Ok(())
    }
}
