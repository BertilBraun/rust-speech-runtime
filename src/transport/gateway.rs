use std::{
    net::{IpAddr, Ipv4Addr, SocketAddr},
    sync::Arc,
    time::Duration,
};

use serde::Serialize;
use tokio::{net::TcpListener, sync::Semaphore, task::JoinSet};
use tokio_util::sync::CancellationToken;

use super::{
    connection::Connection,
    wire::{MAX_MESSAGE_BYTES, WireError},
};
use crate::{
    Node, RuntimeError,
    config::{AudioLimits, RuntimeConfig},
    metrics::{
        Report,
        cpu::{CpuUsage, ProcessCpuMeasurement},
    },
};

#[derive(Clone, Debug, Serialize)]
pub struct GatewayConfig {
    pub listen_address: SocketAddr,
    pub maximum_connections: usize,
    pub message_limit: usize,
    pub io_timeout: Duration,
}

impl GatewayConfig {
    /// Checks transport bounds and space for the negotiated audio replay limits.
    pub fn validate(&self, audio_limits: AudioLimits) -> Result<(), GatewayError> {
        if self.maximum_connections == 0
            || self.message_limit == 0
            || self.message_limit > MAX_MESSAGE_BYTES
            || self.io_timeout.is_zero()
            || self.io_timeout > Duration::from_secs(86400)
        {
            return Err(GatewayError::Configuration(
                "positive bounded connection, message and timeout limits required",
            ));
        }
        let replay_limit = audio_limits
            .max_prefix_packets
            .checked_mul(10)
            .and_then(|bytes| bytes.checked_add(audio_limits.max_prefix_bytes))
            .and_then(|bytes| bytes.checked_add(audio_limits.max_frame_bytes))
            .and_then(|bytes| bytes.checked_add(65536));
        if replay_limit.is_none_or(|limit| limit > self.message_limit) {
            return Err(GatewayError::Configuration(
                "wire limit must hold a full audio prefix replay",
            ));
        }
        Ok(())
    }
}

impl Default for GatewayConfig {
    fn default() -> Self {
        Self {
            listen_address: SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 9000),
            maximum_connections: 4096,
            message_limit: MAX_MESSAGE_BYTES,
            io_timeout: Duration::from_secs(5),
        }
    }
}
#[derive(Debug, thiserror::Error)]
pub enum GatewayError {
    #[error(transparent)]
    Runtime(#[from] RuntimeError),
    #[error(transparent)]
    Wire(#[from] WireError),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Task(#[from] tokio::task::JoinError),
    #[error("invalid gateway configuration: {0}")]
    Configuration(&'static str),
}
#[derive(Debug, Default, Serialize)]
pub struct ConnectionMetrics {
    pub accepted: u64,
    pub peak_active: usize,
    pub rejected_connection_capacity: u64,
    pub failed_connections: u64,
}
#[derive(Debug, Serialize)]
pub struct GatewayReport {
    pub process_cpu: CpuUsage,
    pub runtime_configuration: RuntimeConfig,
    pub gateway_configuration: GatewayConfig,
    pub connections: ConnectionMetrics,
    pub runtime: Report,
}
/// Owns TCP ingress, bounded connection tasks and the inference runtime.
pub struct Gateway {
    listener: TcpListener,
    node: Node,
    runtime_configuration: RuntimeConfig,
    configuration: GatewayConfig,
}
impl Gateway {
    pub async fn bind(
        runtime_configuration: RuntimeConfig,
        configuration: GatewayConfig,
    ) -> Result<Self, GatewayError> {
        runtime_configuration
            .validate()
            .map_err(RuntimeError::from)?;
        configuration.validate(runtime_configuration.audio_limits)?;
        let listener = TcpListener::bind(configuration.listen_address).await?;
        let node = Node::start(runtime_configuration.clone()).await?;
        Ok(Self {
            listener,
            node,
            runtime_configuration,
            configuration,
        })
    }
    pub fn local_address(&self) -> Result<SocketAddr, std::io::Error> {
        self.listener.local_addr()
    }

    pub async fn serve(
        self,
        cancellation: CancellationToken,
    ) -> Result<GatewayReport, GatewayError> {
        let _connection_cleanup = cancellation.clone().drop_guard();
        let cpu = ProcessCpuMeasurement::start()?;
        let mut tasks: JoinSet<Result<(), GatewayError>> = JoinSet::new();
        let permits = Arc::new(Semaphore::new(self.configuration.maximum_connections));
        let mut metrics = ConnectionMetrics::default();
        let mut failure = None;
        loop {
            tokio::select! {
                _ = cancellation.cancelled() => break,
                connection = self.listener.accept() => {
                    match connection {
                        Ok((stream, _)) => {
                            let Ok(permit) = permits.clone().try_acquire_owned() else { metrics.rejected_connection_capacity += 1; continue; };
                            let ingress = self.node.ingress.clone();
                            let configuration = self.configuration.clone();
                            let runtime = self.runtime_configuration.clone();
                            let signal = cancellation.clone();
                            tasks.spawn(async move {
                                let _permit = permit;
                                Connection::new(stream, ingress, runtime, configuration, signal)?.run().await
                            });
                            metrics.accepted += 1;
                            metrics.peak_active = metrics.peak_active.max(tasks.len());
                        }
                        Err(error) => { failure = Some(error); break; }
                    }
                }
                Some(result) = tasks.join_next() => {
                    if !matches!(result, Ok(Ok(()))) { metrics.failed_connections += 1; }
                }
            }
        }
        cancellation.cancel();
        drop(self.listener);
        while let Some(result) = tasks.join_next().await {
            if !matches!(result, Ok(Ok(()))) {
                metrics.failed_connections += 1;
            }
        }
        let runtime = self.node.shutdown().await?;
        if let Some(error) = failure {
            return Err(error.into());
        }
        Ok(GatewayReport {
            process_cpu: cpu.finish()?,
            runtime_configuration: self.runtime_configuration,
            gateway_configuration: self.configuration,
            connections: metrics,
            runtime,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{Gateway, GatewayConfig};
    use crate::{
        config::RuntimeConfig,
        protocol::{CacheOutcome, FrameRejection, SessionId},
        transport::{ClientError, ConnectOutcome, connect_session},
    };
    use bytes::Bytes;
    use std::time::Duration;
    use tokio::time::Instant;
    use tokio_util::sync::CancellationToken;

    #[tokio::test]
    async fn obsolete_tcp_connection_cannot_cancel_its_replacement() {
        let gateway = Gateway::bind(
            RuntimeConfig {
                workers: 1,
                inference_latency: Duration::from_millis(3),
                calibration_samples: 2,
                minimum_packet_interval: Duration::from_millis(250),
                packet_deadline: Duration::from_millis(250),
                max_batch_wait: Duration::ZERO,
                ..RuntimeConfig::default()
            },
            GatewayConfig {
                listen_address: "127.0.0.1:0".parse().unwrap(),
                ..GatewayConfig::default()
            },
        )
        .await
        .unwrap();
        let address = gateway.local_address().unwrap();
        let ingress = gateway.node.ingress.clone();
        let signal = CancellationToken::new();
        let task = tokio::spawn(gateway.serve(signal.clone()));
        let ConnectOutcome::Admitted(mut old) =
            connect_session(address, SessionId(1), Duration::from_secs(2))
                .await
                .unwrap()
        else {
            panic!("expected admission");
        };
        ingress.close_session(SessionId(1)).await.unwrap();
        let ConnectOutcome::Admitted(mut current) =
            connect_session(address, SessionId(1), Duration::from_secs(2))
                .await
                .unwrap()
        else {
            panic!("expected replacement");
        };
        assert!(matches!(
            old.infer_audio(Bytes::from_static(b"old"), Instant::now())
                .await,
            Err(ClientError::Rejected(FrameRejection::Cancelled))
        ));
        drop(old);
        let audio = current
            .infer_audio(Bytes::from_static(b"current"), Instant::now())
            .await
            .unwrap()
            .into_audio();
        assert_eq!(audio.cache, CacheOutcome::Hit);
        assert!(current.close().await.unwrap());
        signal.cancel();
        let report = task.await.unwrap().unwrap();
        assert_eq!(report.runtime.admitted_sessions, 2);
        assert_eq!(report.runtime.inference.delivered_frames, 1);
    }
}
