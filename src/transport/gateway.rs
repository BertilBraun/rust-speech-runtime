use std::{
    net::{IpAddr, Ipv4Addr, SocketAddr},
    sync::Arc,
    time::Duration,
};

use serde::Serialize;
use tokio::{
    net::{TcpListener, TcpStream},
    sync::Semaphore,
    task::JoinSet,
    time::Instant,
};
use tokio_util::sync::CancellationToken;

use super::wire::{ClientRequest, MAX_MESSAGE_BYTES, ServerPeer, ServerReply, WireError};
use crate::{
    Ingress, Node, RuntimeError,
    config::RuntimeConfig,
    metrics::{
        Report,
        cpu::{CpuUsage, ProcessCpuMeasurement},
    },
    protocol::{CreateOutcome, FrameRejection, InputFrame, InputOutcome, SessionLease},
};

#[derive(Clone, Debug, Serialize)]
pub struct GatewayConfig {
    pub listen_address: SocketAddr,
    pub maximum_connections: usize,
    pub message_limit: usize,
    pub io_timeout: Duration,
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
            .unwrap();
        assert_eq!(audio.cache, CacheOutcome::Hit);
        assert!(current.close().await.unwrap());
        signal.cancel();
        let report = task.await.unwrap().unwrap();
        assert_eq!(report.runtime.admitted_sessions, 2);
        assert_eq!(report.runtime.inference.delivered_frames, 1);
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
        if configuration.maximum_connections == 0
            || configuration.message_limit == 0
            || configuration.message_limit > MAX_MESSAGE_BYTES
            || configuration.io_timeout.is_zero()
            || configuration.io_timeout > Duration::from_secs(86400)
        {
            return Err(GatewayError::Configuration(
                "positive bounded connection, message and timeout limits required",
            ));
        }
        let replay_limit = runtime_configuration
            .audio_limits
            .max_prefix_packets
            .checked_mul(10)
            .and_then(|bytes| {
                bytes.checked_add(runtime_configuration.audio_limits.max_prefix_bytes)
            })
            .and_then(|bytes| bytes.checked_add(runtime_configuration.audio_limits.max_frame_bytes))
            .and_then(|bytes| bytes.checked_add(65536));
        if replay_limit.is_none_or(|limit| limit > configuration.message_limit) {
            return Err(GatewayError::Configuration(
                "wire limit must hold a full audio prefix replay",
            ));
        }
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
                                connection_loop(stream, ingress, runtime, configuration, signal).await
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

async fn connection_loop(
    stream: TcpStream,
    ingress: Ingress,
    runtime: RuntimeConfig,
    configuration: GatewayConfig,
    cancellation: CancellationToken,
) -> Result<(), GatewayError> {
    let mut peer = ServerPeer::new(stream, configuration.message_limit)?;
    let mut session = None;
    let outcome = serve_connection(
        &mut peer,
        &ingress,
        &runtime,
        &configuration,
        &cancellation,
        &mut session,
    )
    .await;
    if let Some(lease) = session {
        ingress.close_session(lease).await?;
    }
    outcome
}

async fn send_reply(
    peer: &mut ServerPeer,
    reply: ServerReply,
    configuration: &GatewayConfig,
    cancellation: &CancellationToken,
) -> Result<(), GatewayError> {
    tokio::select! {
        _ = cancellation.cancelled() => Ok(()),
        outcome = tokio::time::timeout(configuration.io_timeout, peer.send(reply)) => {
            outcome.map_err(|_| WireError::Timeout)??;
            Ok(())
        }
    }
}

async fn serve_connection(
    peer: &mut ServerPeer,
    ingress: &Ingress,
    runtime: &RuntimeConfig,
    configuration: &GatewayConfig,
    cancellation: &CancellationToken,
    session: &mut Option<SessionLease>,
) -> Result<(), GatewayError> {
    loop {
        let request = tokio::select! {
            _ = cancellation.cancelled() => return Ok(()),
            outcome = tokio::time::timeout(configuration.io_timeout, peer.receive()) => outcome.map_err(|_| WireError::Timeout)??,
        };
        let Some(request) = request else {
            return Ok(());
        };
        match request {
            ClientRequest::Open(session_id) if session.is_none() => {
                let outcome = ingress.create_session(session_id).await?;
                if let CreateOutcome::Admitted(admission) = outcome {
                    *session = Some(SessionLease {
                        session_id,
                        generation: admission.assignment.generation,
                    });
                }
                send_reply(
                    peer,
                    ServerReply::Opened(outcome),
                    configuration,
                    cancellation,
                )
                .await?;
                if session.is_none() {
                    return Ok(());
                }
            }
            ClientRequest::Audio {
                packet,
                remaining_budget,
            } => {
                let Some(lease) = *session else {
                    return Err(WireError::InvalidMessage("open a session before audio").into());
                };
                if remaining_budget.is_zero()
                    || remaining_budget > runtime.packet_completion_budget()
                {
                    send_reply(
                        peer,
                        ServerReply::Rejected(FrameRejection::DeadlineExceeded),
                        configuration,
                        cancellation,
                    )
                    .await?;
                    return Ok(());
                }
                let timestamp = Instant::now();
                let future = ingress.input_frame(
                    lease,
                    InputFrame {
                        timestamp,
                        deadline: timestamp + remaining_budget - runtime.packet_lateness_grace,
                        packet,
                    },
                );
                tokio::pin!(future);
                let outcome = {
                    tokio::select! {
                        _ = cancellation.cancelled() => return Ok(()),
                        outcome = &mut future => outcome?,
                        incoming = peer.receive() => {
                            match incoming? {
                                None => return Ok(()),
                                Some(ClientRequest::Close) => {
                                    let closed = ingress.close_session(lease).await?;
                                    *session = None;
                                    send_reply(peer, ServerReply::Closed(closed), configuration, cancellation).await?;
                                    return Ok(());
                                }
                                _ => {
                                    ingress.close_session(lease).await?;
                                    *session = None;
                                    send_reply(peer, ServerReply::Rejected(FrameRejection::Overloaded), configuration, cancellation).await?;
                                    return Ok(());
                                }
                            }
                        }
                    }
                };
                let terminal = matches!(outcome, InputOutcome::Rejected(_));
                let reply = match outcome {
                    InputOutcome::Processed(mut output) => {
                        output.audio.timings.gateway_return = Instant::now()
                            .duration_since(output.completed_at)
                            .saturating_sub(output.audio.timings.result_delivery);
                        ServerReply::Audio(output.audio)
                    }
                    InputOutcome::CacheMiss => ServerReply::CacheMiss,
                    InputOutcome::Rejected(reason) => ServerReply::Rejected(reason),
                };
                send_reply(peer, reply, configuration, cancellation).await?;
                if terminal {
                    return Ok(());
                }
            }
            ClientRequest::EvictCache => {
                let Some(lease) = *session else {
                    return Err(WireError::InvalidMessage("open a session before eviction").into());
                };
                let removed = ingress.evict_cache(lease).await?;
                send_reply(
                    peer,
                    ServerReply::CacheEvicted(removed),
                    configuration,
                    cancellation,
                )
                .await?;
            }
            ClientRequest::Close => {
                let closed = match session.take() {
                    Some(lease) => ingress.close_session(lease).await?,
                    None => false,
                };
                send_reply(
                    peer,
                    ServerReply::Closed(closed),
                    configuration,
                    cancellation,
                )
                .await?;
                return Ok(());
            }
            ClientRequest::Open(_) => {
                return Err(WireError::InvalidMessage("connection already owns a session").into());
            }
        }
    }
}
