use std::{net::SocketAddr, sync::Arc};

use serde::Serialize;
use tokio::{
    net::TcpListener,
    sync::{Semaphore, mpsc},
    task::JoinSet,
};
use tokio_util::sync::CancellationToken;

use crate::{Node, RuntimeError, config::RuntimeConfig, metrics::MetricsSnapshot};

use super::{
    GatewayConfig,
    archive::{self, ArchiveRecord, ArchiveReport},
    connection,
};

#[derive(Debug, thiserror::Error)]
pub enum GatewayError {
    #[error("invalid gateway configuration: {0}")]
    Configuration(&'static str),
    #[error("invalid WebSocket protocol: {0}")]
    Protocol(&'static str),
    #[error("connection timed out")]
    Timeout,
    #[error("outbound client channel is saturated")]
    SlowConsumer,
    #[error("session rejected ({0:?}): {1}")]
    Rejected(crate::protocol::ErrorCode, String),
    #[error(transparent)]
    Runtime(#[from] RuntimeError),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    WebSocket(#[from] tokio_tungstenite::tungstenite::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Task(#[from] tokio::task::JoinError),
}

#[derive(Debug, Default, Serialize)]
pub struct GatewayReport {
    pub connections: u64,
    pub rejected_connections: u64,
    pub failed_connections: u64,
    pub archive_queue_rejections: u64,
    pub archives: ArchiveReport,
    pub runtime: MetricsSnapshot,
}

pub struct Gateway {
    listener: TcpListener,
    node: Node,
    configuration: Arc<GatewayConfig>,
}

impl Gateway {
    pub async fn bind(
        runtime: RuntimeConfig,
        configuration: GatewayConfig,
    ) -> Result<Self, GatewayError> {
        configuration.validate()?;
        let listener = TcpListener::bind(configuration.listen_address).await?;
        let node = Node::start(runtime).await?;
        Ok(Self {
            listener,
            node,
            configuration: Arc::new(configuration),
        })
    }

    pub fn local_address(&self) -> Result<SocketAddr, GatewayError> {
        Ok(self.listener.local_addr()?)
    }

    pub async fn serve(
        self,
        cancellation: CancellationToken,
    ) -> Result<GatewayReport, GatewayError> {
        let (archives, archive_task) = match self.configuration.archive_directory.clone() {
            Some(directory) => {
                let (sender, task) = archive::start(directory, self.configuration.archive_capacity);
                (Some(sender), Some(task))
            }
            None => (None, None),
        };
        let capacity = Arc::new(Semaphore::new(self.configuration.max_connections));
        let mut connections = JoinSet::new();
        let mut report = GatewayReport::default();
        loop {
            tokio::select! {
                _ = cancellation.cancelled() => break,
                completed = connections.join_next(), if !connections.is_empty() => {
                    accumulate(&mut report, completed.expect("join set is nonempty"))?;
                }
                accepted = self.listener.accept() => {
                    let (stream, _) = accepted?;
                    let Ok(permit) = capacity.clone().try_acquire_owned() else {
                        report.rejected_connections += 1;
                        continue;
                    };
                    stream.set_nodelay(true)?;
                    report.connections += 1;
                    let ingress = self.node.ingress();
                    let configuration = self.configuration.clone();
                    let archives = archives.clone();
                    let connection_cancellation = cancellation.child_token();
                    connections.spawn(async move {
                        let _permit = permit;
                        connection::run(stream, ingress, configuration, archives, connection_cancellation).await
                    });
                }
            }
        }
        while let Some(completed) = connections.join_next().await {
            accumulate(&mut report, completed)?;
        }
        drop(archives);
        if let Some(task) = archive_task {
            report.archives = task.await?;
        }
        report.runtime = self.node.metrics();
        self.node.shutdown().await?;
        Ok(report)
    }
}

fn accumulate(
    report: &mut GatewayReport,
    completed: Result<connection::ConnectionReport, tokio::task::JoinError>,
) -> Result<(), GatewayError> {
    let connection = completed?;
    report.failed_connections += u64::from(connection.failed);
    report.archive_queue_rejections += u64::from(connection.archive_rejected);
    Ok(())
}

pub(crate) type ArchiveSender = mpsc::Sender<ArchiveRecord>;
