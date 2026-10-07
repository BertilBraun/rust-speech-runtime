use std::{net::SocketAddr, sync::Arc};

use serde::Serialize;
use tokio::{
    net::{TcpListener, TcpStream},
    sync::{OwnedSemaphorePermit, Semaphore, mpsc},
    task::JoinSet,
};
use tokio_util::sync::CancellationToken;

use crate::{Node, RuntimeError, config::RuntimeConfig, metrics::MetricsSnapshot};

use super::{
    GatewayConfig,
    archive::{self, ArchiveRecord, ArchiveReport},
    connection,
};

/// Typed failures at configuration, network, runtime and task boundaries.
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

/// Final connection, archive and runtime observations returned after orderly shutdown.
#[derive(Debug, Default, Serialize)]
pub struct GatewayReport {
    pub connections: u64,
    pub rejected_connections: u64,
    pub failed_connections: u64,
    pub archive_queue_rejections: u64,
    pub archive_backpressure_events: u64,
    pub archives: ArchiveReport,
    pub runtime: MetricsSnapshot,
}

/// Public WebSocket ingress backed by one Node and bounded connection/archive tasks.
/// Bind once, then serve until cancellation; each connection owns one sticky session.
pub struct Gateway {
    listener: TcpListener,
    node: Node,
    configuration: Arc<GatewayConfig>,
}

impl Gateway {
    /// Validates transport limits, binds the listener and connects configured model workers.
    /// Serving starts only after every worker reports readiness.
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

    /// Returns the bound address, including an OS-selected port when configured as zero.
    pub fn local_address(&self) -> Result<SocketAddr, GatewayError> {
        Ok(self.listener.local_addr()?)
    }

    /// Accepts connections until cancelled, then closes sessions and drains archive work.
    /// Runtime shutdown follows connection cleanup so accepted records can be retained.
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
        let mut connections = JoinSet::new();
        let mut report = GatewayReport::default();
        let mut outcome = self
            .accept_connections(
                &cancellation,
                &mut connections,
                archives.as_ref(),
                &mut report,
            )
            .await;
        if outcome.is_err() {
            cancellation.cancel();
        }
        while let Some(completed) = connections.join_next().await {
            outcome = outcome.and(accumulate(&mut report, completed));
        }
        drop(archives);
        if let Some(task) = archive_task {
            match task.await {
                Ok(archives) => report.archives = archives,
                Err(error) => outcome = outcome.and(Err(error.into())),
            }
        }
        report.runtime = self.node.metrics();
        outcome = outcome.and(self.node.shutdown().await.map_err(GatewayError::from));
        outcome.map(|()| report)
    }

    async fn accept_connections(
        &self,
        cancellation: &CancellationToken,
        connections: &mut JoinSet<connection::ConnectionReport>,
        archives: Option<&ArchiveSender>,
        report: &mut GatewayReport,
    ) -> Result<(), GatewayError> {
        let capacity = Arc::new(Semaphore::new(self.configuration.max_connections));
        loop {
            tokio::select! {
                _ = cancellation.cancelled() => break,
                completed = connections.join_next(), if !connections.is_empty() => {
                    accumulate(report, completed.expect("join set is nonempty"))?;
                }
                accepted = self.listener.accept() => {
                    let (stream, _) = accepted?;
                    let Ok(permit) = capacity.clone().try_acquire_owned() else {
                        report.rejected_connections += 1;
                        continue;
                    };
                    stream.set_nodelay(true)?;
                    report.connections += 1;
                    self.spawn_connection(stream, permit, connections, archives, cancellation);
                }
            }
        }
        Ok(())
    }

    fn spawn_connection(
        &self,
        stream: TcpStream,
        permit: OwnedSemaphorePermit,
        connections: &mut JoinSet<connection::ConnectionReport>,
        archives: Option<&ArchiveSender>,
        cancellation: &CancellationToken,
    ) {
        let ingress = self.node.ingress();
        let configuration = self.configuration.clone();
        let archives = archives.cloned();
        let connection_cancellation = cancellation.child_token();
        connections.spawn(async move {
            // Retain the permit through archive backpressure to bound records awaiting disk.
            let _permit = permit;
            connection::run(
                stream,
                ingress,
                configuration,
                archives,
                connection_cancellation,
            )
            .await
        });
    }
}

fn accumulate(
    report: &mut GatewayReport,
    completed: Result<connection::ConnectionReport, tokio::task::JoinError>,
) -> Result<(), GatewayError> {
    let connection = completed?;
    report.failed_connections += u64::from(connection.failed);
    report.archive_queue_rejections += u64::from(connection.archive_rejected);
    report.archive_backpressure_events += u64::from(connection.archive_backpressured);
    Ok(())
}

pub(crate) type ArchiveSender = mpsc::Sender<ArchiveRecord>;
