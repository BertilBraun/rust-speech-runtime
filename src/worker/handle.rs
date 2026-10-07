//! Bounded mailbox access and startup for one sticky worker.

use super::{Command, actor, backend};

use crate::{
    config::{RuntimeConfig, WorkerConfig},
    metrics::Metrics,
    protocol::{ErrorCode, SessionRecord},
    runtime::RuntimeError,
};
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
};
use tokio::{
    sync::{mpsc, oneshot},
    task::JoinHandle,
};
use tokio_util::sync::CancellationToken;

#[derive(Clone)]
pub(crate) struct WorkerHandle {
    pub id: usize,
    sender: mpsc::Sender<Command>,
    load: Arc<AtomicUsize>,
    available: Arc<AtomicBool>,
    metrics: Arc<Metrics>,
}

impl WorkerHandle {
    pub async fn start(
        id: usize,
        worker_config: &WorkerConfig,
        config: RuntimeConfig,
        metrics: Arc<Metrics>,
        clock: Arc<AtomicU64>,
        cancellation: CancellationToken,
    ) -> Result<(Self, JoinHandle<()>), RuntimeError> {
        let (connection, ready) = tokio::time::timeout(
            std::time::Duration::from_millis(config.backend_timeout_ms),
            backend::BackendConnection::connect(worker_config.endpoint),
        )
        .await
        .map_err(|_| {
            RuntimeError::new(ErrorCode::BackendUnavailable, "worker startup timeout")
        })??;
        let (sender, receiver) = mpsc::channel(config.mailbox_capacity);
        let load = Arc::new(AtomicUsize::new(0));
        let available = Arc::new(AtomicBool::new(true));
        let actor = actor::WorkerActor::new(
            id,
            config,
            ready,
            metrics.clone(),
            clock,
            load.clone(),
            available.clone(),
        );
        let task = tokio::spawn(actor.run(receiver, connection, cancellation));
        Ok((
            Self {
                id,
                sender,
                load,
                available,
                metrics,
            },
            task,
        ))
    }

    pub fn load(&self) -> usize {
        self.load.load(Ordering::Relaxed)
    }

    pub fn available(&self) -> bool {
        self.available.load(Ordering::Acquire)
    }

    pub fn send(&self, command: Command) -> Result<(), RuntimeError> {
        self.sender.try_send(command).map_err(|error| {
            self.metrics.saturation.fetch_add(1, Ordering::Relaxed);
            RuntimeError::new(
                match error {
                    mpsc::error::TrySendError::Closed(_) => ErrorCode::BackendUnavailable,
                    mpsc::error::TrySendError::Full(_) => ErrorCode::ChannelSaturated,
                },
                "worker mailbox unavailable",
            )
        })
    }

    pub async fn close(
        &self,
        key: String,
        reply: oneshot::Sender<Result<SessionRecord, RuntimeError>>,
    ) -> Result<(), RuntimeError> {
        self.sender
            .send(Command::Close {
                key,
                reply: Some(reply),
            })
            .await
            .map_err(|_| {
                RuntimeError::new(ErrorCode::BackendUnavailable, "worker stopped before close")
            })
    }
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
