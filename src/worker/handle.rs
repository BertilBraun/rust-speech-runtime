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
    pub worker_id: usize,
    sender: mpsc::Sender<Command>,
    session_count: Arc<AtomicUsize>,
    backend_available: Arc<AtomicBool>,
    metrics: Arc<Metrics>,
}

impl WorkerHandle {
    pub async fn start(
        worker_id: usize,
        worker_configuration: &WorkerConfig,
        configuration: RuntimeConfig,
        metrics: Arc<Metrics>,
        generation_counter: Arc<AtomicU64>,
        cancellation: CancellationToken,
    ) -> Result<(Self, JoinHandle<()>), RuntimeError> {
        let (connection, backend_capabilities) = tokio::time::timeout(
            std::time::Duration::from_millis(configuration.backend_timeout_ms),
            backend::BackendConnection::connect(worker_configuration.endpoint),
        )
        .await
        .map_err(|_| {
            RuntimeError::new(ErrorCode::BackendUnavailable, "worker startup timeout")
        })??;
        let (sender, receiver) = mpsc::channel(configuration.mailbox_capacity);
        let session_count = Arc::new(AtomicUsize::new(0));
        let backend_available = Arc::new(AtomicBool::new(true));
        let actor = actor::WorkerActor::new(
            worker_id,
            configuration,
            backend_capabilities,
            metrics.clone(),
            generation_counter,
            session_count.clone(),
            backend_available.clone(),
        );
        let task = tokio::spawn(actor.run(receiver, connection, cancellation));
        Ok((
            Self {
                worker_id,
                sender,
                session_count,
                backend_available,
                metrics,
            },
            task,
        ))
    }

    /// Includes pending opens and closing sessions until their state is released.
    pub fn session_count(&self) -> usize {
        self.session_count.load(Ordering::Relaxed)
    }

    pub fn is_available(&self) -> bool {
        self.backend_available.load(Ordering::Acquire)
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
        session_key: String,
        reply: oneshot::Sender<Result<SessionRecord, RuntimeError>>,
    ) -> Result<(), RuntimeError> {
        self.sender
            .send(Command::Close {
                session_key,
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
