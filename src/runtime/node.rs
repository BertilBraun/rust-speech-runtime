use std::sync::{Arc, atomic::AtomicU64};

use tokio::{sync::mpsc, task::JoinHandle};
use tokio_util::sync::CancellationToken;

use crate::{
    config::RuntimeConfig,
    metrics::{Metrics, MetricsSnapshot},
    protocol::ErrorCode,
    session::manager::run_manager,
    worker::WorkerHandle,
};

use super::{Ingress, RuntimeError};

/// Owns worker tasks and session admission for one inference node.
///
/// Backends must already be listening when [`Self::start`] is called. Use
/// [`Self::shutdown`] to wait for task cleanup; dropping the node only signals cancellation.
pub struct Node {
    ingress: Ingress,
    cancellation: CancellationToken,
    tasks: Vec<JoinHandle<()>>,
}

struct StartedWorkers {
    handles: Vec<WorkerHandle>,
    tasks: Vec<JoinHandle<()>>,
}

impl Node {
    /// Validates configuration, connects all workers, then starts admission.
    ///
    /// If any worker cannot start, previously started tasks are cancelled and joined.
    pub async fn start(configuration: RuntimeConfig) -> Result<Self, RuntimeError> {
        configuration
            .validate()
            .map_err(|error| RuntimeError::new(ErrorCode::InvalidInput, error.to_string()))?;

        let cancellation = CancellationToken::new();
        let metrics = Arc::new(Metrics::new(configuration.workers.len()));
        let mut workers = start_workers(&configuration, &metrics, &cancellation).await?;
        let (sender, receiver) = mpsc::channel(configuration.mailbox_capacity);
        workers.tasks.push(tokio::spawn(run_manager(
            receiver,
            sender.clone(),
            workers.handles,
            configuration,
            metrics.clone(),
            cancellation.child_token(),
        )));

        Ok(Self {
            ingress: Ingress::new(sender, metrics, cancellation.clone()),
            cancellation,
            tasks: workers.tasks,
        })
    }

    /// Returns a cloneable admission handle for application or gateway tasks.
    pub fn ingress(&self) -> Ingress {
        self.ingress.clone()
    }

    /// Returns cumulative observations for this node's lifetime.
    pub fn metrics(&self) -> MetricsSnapshot {
        self.ingress.metrics()
    }

    /// Cancels admission and worker tasks and waits for their termination.
    ///
    /// Close sessions and archive their records before calling this method when records
    /// must be retained. The WebSocket gateway performs that drain before node shutdown.
    pub async fn shutdown(mut self) -> Result<(), RuntimeError> {
        self.cancellation.cancel();
        for task in self.tasks.drain(..) {
            task.await
                .map_err(|error| RuntimeError::new(ErrorCode::BackendFailed, error.to_string()))?;
        }
        Ok(())
    }
}

impl Drop for Node {
    fn drop(&mut self) {
        self.cancellation.cancel();
    }
}

async fn start_workers(
    configuration: &RuntimeConfig,
    metrics: &Arc<Metrics>,
    cancellation: &CancellationToken,
) -> Result<StartedWorkers, RuntimeError> {
    let generation_counter = Arc::new(AtomicU64::new(1));
    let mut workers = StartedWorkers {
        handles: Vec::with_capacity(configuration.workers.len()),
        tasks: Vec::with_capacity(configuration.workers.len()),
    };

    for (worker_id, worker_configuration) in configuration.workers.iter().enumerate() {
        let started = WorkerHandle::start(
            worker_id,
            worker_configuration,
            configuration.clone(),
            metrics.clone(),
            generation_counter.clone(),
            cancellation.child_token(),
        )
        .await;

        let (handle, task) = match started {
            Ok(worker) => worker,
            Err(error) => {
                cancellation.cancel();
                for task in workers.tasks {
                    let _ = task.await;
                }
                return Err(error);
            }
        };
        workers.handles.push(handle);
        workers.tasks.push(task);
    }
    Ok(workers)
}
