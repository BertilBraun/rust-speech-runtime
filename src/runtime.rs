use crate::{
    config::RuntimeConfig,
    metrics::{Metrics, MetricsSnapshot},
    protocol::{ErrorCode, SessionEvent, SessionId, SessionRecord, TurnId},
    session::manager::{ManagerCommand, run_manager},
    worker::{Command, WorkerHandle},
};
use bytes::Bytes;
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};
use thiserror::Error;
use tokio::{
    sync::{mpsc, oneshot},
    task::JoinHandle,
};
use tokio_util::sync::CancellationToken;

#[derive(Clone, Debug, Error)]
#[error("{code:?}: {message}")]
pub struct RuntimeError {
    code: ErrorCode,
    message: String,
}
impl RuntimeError {
    pub fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
    pub fn code(&self) -> ErrorCode {
        self.code
    }
}

pub struct Node {
    ingress: Ingress,
    cancellation: CancellationToken,
    tasks: Vec<JoinHandle<()>>,
}
#[derive(Clone)]
pub struct Ingress {
    manager: mpsc::Sender<ManagerCommand>,
    metrics: Arc<Metrics>,
    cancellation: CancellationToken,
}

impl Node {
    pub async fn start(config: RuntimeConfig) -> Result<Self, RuntimeError> {
        config
            .validate()
            .map_err(|error| RuntimeError::new(ErrorCode::InvalidInput, error.to_string()))?;
        let cancellation = CancellationToken::new();
        let metrics = Arc::new(Metrics::new(config.workers.len()));
        let clock = Arc::new(AtomicU64::new(1));
        let mut workers = Vec::with_capacity(config.workers.len());
        let mut tasks = Vec::new();
        for (worker_id, worker_config) in config.workers.iter().enumerate() {
            match WorkerHandle::start(
                worker_id,
                worker_config,
                config.clone(),
                metrics.clone(),
                clock.clone(),
                cancellation.child_token(),
            )
            .await
            {
                Ok((worker, task)) => {
                    workers.push(worker);
                    tasks.push(task);
                }
                Err(error) => {
                    cancellation.cancel();
                    for task in tasks {
                        let _ = task.await;
                    }
                    return Err(error);
                }
            }
        }
        let (sender, receiver) = mpsc::channel(config.mailbox_capacity);
        tasks.push(tokio::spawn(run_manager(
            receiver,
            sender.clone(),
            workers,
            config,
            metrics.clone(),
            cancellation.child_token(),
        )));
        Ok(Self {
            ingress: Ingress {
                manager: sender,
                metrics,
                cancellation: cancellation.clone(),
            },
            cancellation,
            tasks,
        })
    }
    pub fn ingress(&self) -> Ingress {
        self.ingress.clone()
    }
    pub fn metrics(&self) -> MetricsSnapshot {
        self.ingress.metrics.snapshot()
    }
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

impl Ingress {
    pub async fn open_session(&self, session_id: SessionId) -> Result<SessionHandle, RuntimeError> {
        if session_id.0.is_empty() || session_id.0.len() > 128 {
            return Err(RuntimeError::new(
                ErrorCode::InvalidInput,
                "session ID must contain 1..128 bytes",
            ));
        }
        let (reply, response) = oneshot::channel();
        self.manager
            .try_send(ManagerCommand::Open { session_id, reply })
            .map_err(|error| {
                self.metrics.saturation.fetch_add(1, Ordering::Relaxed);
                RuntimeError::new(
                    match error {
                        mpsc::error::TrySendError::Closed(_) => ErrorCode::Shutdown,
                        mpsc::error::TrySendError::Full(_) => ErrorCode::ChannelSaturated,
                    },
                    "session manager unavailable",
                )
            })?;
        tokio::select! { _ = self.cancellation.cancelled() => Err(RuntimeError::new(ErrorCode::Shutdown, "node stopped")), result = response => result.map_err(|_| RuntimeError::new(ErrorCode::Shutdown, "session manager stopped"))? }
    }
    pub fn metrics(&self) -> MetricsSnapshot {
        self.metrics.snapshot()
    }
}

pub struct SessionHandle {
    pub(crate) key: String,
    pub(crate) worker_id: usize,
    pub(crate) worker: WorkerHandle,
    pub(crate) events: mpsc::Receiver<SessionEvent>,
    pub(crate) generation: Arc<AtomicU64>,
    pub(crate) cancellation: CancellationToken,
    pub(crate) manager: mpsc::Sender<ManagerCommand>,
    pub(crate) session_id: SessionId,
}
impl SessionHandle {
    pub fn worker_id(&self) -> usize {
        self.worker_id
    }
    pub fn current_generation(&self) -> u64 {
        self.generation.load(Ordering::Acquire)
    }
    pub fn generation_fence(&self) -> Arc<AtomicU64> {
        self.generation.clone()
    }
    pub async fn begin_turn(&self, turn_id: TurnId) -> Result<(), RuntimeError> {
        self.request(|reply| Command::Begin {
            key: self.key.clone(),
            turn_id,
            reply,
        })
        .await
    }
    pub async fn audio(
        &self,
        turn_id: TurnId,
        chunk_index: u32,
        audio: Bytes,
    ) -> Result<(), RuntimeError> {
        self.request(|reply| Command::Audio {
            key: self.key.clone(),
            turn_id,
            chunk_index,
            audio,
            reply,
        })
        .await
    }
    pub async fn commit(
        &self,
        turn_id: TurnId,
        final_chunk_count: u32,
        sample_count: usize,
    ) -> Result<(), RuntimeError> {
        self.request(|reply| Command::Commit {
            key: self.key.clone(),
            turn_id,
            final_chunk_count,
            sample_count,
            reply,
        })
        .await
    }
    pub async fn cancel(&self, turn_id: TurnId) -> Result<(), RuntimeError> {
        self.request(|reply| Command::Cancel {
            key: self.key.clone(),
            turn_id,
            reply,
        })
        .await
    }
    async fn request(
        &self,
        command: impl FnOnce(oneshot::Sender<Result<(), RuntimeError>>) -> Command,
    ) -> Result<(), RuntimeError> {
        let (reply, response) = oneshot::channel();
        self.worker.send(command(reply))?;
        response
            .await
            .map_err(|_| RuntimeError::new(ErrorCode::Shutdown, "worker stopped"))?
    }
    pub async fn next_event(&mut self) -> Option<SessionEvent> {
        if let Ok(event) = self.events.try_recv() {
            return Some(event);
        }
        tokio::select! {biased;event=self.events.recv()=>event,_=self.cancellation.cancelled()=>None}
    }
    pub async fn close(&mut self) -> Result<SessionRecord, RuntimeError> {
        let (reply, response) = oneshot::channel();
        self.worker.send(Command::Close {
            key: self.key.clone(),
            reply: Some(reply),
        })?;
        let record = response
            .await
            .map_err(|_| RuntimeError::new(ErrorCode::Shutdown, "worker stopped"))??;
        self.cancellation.cancel();
        let _ = self.manager.try_send(ManagerCommand::Release {
            session_id: self.session_id.clone(),
            key: self.key.clone(),
        });
        Ok(record)
    }
}
impl Drop for SessionHandle {
    fn drop(&mut self) {
        self.cancellation.cancel();
    }
}
