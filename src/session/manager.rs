//! Owns session routes and placement; pending opens never block control for other workers.

use crate::{
    config::RuntimeConfig,
    metrics::Metrics,
    protocol::{ErrorCode, SessionId},
    runtime::{RuntimeError, SessionHandle},
    worker::{Command, WorkerHandle},
};
use futures_util::{StreamExt, future::BoxFuture, stream::FuturesUnordered};
use std::{
    collections::HashMap,
    sync::{Arc, atomic::Ordering},
    time::Duration,
};
use tokio::sync::{mpsc, oneshot};
use tokio_util::sync::CancellationToken;

pub(crate) enum ManagerCommand {
    Open {
        session_id: SessionId,
        reply: oneshot::Sender<Result<SessionHandle, RuntimeError>>,
    },
    Release {
        session_id: SessionId,
        key: String,
    },
}
struct Route {
    key: String,
    worker_id: usize,
    cancellation: CancellationToken,
}
struct OpenCompletion {
    reply: oneshot::Sender<Result<SessionHandle, RuntimeError>>,
    result: Result<SessionHandle, RuntimeError>,
}
struct SessionManager {
    sender: mpsc::Sender<ManagerCommand>,
    workers: Vec<WorkerHandle>,
    config: RuntimeConfig,
    metrics: Arc<Metrics>,
    cancellation: CancellationToken,
    routes: HashMap<SessionId, Route>,
    pending: FuturesUnordered<BoxFuture<'static, OpenCompletion>>,
    sequence: u64,
}

pub(crate) async fn run_manager(
    mut receiver: mpsc::Receiver<ManagerCommand>,
    sender: mpsc::Sender<ManagerCommand>,
    workers: Vec<WorkerHandle>,
    config: RuntimeConfig,
    metrics: Arc<Metrics>,
    cancellation: CancellationToken,
) {
    let mut manager = SessionManager {
        sender,
        workers,
        config,
        metrics,
        cancellation,
        routes: HashMap::new(),
        pending: FuturesUnordered::new(),
        sequence: 0,
    };
    let mut reap = tokio::time::interval(Duration::from_millis(100));
    loop {
        tokio::select! {
            _ = manager.cancellation.cancelled() => break,
            _ = reap.tick() => manager.reap(),
            Some(completion) = manager.pending.next(), if !manager.pending.is_empty() => {
                manager.complete_open(completion);
            }
            command = receiver.recv() => {
                let Some(command) = command else {
                    break;
                };
                manager.command(command);
            }
        }
    }
    for route in manager.routes.into_values() {
        route.cancellation.cancel();
    }
}

impl SessionManager {
    fn reap(&mut self) {
        self.routes
            .retain(|_, route| !route.cancellation.is_cancelled());
    }

    fn command(&mut self, command: ManagerCommand) {
        match command {
            ManagerCommand::Release { session_id, key } => {
                if self
                    .routes
                    .get(&session_id)
                    .is_some_and(|route| route.key == key)
                {
                    self.routes.remove(&session_id);
                }
            }
            ManagerCommand::Open { session_id, reply } => match self.reserve(session_id) {
                Ok((handle, opened)) => self
                    .pending
                    .push(Box::pin(wait_open(handle, opened, reply))),
                Err(error) => self.complete_open(OpenCompletion {
                    reply,
                    result: Err(error),
                }),
            },
        }
    }

    fn complete_open(&self, completion: OpenCompletion) {
        if completion.result.is_err() {
            self.metrics
                .rejected_sessions
                .fetch_add(1, Ordering::Relaxed);
        }
        let _ = completion.reply.send(completion.result);
    }

    fn choose_worker(&self) -> Result<WorkerHandle, RuntimeError> {
        let load = |worker: &WorkerHandle| {
            worker.load().max(
                self.routes
                    .values()
                    .filter(|route| route.worker_id == worker.id)
                    .count(),
            )
        };
        self.workers
            .iter()
            .filter(|worker| {
                worker.available() && load(worker) < self.config.max_sessions_per_worker
            })
            .min_by_key(|worker| (load(worker), worker.id))
            .cloned()
            .ok_or_else(|| {
                RuntimeError::new(
                    ErrorCode::CapacityExceeded,
                    "no worker has a free session slot",
                )
            })
    }

    fn reserve(
        &mut self,
        session_id: SessionId,
    ) -> Result<(SessionHandle, oneshot::Receiver<Result<(), RuntimeError>>), RuntimeError> {
        self.reap();
        if self.routes.contains_key(&session_id) {
            return Err(RuntimeError::new(
                ErrorCode::SessionExists,
                "session ID already active",
            ));
        }
        let worker = self.choose_worker()?;
        self.sequence += 1;
        let key = format!("{}:{}", self.sequence, session_id.0);
        let cancellation = self.cancellation.child_token();
        let (events, receiver) = mpsc::channel(self.config.event_capacity);
        let (reply, opened) = oneshot::channel();
        worker.send(Command::Open {
            key: key.clone(),
            session_id: session_id.clone(),
            events,
            cancellation: cancellation.clone(),
            reply,
        })?;
        self.routes.insert(
            session_id.clone(),
            Route {
                key: key.clone(),
                worker_id: worker.id,
                cancellation: cancellation.clone(),
            },
        );
        let handle = SessionHandle {
            key,
            worker_id: worker.id,
            worker,
            events: receiver,
            cancellation,
            manager: self.sender.clone(),
            session_id,
        };
        Ok((handle, opened))
    }
}

async fn wait_open(
    handle: SessionHandle,
    opened: oneshot::Receiver<Result<(), RuntimeError>>,
    reply: oneshot::Sender<Result<SessionHandle, RuntimeError>>,
) -> OpenCompletion {
    let result = tokio::select! {
        _ = handle.cancellation.cancelled() => {
            Err(RuntimeError::new(ErrorCode::Shutdown, "session opening cancelled"))
        }
        result = opened => {
            result.unwrap_or_else(|_| {
                Err(RuntimeError::new(ErrorCode::BackendUnavailable, "worker stopped"))
            })
        }
    };
    OpenCompletion {
        reply,
        result: result.map(|()| handle),
    }
}
