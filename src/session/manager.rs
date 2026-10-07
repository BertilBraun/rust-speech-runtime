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
        session_key: String,
    },
}

struct Route {
    session_key: String,
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
    configuration: RuntimeConfig,
    metrics: Arc<Metrics>,
    cancellation: CancellationToken,
    routes: HashMap<SessionId, Route>,
    pending_opens: FuturesUnordered<BoxFuture<'static, OpenCompletion>>,
    next_session_number: u64,
}

pub(crate) async fn run_manager(
    mut receiver: mpsc::Receiver<ManagerCommand>,
    sender: mpsc::Sender<ManagerCommand>,
    workers: Vec<WorkerHandle>,
    configuration: RuntimeConfig,
    metrics: Arc<Metrics>,
    cancellation: CancellationToken,
) {
    let mut manager = SessionManager {
        sender,
        workers,
        configuration,
        metrics,
        cancellation,
        routes: HashMap::new(),
        pending_opens: FuturesUnordered::new(),
        next_session_number: 0,
    };
    let mut cleanup_timer = tokio::time::interval(Duration::from_millis(100));
    loop {
        tokio::select! {
            _ = manager.cancellation.cancelled() => break,
            _ = cleanup_timer.tick() => manager.remove_cancelled_routes(),
            Some(completion) = manager.pending_opens.next(), if !manager.pending_opens.is_empty() => {
                manager.complete_open(completion);
            }
            command = receiver.recv() => {
                let Some(command) = command else {
                    break;
                };
                manager.handle_command(command);
            }
        }
    }
    for route in manager.routes.into_values() {
        route.cancellation.cancel();
    }
}

impl SessionManager {
    fn remove_cancelled_routes(&mut self) {
        self.routes
            .retain(|_, route| !route.cancellation.is_cancelled());
    }

    fn handle_command(&mut self, command: ManagerCommand) {
        match command {
            ManagerCommand::Release {
                session_id,
                session_key,
            } => {
                if self
                    .routes
                    .get(&session_id)
                    .is_some_and(|route| route.session_key == session_key)
                {
                    self.routes.remove(&session_id);
                }
            }
            ManagerCommand::Open { session_id, reply } => self.start_open(session_id, reply),
        }
    }

    fn start_open(
        &mut self,
        session_id: SessionId,
        reply: oneshot::Sender<Result<SessionHandle, RuntimeError>>,
    ) {
        match self.reserve_session(session_id) {
            Ok((handle, opened)) => {
                self.pending_opens
                    .push(Box::pin(wait_open(handle, opened, reply)));
            }
            Err(error) => self.complete_open(OpenCompletion {
                reply,
                result: Err(error),
            }),
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
        let mut selected_worker: Option<&WorkerHandle> = None;
        let mut selected_priority = (usize::MAX, usize::MAX);
        for worker in &self.workers {
            let session_count = self.reserved_session_count(worker);
            if !worker.is_available() || session_count >= self.configuration.max_sessions_per_worker
            {
                continue;
            }
            // Prefer fewer sessions, then lower worker ID, independent of vector order.
            let placement_priority = (session_count, worker.worker_id);
            if placement_priority < selected_priority {
                selected_worker = Some(worker);
                selected_priority = placement_priority;
            }
        }
        selected_worker.cloned().ok_or_else(|| {
            RuntimeError::new(
                ErrorCode::CapacityExceeded,
                "no worker has a free session slot",
            )
        })
    }

    fn reserved_session_count(&self, worker: &WorkerHandle) -> usize {
        let routed_sessions = self
            .routes
            .values()
            .filter(|route| route.worker_id == worker.worker_id)
            .count();
        // Routes include opens not yet visible in the worker's atomic session count.
        worker.session_count().max(routed_sessions)
    }

    fn reserve_session(
        &mut self,
        session_id: SessionId,
    ) -> Result<(SessionHandle, oneshot::Receiver<Result<(), RuntimeError>>), RuntimeError> {
        self.remove_cancelled_routes();
        if self.routes.contains_key(&session_id) {
            return Err(RuntimeError::new(
                ErrorCode::SessionExists,
                "session ID already active",
            ));
        }
        let worker = self.choose_worker()?;
        self.next_session_number += 1;
        let session_key = format!("{}:{}", self.next_session_number, session_id.0);
        let cancellation = self.cancellation.child_token();
        let (events, receiver) = mpsc::channel(self.configuration.event_capacity);
        let (reply, opened) = oneshot::channel();
        worker.send(Command::Open {
            session_key: session_key.clone(),
            session_id: session_id.clone(),
            events,
            cancellation: cancellation.clone(),
            reply,
        })?;
        self.routes.insert(
            session_id.clone(),
            Route {
                session_key: session_key.clone(),
                worker_id: worker.worker_id,
                cancellation: cancellation.clone(),
            },
        );
        let handle = SessionHandle {
            session_key,
            worker_id: worker.worker_id,
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
