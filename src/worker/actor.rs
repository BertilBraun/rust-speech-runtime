//! Coordinates commands and completions; this task alone mutates scheduling state.

use super::{
    Command,
    backend::BackendConnection,
    execution::{Completion, Job, run_backend},
};
use crate::{
    config::RuntimeConfig,
    metrics::Metrics,
    protocol::{
        ErrorCode, SessionEvent,
        backend::{BatchRequest, Ready},
    },
    runtime::RuntimeError,
    scheduler::{BatchKind, CostModel, select_batch},
    session::state::SessionState,
};
use std::{
    collections::HashMap,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
    },
    time::Duration,
};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

pub(super) struct ActiveBatch {
    pub request: BatchRequest,
    pub kind: BatchKind,
}
pub(super) struct WorkerActor {
    pub id: usize,
    pub config: RuntimeConfig,
    pub ready: Ready,
    pub sessions: HashMap<String, SessionState>,
    pub metrics: Arc<Metrics>,
    pub clock: Arc<AtomicU64>,
    pub load: Arc<AtomicUsize>,
    pub available: Arc<AtomicBool>,
    pub costs: CostModel,
    pub next_operation: u64,
    pub active: Option<ActiveBatch>,
    pub consecutive_decode_batches: usize,
}

impl WorkerActor {
    pub fn new(
        id: usize,
        config: RuntimeConfig,
        ready: Ready,
        metrics: Arc<Metrics>,
        clock: Arc<AtomicU64>,
        load: Arc<AtomicUsize>,
        available: Arc<AtomicBool>,
    ) -> Self {
        let costs = CostModel::new(config.initial_forward_estimate_ms);
        Self {
            id,
            config,
            ready,
            sessions: HashMap::new(),
            metrics,
            clock,
            load,
            available,
            costs,
            next_operation: 0,
            active: None,
            consecutive_decode_batches: 0,
        }
    }

    pub async fn run(
        mut self,
        commands: mpsc::Receiver<Command>,
        connection: BackendConnection,
        cancellation: CancellationToken,
    ) {
        let (jobs, job_receiver) = mpsc::channel(1);
        let (results, result_receiver) = mpsc::channel(1);
        let execution = tokio::spawn(run_backend(
            connection,
            job_receiver,
            results,
            Duration::from_millis(self.config.backend_timeout_ms),
            cancellation.child_token(),
        ));
        self.run_control_loop(commands, result_receiver, &jobs, &cancellation)
            .await;
        self.stop_sessions();
        cancellation.cancel();
        let _ = execution.await;
    }

    async fn run_control_loop(
        &mut self,
        mut commands: mpsc::Receiver<Command>,
        mut completions: mpsc::Receiver<Completion>,
        jobs: &mpsc::Sender<Job>,
        cancellation: &CancellationToken,
    ) {
        let mut cleanup_timer = tokio::time::interval(Duration::from_millis(50));

        // Backend I/O runs separately, so input and cancellation stay responsive during a forward.
        loop {
            tokio::select! {
                _ = cancellation.cancelled() => break,
                command = commands.recv() => {
                    let Some(command) = command else {
                        break;
                    };
                    self.command(command);
                }
                completion = completions.recv(), if self.active.is_some() => {
                    self.handle_completion(completion);
                }
                _ = cleanup_timer.tick() => self.reap(),
            }
            self.schedule_next_batch(jobs);
        }
    }

    fn handle_completion(&mut self, completion: Option<Completion>) {
        let Some(completion) = completion else {
            self.fail(RuntimeError::new(
                ErrorCode::BackendUnavailable,
                "execution task stopped",
            ));
            return;
        };
        if let Err(error) = self.complete(completion) {
            self.fail(error);
        }
    }

    fn schedule_next_batch(&mut self, jobs: &mpsc::Sender<Job>) {
        if !self.available.load(Ordering::Acquire) || self.active.is_some() {
            return;
        }

        let Some((kind, session_keys)) = select_batch(
            &self.sessions,
            self.config.max_batch_size.min(self.ready.max_batch_size),
            self.config.max_prefill_batch_size,
            self.consecutive_decode_batches,
            &self.costs,
            Duration::from_millis(self.config.max_prefill_wait_ms),
        ) else {
            return;
        };

        let job = self.build_batch(kind, session_keys);
        if jobs.try_send(job).is_err() {
            self.fail(RuntimeError::new(
                ErrorCode::BackendUnavailable,
                "execution queue stopped",
            ));
        }
    }

    fn stop_sessions(&mut self) {
        self.available.store(false, Ordering::Release);
        self.metrics.active_sessions.fetch_sub(
            self.sessions
                .values()
                .filter(|session| !session.closing)
                .count() as u64,
            Ordering::Relaxed,
        );
        self.load.store(0, Ordering::Relaxed);
        for mut session in self.sessions.drain().map(|(_, session)| session) {
            session.cancellation.cancel();
            if let Some(reply) = session.close_reply.take() {
                let _ = reply.send(Err(RuntimeError::new(
                    ErrorCode::BackendUnavailable,
                    "worker stopped",
                )));
            }
        }
    }

    pub fn generation(&self) -> u64 {
        self.clock.fetch_add(1, Ordering::Relaxed)
    }

    fn reap(&mut self) {
        let generation = self.generation();
        for session in self.sessions.values_mut() {
            if !session.closing
                && (session.cancellation.is_cancelled() || session.events.is_closed())
            {
                session.interrupt(generation);
                session.closing = true;
                self.metrics.active_sessions.fetch_sub(1, Ordering::Relaxed);
            }
        }
        self.sessions.retain(|_, session| {
            !session.closing
                || session.in_flight
                || (!session.backend_closed && session.opened)
                || !session.events.is_closed()
        });
        self.load.store(self.sessions.len(), Ordering::Relaxed);
    }

    fn fail(&mut self, error: RuntimeError) {
        self.available.store(false, Ordering::Release);
        self.active = None;
        self.metrics
            .backend_failures
            .fetch_add(1, Ordering::Relaxed);
        for session in self.sessions.values_mut() {
            session.opened = false;
            session.backend_closed = true;
            session.in_flight = false;
            if session.active() {
                session.finish(crate::protocol::FinishReason::BackendFailed);
            }
            if let Some(reply) = session.open_reply.take() {
                let _ = reply.send(Err(error.clone()));
            }
            let _ = session.events.try_send(SessionEvent::Failed {
                turn_id: session.turn().map(|turn| turn.turn_id),
                code: error.code(),
                message: error.to_string(),
            });
        }
        let closing = self
            .sessions
            .iter()
            .filter(|(_, session)| session.closing && session.close_reply.is_some())
            .map(|(key, _)| key.clone())
            .collect::<Vec<_>>();
        for key in closing {
            self.remove_closed(&key);
        }
    }
}
