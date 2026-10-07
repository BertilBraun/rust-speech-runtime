//! Coordinates commands and completions; this task alone mutates scheduling state.

use super::{
    Command,
    backend::BackendConnection,
    execution::{BatchCompletion, BatchJob, execute_batches},
};
use crate::{
    config::RuntimeConfig,
    metrics::Metrics,
    protocol::{
        ErrorCode,
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

/// Retains request identities until the execution task returns their proposals.
pub(super) struct InFlightBatch {
    pub request: BatchRequest,
    pub kind: BatchKind,
}

/// One task owns scheduling decisions; shared atomics expose only placement status.
pub(super) struct WorkerActor {
    pub worker_id: usize,
    pub configuration: RuntimeConfig,
    pub backend_capabilities: Ready,
    pub sessions: HashMap<String, SessionState>,
    pub metrics: Arc<Metrics>,
    pub generation_counter: Arc<AtomicU64>,
    pub session_count: Arc<AtomicUsize>,
    pub backend_available: Arc<AtomicBool>,
    pub forward_costs: CostModel,
    pub next_operation_id: u64,
    pub in_flight_batch: Option<InFlightBatch>,
    pub consecutive_decode_batches: usize,
}

impl WorkerActor {
    pub fn new(
        worker_id: usize,
        configuration: RuntimeConfig,
        backend_capabilities: Ready,
        metrics: Arc<Metrics>,
        generation_counter: Arc<AtomicU64>,
        session_count: Arc<AtomicUsize>,
        backend_available: Arc<AtomicBool>,
    ) -> Self {
        let forward_costs = CostModel::new(configuration.initial_forward_estimate_ms);
        Self {
            worker_id,
            configuration,
            backend_capabilities,
            sessions: HashMap::new(),
            metrics,
            generation_counter,
            session_count,
            backend_available,
            forward_costs,
            next_operation_id: 0,
            in_flight_batch: None,
            consecutive_decode_batches: 0,
        }
    }

    pub async fn run(
        mut self,
        command_receiver: mpsc::Receiver<Command>,
        connection: BackendConnection,
        cancellation: CancellationToken,
    ) {
        let (job_sender, job_receiver) = mpsc::channel(1);
        let (completion_sender, completion_receiver) = mpsc::channel(1);
        let execution = tokio::spawn(execute_batches(
            connection,
            job_receiver,
            completion_sender,
            Duration::from_millis(self.configuration.backend_timeout_ms),
            cancellation.child_token(),
        ));
        self.run_control_loop(
            command_receiver,
            completion_receiver,
            &job_sender,
            &cancellation,
        )
        .await;
        self.stop_sessions();
        cancellation.cancel();
        let _ = execution.await;
    }

    async fn run_control_loop(
        &mut self,
        mut command_receiver: mpsc::Receiver<Command>,
        mut completion_receiver: mpsc::Receiver<BatchCompletion>,
        job_sender: &mpsc::Sender<BatchJob>,
        cancellation: &CancellationToken,
    ) {
        let mut cleanup_timer = tokio::time::interval(Duration::from_millis(50));

        // Backend I/O runs separately, so input and cancellation stay responsive during a forward.
        loop {
            tokio::select! {
                _ = cancellation.cancelled() => break,
                command = command_receiver.recv() => {
                    let Some(command) = command else {
                        break;
                    };
                    self.process_command(command);
                }
                completion = completion_receiver.recv(), if self.in_flight_batch.is_some() => {
                    self.handle_completion(completion);
                }
                _ = cleanup_timer.tick() => self.cleanup_disconnected_sessions(),
            }
            self.schedule_next_batch(job_sender);
        }
    }

    fn process_command(&mut self, command: Command) {
        let started = tokio::time::Instant::now();
        self.handle_command(command);
        self.metrics
            .runtime_timing
            .command_processing
            .record(started.elapsed().as_secs_f64() * 1000.0);
    }

    fn handle_completion(&mut self, completion: Option<BatchCompletion>) {
        let Some(completion) = completion else {
            self.handle_backend_failure(RuntimeError::new(
                ErrorCode::BackendUnavailable,
                "execution task stopped",
            ));
            return;
        };
        self.metrics
            .runtime_timing
            .completion_handoff
            .record(completion.completed_at.elapsed().as_secs_f64() * 1000.0);
        self.metrics
            .runtime_timing
            .execution_handoff
            .record(completion.execution_handoff_ms);
        let started = tokio::time::Instant::now();
        if let Err(error) = self.apply_completion(completion) {
            self.handle_backend_failure(error);
        }
        self.metrics
            .runtime_timing
            .completion_processing
            .record(started.elapsed().as_secs_f64() * 1000.0);
    }

    fn schedule_next_batch(&mut self, job_sender: &mpsc::Sender<BatchJob>) {
        if !self.backend_available.load(Ordering::Acquire) || self.in_flight_batch.is_some() {
            return;
        }

        let started = tokio::time::Instant::now();
        let Some((kind, session_keys)) = select_batch(
            &self.sessions,
            self.batch_size_limit(),
            self.configuration.max_prefill_batch_size,
            self.consecutive_decode_batches,
            &self.forward_costs,
            Duration::from_millis(self.configuration.max_prefill_wait_ms),
        ) else {
            return;
        };

        let job = self.build_batch(kind, session_keys);
        self.metrics
            .runtime_timing
            .batch_build
            .record(started.elapsed().as_secs_f64() * 1000.0);
        if job_sender.try_send(job).is_err() {
            self.handle_backend_failure(RuntimeError::new(
                ErrorCode::BackendUnavailable,
                "execution queue stopped",
            ));
        }
    }

    /// Allocates an epoch shared across workers and reopened session identities.
    pub(super) fn allocate_generation_id(&self) -> u64 {
        self.generation_counter.fetch_add(1, Ordering::Relaxed)
    }

    /// Applies node policy without exceeding the backend's advertised capability.
    pub(super) fn batch_size_limit(&self) -> usize {
        self.configuration
            .max_batch_size
            .min(self.backend_capabilities.max_batch_size)
    }

    pub(super) fn context_token_limit(&self) -> usize {
        self.configuration
            .max_context_tokens
            .min(self.backend_capabilities.max_context_tokens)
    }
}
