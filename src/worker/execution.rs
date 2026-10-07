//! Executes one backend request at a time without borrowing actor scheduling state.

use super::backend::BackendConnection;
use crate::{
    protocol::{
        ErrorCode,
        backend::{BatchRequest, BatchResponse},
    },
    runtime::RuntimeError,
};
use std::time::Duration;
use tokio::{sync::mpsc, time::Instant};
use tokio_util::sync::CancellationToken;

/// Owned request and packed PCM body; this never borrows scheduling state.
pub(super) struct BatchJob {
    pub request: BatchRequest,
    pub audio: Vec<u8>,
}

/// Backend proposals and complete round-trip time, awaiting actor acceptance.
pub(super) struct BatchCompletion {
    pub response: Result<BatchResponse, RuntimeError>,
    pub elapsed_ms: f64,
}

pub(super) async fn execute_batches(
    mut connection: BackendConnection,
    mut job_receiver: mpsc::Receiver<BatchJob>,
    completion_sender: mpsc::Sender<BatchCompletion>,
    timeout: Duration,
    cancellation: CancellationToken,
) {
    loop {
        let next_job = tokio::select! {
            _ = cancellation.cancelled() => None,
            job = job_receiver.recv() => job,
        };
        let Some(job) = next_job else {
            break;
        };

        let completion = tokio::select! {
            _ = cancellation.cancelled() => break,
            completion = execute_batch(&mut connection, job, timeout) => completion,
        };
        let backend_failed = completion.response.is_err();
        if completion_sender.send(completion).await.is_err() || backend_failed {
            break;
        }
    }
}

async fn execute_batch(
    connection: &mut BackendConnection,
    job: BatchJob,
    timeout: Duration,
) -> BatchCompletion {
    let started = Instant::now();
    let response =
        match tokio::time::timeout(timeout, connection.execute(&job.request, &job.audio)).await {
            Ok(response) => response,
            Err(_) => Err(RuntimeError::new(
                ErrorCode::BackendUnavailable,
                "worker execution timeout",
            )),
        };
    BatchCompletion {
        response,
        elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
    }
}
