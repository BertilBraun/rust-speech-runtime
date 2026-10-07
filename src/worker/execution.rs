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

pub(super) struct Job {
    pub request: BatchRequest,
    pub audio: Vec<u8>,
}
pub(super) struct Completion {
    pub response: Result<BatchResponse, RuntimeError>,
    pub elapsed_ms: f64,
}

pub(super) async fn run_backend(
    mut connection: BackendConnection,
    mut jobs: mpsc::Receiver<Job>,
    results: mpsc::Sender<Completion>,
    timeout: Duration,
    cancellation: CancellationToken,
) {
    loop {
        let job = tokio::select! {
            _ = cancellation.cancelled() => break,
            job = jobs.recv() => {
                let Some(job) = job else {
                    break;
                };
                job
            }
        };
        let started = Instant::now();
        let response = tokio::select! {
            _ = cancellation.cancelled() => break,
            result = tokio::time::timeout(timeout, connection.execute(&job.request, &job.audio)) => {
                result.unwrap_or_else(|_| {
                    Err(RuntimeError::new(
                        ErrorCode::BackendUnavailable,
                        "worker execution timeout",
                    ))
                })
            }
        };
        let failed = response.is_err();
        if results
            .send(Completion {
                response,
                elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
            })
            .await
            .is_err()
            || failed
        {
            break;
        }
    }
}
