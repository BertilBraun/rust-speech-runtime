use std::time::Duration;

use tokio::{sync::mpsc, time::Instant};

use crate::config::WorkerSlowdown;
use crate::protocol::{Assignment, InputFrame, SessionId};

pub(super) struct WorkItem {
    pub session_id: SessionId,
    pub assignment: Assignment,
    pub input: InputFrame,
    pub deadline: Instant,
}

pub(super) struct Batch {
    pub items: Vec<WorkItem>,
}

pub(super) struct BatchResult {
    pub batch: Batch,
    pub started_at: Instant,
    pub completed_at: Instant,
}

pub(super) async fn run(
    mut batches: mpsc::Receiver<Batch>,
    results: mpsc::Sender<BatchResult>,
    latency: Duration,
    slowdown: Option<WorkerSlowdown>,
    epoch: Instant,
) {
    while let Some(batch) = batches.recv().await {
        let started_at = Instant::now();
        let duration = match &slowdown {
            Some(slowdown) if started_at.duration_since(epoch) >= slowdown.after => {
                slowdown.inference_latency
            }
            _ => latency,
        };
        tokio::time::sleep(duration).await;
        let result = BatchResult {
            batch,
            started_at,
            completed_at: Instant::now(),
        };
        if results.send(result).await.is_err() {
            break;
        }
    }
}
