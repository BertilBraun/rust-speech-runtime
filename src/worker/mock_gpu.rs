use crate::protocol::{Assignment, CacheOutcome, InputFrame, InputOutcome, PrefixState, SessionId};
use std::time::Duration;
use tokio::{
    sync::{mpsc, oneshot},
    time::Instant,
};

pub(super) struct WorkItem {
    pub session_id: SessionId,
    pub assignment: Assignment,
    pub input: InputFrame,
    pub prefix: PrefixState,
    pub cache: CacheOutcome,
    pub replay_duration: Duration,
    pub reply: oneshot::Sender<InputOutcome>,
    pub routed_at: Instant,
    pub received_at: Instant,
    pub queued_at: Instant,
}
pub(super) enum DeviceWork {
    Probe,
    Inference(Vec<WorkItem>),
}
pub(super) struct DeviceJob {
    pub work: DeviceWork,
    pub latency: Duration,
    pub submitted_at: Instant,
}
pub(super) struct DeviceResult {
    pub work: DeviceWork,
    pub started_at: Instant,
    pub completed_at: Instant,
    pub submitted_at: Instant,
    pub requested_latency: Duration,
}
pub(super) fn run(mut jobs: mpsc::Receiver<DeviceJob>, results: mpsc::Sender<DeviceResult>) {
    while let Some(job) = jobs.blocking_recv() {
        let started_at = Instant::now();
        std::thread::sleep(job.latency);
        if results
            .blocking_send(DeviceResult {
                work: job.work,
                started_at,
                completed_at: Instant::now(),
                submitted_at: job.submitted_at,
                requested_latency: job.latency,
            })
            .is_err()
        {
            break;
        }
    }
}
