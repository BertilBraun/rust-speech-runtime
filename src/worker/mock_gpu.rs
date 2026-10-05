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
}
pub(super) enum DeviceWork {
    Probe,
    Inference(Vec<WorkItem>),
}
pub(super) struct DeviceJob {
    pub work: DeviceWork,
    pub latency: Duration,
}
pub(super) struct DeviceResult {
    pub work: DeviceWork,
    pub started_at: Instant,
    pub completed_at: Instant,
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
            })
            .is_err()
        {
            break;
        }
    }
}
