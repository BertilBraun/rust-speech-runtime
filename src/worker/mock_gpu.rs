use crate::config::DeviceWait;
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
    pub observed_at: Instant,
    pub submitted_at: Instant,
}
pub(super) fn run(
    mut jobs: mpsc::Receiver<DeviceJob>,
    results: mpsc::Sender<DeviceResult>,
    wait: DeviceWait,
) {
    while let Some(job) = jobs.blocking_recv() {
        let started_at = Instant::now();
        let completed_at = started_at + job.latency;
        wait_until(completed_at, wait);
        if results
            .blocking_send(DeviceResult {
                work: job.work,
                started_at,
                completed_at,
                observed_at: Instant::now(),
                submitted_at: job.submitted_at,
            })
            .is_err()
        {
            break;
        }
    }
}

fn wait_until(deadline: Instant, strategy: DeviceWait) {
    match strategy {
        DeviceWait::Sleep => std::thread::sleep(deadline.saturating_duration_since(Instant::now())),
        DeviceWait::Hybrid { spin_tail } => {
            std::thread::sleep(
                deadline
                    .saturating_duration_since(Instant::now())
                    .saturating_sub(spin_tail),
            );
            while Instant::now() < deadline {
                std::hint::spin_loop();
            }
        }
        DeviceWait::Poll { sleep_interval } => {
            while Instant::now() < deadline {
                std::thread::sleep(
                    sleep_interval.min(deadline.saturating_duration_since(Instant::now())),
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{DeviceJob, DeviceWork, WorkItem, run};
    use crate::{
        config::DeviceWait,
        protocol::{
            Assignment, AudioContext, AudioPacket, CacheOutcome, Generation, InputFrame,
            PacketSequence, PrefixState, SessionId, WorkerId,
        },
    };
    use bytes::Bytes;
    use std::time::Duration;
    use tokio::{
        sync::{mpsc, oneshot},
        time::Instant,
    };

    fn item(session_id: u64) -> WorkItem {
        let now = Instant::now();
        let (reply, _) = oneshot::channel();
        WorkItem {
            session_id: SessionId(session_id),
            assignment: Assignment {
                worker_id: WorkerId(0),
                generation: Generation(1),
            },
            input: InputFrame {
                timestamp: now,
                deadline: now + Duration::from_secs(1),
                packet: AudioPacket {
                    sequence: PacketSequence(0),
                    payload: Bytes::from_static(b"audio"),
                    context: AudioContext::Cached(PrefixState::default()),
                },
            },
            prefix: PrefixState::default(),
            cache: CacheOutcome::Hit,
            replay_duration: Duration::ZERO,
            reply,
            routed_at: now,
            received_at: now,
            queued_at: now,
        }
    }

    #[tokio::test]
    async fn partial_and_full_batches_have_the_same_modeled_compute_cost() {
        let (jobs, mailbox) = mpsc::channel(1);
        let (results, mut responses) = mpsc::channel(1);
        let device = tokio::task::spawn_blocking(move || run(mailbox, results, DeviceWait::Sleep));
        for count in [1, 8, 16] {
            jobs.send(DeviceJob {
                work: DeviceWork::Inference((0..count).map(item).collect()),
                latency: Duration::from_millis(2),
                submitted_at: Instant::now(),
            })
            .await
            .unwrap();
            let result = responses.recv().await.unwrap();
            assert_eq!(
                result.completed_at - result.started_at,
                Duration::from_millis(2)
            );
            assert!(result.observed_at >= result.completed_at);
        }
        drop(jobs);
        device.await.unwrap();
    }

    #[tokio::test]
    async fn wait_strategies_preserve_the_requested_device_timeline() {
        for strategy in [
            DeviceWait::Sleep,
            DeviceWait::Hybrid {
                spin_tail: Duration::from_micros(200),
            },
            DeviceWait::Poll {
                sleep_interval: Duration::from_nanos(500),
            },
        ] {
            let (jobs, mailbox) = mpsc::channel(1);
            let (results, mut responses) = mpsc::channel(1);
            let device = tokio::task::spawn_blocking(move || run(mailbox, results, strategy));
            jobs.send(DeviceJob {
                work: DeviceWork::Probe,
                latency: Duration::from_millis(2),
                submitted_at: Instant::now(),
            })
            .await
            .unwrap();
            let result = responses.recv().await.unwrap();
            assert_eq!(
                result.completed_at - result.started_at,
                Duration::from_millis(2)
            );
            assert!(result.observed_at >= result.completed_at);
            drop(jobs);
            device.await.unwrap();
        }
    }
}
