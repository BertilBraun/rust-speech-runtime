use crate::config::{DeviceWait, ThreadWait};
use crate::metrics::cpu::{CpuUsage, DeviceCpuUsage};
use crate::protocol::{
    Assignment, CacheOutcome, InputOutcome, PacketSequence, PrefixState, SessionId,
};
use bytes::Bytes;
use cpu_time::ThreadTime;
use std::{
    sync::{Arc, OnceLock},
    time::Duration,
};
use tokio::{
    sync::{mpsc, oneshot},
    time::Instant,
};

#[derive(Clone, Default)]
pub(super) struct SessionCancellation(Arc<OnceLock<Instant>>);
impl SessionCancellation {
    pub(super) fn cancel(&self) {
        self.0
            .set(Instant::now())
            .expect("a session is removed once");
    }
    fn was_cancelled_before(&self, started_at: Instant) -> bool {
        self.0
            .get()
            .is_some_and(|cancelled_at| *cancelled_at <= started_at)
    }
}

pub(super) struct WorkItem {
    pub session_id: SessionId,
    pub assignment: Assignment,
    pub timestamp: Instant,
    pub deadline: Instant,
    pub sequence: PacketSequence,
    pub payload: Bytes,
    pub prefix: PrefixState,
    pub cache: CacheOutcome,
    pub replay_duration: Duration,
    pub reply: oneshot::Sender<InputOutcome>,
    pub routed_at: Instant,
    pub received_at: Instant,
    pub queued_at: Instant,
    pub cancellation: SessionCancellation,
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
    pub host_started_at: Instant,
    pub processed_frames: usize,
    pub replay_duration: Duration,
    pub submitted_at: Instant,
}
pub(super) fn spawn(
    jobs: mpsc::Receiver<DeviceJob>,
    results: mpsc::Sender<DeviceResult>,
    wait: DeviceWait,
) -> tokio::task::JoinHandle<DeviceCpuUsage> {
    match wait {
        DeviceWait::Tokio => tokio::spawn(async move {
            run(jobs, results, wait).await;
            DeviceCpuUsage::SharedRuntime
        }),
        DeviceWait::Thread(_) => {
            let runtime = tokio::runtime::Handle::current();
            tokio::task::spawn_blocking(move || {
                let cpu = ThreadTime::now();
                let wall = std::time::Instant::now();
                runtime.block_on(run(jobs, results, wait));
                DeviceCpuUsage::DedicatedThread(CpuUsage::measured(cpu.elapsed(), wall.elapsed()))
            })
        }
    }
}

async fn run(
    mut jobs: mpsc::Receiver<DeviceJob>,
    results: mpsc::Sender<DeviceResult>,
    wait: DeviceWait,
) {
    let mut previous_completion = Instant::now();
    while let Some(job) = jobs.recv().await {
        let host_started_at = Instant::now();
        let started_at = previous_completion.max(job.submitted_at);
        let (processed_frames, latency, replay_duration) = match &job.work {
            DeviceWork::Probe => (0, job.latency, Duration::ZERO),
            DeviceWork::Inference(items) => {
                let mut active = 0;
                let mut cancelled_replay = Duration::ZERO;
                let mut active_replay = Duration::ZERO;
                for item in items {
                    if item.cancellation.was_cancelled_before(started_at) {
                        cancelled_replay += item.replay_duration;
                    } else {
                        active += 1;
                        active_replay += item.replay_duration;
                    }
                }
                (
                    active,
                    if active == 0 {
                        Duration::ZERO
                    } else {
                        job.latency - cancelled_replay
                    },
                    active_replay,
                )
            }
        };
        let completed_at = started_at + latency;
        match wait {
            DeviceWait::Tokio => tokio::time::sleep_until(completed_at).await,
            DeviceWait::Thread(strategy) => wait_until(completed_at, strategy),
        }
        previous_completion = completed_at;
        if results
            .send(DeviceResult {
                work: job.work,
                started_at,
                completed_at,
                observed_at: Instant::now(),
                host_started_at,
                processed_frames,
                replay_duration,
                submitted_at: job.submitted_at,
            })
            .await
            .is_err()
        {
            break;
        }
    }
}

fn wait_until(deadline: Instant, strategy: ThreadWait) {
    let remaining = deadline.saturating_duration_since(Instant::now());
    if remaining.is_zero() {
        return;
    }
    match strategy {
        ThreadWait::Sleep => std::thread::sleep(remaining),
        ThreadWait::Hybrid { spin_tail } => {
            let sleeping = remaining.saturating_sub(spin_tail);
            if !sleeping.is_zero() {
                std::thread::sleep(sleeping);
            }
            while Instant::now() < deadline {
                std::hint::spin_loop();
            }
        }
        ThreadWait::Poll { sleep_interval } => loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                break;
            }
            std::thread::sleep(sleep_interval.min(remaining));
        },
    }
}

#[cfg(test)]
mod tests {
    use super::{DeviceJob, DeviceWork, WorkItem, spawn};
    use crate::{
        config::{DeviceWait, ThreadWait},
        protocol::{
            Assignment, CacheOutcome, Generation, PacketSequence, PrefixState, SessionId, WorkerId,
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
            timestamp: now,
            deadline: now + Duration::from_secs(1),
            sequence: PacketSequence(0),
            payload: Bytes::from_static(b"audio"),
            prefix: PrefixState::default(),
            cache: CacheOutcome::Hit,
            replay_duration: Duration::ZERO,
            reply,
            routed_at: now,
            received_at: now,
            queued_at: now,
            cancellation: super::SessionCancellation::default(),
        }
    }

    #[tokio::test]
    async fn partial_and_full_batches_have_the_same_modeled_compute_cost() {
        for strategy in [DeviceWait::Tokio, DeviceWait::Thread(ThreadWait::Sleep)] {
            let (jobs, mailbox) = mpsc::channel(1);
            let (results, mut responses) = mpsc::channel(1);
            let device = spawn(mailbox, results, strategy);
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
    }

    #[tokio::test]
    async fn async_device_completion_does_not_block_the_single_thread_tokio_runtime() {
        let (jobs, mailbox) = mpsc::channel(1);
        let (results, mut responses) = mpsc::channel(1);
        let device = spawn(mailbox, results, DeviceWait::Tokio);
        jobs.send(DeviceJob {
            work: DeviceWork::Inference(vec![item(1)]),
            latency: Duration::from_millis(100),
            submitted_at: Instant::now(),
        })
        .await
        .unwrap();
        tokio::spawn(async { tokio::task::yield_now().await })
            .await
            .unwrap();
        assert!(matches!(
            responses.try_recv(),
            Err(mpsc::error::TryRecvError::Empty)
        ));
        assert_eq!(responses.recv().await.unwrap().processed_frames, 1);
        drop(jobs);
        assert!(matches!(
            device.await.unwrap(),
            crate::metrics::cpu::DeviceCpuUsage::SharedRuntime
        ));
    }

    #[tokio::test]
    async fn wait_strategies_preserve_the_requested_device_timeline() {
        for strategy in [
            DeviceWait::Tokio,
            DeviceWait::Thread(ThreadWait::Sleep),
            DeviceWait::Thread(ThreadWait::Hybrid {
                spin_tail: Duration::from_micros(200),
            }),
            DeviceWait::Thread(ThreadWait::Poll {
                sleep_interval: Duration::from_nanos(500),
            }),
        ] {
            let (jobs, mailbox) = mpsc::channel(1);
            let (results, mut responses) = mpsc::channel(1);
            let device = spawn(mailbox, results, strategy);
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

    #[tokio::test]
    async fn prequeued_batches_execute_serially_without_waiting_for_host_result_handling() {
        let (jobs, mailbox) = mpsc::channel(3);
        let (results, mut responses) = mpsc::channel(3);
        let device = spawn(mailbox, results, DeviceWait::Thread(ThreadWait::Sleep));
        for session_id in 0..3 {
            jobs.send(DeviceJob {
                work: DeviceWork::Inference(vec![item(session_id)]),
                latency: Duration::from_millis(10),
                submitted_at: Instant::now(),
            })
            .await
            .unwrap();
        }
        tokio::time::sleep(Duration::from_millis(40)).await;
        let first = responses.recv().await.unwrap();
        let second = responses.recv().await.unwrap();
        let third = responses.recv().await.unwrap();
        assert_eq!(second.started_at, first.completed_at);
        assert_eq!(third.started_at, second.completed_at);
        assert_eq!(
            third.completed_at - first.started_at,
            Duration::from_millis(30)
        );
        drop(jobs);
        device.await.unwrap();
    }

    #[tokio::test]
    async fn cancellation_before_device_start_skips_compute_without_losing_the_result() {
        let (jobs, mailbox) = mpsc::channel(2);
        let (results, mut responses) = mpsc::channel(2);
        let device = spawn(mailbox, results, DeviceWait::Thread(ThreadWait::Sleep));
        jobs.send(DeviceJob {
            work: DeviceWork::Inference(vec![item(1)]),
            latency: Duration::from_millis(10),
            submitted_at: Instant::now(),
        })
        .await
        .unwrap();
        let cancelled = item(2);
        cancelled.cancellation.cancel();
        jobs.send(DeviceJob {
            work: DeviceWork::Inference(vec![cancelled]),
            latency: Duration::from_millis(10),
            submitted_at: Instant::now(),
        })
        .await
        .unwrap();
        let first = responses.recv().await.unwrap();
        let second = responses.recv().await.unwrap();
        assert_eq!(first.processed_frames, 1);
        assert_eq!(second.processed_frames, 0);
        assert_eq!(second.completed_at, second.started_at);
        drop(jobs);
        device.await.unwrap();
    }
}
