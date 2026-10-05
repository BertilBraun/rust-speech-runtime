use super::{LatencyDistribution, LatencyHistogram};
use crate::protocol::{PacketSequence, SessionId};
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tokio::{task::JoinHandle, time::Instant};
use tokio_util::sync::CancellationToken;

pub(crate) const RETAINED_TRACES: usize = 8;

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct PacketTimings {
    pub ingress: Duration,
    pub worker_mailbox: Duration,
    pub validation: Duration,
    pub scheduler_queue: Duration,
    pub device_queue: Duration,
    pub device_execution: Duration,
    pub host_completion_delay: Duration,
    pub result_delivery: Duration,
    pub gateway_return: Duration,
}
impl PacketTimings {
    pub fn total(self) -> Duration {
        self.ingress
            + self.worker_mailbox
            + self.validation
            + self.scheduler_queue
            + self.device_queue
            + self.device_execution
            + self.host_completion_delay
            + self.result_delivery
            + self.gateway_return
    }
}

#[derive(Debug, Serialize)]
pub struct SlowWorkerPacket {
    pub session_id: SessionId,
    pub sequence: PacketSequence,
    pub elapsed_secs: f64,
    pub deadline_exceeded: bool,
    pub timings: PacketTimings,
}

#[derive(Default)]
pub(crate) struct WorkerProfile {
    pub mailbox: LatencyHistogram,
    pub validation: LatencyHistogram,
    pub scheduler_queue: LatencyHistogram,
    pub result_delivery: LatencyHistogram,
    pub host_completion_delay: LatencyHistogram,
    pub rejected_queue_delay: LatencyHistogram,
    pub device_queue: LatencyHistogram,
    pub batch_preparation: LatencyHistogram,
    pub batch_assembly: LatencyHistogram,
    pub host_device_wakeup: LatencyHistogram,
    pub slowest_packets: Vec<SlowWorkerPacket>,
}
impl WorkerProfile {
    pub(crate) fn merge(&mut self, other: &Self) {
        self.mailbox.merge(&other.mailbox);
        self.validation.merge(&other.validation);
        self.scheduler_queue.merge(&other.scheduler_queue);
        self.result_delivery.merge(&other.result_delivery);
        self.host_completion_delay
            .merge(&other.host_completion_delay);
        self.rejected_queue_delay.merge(&other.rejected_queue_delay);
        self.device_queue.merge(&other.device_queue);
        self.batch_preparation.merge(&other.batch_preparation);
        self.batch_assembly.merge(&other.batch_assembly);
        self.host_device_wakeup.merge(&other.host_device_wakeup);
    }
    pub(crate) fn record(&mut self, packet: SlowWorkerPacket) {
        self.scheduler_queue.record(packet.timings.scheduler_queue);
        self.result_delivery.record(packet.timings.result_delivery);
        self.device_queue.record(packet.timings.device_queue);
        self.slowest_packets.push(packet);
        self.slowest_packets
            .sort_unstable_by_key(|packet| std::cmp::Reverse(packet.timings.total()));
        self.slowest_packets.truncate(RETAINED_TRACES);
    }
    pub(crate) fn report(&self) -> WorkerProfileReport {
        WorkerProfileReport {
            worker_mailbox: self.mailbox.summary(),
            input_validation: self.validation.summary(),
            scheduler_queue: self.scheduler_queue.summary(),
            result_delivery: self.result_delivery.summary(),
            host_completion_delay: self.host_completion_delay.summary(),
            rejected_queue_delay: self.rejected_queue_delay.summary(),
            device_queue: self.device_queue.summary(),
            batch_preparation: self.batch_preparation.summary(),
            batch_assembly: self.batch_assembly.summary(),
            host_device_wakeup: self.host_device_wakeup.summary(),
        }
    }
}

#[derive(Debug, Serialize)]
pub struct WorkerProfileReport {
    pub worker_mailbox: LatencyDistribution,
    pub input_validation: LatencyDistribution,
    pub scheduler_queue: LatencyDistribution,
    pub result_delivery: LatencyDistribution,
    pub host_completion_delay: LatencyDistribution,
    pub rejected_queue_delay: LatencyDistribution,
    pub device_queue: LatencyDistribution,
    pub batch_preparation: LatencyDistribution,
    pub batch_assembly: LatencyDistribution,
    pub host_device_wakeup: LatencyDistribution,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RuntimeLagSample {
    pub elapsed_secs: f64,
    pub delay: Duration,
}
#[derive(Debug, Serialize, Deserialize)]
pub struct RuntimeLagReport {
    pub timer_delay: LatencyDistribution,
    pub slowest_wakeups: Vec<RuntimeLagSample>,
}
pub(crate) fn monitor_runtime(cancellation: CancellationToken) -> JoinHandle<RuntimeLagReport> {
    tokio::spawn(async move {
        let epoch = Instant::now();
        let mut delays = LatencyHistogram::default();
        let mut slowest = Vec::new();
        loop {
            let expected = Instant::now() + Duration::from_millis(5);
            tokio::select! {
                _ = cancellation.cancelled() => break,
                _ = tokio::time::sleep_until(expected) => {}
            }
            let now = Instant::now();
            let delay = now.saturating_duration_since(expected);
            delays.record(delay);
            slowest.push(RuntimeLagSample {
                elapsed_secs: now.duration_since(epoch).as_secs_f64(),
                delay,
            });
            slowest.sort_unstable_by_key(|sample| std::cmp::Reverse(sample.delay));
            slowest.truncate(RETAINED_TRACES);
        }
        RuntimeLagReport {
            timer_delay: delays.summary(),
            slowest_wakeups: slowest,
        }
    })
}

#[cfg(test)]
mod tests {
    use super::{PacketTimings, RETAINED_TRACES, SlowWorkerPacket, WorkerProfile};
    use crate::protocol::{PacketSequence, SessionId};
    use std::time::Duration;

    #[test]
    fn slow_packet_storage_is_bounded_and_preserves_worst_samples() {
        let mut profile = WorkerProfile::default();
        for index in 0..32 {
            profile.record(SlowWorkerPacket {
                session_id: SessionId(1),
                sequence: PacketSequence(index),
                elapsed_secs: index as f64,
                deadline_exceeded: false,
                timings: PacketTimings {
                    scheduler_queue: Duration::from_millis(index),
                    ..PacketTimings::default()
                },
            });
        }
        assert_eq!(profile.slowest_packets.len(), RETAINED_TRACES);
        assert_eq!(profile.slowest_packets[0].sequence, PacketSequence(31));
        assert_eq!(profile.slowest_packets[7].sequence, PacketSequence(24));
        assert_eq!(profile.report().scheduler_queue.samples, 32);
    }
}
