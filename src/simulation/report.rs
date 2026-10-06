use super::{ClientPacketTrace, SimulationConfig};
use crate::{
    metrics::{LatencyDistribution, profile::RuntimeLagReport},
    protocol::{CreateRejection, FrameRejection},
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum FailureReason {
    DeadlineExceeded,
    ServerRejected(FrameRejection),
    Connection,
    PrefixCapacity,
    InvalidEcho,
    Protocol,
    GeneratorOverrun,
    DeadlineMissBurst,
}
#[derive(Debug, Serialize, Deserialize)]
pub struct FailureCount {
    pub reason: FailureReason,
    pub count: u64,
}
#[derive(Default, Debug, Serialize, Deserialize)]
pub struct ClientCounters {
    pub admitted_sessions: u64,
    pub rejected_capacity: u64,
    pub rejected_duplicate: u64,
    pub closed_sessions: u64,
    pub failed_sessions: u64,
    pub attempted_frames: u64,
    pub echoed_frames: u64,
    pub late_frames: u64,
    pub discarded_output_frames: u64,
    pub unrecovered_deadline_frames: u64,
    pub quality_failed_sessions: u64,
    pub maximum_consecutive_misses: u64,
    pub maximum_window_misses: u64,
    pub cache_hits: u64,
    pub replayed_frames: u64,
    pub replayed_packets: u64,
    pub replayed_bytes: u64,
    pub metric_samples_dropped: u64,
    pub failures: Vec<FailureCount>,
}
impl ClientCounters {
    pub(super) fn failure(&mut self, reason: FailureReason) {
        self.failed_sessions += 1;
        if matches!(
            reason,
            FailureReason::DeadlineExceeded
                | FailureReason::ServerRejected(FrameRejection::DeadlineExceeded)
        ) {
            self.unrecovered_deadline_frames += 1;
        }
        if let Some(entry) = self
            .failures
            .iter_mut()
            .find(|entry| entry.reason == reason)
        {
            entry.count += 1;
        } else {
            self.failures.push(FailureCount { reason, count: 1 });
        }
    }
    pub(super) fn merge(&mut self, other: Self) {
        self.admitted_sessions += other.admitted_sessions;
        self.rejected_capacity += other.rejected_capacity;
        self.rejected_duplicate += other.rejected_duplicate;
        self.closed_sessions += other.closed_sessions;
        self.failed_sessions += other.failed_sessions;
        self.attempted_frames += other.attempted_frames;
        self.echoed_frames += other.echoed_frames;
        self.late_frames += other.late_frames;
        self.discarded_output_frames += other.discarded_output_frames;
        self.unrecovered_deadline_frames += other.unrecovered_deadline_frames;
        self.quality_failed_sessions += other.quality_failed_sessions;
        self.maximum_consecutive_misses = self
            .maximum_consecutive_misses
            .max(other.maximum_consecutive_misses);
        self.maximum_window_misses = self.maximum_window_misses.max(other.maximum_window_misses);
        self.cache_hits += other.cache_hits;
        self.replayed_frames += other.replayed_frames;
        self.replayed_packets += other.replayed_packets;
        self.replayed_bytes += other.replayed_bytes;
        self.metric_samples_dropped += other.metric_samples_dropped;
        for failure in other.failures {
            if let Some(entry) = self
                .failures
                .iter_mut()
                .find(|entry| entry.reason == failure.reason)
            {
                entry.count += failure.count;
            } else {
                self.failures.push(failure);
            }
        }
    }
    pub(super) fn admission(&mut self, reason: CreateRejection) {
        match reason {
            CreateRejection::Capacity => self.rejected_capacity += 1,
            CreateRejection::AlreadyExists => self.rejected_duplicate += 1,
        }
    }
}
#[derive(Debug, Serialize, Deserialize)]
pub struct SimulationReport {
    pub process_cpu: crate::metrics::cpu::CpuUsage,
    pub configuration: SimulationConfig,
    pub elapsed_secs: f64,
    pub admission_secs: f64,
    pub setup_secs: f64,
    pub counters: ClientCounters,
    pub throughput_frames_per_sec: f64,
    pub round_trip_latency: LatencyDistribution,
    pub failed_round_trip_latency: LatencyDistribution,
    pub deadline_lateness: LatencyDistribution,
    pub generator_delay: LatencyDistribution,
    pub generated_intervals: LatencyDistribution,
    pub generator_overruns: u64,
    pub client_start_delay: LatencyDistribution,
    pub outside_server_latency: LatencyDistribution,
    pub runtime_lag: RuntimeLagReport,
    pub slowest_packets: Vec<ClientPacketTrace>,
}
