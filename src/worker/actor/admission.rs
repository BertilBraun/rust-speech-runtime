use super::Worker;
use crate::{
    protocol::{FrameRejection, SessionId},
    scheduler::admission::{WorkerStatus, session_limit},
};
use std::{collections::HashSet, time::Duration};
use tokio::time::Instant;

impl Worker {
    pub(super) fn base_latency(&self) -> Duration {
        match &self.configuration.slowdown {
            Some(slowdown) if Instant::now().duration_since(self.epoch) >= slowdown.after => {
                slowdown.inference_latency
            }
            _ => self.configuration.inference_latency,
        }
    }
    pub(super) fn publish(&self) {
        let sessions: HashSet<SessionId> = self.sessions.keys().copied().collect();
        self.status.send_replace(WorkerStatus {
            worker_id: self.measurements.worker_id,
            sessions,
            session_limit: self.admission_limit(),
            service_time: self.projected_service_time(),
        });
    }
    pub(super) fn projected_device_time(&self) -> Duration {
        self.estimator.device_time().max(self.base_latency())
    }
    pub(super) fn projected_service_time(&self) -> Duration {
        self.estimator.service_time().max(
            self.base_latency()
                .mul_f64(self.configuration.latency_safety_factor),
        )
    }
    pub(super) fn input_has_time_for_compute(&self, deadline: Instant, replay: Duration) -> bool {
        Instant::now()
            + self.projected_device_time()
            + replay
            + self.configuration.scheduling_margin
            <= deadline + self.configuration.packet_lateness_grace
    }
    pub(super) fn recovery_deadline(&self, deadline: Instant) -> Instant {
        deadline
            + (self.configuration.packet_recovery_budget() - self.configuration.packet_deadline)
    }
    pub(super) fn input_has_time_for_recovery(&self, deadline: Instant) -> bool {
        self.projected_device_time() <= self.configuration.compute_budget()
            && Instant::now() + self.projected_device_time() + self.configuration.scheduling_margin
                <= self.recovery_deadline(deadline)
    }
    pub(super) fn admission_limit(&self) -> usize {
        let now = Instant::now();
        let mut queued = Duration::ZERO;
        let mut ready_count: usize = 0;
        let mut replay = Duration::ZERO;
        for pending in self
            .sessions
            .values()
            .filter_map(|session| session.work.ready())
        {
            queued = queued.max(now.duration_since(pending.queued_at));
            ready_count += 1;
            replay += pending.replay_duration;
        }
        let device_queue = self.submitted.back().map_or(Duration::ZERO, |batch| {
            batch.expected_completion.saturating_duration_since(now)
        });
        let ready_compute = self
            .projected_device_time()
            .mul_f64(ready_count.div_ceil(self.configuration.batch_size) as f64)
            + replay;
        self.estimator.available_limit(
            &self.configuration,
            self.session_limit,
            self.sessions.len(),
            queued.max(device_queue + ready_compute),
            self.base_latency(),
        )
    }
    pub(super) fn refresh_capacity(&mut self) {
        self.session_limit = session_limit(&self.configuration, self.projected_service_time());
        let mut sessions: Vec<SessionId> = self.sessions.keys().copied().collect();
        sessions.sort_unstable();
        while self.sessions.len() > self.session_limit {
            self.measurements.counters.capacity_terminations += 1;
            self.terminate(
                sessions.pop().expect("excess sessions exist"),
                FrameRejection::WorkerCapacityLost,
            );
        }
        self.publish();
    }
}
