use super::{SubmittedBatch, Worker};
use crate::{
    metrics::profile::{PacketTimings, SlowWorkerPacket},
    protocol::{
        AudioResult, CacheOutcome, FrameRejection, InferenceOutput, InputOutcome, SessionLease,
    },
    scheduler::deadline::{ReadySession, construct_batch},
    worker::{
        mock_gpu::{DeviceJob, DeviceResult, DeviceWork, WorkItem},
        session::SessionWork,
    },
};
use std::time::Duration;
use tokio::{sync::mpsc, time::Instant};

struct BatchCompletion {
    submitted_at: Instant,
    started_at: Instant,
    observed_at: Instant,
    inference_latency: Duration,
    host_delay: Duration,
}

impl Worker {
    pub(super) fn refresh_prepared(&mut self) {
        let started = Instant::now();
        self.prepared = construct_batch(
            self.sessions
                .iter()
                .filter_map(|(session_id, session)| {
                    session.work.ready().map(|pending| ReadySession {
                        session_id: *session_id,
                        deadline: pending.deadline,
                    })
                })
                .collect(),
            self.configuration.batch_size,
        );
        if self
            .submitted
            .back()
            .is_some_and(|batch| batch.expected_completion > started)
            && !self.prepared.is_empty()
        {
            self.measurements.counters.prepared_while_running += 1;
        }
        self.measurements
            .profile
            .batch_preparation
            .record(started.elapsed());
    }
    pub(super) fn track_submission(&mut self, submitted_at: Instant, latency: Duration) {
        let start = self.submitted.back().map_or(submitted_at, |batch| {
            batch.expected_completion.max(submitted_at)
        });
        self.submitted.push_back(SubmittedBatch {
            submitted_at,
            latency,
            expected_completion: start + latency,
        });
        self.measurements.peak_device_jobs =
            self.measurements.peak_device_jobs.max(self.submitted.len());
    }
    pub(super) fn next_wakeup(&self) -> Instant {
        let now = Instant::now();
        if self.prepared.is_empty()
            || self.submitted.len() > self.configuration.device_queue_capacity
        {
            return now + self.configuration.session_timeout;
        }
        let completion_pending = self
            .submitted
            .front()
            .is_some_and(|batch| batch.expected_completion <= now);
        if self.prepared.len() == self.configuration.batch_size
            || (!completion_pending
                && self
                    .sessions
                    .values()
                    .all(|session| !matches!(session.work, SessionWork::Idle)))
        {
            return now;
        }
        if let Some(previous) = self
            .submitted
            .back()
            .filter(|batch| batch.expected_completion > now)
        {
            return previous.expected_completion - self.configuration.launch_ahead;
        }
        let mut count = 0;
        let mut wakeup = now + self.configuration.session_timeout;
        for pending in self
            .sessions
            .values()
            .filter_map(|session| session.work.ready())
        {
            count += 1;
            let latest = pending.deadline
                - self.projected_device_time()
                - pending.replay_duration
                - self.configuration.scheduling_margin;
            wakeup =
                wakeup.min((pending.queued_at + self.configuration.max_batch_wait).min(latest));
        }
        if count >= self.configuration.batch_size || (count > 0 && count == self.sessions.len()) {
            now
        } else {
            wakeup
        }
    }
    pub(super) fn idle_collection_expired(&self) -> bool {
        let now = Instant::now();
        self.sessions
            .values()
            .filter_map(|session| session.work.ready())
            .any(|pending| {
                now >= pending.queued_at + self.configuration.max_batch_wait
                    || now
                        + self.projected_device_time()
                        + pending.replay_duration
                        + self.configuration.scheduling_margin
                        >= pending.deadline
            })
    }
    pub(super) fn dispatch_batch(&mut self, permit: mpsc::Permit<'_, DeviceJob>) {
        let assembly_started = Instant::now();
        let device_time = self.projected_device_time();
        let selected = std::mem::take(&mut self.prepared);
        let predicted_start = self.submitted.back().map_or(Instant::now(), |batch| {
            batch.expected_completion.max(Instant::now())
        });
        let mut items = Vec::new();
        let mut replay = Duration::ZERO;
        let mut earliest_deadline = None;
        for selected in selected {
            let session = self
                .sessions
                .get_mut(&selected.session_id)
                .expect("selected session exists");
            let pending = session.work.ready().expect("pending frame exists");
            let projected = device_time
                + replay
                + pending.replay_duration
                + self.configuration.scheduling_margin;
            let completion_deadline = pending.deadline
                + match pending.cache {
                    CacheOutcome::Hit => {
                        self.configuration.packet_recovery_budget()
                            - self.configuration.packet_deadline
                    }
                    CacheOutcome::Replayed { .. } => self.configuration.packet_lateness_grace,
                };
            let batch_deadline = earliest_deadline
                .unwrap_or(completion_deadline)
                .min(completion_deadline);
            let own_cost =
                device_time + pending.replay_duration + self.configuration.scheduling_margin;
            if predicted_start + own_cost > completion_deadline {
                let pending = session.work.take_ready();
                self.measurements
                    .profile
                    .rejected_queue_delay
                    .record(pending.timestamp.elapsed());
                self.reject(
                    selected.session_id,
                    pending.reply,
                    if pending.replay_duration.is_zero() {
                        FrameRejection::DeadlineExceeded
                    } else {
                        FrameRejection::ReplayTooExpensive
                    },
                );
                continue;
            }
            if predicted_start + projected > batch_deadline {
                continue;
            }
            let pending = session.work.take_ready();
            session.work = SessionWork::Submitted;
            replay += pending.replay_duration;
            earliest_deadline = Some(batch_deadline);
            items.push(pending);
        }
        if items.is_empty() {
            self.refresh_prepared();
            return;
        }
        if self
            .submitted
            .back()
            .is_some_and(|previous| previous.expected_completion > Instant::now())
        {
            self.measurements.counters.queued_batch_launches += 1;
        }
        let submitted_at = Instant::now();
        let latency = self.base_latency() + replay;
        self.measurements
            .profile
            .batch_assembly
            .record(assembly_started.elapsed());
        permit.send(DeviceJob {
            latency,
            work: DeviceWork::Inference(items),
            submitted_at,
        });
        self.track_submission(submitted_at, latency);
        self.refresh_prepared();
        self.publish();
    }
    pub(super) fn complete(&mut self, result: DeviceResult) {
        self.submitted
            .pop_front()
            .expect("completion matches submitted work");
        let mut expected_completion = result.completed_at;
        for batch in &mut self.submitted {
            expected_completion = expected_completion.max(batch.submitted_at) + batch.latency;
            batch.expected_completion = expected_completion;
        }
        let inference_latency = result.completed_at.duration_since(result.started_at);
        let host_wait = result.observed_at.duration_since(result.completed_at);
        self.estimator
            .observe_host_delay(Instant::now() - result.completed_at);
        self.measurements
            .profile
            .host_completion_delay
            .record(host_wait);
        self.measurements.profile.host_device_wakeup.record(
            result
                .host_started_at
                .saturating_duration_since(result.started_at),
        );
        let completion = BatchCompletion {
            submitted_at: result.submitted_at,
            started_at: result.started_at,
            observed_at: result.observed_at,
            inference_latency,
            host_delay: host_wait,
        };
        match result.work {
            DeviceWork::Probe => {
                self.estimator.observe(inference_latency);
                self.refresh_capacity();
            }
            DeviceWork::Inference(items) => {
                if result.processed_frames > 0 {
                    self.measurements.counters.batches += 1;
                    self.measurements.counters.processed_frames += result.processed_frames as u64;
                    self.measurements
                        .batch_sizes
                        .record(result.processed_frames as u64)
                        .expect("bounded batch size");
                    self.measurements
                        .inference_latency
                        .record(inference_latency);
                    self.estimator
                        .observe(inference_latency - result.replay_duration);
                }
                self.measurements.busy_time += inference_latency;
                let occupied_start = self
                    .measurements
                    .occupied_until
                    .map_or(result.started_at, |previous| {
                        previous.max(result.started_at)
                    });
                self.measurements.occupied_time +=
                    result.observed_at.saturating_duration_since(occupied_start);
                self.measurements.occupied_until = Some(result.observed_at);
                for item in items {
                    self.complete_frame(item, &completion);
                }
                self.refresh_capacity();
            }
        }
    }
    fn complete_frame(&mut self, item: WorkItem, completion: &BatchCompletion) {
        if let Some(cache) = self.retired_cache.remove(&SessionLease {
            session_id: item.session_id,
            generation: item.assignment.generation,
        }) {
            self.cache.free(cache);
        }
        let handled_at = Instant::now();
        let timings = PacketTimings {
            ingress: item.routed_at - item.timestamp,
            worker_mailbox: item.received_at - item.routed_at,
            validation: item.queued_at - item.received_at,
            scheduler_queue: completion.submitted_at - item.queued_at,
            device_queue: completion.started_at - completion.submitted_at,
            device_execution: completion.inference_latency,
            host_completion_delay: completion.host_delay,
            result_delivery: handled_at - completion.observed_at,
            gateway_return: Duration::ZERO,
        };
        self.measurements.profile.record(SlowWorkerPacket {
            session_id: item.session_id,
            sequence: item.sequence,
            elapsed_secs: handled_at.duration_since(self.epoch).as_secs_f64(),
            deadline_exceeded: handled_at > item.deadline,
            timings,
        });
        let Some(session) = self
            .sessions
            .get_mut(&item.session_id)
            .filter(|session| session.assignment == item.assignment)
        else {
            self.measurements.counters.stale_results += 1;
            self.measurements.counters.rejected_frames += 1;
            let _ = item
                .reply
                .send(InputOutcome::Rejected(FrameRejection::Cancelled));
            return;
        };
        session.work = SessionWork::Idle;
        self.measurements
            .queue_delay
            .record(completion.started_at.duration_since(item.timestamp));
        self.measurements
            .end_to_end_latency
            .record(Instant::now().duration_since(item.timestamp));
        if Instant::now() > item.deadline {
            self.measurements.counters.deadline_misses += 1;
            self.measurements
                .deadline_lateness
                .record(Instant::now().duration_since(item.deadline));
        }
        if Instant::now()
            > item.deadline
                + (self.configuration.packet_recovery_budget() - self.configuration.packet_deadline)
        {
            self.reject(
                item.session_id,
                item.reply,
                FrameRejection::DeadlineExceeded,
            );
            return;
        }
        session.prefix = item.prefix;
        self.cache.update(session.cache, item.prefix);
        match item.cache {
            CacheOutcome::Hit => self.measurements.counters.cache_hits += 1,
            CacheOutcome::Replayed { packets, bytes } => {
                self.measurements.counters.replayed_packets += packets;
                self.measurements.counters.replayed_bytes += bytes;
            }
        }
        let output = InferenceOutput {
            session_id: item.session_id,
            input_timestamp: item.timestamp,
            completed_at: completion.observed_at,
            audio: AudioResult {
                assignment: item.assignment,
                sequence: item.sequence,
                payload: item.payload,
                prefix: item.prefix,
                cache: item.cache,
                timings: Box::new(timings),
            },
        };
        if item.reply.send(InputOutcome::Processed(output)).is_ok() {
            self.measurements.counters.delivered_frames += 1;
        } else {
            self.terminate(item.session_id, FrameRejection::Cancelled);
        }
    }
}
