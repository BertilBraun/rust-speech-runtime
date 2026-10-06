use super::Worker;
use crate::{
    protocol::{
        AudioContext, AudioPacket, CacheOutcome, FrameRejection, Generation, InputFrame,
        InputOutcome, SessionId,
    },
    worker::{mock_gpu::WorkItem, session::SessionWork},
};
use std::time::Duration;
use tokio::{sync::oneshot, time::Instant};

impl Worker {
    pub(super) async fn queue_input(
        &mut self,
        session_id: SessionId,
        generation: Generation,
        input: InputFrame,
        reply: oneshot::Sender<InputOutcome>,
        routed_at: Instant,
    ) {
        let InputFrame {
            timestamp,
            deadline,
            packet,
        } = input;
        let AudioPacket {
            sequence,
            payload,
            context,
        } = packet;
        let received_at = Instant::now();
        self.measurements
            .profile
            .mailbox
            .record(received_at - routed_at);
        let Some(session) = self.sessions.get(&session_id) else {
            let _ = reply.send(InputOutcome::Rejected(FrameRejection::UnknownSession));
            return;
        };
        if session.assignment.generation != generation {
            let _ = reply.send(InputOutcome::Rejected(FrameRejection::Cancelled));
            return;
        }
        if !matches!(session.work, SessionWork::Idle) {
            self.reject(session_id, reply, FrameRejection::Overloaded);
            return;
        }
        if self.recovery_deadline(deadline) <= Instant::now() {
            self.reject(session_id, reply, FrameRejection::DeadlineExceeded);
            return;
        }
        if sequence.0 != session.prefix.packets {
            self.reject(session_id, reply, FrameRejection::InvalidSequence);
            return;
        }
        let prior_prefix = session.prefix;
        if prior_prefix.packets >= self.configuration.audio_limits.max_prefix_packets as u64
            || prior_prefix.bytes + payload.len() as u64
                > self.configuration.audio_limits.max_prefix_bytes as u64
        {
            self.reject(session_id, reply, FrameRejection::PrefixCapacity);
            return;
        }
        let replay_cost = self
            .configuration
            .replay_latency_per_packet
            .mul_f64(prior_prefix.packets as f64);
        let recovery_rejection = if replay_cost.is_zero() {
            FrameRejection::DeadlineExceeded
        } else {
            FrameRejection::ReplayTooExpensive
        };
        let cache = match context {
            AudioContext::Cached(expected) => {
                if expected != prior_prefix {
                    self.reject(session_id, reply, FrameRejection::InvalidPrefix);
                    return;
                }
                if self.cache.lookup(session.cache) != Some(prior_prefix) {
                    self.measurements.counters.cache_misses += 1;
                    if !self.input_has_time_for_compute(deadline, replay_cost) {
                        self.reject(session_id, reply, recovery_rejection);
                        return;
                    }
                    let _ = reply.send(InputOutcome::CacheMiss);
                    return;
                }
                CacheOutcome::Hit
            }
            AudioContext::Replay(prefix) => {
                if !self.input_has_time_for_compute(deadline, replay_cost) {
                    self.reject(session_id, reply, recovery_rejection);
                    return;
                }
                if prefix.0.len() > self.configuration.audio_limits.max_prefix_packets
                    || prefix.byte_len() > self.configuration.audio_limits.max_prefix_bytes
                    || prefix
                        .0
                        .iter()
                        .any(|frame| frame.len() > self.configuration.audio_limits.max_frame_bytes)
                {
                    self.reject(session_id, reply, FrameRejection::PrefixCapacity);
                    return;
                }
                let state = tokio::task::spawn_blocking(move || prefix.state())
                    .await
                    .expect("prefix hashing does not panic");
                if state != prior_prefix {
                    self.reject(session_id, reply, FrameRejection::InvalidPrefix);
                    return;
                }
                CacheOutcome::Replayed {
                    packets: state.packets,
                    bytes: state.bytes,
                }
            }
        };
        let replay_duration = match cache {
            CacheOutcome::Hit => Duration::ZERO,
            CacheOutcome::Replayed { .. } => replay_cost,
        };
        let has_time = match cache {
            CacheOutcome::Hit => self.input_has_time_for_recovery(deadline),
            CacheOutcome::Replayed { .. } => {
                self.input_has_time_for_compute(deadline, replay_duration)
            }
        };
        if !has_time {
            self.reject(
                session_id,
                reply,
                if replay_duration.is_zero() {
                    FrameRejection::DeadlineExceeded
                } else {
                    FrameRejection::ReplayTooExpensive
                },
            );
            return;
        }
        let prefix = prior_prefix.append(&payload);
        let queued_at = Instant::now();
        self.measurements
            .profile
            .validation
            .record(queued_at - received_at);
        let session = self
            .sessions
            .get_mut(&session_id)
            .expect("session remains owned during validation");
        session.work = SessionWork::Ready(WorkItem {
            session_id,
            assignment: session.assignment,
            timestamp,
            deadline,
            sequence,
            payload,
            prefix,
            cache,
            replay_duration,
            reply,
            routed_at,
            received_at,
            queued_at,
            cancellation: session.cancellation.clone(),
        });
        self.refresh_prepared();
        self.publish();
    }
}
