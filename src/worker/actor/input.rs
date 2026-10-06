use super::Worker;
use crate::{
    protocol::{
        AudioContext, AudioPacket, AudioPrefix, CacheOutcome, FrameRejection, Generation,
        InputFrame, InputOutcome, PrefixState, SessionId,
    },
    worker::{
        mock_gpu::WorkItem,
        session::{SessionWork, WorkerSession},
    },
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
        if let Err(reason) = self.validate_input(session, &input) {
            self.reject(session_id, reply, reason);
            return;
        }
        let prior_prefix = session.prefix;
        let cache_handle = session.cache;
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
        let replay_cost = self
            .configuration
            .replay_latency_per_packet
            .mul_f64(prior_prefix.packets as f64);
        let cache = match context {
            AudioContext::Cached(expected) => {
                if expected != prior_prefix {
                    self.reject(session_id, reply, FrameRejection::InvalidPrefix);
                    return;
                }
                if self.cache.lookup(cache_handle) != Some(prior_prefix) {
                    self.measurements.counters.cache_misses += 1;
                    if !self.input_has_time_for_compute(deadline, replay_cost) {
                        self.reject(session_id, reply, compute_rejection(replay_cost));
                        return;
                    }
                    let _ = reply.send(InputOutcome::CacheMiss);
                    return;
                }
                CacheOutcome::Hit
            }
            AudioContext::Replay(prefix) => {
                match self
                    .validate_replay(prefix, prior_prefix, deadline, replay_cost)
                    .await
                {
                    Ok(cache) => cache,
                    Err(reason) => {
                        self.reject(session_id, reply, reason);
                        return;
                    }
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
            self.reject(session_id, reply, compute_rejection(replay_duration));
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

    fn validate_input(
        &self,
        session: &WorkerSession,
        input: &InputFrame,
    ) -> Result<(), FrameRejection> {
        if !matches!(session.work, SessionWork::Idle) {
            return Err(FrameRejection::Overloaded);
        }
        if self.recovery_deadline(input.deadline) <= Instant::now() {
            return Err(FrameRejection::DeadlineExceeded);
        }
        if input.packet.sequence.0 != session.prefix.packets {
            return Err(FrameRejection::InvalidSequence);
        }
        if session.prefix.packets >= self.configuration.audio_limits.max_prefix_packets as u64
            || session.prefix.bytes + input.packet.payload.len() as u64
                > self.configuration.audio_limits.max_prefix_bytes as u64
        {
            return Err(FrameRejection::PrefixCapacity);
        }
        Ok(())
    }

    async fn validate_replay(
        &self,
        prefix: AudioPrefix,
        expected: PrefixState,
        deadline: Instant,
        replay_duration: Duration,
    ) -> Result<CacheOutcome, FrameRejection> {
        if !self.input_has_time_for_compute(deadline, replay_duration) {
            return Err(compute_rejection(replay_duration));
        }
        let limits = self.configuration.audio_limits;
        if prefix.0.len() > limits.max_prefix_packets
            || prefix.byte_len() > limits.max_prefix_bytes
            || prefix
                .0
                .iter()
                .any(|frame| frame.len() > limits.max_frame_bytes)
        {
            return Err(FrameRejection::PrefixCapacity);
        }
        let state = tokio::task::spawn_blocking(move || prefix.state())
            .await
            .expect("prefix hashing does not panic");
        if state != expected {
            return Err(FrameRejection::InvalidPrefix);
        }
        Ok(CacheOutcome::Replayed {
            packets: state.packets,
            bytes: state.bytes,
        })
    }
}

fn compute_rejection(replay_duration: Duration) -> FrameRejection {
    if replay_duration.is_zero() {
        FrameRejection::DeadlineExceeded
    } else {
        FrameRejection::ReplayTooExpensive
    }
}
