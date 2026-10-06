use std::{net::SocketAddr, time::Duration};

use bytes::Bytes;
use tokio::time::Instant;

use super::wire::{ClientPeer, ClientRequest, ServerReply, WireError};
use crate::protocol::{
    Assignment, AudioContext, AudioPacket, AudioPrefix, AudioResult, CreateOutcome,
    CreateRejection, FrameRejection, PacketSequence, PrefixState, SessionAdmission, SessionId,
};

#[derive(Debug, thiserror::Error)]
pub enum ClientError {
    #[error(transparent)]
    Wire(#[from] WireError),
    #[error("end-to-end packet deadline exceeded")]
    DeadlineExceeded,
    #[error("audio echo arrived beyond the bounded session recovery budget")]
    RecoveryExceeded(Box<AudioResult>),
    #[error("server rejected packet: {0:?}")]
    Rejected(FrameRejection),
    #[error("invalid server response: {0}")]
    Protocol(&'static str),
    #[error("audio prefix exceeded negotiated limits")]
    PrefixCapacity,
    #[error("audio echo, prefix or sticky assignment does not match")]
    EchoMismatch,
}
pub enum ConnectOutcome {
    Admitted(Box<AudioSession>),
    Rejected(CreateRejection),
}
/// A validated echo advances input history even when its output must be discarded.
#[derive(Debug)]
pub enum AudioDelivery {
    OnTime(AudioResult),
    Late(AudioResult),
    Discarded(AudioResult),
}

impl AudioDelivery {
    pub fn into_audio(self) -> AudioResult {
        match self {
            Self::OnTime(audio) | Self::Late(audio) | Self::Discarded(audio) => audio,
        }
    }
}

/// Owns a persistent connection and its bounded input replay history.
pub struct AudioSession {
    peer: ClientPeer,
    admission: SessionAdmission,
    history: AudioPrefix,
    prefix: PrefixState,
}

pub async fn connect_session(
    address: SocketAddr,
    session_id: SessionId,
    io_timeout: Duration,
) -> Result<ConnectOutcome, ClientError> {
    let operation = async {
        let mut peer = ClientPeer::connect(address).await?;
        peer.send(ClientRequest::Open(session_id)).await?;
        let Some(ServerReply::Opened(outcome)) = peer.receive().await? else {
            return Err(ClientError::Protocol("expected session admission reply"));
        };
        match outcome {
            CreateOutcome::Admitted(admission) => {
                Ok(ConnectOutcome::Admitted(Box::new(AudioSession {
                    peer,
                    admission,
                    history: AudioPrefix::default(),
                    prefix: PrefixState::default(),
                })))
            }
            CreateOutcome::Rejected(reason) => Ok(ConnectOutcome::Rejected(reason)),
        }
    };
    tokio::time::timeout(io_timeout, operation)
        .await
        .map_err(|_| WireError::Timeout)?
}

impl AudioSession {
    pub fn assignment(&self) -> Assignment {
        self.admission.assignment
    }
    pub fn packet_deadline(&self) -> Duration {
        self.admission.packet_deadline
    }
    pub fn packet_completion_budget(&self) -> Duration {
        self.admission.packet_deadline + self.admission.packet_lateness_grace
    }
    pub fn packet_recovery_budget(&self) -> Duration {
        self.admission.packet_recovery_budget
    }
    pub fn next_sequence(&self) -> PacketSequence {
        PacketSequence(self.prefix.packets)
    }

    /// Successful deliveries preserve the session; errors require closing it.
    pub async fn infer_audio(
        &mut self,
        payload: Bytes,
        captured_at: Instant,
    ) -> Result<AudioDelivery, ClientError> {
        if payload.len() > self.admission.audio_limits.max_frame_bytes
            || self.prefix.bytes + payload.len() as u64
                > self.admission.audio_limits.max_prefix_bytes as u64
            || self.history.0.len() >= self.admission.audio_limits.max_prefix_packets
        {
            return Err(ClientError::PrefixCapacity);
        }
        let deadline = captured_at + self.admission.packet_deadline;
        let recovery_deadline = captured_at + self.admission.packet_recovery_budget;
        let operation = self.exchange_audio(payload, deadline);
        let audio = tokio::time::timeout_at(recovery_deadline, operation)
            .await
            .map_err(|_| ClientError::DeadlineExceeded)??;
        self.classify_echo(audio, captured_at.elapsed())
    }

    pub(crate) fn classify_echo(
        &self,
        audio: AudioResult,
        round_trip: Duration,
    ) -> Result<AudioDelivery, ClientError> {
        if round_trip > self.packet_recovery_budget() {
            Err(ClientError::RecoveryExceeded(Box::new(audio)))
        } else if round_trip > self.packet_completion_budget() {
            Ok(AudioDelivery::Discarded(audio))
        } else if round_trip > self.packet_deadline() {
            Ok(AudioDelivery::Late(audio))
        } else {
            Ok(AudioDelivery::OnTime(audio))
        }
    }

    async fn exchange_audio(
        &mut self,
        payload: Bytes,
        deadline: Instant,
    ) -> Result<AudioResult, ClientError> {
        let sequence = self.next_sequence();
        let recovery_deadline =
            deadline + (self.admission.packet_recovery_budget - self.admission.packet_deadline);
        let packet = AudioPacket {
            sequence,
            payload: payload.clone(),
            context: AudioContext::Cached(self.prefix),
        };
        self.peer
            .send(ClientRequest::Audio {
                packet,
                remaining_budget: recovery_deadline.saturating_duration_since(Instant::now()),
            })
            .await?;
        let mut reply = self.peer.receive().await?.ok_or(WireError::Closed)?;
        if matches!(reply, ServerReply::CacheMiss) {
            let packet = AudioPacket {
                sequence,
                payload: payload.clone(),
                context: AudioContext::Replay(self.history.clone()),
            };
            self.peer
                .send(ClientRequest::Audio {
                    packet,
                    remaining_budget: recovery_deadline.saturating_duration_since(Instant::now()),
                })
                .await?;
            reply = self.peer.receive().await?.ok_or(WireError::Closed)?;
        }
        match reply {
            ServerReply::Audio(audio) => {
                let expected = self.prefix.append(&payload);
                if audio.assignment != self.admission.assignment
                    || audio.sequence != sequence
                    || audio.payload != payload
                    || audio.prefix != expected
                {
                    return Err(ClientError::EchoMismatch);
                }
                if Instant::now() > recovery_deadline {
                    return Err(ClientError::RecoveryExceeded(Box::new(audio)));
                }
                self.history.0.push(payload);
                self.prefix = expected;
                Ok(audio)
            }
            ServerReply::Rejected(reason) => Err(ClientError::Rejected(reason)),
            _ => Err(ClientError::Protocol("expected audio reply")),
        }
    }

    pub async fn evict_cache(&mut self) -> Result<bool, ClientError> {
        self.peer.send(ClientRequest::EvictCache).await?;
        match self.peer.receive().await? {
            Some(ServerReply::CacheEvicted(removed)) => Ok(removed),
            _ => Err(ClientError::Protocol("expected cache eviction reply")),
        }
    }

    pub async fn close(mut self) -> Result<bool, ClientError> {
        self.peer.send(ClientRequest::Close).await?;
        match self.peer.receive().await? {
            Some(ServerReply::Closed(closed)) => Ok(closed),
            _ => Err(ClientError::Protocol("expected close reply")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{AudioDelivery, AudioSession, ClientError, ConnectOutcome, connect_session};
    use crate::{
        config::AudioLimits,
        metrics::profile::PacketTimings,
        protocol::{
            Assignment, AudioResult, CacheOutcome, CreateOutcome, Generation, PrefixState,
            SessionAdmission, SessionId, WorkerId,
        },
        transport::wire::{ClientRequest, MAX_MESSAGE_BYTES, ServerPeer, ServerReply},
    };
    use bytes::Bytes;
    use std::time::Duration;
    use tokio::{net::TcpListener, time::Instant};

    async fn delayed_echo(
        delay: Duration,
    ) -> (Result<AudioDelivery, ClientError>, Box<AudioSession>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let assignment = Assignment {
            worker_id: WorkerId(0),
            generation: Generation(1),
        };
        let server = tokio::spawn(async move {
            let (socket, _) = listener.accept().await.unwrap();
            let mut peer = ServerPeer::new(socket, MAX_MESSAGE_BYTES).unwrap();
            assert!(matches!(
                peer.receive().await.unwrap(),
                Some(ClientRequest::Open(_))
            ));
            peer.send(ServerReply::Opened(CreateOutcome::Admitted(
                SessionAdmission {
                    assignment,
                    audio_limits: AudioLimits::default(),
                    packet_deadline: Duration::from_millis(50),
                    packet_lateness_grace: Duration::from_millis(10),
                    packet_recovery_budget: Duration::from_millis(204),
                },
            )))
            .await
            .unwrap();
            let Some(ClientRequest::Audio { packet, .. }) = peer.receive().await.unwrap() else {
                panic!("expected audio request");
            };
            tokio::time::advance(delay).await;
            peer.send(ServerReply::Audio(AudioResult {
                assignment,
                sequence: packet.sequence,
                prefix: PrefixState::default().append(&packet.payload),
                payload: packet.payload,
                cache: CacheOutcome::Hit,
                timings: Box::new(PacketTimings {
                    device_execution: Duration::from_millis(12),
                    ..PacketTimings::default()
                }),
            }))
            .await
            .unwrap();
        });
        let ConnectOutcome::Admitted(mut session) =
            connect_session(address, SessionId(1), Duration::from_secs(2))
                .await
                .unwrap()
        else {
            panic!("expected admission");
        };
        tokio::time::pause();
        let captured_at = Instant::now();
        let outcome = session
            .exchange_audio(
                Bytes::from_static(b"audio"),
                captured_at + Duration::from_millis(50),
            )
            .await
            .and_then(|audio| session.classify_echo(audio, captured_at.elapsed()));
        server.await.unwrap();
        (outcome, session)
    }

    #[tokio::test]
    async fn received_late_echo_preserves_profile_and_advances_audio_history() {
        let (outcome, session) = delayed_echo(Duration::from_millis(55)).await;
        let AudioDelivery::Late(audio) = outcome.unwrap() else {
            panic!("received late audio must retain its profile");
        };
        assert_eq!(audio.payload, Bytes::from_static(b"audio"));
        assert_eq!(audio.timings.device_execution, Duration::from_millis(12));
        assert_eq!(session.prefix, PrefixState::default().append(b"audio"));
        assert_eq!(session.history.0, vec![Bytes::from_static(b"audio")]);
    }

    #[tokio::test]
    async fn discarded_output_advances_input_history_and_retains_its_profile() {
        let (outcome, session) = delayed_echo(Duration::from_millis(65)).await;
        let AudioDelivery::Discarded(audio) = outcome.unwrap() else {
            panic!("output beyond playback budget must be discarded");
        };
        assert_eq!(audio.timings.device_execution, Duration::from_millis(12));
        assert_eq!(session.prefix, PrefixState::default().append(b"audio"));
        assert_eq!(session.history.0, vec![Bytes::from_static(b"audio")]);
    }

    #[tokio::test]
    async fn expired_session_recovery_is_fatal_and_retains_its_profile() {
        let (outcome, session) = delayed_echo(Duration::from_millis(230)).await;
        let ClientError::RecoveryExceeded(audio) = outcome.unwrap_err() else {
            panic!("session recovery must remain bounded");
        };
        assert_eq!(audio.timings.device_execution, Duration::from_millis(12));
        assert_eq!(session.prefix, PrefixState::default());
        assert!(session.history.0.is_empty());
    }
}
