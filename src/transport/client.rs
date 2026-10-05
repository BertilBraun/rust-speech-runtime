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
    pub fn next_sequence(&self) -> PacketSequence {
        PacketSequence(self.prefix.packets)
    }

    pub async fn infer_audio(
        &mut self,
        payload: Bytes,
        captured_at: Instant,
    ) -> Result<AudioResult, ClientError> {
        if payload.len() > self.admission.audio_limits.max_frame_bytes
            || self.prefix.bytes + payload.len() as u64
                > self.admission.audio_limits.max_prefix_bytes as u64
            || self.history.0.len() >= self.admission.audio_limits.max_prefix_packets
        {
            return Err(ClientError::PrefixCapacity);
        }
        let deadline = captured_at + self.admission.packet_deadline;
        let operation = self.exchange_audio(payload, deadline);
        tokio::time::timeout_at(deadline, operation)
            .await
            .map_err(|_| ClientError::DeadlineExceeded)?
    }

    async fn exchange_audio(
        &mut self,
        payload: Bytes,
        deadline: Instant,
    ) -> Result<AudioResult, ClientError> {
        let sequence = self.next_sequence();
        let packet = AudioPacket {
            sequence,
            payload: payload.clone(),
            context: AudioContext::Cached(self.prefix),
        };
        self.peer
            .send(ClientRequest::Audio {
                packet,
                remaining_budget: deadline.saturating_duration_since(Instant::now()),
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
                    remaining_budget: deadline.saturating_duration_since(Instant::now()),
                })
                .await?;
            reply = self.peer.receive().await?.ok_or(WireError::Closed)?;
        }
        match reply {
            ServerReply::Audio(audio) => {
                if Instant::now() > deadline {
                    return Err(ClientError::DeadlineExceeded);
                }
                let expected = self.prefix.append(&payload);
                if audio.assignment != self.admission.assignment
                    || audio.sequence != sequence
                    || audio.payload != payload
                    || audio.prefix != expected
                {
                    return Err(ClientError::EchoMismatch);
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
