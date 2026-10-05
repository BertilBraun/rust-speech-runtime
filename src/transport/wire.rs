use std::{marker::PhantomData, net::SocketAddr, time::Duration};

use bytes::Bytes;
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use tokio::net::{
    TcpStream,
    tcp::{OwnedReadHalf, OwnedWriteHalf},
};
use tokio_util::codec::{FramedRead, FramedWrite, LengthDelimitedCodec};

use crate::protocol::{AudioPacket, AudioResult, CreateOutcome, FrameRejection, SessionId};

pub(crate) const MAX_MESSAGE_BYTES: usize = 32 * 1024 * 1024;
const INLINE_MESSAGE_BYTES: usize = 8192;

#[derive(Debug, Serialize, Deserialize)]
pub(crate) enum ClientRequest {
    Open(SessionId),
    Audio {
        packet: AudioPacket,
        remaining_budget: Duration,
    },
    EvictCache,
    Close,
}

#[cfg(test)]
mod tests {
    use super::{ClientRequest, MAX_MESSAGE_BYTES, ServerPeer};
    use crate::protocol::SessionId;
    use std::time::Duration;
    use tokio::{
        io::AsyncWriteExt,
        net::{TcpListener, TcpStream},
    };

    #[tokio::test]
    async fn cancelled_receive_keeps_partially_received_message() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let mut client = TcpStream::connect(listener.local_addr().unwrap())
            .await
            .unwrap();
        let (socket, _) = listener.accept().await.unwrap();
        let mut server = ServerPeer::new(socket, MAX_MESSAGE_BYTES).unwrap();
        let message = bincode::serde::encode_to_vec(
            ClientRequest::Open(SessionId(42)),
            bincode::config::standard(),
        )
        .unwrap();
        client
            .write_all(&(message.len() as u32).to_be_bytes())
            .await
            .unwrap();
        client.write_all(&message[..1]).await.unwrap();
        assert!(
            tokio::time::timeout(Duration::from_millis(5), server.receive())
                .await
                .is_err()
        );
        client.write_all(&message[1..]).await.unwrap();
        assert!(matches!(
            server.receive().await.unwrap(),
            Some(ClientRequest::Open(SessionId(42)))
        ));
    }

    #[tokio::test]
    async fn cancelled_receive_keeps_pending_decode() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let client = TcpStream::connect(listener.local_addr().unwrap())
            .await
            .unwrap();
        let (socket, _) = listener.accept().await.unwrap();
        let mut server = ServerPeer::new(socket, MAX_MESSAGE_BYTES).unwrap();
        let message = bincode::serde::encode_to_vec(
            ClientRequest::Open(SessionId(7)),
            bincode::config::standard(),
        )
        .unwrap();
        server.pending_decode = Some(tokio::task::spawn_blocking(move || {
            std::thread::sleep(Duration::from_millis(40));
            Ok(
                bincode::serde::decode_from_slice(&message, bincode::config::standard())
                    .unwrap()
                    .0,
            )
        }));
        assert!(
            tokio::time::timeout(Duration::from_millis(5), server.receive())
                .await
                .is_err()
        );
        assert!(matches!(
            server.receive().await.unwrap(),
            Some(ClientRequest::Open(SessionId(7)))
        ));
        drop(client);
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) enum ServerReply {
    Opened(CreateOutcome),
    Audio(AudioResult),
    CacheMiss,
    Rejected(FrameRejection),
    CacheEvicted(bool),
    Closed(bool),
}

pub(crate) trait WireMessage: Serialize + Send + 'static {
    fn size_upper_bound(&self) -> usize;
}
impl WireMessage for ClientRequest {
    fn size_upper_bound(&self) -> usize {
        match self {
            Self::Audio { packet, .. } => {
                let context = match &packet.context {
                    crate::protocol::AudioContext::Cached(_) => 0,
                    crate::protocol::AudioContext::Replay(prefix) => {
                        prefix.byte_len() + prefix.0.len() * 10
                    }
                };
                1024 + packet.payload.len() + context
            }
            _ => 1024,
        }
    }
}
impl WireMessage for ServerReply {
    fn size_upper_bound(&self) -> usize {
        match self {
            Self::Audio(audio) => 1024 + audio.payload.len(),
            _ => 1024,
        }
    }
}
fn encode<Outgoing: Serialize>(message: Outgoing) -> Result<Vec<u8>, bincode::error::EncodeError> {
    bincode::serde::encode_to_vec(
        message,
        bincode::config::standard().with_limit::<MAX_MESSAGE_BYTES>(),
    )
}
fn decode<Incoming: DeserializeOwned>(bytes: &[u8]) -> Result<Incoming, WireError> {
    let (message, consumed) = bincode::serde::decode_from_slice::<Incoming, _>(
        bytes,
        bincode::config::standard().with_limit::<MAX_MESSAGE_BYTES>(),
    )?;
    if consumed != bytes.len() {
        return Err(WireError::InvalidMessage("trailing bytes"));
    }
    Ok(message)
}

#[derive(Debug, thiserror::Error)]
pub enum WireError {
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Encode(#[from] bincode::error::EncodeError),
    #[error(transparent)]
    Decode(#[from] bincode::error::DecodeError),
    #[error(transparent)]
    Task(#[from] tokio::task::JoinError),
    #[error("connection closed")]
    Closed,
    #[error("invalid wire message: {0}")]
    InvalidMessage(&'static str),
    #[error("network operation timed out")]
    Timeout,
}

pub(crate) type ClientPeer = WirePeer<ServerReply, ClientRequest>;
pub(crate) type ServerPeer = WirePeer<ClientRequest, ServerReply>;

pub(crate) struct WirePeer<Incoming, Outgoing> {
    reader: FramedRead<OwnedReadHalf, LengthDelimitedCodec>,
    writer: FramedWrite<OwnedWriteHalf, LengthDelimitedCodec>,
    message_limit: usize,
    pending_decode: Option<tokio::task::JoinHandle<Result<Incoming, WireError>>>,
    outgoing: PhantomData<Outgoing>,
}

impl<Incoming: DeserializeOwned + Send + 'static, Outgoing: WireMessage>
    WirePeer<Incoming, Outgoing>
{
    pub(crate) fn new(stream: TcpStream, message_limit: usize) -> Result<Self, WireError> {
        stream.set_nodelay(true)?;
        let (reader, writer) = stream.into_split();
        let codec = || {
            LengthDelimitedCodec::builder()
                .max_frame_length(message_limit)
                .new_codec()
        };
        Ok(Self {
            reader: FramedRead::new(reader, codec()),
            writer: FramedWrite::new(writer, codec()),
            message_limit,
            pending_decode: None,
            outgoing: PhantomData,
        })
    }

    pub(crate) async fn connect(address: SocketAddr) -> Result<Self, WireError> {
        Self::new(TcpStream::connect(address).await?, MAX_MESSAGE_BYTES)
    }

    pub(crate) async fn send(&mut self, message: Outgoing) -> Result<(), WireError> {
        let bytes = if message.size_upper_bound() <= INLINE_MESSAGE_BYTES {
            encode(message)?
        } else {
            tokio::task::spawn_blocking(move || encode(message)).await??
        };
        if bytes.len() > self.message_limit {
            return Err(WireError::InvalidMessage("encoded message exceeds limit"));
        }
        self.writer.send(Bytes::from(bytes)).await?;
        Ok(())
    }

    pub(crate) async fn receive(&mut self) -> Result<Option<Incoming>, WireError> {
        if self.pending_decode.is_none() {
            let Some(frame) = self.reader.next().await else {
                return Ok(None);
            };
            let bytes = frame?;
            if bytes.len() <= INLINE_MESSAGE_BYTES {
                return decode(&bytes).map(Some);
            }
            self.pending_decode = Some(tokio::task::spawn_blocking(move || decode(&bytes)));
        }
        let result = self
            .pending_decode
            .as_mut()
            .expect("decode is pending")
            .await;
        self.pending_decode = None;
        let message = result??;
        Ok(Some(message))
    }
}
