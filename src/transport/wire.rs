use bytes::{BufMut, Bytes, BytesMut};
use serde::{Deserialize, Serialize};

use crate::protocol::{SessionId, TurnId};

use super::GatewayError;

pub const AUDIO_HEADER_BYTES: usize = 12;

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum ClientControl {
    Open {
        session_id: SessionId,
    },
    StartTurn {
        turn_id: TurnId,
    },
    Commit {
        turn_id: TurnId,
        chunk_count: u32,
        sample_count: usize,
    },
    Cancel {
        turn_id: TurnId,
    },
    Close,
}

#[derive(Debug)]
pub struct AudioChunk {
    pub turn_id: TurnId,
    pub chunk_index: u32,
    pub pcm16: Bytes,
}

impl AudioChunk {
    pub fn decode(bytes: Bytes) -> Result<Self, GatewayError> {
        if bytes.len() <= AUDIO_HEADER_BYTES
            || !(bytes.len() - AUDIO_HEADER_BYTES).is_multiple_of(2)
        {
            return Err(GatewayError::Protocol(
                "audio must contain a 12-byte header and nonempty PCM16",
            ));
        }
        let turn_id = TurnId(u64::from_le_bytes(
            bytes[..8].try_into().expect("header length validated"),
        ));
        let chunk_index =
            u32::from_le_bytes(bytes[8..12].try_into().expect("header length validated"));
        Ok(Self {
            turn_id,
            chunk_index,
            pcm16: bytes.slice(AUDIO_HEADER_BYTES..),
        })
    }

    pub fn encode(&self) -> Bytes {
        let mut encoded = BytesMut::with_capacity(AUDIO_HEADER_BYTES + self.pcm16.len());
        encoded.put_u64_le(self.turn_id.0);
        encoded.put_u32_le(self.chunk_index);
        encoded.extend_from_slice(&self.pcm16);
        encoded.freeze()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binary_audio_round_trips() {
        let chunk = AudioChunk {
            turn_id: TurnId(9),
            chunk_index: 3,
            pcm16: Bytes::from_static(&[1, 2, 3, 4]),
        };
        let decoded = AudioChunk::decode(chunk.encode()).expect("valid packet");
        assert_eq!(decoded.turn_id, chunk.turn_id);
        assert_eq!(decoded.chunk_index, chunk.chunk_index);
        assert_eq!(decoded.pcm16, chunk.pcm16);
    }

    #[test]
    fn rejects_unknown_control_fields() {
        assert!(
            serde_json::from_str::<ClientControl>(
                r#"{"type":"open","session_id":"s","authentication":"unused"}"#
            )
            .is_err()
        );
    }

    #[test]
    fn rejects_empty_and_incomplete_pcm() {
        for length in [0, 11, 12, 13, 15] {
            assert!(AudioChunk::decode(Bytes::from(vec![0; length])).is_err());
        }
    }
}
