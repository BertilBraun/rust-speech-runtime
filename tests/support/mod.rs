use std::{collections::HashMap, net::SocketAddr, time::Duration};

use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream},
    task::JoinHandle,
};
use voice_scheduler::{
    config::{RuntimeConfig, WorkerConfig},
    protocol::backend::{
        BatchRequest, BatchResponse, Memory, Operation, OperationResult, Outcome, Ready, Timing,
    },
};

pub struct Fixture {
    endpoint: SocketAddr,
    task: JoinHandle<()>,
}

impl Fixture {
    pub async fn start(delay_ms: u64) -> Self {
        Self::launch(delay_ms, false).await
    }

    pub async fn failing_decode(delay_ms: u64) -> Self {
        Self::launch(delay_ms, true).await
    }

    async fn launch(delay_ms: u64, fail_decode: bool) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("fixture binds");
        let endpoint = listener.local_addr().expect("fixture address");
        let task = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.expect("runtime connects");
            stream
                .set_nodelay(true)
                .expect("disable fixture delayed writes");
            write_json(
                &mut stream,
                &Ready {
                    r#type: "ready".into(),
                    protocol_version: 1,
                    body_bytes: 0,
                    model_id: "test-worker-v1".into(),
                    max_context_tokens: 16384,
                    max_batch_size: 16,
                    max_audio_samples: 480_000,
                },
            )
            .await;
            let mut contexts = HashMap::<String, usize>::new();
            while let Ok(length) = stream.read_u32().await {
                assert!(length < 1024 * 1024);
                let mut metadata = vec![0; length as usize];
                if stream.read_exact(&mut metadata).await.is_err() {
                    break;
                }
                let request: BatchRequest =
                    serde_json::from_slice(&metadata).expect("typed request");
                assert!(request.body_bytes <= 16 * 960_000);
                let mut audio = vec![0; request.body_bytes];
                stream.read_exact(&mut audio).await.expect("audio body");
                tokio::time::sleep(Duration::from_millis(delay_ms)).await;
                let results = request
                    .operations
                    .into_iter()
                    .map(|operation| result(operation, &mut contexts, fail_decode))
                    .collect();
                write_json(
                    &mut stream,
                    &BatchResponse {
                        request_id: request.request_id,
                        body_bytes: 0,
                        results,
                        timing: Timing {
                            elapsed_ms: delay_ms as f64,
                            decode_ms: delay_ms as f64,
                            ..Timing::default()
                        },
                        memory: Memory::default(),
                    },
                )
                .await;
            }
        });
        Self { endpoint, task }
    }

    pub fn config(&self) -> RuntimeConfig {
        RuntimeConfig {
            workers: vec![WorkerConfig {
                endpoint: self.endpoint,
            }],
            initial_forward_estimate_ms: 1.0,
            ..RuntimeConfig::default()
        }
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        self.task.abort();
    }
}

fn result(
    operation: Operation,
    contexts: &mut HashMap<String, usize>,
    fail_decode: bool,
) -> OperationResult {
    let operation_id = operation.operation_id();
    let session_id = operation.session_id().to_owned();
    let (turn_id, generation, outcome) = match operation {
        Operation::Open { .. } => {
            contexts.insert(session_id.clone(), 0);
            (None, None, Outcome::Opened)
        }
        Operation::Close { .. } => {
            contexts.remove(&session_id);
            (None, None, Outcome::Closed)
        }
        Operation::Prefill {
            turn_id,
            generation,
            audio_bytes,
            accepted,
            ..
        } => {
            let context = contexts.get_mut(&session_id).expect("owned cache");
            *context += audio_bytes.div_ceil(3200) + 12 + usize::from(accepted.is_some());
            (
                Some(turn_id),
                Some(generation),
                Outcome::Token {
                    token_id: 100,
                    text_delta: "a".into(),
                    eos: false,
                    context_tokens: *context,
                },
            )
        }
        Operation::Decode {
            turn_id,
            generation,
            accepted,
            ..
        } => {
            if fail_decode {
                return OperationResult {
                    operation_id,
                    session_id,
                    turn_id: Some(turn_id),
                    generation: Some(generation),
                    outcome: Outcome::Failed {
                        code: voice_scheduler::protocol::ErrorCode::BackendFailed,
                        message: "test decode failure".into(),
                    },
                };
            }
            let context = contexts.get_mut(&session_id).expect("owned cache");
            *context += 1;
            let eos = accepted.index >= 3;
            (
                Some(turn_id),
                Some(generation),
                Outcome::Token {
                    token_id: if eos {
                        248046
                    } else {
                        101 + accepted.index as u32
                    },
                    text_delta: if eos { String::new() } else { "b".into() },
                    eos,
                    context_tokens: *context,
                },
            )
        }
    };
    OperationResult {
        operation_id,
        session_id,
        turn_id,
        generation,
        outcome,
    }
}

async fn write_json<T: serde::Serialize>(stream: &mut TcpStream, value: &T) {
    let metadata = serde_json::to_vec(value).expect("serialize fixture");
    if stream.write_u32(metadata.len() as u32).await.is_err() {
        return;
    }
    let _ = stream.write_all(&metadata).await;
}
