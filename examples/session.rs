use bytes::Bytes;
use std::time::Duration;
use tokio::time::Instant;
use voice_scheduler::{
    Node, RuntimeError,
    config::RuntimeConfig,
    protocol::{
        AudioContext, AudioPacket, CreateOutcome, InputFrame, InputOutcome, PacketSequence,
        PrefixState, SessionId,
    },
};

#[tokio::main]
async fn main() -> Result<(), RuntimeError> {
    let node = Node::start(RuntimeConfig::default()).await?;
    let session_id = SessionId(1);
    let CreateOutcome::Admitted(assignment) = node.ingress.create_session(session_id).await? else {
        panic!("no realtime capacity");
    };
    let timestamp = Instant::now();
    let outcome = node
        .ingress
        .input_frame(
            session_id,
            InputFrame {
                timestamp,
                deadline: timestamp + Duration::from_millis(50),
                packet: AudioPacket {
                    sequence: PacketSequence(0),
                    payload: Bytes::from(vec![0; 1600]),
                    context: AudioContext::Cached(PrefixState::default()),
                },
            },
        )
        .await?;
    match outcome {
        InputOutcome::Processed(output) => {
            assert_eq!(output.audio.assignment, assignment);
            println!(
                "Echoed {} bytes from worker {}",
                output.audio.payload.len(),
                assignment.worker_id.0
            );
        }
        outcome => println!("Packet was not processed: {outcome:?}"),
    }
    node.ingress.close_session(session_id).await?;
    let report = node.shutdown().await?;
    println!("Delivered {} frames", report.inference.delivered_frames);
    Ok(())
}
