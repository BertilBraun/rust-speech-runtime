use std::time::Duration;

use bytes::Bytes;
use tokio::time::Instant;
use voice_scheduler::{
    Node, RuntimeError,
    config::RuntimeConfig,
    protocol::{CreateOutcome, InputFrame, InputOutcome, SessionId},
};

#[tokio::main]
async fn main() -> Result<(), RuntimeError> {
    let mut node = Node::start(RuntimeConfig::default())?;
    let mut outputs = node.take_outputs().expect("first receiver owner");
    let session_id = SessionId(1);
    let CreateOutcome::Admitted(assignment) = node.ingress.create_session(session_id).await? else {
        panic!("empty node has capacity");
    };
    println!("Assigned to worker {}", assignment.worker_id.0);
    let outcome = node
        .ingress
        .input_frame(
            session_id,
            InputFrame {
                timestamp: Instant::now(),
                payload: Bytes::from_static(b"audio frame"),
            },
        )
        .await?;
    assert_eq!(outcome, InputOutcome::Accepted);
    let output = tokio::time::timeout(Duration::from_secs(1), outputs.recv())
        .await
        .expect("mock inference completes within one second")
        .expect("runtime is active");
    assert_eq!(output.assignment, assignment);
    node.ingress.close_session(session_id).await?;
    let report = node.shutdown().await?;
    println!("Delivered {} result", report.delivered_results);
    Ok(())
}
