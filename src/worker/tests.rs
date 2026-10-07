use super::WorkerHandle;
use crate::worker::Command;
use crate::{
    metrics::Metrics,
    protocol::{ErrorCode, SessionId, SessionRecord},
};
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize},
};
use tokio::sync::{mpsc, oneshot};
#[test]
fn bounded_worker_mailbox_rejects_without_allocating_more_queue_slots() {
    let (sender, _receiver) = mpsc::channel(1);
    let metrics = Arc::new(Metrics::default());
    let worker = WorkerHandle {
        id: 0,
        sender,
        load: Arc::new(AtomicUsize::new(0)),
        available: Arc::new(AtomicBool::new(true)),
        metrics: metrics.clone(),
    };
    worker
        .send(Command::Close {
            key: "first".into(),
            reply: None,
        })
        .unwrap();
    assert_eq!(
        worker
            .send(Command::Close {
                key: "second".into(),
                reply: None
            })
            .unwrap_err()
            .code(),
        ErrorCode::ChannelSaturated
    );
    assert_eq!(metrics.snapshot().channel_saturation_events, 1);
}

#[tokio::test]
async fn close_waits_for_bounded_mailbox_space_and_preserves_record_reply() {
    let (sender, mut receiver) = mpsc::channel(1);
    let worker = WorkerHandle {
        id: 0,
        sender,
        load: Arc::new(AtomicUsize::new(1)),
        available: Arc::new(AtomicBool::new(true)),
        metrics: Arc::new(Metrics::default()),
    };
    worker
        .send(Command::Close {
            key: "occupied".into(),
            reply: None,
        })
        .unwrap();
    let (reply, response) = oneshot::channel();
    let close = tokio::spawn(async move { worker.close("archive".into(), reply).await });
    tokio::task::yield_now().await;
    assert!(!close.is_finished());
    let _ = receiver.recv().await;
    close.await.unwrap().unwrap();
    match receiver.recv().await.unwrap() {
        Command::Close {
            key,
            reply: Some(reply),
        } => {
            assert_eq!(key, "archive");
            reply
                .send(Ok(SessionRecord {
                    session_id: SessionId("archive".into()),
                    worker_id: 0,
                    model_id: "test".into(),
                    turns: Vec::new(),
                }))
                .unwrap();
        }
        _ => panic!("close command must preserve reply"),
    }
    assert_eq!(response.await.unwrap().unwrap().session_id.0, "archive");
}
