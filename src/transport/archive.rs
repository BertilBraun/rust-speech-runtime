use std::{
    io::Write,
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};

use serde::Serialize;
use tokio::{sync::mpsc, task::JoinHandle};

use crate::protocol::SessionRecord;

#[derive(Debug, Default, Serialize)]
pub struct ArchiveReport {
    pub saved: u64,
    pub failed: u64,
}

#[derive(Serialize)]
pub struct ArchiveRecord {
    pub created_unix_ms: u128,
    pub closed_unix_ms: u128,
    pub conversation: SessionRecord,
}

pub enum EnqueueOutcome {
    Queued,
    Backpressured,
}

pub async fn enqueue(
    sender: &mpsc::Sender<ArchiveRecord>,
    record: ArchiveRecord,
) -> Result<EnqueueOutcome, mpsc::error::SendError<ArchiveRecord>> {
    match sender.try_send(record) {
        Ok(()) => Ok(EnqueueOutcome::Queued),
        Err(mpsc::error::TrySendError::Full(record)) => {
            sender.send(record).await?;
            Ok(EnqueueOutcome::Backpressured)
        }
        Err(mpsc::error::TrySendError::Closed(record)) => Err(mpsc::error::SendError(record)),
    }
}

pub fn unix_milliseconds() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock follows Unix epoch")
        .as_millis()
}

pub fn start(
    directory: PathBuf,
    capacity: usize,
) -> (mpsc::Sender<ArchiveRecord>, JoinHandle<ArchiveReport>) {
    let (sender, mut receiver) = mpsc::channel::<ArchiveRecord>(capacity);
    let task = tokio::spawn(async move {
        let mut report = ArchiveReport::default();
        while let Some(record) = receiver.recv().await {
            let destination = directory.clone();
            let ordinal = report.saved + report.failed;
            let result =
                tokio::task::spawn_blocking(move || write_record(destination, ordinal, record))
                    .await;
            match result {
                Ok(Ok(())) => report.saved += 1,
                failure => {
                    report.failed += 1;
                    eprintln!("session archive failed: {failure:?}");
                }
            }
        }
        report
    });
    (sender, task)
}

fn write_record(
    directory: PathBuf,
    ordinal: u64,
    record: ArchiveRecord,
) -> Result<(), std::io::Error> {
    std::fs::create_dir_all(&directory)?;
    let destination = directory.join(format!(
        "{}-{}-{ordinal}-{}.json",
        record.closed_unix_ms,
        record.conversation.worker_id,
        rand::random::<u64>()
    ));
    let pending = destination.with_extension("json.partial");
    let result =
        write_pending(&pending, &record).and_then(|()| std::fs::rename(&pending, destination));
    if result.is_err()
        && pending.exists()
        && let Err(error) = std::fs::remove_file(&pending)
    {
        eprintln!("failed to remove incomplete session archive: {error}");
    }
    result
}

fn write_pending(path: &std::path::Path, record: &ArchiveRecord) -> Result<(), std::io::Error> {
    let file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?;
    let mut writer = std::io::BufWriter::new(file);
    serde_json::to_writer(&mut writer, record).map_err(std::io::Error::other)?;
    writer.flush()?;
    writer.get_ref().sync_all()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::SessionId;

    #[tokio::test]
    async fn saturated_archive_mailbox_retains_pending_record() {
        let (sender, mut receiver) = mpsc::channel(1);
        sender.send(record("first")).await.expect("first record");
        let mut pending = tokio::spawn(async move { enqueue(&sender, record("second")).await });
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(20), &mut pending)
                .await
                .is_err()
        );
        assert_eq!(
            receiver
                .recv()
                .await
                .expect("first queued")
                .conversation
                .session_id
                .0,
            "first"
        );
        assert!(matches!(
            pending.await.expect("enqueue task").expect("second queued"),
            EnqueueOutcome::Backpressured
        ));
        assert_eq!(
            receiver
                .recv()
                .await
                .expect("retained record")
                .conversation
                .session_id
                .0,
            "second"
        );
        assert!(receiver.recv().await.is_none());
    }

    fn record(session_id: &str) -> ArchiveRecord {
        ArchiveRecord {
            created_unix_ms: 1,
            closed_unix_ms: 2,
            conversation: SessionRecord {
                session_id: SessionId(session_id.into()),
                worker_id: 0,
                model_id: "test-model".into(),
                turns: Vec::new(),
            },
        }
    }

    #[tokio::test]
    async fn archive_directory_failure_is_reported() {
        let destination =
            std::env::temp_dir().join(format!("voice-archive-blocker-{}", rand::random::<u64>()));
        std::fs::write(&destination, b"not a directory").expect("test blocker");
        let (sender, task) = start(destination.clone(), 1);
        sender
            .send(ArchiveRecord {
                created_unix_ms: 1,
                closed_unix_ms: 2,
                conversation: SessionRecord {
                    session_id: SessionId("test".into()),
                    worker_id: 0,
                    model_id: "test-model".into(),
                    turns: Vec::new(),
                },
            })
            .await
            .expect("record queued");
        drop(sender);
        let report = task.await.expect("archive task");
        assert_eq!(report.failed, 1);
        assert_eq!(report.saved, 0);
        std::fs::remove_file(destination).expect("remove test blocker");
    }
}
