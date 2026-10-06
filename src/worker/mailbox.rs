use crate::{
    metrics::WorkerMeasurements,
    protocol::{Assignment, Generation, InputFrame, InputOutcome, SessionId},
    scheduler::admission::WorkerStatus,
};
use tokio::{
    sync::{mpsc, oneshot, watch},
    task::JoinHandle,
    time::Instant,
};

pub(crate) enum WorkerCommand {
    AddSession {
        session_id: SessionId,
        assignment: Assignment,
        reply: oneshot::Sender<bool>,
    },
    InputReady {
        session_id: SessionId,
        generation: Generation,
        input: InputFrame,
        reply: oneshot::Sender<InputOutcome>,
        routed_at: Instant,
    },
    RemoveSession {
        session_id: SessionId,
        generation: Generation,
        reply: oneshot::Sender<()>,
    },
    EvictCache {
        session_id: SessionId,
        generation: Generation,
        reply: oneshot::Sender<bool>,
    },
}
pub(crate) struct WorkerHandle {
    pub commands: mpsc::Sender<WorkerCommand>,
    pub status: watch::Receiver<WorkerStatus>,
    pub task: JoinHandle<WorkerMeasurements>,
}
