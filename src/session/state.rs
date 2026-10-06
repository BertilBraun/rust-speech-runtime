use crate::{
    protocol::{
        FinishReason, SessionEvent, SessionId, SessionRecord, TurnRecord, backend::AcceptedToken,
    },
    runtime::RuntimeError,
};
use std::{
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};
use tokio::{sync::mpsc, time::Instant};
use tokio_util::sync::CancellationToken;

pub(crate) enum Stage {
    Idle,
    Capturing {
        chunks: u32,
    },
    Prefill {
        queued_at: Instant,
    },
    Generating {
        last_token_at: Instant,
        deadline: Instant,
    },
}
pub(crate) struct SessionState {
    pub record: SessionRecord,
    pub events: mpsc::Sender<SessionEvent>,
    pub cancellation: CancellationToken,
    pub fence: Arc<AtomicU64>,
    pub stage: Stage,
    pub opened: bool,
    pub closing: bool,
    pub in_flight: bool,
    pub pending_token: Option<AcceptedToken>,
    pub context_tokens: usize,
    pub history_bytes: usize,
    pub committed_at: Option<Instant>,
    pub close_reply: Option<tokio::sync::oneshot::Sender<Result<SessionRecord, RuntimeError>>>,
    pub open_reply: Option<tokio::sync::oneshot::Sender<Result<(), RuntimeError>>>,
    pub backend_closed: bool,
    pub chunk_count: u32,
}
impl SessionState {
    pub fn new(
        session_id: SessionId,
        worker_id: usize,
        model_id: String,
        events: mpsc::Sender<SessionEvent>,
        cancellation: CancellationToken,
        fence: Arc<AtomicU64>,
    ) -> Self {
        Self {
            record: SessionRecord {
                session_id,
                worker_id,
                model_id,
                turns: Vec::new(),
            },
            events,
            cancellation,
            fence,
            stage: Stage::Idle,
            opened: false,
            closing: false,
            in_flight: false,
            pending_token: None,
            context_tokens: 0,
            history_bytes: 0,
            committed_at: None,
            close_reply: None,
            open_reply: None,
            backend_closed: false,
            chunk_count: 0,
        }
    }
    pub fn active(&self) -> bool {
        !self.closing && !matches!(self.stage, Stage::Idle)
    }
    pub fn generation(&self) -> u64 {
        self.fence.load(Ordering::Acquire)
    }
    pub fn turn(&self) -> Option<&TurnRecord> {
        self.record.turns.last()
    }
    pub fn turn_mut(&mut self) -> Option<&mut TurnRecord> {
        self.record.turns.last_mut()
    }
    pub fn interrupt(&mut self, next_generation: u64) {
        if self.active() {
            self.finish(FinishReason::Cancelled);
        }
        self.fence.store(next_generation, Ordering::Release);
    }
    pub fn finish(&mut self, reason: FinishReason) {
        let generation = self.generation();
        if let Some(turn) = self.turn_mut() {
            turn.finish_reason = Some(reason);
            let event = SessionEvent::Finished {
                turn_id: turn.turn_id,
                generation,
                reason,
                generated_tokens: turn.tokens.len(),
            };
            let _ = self.events.try_send(event);
        }
        self.stage = Stage::Idle;
    }
    pub fn token_interval(target: f64) -> Duration {
        Duration::from_secs_f64(1.0 / target)
    }
}
