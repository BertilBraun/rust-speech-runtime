use crate::{
    protocol::{
        FinishReason, SessionEvent, SessionId, SessionRecord, TurnRecord, backend::AcceptedToken,
    },
    runtime::RuntimeError,
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct CaptureSnapshot {
    pub turn_id: crate::protocol::TurnId,
    pub generation: u64,
    pub chunk_count: u32,
    pub sample_count: usize,
}

#[derive(Clone, Copy)]
pub(crate) struct PreparationRequest {
    pub snapshot: CaptureSnapshot,
    pub queued_at: Instant,
}

pub(crate) enum Preparation {
    None,
    Queued(PreparationRequest),
    Running(CaptureSnapshot),
    Ready(CaptureSnapshot),
    DiscardPending { next: Option<PreparationRequest> },
    Discarding { next: Option<PreparationRequest> },
}

impl Preparation {
    /// Queue changed audio after any in-flight provisional cache has been discarded.
    pub fn queue(&mut self, request: PreparationRequest) {
        *self = match self {
            Self::None | Self::Queued(_) => Self::Queued(request),
            Self::Discarding { .. } => Self::Discarding {
                next: Some(request),
            },
            Self::Running(_) | Self::Ready(_) | Self::DiscardPending { .. } => {
                Self::DiscardPending {
                    next: Some(request),
                }
            }
        };
    }

    pub fn invalidate(&mut self) {
        *self = match self {
            Self::None | Self::Queued(_) => Self::None,
            Self::Discarding { .. } => Self::Discarding { next: None },
            Self::Running(_) | Self::Ready(_) | Self::DiscardPending { .. } => {
                Self::DiscardPending { next: None }
            }
        };
    }

    pub fn snapshot(&self) -> Option<CaptureSnapshot> {
        match self {
            Self::Queued(request) => Some(request.snapshot),
            Self::Running(snapshot) | Self::Ready(snapshot) => Some(*snapshot),
            Self::DiscardPending { next } | Self::Discarding { next } => {
                next.map(|request| request.snapshot)
            }
            Self::None => None,
        }
    }
}

pub(crate) struct SessionState {
    pub record: SessionRecord,
    pub events: mpsc::Sender<SessionEvent>,
    pub cancellation: CancellationToken,
    pub generation: u64,
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
    pub preparation: Preparation,
}

impl SessionState {
    pub fn new(
        session_id: SessionId,
        worker_id: usize,
        model_id: String,
        events: mpsc::Sender<SessionEvent>,
        cancellation: CancellationToken,
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
            generation: 0,
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
            preparation: Preparation::None,
        }
    }

    pub fn has_active_turn(&self) -> bool {
        !self.closing && !matches!(self.stage, Stage::Idle)
    }

    /// A disconnected record remains owned until its last operation and cache release finish.
    pub fn can_remove_after_disconnect(&self) -> bool {
        self.closing
            && !self.in_flight
            && (!self.opened || self.backend_closed)
            && self.events.is_closed()
    }

    pub fn current_turn(&self) -> Option<&TurnRecord> {
        self.record.turns.last()
    }

    pub fn current_turn_mut(&mut self) -> Option<&mut TurnRecord> {
        self.record.turns.last_mut()
    }

    pub fn interrupt(&mut self, next_generation: u64) {
        self.preparation.invalidate();
        if self.has_active_turn() {
            self.finish_turn(FinishReason::Cancelled);
        }
        self.generation = next_generation;
    }

    pub fn finish_turn(&mut self, reason: FinishReason) {
        let generation = self.generation;
        if let Some(turn) = self.current_turn_mut() {
            turn.finish_reason = Some(reason);
            let event = SessionEvent::Finished {
                turn_id: turn.turn_id,
                generation,
                reason,
                generated_tokens: turn.tokens.len(),
            };
            if self.events.try_send(event).is_err() {
                self.current_turn_mut()
                    .expect("finished turn")
                    .finish_reason = Some(FinishReason::SlowConsumer);
                self.cancellation.cancel();
            }
        }
        self.stage = Stage::Idle;
    }
}
