use super::{Command, actor::WorkerActor};
use crate::{
    protocol::{ErrorCode, SessionEvent, TurnId, TurnRecord},
    runtime::RuntimeError,
    session::state::{Preparation, SessionState, Stage},
};
use std::sync::atomic::Ordering;

impl WorkerActor {
    pub fn command(&mut self, command: Command) {
        match command {
            Command::Open {
                key,
                session_id,
                events,
                cancellation,
                reply,
            } => {
                if !self.available.load(Ordering::Acquire)
                    || self.sessions.len() >= self.config.max_sessions_per_worker
                {
                    let _ = reply.send(Err(error(
                        ErrorCode::CapacityExceeded,
                        "worker session capacity reached",
                    )));
                } else {
                    let mut session = SessionState::new(
                        session_id,
                        self.id,
                        self.ready.model_id.clone(),
                        events,
                        cancellation,
                    );
                    session.open_reply = Some(reply);
                    self.sessions.insert(key, session);
                    self.load.store(self.sessions.len(), Ordering::Relaxed);
                    self.metrics.active_sessions.fetch_add(1, Ordering::Relaxed);
                }
            }
            Command::Begin {
                key,
                turn_id,
                reply,
            } => {
                let result = self.begin(&key, turn_id);
                if result.is_err() {
                    self.metrics.rejected_turns.fetch_add(1, Ordering::Relaxed);
                }
                let _ = reply.send(result);
            }
            Command::Audio {
                key,
                turn_id,
                chunk_index,
                audio,
                reply,
            } => {
                let result = self.audio(&key, turn_id, chunk_index, &audio);
                let _ = reply.send(result);
            }
            Command::Prepare {
                key,
                turn_id,
                chunk_count,
                sample_count,
                reply,
            } => {
                let result = self.prepare(&key, turn_id, chunk_count, sample_count);
                let _ = reply.send(result);
            }
            Command::Commit {
                key,
                turn_id,
                final_chunk_count,
                sample_count,
                reply,
            } => {
                let result = self.commit(&key, turn_id, final_chunk_count, sample_count);
                let _ = reply.send(result);
            }
            Command::Cancel {
                key,
                turn_id,
                reply,
            } => {
                let generation = self.generation();
                let result = self.session(&key).and_then(|session| {
                    check_turn(session, turn_id)?;
                    if session.active() {
                        session.interrupt(generation);
                    }
                    Ok(())
                });
                let _ = reply.send(result);
            }
            Command::Close { key, reply } => {
                let generation = self.generation();
                if let Some(session) = self.sessions.get_mut(&key) {
                    if !session.closing {
                        session.interrupt(generation);
                        session.closing = true;
                        self.metrics.active_sessions.fetch_sub(1, Ordering::Relaxed);
                    }
                    session.close_reply = reply;
                    if (!session.opened || session.backend_closed) && !session.in_flight {
                        self.remove_closed(&key);
                    }
                } else if let Some(reply) = reply {
                    let _ = reply.send(Err(error(
                        ErrorCode::SessionNotFound,
                        "session unavailable",
                    )));
                }
            }
        }
    }
    pub(super) fn session(&mut self, key: &str) -> Result<&mut SessionState, RuntimeError> {
        if !self.available.load(Ordering::Acquire) {
            return Err(error(
                ErrorCode::BackendUnavailable,
                "worker connection failed",
            ));
        }
        self.sessions
            .get_mut(key)
            .filter(|session| !session.closing)
            .ok_or_else(|| error(ErrorCode::SessionNotFound, "session unavailable"))
    }
    fn begin(&mut self, key: &str, turn_id: TurnId) -> Result<(), RuntimeError> {
        if !self.available.load(Ordering::Acquire) {
            return Err(error(
                ErrorCode::BackendUnavailable,
                "worker connection failed",
            ));
        }
        let active = self
            .sessions
            .values()
            .filter(|session| session.active())
            .count();
        let previous = self
            .sessions
            .get(key)
            .ok_or_else(|| error(ErrorCode::SessionNotFound, "session unavailable"))?;
        if previous
            .record
            .turns
            .iter()
            .any(|turn| turn.turn_id == turn_id)
        {
            return Err(error(
                ErrorCode::InvalidInput,
                "turn ID must be unique within session",
            ));
        }
        let reserved = active + usize::from(!previous.active());
        let reserve_prefill = self
            .sessions
            .values()
            .any(|session| session.active() && matches!(session.stage, Stage::Generating { .. }));
        let context = self
            .sessions
            .values()
            .filter(|session| session.active())
            .map(|session| session.context_tokens)
            .chain(std::iter::once(previous.context_tokens))
            .max()
            .unwrap_or(0);
        if reserved > self.config.max_active_turns_per_worker
            || (!previous.active()
                && !self.costs.admits(
                    reserved,
                    self.config.max_batch_size.min(self.ready.max_batch_size),
                    context,
                    &self.config,
                    reserve_prefill,
                ))
        {
            return Err(error(
                ErrorCode::CapacityExceeded,
                "worker cannot reserve another active turn at target token rate",
            ));
        }
        if previous.history_bytes.saturating_add(1024) > self.config.max_history_bytes {
            return Err(error(
                ErrorCode::HistoryLimit,
                "session record limit reached",
            ));
        }
        if previous.context_tokens.saturating_add(32)
            >= self
                .config
                .max_context_tokens
                .min(self.ready.max_context_tokens)
        {
            return Err(error(
                ErrorCode::ContextLimit,
                "conversation context exhausted",
            ));
        }
        let generation = self.generation();
        let session = self.session(key)?;
        session.interrupt(generation);
        session.record.turns.push(TurnRecord {
            turn_id,
            audio_pcm16: Vec::new(),
            tokens: Vec::new(),
            finish_reason: None,
            committed: false,
        });
        session.history_bytes += 1024;
        session.stage = Stage::Capturing { chunks: 0 };
        session.committed_at = None;
        session.chunk_count = 0;
        if session
            .events
            .try_send(SessionEvent::Accepted { turn_id })
            .is_err()
        {
            session.finish(crate::protocol::FinishReason::SlowConsumer);
            session.cancellation.cancel();
            session.closing = true;
            self.metrics.active_sessions.fetch_sub(1, Ordering::Relaxed);
            self.metrics.saturation.fetch_add(1, Ordering::Relaxed);
            return Err(error(ErrorCode::SlowConsumer, "session event queue full"));
        }
        self.metrics.admitted_turns.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
    fn audio(
        &mut self,
        key: &str,
        turn_id: TurnId,
        chunk_index: u32,
        audio: &[u8],
    ) -> Result<(), RuntimeError> {
        if audio.is_empty() || !audio.len().is_multiple_of(2) {
            return Err(error(
                ErrorCode::InvalidInput,
                "audio must contain nonempty PCM16 samples",
            ));
        }
        let audio_limit = self
            .config
            .max_audio_samples
            .min(self.ready.max_audio_samples)
            * 2;
        let history_limit = self.config.max_history_bytes;
        let generation = self.generation();
        let session = self.session(key)?;
        check_turn(session, turn_id)?;
        let chunks = match session.stage {
            Stage::Capturing { chunks } => chunks,
            _ => {
                return Err(error(
                    ErrorCode::InvalidState,
                    "audio only accepted during capture",
                ));
            }
        };
        if chunk_index != chunks || chunks == u32::MAX {
            return Err(error(
                ErrorCode::InvalidInput,
                "audio chunk index must be contiguous",
            ));
        }
        let length = session.turn().expect("capturing turn").audio_pcm16.len();
        if length.saturating_add(audio.len()) > audio_limit {
            return Err(error(
                ErrorCode::InvalidInput,
                "utterance duration limit exceeded",
            ));
        }
        if session.history_bytes.saturating_add(audio.len()) > history_limit {
            return Err(error(
                ErrorCode::HistoryLimit,
                "session record limit exceeded",
            ));
        }
        if !matches!(session.preparation, Preparation::None) {
            session.preparation.invalidate();
            session.generation = generation;
        }
        session
            .turn_mut()
            .expect("capturing turn")
            .audio_pcm16
            .extend_from_slice(audio);
        session.history_bytes += audio.len();
        session.chunk_count = chunks + 1;
        session.stage = Stage::Capturing { chunks: chunks + 1 };
        Ok(())
    }
    fn commit(
        &mut self,
        key: &str,
        turn_id: TurnId,
        final_chunk_count: u32,
        sample_count: usize,
    ) -> Result<(), RuntimeError> {
        let context_limit = self
            .config
            .max_context_tokens
            .min(self.ready.max_context_tokens);
        let session = self.session(key)?;
        check_turn(session, turn_id)?;
        let turn = session.turn().expect("current turn");
        if turn.committed {
            return if turn.audio_pcm16.len() / 2 == sample_count
                && session.chunk_count == final_chunk_count
            {
                Ok(())
            } else {
                Err(error(
                    ErrorCode::InvalidInput,
                    "duplicate commit changed chunk/sample counts",
                ))
            };
        }
        let chunks = match session.stage {
            Stage::Capturing { chunks } => chunks,
            _ => return Err(error(ErrorCode::InvalidState, "turn not capturing")),
        };
        if chunks != final_chunk_count
            || sample_count == 0
            || turn.audio_pcm16.len() / 2 != sample_count
        {
            return Err(error(
                ErrorCode::InvalidInput,
                "commit chunk/sample counts mismatch",
            ));
        }
        let audio_tokens = sample_count.div_ceil(1600);
        if session.context_tokens.saturating_add(audio_tokens + 32) >= context_limit {
            return Err(error(
                ErrorCode::ContextLimit,
                "utterance exceeds remaining context",
            ));
        }
        session.turn_mut().expect("current turn").committed = true;
        let now = tokio::time::Instant::now();
        session.committed_at = Some(now);
        session.stage = Stage::Prefill { queued_at: now };
        Ok(())
    }
    pub fn remove_closed(&mut self, key: &str) {
        if let Some(mut session) = self.sessions.remove(key) {
            let _ = session.events.try_send(SessionEvent::Closed {
                session_id: session.record.session_id.clone(),
            });
            if let Some(reply) = session.close_reply.take() {
                let _ = reply.send(Ok(session.record));
            }
            self.load.store(self.sessions.len(), Ordering::Relaxed);
        }
    }
}
fn error(code: ErrorCode, message: &str) -> RuntimeError {
    RuntimeError::new(code, message)
}
pub(super) fn check_turn(session: &SessionState, turn_id: TurnId) -> Result<(), RuntimeError> {
    if session.turn().is_some_and(|turn| turn.turn_id == turn_id) {
        Ok(())
    } else {
        Err(error(
            ErrorCode::TurnNotFound,
            "turn ID does not match current turn",
        ))
    }
}
