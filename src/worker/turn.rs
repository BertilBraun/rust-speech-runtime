//! Capture validation and turn admission; no backend I/O runs here.

use super::actor::WorkerActor;
use crate::{
    protocol::{ErrorCode, SessionEvent, TurnId, TurnRecord},
    runtime::RuntimeError,
    session::state::{Preparation, SessionState, Stage},
};
use std::sync::atomic::Ordering;

impl WorkerActor {
    pub(super) fn cancel_turn(
        &mut self,
        session_key: &str,
        turn_id: TurnId,
    ) -> Result<(), RuntimeError> {
        let generation = self.allocate_generation_id();
        let session = self.session_mut(session_key)?;
        check_turn(session, turn_id)?;
        if session.has_active_turn() {
            session.interrupt(generation);
        }
        Ok(())
    }

    pub(super) fn begin_turn(
        &mut self,
        session_key: &str,
        turn_id: TurnId,
    ) -> Result<(), RuntimeError> {
        self.validate_turn_start(session_key, turn_id)?;
        let generation = self.allocate_generation_id();
        let session = self.session_mut(session_key)?;
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
            session.finish_turn(crate::protocol::FinishReason::SlowConsumer);
            session.cancellation.cancel();
            session.closing = true;
            self.metrics.active_sessions.fetch_sub(1, Ordering::Relaxed);
            self.metrics.saturation.fetch_add(1, Ordering::Relaxed);
            return Err(RuntimeError::new(
                ErrorCode::SlowConsumer,
                "session event queue full",
            ));
        }
        self.metrics.admitted_turns.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    pub(super) fn append_audio_chunk(
        &mut self,
        session_key: &str,
        turn_id: TurnId,
        chunk_index: u32,
        audio: &[u8],
    ) -> Result<(), RuntimeError> {
        if audio.is_empty() || !audio.len().is_multiple_of(2) {
            return Err(RuntimeError::new(
                ErrorCode::InvalidInput,
                "audio must contain nonempty PCM16 samples",
            ));
        }
        let audio_limit = self
            .configuration
            .max_audio_samples
            .min(self.backend_capabilities.max_audio_samples)
            * 2;
        let history_limit = self.configuration.max_history_bytes;
        let generation = self.allocate_generation_id();
        let session = self.session_mut(session_key)?;
        check_turn(session, turn_id)?;
        let chunks = validate_audio_append(
            session,
            chunk_index,
            audio.len(),
            audio_limit,
            history_limit,
        )?;

        if !matches!(session.preparation, Preparation::None) {
            session.preparation.invalidate();
            session.generation = generation;
        }
        session
            .current_turn_mut()
            .expect("capturing turn")
            .audio_pcm16
            .extend_from_slice(audio);
        session.history_bytes += audio.len();
        session.chunk_count = chunks + 1;
        session.stage = Stage::Capturing { chunks: chunks + 1 };
        Ok(())
    }

    pub(super) fn commit_turn(
        &mut self,
        session_key: &str,
        turn_id: TurnId,
        final_chunk_count: u32,
        sample_count: usize,
    ) -> Result<(), RuntimeError> {
        let context_limit = self.context_token_limit();
        let session = self.session_mut(session_key)?;
        check_turn(session, turn_id)?;
        let turn = session.current_turn().expect("current turn");
        if turn.committed {
            return if turn.audio_pcm16.len() / 2 == sample_count
                && session.chunk_count == final_chunk_count
            {
                Ok(())
            } else {
                Err(RuntimeError::new(
                    ErrorCode::InvalidInput,
                    "duplicate commit changed chunk/sample counts",
                ))
            };
        }
        validate_new_commit(session, final_chunk_count, sample_count, context_limit)?;

        session.current_turn_mut().expect("current turn").committed = true;
        let now = tokio::time::Instant::now();
        session.committed_at = Some(now);
        session.stage = Stage::Prefill { queued_at: now };
        Ok(())
    }
}

pub(super) fn check_turn(session: &SessionState, turn_id: TurnId) -> Result<(), RuntimeError> {
    if session
        .current_turn()
        .is_some_and(|turn| turn.turn_id == turn_id)
    {
        Ok(())
    } else {
        Err(RuntimeError::new(
            ErrorCode::TurnNotFound,
            "turn ID does not match current turn",
        ))
    }
}

fn validate_audio_append(
    session: &SessionState,
    chunk_index: u32,
    audio_bytes: usize,
    audio_limit: usize,
    history_limit: usize,
) -> Result<u32, RuntimeError> {
    let chunks = match session.stage {
        Stage::Capturing { chunks } => chunks,
        _ => {
            return Err(RuntimeError::new(
                ErrorCode::InvalidState,
                "audio only accepted during capture",
            ));
        }
    };
    if chunk_index != chunks || chunks == u32::MAX {
        return Err(RuntimeError::new(
            ErrorCode::InvalidInput,
            "audio chunk index must be contiguous",
        ));
    }
    let length = session
        .current_turn()
        .expect("capturing turn")
        .audio_pcm16
        .len();
    if length.saturating_add(audio_bytes) > audio_limit {
        return Err(RuntimeError::new(
            ErrorCode::InvalidInput,
            "utterance duration limit exceeded",
        ));
    }
    if session.history_bytes.saturating_add(audio_bytes) > history_limit {
        return Err(RuntimeError::new(
            ErrorCode::HistoryLimit,
            "session record limit exceeded",
        ));
    }
    Ok(chunks)
}

fn validate_new_commit(
    session: &SessionState,
    final_chunk_count: u32,
    sample_count: usize,
    context_limit: usize,
) -> Result<(), RuntimeError> {
    let chunks = match session.stage {
        Stage::Capturing { chunks } => chunks,
        _ => {
            return Err(RuntimeError::new(
                ErrorCode::InvalidState,
                "turn not capturing",
            ));
        }
    };
    if chunks != final_chunk_count
        || sample_count == 0
        || session
            .current_turn()
            .expect("capturing turn")
            .audio_pcm16
            .len()
            / 2
            != sample_count
    {
        return Err(RuntimeError::new(
            ErrorCode::InvalidInput,
            "commit chunk/sample counts mismatch",
        ));
    }
    let audio_tokens = sample_count.div_ceil(1600);
    if session.context_tokens.saturating_add(audio_tokens + 32) >= context_limit {
        return Err(RuntimeError::new(
            ErrorCode::ContextLimit,
            "utterance exceeds remaining context",
        ));
    }
    Ok(())
}
