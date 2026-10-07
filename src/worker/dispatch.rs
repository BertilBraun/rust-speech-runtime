//! Dispatches validated mailbox commands to focused lifecycle and turn handlers.

use super::{Command, actor::WorkerActor};
use std::sync::atomic::Ordering;

impl WorkerActor {
    pub fn handle_command(&mut self, command: Command) {
        match command {
            Command::Open {
                session_key,
                session_id,
                events,
                cancellation,
                reply,
            } => self.open_session(session_key, session_id, events, cancellation, reply),
            Command::Begin {
                session_key,
                turn_id,
                reply,
            } => {
                let result = self.begin_turn(&session_key, turn_id);
                if result.is_err() {
                    self.metrics.rejected_turns.fetch_add(1, Ordering::Relaxed);
                }
                let _ = reply.send(result);
            }
            Command::Audio {
                session_key,
                turn_id,
                chunk_index,
                audio,
                reply,
            } => {
                let result = self.append_audio_chunk(&session_key, turn_id, chunk_index, &audio);
                let _ = reply.send(result);
            }
            Command::Prepare {
                session_key,
                turn_id,
                chunk_count,
                sample_count,
                reply,
            } => {
                let result = self.prepare_turn(&session_key, turn_id, chunk_count, sample_count);
                let _ = reply.send(result);
            }
            Command::Commit {
                session_key,
                turn_id,
                final_chunk_count,
                sample_count,
                reply,
            } => {
                let result =
                    self.commit_turn(&session_key, turn_id, final_chunk_count, sample_count);
                let _ = reply.send(result);
            }
            Command::Cancel {
                session_key,
                turn_id,
                reply,
            } => {
                let result = self.cancel_turn(&session_key, turn_id);
                let _ = reply.send(result);
            }
            Command::Close { session_key, reply } => self.close_session(&session_key, reply),
        }
    }
}
