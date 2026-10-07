//! Dispatches validated mailbox commands to focused lifecycle and turn handlers.

use super::{Command, actor::WorkerActor};
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
            } => self.open_session(key, session_id, events, cancellation, reply),
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
                let result = self.cancel_turn(&key, turn_id);
                let _ = reply.send(result);
            }
            Command::Close { key, reply } => self.close_session(&key, reply),
        }
    }
}
