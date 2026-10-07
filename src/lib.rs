//! Bounded turn-based speech serving with sticky model workers and persistent caches.
//!
//! [`Node`] owns worker tasks; clone its [`Ingress`] to admit conversations. Each
//! [`SessionHandle`] remains assigned to one worker and streams bounded events.
//! [`transport::Gateway`] exposes the same lifecycle to ordinary WebSocket clients.
//!
//! Use [`SessionHandle::prepare`] during endpoint confirmation to move candidate
//! inference ahead of commit without exposing provisional text. A matching commit
//! activates the candidate; resumed audio invalidates it. Explicit close returns
//! the bounded conversation record for archival, separately from the GPU cache.
//!
//! ```no_run
//! use voice_scheduler::{Node, config::RuntimeConfig, protocol::SessionId};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let node = Node::start(RuntimeConfig::default()).await?;
//! let mut session = node.ingress().open_session(SessionId("conversation".into())).await?;
//! let record = session.close().await?;
//! assert!(record.turns.is_empty());
//! node.shutdown().await?;
//! # Ok(())
//! # }
//! ```

pub mod config;
pub mod metrics;
pub mod protocol;
mod runtime;
mod scheduler;
mod session;
pub mod simulation;
pub mod transport;
mod worker;

pub use runtime::{Ingress, Node, RuntimeError, SessionHandle};
