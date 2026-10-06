//! Realtime voice scheduling with bounded actor mailboxes and worker-local cache state.

pub mod config;
pub mod metrics;
pub mod protocol;
mod scheduler;
mod session;
pub mod simulation;
pub mod transport;
mod worker;

mod runtime;

pub use runtime::{Ingress, Node, RuntimeError};
