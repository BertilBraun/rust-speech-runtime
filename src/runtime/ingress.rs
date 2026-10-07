use std::sync::{Arc, atomic::Ordering};

use tokio::sync::{mpsc, oneshot};
use tokio_util::sync::CancellationToken;

use crate::{
    metrics::{Metrics, MetricsSnapshot},
    protocol::{ErrorCode, SessionId},
    session::manager::ManagerCommand,
};

use super::{RuntimeError, SessionHandle};

/// Cloneable admission entry point shared by gateway connections.
///
/// Opening chooses a worker once. Subsequent commands use the returned session handle
/// directly, so audio packets do not pass through the session manager again.
#[derive(Clone)]
pub struct Ingress {
    manager: mpsc::Sender<ManagerCommand>,
    metrics: Arc<Metrics>,
    cancellation: CancellationToken,
}

impl Ingress {
    pub(super) fn new(
        manager: mpsc::Sender<ManagerCommand>,
        metrics: Arc<Metrics>,
        cancellation: CancellationToken,
    ) -> Self {
        Self {
            manager,
            metrics,
            cancellation,
        }
    }

    /// Reserves a sticky worker and waits for its backend cache allocation.
    ///
    /// IDs must contain 1–128 bytes and be unique among live sessions. A full admission
    /// mailbox returns `ChannelSaturated` immediately; worker capacity is checked before
    /// the handle is returned. Dropping the returned handle cancels its session.
    pub async fn open_session(&self, session_id: SessionId) -> Result<SessionHandle, RuntimeError> {
        if session_id.0.is_empty() || session_id.0.len() > 128 {
            return Err(RuntimeError::new(
                ErrorCode::InvalidInput,
                "session ID must contain 1..128 bytes",
            ));
        }

        let (reply, response) = oneshot::channel();
        self.manager
            .try_send(ManagerCommand::Open { session_id, reply })
            .map_err(|error| {
                self.metrics.saturation.fetch_add(1, Ordering::Relaxed);
                let code = match error {
                    mpsc::error::TrySendError::Closed(_) => ErrorCode::Shutdown,
                    mpsc::error::TrySendError::Full(_) => ErrorCode::ChannelSaturated,
                };
                RuntimeError::new(code, "session manager unavailable")
            })?;

        tokio::select! {
            _ = self.cancellation.cancelled() => {
                Err(RuntimeError::new(ErrorCode::Shutdown, "node stopped"))
            }
            result = response => {
                result.map_err(|_| {
                    RuntimeError::new(ErrorCode::Shutdown, "session manager stopped")
                })?
            }
        }
    }

    /// Takes a metrics snapshot without acquiring or mutating worker scheduling state.
    pub fn metrics(&self) -> MetricsSnapshot {
        self.metrics.snapshot()
    }
}
