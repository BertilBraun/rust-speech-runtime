//! Boundary timings separate scheduling work from backend execution and IPC.

use serde::{Deserialize, Serialize};

use super::{Distribution, LatencyDistribution};

/// Elapsed boundary timings; socket waits and IPC are not CPU execution time.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RuntimeTimingSnapshot {
    /// Applying a worker mailbox command; excludes queue wait and reply delivery.
    pub command_processing: LatencyDistribution,
    /// Selecting eligible sessions, building metadata and packing audio; no backend await.
    pub batch_build: LatencyDistribution,
    /// Applying returned proposals, updating state and enqueuing output; no socket await.
    pub completion_processing: LatencyDistribution,
    /// Response ready in the execution task until the actor starts handling it.
    pub completion_handoff: LatencyDistribution,
    /// Prepared batch until the execution task starts its RPC.
    pub execution_handoff: LatencyDistribution,
    /// Paired RPC duration minus Python engine duration, including both sides' IPC work.
    pub backend_rpc_overhead: LatencyDistribution,
    pub websocket_serialization: LatencyDistribution,
    /// Socket send/flush, including OS waits or client backpressure.
    pub websocket_write: LatencyDistribution,
}

#[derive(Default)]
pub(crate) struct RuntimeTiming {
    pub command_processing: Distribution,
    pub batch_build: Distribution,
    pub completion_processing: Distribution,
    pub completion_handoff: Distribution,
    pub execution_handoff: Distribution,
    pub backend_rpc_overhead: Distribution,
    pub websocket_serialization: Distribution,
    pub websocket_write: Distribution,
}

impl RuntimeTiming {
    pub fn snapshot(&self) -> RuntimeTimingSnapshot {
        RuntimeTimingSnapshot {
            command_processing: self.command_processing.snapshot(),
            batch_build: self.batch_build.snapshot(),
            completion_processing: self.completion_processing.snapshot(),
            completion_handoff: self.completion_handoff.snapshot(),
            execution_handoff: self.execution_handoff.snapshot(),
            backend_rpc_overhead: self.backend_rpc_overhead.snapshot(),
            websocket_serialization: self.websocket_serialization.snapshot(),
            websocket_write: self.websocket_write.snapshot(),
        }
    }
}
