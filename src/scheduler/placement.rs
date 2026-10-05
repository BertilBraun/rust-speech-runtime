use crate::{protocol::WorkerId, scheduler::admission::WorkerStatus};
pub(crate) trait PlacementPolicy: Send {
    fn select_worker(&self, workers: &[WorkerStatus]) -> Option<WorkerId>;
}
pub(crate) struct LeastLoaded;
impl PlacementPolicy for LeastLoaded {
    fn select_worker(&self, workers: &[WorkerStatus]) -> Option<WorkerId> {
        workers
            .iter()
            .filter(|worker| worker.sessions.len() < worker.session_limit)
            .min_by_key(|worker| {
                (
                    worker.sessions.len(),
                    worker.service_time,
                    worker.worker_id.0,
                )
            })
            .map(|worker| worker.worker_id)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::SessionId;
    use std::time::Duration;
    #[test]
    fn placement_respects_each_workers_measured_capacity() {
        let workers = vec![
            WorkerStatus {
                worker_id: WorkerId(0),
                sessions: std::collections::HashSet::from([SessionId(1)]),
                session_limit: 1,
                service_time: Duration::from_millis(20),
            },
            WorkerStatus {
                worker_id: WorkerId(1),
                sessions: std::collections::HashSet::from([SessionId(2)]),
                session_limit: 2,
                service_time: Duration::from_millis(12),
            },
        ];
        assert_eq!(LeastLoaded.select_worker(&workers), Some(WorkerId(1)));
    }
}
