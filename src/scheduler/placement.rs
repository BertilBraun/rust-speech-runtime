use crate::protocol::WorkerId;
use crate::scheduler::admission::has_capacity;

pub trait PlacementPolicy: Send {
    fn select_worker(&self, loads: &[usize], limit: usize) -> Option<WorkerId>;
}

pub struct LeastLoaded;

impl PlacementPolicy for LeastLoaded {
    fn select_worker(&self, loads: &[usize], limit: usize) -> Option<WorkerId> {
        loads
            .iter()
            .enumerate()
            .filter(|(_, load)| has_capacity(**load, limit))
            .min_by_key(|(index, load)| (**load, *index))
            .map(|(index, _)| WorkerId(index))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selects_least_loaded_with_stable_ties() {
        assert_eq!(LeastLoaded.select_worker(&[3, 1, 1], 4), Some(WorkerId(1)));
        assert_eq!(LeastLoaded.select_worker(&[4, 4], 4), None);
    }
}
