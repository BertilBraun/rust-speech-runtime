use crate::protocol::WorkerId;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct CacheHandle {
    worker_id: WorkerId,
    slot: u32,
}

pub(crate) struct CachePool {
    worker_id: WorkerId,
    free_slots: Vec<u32>,
    allocated: Vec<bool>,
}

impl CachePool {
    pub(crate) fn new(worker_id: WorkerId, capacity: usize) -> Self {
        Self {
            worker_id,
            free_slots: (0..capacity as u32).rev().collect(),
            allocated: vec![false; capacity],
        }
    }

    pub(crate) fn allocate(&mut self) -> Option<CacheHandle> {
        let slot = self.free_slots.pop()?;
        assert!(!self.allocated[slot as usize]);
        self.allocated[slot as usize] = true;
        Some(CacheHandle {
            worker_id: self.worker_id,
            slot,
        })
    }

    pub(crate) fn free(&mut self, handle: CacheHandle) {
        self.assert_owned(handle);
        self.allocated[handle.slot as usize] = false;
        self.free_slots.push(handle.slot);
    }

    pub(crate) fn assert_owned(&self, handle: CacheHandle) {
        assert_eq!(
            handle.worker_id, self.worker_id,
            "cache belongs to another worker"
        );
        assert!(
            self.allocated[handle.slot as usize],
            "cache slot is not allocated"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slots_are_bounded_and_reusable() {
        let mut pool = CachePool::new(WorkerId(0), 1);
        let handle = pool.allocate().unwrap();
        assert_eq!(pool.allocate(), None);
        pool.free(handle);
        assert_eq!(pool.allocate(), Some(handle));
    }

    #[test]
    #[should_panic(expected = "cache belongs to another worker")]
    fn handles_cannot_cross_workers() {
        let handle = CachePool::new(WorkerId(0), 1).allocate().unwrap();
        CachePool::new(WorkerId(1), 1).assert_owned(handle);
    }
}
