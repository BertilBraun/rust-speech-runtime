pub fn has_capacity(active_sessions: usize, limit: usize) -> bool {
    active_sessions < limit
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capacity_boundary() {
        assert!(has_capacity(51, 52));
        assert!(!has_capacity(52, 52));
        assert!(!has_capacity(53, 52));
    }
}
