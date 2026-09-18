use super::*;
use core::cell::Cell;

struct Funding {
    calls: Cell<usize>,
    bytes: Cell<usize>,
    refuse_at: usize,
}
impl Funding {
    fn new(refuse_at: usize) -> Self {
        Self {
            calls: Cell::new(0),
            bytes: Cell::new(0),
            refuse_at,
        }
    }
}
impl Allocation for Funding {
    fn reserve(&self, bytes: usize) -> Result<(), AllocationError> {
        let call = self.calls.get();
        self.calls.set(call + 1);
        if call >= self.refuse_at {
            return Err(AllocationError::Refused);
        }
        self.bytes.set(self.bytes.get().checked_add(bytes).unwrap());
        Ok(())
    }
}

#[test]
fn lazy_source_and_cache_refuse_each_original_destination() {
    let builder = Builder::new();
    let funding = Funding::new(usize::MAX);
    let dfa = builder
        .build_many_with_allocations(&[r"([a-z]+):([0-9]+)"], &funding)
        .unwrap();
    for refuse_at in 0..funding.calls.get() {
        let refuse = Funding::new(refuse_at);
        let error = builder
            .build_many_with_allocations(&[r"([a-z]+):([0-9]+)"], &refuse)
            .unwrap_err();
        assert_eq!(error.allocation_error(), Some(AllocationError::Refused));
        assert_eq!(refuse.calls.get(), refuse_at + 1);
    }
    let funding = Funding::new(usize::MAX);
    let cache = Cache::new_with_allocations(&dfa, &funding).unwrap();
    assert!(funding.bytes.get() > 0);
    drop(cache);
    for refuse_at in 0..funding.calls.get() {
        let refuse = Funding::new(refuse_at);
        assert_eq!(
            Cache::new_with_allocations(&dfa, &refuse)
                .unwrap_err()
                .allocation_error(),
            Some(AllocationError::Refused)
        );
        assert_eq!(refuse.calls.get(), refuse_at + 1);
    }
}

#[test]
fn forward_reverse_growth_refusal_and_recovery() {
    use crate::hybrid::regex::{Cache as RegexCache, Regex};
    let re = Regex::new(r"([a-z]+):([0-9]+)|α+").unwrap();
    let input = Input::new("! item:123 then αα");
    let expected = Some(crate::Match::must(0, 2..10));
    let funding = Funding::new(usize::MAX);
    let mut cache = RegexCache::new(&re);
    assert_eq!(
        re.try_search_with_allocations(&mut cache, &input, &funding)
            .unwrap(),
        expected
    );
    assert!(funding.calls.get() > 0);
    // The same actual adaptive tables are reused; no second engine is selected.
    let refuse = Funding::new(0);
    assert_eq!(
        re.try_search_with_allocations(&mut cache, &input, &refuse)
            .unwrap(),
        expected
    );
    assert_eq!(refuse.calls.get(), 0);
    for refuse_at in 0..funding.calls.get() {
        let mut cache = RegexCache::new(&re);
        let refuse = Funding::new(refuse_at);
        let error = re
            .try_search_with_allocations(&mut cache, &input, &refuse)
            .unwrap_err();
        assert_eq!(error.allocation_error(), Some(AllocationError::Refused));
        assert_eq!(refuse.calls.get(), refuse_at + 1);
        // Failed construction can invalidate internal IDs; restart resets that
        // same cache before any table access, while preserving paid capacities.
        assert_eq!(
            re.try_search_with_allocations(&mut cache, &input, &Unenforced)
                .unwrap(),
            expected
        );
    }
}

#[test]
fn refusal_while_clearing_cache_preserves_error_and_restarts() {
    let pattern = r"[01]*1[01]{10}";
    let dfa = Builder::new()
        .configure(Config::new().cache_capacity(4096))
        .build(pattern)
        .unwrap();
    let input = Input::new("01000110110100111000001010111000101111010111010101100111000010100101011111000010110100101001100010101");
    let expected = Some(HalfMatch::must(0, 99));
    let funding = Funding::new(usize::MAX);
    let mut cache = Cache::new(&dfa);
    let actual = dfa
        .try_search_fwd_with_allocations(&mut cache, &input, &funding)
        .unwrap();
    assert_eq!(actual, expected);
    assert!(cache.clear_count() > 0);
    for refuse_at in 0..funding.calls.get() {
        let mut cache = Cache::new(&dfa);
        let refuse = Funding::new(refuse_at);
        let error = dfa
            .try_search_fwd_with_allocations(&mut cache, &input, &refuse)
            .unwrap_err();
        assert_eq!(error.allocation_error(), Some(AllocationError::Refused));
        assert_eq!(refuse.calls.get(), refuse_at + 1);
        assert_eq!(dfa.try_search_fwd(&mut cache, &input).unwrap(), expected);
    }
}

#[test]
fn overlapping_failure_requires_fresh_saved_state() {
    let dfa = Builder::new()
        .configure(Config::new().match_kind(MatchKind::All))
        .build_many(&["a", "ab"])
        .unwrap();
    let input = Input::new("ab");
    let funding = Funding::new(usize::MAX);
    let mut cache = Cache::new(&dfa);
    let mut state = OverlappingState::start();
    dfa.try_search_overlapping_fwd_with_allocations(&mut cache, &input, &mut state, &funding)
        .unwrap();
    assert_eq!(state.get_match(), Some(HalfMatch::must(0, 1)));
    let refuse = Funding::new(0);
    let error = dfa
        .try_search_overlapping_fwd_with_allocations(&mut cache, &input, &mut state, &refuse)
        .unwrap_err();
    assert_eq!(error.allocation_error(), Some(AllocationError::Refused));
    let error = dfa
        .try_search_overlapping_fwd(&mut cache, &input, &mut state)
        .unwrap_err();
    assert_eq!(error.allocation_error(), Some(AllocationError::Refused));
    state = OverlappingState::start();
    dfa.try_search_overlapping_fwd(&mut cache, &input, &mut state)
        .unwrap();
    assert_eq!(state.get_match(), Some(HalfMatch::must(0, 1)));
}
