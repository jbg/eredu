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
fn original_onepass_destinations_refuse_before_later_work() {
    // NFA source construction is tested separately by the Thompson producer.
    for pattern in [r"([a-z]+):([0-9]+)", r"(α+)([0-9]+)", r"a*[ab]", r"(^|$)a"] {
        let nfa = NFA::new(pattern).unwrap();
        let builder = Builder::new();
        let expected = builder.build_from_nfa(nfa.clone());
        let funding = Funding::new(usize::MAX);
        let actual = builder.build_from_nfa_with_allocations(nfa.clone(), &funding);
        assert_eq!(
            expected
                .as_ref()
                .map(|_| ())
                .map_err(|e| alloc::format!("{e}")),
            actual
                .as_ref()
                .map(|_| ())
                .map_err(|e| alloc::format!("{e}"))
        );
        let requests = funding.calls.get();
        assert!(requests > 0 && funding.bytes.get() > 0);
        drop(actual);
        for refuse_at in 0..requests {
            let funding = Funding::new(refuse_at);
            let error = builder
                .build_from_nfa_with_allocations(nfa.clone(), &funding)
                .unwrap_err();
            assert_eq!(error.allocation_error(), Some(AllocationError::Refused));
            assert_eq!(funding.calls.get(), refuse_at + 1);
        }
    }
}

#[test]
fn caller_owned_slots_preserve_capture_results_and_reuse() {
    let re = DFA::new(r"([a-z]+):([0-9]+)").unwrap();
    let refuse = Funding::new(0);
    assert_eq!(
        Cache::new_with_allocations(&re, &refuse).unwrap_err(),
        AllocationError::Refused
    );
    let funding = Funding::new(usize::MAX);
    let mut cache = Cache::new_with_allocations(&re, &funding).unwrap();
    let mut slots = [None; 6];
    let input = Input::new("item:123").anchored(Anchored::Yes);
    assert_eq!(
        re.try_search_slots(&mut cache, &input, &mut slots).unwrap(),
        Some(PatternID::ZERO)
    );
    assert_eq!(
        slots.map(|x| x.map(|x| x.get())),
        [Some(0), Some(8), Some(0), Some(4), Some(5), Some(8)]
    );
    cache.reset_with_allocations(&re, &refuse).unwrap();
    assert_eq!(
        refuse.calls.get(),
        1,
        "reused slots did not request another allocation"
    );
}

#[test]
fn empty_multipattern_auxiliary_slots_are_prospectively_funded() {
    let dfa = Builder::new().build_many(&["", "z"]).unwrap();
    let mut cache = dfa.create_cache();
    let input = Input::new("é").anchored(Anchored::Yes);
    let refuse = Funding::new(0);
    assert_eq!(dfa.try_search_slots_with_allocations(&mut cache, &input, &mut [], &refuse).unwrap_err().allocation_error(), Some(AllocationError::Refused));
    assert_eq!(dfa.try_search_slots(&mut cache, &input, &mut []).unwrap(), Some(PatternID::ZERO));
}
