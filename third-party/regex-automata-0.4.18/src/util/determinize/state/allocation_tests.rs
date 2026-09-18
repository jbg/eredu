use super::*;
use core::cell::Cell;

struct Funding { calls: Cell<usize>, refuse_at: usize }
impl Allocation for Funding {
    fn reserve(&self, _: usize) -> Result<(), AllocationError> {
        let call = self.calls.get();
        self.calls.set(call + 1);
        if call >= self.refuse_at { Err(AllocationError::Refused) } else { Ok(()) }
    }
}

fn emit(funding: &dyn Allocation) -> Result<State, AllocationError> {
    let mut matches = StateBuilderEmpty::new().into_matches_with_allocations(funding)?;
    for pid in [0, 17, 1_000_000] {
        matches.add_match_pattern_id_with_allocations(PatternID::must(pid), funding)?;
    }
    let mut nfa = matches.into_nfa();
    for sid in [1, 9000, 0, 123_000_000, 7] {
        nfa.add_nfa_state_id_with_allocations(StateID::must(sid), funding)?;
    }
    let nfa = nfa.clone_with_allocations(funding)?;
    let state = nfa.to_state_with_allocations(funding)?;
    assert_eq!(state.match_pattern_ids_with_allocations(funding)?.unwrap(),
        [PatternID::ZERO, PatternID::must(17), PatternID::must(1_000_000)]);
    let mut ids = Vec::new();
    state.iter_nfa_state_ids(|id| ids.push(id.as_usize()));
    assert_eq!(ids, [1, 9000, 0, 123_000_000, 7]);
    Ok(state)
}

#[test]
fn compact_state_emissions_refuse_each_reached_destination() {
    let funding = Funding { calls: Cell::new(0), refuse_at: usize::MAX };
    drop(emit(&funding).unwrap());
    for refuse_at in 0..funding.calls.get() {
        let funding = Funding { calls: Cell::new(0), refuse_at };
        assert_eq!(emit(&funding).unwrap_err(), AllocationError::Refused);
        assert_eq!(funding.calls.get(), refuse_at + 1);
    }
}

#[test]
fn refused_compact_extension_preserves_header_and_delta_base() {
    let mut matches = StateBuilderEmpty::new().into_matches();
    matches.add_match_pattern_id(PatternID::ZERO);
    let refuse = Funding { calls: Cell::new(0), refuse_at: 0 };
    assert_eq!(matches.add_match_pattern_id_with_allocations(PatternID::must(3), &refuse), Err(AllocationError::Refused));
    let mut nfa = matches.into_nfa();
    assert_eq!(nfa.to_state().match_pattern_ids(), Some(alloc::vec![PatternID::ZERO]));
    assert_eq!(nfa.add_nfa_state_id_with_allocations(StateID::must(1000), &refuse), Err(AllocationError::Refused));
    nfa.add_nfa_state_id(StateID::must(2));
    let mut ids = Vec::new();
    nfa.to_state().iter_nfa_state_ids(|id| ids.push(id.as_usize()));
    assert_eq!(ids, [2]);
}
