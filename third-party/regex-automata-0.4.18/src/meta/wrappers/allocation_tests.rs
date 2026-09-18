use super::*;
use crate::util::allocation::Unenforced;
use crate::util::allocation::AllocationError;
use core::cell::Cell;
struct Funding { calls: Cell<usize>, limit: usize }
impl Allocation for Funding {
    fn reserve(&self, _: usize) -> Result<(), AllocationError> {
        let call = self.calls.get(); self.calls.set(call + 1);
        if call < self.limit { Ok(()) } else { Err(AllocationError::Refused) }
    }
}
fn funding(limit: usize) -> Funding { Funding { calls: Cell::new(0), limit } }

#[test]
fn optional_engine_allocation_refusal_cannot_become_an_optional_miss() {
    use crate::meta::Config;
    use crate::nfa::thompson::{Compiler, Config as NfaConfig, WhichCaptures};
    let hir = crate::util::syntax::parse("(a+)(b+)").unwrap();
    let info = RegexInfo::new_with_allocations(Config::new(), &[&hir], &Unenforced).unwrap();
    let nfa = Compiler::new().build_from_hir(&hir).unwrap();
    let reverse = Compiler::new().configure(NfaConfig::new().reverse(true).which_captures(WhichCaptures::None)).build_from_hir(&hir).unwrap();
    #[cfg(feature = "dfa-onepass")]
    {
        let paid = funding(usize::MAX);
        assert!(OnePass::new_with_allocations(&info, &nfa, &paid).unwrap().0.is_some());
        assert!(paid.calls.get() > 0);
        for limit in 0..paid.calls.get() {
            let refuse = funding(limit);
            assert_eq!(OnePass::new_with_allocations(&info, &nfa, &refuse).unwrap_err().allocation_error(), Some(AllocationError::Refused));
            assert_eq!(refuse.calls.get(), limit + 1);
        }
    }
    #[cfg(feature = "dfa-build")]
    {
        let paid = funding(usize::MAX);
        assert!(DFA::new_with_allocations(&info, None, &nfa, &reverse, &paid).unwrap().is_some());
        assert!(paid.calls.get() > 0);
        for limit in 0..paid.calls.get() {
            let refuse = funding(limit);
            assert_eq!(DFA::new_with_allocations(&info, None, &nfa, &reverse, &refuse).unwrap_err().allocation_error(), Some(AllocationError::Refused));
            assert_eq!(refuse.calls.get(), limit + 1);
        }
    }
}

#[test]
fn pike_wrapper_preserves_refusal_and_reuses_original_cache() {
    let hir = crate::util::syntax::parse("(a+)(b+)").unwrap();
    let info = RegexInfo::new_with_allocations(crate::meta::Config::new(), &[&hir], &Unenforced).unwrap();
    let nfa = NFA::compiler().build_from_hir(&hir).unwrap();
    let vm = PikeVM::new(&info, None, &nfa).unwrap();
    let input = Input::new("!aabb!");
    let mut cache = vm.create_cache(); let mut slots = [None; 6];
    let paid = funding(usize::MAX);
    assert_eq!(vm.get().search_slots_with_allocations(&mut cache, &input, &mut slots, &paid).unwrap(), Some(PatternID::ZERO));
    assert_eq!(slots.map(|v| v.map(|v| v.get())), [Some(1), Some(5), Some(1), Some(3), Some(3), Some(5)]);
    for limit in 0..paid.calls.get() {
        let refuse = funding(limit); let mut cache = vm.create_cache();
        let error = vm.get().search_slots_with_allocations(&mut cache, &input, &mut slots, &refuse).unwrap_err();
        assert_eq!(error.allocation_error(), Some(AllocationError::Refused));
        assert_eq!(refuse.calls.get(), limit + 1);
        assert_eq!(vm.get().search_slots_with_allocations(&mut cache, &input, &mut slots, &crate::util::allocation::Unenforced).unwrap(), Some(PatternID::ZERO));
    }
}
