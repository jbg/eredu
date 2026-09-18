use super::*;
use core::cell::Cell;
struct Gate {
    calls: Cell<usize>,
    fail: usize,
}
impl Gate {
    fn pass() -> Self {
        Self {
            calls: Cell::new(0),
            fail: usize::MAX,
        }
    }
    fn at(fail: usize) -> Self {
        Self {
            calls: Cell::new(0),
            fail,
        }
    }
}
impl Allocation for Gate {
    fn reserve(&self, bytes: usize) -> Result<(), AllocationError> {
        assert!(bytes > 0);
        let call = self.calls.get();
        self.calls.set(call + 1);
        if call == self.fail {
            Err(AllocationError::Refused)
        } else {
            Ok(())
        }
    }
}

#[test]
fn pike_cache_original_construction_and_reset_refusals() {
    let re = PikeVM::new(r"(?P<a>ab|a|αβ)+(?P<b>b|β)?").unwrap();
    let pass = Gate::pass();
    Cache::new_with_allocations(&re, &pass).unwrap();
    for fail in 0..pass.calls.get() {
        let gate = Gate::at(fail);
        assert!(matches!(
            Cache::new_with_allocations(&re, &gate),
            Err(AllocationError::Refused)
        ));
        assert_eq!(fail + 1, gate.calls.get());
    }
    let pass = Gate::pass();
    Cache::empty().reset_with_allocations(&re, &pass).unwrap();
    for fail in 0..pass.calls.get() {
        let gate = Gate::at(fail);
        let mut cache = Cache::empty();
        assert_eq!(
            Err(AllocationError::Refused),
            cache.reset_with_allocations(&re, &gate)
        );
        cache.reset_with_allocations(&re, &Unenforced).unwrap();
        assert_eq!(
            Some(PatternID::ZERO),
            re.search_slots(&mut cache, &Input::new("αβab"), &mut [])
        );
    }
}

#[test]
fn pike_epsilon_refusal_restores_every_pending_capture() {
    let re = PikeVM::new(r"((a?)|(?P<n>b?)|((c?)))d").unwrap();
    instrument!(|c| c.reset(re.get_nfa()));
    let width = re.get_nfa().group_info().slot_len();
    let original = vec![NonMaxUsize::new(19); width];
    let pass = Gate::pass();
    let mut cache = Cache::new(&re);
    let mut slots = original.clone();
    re.epsilon_closure(
        &mut cache.stack,
        &mut slots,
        &mut cache.curr,
        &Input::new("bd"),
        0,
        re.nfa.start_anchored(),
        Allocator::new(&pass),
    )
    .unwrap();
    assert_eq!(original, slots);
    assert!(pass.calls.get() > 1);
    for fail in 0..pass.calls.get() {
        let gate = Gate::at(fail);
        let mut cache = Cache::new(&re);
        let mut slots = original.clone();
        let error = re
            .epsilon_closure(
                &mut cache.stack,
                &mut slots,
                &mut cache.curr,
                &Input::new("bd"),
                0,
                re.nfa.start_anchored(),
                Allocator::new(&gate),
            )
            .unwrap_err();
        assert_eq!(Some(AllocationError::Refused), error.allocation_error());
        assert_eq!(original, slots, "restore at refusal {fail}");
        assert!(cache.stack.is_empty());
        assert_eq!(fail + 1, gate.calls.get());
    }
}

#[test]
fn pike_search_refusal_then_same_cache_recovery() {
    for patterns in [
        &[r"(?P<a>ab|a|αβ)+(?P<b>b|β)?", r"z+"][..],
        &[r"a*", r"(?-u:\B)"][..],
    ] {
        let re = PikeVM::new_many(patterns).unwrap();
        for input in [Input::new("!αβabb"), Input::new("a☃z").range(1..5)] {
            for width in [
                0,
                re.nfa.group_info().implicit_slot_len(),
                re.nfa.group_info().slot_len(),
                re.nfa.group_info().slot_len() + 7,
            ] {
                let mut ordinary = vec![None; width];
                let expected = re.search_slots(&mut Cache::new(&re), &input, &mut ordinary);
                let pass = Gate::pass();
                let mut slots = vec![None; width];
                let actual = re
                    .search_slots_with_allocations(&mut Cache::new(&re), &input, &mut slots, &pass)
                    .unwrap();
                assert_eq!((expected, &ordinary), (actual, &slots));
                for fail in 0..pass.calls.get() {
                    let gate = Gate::at(fail);
                    let mut cache = Cache::new(&re);
                    let mut slots = vec![None; width];
                    let error = re
                        .search_slots_with_allocations(&mut cache, &input, &mut slots, &gate)
                        .unwrap_err();
                    assert_eq!(Some(AllocationError::Refused), error.allocation_error());
                    assert_eq!(fail + 1, gate.calls.get());
                    assert!(cache.stack.is_empty());
                    let recovered = re.search_slots(&mut cache, &input, &mut slots);
                    assert_eq!((expected, &ordinary), (recovered, &slots));
                }
            }
        }
    }
}

#[test]
fn pike_overlapping_uses_the_same_fallible_epsilon_worker() {
    let re = PikeVM::builder()
        .configure(Config::new().match_kind(MatchKind::All))
        .build_many(&["a+", "ab+", "b+"])
        .unwrap();
    let input = Input::new("aaabb");
    let pass = Gate::pass();
    let mut patterns = PatternSet::new(3);
    re.which_overlapping_matches_with_allocations(
        &mut Cache::new(&re),
        &input,
        &mut patterns,
        &pass,
    )
    .unwrap();
    assert_eq!(3, patterns.len());
    for fail in 0..pass.calls.get() {
        let gate = Gate::at(fail);
        let mut cache = Cache::new(&re);
        let error = re
            .which_overlapping_matches_with_allocations(
                &mut cache,
                &input,
                &mut PatternSet::new(3),
                &gate,
            )
            .unwrap_err();
        assert_eq!(Some(AllocationError::Refused), error.allocation_error());
        assert!(cache.stack.is_empty());
        assert_eq!(fail + 1, gate.calls.get());
    }
}

#[cfg(feature = "nfa-backtrack")]
#[test]
fn backtracking_refuses_each_visited_stack_and_utf8_destination() {
    use crate::nfa::thompson::backtrack::{BoundedBacktracker, Cache};
    for patterns in [
        &[r"(?P<a>ab|a|αβ)+(?P<b>b|β)?", r"z+"][..],
        &[r"a*", r"(?-u:\B)"][..],
    ] {
        let re = BoundedBacktracker::new_many(patterns).unwrap();
        for input in [Input::new("!αβabb"), Input::new("a☃z").range(1..5)] {
            for width in [
                0,
                re.get_nfa().group_info().implicit_slot_len(),
                re.get_nfa().group_info().slot_len(),
            ] {
                let mut ordinary = vec![None; width];
                let expected = re
                    .try_search_slots(&mut Cache::new(&re), &input, &mut ordinary)
                    .unwrap();
                let pass = Gate::pass();
                let mut slots = vec![None; width];
                let actual = re
                    .try_search_slots_with_allocations(
                        &mut Cache::new(&re),
                        &input,
                        &mut slots,
                        &pass,
                    )
                    .unwrap();
                assert_eq!((expected, &ordinary), (actual, &slots));
                for fail in 0..pass.calls.get() {
                    let gate = Gate::at(fail);
                    let mut cache = Cache::new(&re);
                    let mut slots = vec![None; width];
                    let error = re
                        .try_search_slots_with_allocations(&mut cache, &input, &mut slots, &gate)
                        .unwrap_err();
                    assert_eq!(Some(AllocationError::Refused), error.allocation_error());
                    assert_eq!(fail + 1, gate.calls.get());
                    assert!(slots.iter().all(Option::is_none));
                    let recovered = re.try_search_slots(&mut cache, &input, &mut slots).unwrap();
                    assert_eq!((expected, &ordinary), (recovered, &slots));
                }
            }
        }
    }
}

#[test]
fn oversized_slot_buffer_uses_only_original_group_rows() {
    let re = PikeVM::new(r"(a)|(bc)").unwrap();
    let used = re.get_nfa().group_info().slot_len();
    let sentinel = NonMaxUsize::new(99);
    let mut slots = vec![sentinel; used + 7];
    let mut cache = Cache::new(&re);
    let policy = Gate::pass();
    assert_eq!(Some(PatternID::ZERO), re.search_slots_with_allocations(&mut cache, &Input::new("!bc"), &mut slots, &policy).unwrap());
    assert_eq!(slots[0].map(|n| n.get()), Some(1));
    assert_eq!(slots[1].map(|n| n.get()), Some(3));
    assert!(slots[used..].iter().all(|&value| value == sentinel));
    let mut exact = vec![None; used];
    assert_eq!(Some(PatternID::ZERO), re.search_slots(&mut cache, &Input::new("a"), &mut exact));
    assert_eq!(exact[0].map(|n| n.get()), Some(0));
    assert_eq!(exact[1].map(|n| n.get()), Some(1));
}
