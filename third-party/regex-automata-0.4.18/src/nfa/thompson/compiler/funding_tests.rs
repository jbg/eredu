use super::*;
use crate::{
    nfa::thompson::{pikevm::PikeVM, WhichCaptures},
    util::{
        captures::{Captures, GroupInfo},
        primitives::PatternID,
    },
    Match,
};
use alloc::{format, string::ToString};
use core::cell::Cell;

struct Funding {
    calls: Cell<usize>,
    fail: Option<usize>,
}
impl Funding {
    fn new(fail: Option<usize>) -> Self {
        Self {
            calls: Cell::new(0),
            fail,
        }
    }
}
impl Allocation for Funding {
    fn reserve(&self, bytes: usize) -> Result<(), AllocationError> {
        assert!(bytes > 0);
        let call = self.calls.get();
        self.calls.set(call + 1);
        if self.fail == Some(call) {
            Err(AllocationError::Refused)
        } else {
            Ok(())
        }
    }
}

#[test]
fn original_thompson_growth_refuses_each_reached_forward_reverse_and_literal_producer() {
    let hir = regex_syntax::Parser::new()
        .parse(r"(?P<choice>abc|xyz|ab)(?:[\u0080-\u08FF\U00010000-\U000103FF]){1,3}")
        .unwrap();
    for (reverse, shrink) in [(false, false), (true, false), (true, true)] {
        let config = Config::new()
            .reverse(reverse)
            .shrink(shrink)
            .which_captures(if reverse {
                WhichCaptures::None
            } else {
                WhichCaptures::All
            });
        let funding = Funding::new(None);
        let nfa = Compiler::new()
            .configure(config.clone())
            .build_from_hir_with_allocations(&hir, &funding)
            .unwrap();
        let ordinary = Compiler::new()
            .configure(config.clone())
            .build_from_hir(&hir)
            .unwrap();
        assert_eq!(format!("{nfa:?}"), format!("{ordinary:?}"));
        assert!(funding.calls.get() > 20);
        for fail in 0..funding.calls.get() {
            let refused = Funding::new(Some(fail));
            let mut compiler = Compiler::new();
            compiler.configure(config.clone());
            let error = compiler
                .build_from_hir_with_allocations(&hir, &refused)
                .unwrap_err();
            assert_eq!(
                error.allocation_error(),
                Some(AllocationError::Refused),
                "reverse={reverse}, shrink={shrink}, request={fail}: {error}"
            );
            assert_eq!(refused.calls.get(), fail + 1);
        }
    }
}

#[test]
fn funded_original_nfa_keeps_literal_priority_and_named_capture_results() {
    let funding = Funding::new(None);
    let nfa = Compiler::new()
        .build_many_with_allocations(&[r"(?P<name>sam|samwise|sage)(?P<tail>[0-9]+)"], &funding)
        .unwrap();
    let vm = PikeVM::new_from_nfa(nfa).unwrap();
    let mut cache = vm.create_cache();
    let mut captures =
        Captures::all_with_allocations(vm.get_nfa().group_info().clone(), &funding).unwrap();
    vm.captures(&mut cache, "!samwise27", &mut captures);
    assert_eq!(captures.get_match(), Some(Match::must(0, 1..10)));
    assert_eq!(captures.get_group_by_name("name"), Some((1..8).into()));
    assert_eq!(captures.get_group_by_name("tail"), Some((8..10).into()));
    let copy = captures.clone_with_allocations(&funding).unwrap();
    captures.clear();
    assert_eq!(copy.get_match(), Some(Match::must(0, 1..10)));
    captures
        .clone_from_with_allocations(&copy, &Funding::new(Some(0)))
        .unwrap();
    assert_eq!(captures.get_match(), copy.get_match());
}

#[test]
fn capture_name_tables_and_slot_copies_preserve_exact_fixed_refusals() {
    let rows = [
        [None, Some("first"), Some("second")],
        [None, Some("second"), None],
    ];
    let funding = Funding::new(None);
    let groups = GroupInfo::new_with_allocations(rows, &funding).unwrap();
    assert_eq!(groups.to_index(PatternID::ZERO, "second"), Some(2));
    assert_eq!(groups.to_index(PatternID::must(1), "second"), Some(1));
    assert_eq!(groups.to_name(PatternID::ZERO, 1), Some("first"));
    for fail in 0..funding.calls.get() {
        let refused = Funding::new(Some(fail));
        let error = GroupInfo::new_with_allocations(rows, &refused).unwrap_err();
        assert_eq!(error.allocation_error(), Some(AllocationError::Refused));
        assert_eq!(refused.calls.get(), fail + 1);
    }
    let refused = Funding::new(Some(0));
    assert_eq!(
        Captures::all_with_allocations(groups.clone(), &refused).unwrap_err(),
        AllocationError::Refused
    );
    let captures = Captures::matches(groups);
    assert_eq!(
        captures
            .clone_with_allocations(&Funding::new(Some(0)))
            .unwrap_err(),
        AllocationError::Refused
    );
    let error = GroupInfo::new_with_allocations([[None, Some("same"), Some("same")]], &funding)
        .unwrap_err();
    assert_eq!(error.allocation_error(), None);
    assert!(error
        .to_string()
        .contains("duplicate capture group name 'same'"));
}

#[test]
fn hir_ancestry_uses_paid_work_storage_on_a_small_thread_stack() {
    let mut hir = Hir::literal(b"a".as_slice());
    for index in 1..=12_000 {
        hir = Hir::capture(hir::Capture {
            index,
            name: None,
            sub: alloc::boxed::Box::new(hir),
        });
    }
    std::thread::Builder::new()
        .stack_size(256 * 1024)
        .spawn(move || {
            let funding = Funding::new(None);
            let nfa = Compiler::new()
                .configure(Config::new().which_captures(WhichCaptures::All))
                .build_from_hir_with_allocations(&hir, &funding)
                .unwrap();
            assert!(nfa.states().len() > 24_000);
            assert!(funding.calls.get() > 0);
        })
        .unwrap()
        .join()
        .unwrap();
}

#[test]
fn fixed_nfa_constructors_pay_original_graphs_and_preserve_refusals() {
    for (build, patterns) in [
        (
            NFA::always_match_with_allocations as fn(&dyn Allocation) -> Result<NFA, BuildError>,
            1,
        ),
        (NFA::never_match_with_allocations, 0),
    ] {
        let funding = Funding::new(None);
        let nfa = build(&funding).unwrap();
        assert_eq!(nfa.pattern_len(), patterns);
        for fail in 0..funding.calls.get() {
            let refused = Funding::new(Some(fail));
            assert_eq!(
                build(&refused).unwrap_err().allocation_error(),
                Some(AllocationError::Refused)
            );
            assert_eq!(refused.calls.get(), fail + 1);
        }
    }
}
