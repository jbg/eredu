use super::*;
use core::cell::Cell;

struct Gate {
    calls: Cell<usize>,
    fail: usize,
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
fn sweep(mut build: impl FnMut(&dyn Allocation) -> Result<OwnedDFA, BuildError>) {
    let pass = Gate {
        calls: Cell::new(0),
        fail: usize::MAX,
    };
    let dfa = build(&pass).unwrap();
    assert!(pass.calls.get() > 0);
    for fail in 0..pass.calls.get() {
        let gate = Gate {
            calls: Cell::new(0),
            fail,
        };
        let error = build(&gate).expect_err("every original producer must propagate refusal");
        assert_eq!(
            Some(AllocationError::Refused),
            error.allocation_error(),
            "request {fail}"
        );
        assert_eq!(
            fail + 1,
            gate.calls.get(),
            "no allocation callback after refusal"
        );
        assert!(
            !error.is_size_limit_exceeded(),
            "resource refusal cannot select a fallback engine"
        );
    }
    drop(dfa);
}

#[test]
fn every_original_dense_producer_refuses_before_growth() {
    let patterns = [r"(?-u:\b[a-z]+)z", r"(?:ab|a|bc){1,3}", r"[é-ê]+", r"^foo$"];
    for minimize in [false, true] {
        for accelerate in [false, true] {
            let mut builder = Builder::new();
            builder.configure(
                Config::new()
                    .minimize(minimize)
                    .accelerate(accelerate)
                    .starts_for_each_pattern(true),
            );
            sweep(|funding| builder.build_many_with_allocations(&patterns, funding));
        }
    }
}

#[test]
fn admitted_dense_semantics_and_fixed_graphs() {
    for minimize in [false, true] {
        let mut builder = Builder::new();
        builder.configure(Config::new().minimize(minimize).match_kind(MatchKind::All));
        let gate = Gate {
            calls: Cell::new(0),
            fail: usize::MAX,
        };
        let dfa = builder
            .build_many_with_allocations(&["a+", "ab+", "x"], &gate)
            .unwrap();
        assert_eq!(
            Some(crate::HalfMatch::must(1, 5)),
            dfa.try_search_fwd(&crate::Input::new("zaaab")).unwrap()
        );
        assert_eq!(None, dfa.try_search_fwd(&crate::Input::new("zzz")).unwrap());
    }
    sweep(|funding| OwnedDFA::always_match_with_allocations(funding));
    sweep(|funding| OwnedDFA::never_match_with_allocations(funding));
}

#[test]
fn dense_refusal_keeps_original_nfa_reusable() {
    let nfa = thompson::NFA::new("(?-u:[a-z]+)[0-9]{1,3}").unwrap();
    let before = alloc::format!("{nfa:?}");
    let mut builder = Builder::new();
    builder.configure(Config::new().minimize(true));
    sweep(|funding| builder.build_from_nfa_with_allocations(&nfa, funding));
    assert_eq!(before, alloc::format!("{nfa:?}"));
}
