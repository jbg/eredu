use super::{Allocation, AllocationError};
use crate::{AhoCorasick, AhoCorasickKind, MatchKind, StartKind};
use core::cell::Cell;

struct Policy {
    calls: Cell<usize>,
    refuse: usize,
}
impl Policy {
    fn new(refuse: usize) -> Self {
        Self { calls: Cell::new(0), refuse }
    }
}
impl Allocation for Policy {
    fn reserve(&self, bytes: usize) -> Result<(), AllocationError> {
        assert!(bytes > 0);
        let call = self.calls.get();
        self.calls.set(call + 1);
        if call == self.refuse {
            Err(AllocationError::Refused)
        } else {
            Ok(())
        }
    }
}

#[test]
fn every_selected_automaton_producer_refuses_without_engine_fallback() {
    let patterns = ["hers", "he", "she", "His", "a", "aa", "aaa"];
    for kind in [
        None,
        Some(AhoCorasickKind::NoncontiguousNFA),
        Some(AhoCorasickKind::ContiguousNFA),
        Some(AhoCorasickKind::DFA),
    ] {
        for semantics in [
            MatchKind::Standard,
            MatchKind::LeftmostFirst,
            MatchKind::LeftmostLongest,
        ] {
            for prefilter in [false, true] {
                let mut builder = AhoCorasick::builder();
                builder
                    .kind(kind)
                    .match_kind(semantics)
                    .prefilter(prefilter)
                    .ascii_case_insensitive(true)
                    .start_kind(StartKind::Both);
                let policy = Policy::new(usize::MAX);
                let paid =
                    builder.build_with_allocations(patterns, &policy).unwrap();
                let ordinary = builder.build(patterns).unwrap();
                assert_eq!(paid.kind(), ordinary.kind());
                assert_eq!(
                    paid.find("His hers aaa"),
                    ordinary.find("His hers aaa")
                );
                for refuse in 0..policy.calls.get() {
                    let refused = Policy::new(refuse);
                    let error = builder
                        .build_with_allocations(patterns, &refused)
                        .unwrap_err();
                    assert_eq!(
                        error.allocation_error(),
                        Some(AllocationError::Refused)
                    );
                    assert_eq!(
                        refused.calls.get(),
                        refuse + 1,
                        "construction continued after refusal"
                    );
                }
            }
        }
    }
}

#[test]
fn internal_prefilter_sources_and_packed_searchers_refuse_each_producer() {
    for patterns in
        [&["shared needle"][..], &["sam", "samwise", "Sam", "maple"][..]]
    {
        let builder = AhoCorasick::builder();
        let policy = Policy::new(usize::MAX);
        let paid = builder.build_with_allocations(patterns, &policy).unwrap();
        assert_eq!(
            paid.find("xx samwise shared needle"),
            builder.build(patterns).unwrap().find("xx samwise shared needle")
        );
        for refuse in 0..policy.calls.get() {
            let policy = Policy::new(refuse);
            assert_eq!(
                builder
                    .build_with_allocations(patterns, &policy)
                    .unwrap_err()
                    .allocation_error(),
                Some(AllocationError::Refused)
            );
            assert_eq!(policy.calls.get(), refuse + 1);
        }
    }
    for kind in [
        crate::packed::MatchKind::LeftmostFirst,
        crate::packed::MatchKind::LeftmostLongest,
    ] {
        let construct = |policy: &Policy| {
            let mut builder =
                crate::packed::Config::new().match_kind(kind).builder();
            builder.extend_with_allocations(
                ["sam", "samwise", "Sam", "maple"],
                policy,
            )?;
            builder.build_with_allocations(policy)
        };
        let policy = Policy::new(usize::MAX);
        let searcher = construct(&policy).unwrap();
        if let Some(searcher) = searcher {
            assert!(searcher.find("samwise and maple").is_some());
        }
        for refuse in 0..policy.calls.get() {
            let policy = Policy::new(refuse);
            assert_eq!(
                construct(&policy).unwrap_err(),
                AllocationError::Refused
            );
            assert_eq!(policy.calls.get(), refuse + 1);
        }
    }
}

#[test]
fn refused_pattern_add_keeps_builder_reusable_and_precedence_intact() {
    let mut builder = crate::packed::Builder::new();
    builder.add("sam");
    let policy = Policy::new(0);
    assert_eq!(
        builder.add_with_allocations("samwise", &policy).unwrap_err(),
        AllocationError::Refused
    );
    assert_eq!(builder.len(), 1);
    builder.add("samwise");
    if let Some(searcher) = builder.build() {
        assert_eq!(searcher.find("samwise").unwrap().pattern().as_usize(), 0);
    }
}
