use super::Prefilter;
use crate::util::{
    allocation::{Allocation, AllocationError},
    search::{MatchKind, Span},
};
use alloc::{format, vec, vec::Vec};
use core::cell::Cell;
struct Policy {
    calls: Cell<usize>,
    refuse: usize,
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
fn original_choices_keep_match_semantics_and_stop_at_each_refusal() {
    let mut cases = vec![
        vec![b"x".to_vec()],
        vec![b"x".to_vec(), b"y".to_vec()],
        vec![b"x".to_vec(), b"y".to_vec(), b"z".to_vec()],
        vec![b"needle".to_vec()],
        vec![b"sam".to_vec(), b"samwise".to_vec()],
    ];
    cases.push(
        (0..131)
            .map(|i| format!("prefix-{i:03}").into_bytes())
            .collect::<Vec<_>>(),
    );
    cases.push(
        (0..501)
            .map(|i| format!("prefix-{i:03}").into_bytes())
            .collect::<Vec<_>>(),
    );
    for needles in cases {
        for kind in [MatchKind::LeftmostFirst, MatchKind::All] {
            let policy = Policy {
                calls: Cell::new(0),
                refuse: usize::MAX,
            };
            let paid = Prefilter::new_with_allocations(kind, &needles, &policy).unwrap();
            let ordinary = Prefilter::new(kind, &needles);
            assert_eq!(paid.is_some(), ordinary.is_some());
            if let (Some(paid), Some(ordinary)) = (paid, ordinary) {
                assert_eq!(paid.pre.name(), ordinary.pre.name());
                for haystack in [
                    b"samwise prefix-130 xyz needle".as_slice(),
                    b"prefix-500",
                    b"",
                ] {
                    let span = Span::from(0..haystack.len());
                    assert_eq!(paid.find(haystack, span), ordinary.find(haystack, span));
                    assert_eq!(paid.prefix(haystack, span), ordinary.prefix(haystack, span));
                }
                let calls = policy.calls.get();
                let alias = paid.clone();
                drop(paid);
                drop(alias);
                assert_eq!(
                    policy.calls.get(),
                    calls,
                    "alias created another source shell"
                );
            }
            for refuse in 0..policy.calls.get() {
                let refusal = Policy {
                    calls: Cell::new(0),
                    refuse,
                };
                assert_eq!(
                    Prefilter::new_with_allocations(kind, &needles, &refusal).unwrap_err(),
                    AllocationError::Refused
                );
                assert_eq!(
                    refusal.calls.get(),
                    refuse + 1,
                    "optional engine fallback after refusal"
                );
            }
        }
    }
}

#[test]
fn retained_prefilter_census_preserves_aliases_and_scoped_searches() {
    use alloc::collections::BTreeMap;
    let mut cases = vec![
        vec![b"x".to_vec()],
        vec![b"x".to_vec(), b"y".to_vec()],
        vec![b"x".to_vec(), b"y".to_vec(), b"z".to_vec()],
        vec![b"needle".to_vec()],
        vec![b"sam".to_vec(), b"samwise".to_vec()],
    ];
    cases.push(
        (0..131)
            .map(|i| format!("prefix-{i:03}").into_bytes())
            .collect(),
    );
    cases.push(
        (0..501)
            .map(|i| format!("prefix-{i:03}").into_bytes())
            .collect(),
    );
    for needles in cases {
        for kind in [MatchKind::LeftmostFirst, MatchKind::All] {
            let Some(prefilter) = Prefilter::new(kind, &needles) else {
                continue;
            };
            let mut seen = BTreeMap::new();
            let mut total = 0;
            prefilter
                .visit_source_storage(&mut |id: *const (), bytes| {
                    assert!(bytes > 0);
                    if let Some(prior) = seen.insert(id as usize, bytes) {
                        assert_eq!(prior, bytes);
                        false
                    } else {
                        total += bytes;
                        true
                    }
                })
                .unwrap();
            let shell = crate::util::source_storage::arc_bytes(&*prefilter.pre).unwrap();
            assert!(total >= shell);
            if prefilter.pre.name() == "memmem" {
                assert_eq!(total, shell + needles[0].len());
                assert_eq!(
                    seen.len(),
                    2,
                    "owned needle backing must be visited below the shared shell"
                );
            }
            let alias = prefilter.clone();
            alias
                .visit_source_storage(&mut |id: *const (), bytes| {
                    assert_eq!(seen.get(&(id as usize)), Some(&bytes));
                    false
                })
                .unwrap();
            let haystack = b"samwise prefix-130 xyz needle";
            prefilter.find(haystack, Span::from(0..haystack.len()));
            let mut after = BTreeMap::new();
            prefilter
                .visit_source_storage(&mut |id: *const (), bytes| {
                    if let Some(prior) = after.insert(id as usize, bytes) {
                        assert_eq!(prior, bytes);
                        false
                    } else {
                        true
                    }
                })
                .unwrap();
            assert_eq!(seen, after);
        }
    }
}
