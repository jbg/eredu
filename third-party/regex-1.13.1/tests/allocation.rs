use regex::{
    allocation::{Allocation, AllocationError, Unenforced},
    Error,
};
use std::{cell::Cell, collections::BTreeMap, sync::Arc};

struct Meter {
    calls: Cell<usize>,
    bytes: Cell<usize>,
    refuse: usize,
}
impl Meter {
    fn new(refuse: usize) -> Self {
        Self {
            calls: Cell::new(0),
            bytes: Cell::new(0),
            refuse,
        }
    }
}
impl Allocation for Meter {
    fn reserve(&self, bytes: usize) -> Result<(), AllocationError> {
        let call = self.calls.get();
        assert!(
            call <= self.refuse,
            "producer continued after its first refusal"
        );
        self.calls.set(call + 1);
        if call == self.refuse {
            return Err(AllocationError::Refused);
        }
        self.bytes.set(self.bytes.get().checked_add(bytes).unwrap());
        Ok(())
    }
}
fn sweep(build: impl Fn(&dyn Allocation) -> Result<(), Error>) {
    let meter = Meter::new(usize::MAX);
    build(&meter).unwrap();
    assert!(meter.calls.get() > 0);
    for refuse in 0..meter.calls.get() {
        let meter = Meter::new(refuse);
        let error = build(&meter).unwrap_err();
        assert_eq!(error, Error::Allocation(AllocationError::Refused));
        assert_eq!(meter.calls.get(), refuse + 1);
        #[cfg(feature = "std")]
        assert!(std::error::Error::source(&error)
            .unwrap()
            .is::<AllocationError>());
    }
}

#[test]
fn every_builder_destination_refuses_without_retry() {
    for pattern in [
        r"^([a-z]+)[0-9]{2,6}$",
        r"alpha|beta|gamma|delta",
        r"[a-z]+z$",
    ] {
        sweep(|payer| {
            regex::RegexBuilder::new_with_allocations(pattern, payer)?
                .build_with_allocations(payer)?;
            Ok(())
        });
        sweep(|payer| {
            regex::bytes::RegexBuilder::new_with_allocations(pattern, payer)?
                .build_with_allocations(payer)?;
            Ok(())
        });
    }
    let patterns = [r"^foo", r"[0-9]{2,4}$", r"\x7F", r"foo|bar"];
    sweep(|payer| {
        regex::RegexSetBuilder::new_with_allocations(patterns, payer)?
            .build_with_allocations(payer)?;
        Ok(())
    });
    sweep(|payer| {
        regex::bytes::RegexSetBuilder::new_with_allocations(patterns, payer)?
            .build_with_allocations(payer)?;
        Ok(())
    });
}

#[test]
fn diagnostics_and_size_limits_preserve_original_outcomes() {
    for pattern in ["(", "[z-a]", "a{3,1}", r"(?=a)", r"(a)\1"] {
        let ordinary = regex::Regex::new(pattern).unwrap_err();
        let payer = Meter::new(usize::MAX);
        let actual = regex::RegexBuilder::new_with_allocations(pattern, &payer)
            .unwrap()
            .build_with_allocations(&payer)
            .unwrap_err();
        assert_eq!(format!("{actual}"), format!("{ordinary}"));
        assert_eq!(format!("{actual:?}"), format!("{ordinary:?}"));
        for refuse in 0..payer.calls.get() {
            let payer = Meter::new(refuse);
            let actual = regex::RegexBuilder::new_with_allocations(pattern, &payer)
                .and_then(|b| b.build_with_allocations(&payer))
                .unwrap_err();
            assert_eq!(actual, Error::Allocation(AllocationError::Refused));
            assert_eq!(payer.calls.get(), refuse + 1);
        }
    }
    let ordinary = regex::RegexBuilder::new("[a-z]{1000}")
        .size_limit(1)
        .build()
        .unwrap_err();
    let paid = regex::RegexBuilder::new_with_allocations("[a-z]{1000}", &Unenforced)
        .unwrap()
        .size_limit(1)
        .build_with_allocations(&Unenforced)
        .unwrap_err();
    assert_eq!(ordinary, paid);
    assert!(matches!(paid, Error::CompiledTooBig(1)));
}

fn census(regex: &regex::Regex) -> BTreeMap<usize, usize> {
    let mut allocations = BTreeMap::new();
    regex
        .visit_source_storage(&mut |identity: *const (), bytes| match allocations
            .entry(identity as usize)
        {
            std::collections::btree_map::Entry::Vacant(entry) => {
                entry.insert(bytes);
                true
            }
            std::collections::btree_map::Entry::Occupied(entry) => {
                assert_eq!(*entry.get(), bytes);
                false
            }
        })
        .unwrap();
    allocations
}

#[test]
fn scoped_cache_refusal_recovery_and_source_retirement() {
    for pattern in [r"^[a-z][a-z0-9_-]{1,32}$", r"a.*b[0-9]+", r"[a-z]+z$"] {
        let source = Arc::new(regex::Regex::new(pattern).unwrap());
        let equal = Arc::new(regex::Regex::new(pattern).unwrap());
        let before = census(&source);
        let meter = Meter::new(usize::MAX);
        let mut cache =
            regex::OwnedSearchWorkspace::new_with_allocations(source.clone(), &meter).unwrap();
        assert!(cache.matches_source(&source));
        assert!(!cache.matches_source(&equal));
        let creation_calls = meter.calls.get();
        for refuse in 0..creation_calls {
            let payer = Meter::new(refuse);
            assert_eq!(
                regex::OwnedSearchWorkspace::new_with_allocations(source.clone(), &payer)
                    .unwrap_err()
                    .allocation_error(),
                Some(AllocationError::Refused)
            );
            assert_eq!(payer.calls.get(), refuse + 1);
        }
        for input in ["a_field_42", "a field b42", "abcdefz", "none", ""] {
            // Use a distinct equivalent source for the ordinary pooled oracle.
            let expected = equal.is_match(input);
            let mut fresh =
                regex::OwnedSearchWorkspace::new_with_allocations(source.clone(), &Unenforced)
                    .unwrap();
            let payer = Meter::new(usize::MAX);
            assert_eq!(fresh.is_match(input, &payer).unwrap(), expected);
            for refuse in 0..payer.calls.get() {
                let mut refused =
                    regex::OwnedSearchWorkspace::new_with_allocations(source.clone(), &Unenforced)
                        .unwrap();
                let payer = Meter::new(refuse);
                assert_eq!(
                    refused
                        .is_match(input, &payer)
                        .unwrap_err()
                        .allocation_error(),
                    Some(AllocationError::Refused)
                );
                assert_eq!(payer.calls.get(), refuse + 1);
                assert_eq!(refused.is_match(input, &Unenforced).unwrap(), expected);
            }
            assert_eq!(cache.is_match(input, &meter).unwrap(), expected);
            let calls = meter.calls.get();
            assert_eq!(cache.is_match(input, &meter).unwrap(), expected);
            let repeated = meter.calls.get() - calls;
            assert_eq!(cache.is_match(input, &meter).unwrap(), expected);
            // Minimal PikeVM charges its fixed search controls each time;
            // repeated input never requires additional backing growth.
            assert_eq!(meter.calls.get() - calls, repeated * 2);
        }
        assert_eq!(before, census(&source));
        let weak = Arc::downgrade(&source);
        drop(source);
        assert!(weak.upgrade().is_some());
        drop(cache);
        assert!(weak.upgrade().is_none());
    }
}

#[test]
fn source_census_deduplicates_all_four_wrapper_aliases() {
    macro_rules! check {
        ($source:expr) => {{
            let source = $source;
            let clone = source.clone();
            let mut seen = BTreeMap::new();
            let mut visitor = |identity: *const (), bytes| match seen.entry(identity as usize) {
                std::collections::btree_map::Entry::Vacant(entry) => {
                    entry.insert(bytes);
                    true
                }
                std::collections::btree_map::Entry::Occupied(entry) => {
                    assert_eq!(*entry.get(), bytes);
                    false
                }
            };
            source.visit_source_storage(&mut visitor).unwrap();
            clone.visit_source_storage(&mut visitor).unwrap();
            assert!(seen.values().sum::<usize>() > 0);
            assert!(seen.len() > 1);
        }};
    }
    check!(regex::Regex::new("alpha|beta").unwrap());
    check!(regex::bytes::Regex::new("alpha|beta").unwrap());
    check!(regex::RegexSet::new(["alpha", "beta"]).unwrap());
    check!(regex::bytes::RegexSet::new(["alpha", "beta"]).unwrap());
}
