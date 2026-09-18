use super::*;
use crate::util::allocation::{Allocation, AllocationError};
use core::cell::Cell;

struct Funding {
    calls: Cell<usize>,
    limit: usize,
}
impl Allocation for Funding {
    fn reserve(&self, _: usize) -> Result<(), AllocationError> {
        let call = self.calls.get();
        self.calls.set(call + 1);
        if call < self.limit {
            Ok(())
        } else {
            Err(AllocationError::Refused)
        }
    }
}

#[test]
fn meta_source_refuses_every_reached_original_destination() {
    for pattern in [
        "needle",
        "(a+)(b+)",
        r"[a-z]+\s+END",
        r"\d+@!\w+",
        r"\d+XYZ",
        "(α+)([0-9]+)",
        "(?P<name>a)(",
    ] {
        let builder = Builder::new();
        let paid = Funding {
            calls: Cell::new(0),
            limit: usize::MAX,
        };
        let result = builder.build_with_allocations(pattern, &paid);
        assert!(paid.calls.get() > 0);
        if pattern.ends_with('(') {
            assert!(result.unwrap_err().syntax_error().is_some());
        } else {
            assert!(result.is_ok(), "{pattern}: {result:?}");
        }
        for limit in 0..paid.calls.get() {
            let refuse = Funding {
                calls: Cell::new(0),
                limit,
            };
            let error = builder
                .build_with_allocations(pattern, &refuse)
                .unwrap_err();
            assert_eq!(
                error.allocation_error(),
                Some(AllocationError::Refused),
                "{pattern}, request {limit}"
            );
            assert_eq!(
                refuse.calls.get(),
                limit + 1,
                "must stop at first refused source destination"
            );
        }
    }
}

#[test]
fn scoped_meta_search_refuses_without_engine_fallback_and_recovers() {
    use crate::util::allocation::Unenforced;
    use crate::{Anchored, Input};
    let configs = [
        Config::new(),
        Config::new().dfa(false).onepass(false).backtrack(false),
        Config::new().dfa(false).hybrid(false).onepass(false),
        Config::new()
            .dfa(false)
            .hybrid(false)
            .onepass(false)
            .backtrack(false),
    ];
    for config in configs {
        for (pattern, text) in [
            ("needle", "a needle here"),
            ("(a+)(b+)", "xxaaaabbb!"),
            (r"[a-z]+\s+END", "aa aa aa END"),
            (r"\d+@!\w+", "22@!alpha"),
            (r"\d+XYZ", "11223XYZ"),
            (r"\w+$", "αβ 123xyz"),
            ("(α+)([0-9]+)", "αα123!"),
            (r"\b(α|β)+\b", "αβα !"),
        ] {
            let regex = Builder::new()
                .configure(config.clone())
                .build(pattern)
                .unwrap();
            for anchored in [Anchored::No, Anchored::Yes] {
                let input = Input::new(text).anchored(anchored);
                for operation in 0..4 {
                    let run = |cache: &mut Cache, funding: &dyn Allocation| {
                        let mut slots = [None; 12];
                        match operation {
                            0 => regex
                                .search_with_allocations(cache, &input, funding)
                                .map(|m| (m.map(|m| (m.start(), m.end())), slots)),
                            1 => regex
                                .search_half_with_allocations(cache, &input, funding)
                                .map(|m| (m.map(|m| (0, m.offset())), slots)),
                            2 => regex
                                .is_match_with_allocations(cache, &input, funding)
                                .map(|yes| (yes.then_some((0, 0)), slots)),
                            _ => regex
                                .search_slots_with_allocations(cache, &input, &mut slots, funding)
                                .map(|m| (m.map(|p| (p.as_usize(), 0)), slots)),
                        }
                    };
                    let paid = Funding {
                        calls: Cell::new(0),
                        limit: usize::MAX,
                    };
                    let mut cache = regex.create_cache_with_allocations(&paid).unwrap();
                    let expected = run(&mut cache, &paid).unwrap();
                    assert_eq!(
                        run(&mut regex.create_cache(), &Unenforced).unwrap(),
                        expected
                    );
                    for limit in 0..paid.calls.get() {
                        let refuse = Funding {
                            calls: Cell::new(0),
                            limit,
                        };
                        let result = regex.create_cache_with_allocations(&refuse).and_then(|mut cache| {
                            let result = run(&mut cache, &refuse);
                            if result.is_err() {
                                assert_eq!(run(&mut cache, &Unenforced).unwrap(), expected, "recovery {pattern}, operation {operation}, request {limit}");
                            }
                            result
                        });
                        let error = result.unwrap_err();
                        assert_eq!(
                            error.allocation_error(),
                            Some(AllocationError::Refused),
                            "{pattern}, operation {operation}, request {limit}"
                        );
                        assert_eq!(
                            refuse.calls.get(),
                            limit + 1,
                            "allocation refusal must not select another engine"
                        );
                    }
                }
            }
        }
    }
}
