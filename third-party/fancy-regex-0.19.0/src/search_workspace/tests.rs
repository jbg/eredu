use alloc::string::ToString;
use super::*;
use crate::{
    allocation::{AllocationError, Unenforced},
    Error, RegexOptionsBuilder,
};
use core::cell::Cell;

struct Funding {
    calls: Cell<usize>,
    limit: usize,
}
impl Allocation for Funding {
    fn reserve(&self, _: usize) -> core::result::Result<(), AllocationError> {
        let call = self.calls.get();
        self.calls.set(call + 1);
        if call < self.limit {
            Ok(())
        } else {
            Err(AllocationError::Refused)
        }
    }
}
const CASES: &[(&str, &str)] = &[
    ("needle", "one needle here"),
    ("a{3,1}", "aaaa"),
    (r"(a+)(b+)", "xaaabbb!"),
    (r"(?=a+)(a+)\1", "aaaaaa!"),
    (r"[α-ω]+\d+", "!αβ123"),
    (r"(?i:(α+))\1", "ααΑΑ!"),
    (r"(?>a+)b|a+c", "aaaac"),
    (r"(?<=ab)c", "abc"),
    (r"(?<=a+)b", "aaab"),
    (r"(?<=(a+))b", "aaab"),
    (r"(?~END)", "aabbENDcc"),
    (r"\b\w+\b(?=!)", "αβ! xyz!"),
    (r"(?<name>a+)\g<name>", "aaaa"),
];

#[test]
fn original_source_constructor_refuses_every_reached_destination() {
    for pattern in
        CASES
            .iter()
            .map(|case| case.0)
            .chain(["(?P<name>a)(", r"\p{NoSuchProperty}", r"(a)\9"])
    {
        let paid = Funding {
            calls: Cell::new(0),
            limit: usize::MAX,
        };
        let result = RegexOptionsBuilder::new().build_with_allocations(pattern, &paid);
        let ordinary = Regex::new(pattern);
        assert_eq!(result.is_ok(), ordinary.is_ok(), "{pattern}");
        if let (Err(paid), Err(ordinary)) = (result, ordinary) {
            assert_eq!(paid.to_string(), ordinary.to_string());
        }
        assert!(paid.calls.get() > 0);
        for limit in 0..paid.calls.get() {
            let refuse = Funding {
                calls: Cell::new(0),
                limit,
            };
            let result = RegexOptionsBuilder::new().build_with_allocations(pattern, &refuse);
            assert!(
                matches!(result, Err(Error::Allocation(AllocationError::Refused))),
                "{}, request {}: {:?}", pattern, limit, result
            );
            assert_eq!(
                refuse.calls.get(),
                limit + 1,
                "must stop at first source refusal: {pattern}"
            );
        }
    }
}

#[test]
fn scoped_original_search_parity_refusal_and_recovery() {
    for &(pattern, text) in CASES {
        let Ok(regex) = Regex::new(pattern) else {
            continue;
        };
        for pos in text.char_indices().map(|(at, _)| at).chain([text.len()]) {
            for anchored in [false, true] {
                let input = RegexInput::new(text).from_pos(pos).anchored(anchored);
                let expected = regex
                    .find_input(input.clone())
                    .unwrap()
                    .map(|m| (m.start(), m.end()));
                for is_match in [false, true] {
                    let run = |workspace: &mut SearchWorkspace<'_>, funding: &dyn Allocation| {
                        if is_match {
                            workspace
                                .is_match_input(input.clone(), funding)
                                .map(|yes| yes.then_some((0, 0)))
                        } else {
                            workspace
                                .find_input(input.clone(), funding)
                                .map(|m| m.map(|m| (m.start(), m.end())))
                        }
                    };
                    let paid = Funding {
                        calls: Cell::new(0),
                        limit: usize::MAX,
                    };
                    let mut workspace = regex.search_workspace_with_allocations(&paid).unwrap();
                    let actual = run(&mut workspace, &paid).unwrap();
                    if is_match {
                        assert_eq!(
                            actual.is_some(),
                            regex.is_match_input(input.clone()).unwrap()
                        );
                    } else {
                        assert_eq!(actual, expected, "{pattern}, {pos}, {anchored}");
                    }
                    for limit in 0..paid.calls.get() {
                        let refuse = Funding {
                            calls: Cell::new(0),
                            limit,
                        };
                        let result = regex.search_workspace_with_allocations(&refuse).and_then(
                            |mut workspace| {
                                let result = run(&mut workspace, &refuse);
                                if result.is_err() {
                                    assert_eq!(
                                        run(&mut workspace, &Unenforced).unwrap(),
                                        actual,
                                        "recovery {pattern}, request {limit}"
                                    );
                                }
                                result
                            },
                        );
                        assert!(
                            matches!(result, Err(Error::Allocation(AllocationError::Refused))),
                            "{}, request {}: {:?}", pattern, limit, result
                        );
                        assert_eq!(
                            refuse.calls.get(),
                            limit + 1,
                            "must stop at first invocation refusal"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn descending_explicit_repeat_preserves_original_hir_semantics() {
    let regex = RegexOptionsBuilder::new()
        .build_with_allocations("a{3,1}", &Unenforced)
        .unwrap();
    let mut workspace = regex
        .search_workspace_with_allocations(&Unenforced)
        .unwrap();
    let found = workspace.find("zaaaa", &Unenforced).unwrap().unwrap();
    assert_eq!((found.start(), found.end()), (1, 4));
}

#[test]
fn owning_invocation_retains_exact_source_and_reuses_admitted_storage() {
    let source = Arc::new(Regex::new(r"^[a-z_][a-z_0-9]*$").unwrap());
    let equal_pattern = Arc::new(Regex::new(source.as_str()).unwrap());
    let weak = Arc::downgrade(&source);
    let paid = Funding {
        calls: Cell::new(0),
        limit: usize::MAX,
    };
    let mut workspace = OwnedSearchWorkspace::new_with_allocations(source.clone(), &paid).unwrap();
    assert!(workspace.matches_source(&source));
    assert!(!workspace.matches_source(&equal_pattern));
    assert!(workspace.is_match("identifier_12", &paid).unwrap());
    drop(source);
    assert!(weak.upgrade().is_some());
    let refuse = Funding {
        calls: Cell::new(0),
        limit: 0,
    };
    for _ in 0..8 {
        assert!(workspace.is_match("identifier_12", &refuse).unwrap());
    }
    assert_eq!(refuse.calls.get(), 0);
    drop(workspace);
    assert!(weak.upgrade().is_none());
}
