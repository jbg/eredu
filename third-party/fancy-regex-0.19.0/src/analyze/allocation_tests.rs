use alloc::format;
use super::*;
use crate::allocation::AllocationError;
use core::cell::Cell;
struct Funding {
    calls: Cell<usize>,
    refuse: usize,
}
impl Allocation for Funding {
    fn reserve(&self, bytes: usize) -> core::result::Result<(), AllocationError> {
        assert!(bytes > 0);
        let call = self.calls.get();
        self.calls.set(call + 1);
        if call >= self.refuse {
            Err(AllocationError::Refused)
        } else {
            Ok(())
        }
    }
}
#[test]
fn analysis_graph_and_diagnostic_storage_refuses_at_each_reached_destination() {
    for pattern in [
        r"(a|ab)\1",
        r"(?<word>a\g<word>?b)\g<word>",
        r"(?(DEFINE)(?<word>ab))\g<word>",
        r"(?<word>\g<word>)",
        r"(?<word>a\g<word>)",
        r"\g<missing>",
        r"\9(a)",
        r"(?~|ab|c+)",
        r"(a)?(?(1)b|c)",
    ] {
        let tree = Expr::parse_tree(pattern).unwrap();
        let expected = analyze(&tree, AnalyzeContext::default());
        let funding = Funding {
            calls: Cell::new(0),
            refuse: usize::MAX,
        };
        let actual = analyze_with_allocations(&tree, AnalyzeContext::default(), &funding);
        assert_eq!(format!("{actual:?}"), format!("{expected:?}"), "{pattern}");
        let requests = funding.calls.get();
        assert!(requests > 0, "{}", pattern);
        drop(actual);
        for refuse in 0..requests {
            let funding = Funding {
                calls: Cell::new(0),
                refuse,
            };
            let failure =
                analyze_with_allocations(&tree, AnalyzeContext::default(), &funding).unwrap_err();
            assert!(
                matches!(failure, Error::Allocation(AllocationError::Refused)),
                "{} at {}: {}",
                pattern,
                refuse,
                failure
            );
            assert_eq!(funding.calls.get(), refuse + 1);
        }
    }
}
