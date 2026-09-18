use super::*;
use core::cell::Cell;
struct Funding {
    calls: Cell<usize>,
    refuse: usize,
}
impl Allocation for Funding {
    fn reserve(&self, bytes: usize) -> Result<()> {
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
fn every_rewrite_destination_is_admitted_before_growth() {
    for pattern in [
        r"(?:a+)+",
        r"(a+)+",
        r"\w+\.?\w+",
        r"(?:\w+\.?\w+)*",
        r"prefix\Ktail(?=end)",
        r"abc(?=tail)",
        r"(?=tail)",
        r"(?:(a+)+b)",
    ] {
        let original = Expr::parse_tree(pattern).unwrap();
        let mut expected = original.clone();
        let expected_capture = optimize(&mut expected);
        // Keeping original alive also exercises copy-on-write capture Arcs.
        let mut actual = original.clone();
        let funding = Funding {
            calls: Cell::new(0),
            refuse: usize::MAX,
        };
        assert_eq!(
            optimize_with_allocations(&mut actual, &funding).unwrap(),
            expected_capture
        );
        assert_eq!(actual.expr, expected.expr);
        let requests = funding.calls.get();
        assert!(requests > 0, "{}", pattern);
        drop(actual);
        for refuse in 0..requests {
            let mut candidate = original.clone();
            let funding = Funding {
                calls: Cell::new(0),
                refuse,
            };
            assert_eq!(
                optimize_with_allocations(&mut candidate, &funding),
                Err(AllocationError::Refused),
                "{pattern} at {refuse}"
            );
            assert_eq!(funding.calls.get(), refuse + 1);
        }
    }
}
