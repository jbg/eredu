use alloc::{format, string::ToString};
use super::*;
use crate::allocation::{Allocation, AllocationError};
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
fn vm_emission_and_native_class_frontiers_preserve_refusal() {
    // This covers the compiler-owned producers. Lower automata and pool
    // producers are a separate integration frontier, not inferred from this test.
    for pattern in [
        r"(ab)\1",
        r"(?:(?=a)a|(?=b)b)(a)\1",
        r"(a)(?>\1{0,2})b",
        r"[a-z](a)\1",
        r"(a)\g<1>",
        r"(?~|a|b)",
    ] {
        let source =
            crate::source::ParsedSource::new(pattern, &crate::RegexOptions::default()).unwrap();
        let info = source.analyze().unwrap();
        let options = source.compile_options(&crate::RegexOptions::default());
        let funding = Funding {
            calls: Cell::new(0),
            refuse: usize::MAX,
        };
        let completed = compile_with_context(&info, options.clone(), false, Context::new(&funding));
        let expected = compile(&info, options.clone());
        match (&completed, &expected) {
            (Ok(actual), Ok(expected)) => {
                assert_eq!(format!("{:?}", actual.body), format!("{:?}", expected.body))
            }
            (Err(actual), Err(expected)) => assert_eq!(actual.to_string(), expected.to_string()),
            _ => panic!("compiler behavior differs for {}", pattern),
        }
        let requests = funding.calls.get();
        assert!(requests > 0, "{}", pattern);
        drop(completed);
        for refuse in 0..requests {
            let funding = Funding {
                calls: Cell::new(0),
                refuse,
            };
            let failure =
                compile_with_context(&info, options.clone(), false, Context::new(&funding))
                    .unwrap_err();
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
