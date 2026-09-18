use super::*;
use crate::allocation::{Allocation, AllocationError};
use core::cell::Cell;
struct Funding {
    calls: Cell<usize>,
    bytes: Cell<usize>,
    refuse: usize,
}
impl Allocation for Funding {
    fn reserve(&self, bytes: usize) -> core::result::Result<(), AllocationError> {
        let call = self.calls.get();
        self.calls.set(call + 1);
        if call >= self.refuse {
            return Err(AllocationError::Refused);
        }
        self.bytes.set(self.bytes.get().checked_add(bytes).unwrap());
        Ok(())
    }
}
#[test]
fn every_parser_destination_refuses_before_following_storage_or_diagnostics() {
    let flags = crate::RegexOptions::default().compute_flags();
    for pattern in [
        r"(?<first>a+|b)(?<second>c\x{41})\k<first>",
        r"[\p{Alpha}\P{^Uppercase}\d\x{3a3}]+(?:x|y){0,2}",
        r"(?<first>ab)\g<first>",
        r"(?~ab)",
        r"(a)?(?(1)b|c)",
        r"\b{start-half}",
        r"[\p{ΟΣ'Α}\p{ΟΣ'}]",
        r"\p{BLANK}",
        r"\k<forward>(?<forward>a)",
        r"(?q:a)",
        r"\y",
        r"\B{wrong}",
        r"(?<name>unclosed",
        r"(a|[\p{Greek}",
    ] {
        let expected = Parser::parse_with_flags(pattern, flags);
        let funding = Funding {
            calls: Cell::new(0),
            bytes: Cell::new(0),
            refuse: usize::MAX,
        };
        let actual = Parser::parse_with_allocations(pattern, flags, &funding);
        match (&actual, &expected) {
            (Ok(actual), Ok(expected)) => {
                assert_eq!(actual.expr, expected.expr);
                assert_eq!(actual.named_groups, expected.named_groups);
                assert_eq!(
                    actual.backrefs.iter().collect::<Vec<_>>(),
                    expected.backrefs.iter().collect::<Vec<_>>()
                );
            }
            (Err(actual), Err(expected)) => assert_eq!(actual.to_string(), expected.to_string()),
            _ => panic!("parser acceptance differs: {}", pattern),
        }
        let requests = funding.calls.get();
        assert!(requests > 0 && funding.bytes.get() > 0, "{}", pattern);
        drop(actual);
        for refuse in 0..requests {
            let funding = Funding {
                calls: Cell::new(0),
                bytes: Cell::new(0),
                refuse,
            };
            let failure = Parser::parse_with_allocations(pattern, flags, &funding).unwrap_err();
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
