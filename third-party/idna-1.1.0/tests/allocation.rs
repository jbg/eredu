use idna::{
    allocation::{Allocation, AllocationError},
    uts46::{AsciiDenyList, DnsLength, Hyphens, Uts46},
};
use std::cell::Cell;
struct Funding {
    calls: Cell<usize>,
    refuse_at: usize,
}
impl Allocation for Funding {
    fn reserve(&self, _: usize) -> Result<(), AllocationError> {
        let call = self.calls.get();
        assert!(call <= self.refuse_at, "another producer ran after refusal");
        self.calls.set(call + 1);
        if call == self.refuse_at {
            Err(AllocationError::Refused)
        } else {
            Ok(())
        }
    }
}
fn funding(refuse_at: usize) -> Funding {
    Funding {
        calls: Cell::new(0),
        refuse_at,
    }
}
#[test]
fn every_reached_idna_destination_stops_at_its_first_refusal() {
    let mut inputs = vec![
        "example.org".to_owned(),
        "BÜCHER.example".to_owned(),
        "xn--bcher-kva.example".to_owned(),
        "a\u{200d}b.example".to_owned(),
        "\u{fdfa}.example".to_owned(),
        "a\u{315}\u{301}\u{300}.example".to_owned(),
        format!("a{}.example", "\u{315}\u{301}\u{300}".repeat(40)),
        format!("{}.example", "é.".repeat(140)),
        format!("{}.example", "音".repeat(300)),
    ];
    let encoded = Uts46::new()
        .to_ascii(
            inputs.last().unwrap().as_bytes(),
            AsciiDenyList::STD3,
            Hyphens::Check,
            DnsLength::Ignore,
        )
        .unwrap()
        .into_owned();
    inputs.push(encoded);
    for input in inputs {
        for dns in [DnsLength::Ignore, DnsLength::Verify] {
            let expected =
                Uts46::new().to_ascii(input.as_bytes(), AsciiDenyList::STD3, Hyphens::Check, dns);
            let source = funding(usize::MAX);
            let actual = Uts46::new()
                .to_ascii_with_allocations(
                    input.as_bytes(),
                    AsciiDenyList::STD3,
                    Hyphens::Check,
                    dns,
                    &source,
                )
                .unwrap();
            assert_eq!(actual.map_err(|_| ()), expected.map_err(|_| ()), "{input}");
            for refuse_at in 0..source.calls.get() {
                let refused = funding(refuse_at);
                let result = Uts46::new().to_ascii_with_allocations(
                    input.as_bytes(),
                    AsciiDenyList::STD3,
                    Hyphens::Check,
                    dns,
                    &refused,
                );
                assert!(
                    matches!(result, Err(AllocationError::Refused)),
                    "{}: {}: {:?}",
                    input,
                    refuse_at,
                    result
                );
                assert_eq!(refused.calls.get(), refuse_at + 1);
            }
        }
        let expected =
            Uts46::new().to_unicode(input.as_bytes(), AsciiDenyList::EMPTY, Hyphens::Allow);
        let source = funding(usize::MAX);
        let actual = Uts46::new()
            .to_unicode_with_allocations(
                input.as_bytes(),
                AsciiDenyList::EMPTY,
                Hyphens::Allow,
                &source,
            )
            .unwrap();
        assert_eq!(
            (actual.0, actual.1.is_ok()),
            (expected.0, expected.1.is_ok()),
            "{input}"
        );
        for refuse_at in 0..source.calls.get() {
            let refused = funding(refuse_at);
            let result = Uts46::new().to_unicode_with_allocations(
                input.as_bytes(),
                AsciiDenyList::EMPTY,
                Hyphens::Allow,
                &refused,
            );
            assert!(
                matches!(result, Err(AllocationError::Refused)),
                "{}: {}: {:?}",
                input,
                refuse_at,
                result
            );
            assert_eq!(refused.calls.get(), refuse_at + 1);
        }
    }
}
