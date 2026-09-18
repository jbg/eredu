use super::*;
use crate::allocation::Allocation;
use core::cell::Cell;

struct Funding {
    calls: Cell<usize>,
    bytes: Cell<usize>,
    refuse_at: usize,
}
impl Funding {
    fn new(refuse_at: usize) -> Self {
        Self {
            calls: Cell::new(0),
            bytes: Cell::new(0),
            refuse_at,
        }
    }
}
impl Allocation for Funding {
    fn reserve(&self, bytes: usize) -> Result<(), AllocationError> {
        let call = self.calls.get();
        self.calls.set(call + 1);
        if call >= self.refuse_at {
            return Err(AllocationError::Refused);
        }
        self.bytes.set(self.bytes.get().checked_add(bytes).unwrap());
        Ok(())
    }
}

fn exhaust<T: core::fmt::Debug>(
    mut worker: impl FnMut(Allocator<'_>) -> Result<T, AllocationError>,
) {
    let funding = Funding::new(usize::MAX);
    drop(worker(Allocator::new(&funding)).unwrap());
    assert!(funding.calls.get() > 0 && funding.bytes.get() > 0);
    for refuse_at in 0..funding.calls.get() {
        let funding = Funding::new(refuse_at);
        assert_eq!(
            worker(Allocator::new(&funding)).unwrap_err(),
            AllocationError::Refused
        );
        assert_eq!(
            funding.calls.get(),
            refuse_at + 1,
            "producer ran after refusal"
        );
    }
}

#[test]
fn extraction_and_optimization_refuse_each_reached_destination() {
    for pattern in [
        "",
        "look\\b",
        "(a|bc|d)[xyz]{2,4}",
        "[α-γ][0-3]?tail",
        "(?-u:[a-c]){2}",
        "(?:samwise|sam|samantha|frodo|farmer|farmyard)",
        "(a||f)+foo",
        "(?:ab|ac|ad){3}",
        "(?:a*?|ba*?|c*)suffix",
        "(?:abc|xyz){30}",
    ] {
        let hir = crate::parse(pattern).unwrap();
        for kind in [ExtractKind::Prefix, ExtractKind::Suffix] {
            let mut extractor = Extractor::new();
            extractor.kind(kind.clone());
            exhaust(|allocation| {
                let mut seq = extractor.extract_with_allocations(&hir, allocation)?;
                match kind {
                    ExtractKind::Prefix => {
                        seq.optimize_for_prefix_by_preference_with_allocations(allocation)?
                    }
                    ExtractKind::Suffix => {
                        seq.optimize_for_suffix_by_preference_with_allocations(allocation)?
                    }
                }
                Ok(seq)
            });
        }
    }
}

#[test]
fn sequence_splice_and_preference_trie_keep_expected_order() {
    exhaust(|allocation| {
        let mut seq = Seq::new_with_allocations(["a", "", "f", ""], allocation)?;
        let mut other = Seq::new_with_allocations(["foo", "far"], allocation)?;
        seq.union_into_empty_with_allocations(&mut other, allocation)?;
        assert_eq!(seq, Seq::new(["a", "foo", "far", "f"]));
        assert_eq!(other.len(), Some(0));
        Ok(seq)
    });
    exhaust(|allocation| {
        let mut seq = Seq::new_with_allocations(
            ["samwise", "sam", "samantha", "frodo", "frodo"],
            allocation,
        )?;
        seq.minimize_by_preference_with_allocations(allocation)?;
        assert_eq!(
            seq.literals()
                .unwrap()
                .iter()
                .map(Literal::as_bytes)
                .collect::<Vec<_>>(),
            [b"samwise".as_slice(), b"sam", b"frodo"]
        );
        assert!(!seq.literals().unwrap()[1].is_exact());
        Ok(seq)
    });
}

#[cfg(feature = "std")]
#[test]
fn deeply_nested_hir_uses_the_paid_traversal_stack() {
    let mut hir = Hir::literal(b"needle".as_slice());
    for index in 1..=4096 {
        hir = Hir::capture(hir::Capture {
            index,
            name: None,
            sub: alloc::boxed::Box::new(hir),
        });
    }
    std::thread::Builder::new()
        .stack_size(64 * 1024)
        .spawn(move || {
            let funding = Funding::new(usize::MAX);
            let seq = Extractor::new()
                .extract_with_allocations(&hir, Allocator::new(&funding))
                .unwrap();
            assert_eq!(seq, Seq::new(["needle"]));
            assert!(funding.calls.get() > 0);
        })
        .unwrap()
        .join()
        .unwrap();
}
