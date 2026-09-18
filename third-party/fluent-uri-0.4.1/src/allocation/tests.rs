use super::*;
use crate::{normalize::NormalizeError, resolve::ResolveError, Iri, Uri, UriRef};
use core::sync::atomic::{AtomicUsize, Ordering};

struct Funding {
    calls: AtomicUsize,
    stop: usize,
}
impl Funding {
    fn new(stop: usize) -> Self {
        Self {
            calls: AtomicUsize::new(0),
            stop,
        }
    }
}
impl Allocation for Funding {
    fn reserve(&self, bytes: usize) -> Result<(), AllocationError> {
        assert!(bytes > 0);
        let index = self.calls.fetch_add(1, Ordering::Relaxed);
        if index == self.stop {
            Err(AllocationError::Refused)
        } else {
            Ok(())
        }
    }
}

#[test]
fn normalization_refuses_each_reached_allocation() {
    // The IPv6 expansion exceeds the initial input length and exercises replacement growth.
    for (input, expected) in [
        ("a://[::ffff:5:9]/", "a://[::ffff:0.5.0.9]/"),
        (
            "HTTPS://EXAMPLE.COM:443/a/../%62?x=%2f#%7e",
            "https://example.com/b?x=%2F#~",
        ),
        (
            "https://例.example/%C3%A9/%E6%B0%B4",
            "https://例.example/é/水",
        ),
    ] {
        let value = Iri::parse(input).unwrap();
        let funding = Funding::new(usize::MAX);
        assert_eq!(
            value.normalize_with_allocations(&funding).unwrap().as_str(),
            expected
        );
        let calls = funding.calls.load(Ordering::Relaxed);
        assert!(calls >= 2);
        for stop in 0..calls {
            let funding = Funding::new(stop);
            assert_eq!(
                value.normalize_with_allocations(&funding),
                Err(NormalizeError::Allocation(AllocationError::Refused))
            );
            assert_eq!(funding.calls.load(Ordering::Relaxed), stop + 1);
        }
    }
}

#[test]
fn resolution_and_copy_refuse_before_storage() {
    let base = Uri::parse("https://example.com/a/b/c?old#ignored").unwrap();
    let base = base.strip_fragment();
    let relative = UriRef::parse("../../d?x#y").unwrap();
    let funding = Funding::new(usize::MAX);
    assert_eq!(
        relative
            .resolve_against_with_allocations(&base, &funding)
            .unwrap(),
        "https://example.com/d?x#y"
    );
    for stop in 0..funding.calls.load(Ordering::Relaxed) {
        assert_eq!(
            relative.resolve_against_with_allocations(&base, &Funding::new(stop)),
            Err(ResolveError::Allocation(AllocationError::Refused))
        );
    }
    assert_eq!(
        base.to_owned_with_allocations(&Funding::new(0)),
        Err(AllocationError::Refused)
    );
    assert_eq!(
        base.with_fragment_with_allocations(None, &Funding::new(0)),
        Err(AllocationError::Refused)
    );
}

#[test]
fn fragment_refusal_preserves_original_and_removal_does_not_allocate() {
    let mut value = Uri::parse("https://example.com/#old").unwrap().to_owned();
    let before = value.clone();
    let fragment = crate::pct_enc::EStr::new_or_panic("a-much-longer-fragment-than-the-original");
    assert_eq!(
        value.set_fragment_with_allocations(Some(fragment), &Funding::new(0)),
        Err(AllocationError::Refused)
    );
    assert_eq!(value, before);
    let funding = Funding::new(0);
    value.set_fragment_with_allocations(None, &funding).unwrap();
    assert_eq!(value, "https://example.com/");
    assert_eq!(funding.calls.load(Ordering::Relaxed), 0);
}
