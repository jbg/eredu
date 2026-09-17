use super::*;
use eredu_core::{SharedStorageDomain, cache::LayerCachePolicy};
use std::sync::{Arc, atomic::{AtomicBool, Ordering}};

struct Preparation(Arc<AtomicBool>);
impl Drop for Preparation {
    fn drop(&mut self) { self.0.store(true, Ordering::SeqCst); }
}

#[test]
fn copied_empty_children_keep_prepared_identity_and_host_custody() {
    let retired = Arc::new(AtomicBool::new(false));
    let host = HostPreparationAuthority::retain(Preparation(retired.clone()));
    let domain = SharedStorageDomain::default();
    let source = FixedStateSlots::from_policy(&LayerCachePolicy::NoState).unwrap();
    assert!(source.empty_copy_preparation_bytes().unwrap() > 0);
    let ordinary = source.copy_empty().unwrap();
    assert!(matches!(ordinary.metadata().prepare_copy_attachment(&domain),
        Err(WorkingMemoryError::IdentityMismatch)));
    let first = source.copy_empty_prepared(&host).unwrap();
    let second = first.copy_empty_prepared(&host).unwrap();
    assert!(first.is_empty() && second.is_empty());
    assert_eq!(first.payload_bytes(), Some(0));
    assert!(!first.metadata().same_storage(source.metadata()));
    assert!(!second.metadata().same_storage(first.metadata()));
    first.metadata().prepare_copy_attachment(&domain).unwrap();
    second.metadata().prepare_copy_attachment(&domain).unwrap();
    // A nonempty fixed-role table cannot enter this zero-payload branch.
    let nonempty = FixedStateSlots::from_published_slots(eredu_runtime::HostSlotTable::new(
        vec![(eredu_core::cache::StateTensorRole::PositionDelta, None)].into_boxed_slice()));
    assert!(nonempty.empty_copy_preparation_bytes().is_none());
    assert!(matches!(nonempty.copy_empty_prepared(&host),
        Err(Error::PrefillControl(WorkingMemoryError::UnknownBound))));
    let escaped = second.metadata().clone();
    drop((source, ordinary, first, second, nonempty, host));
    assert!(!retired.load(Ordering::SeqCst));
    drop(escaped);
    assert!(retired.load(Ordering::SeqCst));
}
