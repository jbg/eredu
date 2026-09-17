use super::*;
use eredu_core::SharedStorageDomain;
use eredu_runtime::{HostMetadataIdentity, HostSlotAttachmentError, HostSlotMetadata};
use std::{
    convert::Infallible,
    sync::{
        atomic::{AtomicU64, AtomicUsize, Ordering},
        Arc,
    },
};

struct Charge {
    bytes: u64,
    used: Arc<AtomicU64>,
    _identity: HostMetadataIdentity,
}
impl Drop for Charge {
    fn drop(&mut self) {
        self.used.fetch_sub(self.bytes, Ordering::SeqCst);
    }
}

fn attach(
    metadata: &HostSlotMetadata,
    domain: &SharedStorageDomain,
    used: &Arc<AtomicU64>,
) -> bool {
    let bytes = metadata.capacity_bytes().unwrap();
    let identity = metadata.identity().clone();
    metadata
        .try_attach(domain, || {
            used.fetch_add(bytes, Ordering::SeqCst);
            Ok::<Box<dyn Send + Sync>, Infallible>(Box::new(Charge {
                bytes,
                used: used.clone(),
                _identity: identity,
            }))
        })
        .unwrap()
}

#[test]
fn slot_extent_reuse_keeps_custody_and_independent_clones_get_fresh_owners() {
    let stream = stream();
    let used = Arc::new(AtomicU64::new(0));
    let domain = SharedStorageDomain::default();
    let mut source = FixedStateSlots::from_policy(&policy(&[RECURRENT, CONV])).unwrap();
    *source.get_mut(&CONV).unwrap() = Some(tensor([1., 3., 5., 7.]));
    let mut destination = FixedStateSlots::from_policy(&policy(&[PREFIX, CONV])).unwrap();
    *destination.get_mut(&PREFIX).unwrap() = Some(tensor([2., 4., 6., 8.]));
    let token = destination.metadata().clone();
    let capacity = token.capacity_bytes().unwrap();
    assert!(capacity > 0);
    assert!(attach(&token, &domain, &used));
    destination.clone_from(&source);
    assert!(token.same_storage(destination.metadata()));
    assert!(!token.same_storage(source.metadata()));
    assert_eq!(roles(&destination), [CONV, RECURRENT]);
    assert_eq!(
        values(
            destination.get_mut(&CONV).unwrap().as_ref().unwrap(),
            &stream
        ),
        [1., 3., 5., 7.]
    );
    assert!(destination.get_mut(&RECURRENT).unwrap().is_none());
    assert!(!destination
        .metadata()
        .try_attach::<Infallible>(&domain, || panic!(
            "same extent must retain existing custody"
        ))
        .unwrap());
    assert_eq!(used.load(Ordering::SeqCst), capacity);

    let independent = destination.clone();
    assert!(!token.same_storage(independent.metadata()));
    assert_eq!(independent.metadata().capacity_bytes(), Some(capacity));
    assert!(attach(independent.metadata(), &domain, &used));
    assert_eq!(used.load(Ordering::SeqCst), capacity * 2);
    drop(independent);
    assert_eq!(used.load(Ordering::SeqCst), capacity);

    let larger = FixedStateSlots::from_policy(&policy(&[PREFIX, RECURRENT, CONV])).unwrap();
    destination.clone_from(&larger);
    assert!(!token.same_storage(destination.metadata()));
    assert!(!larger.metadata().same_storage(destination.metadata()));
    assert_eq!(
        destination.metadata().capacity_bytes(),
        larger.metadata().capacity_bytes()
    );
    assert!(matches!(
        token.try_attach::<Infallible>(&domain, || panic!("retired token must not acquire")),
        Err(HostSlotAttachmentError::Retired)
    ));
    assert_eq!(
        used.load(Ordering::SeqCst),
        capacity,
        "escaped old token retains only old custody"
    );
    drop(token);
    assert_eq!(used.load(Ordering::SeqCst), 0);
    assert!(destination.values().all(Option::is_none));
}

#[test]
fn escaped_fixed_metadata_retains_custody_without_retaining_native_slot_payload() {
    struct NativeRetired(Arc<AtomicUsize>);
    impl Drop for NativeRetired {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }
    let mut source = MlxHybridLayerState::device(0, &policy(&[PREFIX, CONV])).unwrap();
    *source.fixed_component(CONV).unwrap() = Some(tensor([11., 13., 17., 19.]));
    let native_retired = Arc::new(AtomicUsize::new(0));
    {
        let array = source
            .fixed_component(CONV)
            .unwrap()
            .as_ref()
            .unwrap()
            .as_array();
        array.evaluated().unwrap();
        array
            .retain_allocation_owner(NativeRetired(native_retired.clone()))
            .unwrap();
    }
    let metadata = source.fixed_slot_metadata().clone();
    let earlier = metadata.clone();
    let identity = metadata.identity().clone();
    let capacity = source.fixed_slot_payload_bytes().unwrap();
    assert_eq!(metadata.capacity_bytes(), Some(capacity));
    let used = Arc::new(AtomicU64::new(0));
    let first = SharedStorageDomain::default();
    let second = SharedStorageDomain::default();
    assert!(attach(&metadata, &first, &used));
    assert!(!earlier
        .try_attach::<Infallible>(&first, || panic!("earlier token sees attachment"))
        .unwrap());
    assert!(attach(&earlier, &second, &used));
    assert_eq!(used.load(Ordering::SeqCst), capacity * 2);
    drop(source);
    crate::backend::submission_recovery::wait_for_retirement(|| {
        safemlx::reclaim_allocation_owners();
        native_retired.load(Ordering::SeqCst) == 1
    });
    assert_eq!(used.load(Ordering::SeqCst), capacity * 2);
    for domain in [&first, &SharedStorageDomain::default()] {
        assert!(matches!(
            metadata.try_attach::<Infallible>(domain, || panic!(
                "retired table cannot reacquire custody"
            )),
            Err(HostSlotAttachmentError::Retired)
        ));
    }
    assert_eq!(metadata.identity(), &identity);
    drop(metadata);
    assert_eq!(used.load(Ordering::SeqCst), capacity * 2);
    drop(earlier);
    assert_eq!(used.load(Ordering::SeqCst), 0);
    // The independently retained registry key neither owns slots nor charges.
    drop(identity);
    assert_eq!(used.load(Ordering::SeqCst), 0);
}
