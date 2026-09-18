use super::*;
use crate::{TextControllerStorage, TokenFilter};
use std::{
    collections::{BTreeSet, HashSet},
    convert::Infallible,
    error::Error,
    panic::{catch_unwind, AssertUnwindSafe},
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Barrier,
    },
};

fn bytes() -> SharedControllerBytes {
    let mut bytes = Vec::with_capacity(73);
    bytes.extend_from_slice(&[0, 17, 255, 63]);
    SharedControllerBytes::new(bytes, crate::HostPreparationAuthority::unmanaged())
}

fn bytes_with_retirement(retired: &Arc<AtomicBool>) -> SharedControllerBytes {
    SharedControllerBytes(SharedStorageOwner::new(BytesInner {
        bytes: vec![0, 17, 255, 63],
        custody: SharedStorageCustody::new(),
        authority: HostPreparationAuthority::unmanaged(),
        payload_retired: Some(retired.clone()),
    }))
}

#[test]
fn source_authority_survives_byte_aliases_but_detached_value_keys_need_no_storage() {
    struct Authority {
        payload_retired: Arc<AtomicBool>,
        retired: Arc<AtomicBool>,
    }
    impl Drop for Authority {
        fn drop(&mut self) {
            assert!(self.payload_retired.load(Ordering::SeqCst));
            self.retired.store(true, Ordering::SeqCst);
        }
    }
    let payload_retired = Arc::new(AtomicBool::new(false));
    let authority_retired = Arc::new(AtomicBool::new(false));
    let source = SharedControllerBytes(SharedStorageOwner::new(BytesInner {
        bytes: vec![1, 7, 19],
        custody: SharedStorageCustody::new(),
        authority: HostPreparationAuthority::retain(Authority {
            payload_retired: payload_retired.clone(),
            retired: authority_retired.clone(),
        }),
        payload_retired: Some(payload_retired.clone()),
    }));
    let key = *source.identity();
    let alias = source.clone();
    drop(source);
    assert!(!authority_retired.load(Ordering::SeqCst));
    assert_eq!(alias.as_ref(), &[1, 7, 19]);
    drop(alias);
    assert!(authority_retired.load(Ordering::SeqCst));
    assert_ne!(key, *bytes().identity());
}

#[test]
fn concurrent_source_identities_do_not_repeat_after_payload_retirement() {
    let keys = std::thread::scope(|scope| {
        let workers = (0..4).map(|_| scope.spawn(|| {
            (0..64).map(|_| *bytes().identity()).collect::<Vec<_>>()
        })).collect::<Vec<_>>();
        workers.into_iter().flat_map(|worker| worker.join().unwrap()).collect::<Vec<_>>()
    });
    assert_eq!(keys.iter().copied().collect::<BTreeSet<_>>().len(), keys.len());
}

struct Charge {
    used: Arc<AtomicUsize>,
    bytes: usize,
    _identity: SharedStorageIdentity,
    _domain: SharedStorageDomain,
}
impl Charge {
    fn acquire(
        source: SharedControllerSource<'_>,
        domain: &SharedStorageDomain,
        used: &Arc<AtomicUsize>,
    ) -> Box<dyn Send + Sync> {
        let bytes = source.capacity_bytes().unwrap() as usize;
        used.fetch_add(bytes, Ordering::SeqCst);
        Box::new(Self {
            used: used.clone(),
            bytes,
            _identity: source.identity().clone(),
            _domain: domain.clone(),
        })
    }
}
impl Drop for Charge {
    fn drop(&mut self) {
        self.used.fetch_sub(self.bytes, Ordering::SeqCst);
    }
}
fn attach(
    bytes: &SharedControllerBytes,
    domain: &SharedStorageDomain,
    used: &Arc<AtomicUsize>,
) -> bool {
    bytes
        .try_attach(domain, || {
            Ok::<_, Infallible>(Charge::acquire(
                SharedControllerSource::Bytes(bytes),
                domain,
                used,
            ))
        })
        .unwrap()
}

#[test]
fn transfer_preserves_pointer_spare_capacity_and_distinct_equal_owner_identity() {
    fn send_sync<T: Send + Sync>() {}
    send_sync::<SharedControllerBytes>();
    send_sync::<SharedStorageIdentity>();
    send_sync::<SharedControllerSource<'static>>();
    let mut raw = Vec::with_capacity(73);
    raw.extend_from_slice(&[0, 17, 255, 63]);
    let pointer = raw.as_ptr();
    let capacity = raw.capacity();
    let source = SharedControllerBytes::new(raw, crate::HostPreparationAuthority::unmanaged());
    let alias = source.clone();
    let equal = bytes();
    assert_eq!(source.as_ref(), &[0, 17, 255, 63]);
    assert_eq!(AsRef::<[u8]>::as_ref(&source).as_ptr(), pointer);
    assert_eq!(alias.as_ref().as_ptr(), pointer);
    assert_eq!(source.capacity_bytes(), Some(capacity as u64));
    assert!(capacity > source.as_ref().len());
    assert!(source.same_storage(&alias));
    assert!(!source.same_storage(&equal));
    assert_eq!(source.identity(), alias.identity());
    assert_ne!(source.identity(), equal.identity());
    assert_eq!(
        SharedControllerBytes::new(Vec::new(), crate::HostPreparationAuthority::unmanaged()).capacity_bytes(),
        Some(0)
    );
    let empty_spare = SharedControllerBytes::new(Vec::with_capacity(19), crate::HostPreparationAuthority::unmanaged());
    assert!(empty_spare.as_ref().is_empty());
    assert_eq!(empty_spare.capacity_bytes(), Some(19));
}

#[test]
fn identity_keys_survive_without_retaining_payload_or_domain_charges() {
    let retired = Arc::new(AtomicBool::new(false));
    let source = bytes_with_retirement(&retired);
    let identity = source.identity().clone();
    let domain = SharedStorageDomain::default();
    let used = Arc::new(AtomicUsize::new(0));
    attach(&source, &domain, &used);
    drop(source);
    assert!(retired.load(Ordering::SeqCst));
    assert_eq!(used.load(Ordering::SeqCst), 0);
    let other = bytes();
    let filter = SharedTokenFilter::new(TokenFilter::All);
    let filter_identity: &SharedStorageIdentity = filter.identity();
    assert_ne!(&identity, other.identity());
    assert_ne!(&identity, filter_identity);
    assert_eq!(
        BTreeSet::from([
            identity.clone(),
            identity.clone(),
            other.identity().clone(),
            filter_identity.clone()
        ])
        .len(),
        3
    );
    assert_eq!(
        HashSet::from([
            identity.clone(),
            identity,
            other.identity().clone(),
            filter_identity.clone()
        ])
        .len(),
        3
    );
}

#[test]
fn earlier_aliases_keep_each_domain_custody_until_final_retirement() {
    let source = bytes();
    let earlier_alias = source.clone();
    let first = SharedStorageDomain::default();
    let second = SharedStorageDomain::default();
    let used = Arc::new(AtomicUsize::new(0));
    let capacity = source.capacity_bytes().unwrap() as usize;
    assert!(attach(&source, &first, &used));
    assert!(!earlier_alias
        .try_attach::<Infallible>(&first.clone(), || panic!(
            "duplicate domain called provider"
        ))
        .unwrap());
    assert!(attach(&earlier_alias, &second, &used));
    assert_eq!(used.load(Ordering::SeqCst), 2 * capacity);
    drop(source);
    assert_eq!(used.load(Ordering::SeqCst), 2 * capacity);
    assert_eq!(earlier_alias.as_ref(), &[0, 17, 255, 63]);
    drop(earlier_alias);
    assert_eq!(used.load(Ordering::SeqCst), 0);
}

#[test]
fn simultaneous_attachments_publish_one_handle_per_domain() {
    let source = bytes();
    let domain = SharedStorageDomain::default();
    let barrier = Barrier::new(8);
    let used = Arc::new(AtomicUsize::new(0));
    let acquisitions = AtomicUsize::new(0);
    let published = std::thread::scope(|threads| {
        let joins = (0..8)
            .map(|_| {
                threads.spawn(|| {
                    barrier.wait();
                    source
                        .try_attach(&domain, || {
                            acquisitions.fetch_add(1, Ordering::SeqCst);
                            Ok::<_, Infallible>(Charge::acquire(
                                SharedControllerSource::Bytes(&source),
                                &domain,
                                &used,
                            ))
                        })
                        .unwrap()
                })
            })
            .collect::<Vec<_>>();
        joins
            .into_iter()
            .map(|join| usize::from(join.join().unwrap()))
            .sum::<usize>()
    });
    assert_eq!(published, 1);
    assert_eq!(acquisitions.load(Ordering::SeqCst), 1);
    assert_eq!(
        used.load(Ordering::SeqCst),
        source.capacity_bytes().unwrap() as usize
    );
    drop(source);
    assert_eq!(used.load(Ordering::SeqCst), 0);
}

#[derive(Debug)]
struct Rejected {
    owner: SharedControllerBytes,
    dropped: Arc<AtomicBool>,
}
impl fmt::Display for Rejected {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("source capacity rejected")
    }
}
impl Error for Rejected {}
impl Drop for Rejected {
    fn drop(&mut self) {
        assert!(self.owner.0.custody.attachments.try_lock().is_ok());
        self.dropped.store(true, Ordering::SeqCst);
    }
}

#[test]
fn rejected_provider_preserves_prior_custody_and_error_drops_after_unlock() {
    let source = bytes();
    let first = SharedStorageDomain::default();
    let second = SharedStorageDomain::default();
    let used = Arc::new(AtomicUsize::new(0));
    attach(&source, &first, &used);
    let capacity = source.capacity_bytes().unwrap() as usize;
    let dropped = Arc::new(AtomicBool::new(false));
    let error = source
        .try_attach(&second, || {
            Err(Rejected {
                owner: source.clone(),
                dropped: dropped.clone(),
            })
        })
        .unwrap_err();
    assert!(matches!(error, SharedStorageAttachmentError::Provider(_)));
    assert_eq!(
        error.source().unwrap().to_string(),
        "source capacity rejected"
    );
    assert_eq!(used.load(Ordering::SeqCst), capacity);
    assert_eq!(source.0.custody.attachments.lock().unwrap().len(), 1);
    drop(error);
    assert!(dropped.load(Ordering::SeqCst));
    assert!(attach(&source, &second, &used));
    assert_eq!(used.load(Ordering::SeqCst), capacity * 2);
    drop(source);
    assert_eq!(used.load(Ordering::SeqCst), 0);
}

#[test]
fn provider_panic_preserves_old_charge_and_poison_blocks_all_new_acquisition() {
    let source = bytes();
    let alias = source.clone();
    let first = SharedStorageDomain::default();
    let second = SharedStorageDomain::default();
    let used = Arc::new(AtomicUsize::new(0));
    attach(&source, &first, &used);
    let capacity = source.capacity_bytes().unwrap() as usize;
    assert!(catch_unwind(AssertUnwindSafe(|| {
        let _ = source.try_attach::<Infallible>(&second, || panic!("closed provider panic"));
    }))
    .is_err());
    for domain in [&first, &second] {
        assert!(matches!(
            source.try_attach::<Infallible>(domain, || panic!("poison must reject")),
            Err(SharedStorageAttachmentError::Poisoned)
        ));
    }
    drop(source);
    assert_eq!(used.load(Ordering::SeqCst), capacity);
    drop(alias);
    assert_eq!(used.load(Ordering::SeqCst), 0);
}

struct RetireProbe {
    payload_retired: Arc<AtomicBool>,
    custody_retired: Arc<AtomicBool>,
    other: SharedControllerBytes,
    domain: SharedStorageDomain,
}
impl Drop for RetireProbe {
    fn drop(&mut self) {
        assert!(self.payload_retired.load(Ordering::SeqCst));
        assert!(self
            .other
            .try_attach(&self.domain, || Ok::<_, Infallible>(
                Box::new(()) as Box<dyn Send + Sync>
            ))
            .unwrap());
        self.custody_retired.store(true, Ordering::SeqCst);
    }
}

#[test]
fn closed_payload_retires_before_reentrant_custody_drop_even_after_poison() {
    for poison in [false, true] {
        let payload_retired = Arc::new(AtomicBool::new(false));
        let source = bytes_with_retirement(&payload_retired);
        let custody_retired = Arc::new(AtomicBool::new(false));
        let other = bytes();
        let other_alias = other.clone();
        let domain = SharedStorageDomain::default();
        source
            .try_attach(&domain, || {
                Ok::<_, Infallible>(Box::new(RetireProbe {
                    payload_retired: payload_retired.clone(),
                    custody_retired: custody_retired.clone(),
                    other,
                    domain: domain.clone(),
                }) as Box<dyn Send + Sync>)
            })
            .unwrap();
        let alias = source.clone();
        if poison {
            assert!(catch_unwind(AssertUnwindSafe(|| {
                let _ = source.try_attach::<Infallible>(&SharedStorageDomain::default(), || {
                    panic!("poison before final retirement")
                });
            }))
            .is_err());
        }
        drop(source);
        assert!(!payload_retired.load(Ordering::SeqCst));
        assert!(!custody_retired.load(Ordering::SeqCst));
        drop(alias);
        assert!(payload_retired.load(Ordering::SeqCst));
        assert!(custody_retired.load(Ordering::SeqCst));
        assert!(!other_alias
            .try_attach::<Infallible>(&domain, || panic!("reentrant attachment was lost"))
            .unwrap());
    }
}

#[test]
fn mixed_inventory_preserves_kind_and_never_misreports_complete_filter_only_storage() {
    assert!(TextControllerStorage::Unknown.shared_sources().is_none());
    assert!(TextControllerStorage::Unknown.shared_filters().is_none());
    assert_eq!(
        TextControllerStorage::RunOwned
            .shared_sources()
            .unwrap()
            .count(),
        0
    );
    let filters = [SharedTokenFilter::new(TokenFilter::Allowed(vec![
        true, false, true,
    ]))];
    let bytes = [bytes(), SharedControllerBytes::new(Vec::new(), crate::HostPreparationAuthority::unmanaged())];
    let mixed = TextControllerStorage::RunOwnedWithSharedStorage {
        filters: &filters,
        bytes: &bytes,
    };
    assert!(mixed.shared_filters().is_none());
    let sources = mixed.shared_sources().unwrap().collect::<Vec<_>>();
    assert_eq!(sources.len(), 3);
    assert!(matches!(sources[0], SharedControllerSource::Filter(_)));
    assert!(matches!(sources[1], SharedControllerSource::Bytes(_)));
    assert_eq!(sources[0].identity(), filters[0].identity());
    assert_eq!(sources[1].identity(), bytes[0].identity());
    assert_eq!(sources[2].capacity_bytes(), Some(0));
    let legacy = TextControllerStorage::RunOwnedWithSharedFilters(&filters);
    assert_eq!(legacy.shared_sources().unwrap().count(), 1);
    assert_eq!(legacy.shared_filters().unwrap().as_ptr(), filters.as_ptr());
    let compatible = TextControllerStorage::RunOwnedWithSharedStorage {
        filters: &filters,
        bytes: &[],
    };
    assert_eq!(
        compatible.shared_filters().unwrap().as_ptr(),
        filters.as_ptr()
    );
    let zero_bytes = TextControllerStorage::RunOwnedWithSharedStorage {
        filters: &[],
        bytes: &bytes[1..],
    };
    assert!(zero_bytes.shared_filters().is_none());
    let domain = SharedStorageDomain::default();
    let used = Arc::new(AtomicUsize::new(0));
    let expected = sources
        .iter()
        .map(|s| s.capacity_bytes().unwrap() as usize)
        .sum::<usize>();
    for source in sources {
        assert!(source
            .try_attach(&domain, || Ok::<_, Infallible>(Charge::acquire(
                source, &domain, &used
            )))
            .unwrap());
    }
    assert_eq!(used.load(Ordering::SeqCst), expected);
    drop((filters, bytes));
    assert_eq!(used.load(Ordering::SeqCst), 0);
}

#[test]
fn forced_overrides_preserve_borrowed_post_callback_source_witness() {
    let filters = [SharedTokenFilter::new(TokenFilter::Allowed(vec![
        true, true, false,
    ]))];
    let retired = Arc::new(AtomicBool::new(false));
    let bytes = [bytes_with_retirement(&retired)];
    let legacy = crate::TokenSamplingDecision::new(TokenFilter::All)
        .with_shared_tokenizer_validity(&filters[0]);
    assert!(legacy.controller_storage().is_none());
    let mut decision =
        crate::TokenSamplingDecision::new(TokenFilter::Allowed(vec![true, true, false]))
            .with_shared_tokenizer_validity(&filters[0])
            .with_controller_storage(TextControllerStorage::RunOwnedWithSharedStorage {
                filters: &filters,
                bytes: &bytes,
            });
    assert!(!retired.load(Ordering::SeqCst));
    for forced in [vec![false, true, false], vec![true, false, false]] {
        decision.override_filter(TokenFilter::Allowed(forced.clone()));
        assert_eq!(decision.filter().allowed_mask(), Some(forced.as_slice()));
        let storage = decision
            .controller_storage()
            .expect("source witness survives override");
        assert!(storage.shared_filters().is_none());
        let sources = storage.shared_sources().unwrap().collect::<Vec<_>>();
        assert_eq!(sources.len(), 2);
        assert_eq!(sources[0].identity(), filters[0].identity());
        assert_eq!(sources[1].identity(), bytes[0].identity());
        assert!(decision
            .shared_tokenizer_validity()
            .unwrap()
            .same_storage(&filters[0]));
        let crate::capture::CaptureTokenFilter::Fixed(capture_filter) = decision.capture_domain().unwrap().filter else { panic!("fixed pre-forcing filter"); };
        assert_eq!(
            capture_filter.allowed_mask(),
            Some(&[true, true, false][..])
        );
    }
    drop(decision);
    drop(bytes);
    assert!(retired.load(Ordering::SeqCst));
}
