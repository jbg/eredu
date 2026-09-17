use super::*;
use std::{
    convert::Infallible,
    error::Error,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{
        Barrier,
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
        mpsc,
    },
};

struct Charge {
    used: Arc<AtomicU64>,
    bytes: u64,
    retired: Option<(Arc<AtomicUsize>, usize)>,
    _identity: HostMetadataIdentity,
}
impl Drop for Charge {
    fn drop(&mut self) {
        if let Some((count, expected)) = &self.retired {
            assert_eq!(
                count.load(Ordering::SeqCst),
                *expected,
                "all slots precede the charge"
            );
        }
        self.used.fetch_sub(self.bytes, Ordering::SeqCst);
    }
}
fn attach(
    token: &HostSlotMetadata,
    domain: &SharedStorageDomain,
    used: &Arc<AtomicU64>,
    retired: Option<(Arc<AtomicUsize>, usize)>,
) -> bool {
    token
        .try_attach(domain, || {
            let bytes = token.capacity_bytes().unwrap();
            used.fetch_add(bytes, Ordering::SeqCst);
            Ok::<Box<dyn Send + Sync>, Infallible>(Box::new(Charge {
                used: used.clone(),
                bytes,
                retired,
                _identity: token.identity().clone(),
            }))
        })
        .unwrap()
}

#[test]
fn fixed_extent_covers_inline_optional_slots_without_pricing_nested_capacity() {
    let mut string = String::with_capacity(4096);
    string.push_str("nested payload is separate");
    let mut table = HostSlotTable::new(Box::new([Some(string), None]));
    let id = table.metadata().identity().clone();
    let bytes = 2 * size_of::<Option<String>>() as u64;
    assert_eq!(table.metadata().capacity_bytes(), Some(bytes));
    assert_eq!(table.metadata().slot_size(), size_of::<Option<String>>());
    assert_eq!(table.len(), 2);
    let moved = table.slots_mut()[0].take();
    table.slots_mut()[1] = moved;
    assert_eq!(
        table.slots()[1].as_deref(),
        Some("nested payload is separate")
    );
    assert_eq!(table.metadata().identity(), &id);
    assert_eq!(table.metadata().capacity_bytes(), Some(bytes));
    let token = table.metadata().clone();
    let table = table;
    assert!(table.metadata().same_storage(&token));
    let independent = HostSlotTable::<Option<String>>::new(Box::new([None, None]));
    assert_eq!(independent.metadata().capacity_bytes(), Some(bytes));
    assert_ne!(independent.metadata().identity(), token.identity());
}

#[test]
fn empty_and_zero_sized_tables_preserve_distinct_identity_and_zero_capacity() {
    let empty = HostSlotTable::<u64>::new(Box::new([]));
    let zst = HostSlotTable::new(Box::new([(); 19]));
    assert!(empty.is_empty());
    assert!(empty.metadata().is_empty());
    assert!(!zst.is_empty());
    assert_eq!(zst.metadata().len(), 19);
    assert_eq!(zst.metadata().slot_size(), 0);
    assert_eq!(empty.metadata().capacity_bytes(), Some(0));
    assert_eq!(zst.metadata().capacity_bytes(), Some(0));
    assert_ne!(empty.metadata().identity(), zst.metadata().identity());
    let equal = HostSlotTable::<u64>::new(Box::new([]));
    assert_eq!(empty.slots(), equal.slots());
    assert_ne!(empty.metadata().identity(), equal.metadata().identity());
    let token = zst.metadata().clone();
    let domain = SharedStorageDomain::default();
    assert!(
        token
            .try_attach(&domain, || Ok::<Box<dyn Send + Sync>, Infallible>(
                Box::new(())
            ))
            .unwrap()
    );
    drop(zst);
    assert!(matches!(
        token.try_attach::<Infallible>(&domain, || panic!("retired zero extent")),
        Err(HostSlotAttachmentError::Retired)
    ));
}

struct SlotDrop(Arc<AtomicUsize>);
impl Drop for SlotDrop {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn prepared_table_metadata_and_identity_retain_host_custody_but_registry_key_does_not() {
    struct Preparation {
        slots_retired: Arc<AtomicUsize>,
        retired: Arc<AtomicUsize>,
    }
    impl Drop for Preparation {
        fn drop(&mut self) {
            assert_eq!(self.slots_retired.load(Ordering::SeqCst), 2);
            self.retired.fetch_add(1, Ordering::SeqCst);
        }
    }
    for identity_first in [false, true] {
        let slots_retired = Arc::new(AtomicUsize::new(0));
        let retired = Arc::new(AtomicUsize::new(0));
        let authority = eredu_core::HostPreparationAuthority::retain(Preparation {
            slots_retired: slots_retired.clone(),
            retired: retired.clone(),
        });
        let identity = HostMetadataIdentity::prepared_host(&authority).unwrap();
        let table = HostSlotTable::new_prepared_host(
            Box::new([
                SlotDrop(slots_retired.clone()),
                SlotDrop(slots_retired.clone()),
            ]),
            identity,
            &authority,
        );
        let metadata = table.metadata().clone();
        let identity = metadata.identity().clone();
        let key = identity.registry_key().clone();
        let domain = SharedStorageDomain::default();
        let used = Arc::new(AtomicU64::new(0));
        assert!(attach(
            &metadata,
            &domain,
            &used,
            Some((slots_retired.clone(), 2))
        ));
        drop((authority, table));
        assert_eq!(slots_retired.load(Ordering::SeqCst), 2);
        assert_eq!(retired.load(Ordering::SeqCst), 0);
        assert!(matches!(
            metadata.try_attach::<Infallible>(&domain, || panic!("retired table cannot be reused")),
            Err(HostSlotAttachmentError::Retired)
        ));
        if identity_first {
            drop(identity);
            assert_eq!(retired.load(Ordering::SeqCst), 0);
            drop(metadata);
        } else {
            drop(metadata);
            assert_eq!(used.load(Ordering::SeqCst), 0);
            assert_eq!(retired.load(Ordering::SeqCst), 0);
            drop(identity);
        }
        assert_eq!(used.load(Ordering::SeqCst), 0);
        assert_eq!(retired.load(Ordering::SeqCst), 1);
        // Retaining the accounting key cannot keep its own account alive.
        drop(key);
        assert_eq!(retired.load(Ordering::SeqCst), 1);
    }
}

#[test]
fn payload_precedes_charge_and_escaped_tokens_only_prolong_existing_custody() {
    for escape in [false, true] {
        let retired = Arc::new(AtomicUsize::new(0));
        let table = HostSlotTable::new(Box::new([
            SlotDrop(retired.clone()),
            SlotDrop(retired.clone()),
        ]));
        let used = Arc::new(AtomicU64::new(0));
        let bytes = table.metadata().capacity_bytes().unwrap();
        let first = SharedStorageDomain::default();
        let second = SharedStorageDomain::default();
        assert!(attach(
            table.metadata(),
            &first,
            &used,
            Some((retired.clone(), 2))
        ));
        assert!(attach(
            table.metadata(),
            &second,
            &used,
            Some((retired.clone(), 2))
        ));
        let key = table.metadata().identity().clone();
        let tokens = escape.then(|| (table.metadata().clone(), table.metadata().clone()));
        drop(table);
        assert_eq!(retired.load(Ordering::SeqCst), 2);
        if let Some((token, last)) = tokens {
            assert_eq!(used.load(Ordering::SeqCst), 2 * bytes);
            assert_eq!(token.identity(), &key);
            assert_eq!(token.capacity_bytes(), Some(bytes));
            for domain in [&first, &SharedStorageDomain::default()] {
                assert!(matches!(
                    token.try_attach::<Infallible>(domain, || panic!(
                        "retired token is not a source"
                    )),
                    Err(HostSlotAttachmentError::Retired)
                ));
            }
            drop(token);
            assert_eq!(used.load(Ordering::SeqCst), 2 * bytes);
            drop(last);
        }
        // An independent identity key never holds the payload or registrations.
        assert_eq!(used.load(Ordering::SeqCst), 0);
        drop(key);
    }
}

#[test]
fn concurrent_live_attachments_acquire_once_for_each_domain() {
    fn send_sync<T: Send + Sync>() {}
    send_sync::<HostSlotMetadata>();
    let table = HostSlotTable::new(Box::new([17u32, 29, 31]));
    let domains = [
        SharedStorageDomain::default(),
        SharedStorageDomain::default(),
    ];
    let barrier = Barrier::new(8);
    let used = Arc::new(AtomicU64::new(0));
    let count = std::thread::scope(|threads| {
        let handles = (0..8)
            .map(|index| {
                let token = table.metadata();
                let domain = &domains[index % 2];
                let barrier = &barrier;
                let used = &used;
                threads.spawn(move || {
                    barrier.wait();
                    attach(token, domain, used, None)
                })
            })
            .collect::<Vec<_>>();
        handles
            .into_iter()
            .map(|handle| usize::from(handle.join().unwrap()))
            .sum::<usize>()
    });
    assert_eq!(count, 2);
    assert_eq!(used.load(Ordering::SeqCst), 24);
    drop(table);
    assert_eq!(used.load(Ordering::SeqCst), 0);
}

#[derive(Debug)]
struct ReenterError {
    token: HostSlotMetadata,
    domain: SharedStorageDomain,
    dropped: Arc<AtomicBool>,
}
impl fmt::Display for ReenterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("exact provider rejection")
    }
}
impl Error for ReenterError {}
impl Drop for ReenterError {
    fn drop(&mut self) {
        assert!(
            self.token
                .try_attach(&self.domain, || Ok::<Box<dyn Send + Sync>, Infallible>(
                    Box::new(())
                ))
                .unwrap()
        );
        self.dropped.store(true, Ordering::SeqCst);
    }
}

#[test]
fn provider_error_keeps_prior_custody_and_reenters_after_both_locks_release() {
    let table = HostSlotTable::new(Box::new([17u64, 29]));
    let token = table.metadata().clone();
    let used = Arc::new(AtomicU64::new(0));
    attach(&token, &SharedStorageDomain::default(), &used, None);
    let domain = SharedStorageDomain::default();
    let dropped = Arc::new(AtomicBool::new(false));
    let error = token
        .try_attach(&domain, || {
            Err(ReenterError {
                token: token.clone(),
                domain: domain.clone(),
                dropped: dropped.clone(),
            })
        })
        .unwrap_err();
    assert!(matches!(
        &error,
        HostSlotAttachmentError::Attachment(SharedStorageAttachmentError::Provider(_))
    ));
    assert_eq!(
        error.source().unwrap().source().unwrap().to_string(),
        "exact provider rejection"
    );
    assert_eq!(used.load(Ordering::SeqCst), 16);
    drop(error);
    assert!(dropped.load(Ordering::SeqCst));
    assert!(
        !token
            .try_attach::<Infallible>(&domain, || panic!("error destructor already attached"))
            .unwrap()
    );
    drop((table, token));
    assert_eq!(used.load(Ordering::SeqCst), 0);
}

struct SkippedProviderDrop {
    token: HostSlotMetadata,
    domain: SharedStorageDomain,
    dropped: Arc<AtomicBool>,
}
impl Drop for SkippedProviderDrop {
    fn drop(&mut self) {
        assert!(
            self.token
                .try_attach(&self.domain, || Ok::<Box<dyn Send + Sync>, Infallible>(
                    Box::new(())
                ))
                .unwrap()
        );
        self.dropped.store(true, Ordering::SeqCst);
    }
}

#[test]
fn duplicate_provider_captures_retire_after_the_outer_lifecycle_gate() {
    let table = HostSlotTable::new(Box::new([17u32]));
    let token = table.metadata();
    let domain = SharedStorageDomain::default();
    token
        .try_attach(&domain, || {
            Ok::<Box<dyn Send + Sync>, Infallible>(Box::new(()))
        })
        .unwrap();
    let dropped = Arc::new(AtomicBool::new(false));
    let capture = SkippedProviderDrop {
        token: token.clone(),
        domain: SharedStorageDomain::default(),
        dropped: dropped.clone(),
    };
    assert!(
        !token
            .try_attach::<Infallible>(&domain, move || {
                let _capture = capture;
                panic!("duplicate must not invoke provider")
            })
            .unwrap()
    );
    assert!(dropped.load(Ordering::SeqCst));
}

#[test]
fn poison_preserves_existing_custody_and_does_not_lock_hot_payload_mutation() {
    let mut table = HostSlotTable::new(Box::new([17u32, 29]));
    let token = table.metadata().clone();
    let used = Arc::new(AtomicU64::new(0));
    attach(&token, &SharedStorageDomain::default(), &used, None);
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            let _ = token.try_attach::<Infallible>(&SharedStorageDomain::default(), || {
                panic!("injected acquisition panic")
            });
        }))
        .is_err()
    );
    table.slots_mut()[0] = 41;
    assert_eq!(table.slots(), [41, 29]);
    assert!(matches!(
        token.try_attach::<Infallible>(&SharedStorageDomain::default(), || panic!(
            "poison rejects before provider"
        )),
        Err(HostSlotAttachmentError::Attachment(
            SharedStorageAttachmentError::Poisoned
        ))
    ));
    assert_eq!(used.load(Ordering::SeqCst), 8);
    drop(table);
    assert_eq!(used.load(Ordering::SeqCst), 8);
    drop(token);
    assert_eq!(used.load(Ordering::SeqCst), 0);
}

struct RetiringSlot {
    token: Option<HostSlotMetadata>,
    retired: Arc<AtomicBool>,
}
impl Drop for RetiringSlot {
    fn drop(&mut self) {
        let token = self.token.as_ref().unwrap();
        assert!(matches!(
            token.try_attach::<Infallible>(&SharedStorageDomain::default(), || panic!(
                "payload Drop sees retired source"
            )),
            Err(HostSlotAttachmentError::Retired)
        ));
        self.retired.store(true, Ordering::SeqCst);
    }
}

#[test]
fn concurrent_provider_finishes_before_retirement_and_payload_drop_reenters_unlocked() {
    let retired = Arc::new(AtomicBool::new(false));
    let mut table = HostSlotTable::new(Box::new([RetiringSlot {
        token: None,
        retired: retired.clone(),
    }]));
    let token = table.metadata().clone();
    table.slots_mut()[0].token = Some(token.clone());
    let (entered_tx, entered_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let (dropping_tx, dropping_rx) = mpsc::channel();
    let token_copy = token.clone();
    let retired_copy = retired.clone();
    let provider = std::thread::spawn(move || {
        token_copy
            .try_attach(&SharedStorageDomain::default(), || {
                entered_tx.send(()).unwrap();
                release_rx.recv().unwrap();
                assert!(!retired_copy.load(Ordering::SeqCst));
                Ok::<Box<dyn Send + Sync>, Infallible>(Box::new(()))
            })
            .unwrap()
    });
    entered_rx.recv().unwrap();
    let dropping = std::thread::spawn(move || {
        dropping_tx.send(()).unwrap();
        drop(table);
    });
    dropping_rx.recv().unwrap();
    assert!(!retired.load(Ordering::SeqCst));
    release_tx.send(()).unwrap();
    assert!(provider.join().unwrap());
    dropping.join().unwrap();
    assert!(retired.load(Ordering::SeqCst));
    assert!(matches!(
        token.try_attach::<Infallible>(&SharedStorageDomain::default(), || panic!("retired")),
        Err(HostSlotAttachmentError::Retired)
    ));
}

struct AttachOtherOnDrop {
    other: HostSlotMetadata,
    domain: SharedStorageDomain,
    dropped: Arc<AtomicBool>,
}
impl Drop for AttachOtherOnDrop {
    fn drop(&mut self) {
        assert!(
            self.other
                .try_attach(&self.domain, || Ok::<Box<dyn Send + Sync>, Infallible>(
                    Box::new(())
                ))
                .unwrap()
        );
        self.dropped.store(true, Ordering::SeqCst);
    }
}

#[test]
fn final_attached_handle_can_reenter_another_live_table_outside_owner_locks() {
    let source = HostSlotTable::new(Box::new([17u32]));
    let other = HostSlotTable::new(Box::new([29u32]));
    let domain = SharedStorageDomain::default();
    let dropped = Arc::new(AtomicBool::new(false));
    source
        .metadata()
        .try_attach(&SharedStorageDomain::default(), || {
            Ok::<Box<dyn Send + Sync>, Infallible>(Box::new(AttachOtherOnDrop {
                other: other.metadata().clone(),
                domain: domain.clone(),
                dropped: dropped.clone(),
            }))
        })
        .unwrap();
    drop(source);
    assert!(dropped.load(Ordering::SeqCst));
    assert!(
        !other
            .metadata()
            .try_attach::<Infallible>(&domain, || panic!("attached during other owner retirement"))
            .unwrap()
    );
}
