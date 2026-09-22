//! Representation tests only; no original execution authority is constructed.
use super::*;
use std::sync::{
    Mutex, Weak,
    atomic::{AtomicBool, AtomicUsize, Ordering as A},
};
#[derive(Debug)]
struct Key {
    id: u8,
    panic: Arc<AtomicBool>,
    drops: Arc<AtomicUsize>,
    locked: Arc<AtomicBool>,
    registry: Weak<Mutex<Registry<Key>>>,
}
impl PartialEq for Key {
    fn eq(&self, o: &Self) -> bool {
        self.cmp(o) == Ordering::Equal
    }
}
impl Eq for Key {}
impl PartialOrd for Key {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Key {
    fn cmp(&self, o: &Self) -> Ordering {
        assert!(!self.panic.load(A::SeqCst), "provider compare");
        self.id.cmp(&o.id)
    }
}
impl Drop for Key {
    fn drop(&mut self) {
        self.drops.fetch_add(1, A::SeqCst);
        if let Some(r) = self.registry.upgrade() {
            if matches!(r.try_lock(), Err(std::sync::TryLockError::WouldBlock)) {
                self.locked.store(true, A::SeqCst);
            }
        }
    }
}
fn key(
    id: u8,
    registry: &Arc<Mutex<Registry<Key>>>,
    panic: &Arc<AtomicBool>,
    drops: &Arc<AtomicUsize>,
    locked: &Arc<AtomicBool>,
) -> Key {
    Key {
        id,
        panic: panic.clone(),
        drops: drops.clone(),
        locked: locked.clone(),
        registry: Arc::downgrade(registry),
    }
}
fn install(registry: &Arc<Mutex<Registry<Key>>>, key: RegistryKey<Key>, entry: Entry) {
    let mut batch = RegistryBatch::prepare_source_registration(1);
    batch.entries[0] = Some((key, entry));
    registry.lock().unwrap().link(batch);
}
#[test]
fn prepared_batches_use_one_namespace_and_last_node_drops_outside_lock() {
    let registry = Arc::new(Mutex::new(Registry::new()));
    let panic = Arc::new(AtomicBool::new(false));
    let drops = Arc::new(AtomicUsize::new(0));
    let locked = Arc::new(AtomicBool::new(false));
    let make = |id| key(id, &registry, &panic, &drops, &locked);
    let old = make(1);
    let new = make(2);
    let zero = make(3);
    install(
        &registry,
        RegistryKey::Owned(make(1)),
        Entry {
            reset_layout_id: None,
            prepaid: None,
            placement: crate::working_memory::memory_fixture::host_ledger(u64::MAX, 0)
                .unwrap()
                .host_placement_handle(),
            native_retired: false,
            pending_allocation: false,
            funding_allowance_bytes: 0,
            bytes: 9,
            owners: 1,
            funding: None,
        },
    );
    let node = Box::new(RegistryBatch {
        entries: RegistrySlots::Ordinary(
            vec![
                Some((
                    RegistryKey::Owned(make(2)),
                    Entry {
                        reset_layout_id: None,
                        prepaid: None,
                        placement: crate::working_memory::memory_fixture::host_ledger(u64::MAX, 0)
                            .unwrap()
                            .host_placement_handle(),
                        native_retired: false,
                        pending_allocation: false,
                        funding_allowance_bytes: 0,
                        bytes: 7,
                        owners: 2,
                        funding: Some(4),
                    },
                )),
                Some((
                    RegistryKey::Owned(make(3)),
                    Entry {
                        reset_layout_id: None,
                        prepaid: None,
                        placement: crate::working_memory::memory_fixture::host_ledger(u64::MAX, 0)
                            .unwrap()
                            .host_placement_handle(),
                        native_retired: false,
                        pending_allocation: false,
                        funding_allowance_bytes: 0,
                        bytes: 0,
                        owners: 1,
                        funding: Some(4),
                    },
                )),
            ]
            .into_boxed_slice(),
        ),
        next: None,
        raw: None,
        native: None,
        preparation: None,
    });
    registry.lock().unwrap().link(node);
    {
        let r = registry.lock().unwrap();
        assert_eq!(r.get(&old).unwrap().bytes, 9);
        assert_eq!(r.get(&new).unwrap().owners, 2);
        assert_eq!(r.get(&zero).unwrap().funding, Some(4));
    }
    let retired = registry.lock().unwrap().retire_owner(&new);
    assert!(retired.entry.is_none());
    drop(retired);
    let retired = registry.lock().unwrap().retire_owner(&zero);
    assert_eq!(retired.entry.as_ref().unwrap().1.bytes, 0);
    assert!(retired._batch.is_none());
    retired.retire_with_key(zero);
    let retired = registry.lock().unwrap().retire_owner(&new);
    assert!(retired._batch.is_some());
    assert_eq!(retired.entry.as_ref().unwrap().1.funding, Some(4));
    assert_eq!(drops.load(A::SeqCst), 2);
    retired.retire_with_key(new);
    let retired = registry.lock().unwrap().retire_owner(&old);
    retired.retire_with_key(old);
    assert!(registry.lock().unwrap().is_empty());
    assert_eq!(drops.load(A::SeqCst), 6);
    assert!(!locked.load(A::SeqCst));
}
#[test]
fn integer_locator_commit_performs_no_provider_comparison_and_lookup_panic_is_atomic() {
    let registry = Arc::new(Mutex::new(Registry::new()));
    let panic = Arc::new(AtomicBool::new(false));
    let drops = Arc::new(AtomicUsize::new(0));
    let locked = Arc::new(AtomicBool::new(false));
    let make = |id| key(id, &registry, &panic, &drops, &locked);
    let probe = make(1);
    install(
        &registry,
        RegistryKey::Owned(make(1)),
        Entry {
            reset_layout_id: None,
            prepaid: None,
            placement: crate::working_memory::memory_fixture::host_ledger(u64::MAX, 0)
                .unwrap()
                .host_placement_handle(),
            native_retired: false,
            pending_allocation: false,
            funding_allowance_bytes: 0,
            bytes: 9,
            owners: 1,
            funding: None,
        },
    );
    {
        let mut r = registry.lock().unwrap();
        let (locator, _) = r.locate(&probe).unwrap();
        panic.store(true, A::SeqCst);
        r.at_mut(locator).owners += 1;
        panic.store(false, A::SeqCst);
        assert_eq!(r.get(&probe).unwrap().owners, 2);
    }
    panic.store(true, A::SeqCst);
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| registry
            .lock()
            .unwrap()
            .retire_owner(&probe)))
        .is_err()
    );
    panic.store(false, A::SeqCst);
    let r = registry
        .lock()
        .err()
        .expect("poisoned registry")
        .into_inner();
    assert_eq!(r.get(&probe).unwrap().owners, 2);
    drop(r);
    assert_eq!(drops.load(A::SeqCst), 0);
    assert!(!locked.load(A::SeqCst));
}
