use super::*;
use std::{
    cmp::Ordering,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering as AtomicOrdering},
        Mutex,
    },
};

struct ProviderKey {
    id: u32,
    pool: WorkingMemoryPool,
    clones: Arc<AtomicUsize>,
    panic_on_clone: usize,
}

impl Clone for ProviderKey {
    fn clone(&self) -> Self {
        assert!(
            self.pool.0.usage.try_lock().is_ok(),
            "provider clone ran while accounting was locked"
        );
        let clone = self.clones.fetch_add(1, AtomicOrdering::SeqCst) + 1;
        assert_ne!(clone, self.panic_on_clone, "provider clone failure");
        Self {
            id: self.id,
            pool: self.pool.clone(),
            clones: self.clones.clone(),
            panic_on_clone: self.panic_on_clone,
        }
    }
}

impl PartialEq for ProviderKey {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}
impl Eq for ProviderKey {}
impl PartialOrd for ProviderKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for ProviderKey {
    fn cmp(&self, other: &Self) -> Ordering {
        self.id.cmp(&other.id)
    }
}

#[test]
fn provider_clone_failure_precedes_accounting_and_drops_only_unregistered_owners() {
    for individual in [false, true] {
        let pool = WorkingMemoryPool::new(100, 5).unwrap();
        let existing = pool.register_storage([(7u32, 10)]).unwrap();
        let clones = Arc::new(AtomicUsize::new(0));
        let key = |id| ProviderKey {
            id,
            pool: pool.clone(),
            clones: clones.clone(),
            // Individual staging has already built one complete result entry
            // before the next entry's key fails to clone.
            panic_on_clone: if individual { 4 } else { 2 },
        };
        let failure = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let inventory = [(key(1), 20), (key(2), 30)];
            if individual {
                drop(pool.register_storage_individually(inventory));
            } else {
                drop(pool.register_storage(inventory));
            }
        }));
        assert!(failure.is_err());
        assert_eq!(pool.used_bytes().unwrap(), 15);
        assert_eq!(pool.peak_bytes().unwrap(), 15);
        drop(existing);
        assert_eq!(pool.used_bytes().unwrap(), 5);
        let fresh = pool.register_storage_individually([(1u32, 95)]).unwrap();
        assert_eq!(pool.used_bytes().unwrap(), 100);
        drop(fresh);
        assert_eq!(pool.used_bytes().unwrap(), 5);
    }
}

#[test]
fn individual_registration_rejects_poison_without_losing_existing_retirement() {
    let pool = WorkingMemoryPool::new(100, 0).unwrap();
    let mut owners = pool
        .register_storage_individually([(1u32, 20), (2, 30)])
        .unwrap();
    let failure = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _guard = pool.0.usage.lock().unwrap();
        panic!("poison accounting fixture");
    }));
    assert!(failure.is_err());
    for inventory in [vec![], vec![(1u32, 20)], vec![(3, 10)]] {
        assert!(matches!(
            pool.register_storage_individually(inventory.clone()),
            Err(WorkingMemoryError::Poisoned)
        ));
        assert!(matches!(
            pool.register_storage(inventory),
            Err(WorkingMemoryError::Poisoned)
        ));
    }
    drop(owners.remove(&1));
    {
        let usage = pool.0.usage.lock().unwrap_err().into_inner();
        assert_eq!(usage.registered, 30);
        assert_eq!(usage.peak, 50);
    }
    drop(owners);
    let usage = pool.0.usage.lock().unwrap_err().into_inner();
    assert_eq!(usage.registered, 0);
    assert_eq!(usage.peak, 50);
    assert!(usage.storage.is_empty());
}

#[test]
fn per_key_owner_overflow_rejects_the_whole_inventory_before_commit() {
    let pool = WorkingMemoryPool::new(100, 0).unwrap();
    let original = pool.register_storage([(2u32, 40)]).unwrap();
    let set_owners = |owners| {
        pool.0
            .usage
            .lock()
            .unwrap()
            .storage
            .get_mut(&TypeId::of::<u32>())
            .unwrap()
            .downcast_mut::<Registry<u32>>()
            .unwrap()
            .get_mut(&2)
            .unwrap()
            .owners = owners;
    };
    set_owners(usize::MAX);
    assert!(matches!(
        pool.register_storage_individually([(1u32, 10), (2, 40)]),
        Err(WorkingMemoryError::Overflow)
    ));
    assert!(matches!(
        pool.register_storage([(1u32, 10), (2, 40)]),
        Err(WorkingMemoryError::Overflow)
    ));
    assert_eq!(pool.used_bytes().unwrap(), 40);
    assert_eq!(pool.peak_bytes().unwrap(), 40);
    set_owners(1);
    drop(original);
    assert_eq!(pool.used_bytes().unwrap(), 0);
    let replacement = pool.register_storage_individually([(1u32, 100)]).unwrap();
    assert_eq!(pool.used_bytes().unwrap(), 100);
    drop(replacement);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[derive(Default)]
struct KeyEvents {
    armed: AtomicBool,
    reregister: AtomicBool,
    panic_on_drop: AtomicBool,
    key_drops: AtomicUsize,
    payload_drops: AtomicUsize,
    replacement: Mutex<Option<WorkingMemoryStorage<PinnedKey>>>,
}

struct PayloadPin {
    pool: WorkingMemoryPool,
    events: Arc<KeyEvents>,
}

impl Drop for PayloadPin {
    fn drop(&mut self) {
        assert_eq!(self.pool.used_bytes().unwrap(), 60);
        self.events
            .payload_drops
            .fetch_add(1, AtomicOrdering::SeqCst);
    }
}

#[derive(Clone)]
struct PinnedKey {
    id: u32,
    pin: Arc<PayloadPin>,
}

impl PartialEq for PinnedKey {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}
impl Eq for PinnedKey {}
impl PartialOrd for PinnedKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for PinnedKey {
    fn cmp(&self, other: &Self) -> Ordering {
        self.id.cmp(&other.id)
    }
}

impl Drop for PinnedKey {
    fn drop(&mut self) {
        let events = &self.pin.events;
        if !events.armed.load(AtomicOrdering::SeqCst) {
            return;
        }
        assert!(self.pin.pool.0.usage.try_lock().is_ok());
        assert!(self.pin.pool.used_bytes().unwrap() >= 60);
        assert_eq!(events.payload_drops.load(AtomicOrdering::SeqCst), 0);
        events.key_drops.fetch_add(1, AtomicOrdering::SeqCst);
        assert!(
            !events.panic_on_drop.swap(false, AtomicOrdering::SeqCst),
            "provider key destructor failure"
        );
        if events.reregister.swap(false, AtomicOrdering::SeqCst) {
            // The old registry entry has retired, but its identity keys still
            // pin payload. The replacement must overlap that old charge.
            let replacement = self
                .pin
                .pool
                .register_storage([(self.clone(), 60)])
                .unwrap();
            assert_eq!(self.pin.pool.used_bytes().unwrap(), 120);
            *events.replacement.lock().unwrap() = Some(replacement);
        }
    }
}

#[test]
fn provider_keys_drop_unlocked_before_payload_bytes_are_refunded() {
    let pool = WorkingMemoryPool::new(120, 0).unwrap();
    let events = Arc::new(KeyEvents::default());
    let pin = Arc::new(PayloadPin {
        pool: pool.clone(),
        events: events.clone(),
    });
    let key = || PinnedKey {
        id: 1,
        pin: pin.clone(),
    };
    let grouped = pool.register_storage([(key(), 60)]).unwrap();
    events.armed.store(true, AtomicOrdering::SeqCst);
    // Duplicate incoming provider keys must also be destroyed outside the
    // commit lock. The result map's identity key retires before its handle.
    let individual = pool
        .register_storage_individually([(key(), 60)])
        .unwrap()
        .into_values()
        .next()
        .unwrap();
    assert!(events.key_drops.load(AtomicOrdering::SeqCst) >= 2);
    drop(pin);
    drop(grouped);
    assert_eq!(pool.used_bytes().unwrap(), 60);
    assert_eq!(events.payload_drops.load(AtomicOrdering::SeqCst), 0);
    events.reregister.store(true, AtomicOrdering::SeqCst);
    drop(individual);
    assert_eq!(pool.used_bytes().unwrap(), 60);
    assert_eq!(pool.peak_bytes().unwrap(), 120);
    assert_eq!(events.payload_drops.load(AtomicOrdering::SeqCst), 0);
    let replacement = events.replacement.lock().unwrap().take().unwrap();
    drop(replacement);
    assert_eq!(events.payload_drops.load(AtomicOrdering::SeqCst), 1);
    assert_eq!(pool.used_bytes().unwrap(), 0);
    assert_eq!(pool.peak_bytes().unwrap(), 120);
}

#[test]
fn provider_destructor_panic_retains_the_charge_without_poisoning_accounting() {
    let pool = WorkingMemoryPool::new(120, 0).unwrap();
    let events = Arc::new(KeyEvents::default());
    let pin = Arc::new(PayloadPin {
        pool: pool.clone(),
        events: events.clone(),
    });
    let owner = pool
        .register_storage([(PinnedKey { id: 1, pin }, 60)])
        .unwrap();
    events.armed.store(true, AtomicOrdering::SeqCst);
    events.panic_on_drop.store(true, AtomicOrdering::SeqCst);
    let failure = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(owner)));
    assert!(failure.is_err());
    assert_eq!(events.payload_drops.load(AtomicOrdering::SeqCst), 1);
    assert_eq!(pool.used_bytes().unwrap(), 60);
    assert_eq!(pool.peak_bytes().unwrap(), 60);
    let next = pool.register_storage_individually([(2u32, 60)]).unwrap();
    assert_eq!(pool.used_bytes().unwrap(), 120);
    drop(next);
    assert_eq!(pool.used_bytes().unwrap(), 60);
}
