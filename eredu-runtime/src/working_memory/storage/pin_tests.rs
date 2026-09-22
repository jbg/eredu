use super::*;
use crate::working_memory::memory_fixture::separate::*;
use std::{
    cmp::Ordering,
    sync::atomic::{AtomicBool, AtomicUsize, Ordering as AtomicOrdering},
};

fn accounting(pool: &MemoryLedger) -> (u64, u64, u64, usize, usize) {
    let usage = pool.0.usage.lock().unwrap();
    (
        usage.domains[1].reserved,
        usage.domains[1].registered,
        usage.domains[1].peak,
        usage.reservations,
        usage.funding.len(),
    )
}

#[test]
fn grouped_pins_retain_exact_aliases_and_zero_byte_keys_without_new_charges() {
    let pool = device_ledger(60, 0).unwrap();
    let mut original = pool
        .register_device_storage_individually([(1u32, 20), (2, 40), (3, 0)])
        .unwrap();
    let before = accounting(&pool);
    let pin = pool
        .pin_registered_storage([(1u32, 20), (3, 0), (1, 20)])
        .unwrap();
    assert_eq!(pin.bytes(), Some(20));
    assert_eq!(accounting(&pool), before);
    assert_eq!(pool.device_capacity().unwrap(), 60);
    let clone = pin.clone();
    drop(original.remove(&1));
    drop(original.remove(&3));
    drop(pin);
    assert_eq!(pool.device_used_bytes().unwrap(), 60);
    let zero = pool.pin_registered_storage([(3u32, 0)]).unwrap();
    drop(clone);
    assert_eq!(pool.device_used_bytes().unwrap(), 40);
    assert!(matches!(
        pool.pin_registered_storage([(1u32, 20)]),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    drop(zero);
    assert!(matches!(
        pool.pin_registered_storage([(3u32, 0)]),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    drop(original);
    assert_eq!(pool.device_used_bytes().unwrap(), 0);
    assert_eq!(pool.device_peak_bytes().unwrap(), 60);
    let empty = pool
        .pin_registered_storage(Vec::<(u32, u64)>::new())
        .unwrap();
    assert_eq!(empty.bytes(), Some(0));
    assert_eq!(accounting(&pool), (0, 0, 60, 0, 0));
}

#[test]
fn missing_capacity_conflicts_and_foreign_namespaces_reject_every_pin_atomically() {
    let pool = device_ledger(100, 0).unwrap();
    let original = pool.register_device_storage([(1u32, 20), (2, 40)]).unwrap();
    for inventory in [
        vec![(1u32, 20), (3, 10)],
        vec![(1, 20), (2, 41)],
        vec![(1, 20), (1, 21)],
        vec![(1, 20), (3, 0)],
    ] {
        assert!(matches!(
            pool.pin_registered_storage(inventory),
            Err(WorkingMemoryError::IdentityMismatch)
        ));
        assert_eq!(accounting(&pool), (0, 60, 60, 0, 0));
    }
    assert!(matches!(
        pool.pin_registered_storage([(1u64, 20)]),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    let foreign = device_ledger(100, 0).unwrap();
    assert!(matches!(
        foreign.pin_registered_storage([(1u32, 20)]),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    assert_eq!(accounting(&foreign), (0, 0, 0, 0, 0));
    drop(original);
    // An earlier valid key in a rejected batch did not acquire an owner.
    assert_eq!(pool.device_used_bytes().unwrap(), 0);
    assert!(pool.0.usage.lock().unwrap().storage.is_empty());
}

fn funded(
    pool: &MemoryLedger,
) -> (
    crate::working_memory::WorkingMemoryReservation,
    crate::working_memory::WorkingMemoryFundingRun,
) {
    device_reservation(pool, 100, 150).into_funding().unwrap()
}

#[test]
fn pins_preserve_funding_origin_credit_and_ceiling_without_a_new_account() {
    for close_before_pin_retires in [false, true] {
        let pool = device_ledger(500, 0).unwrap();
        let (metadata, run) = funded(&pool);
        let scope = run.scope().unwrap();
        let adopted = scope.adopt_device_storage([(1u32, 40)]).unwrap();
        let before = accounting(&pool);
        let pin = pool.pin_registered_storage([(1u32, 40)]).unwrap();
        assert_eq!(accounting(&pool), before);
        drop(adopted);
        // The pin preserves the allocation origin, not another funded handle.
        {
            let usage = pool.0.usage.lock().unwrap();
            assert_eq!(usage.funding.values().next().unwrap().registrations, 0);
            assert_eq!(usage.funding.values().next().unwrap().allocations, 2);
        }
        scope.certify().unwrap();
        if close_before_pin_retires {
            drop((metadata, run));
            assert_eq!(accounting(&pool), (0, 40, 100, 1, 1));
            assert_eq!(pool.device_capacity().unwrap(), 150);
            assert!(matches!(
                pool.acquire_unquoted(),
                Err(WorkingMemoryError::ReservedWorkActive)
            ));
            drop(pin);
        } else {
            drop(pin);
            assert_eq!(accounting(&pool), (100, 0, 100, 1, 1));
            drop((metadata, run));
        }
        assert_eq!(accounting(&pool), (0, 0, 100, 0, 0));
        assert_eq!(pool.device_capacity().unwrap(), 500);
        crate::working_memory::memory_fixture::assert_unquoted_idle(&pool);
    }
}

#[test]
fn concurrent_pins_share_ownership_and_cannot_resurrect_a_retired_key() {
    let pool = device_ledger(60, 0).unwrap();
    let original = pool.register_device_storage([(1u32, 20), (2, 40)]).unwrap();
    let barrier = std::sync::Barrier::new(8);
    let mut pins = std::thread::scope(|threads| {
        let workers = (0..8)
            .map(|_| {
                let pool = &pool;
                let barrier = &barrier;
                threads.spawn(move || {
                    barrier.wait();
                    pool.pin_registered_storage([(1u32, 20), (2, 40)]).unwrap()
                })
            })
            .collect::<Vec<_>>();
        workers
            .into_iter()
            .map(|worker| worker.join().unwrap())
            .collect::<Vec<_>>()
    });
    assert_eq!(accounting(&pool), (0, 60, 60, 0, 0));
    drop(original);
    let last = pins.pop().unwrap();
    drop(pins);
    assert_eq!(pool.device_used_bytes().unwrap(), 60);
    drop(last);
    assert_eq!(pool.device_used_bytes().unwrap(), 0);
    assert!(matches!(
        pool.pin_registered_storage([(1u32, 20), (2, 40)]),
        Err(WorkingMemoryError::IdentityMismatch)
    ));

    // Pinning and final retirement linearize on the same registry lock. Either
    // the pin wins and preserves the charge, or it rejects the retired identity.
    for _ in 0..16 {
        let original = pool.register_device_storage([(3u32, 60)]).unwrap();
        let barrier = std::sync::Barrier::new(2);
        let result = std::thread::scope(|threads| {
            let worker = threads.spawn(|| {
                barrier.wait();
                pool.pin_registered_storage([(3u32, 60)])
            });
            barrier.wait();
            drop(original);
            worker.join().unwrap()
        });
        match result {
            Ok(pin) => {
                assert_eq!(pool.device_used_bytes().unwrap(), 60);
                drop(pin);
            }
            Err(error) => assert_eq!(error, WorkingMemoryError::IdentityMismatch),
        }
        assert_eq!(pool.device_used_bytes().unwrap(), 0);
        assert_eq!(pool.device_peak_bytes().unwrap(), 60);
    }
}

#[test]
fn pin_overflow_and_poison_do_not_mutate_existing_ownership() {
    let pool = device_ledger(60, 0).unwrap();
    let original = pool.register_device_storage([(1u32, 20), (2, 40)]).unwrap();
    {
        let mut usage = pool.0.usage.lock().unwrap();
        usage
            .storage
            .get_mut(&TypeId::of::<u32>())
            .unwrap()
            .downcast_mut::<Registry<u32>>()
            .unwrap()
            .get_mut(&2)
            .unwrap()
            .owners = usize::MAX;
    }
    assert!(matches!(
        pool.pin_registered_storage([(1u32, 20), (2, 40)]),
        Err(WorkingMemoryError::Overflow)
    ));
    {
        let mut usage = pool.0.usage.lock().unwrap();
        let registry = usage
            .storage
            .get_mut(&TypeId::of::<u32>())
            .unwrap()
            .downcast_mut::<Registry<u32>>()
            .unwrap();
        assert_eq!(registry.get(&1).unwrap().owners, 1);
        registry.get_mut(&2).unwrap().owners = 1;
    }
    let pin = pool.pin_registered_storage([(1u32, 20)]).unwrap();
    let failure = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _guard = pool.0.usage.lock().unwrap();
        panic!("poison storage pin fixture");
    }));
    assert!(failure.is_err());
    for inventory in [vec![], vec![(1u32, 20)]] {
        assert!(matches!(
            pool.pin_registered_storage(inventory),
            Err(WorkingMemoryError::Poisoned)
        ));
    }
    drop((original, pin));
    let usage = pool.0.usage.lock().unwrap_err().into_inner();
    assert_eq!(usage.domains[1].registered, 0);
    assert_eq!(usage.domains[1].peak, 60);
    assert!(usage.storage.is_empty());
}

#[derive(Default)]
struct Events {
    armed: AtomicBool,
    clone_failure: AtomicBool,
    drop_failure: AtomicBool,
    payload_drops: AtomicUsize,
}
struct Payload {
    pool: MemoryLedger,
    events: Arc<Events>,
}
impl Drop for Payload {
    fn drop(&mut self) {
        if self.events.armed.load(AtomicOrdering::SeqCst) {
            assert_eq!(self.pool.device_used_bytes().unwrap(), 40);
            self.events
                .payload_drops
                .fetch_add(1, AtomicOrdering::SeqCst);
        }
    }
}
struct Key {
    id: u32,
    payload: Arc<Payload>,
}
impl Clone for Key {
    fn clone(&self) -> Self {
        assert!(self.payload.pool.0.usage.try_lock().is_ok());
        assert!(
            !self
                .payload
                .events
                .clone_failure
                .swap(false, AtomicOrdering::SeqCst),
            "provider clone failure"
        );
        Self {
            id: self.id,
            payload: self.payload.clone(),
        }
    }
}
impl PartialEq for Key {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}
impl Eq for Key {}
impl PartialOrd for Key {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Key {
    fn cmp(&self, other: &Self) -> Ordering {
        self.id.cmp(&other.id)
    }
}
impl Drop for Key {
    fn drop(&mut self) {
        let events = &self.payload.events;
        if events.armed.load(AtomicOrdering::SeqCst) {
            // Reenter accounting while every provider identity still pins its
            // payload. No registry/refund operation may hold the lock here.
            assert!(self.payload.pool.0.usage.try_lock().is_ok());
            assert_eq!(self.payload.pool.device_used_bytes().unwrap(), 40);
            assert_eq!(events.payload_drops.load(AtomicOrdering::SeqCst), 0);
            assert!(
                !events.drop_failure.swap(false, AtomicOrdering::SeqCst),
                "provider drop failure"
            );
        }
    }
}

#[test]
fn pin_moves_provider_identity_without_cloning_and_preserves_final_custody() {
    let pool = device_ledger(40, 0).unwrap();
    let events = Arc::new(Events::default());
    let payload = Arc::new(Payload {
        pool: pool.clone(),
        events: events.clone(),
    });
    let original = pool
        .register_device_storage([(
            Key {
                id: 1,
                payload: payload.clone(),
            },
            40,
        )])
        .unwrap();
    events.clone_failure.store(true, AtomicOrdering::SeqCst);
    let pin = pool
        .pin_registered_storage([(
            Key {
                id: 1,
                payload: payload.clone(),
            },
            40,
        )])
        .unwrap();
    assert!(events.clone_failure.load(AtomicOrdering::SeqCst));
    assert_eq!(accounting(&pool), (0, 40, 40, 0, 0));
    drop(payload);
    events.armed.store(true, AtomicOrdering::SeqCst);
    drop(original);
    assert_eq!(pool.device_used_bytes().unwrap(), 40);
    drop(pin);
    assert_eq!(events.payload_drops.load(AtomicOrdering::SeqCst), 1);
    assert_eq!(pool.device_used_bytes().unwrap(), 0);
}

#[test]
fn final_pin_destroys_provider_keys_unlocked_before_refund_and_retains_charge_on_panic() {
    for panic_on_drop in [false, true] {
        let pool = device_ledger(40, 0).unwrap();
        let events = Arc::new(Events::default());
        let payload = Arc::new(Payload {
            pool: pool.clone(),
            events: events.clone(),
        });
        let original = pool
            .register_device_storage([(
                Key {
                    id: 1,
                    payload: payload.clone(),
                },
                40,
            )])
            .unwrap();
        events.armed.store(true, AtomicOrdering::SeqCst);
        let pin = pool
            .pin_registered_storage([(
                Key {
                    id: 1,
                    payload: payload.clone(),
                },
                40,
            )])
            .unwrap();
        drop((payload, original));
        events
            .drop_failure
            .store(panic_on_drop, AtomicOrdering::SeqCst);
        let failure = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(pin)));
        assert_eq!(failure.is_err(), panic_on_drop);
        assert_eq!(events.payload_drops.load(AtomicOrdering::SeqCst), 1);
        assert_eq!(
            pool.device_used_bytes().unwrap(),
            if panic_on_drop { 40 } else { 0 }
        );
        assert_eq!(pool.device_peak_bytes().unwrap(), 40);
    }
}
