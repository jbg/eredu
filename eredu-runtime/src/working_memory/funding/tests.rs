use crate::working_memory::memory_fixture::separate::*;

mod quarantine_pins;
mod reservation_validation;
mod terminal_host;

use super::*;
use crate::working_memory::InferenceExecutionIdentity;
use eredu_core::{
    Admission, EstimationCompleteness, ExecutionWorkspaceEstimate, InferenceGeometry,
    InputTokenCount, LayerSchedule, OutputDemand, StateMemoryLayout, WorkspaceBound,
    cache::LayerCachePolicy,
};
use std::{
    cmp::Ordering,
    sync::{
        Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering as AtomicOrdering},
    },
};

// A portable stateless execution whose explicit workspace is the envelope being
// tested. These ledger tests make no native allocation or quote claim.
pub(in crate::working_memory) fn reservation(
    pool: &MemoryLedger,
    bytes: u64,
    capacity: u64,
) -> WorkingMemoryReservation {
    try_reservation(pool, bytes, capacity).unwrap()
}

fn try_reservation(
    pool: &MemoryLedger,
    bytes: u64,
    capacity: u64,
) -> Result<WorkingMemoryReservation, WorkingMemoryError> {
    pool.reserve_with_capacity(
        &InferenceExecutionIdentity::default(),
        &crate::working_memory::memory_fixture::host_admission(pool, bytes),
        crate::working_memory::memory_fixture::resolved_host_limits(pool, capacity),
    )
}

fn blocked(pool: &MemoryLedger) {
    assert!(matches!(
        pool.acquire_unquoted(),
        Err(WorkingMemoryError::ReservedWorkActive)
    ));
}

fn balances(pool: &MemoryLedger) -> (u64, u64, u64) {
    let usage = pool.0.usage.lock().unwrap();
    (
        usage.domains[1].reserved,
        usage.domains[1].registered,
        usage.domains[1].peak,
    )
}

fn with_borrowed_storage<K: Ord + Send + Sync + 'static>(
    mut reservation: WorkingMemoryReservation,
    storage: WorkingMemoryStorage<K>,
) -> WorkingMemoryReservation {
    reservation.0.get_mut().unwrap().borrowed_storage = Some(RegisteredStoragePin::new(storage));
    reservation
}

#[test]
fn conversion_moves_borrowed_pins_from_metadata_into_run_and_independent_scopes() {
    for scope_count in [0, 2] {
        let pool = device_ledger(400, 0).unwrap();
        let original_storage = pool.register_device_storage([(1u32, 40)]).unwrap();
        let pin = pool.pin_registered_storage([(1u32, 40)]).unwrap();
        let original = with_borrowed_storage(device_reservation(&pool, 100, 200), pin);
        drop(original_storage);
        assert_eq!(balances(&pool), (100, 40, 140));
        let (metadata, run) = original.into_funding().unwrap();
        assert!(metadata.0.borrowed_storage.is_none());
        let metadata_clone = metadata.clone();
        assert!(metadata_clone.0.borrowed_storage.is_none());
        let scopes = (0..scope_count)
            .map(|_| run.scope().unwrap())
            .collect::<Vec<_>>();
        drop((run, metadata));
        if scope_count != 0 {
            assert_eq!(balances(&pool), (100, 40, 140));
        }
        let mut scopes = scopes.into_iter();
        while let Some(scope) = scopes.next() {
            scope.certify().unwrap();
            if scopes.len() != 0 {
                assert_eq!(balances(&pool), (100, 40, 140));
            }
        }
        assert_eq!(balances(&pool), (0, 0, 140));
        assert_eq!(device_request_bytes(&metadata_clone), Some(100));
        assert_eq!(pool.device_capacity().unwrap(), 200);
        blocked(&pool);
        drop(metadata_clone);
        assert_eq!(pool.device_capacity().unwrap(), 400);
        crate::working_memory::memory_fixture::assert_unquoted_idle(&pool);
    }
}

#[test]
fn uncertified_scopes_keep_one_borrowed_pin_in_permanent_account_custody() {
    for poison in [false, true] {
        let pool = device_ledger(400, 0).unwrap();
        let original_storage = pool.register_device_storage([(1u32, 40)]).unwrap();
        let pin = pool.pin_registered_storage([(1u32, 40)]).unwrap();
        let (metadata, run) = with_borrowed_storage(device_reservation(&pool, 100, 200), pin)
            .into_funding()
            .unwrap();
        let first = run.scope().unwrap();
        let second = run.scope().unwrap();
        drop(original_storage);
        if poison {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _guard = pool.0.usage.lock().unwrap();
                panic!("poison before certifying borrowed storage");
            }));
            assert!(result.is_err());
            assert!(matches!(first.certify(), Err(WorkingMemoryError::Poisoned)));
        } else {
            drop(first);
        }
        // A later failed scope must neither replace nor drop the account's pin.
        let id = second.id;
        drop((second, run, metadata));
        let usage = pool.0.usage.lock().unwrap_or_else(|p| p.into_inner());
        let account = usage.funding.get(&id).unwrap();
        assert!(account.quarantined);
        assert!(account.quarantined_borrowed.is_some());
        assert_eq!(account.scopes, 0);
        assert!(!account.run_open);
        assert!(!account.metadata_live);
        assert_eq!(
            (
                usage.domains[1].reserved,
                usage.domains[1].registered,
                usage.domains[1].peak
            ),
            (100, 40, 140)
        );
        drop(usage);
        if !poison {
            blocked(&pool);
            assert_eq!(pool.device_capacity().unwrap(), 200);
        }
    }
}

#[test]
fn conversion_requires_fresh_unique_evidence_and_preserves_legacy_lifetime() {
    let pool = device_ledger(500, 0).unwrap();
    let original = device_reservation(&pool, 100, 200);
    assert!(matches!(
        original.clone().into_funding(),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    assert_eq!(balances(&pool), (100, 0, 100));
    assert_eq!(pool.device_capacity().unwrap(), 200);
    *original.0.start.lock().unwrap() = RequestStart::Started(None);
    assert!(matches!(
        original.into_funding(),
        Err(WorkingMemoryError::AlreadyStarted)
    ));
    assert_eq!(pool.device_used_bytes().unwrap(), 0);
    assert_eq!(pool.device_capacity().unwrap(), 500);

    let original = device_reservation(&pool, 100, 200);
    let clone = original.clone();
    drop(original);
    assert_eq!(pool.device_used_bytes().unwrap(), 100);
    drop(clone);
    assert_eq!(pool.device_used_bytes().unwrap(), 0);
}

#[test]
fn closing_releases_workspace_but_metadata_keeps_domain_and_ceiling() {
    for bytes in [0, 100] {
        let pool = device_ledger(500, 0).unwrap();
        let original = device_reservation(&pool, bytes, 200);
        assert!(!original.requires_funding_scope());
        let geometry = original.geometry();
        let (metadata, run) = original.into_funding().unwrap();
        assert!(metadata.requires_funding_scope());
        assert_eq!(metadata.geometry(), geometry);
        assert_eq!(device_request_bytes(&metadata), Some(bytes));
        assert!(run.pool().same_ledger(&pool));
        let clone = metadata.clone();
        run.close().unwrap();
        assert!(metadata.requires_funding_scope());
        let request = crate::working_memory::InferenceRequest::from(&metadata);
        assert!(request.requires_funding_scope());
        drop(request);
        assert_eq!(pool.device_used_bytes().unwrap(), 0);
        assert_eq!(device_request_bytes(&metadata), Some(bytes));
        assert_eq!(pool.device_peak_bytes().unwrap(), bytes);
        assert_eq!(pool.device_capacity().unwrap(), 200);
        blocked(&pool);
        drop(metadata);
        blocked(&pool);
        drop(clone);
        assert_eq!(pool.device_capacity().unwrap(), 500);
        crate::working_memory::memory_fixture::assert_unquoted_idle(&pool);
    }
}

#[test]
fn owned_scopes_keep_workspace_after_run_close_until_every_certification() {
    let pool = device_ledger(100, 0).unwrap();
    let (metadata, run) = device_reservation(&pool, 100, 100).into_funding().unwrap();
    let first = run.scope().unwrap();
    let second = run.scope().unwrap();
    drop((run, metadata));
    assert_eq!(pool.device_used_bytes().unwrap(), 100);
    first.certify().unwrap();
    assert_eq!(pool.device_used_bytes().unwrap(), 100);
    blocked(&pool);
    second.certify().unwrap();
    assert_eq!(pool.device_used_bytes().unwrap(), 0);
    crate::working_memory::memory_fixture::assert_unquoted_idle(&pool);
}

#[test]
fn adoption_is_atomic_and_retirement_returns_only_the_origin_credit() {
    let pool = device_ledger(500, 0).unwrap();
    let (metadata, run) = device_reservation(&pool, 100, 150).into_funding().unwrap();
    let scope = run.scope().unwrap();
    let mut storage = scope
        .adopt_device_storage([(1u32, 30), (2, 50), (1, 30)])
        .unwrap();
    assert_eq!(balances(&pool), (20, 80, 100));
    let alias = pool.register_device_storage([(1u32, 30)]).unwrap();
    drop(storage.remove(&1));
    assert_eq!(balances(&pool), (20, 80, 100));
    drop(alias);
    assert_eq!(balances(&pool), (50, 50, 100));
    let alias = pool.register_device_storage([(2u32, 50)]).unwrap();
    drop(storage);
    scope.certify().unwrap();
    drop((run, metadata));
    assert_eq!(balances(&pool), (0, 50, 100));
    assert_eq!(pool.device_capacity().unwrap(), 150);
    blocked(&pool);
    drop(alias);
    assert_eq!(balances(&pool), (0, 0, 100));
    assert_eq!(pool.device_capacity().unwrap(), 500);
    crate::working_memory::memory_fixture::assert_unquoted_idle(&pool);
}

#[test]
fn adoption_rejects_whole_batches_without_changing_balances_or_peak() {
    let pool = device_ledger(200, 0).unwrap();
    let (metadata, run) = device_reservation(&pool, 100, 150).into_funding().unwrap();
    let scope = run.scope().unwrap();
    let storage = scope.adopt_device_storage([(1u32, 40)]).unwrap();
    for inventory in [vec![(2u32, 20), (1, 41)], vec![(2, 20), (2, 21)]] {
        assert!(matches!(
            scope.adopt_device_storage(inventory),
            Err(WorkingMemoryError::StorageCapacityMismatch { .. })
        ));
        assert_eq!(balances(&pool), (60, 40, 100));
    }
    assert!(matches!(
        scope.adopt_device_storage([(2u32, 30), (3, 31)]),
        Err(WorkingMemoryError::DomainAllowanceExceeded {
            domain: _,
            required_bytes: 61,
            available_bytes: 60
        })
    ));
    assert!(matches!(
        scope.adopt_device_storage([(2u32, u64::MAX), (3, 1)]),
        Err(WorkingMemoryError::Domain(
            eredu_core::MemoryDomainError::Overflow
        ))
    ));
    let foreign = device_ledger(200, 0).unwrap();
    assert!(matches!(
        foreign.adopt_storage_individually(
            &scope,
            [(
                2u32,
                crate::working_memory::StorageAllocation::new(1, device_placement(&foreign))
            )]
        ),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    assert_eq!(balances(&pool), (60, 40, 100));
    assert_eq!(balances(&foreign), (0, 0, 0));
    let exact = scope.adopt_device_storage([(2u32, 60)]).unwrap();
    assert_eq!(balances(&pool), (0, 100, 100));
    drop((storage, exact));
    assert_eq!(balances(&pool), (100, 0, 100));
    scope.certify().unwrap();
    drop((run, metadata));
    assert_eq!(pool.device_used_bytes().unwrap(), 0);
}

#[test]
fn partial_publication_failure_restores_and_quarantines_the_complete_envelope() {
    let pool = device_ledger(100, 0).unwrap();
    let (metadata, run) = device_reservation(&pool, 100, 100).into_funding().unwrap();
    let scope = run.scope().unwrap();
    let mut inventory = scope.adopt_device_storage([(1u32, 40), (2, 30)]).unwrap();
    let attached = inventory.remove(&1).unwrap();
    drop(inventory); // Attachment of key 2 failed; no physical handle escaped.
    assert_eq!(balances(&pool), (60, 40, 100));
    drop(scope); // No settlement/publication certificate was available.
    assert!(matches!(
        run.scope(),
        Err(WorkingMemoryError::ExecutionFenced)
    ));
    drop((run, metadata));
    assert_eq!(balances(&pool), (60, 40, 100));
    drop(attached);
    assert_eq!(balances(&pool), (100, 0, 100));
    blocked(&pool);
}

#[test]
fn cross_funding_aliases_transfer_zero_and_keep_each_original_ceiling() {
    let pool = device_ledger(1000, 0).unwrap();
    let baseline = pool.register_device_storage([(0u32, 20)]).unwrap();
    let (a, a_run) = device_reservation(&pool, 100, 250).into_funding().unwrap();
    let (b, b_run) = device_reservation(&pool, 100, 300).into_funding().unwrap();
    let a_scope = a_run.scope().unwrap();
    let b_scope = b_run.scope().unwrap();
    let a_storage = a_scope.adopt_device_storage([(1u32, 40)]).unwrap();
    let b_storage = b_scope
        .adopt_device_storage([(0u32, 20), (1, 40), (2, 30)])
        .unwrap();
    assert_eq!(balances(&pool), (130, 90, 220));
    a_scope.certify().unwrap();
    b_scope.certify().unwrap();
    drop((a_run, a, a_storage));
    assert_eq!(balances(&pool), (70, 90, 220));
    assert_eq!(pool.device_capacity().unwrap(), 250);
    drop((b_run, b, baseline));
    assert_eq!(balances(&pool), (0, 90, 220));
    drop(b_storage);
    assert_eq!(balances(&pool), (0, 0, 220));
    assert_eq!(pool.device_capacity().unwrap(), 1000);
    crate::working_memory::memory_fixture::assert_unquoted_idle(&pool);
}

#[test]
fn zero_transfer_alias_descendant_keeps_exclusion_after_original_metadata_retires() {
    let pool = device_ledger(200, 0).unwrap();
    let baseline = pool.register_device_storage([(0u32, 40)]).unwrap();
    let (metadata, run) = device_reservation(&pool, 0, 80).into_funding().unwrap();
    let scope = run.scope().unwrap();
    let storage = scope.adopt_device_storage([(0u32, 40), (1, 0)]).unwrap();
    scope.certify().unwrap();
    drop((metadata, run));
    blocked(&pool);
    assert_eq!(pool.device_capacity().unwrap(), 80);
    assert_eq!(balances(&pool), (0, 40, 40));
    drop(storage);
    assert_eq!(pool.device_capacity().unwrap(), 200);
    crate::working_memory::memory_fixture::assert_unquoted_idle(&pool);
    drop(baseline);
}

#[test]
fn concurrent_adoption_deduplicates_aliases_and_consumes_one_atomic_envelope() {
    let pool = device_ledger(100, 0).unwrap();
    let (metadata, run) = device_reservation(&pool, 100, 100).into_funding().unwrap();
    let scope = run.scope().unwrap();
    let barrier = std::sync::Barrier::new(8);
    let results = std::thread::scope(|threads| {
        let workers = (1u32..=8)
            .map(|id| {
                let scope = &scope;
                let barrier = &barrier;
                threads.spawn(move || {
                    barrier.wait();
                    scope.adopt_device_storage([(0u32, 60), (id, 10)])
                })
            })
            .collect::<Vec<_>>();
        workers
            .into_iter()
            .map(|worker| worker.join().unwrap())
            .collect::<Vec<_>>()
    });
    assert_eq!(results.iter().filter(|result| result.is_ok()).count(), 4);
    for error in results.iter().filter_map(|result| result.as_ref().err()) {
        assert_eq!(
            *error,
            WorkingMemoryError::DomainAllowanceExceeded {
                domain: device_domain(&pool),
                required_bytes: 10,
                available_bytes: 0
            }
        );
    }
    assert_eq!(balances(&pool), (0, 100, 100));
    drop(results);
    assert_eq!(balances(&pool), (100, 0, 100));
    scope.certify().unwrap();
    drop((metadata, run));
    assert_eq!(pool.device_used_bytes().unwrap(), 0);
}

struct Events {
    pool: MemoryLedger,
    close: Mutex<Option<WorkingMemoryFundingRun>>,
    armed: AtomicBool,
    panic_once: AtomicBool,
    clone_failure: AtomicBool,
    payload_drops: AtomicUsize,
}

struct Pin(Arc<Events>);
impl Drop for Pin {
    fn drop(&mut self) {
        if self.0.armed.load(AtomicOrdering::SeqCst) {
            assert_eq!(balances(&self.0.pool).1, 40);
            self.0.payload_drops.fetch_add(1, AtomicOrdering::SeqCst);
        }
    }
}

struct Key {
    id: u32,
    pin: Arc<Pin>,
}
impl Clone for Key {
    fn clone(&self) -> Self {
        assert!(self.pin.0.pool.0.usage.try_lock().is_ok());
        assert!(
            !self.pin.0.clone_failure.swap(false, AtomicOrdering::SeqCst),
            "provider clone failure"
        );
        Self {
            id: self.id,
            pin: self.pin.clone(),
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
        let events = &self.pin.0;
        if events.armed.load(AtomicOrdering::SeqCst) {
            assert!(events.pool.0.usage.try_lock().is_ok());
            if let Some(run) = events.close.lock().unwrap().take() {
                run.close().unwrap();
            }
            assert_eq!(balances(&events.pool).1, 40);
            assert_eq!(events.payload_drops.load(AtomicOrdering::SeqCst), 0);
            assert!(
                !events.panic_once.swap(false, AtomicOrdering::SeqCst),
                "provider drop failure"
            );
        }
    }
}

#[test]
fn borrowed_pin_keys_drop_unlocked_after_certification_and_can_reenter_funding() {
    let pool = device_ledger(400, 0).unwrap();
    let events = Arc::new(Events {
        pool: pool.clone(),
        close: Mutex::new(None),
        armed: AtomicBool::new(false),
        panic_once: AtomicBool::new(false),
        clone_failure: AtomicBool::new(false),
        payload_drops: AtomicUsize::new(0),
    });
    let key = Key {
        id: 1,
        pin: Arc::new(Pin(events.clone())),
    };
    let original_storage = pool.register_device_storage([(key.clone(), 40)]).unwrap();
    let pin = pool.pin_registered_storage([(key, 40)]).unwrap();
    let (metadata, run) = with_borrowed_storage(device_reservation(&pool, 100, 200), pin)
        .into_funding()
        .unwrap();
    let (other_metadata, other_run) = device_reservation(&pool, 10, 200).into_funding().unwrap();
    *events.close.lock().unwrap() = Some(other_run);
    let scope = run.scope().unwrap();
    drop((original_storage, run, metadata, other_metadata));
    assert_eq!(balances(&pool), (110, 40, 150));
    events.armed.store(true, AtomicOrdering::SeqCst);
    // Key::drop verifies the usage mutex is unlocked and closes the other
    // funding account. Pin::drop checks its physical bytes are still charged.
    scope.certify().unwrap();
    assert_eq!(events.payload_drops.load(AtomicOrdering::SeqCst), 1);
    assert!(events.close.lock().unwrap().is_none());
    assert_eq!(balances(&pool), (0, 0, 150));
    crate::working_memory::memory_fixture::assert_unquoted_idle(&pool);
}

#[test]
fn provider_keys_retire_outside_lock_before_credit_or_ceiling_is_released() {
    for panic_on_drop in [false, true] {
        let pool = device_ledger(100, 0).unwrap();
        let (metadata, run) = device_reservation(&pool, 100, 100).into_funding().unwrap();
        let scope = run.scope().unwrap();
        let events = Arc::new(Events {
            pool: pool.clone(),
            close: Mutex::new(Some(run)),
            armed: AtomicBool::new(false),
            panic_once: AtomicBool::new(panic_on_drop),
            clone_failure: AtomicBool::new(false),
            payload_drops: AtomicUsize::new(0),
        });
        let key = Key {
            id: 1,
            pin: Arc::new(Pin(events.clone())),
        };
        let storage = scope
            .adopt_device_storage([(key, 40)])
            .unwrap()
            .into_values()
            .next()
            .unwrap();
        scope.certify().unwrap();
        drop(metadata);
        events.armed.store(true, AtomicOrdering::SeqCst);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(storage)));
        assert_eq!(result.is_err(), panic_on_drop);
        assert_eq!(events.payload_drops.load(AtomicOrdering::SeqCst), 1);
        assert_eq!(
            pool.device_used_bytes().unwrap(),
            if panic_on_drop { 40 } else { 0 }
        );
        if panic_on_drop {
            blocked(&pool);
        } else {
            crate::working_memory::memory_fixture::assert_unquoted_idle(&pool);
        }
    }
}

#[test]
fn provider_clone_failure_and_poison_do_not_commit_funding() {
    let pool = device_ledger(100, 0).unwrap();
    let (metadata, run) = device_reservation(&pool, 100, 100).into_funding().unwrap();
    let scope = run.scope().unwrap();
    let events = Arc::new(Events {
        pool: pool.clone(),
        close: Mutex::new(None),
        armed: AtomicBool::new(false),
        panic_once: AtomicBool::new(false),
        clone_failure: AtomicBool::new(true),
        payload_drops: AtomicUsize::new(0),
    });
    let key = Key {
        id: 1,
        pin: Arc::new(Pin(events)),
    };
    let failure = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        drop(scope.adopt_device_storage([(key, 40)]));
    }));
    assert!(failure.is_err());
    assert_eq!(balances(&pool), (100, 0, 100));
    assert!(matches!(
        run.scope(),
        Err(WorkingMemoryError::ExecutionFenced)
    ));
    drop((scope, run, metadata));
    blocked(&pool);
    let pool = device_ledger(100, 0).unwrap();
    let (metadata, run) = device_reservation(&pool, 100, 100).into_funding().unwrap();
    let scope = run.scope().unwrap();
    let storage = scope.adopt_device_storage([(1u32, 40)]).unwrap();
    let failure = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _guard = pool.0.usage.lock().unwrap();
        panic!("poison accounting fixture");
    }));
    assert!(failure.is_err());
    assert!(matches!(
        scope.adopt_device_storage([(2u32, 30)]),
        Err(WorkingMemoryError::Poisoned)
    ));
    assert!(matches!(scope.certify(), Err(WorkingMemoryError::Poisoned)));
    drop((storage, metadata, run));
    let usage = pool.0.usage.lock().unwrap_err().into_inner();
    assert_eq!(
        (
            usage.domains[1].reserved,
            usage.domains[1].registered,
            usage.domains[1].peak
        ),
        (100, 0, 100)
    );
    assert_eq!(usage.reservations, 1);
    assert!(usage.storage.is_empty());
}

mod start_funding;

mod prepared_scope;
