use super::*;
use std::sync::{Barrier, TryLockError};

// Scope custody is private. This ledger fixture uses real registered roots and
// real counted scopes, then attaches the exact inventories that closed copy and
// transfer workers can add. The residual companion exercises the public Prompt
// constructor instead of this fixture-only field attachment.
fn attach_pin(
    run: &WorkingMemoryFundingRun,
    pin: WorkingMemoryStorage<u32>,
) -> WorkingMemoryFundingScope {
    let mut scope = run.scope().unwrap();
    scope.borrowed_storage = Some(RegisteredStoragePin::new(pin));
    scope
}

fn distinct_scopes() -> (
    MemoryLedger,
    WorkingMemoryReservation,
    WorkingMemoryFundingRun,
    Vec<WorkingMemoryFundingScope>,
) {
    let pool = device_ledger(400, 0).unwrap();
    let originals = pool
        .register_device_storage_individually([(1u32, 40), (2, 24), (3, 16), (4, 0)])
        .unwrap();
    let inherited = pool.pin_registered_storage([(1u32, 40)]).unwrap();
    let (metadata, run) = with_borrowed_storage(device_reservation(&pool, 100, 400), inherited)
        .into_funding()
        .unwrap();
    let scopes = vec![
        run.scope().unwrap(),
        attach_pin(
            &run,
            pool.pin_registered_storage([(1u32, 40), (2, 24)]).unwrap(),
        ),
        attach_pin(&run, pool.pin_registered_storage([(3u32, 16)]).unwrap()),
        attach_pin(&run, pool.pin_registered_storage([(4u32, 0)]).unwrap()),
    ];
    drop(originals);
    assert_eq!(balances(&pool), (100, 80, 180));
    (pool, metadata, run, scopes)
}

fn assert_roots(pool: &MemoryLedger, retained: [bool; 4]) {
    for ((key, bytes), retained) in [(1u32, 40), (2, 24), (3, 16), (4, 0)]
        .into_iter()
        .zip(retained)
    {
        match pool.pin_registered_storage([(key, bytes)]) {
            Ok(pin) => {
                assert!(retained, "unexpected retained source {key}");
                drop(pin);
            }
            Err(error) => {
                assert!(!retained, "missing retained source {key}: {error}");
                assert_eq!(error, WorkingMemoryError::IdentityMismatch);
            }
        }
    }
}

#[test]
fn quarantine_preserves_distinct_overlapping_and_zero_byte_pins_in_every_drop_order() {
    for a in 0..4 {
        for b in 0..4 {
            for c in 0..4 {
                if a == b || a == c || b == c {
                    continue;
                }
                let d = (0..4).find(|i| *i != a && *i != b && *i != c).unwrap();
                let (pool, metadata, run, scopes) = distinct_scopes();
                drop((run, metadata));
                let mut scopes = scopes.into_iter().map(Some).collect::<Vec<_>>();
                for index in [a, b, c, d] {
                    drop(scopes[index].take());
                    assert_eq!(balances(&pool), (100, 80, 180));
                }
                assert_roots(&pool, [true; 4]);
                assert_eq!(pool.device_capacity().unwrap(), 400);
                blocked(&pool);
            }
        }
    }
}

#[test]
fn certified_siblings_release_only_their_own_distinct_pins() {
    for certified in 0u8..16 {
        let (pool, metadata, run, scopes) = distinct_scopes();
        drop((run, metadata));
        for (index, scope) in scopes.into_iter().enumerate() {
            if certified & (1 << index) != 0 {
                scope.certify().unwrap();
            } else {
                drop(scope);
            }
        }
        let retained = [
            certified & 0b0011 != 0b0011,
            certified & 0b0010 == 0,
            certified & 0b0100 == 0,
            certified & 0b1000 == 0,
        ];
        let registered = [40, 24, 16, 0]
            .into_iter()
            .zip(retained)
            .filter_map(|(bytes, keep)| keep.then_some(bytes))
            .sum();
        assert_eq!(
            balances(&pool),
            (if certified == 15 { 0 } else { 100 }, registered, 180)
        );
        assert_roots(&pool, retained);
        if certified == 15 {
            crate::working_memory::memory_fixture::assert_unquoted_idle(&pool);
        } else {
            blocked(&pool);
        }
    }
}

#[test]
fn poisoned_cleanup_keeps_every_later_source_bundle_and_does_not_certify_it() {
    for reverse in [false, true] {
        let (pool, metadata, run, mut scopes) = distinct_scopes();
        let id = run.id;
        drop((run, metadata));
        if reverse {
            scopes.reverse();
        }
        let failure = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _usage = pool.0.usage.lock().unwrap();
            panic!("poison before distinct source cleanup");
        }));
        assert!(failure.is_err());
        // certify returning Poisoned must leave its still-active scope to the
        // same conservative Drop path, including that scope's complete pins.
        assert_eq!(
            scopes.remove(0).certify(),
            Err(WorkingMemoryError::Poisoned)
        );
        drop(scopes);
        assert_eq!(pool.device_used_bytes(), Err(WorkingMemoryError::Poisoned));
        let usage = pool.0.usage.lock().unwrap_or_else(|p| p.into_inner());
        let account = usage.funding.get(&id).unwrap();
        assert!(account.quarantined);
        assert_eq!(account.scopes, 0);
        assert!(!account.run_open);
        assert!(!account.metadata_live);
        assert_eq!(
            (
                usage.domains[1].reserved,
                usage.domains[1].registered,
                usage.domains[1].peak
            ),
            (100, 80, 180)
        );
        // No poison clearing or settlement is inferred by this inspection.
    }
}

#[test]
fn concurrent_failed_scopes_retain_the_union_of_actual_registered_roots() {
    let (pool, metadata, run, scopes) = distinct_scopes();
    drop((run, metadata));
    let barrier = Arc::new(Barrier::new(scopes.len()));
    std::thread::scope(|threads| {
        for scope in scopes {
            let barrier = barrier.clone();
            threads.spawn(move || {
                barrier.wait();
                drop(scope);
            });
        }
    });
    assert_eq!(balances(&pool), (100, 80, 180));
    assert_roots(&pool, [true; 4]);
    blocked(&pool);
}

struct ProviderProbe {
    pool: MemoryLedger,
    clones: AtomicUsize,
    drops: AtomicUsize,
}

struct ProbeKey {
    id: u32,
    probe: Arc<ProviderProbe>,
}

impl Clone for ProbeKey {
    fn clone(&self) -> Self {
        assert!(!matches!(
            self.probe.pool.0.usage.try_lock(),
            Err(TryLockError::WouldBlock)
        ));
        self.probe.clones.fetch_add(1, AtomicOrdering::SeqCst);
        Self {
            id: self.id,
            probe: self.probe.clone(),
        }
    }
}
impl PartialEq for ProbeKey {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}
impl Eq for ProbeKey {}
impl PartialOrd for ProbeKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for ProbeKey {
    fn cmp(&self, other: &Self) -> Ordering {
        self.id.cmp(&other.id)
    }
}
impl Drop for ProbeKey {
    fn drop(&mut self) {
        assert!(!matches!(
            self.probe.pool.0.usage.try_lock(),
            Err(TryLockError::WouldBlock)
        ));
        self.probe.drops.fetch_add(1, AtomicOrdering::SeqCst);
    }
}

#[test]
fn quarantine_moves_provider_pins_without_callbacks_and_certified_keys_drop_unlocked() {
    for certify_second in [false, true] {
        let pool = device_ledger(400, 0).unwrap();
        let probe = Arc::new(ProviderProbe {
            pool: pool.clone(),
            clones: AtomicUsize::new(0),
            drops: AtomicUsize::new(0),
        });
        let key = |id| ProbeKey {
            id,
            probe: probe.clone(),
        };
        let original = pool
            .register_device_storage([(key(1), 40), (key(2), 24)])
            .unwrap();
        let first_pin = pool.pin_registered_storage([(key(1), 40)]).unwrap();
        let second_pin = pool.pin_registered_storage([(key(2), 24)]).unwrap();
        let (metadata, run) = with_borrowed_storage(device_reservation(&pool, 100, 400), first_pin)
            .into_funding()
            .unwrap();
        let first = run.scope().unwrap();
        let mut second = run.scope().unwrap();
        second.borrowed_storage = Some(RegisteredStoragePin::new(second_pin));
        drop((original, run, metadata));
        let before = (
            probe.clones.load(AtomicOrdering::SeqCst),
            probe.drops.load(AtomicOrdering::SeqCst),
        );
        drop(first);
        assert_eq!(
            (
                probe.clones.load(AtomicOrdering::SeqCst),
                probe.drops.load(AtomicOrdering::SeqCst)
            ),
            before
        );
        if certify_second {
            second.certify().unwrap();
            assert_eq!(probe.clones.load(AtomicOrdering::SeqCst), before.0);
            assert!(probe.drops.load(AtomicOrdering::SeqCst) > before.1);
            assert_eq!(balances(&pool), (100, 40, 164));
        } else {
            drop(second);
            assert_eq!(
                (
                    probe.clones.load(AtomicOrdering::SeqCst),
                    probe.drops.load(AtomicOrdering::SeqCst)
                ),
                before
            );
            assert_eq!(balances(&pool), (100, 64, 164));
        }
    }
}
