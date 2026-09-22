use super::*;
use std::panic::{AssertUnwindSafe, catch_unwind};

#[test]
fn unused_scopes_return_the_same_account_capacity_and_activation_keeps_its_count() {
    let pool = device_ledger(100, 0).unwrap();
    for pass in 0..8 {
        let (metadata, run) = device_reservation(&pool, 100, 100).into_funding().unwrap();
        let id = run.id;
        let prepared = run.prepare_scope().unwrap();
        drop((metadata, run));
        assert_eq!(
            balances(&pool).0,
            100,
            "prepared native count retains headroom"
        );
        {
            let usage = pool.0.usage.lock().unwrap();
            let account = usage.funding.get(&id).unwrap();
            assert_eq!((account.scopes, account.native_scopes), (1, 1));
            assert!(!account.run_open && !account.quarantined);
        }
        if pass % 2 == 0 {
            drop(prepared);
        } else {
            let active = prepared.activate();
            assert_eq!(active.id, id);
            {
                let usage = pool.0.usage.lock().unwrap();
                let account = usage.funding.get(&id).unwrap();
                assert_eq!((account.scopes, account.native_scopes), (1, 1));
            }
            assert!(matches!(
                active.adopt_device_storage([(7u32, 101)]),
                Err(WorkingMemoryError::DomainAllowanceExceeded {
                    domain: _,
                    required_bytes: 101,
                    available_bytes: 100
                })
            ));
            let actual = active.adopt_device_storage([(7u32, 100)]).unwrap();
            assert_eq!(balances(&pool).0, 0);
            assert_eq!(balances(&pool).1, 100);
            active.certify().unwrap();
            assert_eq!(pool.device_used_bytes().unwrap(), 100);
            drop(actual);
        }
        assert_eq!(pool.device_used_bytes().unwrap(), 0);
        {
            let usage = pool.0.usage.lock().unwrap();
            assert!(usage.funding.get(&id).is_none());
            assert!(usage.funding.live_identity(id).is_none());
            assert_eq!(usage.reservations, 0);
        }
        crate::working_memory::memory_fixture::assert_unquoted_idle(&pool);
    }
}

struct DropProbe {
    pool: MemoryLedger,
    drops: AtomicUsize,
}
struct BorrowedKey {
    id: u32,
    probe: Arc<DropProbe>,
}
impl Clone for BorrowedKey {
    fn clone(&self) -> Self {
        assert!(self.probe.pool.0.usage.try_lock().is_ok());
        Self {
            id: self.id,
            probe: self.probe.clone(),
        }
    }
}
impl PartialEq for BorrowedKey {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}
impl Eq for BorrowedKey {}
impl PartialOrd for BorrowedKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for BorrowedKey {
    fn cmp(&self, other: &Self) -> Ordering {
        self.id.cmp(&other.id)
    }
}
impl Drop for BorrowedKey {
    fn drop(&mut self) {
        assert!(
            self.probe.pool.0.usage.try_lock().is_ok(),
            "provider drop outside Usage"
        );
        self.probe.drops.fetch_add(1, AtomicOrdering::SeqCst);
    }
}

#[test]
fn unused_scope_retires_its_real_borrowed_and_zero_byte_sources_after_unlock_even_on_unwind() {
    for unwind in [false, true] {
        let pool = device_ledger(140, 0).unwrap();
        let probe = Arc::new(DropProbe {
            pool: pool.clone(),
            drops: AtomicUsize::new(0),
        });
        let key = |id| BorrowedKey {
            id,
            probe: probe.clone(),
        };
        let roots = pool
            .register_device_storage([(key(1), 40), (key(2), 0)])
            .unwrap();
        let inherited = pool
            .pin_registered_storage([(key(1), 40), (key(2), 0)])
            .unwrap();
        let (metadata, run) = with_borrowed_storage(device_reservation(&pool, 100, 140), inherited)
            .into_funding()
            .unwrap();
        let prepared = run.prepare_scope().unwrap();
        drop((roots, metadata, run));
        assert_eq!(pool.device_used_bytes().unwrap(), 140);
        let before = probe.drops.load(AtomicOrdering::SeqCst);
        let marker = Arc::new(());
        if unwind {
            let expected = marker.clone();
            let failure = catch_unwind(AssertUnwindSafe(move || {
                let _prepared = prepared;
                std::panic::panic_any(marker);
            }))
            .unwrap_err();
            assert!(Arc::ptr_eq(
                failure.downcast_ref::<Arc<()>>().unwrap(),
                &expected
            ));
        } else {
            drop(prepared);
        }
        assert_eq!(pool.device_used_bytes().unwrap(), 0);
        assert!(probe.drops.load(AtomicOrdering::SeqCst) > before);
        assert!(matches!(
            pool.pin_registered_storage([(key(1), 40)]),
            Err(WorkingMemoryError::IdentityMismatch)
        ));
        assert!(matches!(
            pool.pin_registered_storage([(key(2), 0)]),
            Err(WorkingMemoryError::IdentityMismatch)
        ));
    }
}

#[test]
fn unused_cancellation_cannot_clear_sibling_quarantine_and_poison_preserves_every_pin() {
    for poison in [false, true] {
        let pool = device_ledger(400, 0).unwrap();
        let roots = pool.register_device_storage([(1u32, 40), (2, 0)]).unwrap();
        let inherited = pool.pin_registered_storage([(1u32, 40), (2, 0)]).unwrap();
        let (metadata, run) = with_borrowed_storage(device_reservation(&pool, 100, 400), inherited)
            .into_funding()
            .unwrap();
        let id = run.id;
        let prepared = run.prepare_scope().unwrap();
        let sibling = run.scope().unwrap();
        let (other_metadata, other_run) =
            device_reservation(&pool, 70, 400).into_funding().unwrap();
        let other = other_run.prepare_scope().unwrap();
        let other_id = other_run.id;
        drop((roots, metadata, run, other_metadata, other_run));
        drop(sibling); // A real uncertified active scope fences its account.
        if poison {
            assert!(
                catch_unwind(AssertUnwindSafe(|| {
                    let _loan = pool.0.usage.lock().unwrap();
                    panic!("poison before unused cancellation");
                }))
                .is_err()
            );
        }
        drop((prepared, other));
        if poison {
            assert_eq!(pool.device_used_bytes(), Err(WorkingMemoryError::Poisoned));
            let usage = pool.0.usage.lock().unwrap_or_else(|e| e.into_inner());
            for id in [id, other_id] {
                let account = usage.funding.get(&id).unwrap();
                assert!(account.quarantined);
                assert_eq!((account.scopes, account.native_scopes), (0, 0));
            }
            assert_eq!(
                (usage.domains[1].reserved, usage.domains[1].registered),
                (170, 40)
            );
            // Two real bundles: sibling abandonment and prepared poison Drop.
            let pins = usage
                .funding
                .get(&id)
                .unwrap()
                .quarantined_borrowed
                .as_ref()
                .unwrap();
            assert!(pins.next.is_some());
        } else {
            assert_eq!(pool.device_used_bytes().unwrap(), 140);
            let usage = pool.0.usage.lock().unwrap();
            let account = usage.funding.get(&id).unwrap();
            assert!(account.quarantined);
            assert_eq!((account.scopes, account.native_scopes), (0, 0));
            assert!(account.quarantined_borrowed.is_some());
            assert!(
                usage.funding.get(&other_id).is_none(),
                "independent healthy count canceled"
            );
            drop(usage);
            drop(pool.pin_registered_storage([(1u32, 40), (2, 0)]).unwrap());
            blocked(&pool);
        }
    }
}
