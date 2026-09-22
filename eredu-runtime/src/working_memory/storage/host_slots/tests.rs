use super::*;
use crate::HostMetadataKey;
use crate::working_memory::memory_fixture::host_ledger;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
struct Key(HostMetadataKey);
impl HostSlotStorageKey for Key {
    fn from_host_slot_identity(identity: HostMetadataKey) -> Option<Self> {
        Some(Self(identity))
    }
    fn host_slot_identity(&self) -> Option<&HostMetadataKey> {
        Some(&self.0)
    }
}

#[test]
fn closed_table_publishes_full_backing_without_duplicate_charge_and_keeps_custody() {
    for len in [0, 3] {
        let pool = host_ledger(u64::MAX, 0).unwrap();
        let foreign = host_ledger(u64::MAX, 0).unwrap();
        let initial = pool.snapshot().unwrap();
        let construction = pool.prepare_construction_metadata().unwrap();
        let source = construction.storage_funding().unwrap();
        assert!(source.funding().same_account(construction.funding()));
        let mut builder = source.prepare_host_slots::<u64, Key>(len).unwrap();
        for value in 0..len {
            builder.push(value as u64 + 1).unwrap();
        }
        let reserved = pool.snapshot().unwrap();
        let table = builder.finish().unwrap();
        let published = pool.snapshot().unwrap();
        assert_eq!(
            published.domains[0].current_charge_bytes,
            reserved.domains[0].current_charge_bytes
        );
        assert_eq!(
            published.domains[0].historical_peak_bytes,
            reserved.domains[0].historical_peak_bytes
        );
        assert_eq!(
            published.domains[0].registered_storage_bytes,
            initial.domains[0].registered_storage_bytes + (len * 8) as u64
        );
        let key = Key(table.metadata().identity().registry_key().clone());
        let allocation = pool.registered_allocation(&key).unwrap().unwrap();
        assert_eq!(allocation.capacity_bytes(), (len * 8) as u64);
        assert_eq!(
            allocation.placement(),
            pool.host_placement_handle().as_ref()
        );
        table
            .metadata()
            .validate_original_attachment(pool.shared_storage_accounting_id())
            .unwrap();
        assert!(
            table
                .metadata()
                .validate_original_attachment(foreign.shared_storage_accounting_id())
                .is_err()
        );
        let pin = StoragePublicationLayout::<Key>::new(1)
            .unwrap()
            .prepare(&pool, &source)
            .unwrap()
            .pin_registered_storage([(key.clone(), (len * 8) as u64)])
            .unwrap();
        let alias = table.metadata().clone();
        drop((construction, source));
        assert_eq!(table.slots().len(), len);
        drop(table);
        assert!(pool.registered_allocation(&key).unwrap().is_some());
        drop((pin, alias));
        assert!(pool.registered_allocation(&key).unwrap().is_none());
        assert_eq!(
            pool.snapshot().unwrap().domains[0].current_charge_bytes,
            initial.domains[0].current_charge_bytes
        );
    }
}

#[test]
fn closed_table_exact_allowance_and_one_short_refuse_before_payload_allocation() {
    let required = {
        let pool = host_ledger(u64::MAX, 0).unwrap();
        let initial = pool.snapshot().unwrap().domains[0].current_charge_bytes;
        let construction = pool.prepare_construction_metadata().unwrap();
        let source = construction.storage_funding().unwrap();
        let builder = source.prepare_host_slots::<u64, Key>(3).unwrap();
        let required = pool.snapshot().unwrap().domains[0].current_charge_bytes - initial;
        drop(builder);
        required
    };
    for short in [false, true] {
        let pool = host_ledger(required - u64::from(short), 0).unwrap();
        let initial = pool.snapshot().unwrap().domains[0].current_charge_bytes;
        let construction = pool.prepare_construction_metadata().unwrap();
        let source = construction.storage_funding().unwrap();
        let builder = source.prepare_host_slots::<u64, Key>(3);
        if short {
            assert!(
                matches!(
                    &builder,
                    Err(WorkingMemoryError::Domain(
                        eredu_core::MemoryDomainError::BudgetExceeded { .. }
                    ))
                ),
                "one-byte-short table preparation: {builder:?}"
            );
        } else {
            let mut builder = builder.unwrap();
            for value in 1..=3 {
                builder.push(value).unwrap();
            }
            let table = builder.finish().unwrap();
            assert_eq!(table.slots(), &[1, 2, 3]);
            assert_eq!(
                pool.snapshot().unwrap().domains[0].current_charge_bytes,
                initial + required
            );
        }
        drop((source, construction));
        assert_eq!(
            pool.snapshot().unwrap().domains[0].current_charge_bytes,
            initial
        );
    }
}

#[test]
fn closed_table_partial_values_retire_before_reservation_outside_locks() {
    #[derive(Debug)]
    struct Probe {
        pool: MemoryLedger,
        dropped: Arc<AtomicBool>,
    }
    impl Drop for Probe {
        fn drop(&mut self) {
            assert!(
                self.pool.snapshot().unwrap().domains[0].outstanding_reservation_bytes
                    >= (2 * size_of::<Self>()) as u64
            );
            self.dropped.store(true, Ordering::Release);
        }
    }
    let pool = host_ledger(u64::MAX, 0).unwrap();
    let initial = pool.snapshot().unwrap().domains[0].current_charge_bytes;
    let construction = pool.prepare_construction_metadata().unwrap();
    let source = construction.storage_funding().unwrap();
    let before = pool.snapshot().unwrap();
    assert!(matches!(
        source.prepare_host_slots::<u64, Key>(usize::MAX),
        Err(WorkingMemoryError::Overflow)
    ));
    assert_eq!(pool.snapshot().unwrap(), before);
    let dropped = Arc::new(AtomicBool::new(false));
    let mut builder = source.prepare_host_slots::<Probe, Key>(2).unwrap();
    builder
        .push(Probe {
            pool: pool.clone(),
            dropped: dropped.clone(),
        })
        .unwrap();
    drop((construction, source));
    assert!(matches!(
        builder.finish(),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    assert!(dropped.load(Ordering::Acquire));
    assert_eq!(
        pool.snapshot().unwrap().domains[0].current_charge_bytes,
        initial
    );
}
