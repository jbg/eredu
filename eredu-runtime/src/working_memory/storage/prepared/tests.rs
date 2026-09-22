use super::*;
use crate::working_memory::{InferenceExecutionIdentity, memory_fixture};

#[test]
fn bounded_constructor_is_funded_before_allocation_and_retained_by_published_owners() {
    let pool = memory_fixture::host_ledger(u64::MAX, 0).unwrap();
    let baseline = pool.snapshot().unwrap().domains[0].current_charge_bytes;
    let unquoted = pool.acquire_unquoted().unwrap();
    let funding = pool.prepare_storage_metadata().unwrap();
    let layout = StoragePublicationLayout::<u32>::new(2).unwrap();
    let quote = layout.requested_bytes();
    let prepared = layout.prepare(&pool, &funding).unwrap();
    assert!(pool.snapshot().unwrap().domains[0].registry_metadata_bytes >= quote);
    let mut owners = prepared
        .register_storage_individually([
            (1, StorageAllocation::new(16, pool.host_placement_handle())),
            (2, StorageAllocation::new(64, pool.host_placement_handle())),
        ])
        .unwrap();
    drop(funding);
    let destination = owners.remove(&2).unwrap();
    drop(owners);
    let live = pool.snapshot().unwrap();
    assert_eq!(
        live.domains[0].registered_storage_bytes - live.domains[0].registry_metadata_bytes,
        64
    );
    assert!(live.domains[0].registry_metadata_bytes >= quote);
    drop(destination);
    drop(unquoted);
    assert_eq!(
        pool.snapshot().unwrap().domains[0].current_charge_bytes,
        baseline
    );
}

#[test]
fn rejected_publication_preserves_every_counter_after_explicit_preparation() {
    let pool = memory_fixture::host_ledger(u64::MAX, 0).unwrap();
    let funding = pool.prepare_storage_metadata().unwrap();
    let prepared = StoragePublicationLayout::<u32>::new(1)
        .unwrap()
        .prepare(&pool, &funding)
        .unwrap();
    let before = pool.snapshot().unwrap();
    assert!(
        prepared
            .register_storage([(
                1,
                StorageAllocation::new(u64::MAX, pool.host_placement_handle())
            )])
            .is_err()
    );
    assert_eq!(pool.snapshot().unwrap(), before);
}

#[test]
fn funded_preparation_converts_the_assigned_host_amount_before_buffers_are_built() {
    let pool = memory_fixture::host_ledger(u64::MAX, 0).unwrap();
    let baseline = pool.snapshot().unwrap().domains[0].current_charge_bytes;
    let layout = StoragePublicationLayout::<u32>::new(1).unwrap();
    let required =
        layout.requested_bytes() + MemoryLedger::storage_metadata_control_bytes().unwrap() + 64;
    let reservation = pool
        .reserve(
            &InferenceExecutionIdentity::default(),
            &memory_fixture::host_admission(&pool, required),
        )
        .unwrap();
    let (report, run) = reservation.into_funding().unwrap();
    let scope = run.scope().unwrap();
    let before = pool.snapshot().unwrap().domains[0].current_charge_bytes;
    let prepared = layout.fund_from(&scope).unwrap();
    assert_eq!(
        pool.snapshot().unwrap().domains[0].current_charge_bytes,
        before
    );
    let owners = prepared
        .adopt_storage_individually(
            &scope,
            [(1, StorageAllocation::new(64, pool.host_placement_handle()))],
        )
        .unwrap();
    assert_eq!(
        pool.snapshot().unwrap().domains[0].current_charge_bytes,
        before
    );
    scope.certify().unwrap();
    drop((run, report));
    assert_eq!(
        pool.snapshot().unwrap().domains[0].registered_storage_bytes
            - pool.snapshot().unwrap().domains[0].registry_metadata_bytes,
        64
    );
    drop(owners);
    assert_eq!(
        pool.snapshot().unwrap().domains[0].current_charge_bytes,
        baseline
    );
}
