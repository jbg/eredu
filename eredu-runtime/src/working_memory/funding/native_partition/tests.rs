use super::*;
use crate::working_memory::{
    InferenceExecutionIdentity, StorageRegistrations, WorkingMemoryReservation, memory_fixture,
};
fn control_floor() -> u64 {
    let pool = memory_fixture::host_ledger(u64::MAX, 0).unwrap();
    let admission = memory_fixture::host_admission(&pool, 200);
    pool.reservation_requirements(&admission, None)
        .unwrap()
        .get(pool.topology().host_domain())
        .unwrap()
        .total()
        .unwrap()
        - 200
}
fn test_capacity(capacity: u64) -> u64 {
    capacity
        .checked_add(control_floor().checked_mul(2).unwrap())
        .unwrap()
}
fn test_ledger(capacity: u64, existing: u64) -> Result<MemoryLedger, WorkingMemoryError> {
    memory_fixture::host_ledger(test_capacity(capacity), existing)
}
fn reservation(pool: &MemoryLedger, bytes: u64, capacity: u64) -> WorkingMemoryReservation {
    pool.reserve_with_capacity(
        &InferenceExecutionIdentity::default(),
        &memory_fixture::host_admission(pool, bytes),
        memory_fixture::resolved_host_limits(pool, test_capacity(capacity)),
    )
    .unwrap()
}
fn publication_bytes() -> u64 {
    crate::working_memory::StoragePublicationLayout::<u32>::new(1)
        .unwrap()
        .requested_bytes()
        + MemoryLedger::storage_metadata_control_bytes().unwrap()
}

use std::sync::Barrier;

fn balances(pool: &MemoryLedger) -> (u64, u64, u64) {
    let usage = pool.0.usage.lock().unwrap();
    (
        usage.reserved
            - usage
                .funding
                .values()
                .map(|state| state.control_floor)
                .sum::<u64>(),
        usage.registered - usage.registry_metadata,
        usage.reserved
            - usage
                .funding
                .values()
                .map(|state| state.control_floor)
                .sum::<u64>()
            + usage.registered
            - usage.registry_metadata,
    )
}

#[test]
fn native_partition_protects_host_and_ordinary_spending_until_both_retire() {
    for pending_input in [false, true] {
        let pool = test_ledger(300, 0).unwrap();
        let (metadata, run) = reservation(&pool, 200 + publication_bytes(), 300)
            .into_funding()
            .unwrap();
        let partition = run.take_native_partition(test_receipt(&run, 100)).unwrap();
        let mut sampler = run.sampler_scope().unwrap();
        let pending = if pending_input {
            Some(run.open_pending_input_scope(&metadata, 40).unwrap())
        } else {
            sampler.hold_sampler_payload(40).unwrap();
            None
        };
        let scope = run.scope().unwrap();
        assert!(matches!(
            scope.adopt_host_storage_individually([(1u32, 61)]),
            Err(WorkingMemoryError::DomainAllowanceExceeded { .. })
        ));
        let storage = scope.adopt_host_storage_individually([(1u32, 60)]).unwrap();
        assert_eq!(balances(&pool).1, 60);
        let available = {
            let usage = pool.0.usage.lock().unwrap();
            let state = usage.funding.get(&run.id).unwrap();
            state.remaining - state.native_held.unwrap() - state.host_held
        };
        assert!(matches!(
            run.open_pending_input_scope(&metadata, available + 1),
            Err(WorkingMemoryError::DomainAllowanceExceeded { .. })
        ));
        scope.certify().unwrap();
        drop(run);
        drop(metadata);
        drop(storage);
        assert_eq!(balances(&pool), (140, 0, 140));
        drop(pending);
        drop(sampler);
        assert_eq!(balances(&pool), (100, 0, 100));
        drop(partition);
        assert_eq!(balances(&pool), (0, 0, 0));
        assert!(pool.0.usage.lock().unwrap().funding.is_empty());
    }
}

#[test]
fn concurrent_partition_host_hold_and_publication_share_one_balance() {
    let pool = test_ledger(200, 0).unwrap();
    let (metadata, run) = reservation(&pool, 200 + publication_bytes(), 200)
        .into_funding()
        .unwrap();
    let receipt = test_receipt(&run, 120);
    let mut sampler = run.sampler_scope().unwrap();
    let native = run.scope().unwrap();
    let barrier = Barrier::new(3);
    let (partition, host, storage) = std::thread::scope(|threads| {
        let a = threads.spawn(|| {
            barrier.wait();
            run.take_native_partition(receipt)
        });
        let b = threads.spawn(|| {
            barrier.wait();
            let result = sampler.hold_sampler_payload(70);
            (result, sampler)
        });
        let c = threads.spawn(|| {
            barrier.wait();
            let result = native.adopt_host_storage_individually([(7u32, 70)]);
            native.certify().unwrap();
            result
        });
        (a.join().unwrap(), b.join().unwrap(), c.join().unwrap())
    });
    assert_eq!(
        usize::from(partition.is_ok()) + usize::from(host.0.is_ok()) + usize::from(storage.is_ok()),
        2
    );
    for error in [
        partition.as_ref().err(),
        host.0.as_ref().err(),
        storage.as_ref().err(),
    ]
    .into_iter()
    .flatten()
    {
        assert!(matches!(
            error,
            WorkingMemoryError::DomainAllowanceExceeded { .. }
        ));
    }
    let snapshot = pool.snapshot().unwrap();
    let domain = &snapshot.domains[0];
    assert_eq!(
        domain.current_charge_bytes
            - domain.fixed_baseline.total().unwrap()
            - domain.reservation_control_bytes,
        200 + publication_bytes()
    );
    assert!(domain.historical_peak_bytes >= domain.current_charge_bytes);
    drop((partition, host, storage));
    drop(run);
    drop(metadata);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

#[test]
fn partition_receipt_is_account_bound_and_two_requests_retain_distinct_holds() {
    let pool = test_ledger(300, 0).unwrap();
    let (a, a_run) = reservation(&pool, 120, 300).into_funding().unwrap();
    let (b, b_run) = reservation(&pool, 120, 300).into_funding().unwrap();
    assert!(matches!(
        b_run.take_native_partition(test_receipt(&a_run, 100)),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    let a_partition = a_run
        .take_native_partition(test_receipt(&a_run, 100))
        .unwrap();
    let b_partition = b_run
        .take_native_partition(test_receipt(&b_run, 100))
        .unwrap();
    assert!(!a_partition.same_origin(&b_partition));
    assert!(matches!(
        pool.register_host_storage([(9u32, 61)]),
        Err(WorkingMemoryError::Domain(_))
    ));
    drop((a_run, b_run, a, b));
    assert_eq!(pool.payload_used_bytes().unwrap(), 200);
    drop(a_partition);
    assert_eq!(pool.payload_used_bytes().unwrap(), 100);
    drop(b_partition);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

#[test]
fn zero_partition_is_once_only_and_pins_account_after_run_closure() {
    for bytes in [0, 80] {
        let pool = test_ledger(200, 0).unwrap();
        let (metadata, mut run) = reservation(&pool, 100, 150).into_funding().unwrap();
        let first = run
            .take_native_partition(test_receipt(&run, bytes))
            .unwrap();
        let last = first.clone();
        drop(first);
        assert!(matches!(
            run.take_native_partition(test_receipt(&run, bytes)),
            Err(WorkingMemoryError::PreparationAlreadyStarted)
        ));
        let handoff = run.take_capacity_handoff().unwrap();
        drop(run);
        drop(metadata);
        assert_eq!(pool.payload_used_bytes().unwrap(), bytes);
        assert!(!handoff.is_retired().unwrap());
        assert_eq!(
            pool.payload_effective_capacity().unwrap(),
            test_capacity(150)
        );
        drop(last);
        assert!(handoff.is_retired().unwrap());
        assert_eq!(
            pool.payload_effective_capacity().unwrap(),
            test_capacity(200)
        );
    }
    // Even destruction before closure cannot reset the issuance latch.
    let pool = test_ledger(100, 0).unwrap();
    let (metadata, run) = reservation(&pool, 100, 100).into_funding().unwrap();
    drop(run.take_native_partition(test_receipt(&run, 0)).unwrap());
    assert!(matches!(
        run.take_native_partition(test_receipt(&run, 0)),
        Err(WorkingMemoryError::PreparationAlreadyStarted)
    ));
    drop((run, metadata));
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

#[test]
fn native_partition_retirement_never_clears_quarantine_or_poison() {
    for poison in [false, true] {
        let pool = test_ledger(200, 0).unwrap();
        let (metadata, run) = reservation(&pool, 160, 200).into_funding().unwrap();
        let partition = run.take_native_partition(test_receipt(&run, 100)).unwrap();
        let scope = run.scope().unwrap();
        if poison {
            assert!(
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let _usage = pool.0.usage.lock().unwrap();
                    panic!("poison with a protected native partition");
                }))
                .is_err()
            );
        }
        drop(scope);
        drop(partition);
        drop(run);
        drop(metadata);
        let usage = pool
            .0
            .usage
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let state = usage.funding.values().next().unwrap();
        assert!(state.quarantined);
        assert!(state.native_held.is_none());
        assert_eq!(usage.reserved - state.control_floor, 160);
        assert_eq!(usage.registered, 0);
    }
}

#[test]
fn completed_host_transfer_preserves_the_native_protected_remainder() {
    use crate::working_memory::{HostSlotStorageKey, storage::publish_dense_host_slots};
    use crate::{HostMetadataKey, HostSlotTable};
    #[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
    struct Key(HostMetadataKey);
    impl HostSlotStorageKey for Key {
        fn host_slot_identity(&self) -> Option<&HostMetadataKey> {
            Some(&self.0)
        }
    }
    let pool = test_ledger(10_000, 0).unwrap();
    let original = HostSlotTable::new(vec![11u32, 18, 25].into_boxed_slice());
    let source_bytes = original.metadata().capacity_bytes().unwrap();
    let source_key = Key(original.metadata().identity().registry_key().clone());
    let registered = pool
        .register_host_storage([(source_key, source_bytes)])
        .unwrap();
    let plan = original
        .prepare_copy_slots()
        .unwrap()
        .for_dense_destination::<u32>()
        .unwrap();
    let retained = plan.retained_bytes();
    let protected = plan.initialization_peak_bytes();
    let controls =
        crate::working_memory::storage::ordinary_dense_preparation_bytes::<u32, u32, Key>()
            .unwrap();
    let (metadata, run) = reservation(&pool, protected + 180 + controls, 10_000 + controls)
        .into_funding()
        .unwrap();
    let partition = run.take_native_partition(test_receipt(&run, 100)).unwrap();
    let (execution, mut host, native) = run
        .open_dense_prompt_scopes(
            &metadata,
            DecoderCopySource::Registered(&registered),
            &registered,
            RegisteredStoragePin::new(registered.clone()),
            protected,
        )
        .unwrap();
    native.certify().unwrap();
    let preparation = crate::working_memory::storage::prepare_dense_host_metadata::<u32, u32, Key>(
        &host, &execution,
    )
    .unwrap();
    let identity = crate::HostMetadataIdentity::prepared_host(&preparation).unwrap();
    let mut builder = plan.initialize();
    for value in original.slots() {
        builder.push(*value).unwrap();
    }
    let completed = builder
        .finish_with_preparation(Some((identity, &preparation)))
        .unwrap();
    let before = pool.payload_used_bytes().unwrap();
    let key = Key(completed.metadata().identity().registry_key().clone());
    publish_dense_host_slots(
        &completed,
        &mut host,
        &execution,
        retained,
        protected,
        key,
        Some(&preparation),
    )
    .unwrap();
    drop(preparation);
    assert_eq!(
        completed.iter().copied().collect::<Vec<_>>(),
        original.slots()
    );
    assert_eq!(pool.payload_used_bytes().unwrap(), before);
    assert!(matches!(
        run.open_pending_input_scope(&metadata, 81),
        Err(WorkingMemoryError::DomainAllowanceExceeded {
            required_bytes: 81,
            available_bytes: 80,
            ..
        })
    ));
    let remaining = run.open_pending_input_scope(&metadata, 80).unwrap();
    drop((
        completed, remaining, host, run, metadata, registered, original,
    ));
    assert_eq!(pool.payload_used_bytes().unwrap(), 100);
    drop(partition);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

#[test]
fn completed_native_occupancy_retires_staging_and_slack_with_destination_alias_alive() {
    let pool = test_ledger(256, 0).unwrap();
    let (metadata, run) = reservation(&pool, 200, 256).into_funding().unwrap();
    let partition = run.take_native_partition(test_receipt(&run, 144)).unwrap();
    let destination = partition.clone();
    drop((run, metadata));
    assert_eq!(pool.payload_used_bytes().unwrap(), 144);
    partition.retire_completed_occupancy(80);
    assert_eq!(pool.payload_used_bytes().unwrap(), 80);
    // The 16-byte staging backing has physically retired; the independent
    // 64-byte destination remains authenticated by its original capacity.
    partition.retire_completed_occupancy(64);
    assert_eq!(pool.payload_used_bytes().unwrap(), 64);
    assert_eq!(destination.capacity(), 144);
    partition.retire_completed_occupancy(80); // stale concurrent callback
    assert_eq!(pool.payload_used_bytes().unwrap(), 64);
    drop(partition);
    assert_eq!(pool.payload_used_bytes().unwrap(), 64);
    destination.retire_completed_occupancy(0);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
    drop(destination);
    assert!(pool.0.usage.lock().unwrap().funding.is_empty());
}
