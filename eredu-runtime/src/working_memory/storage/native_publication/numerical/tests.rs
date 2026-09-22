use super::*;
use crate::working_memory::{
    HostSourceConstructionFacts, InferenceExecutionIdentity, NumericalSourceRequirements,
    OriginalNumericalSource,
    memory_fixture::{self, separate},
};

fn source(pool: &MemoryLedger, placement: &MemoryPlacement, bytes: u64) -> OriginalNumericalSource {
    let mut native = eredu_core::DomainMemoryRequirements::zero(pool.topology());
    native.add_allocation(bytes, placement).unwrap();
    pool.reserve_numerical_source(
        &InferenceExecutionIdentity::default(),
        NumericalSourceRequirements::new(
            native,
            Some(128),
            Some(0),
            HostSourceConstructionFacts::new(0, 0, 0).unwrap(),
        )
        .unwrap(),
        pool.configured_limits().clone(),
    )
    .unwrap()
}
fn prepare(
    account: &OriginalNumericalBudgetCustody,
    pool: &MemoryLedger,
    funding: &HostMetadataFunding,
    slots: usize,
) -> PreparedNumericalStoragePublication<u64> {
    NumericalStoragePublicationPlan::new(slots, 0)
        .unwrap()
        .prepare(account, pool, funding)
        .unwrap()
}
fn charges(pool: &MemoryLedger) -> Vec<u64> {
    pool.snapshot()
        .unwrap()
        .domains
        .iter()
        .map(|domain| domain.current_charge_bytes)
        .collect()
}

#[test]
fn canonical_numerical_rows_share_payload_and_metadata_without_duplicate_charge() {
    for unified in [false, true] {
        for possible in [false, true] {
            let pool = if unified {
                memory_fixture::host_ledger(1 << 24, 0).unwrap()
            } else {
                separate::device_ledger(64, 0).unwrap()
            };
            let baseline = charges(&pool);
            let target = if unified {
                pool.topology().host_domain()
            } else {
                separate::device_domain(&pool)
            };
            let placement = Arc::new(if possible {
                MemoryPlacement::possible(
                    pool.topology(),
                    vec![pool.topology().host_domain(), target],
                    "actual fixture allocator candidates".into(),
                )
                .unwrap()
            } else {
                MemoryPlacement::fixed(pool.topology(), target).unwrap()
            });
            let source = source(&pool, &placement, 64);
            let account = source.budget_custody();
            let metadata = pool.prepare_construction_metadata().unwrap();
            let mut publication = prepare(&account, &pool, metadata.funding(), 3);
            publication.push(10, 64, placement.clone()).unwrap();
            publication.push_host_controls(11, 128).unwrap();
            publication.push(10, 64, placement.clone()).unwrap();
            let before = pool.snapshot().unwrap();
            publication.publish().unwrap();
            assert_eq!(pool.snapshot().unwrap(), before);
            let payload = pool.registered_allocation(&10u64).unwrap().unwrap();
            assert_eq!(payload.capacity_bytes(), 64);
            assert_eq!(payload.placement(), placement.as_ref());
            assert_eq!(
                pool.registered_allocation(&11u64)
                    .unwrap()
                    .unwrap()
                    .capacity_bytes(),
                128
            );
            let first = publication.take_input(0).unwrap();
            let controls = publication.take_input(1).unwrap();
            assert!(publication.take_input(2).is_none());
            first.arm_attachment().unwrap();
            controls.arm_attachment().unwrap();
            let backing = Arc::new((first, controls));
            let final_view = backing.clone();
            drop((publication, metadata, source, account, backing));
            assert!(pool.registered_allocation(&10u64).unwrap().is_some());
            assert_ne!(charges(&pool), baseline);
            drop(final_view);
            assert!(pool.registered_allocation(&10u64).unwrap().is_none());
            assert!(pool.registered_allocation(&11u64).unwrap().is_none());
            assert_eq!(charges(&pool), baseline);
        }
    }
}

#[test]
fn canonical_numerical_final_row_refusals_are_atomic_and_source_specific() {
    let pool = separate::device_ledger(128, 0).unwrap();
    let placement = separate::device_placement(&pool);
    let source = source(&pool, &placement, 64);
    let account = source.budget_custody();
    let metadata = pool.prepare_construction_metadata().unwrap();
    for kind in 0..4 {
        let mut publication = prepare(&account, &pool, metadata.funding(), 2);
        publication.push(1, 32, placement.clone()).unwrap();
        match kind {
            0 => publication.push(2, 33, placement.clone()).unwrap(),
            1 => publication.push_host_controls(2, 129).unwrap(),
            2 => publication
                .push(2, 32, pool.host_placement_handle())
                .unwrap(),
            _ => publication
                .push(1, 32, pool.host_placement_handle())
                .unwrap(),
        };
        let before = pool.snapshot().unwrap();
        assert!(matches!(
            publication.publish(),
            Err(WorkingMemoryError::IdentityMismatch)
        ));
        assert_eq!(pool.snapshot().unwrap(), before);
        assert!(pool.registered_allocation(&1u64).unwrap().is_none());
        assert!(pool.registered_allocation(&2u64).unwrap().is_none());
    }
    let exhausted = HostMetadataFunding::from_prepaid(
        HostMetadataFunding::prepaid_control_bytes().unwrap(),
        HostPreparationAuthority::retain(metadata.funding().clone()),
    )
    .unwrap();
    let before = pool.snapshot().unwrap();
    assert!(matches!(
        NumericalStoragePublicationPlan::<u64>::new(1, 0)
            .unwrap()
            .prepare(&account, &pool, &exhausted),
        Err(WorkingMemoryError::MetadataConstruction(_))
    ));
    assert_eq!(pool.snapshot().unwrap(), before);
    let foreign = separate::device_ledger(128, 0).unwrap();
    let before = pool.snapshot().unwrap();
    assert!(matches!(
        NumericalStoragePublicationPlan::<u64>::new(1, 0)
            .unwrap()
            .prepare(&account, &foreign, metadata.funding()),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    assert_eq!(pool.snapshot().unwrap(), before);
    account.quarantine();
    assert!(
        NumericalStoragePublicationPlan::<u64>::new(1, 0)
            .unwrap()
            .prepare(&account, &pool, metadata.funding())
            .is_err()
    );
}

#[test]
fn canonical_numerical_batches_preserve_identity_and_failed_attachment_custody() {
    let pool = separate::device_ledger(128, 0).unwrap();
    let baseline = charges(&pool);
    let placement = separate::device_placement(&pool);
    let source = source(&pool, &placement, 64);
    let account = source.budget_custody();
    let metadata = pool.prepare_construction_metadata().unwrap();
    let mut first = prepare(&account, &pool, metadata.funding(), 1);
    first.push(1, 32, placement.clone()).unwrap();
    first.publish().unwrap();
    let first_receipt = first.take_input(0).unwrap();
    drop(first);
    let first = first_receipt;
    first.arm_attachment().unwrap();
    let foreign_source = self::source(&pool, &placement, 64);
    let mut conflict = prepare(
        &foreign_source.budget_custody(),
        &pool,
        metadata.funding(),
        1,
    );
    conflict.push(1, 32, placement.clone()).unwrap();
    let before = pool.snapshot().unwrap();
    assert!(matches!(
        conflict.publish(),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    assert_eq!(pool.snapshot().unwrap(), before);
    drop((conflict, foreign_source));
    let mut over = prepare(&account, &pool, metadata.funding(), 1);
    over.push(2, 33, placement.clone()).unwrap();
    let before = pool.snapshot().unwrap();
    assert!(over.publish().is_err());
    assert_eq!(pool.snapshot().unwrap(), before);
    assert!(pool.registered_allocation(&2u64).unwrap().is_none());
    let mut duplicate = prepare(&account, &pool, metadata.funding(), 1);
    duplicate.push(1, 32, placement.clone()).unwrap();
    duplicate.publish().unwrap();
    let duplicate_receipt = duplicate.take_input(0).unwrap();
    drop(duplicate);
    let duplicate = duplicate_receipt;
    assert!(duplicate.arm_attachment().is_err());
    drop(duplicate);
    assert!(pool.registered_allocation(&1u64).unwrap().is_some());
    let mut unused = prepare(&account, &pool, metadata.funding(), 1);
    unused.push(2, 32, placement).unwrap();
    unused.publish().unwrap();
    let unused_receipt = unused.take_input(0).unwrap();
    drop(unused);
    let unused = unused_receipt;
    unused.arm_attachment().unwrap();
    unused.disarm_attachment();
    let before = charges(&pool);
    drop(unused);
    assert!(pool.registered_allocation(&2u64).unwrap().is_none());
    assert_eq!(charges(&pool), before);
    assert!(pool.registered_allocation(&1u64).unwrap().is_some());
    for (key, take_receipt) in [(3u64, false), (4u64, true)] {
        let mut abandoned = prepare(&account, &pool, metadata.funding(), 1);
        abandoned
            .push(key, 32, separate::device_placement(&pool))
            .unwrap();
        abandoned.publish().unwrap();
        let pin = pool.pin_registered_storage([(key, 32)]).unwrap();
        let before = charges(&pool);
        if take_receipt {
            let receipt = abandoned.take_input(0).unwrap();
            receipt.arm_attachment().unwrap();
            receipt.disarm_attachment();
            drop(receipt);
        }
        drop(abandoned);
        assert!(pool.registered_allocation(&key).unwrap().is_none());
        assert!(pool.pin_registered_storage([(key, 32)]).is_err());
        assert!(
            pin.validate_copy_source(&pool, &pool.0.usage.lock().unwrap())
                .is_err()
        );
        assert_eq!(charges(&pool), before);
        // The independently attached first row remains live throughout either
        // failure; only the fresh abandoned rows lose publication authority.
        assert!(pool.registered_allocation(&1u64).unwrap().is_some());
        drop(pin);
    }
    drop((first, over, source, account, metadata));
    assert_eq!(charges(&pool), baseline);
}

#[test]
fn canonical_numerical_observer_refund_and_directory_retirement_are_independent() {
    for observer_first in [false, true] {
        let pool = separate::device_ledger(64, 0).unwrap();
        let baseline = charges(&pool);
        let placement = separate::device_placement(&pool);
        let mut source = source(&pool, &placement, 64);
        let account = source.budget_custody();
        let observer = source
            .claim_native()
            .unwrap()
            .bind_lifetime(64, placement.clone())
            .unwrap();
        let metadata = pool.prepare_construction_metadata().unwrap();
        let mut publication = prepare(&account, &pool, metadata.funding(), 1);
        publication.push(1, 64, placement).unwrap();
        publication.publish().unwrap();
        let receipt = publication.take_input(0).unwrap();
        receipt.arm_attachment().unwrap();
        let pin = pool.pin_registered_storage([(1u64, 64)]).unwrap();
        let device = separate::device_domain(&pool);
        let device_charge = || {
            pool.snapshot()
                .unwrap()
                .domains
                .iter()
                .find(|row| row.domain == device)
                .unwrap()
                .current_charge_bytes
        };
        let charged = device_charge();
        if observer_first {
            observer.retire_completed_occupancy(0);
        }
        drop(receipt);
        assert!(pool.registered_allocation(&1u64).unwrap().is_none());
        assert!(pool.pin_registered_storage([(1u64, 64)]).is_err());
        assert!(
            pin.validate_copy_source(&pool, &pool.0.usage.lock().unwrap())
                .is_err()
        );
        assert_eq!(
            device_charge(),
            if observer_first {
                charged - 64
            } else {
                charged
            }
        );
        if !observer_first {
            observer.retire_completed_occupancy(0);
        }
        assert_eq!(device_charge(), charged - 64);
        drop((pin, publication, source, observer, account, metadata));
        assert_eq!(charges(&pool), baseline);
    }
}
