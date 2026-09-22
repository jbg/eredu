use super::*;

pub(super) fn prepared_ledger(
    capacity: u64,
    existing: u64,
    populations: &[usize],
) -> (
    MemoryLedger,
    Vec<StorageMetadataFunding>,
    Vec<PreparedStoragePublication<u32>>,
    u64,
) {
    let controls = populations
        .iter()
        .try_fold(0u64, |total, &population| {
            total
                .checked_add(MemoryLedger::storage_metadata_control_bytes().unwrap())?
                .checked_add(
                    StoragePublicationLayout::<u32>::new(population)
                        .unwrap()
                        .requested_bytes(),
                )
        })
        .unwrap();
    let pool = memory::host_ledger(capacity.checked_add(controls).unwrap(), existing).unwrap();
    let mut funding = Vec::new();
    let mut prepared = Vec::new();
    for &population in populations {
        let grant = pool.prepare_storage_metadata().unwrap();
        prepared.push(
            StoragePublicationLayout::new(population)
                .unwrap()
                .prepare(&pool, &grant)
                .unwrap(),
        );
        funding.push(grant);
    }
    (pool, funding, prepared, controls)
}
fn allocations(
    entries: impl IntoIterator<Item = (u32, u64)>,
) -> impl Iterator<Item = (u32, StorageAllocation)> {
    entries
        .into_iter()
        .map(|(key, bytes)| (key, StorageAllocation::new(bytes, memory::placement())))
}

#[test]
fn shared_storage_charges_overlapping_inventories_until_the_last_owner_retires() {
    let (pool, _funding, mut prepared, controls) = prepared_ledger(256, 20, &[0, 3, 2, 1]);
    let empty = prepared
        .remove(0)
        .register_host_storage(Vec::<(u32, u64)>::new())
        .unwrap();
    let first = prepared
        .remove(0)
        .register_host_storage([(1u32, 40), (2, 60), (1, 40)])
        .unwrap();
    let second = prepared
        .remove(0)
        .register_host_storage([(2u32, 60), (3, 80)])
        .unwrap();
    assert_eq!(first.bytes(), Some(100));
    assert_eq!(second.bytes(), Some(140));
    assert_eq!(pool.payload_used_bytes().unwrap(), 200);
    let clone = first.clone();
    drop(first);
    assert_eq!(pool.payload_used_bytes().unwrap(), 200);
    drop(second);
    assert_eq!(pool.payload_used_bytes().unwrap(), 120);
    drop(clone);
    assert_eq!(pool.payload_used_bytes().unwrap(), 20);
    drop(empty);
    assert_eq!(pool.payload_peak_bytes().unwrap(), controls + 200);
    // Retirement removes the old identity: a later physical owner may reuse it.
    let replacement = prepared
        .remove(0)
        .register_host_storage([(1u32, 70)])
        .unwrap();
    assert_eq!(pool.payload_used_bytes().unwrap(), 90);
    drop(replacement);
    assert_eq!(pool.payload_used_bytes().unwrap(), 20);
    assert_eq!(pool.payload_peak_bytes().unwrap(), controls + 200);
}

#[test]
fn rejected_storage_is_atomic_for_conflicts_capacity_and_overflow() {
    let (pool, _funding, mut prepared, controls) = prepared_ledger(100, 10, &[1, 2, 2, 2, 2, 1]);
    let retained = prepared
        .remove(0)
        .register_host_storage([(2u32, 40)])
        .unwrap();
    for inventory in [vec![(1u32, 20), (2, 41)], vec![(1u32, 20), (1, 21)]] {
        assert!(matches!(
            prepared.remove(0).register_host_storage(inventory),
            Err(WorkingMemoryError::StorageCapacityMismatch { .. })
        ));
        assert_eq!(pool.payload_used_bytes().unwrap(), 50);
        assert_eq!(pool.payload_peak_bytes().unwrap(), controls + 50);
    }
    assert_eq!(
        prepared
            .remove(0)
            .register_host_storage([(1u32, 20), (3, 31)])
            .unwrap_err(),
        budget_error(&pool, 51, 50)
    );
    assert!(matches!(
        prepared
            .remove(0)
            .register_host_storage([(1u32, u64::MAX), (3, 1)]),
        Err(WorkingMemoryError::Domain(
            eredu_core::MemoryDomainError::Overflow
        ))
    ));
    assert_eq!(pool.payload_used_bytes().unwrap(), 50);
    assert_eq!(pool.payload_peak_bytes().unwrap(), controls + 50);
    let exact = prepared
        .remove(0)
        .register_host_storage([(1u32, 50)])
        .unwrap();
    assert_eq!(pool.payload_used_bytes().unwrap(), 100);
    drop((retained, exact));
    assert_eq!(pool.payload_used_bytes().unwrap(), 10);
    assert_eq!(pool.payload_peak_bytes().unwrap(), controls + 100);
}

#[test]
fn concurrent_inventories_deduplicate_shared_backing_and_reserve_unique_capacity_atomically() {
    let (pool, _funding, mut prepared, controls) = prepared_ledger(140, 0, &[2; 9]);
    let barrier = Arc::new(std::sync::Barrier::new(8));
    let workers = (1u32..=8)
        .zip(prepared.drain(..8))
        .map(|(id, publication)| {
            let pool = pool.clone();
            let barrier = barrier.clone();
            std::thread::spawn(move || {
                barrier.wait();
                publication.register_host_storage([(0u32, 100), (id, 10)])
            })
        })
        .collect::<Vec<_>>();
    let mut owners = Vec::new();
    for worker in workers {
        match worker.join().unwrap() {
            Ok(owner) => owners.push(owner),
            Err(error) => assert_eq!(error, budget_error(&pool, 10, 0)),
        }
    }
    assert_eq!(owners.len(), 4);
    assert_eq!(pool.payload_used_bytes().unwrap(), 140);
    owners.pop();
    assert_eq!(pool.payload_used_bytes().unwrap(), 130);
    let next = prepared
        .remove(0)
        .register_host_storage([(0u32, 100), (9, 10)])
        .unwrap();
    assert_eq!(pool.payload_used_bytes().unwrap(), 140);
    drop(owners);
    assert_eq!(pool.payload_used_bytes().unwrap(), 110);
    drop(next);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
    assert_eq!(pool.payload_peak_bytes().unwrap(), controls + 140);
}

#[test]
fn storage_and_inference_admission_compete_for_the_same_capacity() {
    let g = geometry(3, OutputDemand::LastPosition);
    let admitted = admission(g);
    let bytes = memory::reservation_bytes(&admitted);
    for _ in 0..16 {
        let (pool, _funding, mut prepared, controls) = prepared_ledger(bytes, 0, &[1]);
        let publication = prepared.pop().unwrap();
        let barrier = Arc::new(std::sync::Barrier::new(2));
        let workers = [None, Some(publication)].map(|storage| {
            let pool = pool.clone();
            let barrier = barrier.clone();
            let admitted = admitted.clone();
            std::thread::spawn(
                move || -> Result<Box<dyn std::any::Any + Send>, WorkingMemoryError> {
                    barrier.wait();
                    if let Some(publication) = storage {
                        Ok(Box::new(
                            publication.register_host_storage([(1u32, bytes)])?,
                        ))
                    } else {
                        Ok(Box::new(pool.reserve(
                            &InferenceExecutionIdentity::default(),
                            &admitted,
                        )?))
                    }
                },
            )
        });
        let owners = workers
            .into_iter()
            .map(|worker| worker.join().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(owners.iter().filter(|owner| owner.is_ok()).count(), 1);
        assert!(owners
            .iter()
            .filter_map(|owner| owner.as_ref().err())
            .all(|error| matches!(
                error,
                WorkingMemoryError::Domain(eredu_core::MemoryDomainError::BudgetExceeded { .. })
            )));
        assert_eq!(pool.funded_used_bytes().unwrap(), bytes);
        drop(owners);
        assert_eq!(pool.payload_used_bytes().unwrap(), 0);
        assert_eq!(pool.payload_peak_bytes().unwrap(), controls + bytes);
    }
}

#[test]
fn individual_storage_retires_each_allocation_across_grouped_and_individual_aliases() {
    let (pool, _funding, mut prepared, controls) = prepared_ledger(512, 10, &[2, 3, 3, 1]);
    let grouped = prepared
        .remove(0)
        .register_host_storage([(2u32, 40), (3, 80)])
        .unwrap();
    let mut first = prepared
        .remove(0)
        .register_storage_individually(allocations([(1u32, 20), (2, 40), (1, 20)]))
        .unwrap();
    let mut second = prepared
        .remove(0)
        .register_storage_individually(allocations([(1u32, 20), (3, 80), (4, 100)]))
        .unwrap();
    assert_eq!(first.len(), 2);
    assert_eq!(first[&1].bytes(), Some(20));
    assert_eq!(first[&2].bytes(), Some(40));
    assert_eq!(pool.payload_used_bytes().unwrap(), 250);
    drop(first.remove(&2));
    assert_eq!(pool.payload_used_bytes().unwrap(), 250);
    drop(grouped);
    assert_eq!(pool.payload_used_bytes().unwrap(), 210);
    drop(first);
    assert_eq!(pool.payload_used_bytes().unwrap(), 210);
    drop(second.remove(&3));
    assert_eq!(pool.payload_used_bytes().unwrap(), 130);
    let last_alias = second[&4].clone();
    drop(second.remove(&4));
    assert_eq!(pool.payload_used_bytes().unwrap(), 130);
    drop(second);
    assert_eq!(pool.payload_used_bytes().unwrap(), 110);
    drop(last_alias);
    assert_eq!(pool.payload_used_bytes().unwrap(), 10);
    assert_eq!(pool.payload_peak_bytes().unwrap(), controls + 250);
    // Retiring one key permits reuse independently of any earlier batch.
    let replacement = prepared
        .remove(0)
        .register_storage_individually(allocations([(1u32, 70)]))
        .unwrap();
    assert_eq!(pool.payload_used_bytes().unwrap(), 80);
    drop(replacement);
    assert_eq!(pool.payload_used_bytes().unwrap(), 10);
}

#[test]
fn individual_storage_rejects_the_whole_batch_without_retiring_or_installing_keys() {
    let (pool, _funding, mut prepared, controls) = prepared_ledger(100, 10, &[1, 0, 2, 2, 2, 2, 2]);
    let existing = prepared
        .remove(0)
        .register_host_storage([(2u32, 40)])
        .unwrap();
    assert!(prepared
        .remove(0)
        .register_storage_individually(allocations(Vec::<(u32, u64)>::new()))
        .unwrap()
        .is_empty());
    for (inventory, expected) in [
        (
            vec![(1u32, 20), (2, 41)],
            WorkingMemoryError::StorageCapacityMismatch {
                expected_bytes: 40,
                actual_bytes: 41,
            },
        ),
        (
            vec![(1u32, 20), (1, 21)],
            WorkingMemoryError::StorageCapacityMismatch {
                expected_bytes: 20,
                actual_bytes: 21,
            },
        ),
        (vec![(1u32, 20), (3, 31)], budget_error(&pool, 51, 50)),
        (
            vec![(1u32, u64::MAX), (3, 1)],
            WorkingMemoryError::Domain(eredu_core::MemoryDomainError::Overflow),
        ),
    ] {
        assert_eq!(
            prepared
                .remove(0)
                .register_storage_individually(allocations(inventory))
                .unwrap_err(),
            expected
        );
        assert_eq!(pool.payload_used_bytes().unwrap(), 50);
        assert_eq!(pool.payload_peak_bytes().unwrap(), controls + 50);
    }
    let mut exact = prepared
        .remove(0)
        .register_storage_individually(allocations([(1u32, 25), (3, 25)]))
        .unwrap();
    assert_eq!(pool.payload_used_bytes().unwrap(), 100);
    drop(exact.remove(&1));
    assert_eq!(pool.payload_used_bytes().unwrap(), 75);
    drop((existing, exact));
    assert_eq!(pool.payload_used_bytes().unwrap(), 10);
    assert_eq!(pool.payload_peak_bytes().unwrap(), controls + 100);
}

#[test]
fn individual_storage_batch_obeys_the_live_request_capacity_ceiling() {
    let admitted = admission(geometry(3, OutputDemand::LastPosition));
    let request_bytes = memory::reservation_bytes(&admitted);
    let (pool, _funding, mut prepared, controls) =
        prepared_ledger(request_bytes + 100, 10, &[2, 2, 1]);
    let request = pool
        .reserve_with_capacity(
            &InferenceExecutionIdentity::default(),
            &admitted,
            memory::resolved_limits(controls + request_bytes + 40),
        )
        .unwrap();
    assert_eq!(
        prepared
            .remove(0)
            .register_storage_individually(allocations([(1u32, 10), (2, 21)]))
            .unwrap_err(),
        budget_error(&pool, 31, 30)
    );
    assert_eq!(pool.funded_used_bytes().unwrap(), request_bytes + 10);
    assert_eq!(
        pool.payload_peak_bytes().unwrap(),
        controls + request_bytes + 10
    );
    let mut exact = prepared
        .remove(0)
        .register_storage_individually(allocations([(1u32, 10), (2, 20)]))
        .unwrap();
    assert_eq!(pool.funded_used_bytes().unwrap(), request_bytes + 40);
    drop(exact.remove(&1));
    let replacement = prepared
        .remove(0)
        .register_host_storage([(3u32, 10)])
        .unwrap();
    assert_eq!(pool.funded_used_bytes().unwrap(), request_bytes + 40);
    drop((request, exact, replacement));
    assert_eq!(pool.payload_used_bytes().unwrap(), 10);
    assert_eq!(
        pool.payload_peak_bytes().unwrap(),
        controls + request_bytes + 40
    );
}
