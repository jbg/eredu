use super::*;

#[test]
fn shared_storage_charges_overlapping_inventories_until_the_last_owner_retires() {
    let pool = WorkingMemoryPool::new(256, 20).unwrap();
    let empty = pool.register_storage(Vec::<(u32, u64)>::new()).unwrap();
    let first = pool
        .register_storage([(1u32, 40), (2, 60), (1, 40)])
        .unwrap();
    let second = pool.register_storage([(2u32, 60), (3, 80)]).unwrap();
    assert_eq!(first.bytes(), 100);
    assert_eq!(second.bytes(), 140);
    assert_eq!(pool.used_bytes().unwrap(), 200);
    let clone = first.clone();
    drop(first);
    assert_eq!(pool.used_bytes().unwrap(), 200);
    drop(second);
    assert_eq!(pool.used_bytes().unwrap(), 120);
    drop(clone);
    assert_eq!(pool.used_bytes().unwrap(), 20);
    drop(empty);
    assert_eq!(pool.peak_bytes().unwrap(), 200);
    // Retirement removes the old identity: a later physical owner may reuse it.
    let replacement = pool.register_storage([(1u32, 70)]).unwrap();
    assert_eq!(pool.used_bytes().unwrap(), 90);
    drop(replacement);
    assert_eq!(pool.used_bytes().unwrap(), 20);
    assert_eq!(pool.peak_bytes().unwrap(), 200);
}

#[test]
fn rejected_storage_is_atomic_for_conflicts_capacity_and_overflow() {
    let pool = WorkingMemoryPool::new(100, 10).unwrap();
    let retained = pool.register_storage([(2u32, 40)]).unwrap();
    for inventory in [vec![(1u32, 20), (2, 41)], vec![(1u32, 20), (1, 21)]] {
        assert!(matches!(
            pool.register_storage(inventory),
            Err(WorkingMemoryError::StorageCapacityMismatch { .. })
        ));
        assert_eq!(pool.used_bytes().unwrap(), 50);
        assert_eq!(pool.peak_bytes().unwrap(), 50);
    }
    assert!(matches!(
        pool.register_storage([(1u32, 20), (3, 31)]),
        Err(WorkingMemoryError::BudgetExceeded {
            required_bytes: 51,
            available_bytes: 50
        })
    ));
    assert!(matches!(
        pool.register_storage([(1u32, u64::MAX), (3, 1)]),
        Err(WorkingMemoryError::Overflow)
    ));
    assert_eq!(pool.used_bytes().unwrap(), 50);
    assert_eq!(pool.peak_bytes().unwrap(), 50);
    let exact = pool.register_storage([(1u32, 50)]).unwrap();
    assert_eq!(pool.used_bytes().unwrap(), 100);
    drop((retained, exact));
    assert_eq!(pool.used_bytes().unwrap(), 10);
    assert_eq!(pool.peak_bytes().unwrap(), 100);
}

#[test]
fn concurrent_inventories_deduplicate_shared_backing_and_reserve_unique_capacity_atomically() {
    let pool = WorkingMemoryPool::new(140, 0).unwrap();
    let barrier = Arc::new(std::sync::Barrier::new(8));
    let workers = (1u32..=8)
        .map(|id| {
            let pool = pool.clone();
            let barrier = barrier.clone();
            std::thread::spawn(move || {
                barrier.wait();
                pool.register_storage([(0u32, 100), (id, 10)])
            })
        })
        .collect::<Vec<_>>();
    let mut owners = Vec::new();
    for worker in workers {
        match worker.join().unwrap() {
            Ok(owner) => owners.push(owner),
            Err(error) => assert_eq!(
                error,
                WorkingMemoryError::BudgetExceeded {
                    required_bytes: 10,
                    available_bytes: 0
                }
            ),
        }
    }
    assert_eq!(owners.len(), 4);
    assert_eq!(pool.used_bytes().unwrap(), 140);
    owners.pop();
    assert_eq!(pool.used_bytes().unwrap(), 130);
    let next = pool.register_storage([(0u32, 100), (9, 10)]).unwrap();
    assert_eq!(pool.used_bytes().unwrap(), 140);
    drop(owners);
    assert_eq!(pool.used_bytes().unwrap(), 110);
    drop(next);
    assert_eq!(pool.used_bytes().unwrap(), 0);
    assert_eq!(pool.peak_bytes().unwrap(), 140);
}

#[test]
fn storage_and_inference_admission_compete_for_the_same_capacity() {
    let g = geometry(3, OutputDemand::LastPosition);
    let admitted = admission(g);
    let bytes = admitted.incremental_required_bytes;
    for _ in 0..16 {
        let pool = WorkingMemoryPool::new(bytes, 0).unwrap();
        let barrier = Arc::new(std::sync::Barrier::new(2));
        let workers = [false, true].map(|storage| {
            let pool = pool.clone();
            let barrier = barrier.clone();
            let admitted = admitted.clone();
            std::thread::spawn(
                move || -> Result<Box<dyn std::any::Any + Send>, WorkingMemoryError> {
                    barrier.wait();
                    if storage {
                        Ok(Box::new(pool.register_storage([(1u32, bytes)])?))
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
                WorkingMemoryError::BudgetExceeded {
                    available_bytes: 0,
                    ..
                }
            )));
        assert_eq!(pool.used_bytes().unwrap(), bytes);
        drop(owners);
        assert_eq!(pool.used_bytes().unwrap(), 0);
        assert_eq!(pool.peak_bytes().unwrap(), bytes);
    }
}

#[test]
fn individual_storage_retires_each_allocation_across_grouped_and_individual_aliases() {
    let pool = WorkingMemoryPool::new(512, 10).unwrap();
    let grouped = pool.register_storage([(2u32, 40), (3, 80)]).unwrap();
    let mut first = pool
        .register_storage_individually([(1u32, 20), (2, 40), (1, 20)])
        .unwrap();
    let mut second = pool
        .register_storage_individually([(1u32, 20), (3, 80), (4, 100)])
        .unwrap();
    assert_eq!(first.len(), 2);
    assert_eq!(first[&1].bytes(), 20);
    assert_eq!(first[&2].bytes(), 40);
    assert_eq!(pool.used_bytes().unwrap(), 250);
    drop(first.remove(&2));
    assert_eq!(pool.used_bytes().unwrap(), 250);
    drop(grouped);
    assert_eq!(pool.used_bytes().unwrap(), 210);
    drop(first);
    assert_eq!(pool.used_bytes().unwrap(), 210);
    drop(second.remove(&3));
    assert_eq!(pool.used_bytes().unwrap(), 130);
    let last_alias = second[&4].clone();
    drop(second.remove(&4));
    assert_eq!(pool.used_bytes().unwrap(), 130);
    drop(second);
    assert_eq!(pool.used_bytes().unwrap(), 110);
    drop(last_alias);
    assert_eq!(pool.used_bytes().unwrap(), 10);
    assert_eq!(pool.peak_bytes().unwrap(), 250);
    // Retiring one key permits reuse independently of any earlier batch.
    let replacement = pool.register_storage_individually([(1u32, 70)]).unwrap();
    assert_eq!(pool.used_bytes().unwrap(), 80);
    drop(replacement);
    assert_eq!(pool.used_bytes().unwrap(), 10);
}

#[test]
fn individual_storage_rejects_the_whole_batch_without_retiring_or_installing_keys() {
    let pool = WorkingMemoryPool::new(100, 10).unwrap();
    let existing = pool.register_storage([(2u32, 40)]).unwrap();
    assert!(pool
        .register_storage_individually(Vec::<(u32, u64)>::new())
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
        (
            vec![(1u32, 20), (3, 31)],
            WorkingMemoryError::BudgetExceeded {
                required_bytes: 51,
                available_bytes: 50,
            },
        ),
        (vec![(1u32, u64::MAX), (3, 1)], WorkingMemoryError::Overflow),
    ] {
        assert_eq!(
            pool.register_storage_individually(inventory).unwrap_err(),
            expected
        );
        assert_eq!(pool.used_bytes().unwrap(), 50);
        assert_eq!(pool.peak_bytes().unwrap(), 50);
    }
    let mut exact = pool
        .register_storage_individually([(1u32, 25), (3, 25)])
        .unwrap();
    assert_eq!(pool.used_bytes().unwrap(), 100);
    drop(exact.remove(&1));
    assert_eq!(pool.used_bytes().unwrap(), 75);
    drop((existing, exact));
    assert_eq!(pool.used_bytes().unwrap(), 10);
    assert_eq!(pool.peak_bytes().unwrap(), 100);
}

#[test]
fn individual_storage_batch_obeys_the_live_request_capacity_ceiling() {
    let admitted = admission(geometry(3, OutputDemand::LastPosition));
    let request_bytes = admitted.incremental_required_bytes;
    let pool = WorkingMemoryPool::new(request_bytes + 100, 10).unwrap();
    let request = pool
        .reserve_with_capacity(
            &InferenceExecutionIdentity::default(),
            &admitted,
            request_bytes + 40,
        )
        .unwrap();
    assert!(matches!(
        pool.register_storage_individually([(1u32, 10), (2, 21)]),
        Err(WorkingMemoryError::BudgetExceeded {
            required_bytes: 31,
            available_bytes: 30,
        })
    ));
    assert_eq!(pool.used_bytes().unwrap(), request_bytes + 10);
    assert_eq!(pool.peak_bytes().unwrap(), request_bytes + 10);
    let mut exact = pool
        .register_storage_individually([(1u32, 10), (2, 20)])
        .unwrap();
    assert_eq!(pool.used_bytes().unwrap(), request_bytes + 40);
    drop(exact.remove(&1));
    let replacement = pool.register_storage([(3u32, 10)]).unwrap();
    assert_eq!(pool.used_bytes().unwrap(), request_bytes + 40);
    drop((request, exact, replacement));
    assert_eq!(pool.used_bytes().unwrap(), 10);
    assert_eq!(pool.peak_bytes().unwrap(), request_bytes + 40);
}
