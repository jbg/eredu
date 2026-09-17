use super::*;

#[test]
fn live_request_capacity_cannot_be_weakened_by_later_requests_or_storage() {
    let execution = InferenceExecutionIdentity::default();
    let admitted = admission(geometry(3, OutputDemand::LastPosition));
    let bytes = admitted.incremental_required_bytes;
    let pool = WorkingMemoryPool::new(100 + 4 * bytes, 40).unwrap();
    let smaller = 40 + 2 * bytes + 10;
    let larger = 40 + 3 * bytes + 10;
    let first = pool
        .reserve_with_capacity(&execution, &admitted, smaller)
        .unwrap();
    let completion = first.clone();
    let second = pool
        .reserve_with_capacity(&execution, &admitted, larger)
        .unwrap();
    let retained = pool.register_storage([(1_u32, 10)]).unwrap();
    let alias = pool.register_storage([(1_u32, 10)]).unwrap();
    assert_eq!(pool.used_bytes().unwrap(), smaller);
    assert_eq!(pool.effective_capacity().unwrap(), smaller);
    for result in [
        pool.reserve(&execution, &admitted),
        pool.reserve_with_capacity(&execution, &admitted, u64::MAX),
    ] {
        assert!(matches!(
            result,
            Err(WorkingMemoryError::BudgetExceeded {
                available_bytes: 0,
                ..
            })
        ));
    }
    assert!(matches!(
        pool.register_storage([(2_u32, 1)]),
        Err(WorkingMemoryError::BudgetExceeded {
            available_bytes: 0,
            ..
        })
    ));
    drop(first);
    assert_eq!(pool.effective_capacity().unwrap(), smaller);
    drop(completion);
    assert_eq!(pool.effective_capacity().unwrap(), larger);
    let third = pool.reserve(&execution, &admitted).unwrap();
    assert_eq!(pool.effective_capacity().unwrap(), larger);
    drop((second, third, retained, alias));
    assert_eq!(pool.effective_capacity().unwrap(), 100 + 4 * bytes);
    assert_eq!(pool.used_bytes().unwrap(), 40);
    assert_eq!(pool.peak_bytes().unwrap(), smaller);
}

#[test]
fn equal_request_limits_count_distinct_reservations_and_share_cloned_owners() {
    let execution = InferenceExecutionIdentity::default();
    let admitted = admission(geometry(1, OutputDemand::LastPosition));
    let limit = 2 * admitted.incremental_required_bytes;
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let first = pool
        .reserve_with_capacity(&execution, &admitted, limit)
        .unwrap();
    let second = pool
        .reserve_with_capacity(&execution, &admitted, limit)
        .unwrap();
    let snapshot = second.clone();
    drop((first, second));
    assert_eq!(pool.effective_capacity().unwrap(), limit);
    assert_eq!(
        pool.used_bytes().unwrap(),
        admitted.incremental_required_bytes
    );
    drop(snapshot);
    assert_eq!(pool.effective_capacity().unwrap(), u64::MAX);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn rejected_capacity_does_not_install_a_limit_or_change_accounting() {
    let execution = InferenceExecutionIdentity::default();
    let admitted = admission(geometry(3, OutputDemand::LastPosition));
    let pool = WorkingMemoryPool::new(4096, 40).unwrap();
    let storage = pool.register_storage([(1_u32, 60)]).unwrap();
    assert!(matches!(
        pool.reserve_with_capacity(&execution, &admitted, 99),
        Err(WorkingMemoryError::CapacityBelowUsage {
            capacity_bytes: 99,
            used_bytes: 100
        })
    ));
    assert!(matches!(
        pool.reserve_with_capacity(&execution, &admitted, 100),
        Err(WorkingMemoryError::BudgetExceeded {
            available_bytes: 0,
            ..
        })
    ));
    let mut unknown = admitted.clone();
    unknown.state.execution_workspace.as_mut().unwrap().retained = WorkspaceBound::Unknown {
        reason: "missing capture lifetime".into(),
    };
    assert!(matches!(
        pool.reserve_with_capacity(&execution, &unknown, 512),
        Err(WorkingMemoryError::UnknownBound)
    ));
    assert_eq!(pool.effective_capacity().unwrap(), 4096);
    assert_eq!(pool.used_bytes().unwrap(), 100);
    assert_eq!(pool.peak_bytes().unwrap(), 100);
    drop(storage);
}

#[test]
fn concurrent_capacity_claims_preserve_the_winning_live_ceiling() {
    let admitted = admission(geometry(1, OutputDemand::LastPosition));
    let bytes = admitted.incremental_required_bytes;
    for _ in 0..16 {
        let pool = WorkingMemoryPool::new(4 * bytes, 0).unwrap();
        let barrier = Arc::new(std::sync::Barrier::new(2));
        let workers = [bytes, 2 * bytes].map(|limit| {
            let pool = pool.clone();
            let barrier = barrier.clone();
            let admitted = admitted.clone();
            std::thread::spawn(move || {
                barrier.wait();
                (
                    limit,
                    pool.reserve_with_capacity(
                        &InferenceExecutionIdentity::default(),
                        &admitted,
                        limit,
                    ),
                )
            })
        });
        let results = workers.map(|worker| worker.join().unwrap());
        let (limit, _) = results.iter().find(|(_, result)| result.is_ok()).unwrap();
        assert_eq!(
            results.iter().filter(|(_, result)| result.is_ok()).count(),
            1
        );
        assert_eq!(pool.effective_capacity().unwrap(), *limit);
        assert_eq!(pool.used_bytes().unwrap(), bytes);
        assert_eq!(pool.peak_bytes().unwrap(), bytes);
        drop(results);
        assert_eq!(pool.effective_capacity().unwrap(), 4 * bytes);
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
}

#[test]
fn planner_shrinks_chunks_against_domain_capacity_including_existing_work() {
    let execution = InferenceExecutionIdentity::default();
    let pool = WorkingMemoryPool::new(1 << 20, 10).unwrap();
    let retained = pool.register_storage([(1_u32, 20)]).unwrap();
    let other = pool
        .reserve(
            &execution,
            &admission(geometry(1, OutputDemand::LastPosition)),
        )
        .unwrap();
    let expected = admission(geometry(3, OutputDemand::LastPosition));
    let capacity = pool.used_bytes().unwrap() + expected.incremental_required_bytes;
    let mut seen = Vec::new();
    let initial = geometry(7, OutputDemand::LastPosition);
    let (actual, reservation) = plan_prefill_with_capacity(
        &execution,
        &pool,
        &capabilities(),
        request(initial),
        initial,
        capacity,
        |g| {
            seen.push(g.prefill_chunk_positions);
            quote(g)
        },
    )
    .unwrap();
    assert_eq!(seen, [7, 6, 5, 4, 3]);
    assert_eq!(actual, expected);
    assert_eq!(reservation.geometry().prefill_chunk_positions, 3);
    assert_eq!(pool.used_bytes().unwrap(), capacity);
    assert_eq!(pool.effective_capacity().unwrap(), capacity);
    drop(reservation);
    assert_eq!(pool.effective_capacity().unwrap(), 1 << 20);
    drop((other, retained));
    assert_eq!(pool.used_bytes().unwrap(), 10);
}
