use super::*;

#[test]
fn live_request_capacity_cannot_be_weakened_by_later_requests_or_storage() {
    let execution = InferenceExecutionIdentity::default();
    let admitted = admission(geometry(3, OutputDemand::LastPosition));
    let bytes = memory::reservation_bytes(&admitted);
    let (pool, _funding, mut prepared, controls) =
        storage::prepared_ledger(100 + 4 * bytes, 40, &[1, 1, 1]);
    let smaller = controls + 40 + 2 * bytes + 10;
    let larger = controls + 40 + 3 * bytes + 10;
    let first = pool
        .reserve_with_capacity(&execution, &admitted, memory::resolved_limits(smaller))
        .unwrap();
    let completion = first.clone();
    let second = pool
        .reserve_with_capacity(&execution, &admitted, memory::resolved_limits(larger))
        .unwrap();
    let retained = prepared
        .remove(0)
        .register_host_storage([(1_u32, 10)])
        .unwrap();
    let alias = prepared
        .remove(0)
        .register_host_storage([(1_u32, 10)])
        .unwrap();
    assert_eq!(pool.funded_used_bytes().unwrap(), smaller - controls);
    assert_eq!(pool.payload_effective_capacity().unwrap(), smaller);
    for result in [
        pool.reserve(&execution, &admitted),
        pool.reserve_with_capacity(&execution, &admitted, memory::resolved_limits(u64::MAX)),
    ] {
        assert!(matches!(
            result,
            Err(WorkingMemoryError::Domain(
                eredu_core::MemoryDomainError::BudgetExceeded { .. }
            ))
        ));
    }
    assert!(matches!(
        prepared.remove(0).register_host_storage([(2_u32, 1)]),
        Err(WorkingMemoryError::Domain(
            eredu_core::MemoryDomainError::BudgetExceeded { .. }
        ))
    ));
    drop(first);
    assert_eq!(pool.payload_effective_capacity().unwrap(), smaller);
    drop(completion);
    assert_eq!(pool.payload_effective_capacity().unwrap(), larger);
    let third = pool.reserve(&execution, &admitted).unwrap();
    assert_eq!(pool.payload_effective_capacity().unwrap(), larger);
    drop((second, third, retained, alias));
    assert_eq!(
        pool.payload_effective_capacity().unwrap(),
        controls + 100 + 4 * bytes
    );
    assert_eq!(pool.funded_used_bytes().unwrap(), 40);
    assert_eq!(pool.payload_peak_bytes().unwrap(), smaller);
}

#[test]
fn equal_request_limits_count_distinct_reservations_and_share_cloned_owners() {
    let execution = InferenceExecutionIdentity::default();
    let admitted = admission(geometry(1, OutputDemand::LastPosition));
    let limit = 2 * memory::reservation_bytes(&admitted);
    let pool = memory::host_ledger(u64::MAX, 0).unwrap();
    let first = pool
        .reserve_with_capacity(&execution, &admitted, memory::resolved_limits(limit))
        .unwrap();
    let second = pool
        .reserve_with_capacity(&execution, &admitted, memory::resolved_limits(limit))
        .unwrap();
    let snapshot = second.clone();
    drop((first, second));
    assert_eq!(pool.payload_effective_capacity().unwrap(), limit);
    assert_eq!(
        pool.funded_used_bytes().unwrap(),
        memory::reservation_bytes(&admitted)
    );
    drop(snapshot);
    assert_eq!(
        pool.snapshot().unwrap().domains[0].effective_limit,
        eredu_core::MemoryLimit::Finite(u64::MAX)
    );
    assert_eq!(pool.funded_used_bytes().unwrap(), 0);
}

#[test]
fn rejected_capacity_does_not_install_a_limit_or_change_accounting() {
    let execution = InferenceExecutionIdentity::default();
    let admitted = admission(geometry(3, OutputDemand::LastPosition));
    let pool = memory::host_ledger(65536, 40).unwrap();
    let storage = pool.register_host_storage([(1_u32, 60)]).unwrap();
    let before = pool.snapshot().unwrap();
    assert!(matches!(
        pool.reserve_with_capacity(&execution, &admitted, memory::resolved_limits(99)),
        Err(WorkingMemoryError::Domain(
            eredu_core::MemoryDomainError::BudgetExceeded { .. }
        ))
    ));
    assert!(matches!(
        pool.reserve_with_capacity(&execution, &admitted, memory::resolved_limits(100)),
        Err(WorkingMemoryError::Domain(
            eredu_core::MemoryDomainError::BudgetExceeded { .. }
        ))
    ));
    let mut unknown = admitted.clone();
    unknown.state.execution_workspace.as_mut().unwrap().retained = WorkspaceBound::Unknown {
        reason: "missing capture lifetime".into(),
    };
    assert!(matches!(
        pool.reserve_with_capacity(&execution, &unknown, memory::resolved_limits(512)),
        Err(WorkingMemoryError::UnknownBound)
    ));
    assert_eq!(pool.payload_effective_capacity().unwrap(), 65536);
    assert_eq!(pool.funded_used_bytes().unwrap(), 100);
    assert_eq!(pool.snapshot().unwrap(), before);
    drop(storage);
}

#[test]
fn concurrent_capacity_claims_preserve_the_winning_live_ceiling() {
    let admitted = admission(geometry(1, OutputDemand::LastPosition));
    let bytes = memory::reservation_bytes(&admitted);
    for _ in 0..16 {
        let pool = memory::host_ledger(4 * bytes, 0).unwrap();
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
                        memory::resolved_limits(limit),
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
        assert_eq!(pool.payload_effective_capacity().unwrap(), *limit);
        assert_eq!(pool.funded_used_bytes().unwrap(), bytes);
        assert_eq!(pool.payload_peak_bytes().unwrap(), bytes);
        drop(results);
        assert_eq!(pool.payload_effective_capacity().unwrap(), 4 * bytes);
        assert_eq!(pool.funded_used_bytes().unwrap(), 0);
    }
}

#[test]
fn planner_shrinks_chunks_against_domain_capacity_including_existing_work() {
    let execution = InferenceExecutionIdentity::default();
    let pool = memory::host_ledger(1 << 20, 10).unwrap();
    let retained = pool.register_host_storage([(1_u32, 20)]).unwrap();
    let other = pool
        .reserve(
            &execution,
            &admission(geometry(1, OutputDemand::LastPosition)),
        )
        .unwrap();
    let expected = admission(geometry(3, OutputDemand::LastPosition));
    let capacity = pool.charged_bytes().unwrap() + memory::reservation_bytes(&expected);
    let mut seen = Vec::new();
    let initial = geometry(7, OutputDemand::LastPosition);
    let (actual, reservation) = plan_prefill_with_capacity(
        &execution,
        &pool,
        &capabilities(),
        request(initial),
        initial,
        memory::resolved_limits(capacity),
        |g| {
            seen.push(g.prefill_chunk_positions);
            quote(g)
        },
    )
    .unwrap();
    assert_eq!(seen, [7, 6, 5, 4, 3]);
    assert_eq!(actual, expected);
    assert_eq!(reservation.geometry().prefill_chunk_positions, 3);
    assert_eq!(pool.charged_bytes().unwrap(), capacity);
    assert_eq!(pool.payload_effective_capacity().unwrap(), capacity);
    drop(reservation);
    assert_eq!(pool.payload_effective_capacity().unwrap(), 1 << 20);
    drop((other, retained));
    assert_eq!(pool.funded_used_bytes().unwrap(), 10);
}
