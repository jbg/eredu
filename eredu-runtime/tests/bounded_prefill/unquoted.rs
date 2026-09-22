use super::*;

fn zero_admission() -> Admission {
    let g = geometry(1, OutputDemand::StateOnly);
    let mut state = quote(g).unwrap();
    state.fixed_state_bytes = 0;
    state.bytes_per_position_per_batch = 0;
    state.context_state_bytes = 0;
    state.requested_state_bytes = 0;
    let zero = || WorkspaceBound::bounded(0, "stateless fixture without native payloads");
    state.execution_workspace = Some(ExecutionWorkspaceEstimate {
        physical_domains: None,
        geometry: g,
        activations: zero(),
        attention: zero(),
        vocabulary: zero(),
        state_update: zero(),
        materialization: zero(),
        retained: zero(),
    });
    match eredu_core::apply_admission_policy(&capabilities(), request(g), memory::state(state))
        .unwrap()
    {
        AdmissionResult::Admitted(admitted) => {
            assert_eq!(admitted.incremental_required_bytes.unwrap(), 0);
            admitted
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn independent_unquoted_owners_and_their_clones_block_both_reservation_methods() {
    let pool = memory::host_ledger(65536, 40).unwrap();
    let execution = InferenceExecutionIdentity::default();
    let admitted = admission(geometry(3, OutputDemand::LastPosition));
    let first = pool.acquire_unquoted().unwrap();
    let completion = first.clone();
    let second = pool.acquire_unquoted().unwrap();
    assert_eq!(pool.unquoted_owner_count().unwrap(), 2);
    for result in [
        pool.reserve(&execution, &admitted),
        pool.reserve_with_capacity(&execution, &admitted, memory::resolved_limits(512)),
    ] {
        assert!(matches!(result, Err(WorkingMemoryError::UnknownBound)));
    }
    assert_eq!(pool.funded_used_bytes().unwrap(), 40);
    assert_eq!(
        pool.payload_peak_bytes().unwrap(),
        40 + 2 * MemoryLedger::unquoted_owner_control_bytes().unwrap()
    );
    assert_eq!(pool.payload_effective_capacity().unwrap(), 65536);
    drop(first);
    assert_eq!(pool.unquoted_owner_count().unwrap(), 2);
    drop(second);
    assert_eq!(pool.unquoted_owner_count().unwrap(), 1);
    assert!(matches!(
        pool.reserve(&execution, &admitted),
        Err(WorkingMemoryError::UnknownBound)
    ));
    drop(completion);
    assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
    let reservation = pool.reserve(&execution, &admitted).unwrap();
    assert_eq!(
        pool.funded_used_bytes().unwrap(),
        40 + reservation
            .requirements()
            .get(pool.topology().host_domain())
            .unwrap()
            .total()
            .unwrap()
    );
}

#[test]
fn all_live_reservations_including_zero_bytes_exclude_unquoted_work() {
    let execution = InferenceExecutionIdentity::default();
    for admitted in [
        zero_admission(),
        admission(geometry(1, OutputDemand::LastPosition)),
    ] {
        for explicit_capacity in [false, true] {
            let bytes = memory::reservation_bytes(&admitted);
            let pool = memory::host_ledger(bytes, 0).unwrap();
            let reservation = if explicit_capacity {
                pool.reserve_with_capacity(&execution, &admitted, memory::resolved_limits(bytes))
            } else {
                pool.reserve(&execution, &admitted)
            }
            .unwrap();
            let completion = reservation.clone();
            let assert_blocked = || {
                assert!(matches!(
                    pool.acquire_unquoted(),
                    Err(WorkingMemoryError::ReservedWorkActive)
                ));
                assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
                assert_eq!(pool.funded_used_bytes().unwrap(), bytes);
                assert_eq!(pool.payload_peak_bytes().unwrap(), bytes);
            };
            assert_blocked();
            drop(reservation);
            assert_blocked();
            drop(completion);
            let unquoted = pool.acquire_unquoted().unwrap();
            assert_eq!(pool.funded_used_bytes().unwrap(), 0);
            assert_eq!(pool.unquoted_owner_count().unwrap(), 1);
            drop(unquoted);
        }
    }
}

#[test]
fn independent_zero_byte_reservations_retain_separate_exclusion_owners() {
    let execution = InferenceExecutionIdentity::default();
    let admitted = zero_admission();
    let bytes = memory::reservation_bytes(&admitted);
    let pool = memory::host_ledger(2 * bytes, 0).unwrap();
    let first = pool.reserve(&execution, &admitted).unwrap();
    let second = pool
        .reserve_with_capacity(&execution, &admitted, memory::resolved_limits(2 * bytes))
        .unwrap();
    drop(first);
    assert!(matches!(
        pool.acquire_unquoted(),
        Err(WorkingMemoryError::ReservedWorkActive)
    ));
    assert_eq!(pool.funded_used_bytes().unwrap(), bytes);
    drop(second);
    let _lease = pool.acquire_unquoted().unwrap();
    assert_eq!(pool.unquoted_owner_count().unwrap(), 1);
}

#[test]
fn completed_storage_can_be_registered_before_retiring_unquoted_ownership() {
    let execution = InferenceExecutionIdentity::default();
    let admitted = admission(geometry(3, OutputDemand::LastPosition));
    let bytes = memory::reservation_bytes(&admitted);
    let (pool, _funding, mut prepared, controls) =
        storage::prepared_ledger(40 + 100 + bytes, 40, &[2, 1, 1, 1, 1]);
    let unquoted = pool.acquire_unquoted().unwrap();
    let storage = prepared
        .remove(0)
        .register_host_storage([(1_u32, 60), (2, 40)])
        .unwrap();
    let alias = prepared
        .remove(0)
        .register_host_storage([(1_u32, 60)])
        .unwrap();
    assert_eq!(pool.funded_used_bytes().unwrap(), 140);
    assert_eq!(
        pool.payload_peak_bytes().unwrap(),
        controls + 140 + MemoryLedger::unquoted_owner_control_bytes().unwrap()
    );
    assert_eq!(pool.unquoted_owner_count().unwrap(), 1);
    assert!(matches!(
        prepared.remove(0).register_host_storage([(1_u32, 61)]),
        Err(WorkingMemoryError::StorageCapacityMismatch { .. })
    ));
    assert!(matches!(
        prepared
            .remove(0)
            .register_host_storage([(3_u32, bytes + 1)]),
        Err(WorkingMemoryError::Domain(
            eredu_core::MemoryDomainError::BudgetExceeded { .. }
        ))
    ));
    assert!(matches!(
        pool.reserve(&execution, &admitted),
        Err(WorkingMemoryError::UnknownBound)
    ));
    assert_eq!(pool.funded_used_bytes().unwrap(), 140);
    assert_eq!(
        pool.payload_peak_bytes().unwrap(),
        controls + 140 + MemoryLedger::unquoted_owner_control_bytes().unwrap()
    );
    drop(unquoted);
    let reservation = pool.reserve(&execution, &admitted).unwrap();
    // Aliases remain registrable while requests are live and capacity is full.
    let live_alias = prepared
        .remove(0)
        .register_host_storage([(2_u32, 40)])
        .unwrap();
    assert_eq!(pool.funded_used_bytes().unwrap(), 140 + bytes);
    drop((storage, alias, live_alias, reservation));
    assert_eq!(pool.funded_used_bytes().unwrap(), 40);
    assert_eq!(pool.payload_peak_bytes().unwrap(), controls + 140 + bytes);
}

#[test]
fn rejected_reservations_do_not_prevent_later_unquoted_work() {
    let execution = InferenceExecutionIdentity::default();
    let admitted = admission(geometry(1, OutputDemand::LastPosition));
    let pool = memory::host_ledger(memory::reservation_bytes(&admitted) - 1, 0).unwrap();
    assert!(matches!(
        pool.reserve(&execution, &admitted),
        Err(WorkingMemoryError::Domain(
            eredu_core::MemoryDomainError::BudgetExceeded { .. }
        ))
    ));
    let lease = pool.acquire_unquoted().unwrap();
    assert_eq!(pool.unquoted_owner_count().unwrap(), 1);
    assert_eq!(pool.funded_used_bytes().unwrap(), 0);
    assert_eq!(
        pool.payload_peak_bytes().unwrap(),
        MemoryLedger::unquoted_owner_control_bytes().unwrap()
    );
    drop(lease);
    let mut unknown = admitted.clone();
    unknown.state.execution_workspace = None;
    assert!(matches!(
        pool.reserve_with_capacity(&execution, &unknown, memory::resolved_limits(0)),
        Err(WorkingMemoryError::UnknownBound)
    ));
    let _lease = pool.acquire_unquoted().unwrap();
    assert_eq!(pool.unquoted_owner_count().unwrap(), 1);
}

#[test]
fn unquoted_work_and_reservation_races_have_one_winner_even_for_zero_bytes() {
    for admitted in [
        zero_admission(),
        admission(geometry(1, OutputDemand::LastPosition)),
    ] {
        for explicit_capacity in [false, true] {
            for _ in 0..16 {
                let bytes = memory::reservation_bytes(&admitted);
                let pool = memory::host_ledger(bytes, 0).unwrap();
                let barrier = Arc::new(std::sync::Barrier::new(2));
                let quoted_pool = pool.clone();
                let quoted_barrier = barrier.clone();
                let admitted = admitted.clone();
                let quoted = std::thread::spawn(move || {
                    quoted_barrier.wait();
                    let execution = InferenceExecutionIdentity::default();
                    if explicit_capacity {
                        quoted_pool.reserve_with_capacity(
                            &execution,
                            &admitted,
                            memory::resolved_limits(bytes),
                        )
                    } else {
                        quoted_pool.reserve(&execution, &admitted)
                    }
                });
                let unquoted_pool = pool.clone();
                let unquoted = std::thread::spawn(move || {
                    barrier.wait();
                    unquoted_pool.acquire_unquoted()
                });
                let quoted = quoted.join().unwrap();
                let unquoted = unquoted.join().unwrap();
                match (&quoted, &unquoted) {
                    (Ok(_), Err(WorkingMemoryError::ReservedWorkActive)) => {
                        assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
                        assert_eq!(pool.funded_used_bytes().unwrap(), bytes);
                        assert_eq!(pool.payload_peak_bytes().unwrap(), bytes);
                    }
                    (Err(WorkingMemoryError::UnknownBound), Ok(_)) => {
                        assert_eq!(pool.unquoted_owner_count().unwrap(), 1);
                        assert_eq!(pool.funded_used_bytes().unwrap(), 0);
                        assert_eq!(
                            pool.payload_peak_bytes().unwrap(),
                            MemoryLedger::unquoted_owner_control_bytes().unwrap()
                        );
                    }
                    other => panic!("nonexclusive admission: {other:?}"),
                }
                drop((quoted, unquoted));
                assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
                assert_eq!(pool.funded_used_bytes().unwrap(), 0);
            }
        }
    }
}

#[test]
fn planner_does_not_retry_smaller_chunks_while_domain_bound_is_unknown() {
    let pool = memory::host_ledger(65536, 0).unwrap();
    let _unquoted = pool.acquire_unquoted().unwrap();
    let initial = geometry(7, OutputDemand::LastPosition);
    let mut seen = Vec::new();
    let result = plan_prefill_with_capacity(
        &InferenceExecutionIdentity::default(),
        &pool,
        &capabilities(),
        request(initial),
        initial,
        memory::resolved_limits(4096),
        |g| {
            seen.push(g.prefill_chunk_positions);
            quote(g)
        },
    );
    assert!(matches!(
        result,
        Err(PrefillPlanningError::Reservation(
            WorkingMemoryError::UnknownBound
        ))
    ));
    assert_eq!(seen, [7]);
    assert_eq!(pool.unquoted_owner_count().unwrap(), 1);
    assert_eq!(pool.funded_used_bytes().unwrap(), 0);
}
