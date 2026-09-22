use super::*;
use crate::working_memory::{
    OriginalHostSourceCustody, OriginalOperationMetadataCustody,
    memory_fixture::{self, separate},
};
use eredu_core::MemoryLimit;

fn requirements(pool: &MemoryLedger, bytes: u64) -> NumericalSourceRequirements {
    let mut native = DomainMemoryRequirements::zero(pool.topology());
    native
        .add_allocation(bytes, &separate::device_placement(pool))
        .unwrap();
    NumericalSourceRequirements::new(
        native,
        Some(128),
        Some(1024),
        HostSourceConstructionFacts::new(64, 2, 1).unwrap(),
    )
    .unwrap()
}

#[test]
fn numerical_source_claims_keep_one_atomic_account_without_inference_authority() {
    let pool = separate::device_ledger(64, 0).unwrap();
    let before = pool.snapshot().unwrap();
    let execution = InferenceExecutionIdentity::default();
    let mut source = pool
        .reserve_numerical_source(
            &execution,
            requirements(&pool, 64),
            pool.configured_limits().clone(),
        )
        .unwrap();
    source.validate(&execution).unwrap();
    assert_eq!(
        source.validate(&InferenceExecutionIdentity::default()),
        Err(WorkingMemoryError::IdentityMismatch)
    );
    let custody = source.budget_custody();
    let host: OriginalHostSourceCustody = custody.clone().into();
    let metadata: OriginalOperationMetadataCustody = custody.clone().into();
    metadata.validate_retained_origin(&pool).unwrap();
    let foreign = memory_fixture::host_ledger(1 << 20, 0).unwrap();
    assert_eq!(
        metadata.validate_retained_origin(&foreign),
        Err(WorkingMemoryError::IdentityMismatch)
    );
    let mut bank = source.take_source_constructions().unwrap();
    assert!(bank.belongs_to_source(&host));
    assert!(source.take_source_constructions().is_err());
    let receipt = bank.try_debit(64).unwrap();
    let failure = bank.try_debit(1).unwrap_err();
    assert!(failure.retains_receipt());
    let native = source.claim_native().unwrap();
    assert!(source.claim_native().is_err());
    native
        .validate_allocation(64, &separate::device_placement(&pool))
        .unwrap();
    assert_eq!(
        native.validate_allocation(63, &separate::device_placement(&pool)),
        Err(WorkingMemoryError::IdentityMismatch)
    );
    assert_eq!(
        native.validate_allocation(64, &pool.host_placement_handle()),
        Err(WorkingMemoryError::IdentityMismatch)
    );
    assert_eq!(native.metadata_bytes(), 128);
    let funding = source.metadata_funding().unwrap();
    funding.reserve_metadata(1024).unwrap();
    assert!(matches!(
        funding.reserve_metadata(1),
        Err(HostMetadataFundingError::Capacity { .. })
    ));
    assert!(source.metadata_funding().is_err());
    drop((
        source, bank, native, custody, host, metadata, funding, receipt,
    ));
    assert_ne!(
        pool.snapshot().unwrap().domains[1].current_charge_bytes,
        before.domains[1].current_charge_bytes
    );
    drop(failure);
    let after = pool.snapshot().unwrap();
    for (a, b) in after.domains.iter().zip(before.domains.iter()) {
        assert_eq!(a.current_charge_bytes, b.current_charge_bytes);
    }
}

#[test]
fn numerical_last_domain_refusal_is_atomic_under_finite_and_unlimited_host_limits() {
    for host_limit in [MemoryLimit::Finite(1 << 20), MemoryLimit::Unlimited] {
        let pool = separate::device_ledger(63, 0).unwrap();
        let limits = MemoryLimits::resolve(
            pool.topology(),
            [
                (pool.topology().host_domain(), host_limit),
                (separate::device_domain(&pool), MemoryLimit::Finite(63)),
            ],
        )
        .unwrap();
        let before = pool.snapshot().unwrap();
        let error = pool
            .reserve_numerical_source(
                &InferenceExecutionIdentity::default(),
                requirements(&pool, 64),
                limits,
            )
            .unwrap_err();
        assert!(
            matches!(error.cause(), WorkingMemoryError::Domain(eredu_core::MemoryDomainError::BudgetExceeded { domain, .. }) if *domain==separate::device_domain(&pool))
        );
        assert_eq!(pool.snapshot().unwrap(), before);
    }
}

#[test]
fn numerical_source_quarantine_prevents_all_new_claims_and_preserves_charge() {
    let pool = separate::device_ledger(64, 0).unwrap();
    let mut source = pool
        .reserve_numerical_source(
            &InferenceExecutionIdentity::default(),
            requirements(&pool, 64),
            pool.configured_limits().clone(),
        )
        .unwrap();
    let custody = source.budget_custody();
    let before = pool.snapshot().unwrap();
    custody.quarantine();
    assert!(matches!(
        source.claim_native(),
        Err(WorkingMemoryError::ExecutionFenced)
    ));
    assert!(matches!(
        source.take_source_constructions(),
        Err(WorkingMemoryError::ExecutionFenced)
    ));
    let metadata: OriginalOperationMetadataCustody = custody.clone().into();
    assert_eq!(
        metadata.validate_retained_origin(&pool),
        Err(WorkingMemoryError::ExecutionFenced)
    );
    drop((metadata, source, custody));
    let after = pool.snapshot().unwrap();
    for (a, b) in after.domains.iter().zip(before.domains.iter()) {
        assert_eq!(a.current_charge_bytes, b.current_charge_bytes);
    }
}

#[test]
fn numerical_native_claim_preserves_candidate_domains_and_estimation_basis() {
    let pool = separate::device_ledger(64, 0).unwrap();
    let placement = MemoryPlacement::possible(
        pool.topology(),
        vec![
            separate::device_domain(&pool),
            pool.topology().host_domain(),
            separate::device_domain(&pool),
        ],
        "managed allocator candidate domains".into(),
    )
    .unwrap();
    let mut native = DomainMemoryRequirements::zero(pool.topology());
    native.add_allocation(64, &placement).unwrap();
    let requirements = NumericalSourceRequirements::new(
        native,
        Some(128),
        Some(1024),
        HostSourceConstructionFacts::new(0, 0, 0).unwrap(),
    )
    .unwrap();
    let mut source = pool
        .reserve_numerical_source(
            &InferenceExecutionIdentity::default(),
            requirements,
            pool.configured_limits().clone(),
        )
        .unwrap();
    let native = source.claim_native().unwrap();
    native.validate_allocation(64, &placement).unwrap();
    let changed_basis = MemoryPlacement::possible(
        pool.topology(),
        placement.domains().to_vec(),
        "different allocation mechanism".into(),
    )
    .unwrap();
    assert_eq!(
        native.validate_allocation(64, &changed_basis),
        Err(WorkingMemoryError::IdentityMismatch)
    );
    assert_eq!(
        native.validate_allocation(64, &separate::device_placement(&pool)),
        Err(WorkingMemoryError::IdentityMismatch)
    );
    drop(source);
    for domain in pool.snapshot().unwrap().domains {
        assert_eq!(domain.estimated_placement_allowance_bytes, 64);
    }
    drop(native);
    for domain in pool.snapshot().unwrap().domains {
        assert_eq!(domain.estimated_placement_allowance_bytes, 0);
    }
}

#[test]
fn competing_numerical_sources_accept_one_complete_account() {
    let pool = separate::device_ledger(64, 0).unwrap();
    let before = pool.snapshot().unwrap();
    let execution = InferenceExecutionIdentity::default();
    let start = std::sync::Barrier::new(2);
    let accepted = std::sync::Barrier::new(2);
    let results = std::thread::scope(|threads| {
        let workers = [0, 1].map(|_| {
            threads.spawn(|| {
                let requirements = requirements(&pool, 64);
                let limits = pool.configured_limits().clone();
                start.wait();
                let result = pool.reserve_numerical_source(&execution, requirements, limits);
                // Keep the winning account alive through both admission attempts.
                accepted.wait();
                result
            })
        });
        workers.map(|worker| worker.join().unwrap())
    });
    assert_eq!(results.iter().filter(|result| result.is_ok()).count(), 1);
    let error = results
        .iter()
        .find_map(|result| result.as_ref().err())
        .unwrap();
    assert!(matches!(
        error.cause(),
        WorkingMemoryError::Domain(eredu_core::MemoryDomainError::BudgetExceeded { domain, .. })
            if *domain == separate::device_domain(&pool)
    ));
    assert_eq!(
        pool.snapshot().unwrap().reservations,
        before.reservations + 1
    );
    drop(results);
    let after = pool.snapshot().unwrap();
    for (a, b) in after.domains.iter().zip(before.domains.iter()) {
        assert_eq!(a.current_charge_bytes, b.current_charge_bytes);
    }
}

#[derive(Debug)]
struct CompletedFacts(Arc<MemoryTopology>);
impl eredu_nn::workspace::WorkspaceMechanisms for CompletedFacts {
    fn memory_topology(&self) -> Option<&MemoryTopology> {
        Some(&self.0)
    }
    fn operation_bound(
        &self,
        _: &eredu_nn::workspace::WorkspaceOperation,
    ) -> Result<Option<eredu_nn::workspace::WorkspaceOperationBound>, eredu_nn::Error> {
        Ok(None)
    }
}

#[test]
fn completed_standalone_sources_preserve_domain_categories_and_last_alias_custody() {
    use crate::working_memory::{
        CompletedWorkspaceSourceAccount, CompletedWorkspaceSourceLayout,
        RegisteredWorkspaceStorageLayout, RegisteredWorkspaceStorageRow,
    };
    use eredu_core::HostPreparationAuthority;
    use eredu_nn::workspace::{WorkspaceContext, WorkspaceExistingStorage};
    for unified in [false, true] {
        for managed in [false, true] {
            let pool = if unified {
                memory_fixture::host_ledger(1 << 24, 0).unwrap()
            } else {
                separate::device_ledger(64, 0).unwrap()
            };
            let before = pool.snapshot().unwrap();
            let target = if unified {
                pool.topology().host_domain()
            } else {
                separate::device_domain(&pool)
            };
            let placement = if managed {
                MemoryPlacement::possible(
                    pool.topology(),
                    vec![pool.topology().host_domain(), target],
                    "completed fixture candidate domains".into(),
                )
                .unwrap()
            } else {
                MemoryPlacement::fixed(pool.topology(), target).unwrap()
            };
            let mut native = DomainMemoryRequirements::zero(pool.topology());
            native.add_allocation(64, &placement).unwrap();
            let source = pool
                .reserve_numerical_source(
                    &InferenceExecutionIdentity::default(),
                    NumericalSourceRequirements::new(
                        native,
                        Some(128),
                        Some(1 << 20),
                        HostSourceConstructionFacts::new(0, 0, 0).unwrap(),
                    )
                    .unwrap(),
                    pool.configured_limits().clone(),
                )
                .unwrap();
            let custody = source.budget_custody();
            let account = CompletedWorkspaceSourceAccount::Standalone(custody.clone());
            let host = HostPreparationAuthority::retain(custody.clone());
            let context = WorkspaceContext::new(CompletedFacts(pool.topology_handle()));
            let root = |bytes, placement: &MemoryPlacement| {
                WorkspaceExistingStorage::try_new_placed_with_host_controls(
                    Some(bytes),
                    placement,
                    Some(32),
                    &context,
                )
                .unwrap()
            };
            let make = |roots: Vec<WorkspaceExistingStorage>| {
                CompletedWorkspaceSourceLayout::new_accounts(roots.len())
                    .unwrap()
                    .construct_accounts(
                        &context,
                        roots
                            .into_iter()
                            .map(|root| (root, account.clone()))
                            .collect(),
                        &host,
                    )
            };
            assert!(matches!(
                make(vec![root(33, &placement), root(32, &placement)]),
                Err(WorkingMemoryError::IdentityMismatch)
            ));
            if managed {
                let wrong = MemoryPlacement::possible(
                    pool.topology(),
                    placement.domains().to_vec(),
                    "different mechanism".into(),
                )
                .unwrap();
                assert!(matches!(
                    make(vec![root(64, &wrong)]),
                    Err(WorkingMemoryError::IdentityMismatch)
                ));
                let fixed = MemoryPlacement::fixed(pool.topology(), target).unwrap();
                assert!(matches!(
                    make(vec![root(64, &fixed)]),
                    Err(WorkingMemoryError::IdentityMismatch)
                ));
            } else if !unified {
                assert!(matches!(
                    make(vec![root(64, &pool.host_placement_handle())]),
                    Err(WorkingMemoryError::IdentityMismatch)
                ));
            }
            let first = root(32, &placement);
            assert!(matches!(
                make(vec![first.clone(), first.clone()]),
                Err(WorkingMemoryError::IdentityMismatch)
            ));
            let completed = make(vec![first, root(32, &placement)]).unwrap();
            let charged = pool.snapshot().unwrap();
            let binding =
                RegisteredWorkspaceStorageLayout::<u64>::new_with_completed_source(0, &completed)
                    .unwrap()
                    .construct_with_completed_source(
                        &pool,
                        &context,
                        std::iter::empty::<RegisteredWorkspaceStorageRow<u64>>(),
                        completed,
                    )
                    .unwrap();
            let alias = binding.clone();
            {
                let usage = pool.0.usage.lock().unwrap();
                binding
                    .registration()
                    .validate_copy_source(&pool, &usage)
                    .unwrap();
            }
            for (a, b) in pool
                .snapshot()
                .unwrap()
                .domains
                .iter()
                .zip(charged.domains.iter())
            {
                assert_eq!(
                    a.registered_storage_bytes, b.registered_storage_bytes,
                    "completed backing is not republished"
                );
                assert_eq!(
                    a.estimated_placement_allowance_bytes,
                    b.estimated_placement_allowance_bytes
                );
            }
            drop((source, custody, host, account, binding, context));
            assert!(
                pool.snapshot()
                    .unwrap()
                    .domains
                    .iter()
                    .zip(before.domains.iter())
                    .any(|(a, b)| a.current_charge_bytes > b.current_charge_bytes)
            );
            drop(alias);
            for (a, b) in pool
                .snapshot()
                .unwrap()
                .domains
                .iter()
                .zip(before.domains.iter())
            {
                assert_eq!(a.current_charge_bytes, b.current_charge_bytes);
            }
        }
    }
}

#[test]
fn completed_standalone_sources_reject_quarantined_accounts() {
    use eredu_nn::workspace::{WorkspaceContext, WorkspaceExistingStorage};
    let pool = separate::device_ledger(64, 0).unwrap();
    let source = pool
        .reserve_numerical_source(
            &InferenceExecutionIdentity::default(),
            requirements(&pool, 64),
            pool.configured_limits().clone(),
        )
        .unwrap();
    let custody = source.budget_custody();
    let context = WorkspaceContext::new(CompletedFacts(pool.topology_handle()));
    let root = WorkspaceExistingStorage::try_new_placed_with_host_controls(
        Some(64),
        &separate::device_placement(&pool),
        Some(0),
        &context,
    )
    .unwrap();
    custody
        .validate_completed_roots([&root].into_iter())
        .unwrap();
    custody.quarantine();
    assert!(
        custody
            .validate_completed_roots([&root].into_iter())
            .is_err()
    );
}

fn lifetime_source(pool: &MemoryLedger, placement: &MemoryPlacement) -> OriginalNumericalSource {
    let mut native = DomainMemoryRequirements::zero(pool.topology());
    native.add_allocation(64, placement).unwrap();
    pool.reserve_numerical_source(
        &InferenceExecutionIdentity::default(),
        NumericalSourceRequirements::new(
            native,
            Some(128),
            Some(1024),
            HostSourceConstructionFacts::new(0, 0, 0).unwrap(),
        )
        .unwrap(),
        pool.configured_limits().clone(),
    )
    .unwrap()
}

#[test]
fn completed_native_backings_retire_independently_in_every_candidate_domain() {
    for unified in [false, true] {
        for managed in [false, true] {
            let pool = if unified {
                memory_fixture::host_ledger(1 << 24, 0).unwrap()
            } else {
                separate::device_ledger(64, 0).unwrap()
            };
            let before = pool.snapshot().unwrap();
            let target = if unified {
                pool.topology().host_domain()
            } else {
                separate::device_domain(&pool)
            };
            let placement = Arc::new(if managed {
                MemoryPlacement::possible(
                    pool.topology(),
                    vec![pool.topology().host_domain(), target],
                    "completed native backing candidate set".into(),
                )
                .unwrap()
            } else {
                MemoryPlacement::fixed(pool.topology(), target).unwrap()
            });
            let mut source = lifetime_source(&pool, &placement);
            let custody = source.budget_custody();
            let alias = custody.clone();
            let lifetime = source
                .claim_native()
                .unwrap()
                .bind_lifetime(64, Arc::clone(&placement))
                .unwrap();
            let admitted = pool.snapshot().unwrap();
            drop(source);
            let mut minimum = 64;
            for occupancy in [64, 48, 32, 48, 32, 0, 16] {
                lifetime.retire_completed_occupancy(occupancy);
                minimum = minimum.min(occupancy);
                let current = pool.snapshot().unwrap();
                for (a, b) in current.domains.iter().zip(&admitted.domains) {
                    let released = if placement.domains().contains(&a.domain) {
                        64 - minimum
                    } else {
                        0
                    };
                    assert_eq!(a.current_charge_bytes, b.current_charge_bytes - released);
                    assert_eq!(
                        a.outstanding_reservation_bytes,
                        b.outstanding_reservation_bytes - released
                    );
                    assert_eq!(a.historical_peak_bytes, b.historical_peak_bytes);
                    assert_eq!(a.effective_limit, b.effective_limit);
                    assert_eq!(a.reservation_control_bytes, b.reservation_control_bytes);
                    assert_eq!(
                        a.estimated_placement_allowance_bytes,
                        b.estimated_placement_allowance_bytes - if managed { released } else { 0 }
                    );
                }
                assert_eq!(current.reservations, admitted.reservations);
                custody
                    .validate_completed_allocations([(minimum, &*placement)].into_iter())
                    .unwrap();
                assert_eq!(
                    custody
                        .validate_completed_allocations([(minimum + 1, &*placement)].into_iter()),
                    Err(WorkingMemoryError::IdentityMismatch)
                );
            }
            let retired = pool.snapshot().unwrap();
            drop((custody, lifetime));
            assert_eq!(pool.snapshot().unwrap(), retired);
            drop(alias);
            for (a, b) in pool.snapshot().unwrap().domains.iter().zip(&before.domains) {
                assert_eq!(a.current_charge_bytes, b.current_charge_bytes);
            }
        }
    }
}

#[test]
fn released_native_capacity_can_fund_an_independent_live_destination() {
    let pool = separate::device_ledger(64, 0).unwrap();
    let before = pool.snapshot().unwrap();
    let placement = separate::device_placement(&pool);
    let mut source = lifetime_source(&pool, &placement);
    let custody = source.budget_custody();
    let lifetime = source
        .claim_native()
        .unwrap()
        .bind_lifetime(64, placement)
        .unwrap();
    drop(source);
    lifetime.retire_completed_occupancy(48);
    let destination = pool
        .reserve_numerical_source(
            &InferenceExecutionIdentity::default(),
            requirements(&pool, 16),
            pool.configured_limits().clone(),
        )
        .unwrap();
    let full = pool.snapshot().unwrap();
    assert_eq!(full.domains[1].current_charge_bytes, 64);
    assert!(
        pool.reserve_numerical_source(
            &InferenceExecutionIdentity::default(),
            requirements(&pool, 1),
            pool.configured_limits().clone(),
        )
        .is_err()
    );
    assert_eq!(pool.snapshot().unwrap(), full);
    lifetime.retire_completed_occupancy(0);
    assert_eq!(pool.snapshot().unwrap().domains[1].current_charge_bytes, 16);
    drop((lifetime, custody, destination));
    for (a, b) in pool.snapshot().unwrap().domains.iter().zip(&before.domains) {
        assert_eq!(a.current_charge_bytes, b.current_charge_bytes);
    }
}

#[test]
fn concurrent_closed_occupancy_observations_retire_each_byte_once() {
    let pool = separate::device_ledger(64, 0).unwrap();
    let placement = separate::device_placement(&pool);
    let mut source = lifetime_source(&pool, &placement);
    let lifetime = source
        .claim_native()
        .unwrap()
        .bind_lifetime(64, placement)
        .unwrap();
    let start = std::sync::Barrier::new(4);
    std::thread::scope(|threads| {
        for occupied in [48, 32, 16, 0] {
            let start = &start;
            let lifetime = &lifetime;
            threads.spawn(move || {
                start.wait();
                lifetime.retire_completed_occupancy(occupied);
            });
        }
    });
    assert_eq!(pool.snapshot().unwrap().domains[1].current_charge_bytes, 0);
    source.validate(&source.account.value().execution).unwrap();
    assert_eq!(
        source
            .account
            .value()
            .native_released
            .load(Ordering::Acquire),
        64
    );
}

#[test]
fn invalid_occupancy_and_unwinding_aliases_quarantine_all_remaining_charges() {
    for unwind in [false, true] {
        let pool = separate::device_ledger(64, 0).unwrap();
        let placement = separate::device_placement(&pool);
        let mut source = lifetime_source(&pool, &placement);
        let custody = source.budget_custody();
        let lifetime = source
            .claim_native()
            .unwrap()
            .bind_lifetime(64, placement)
            .unwrap();
        lifetime.retire_completed_occupancy(48);
        let before = pool.snapshot().unwrap();
        if unwind {
            let alias = custody.clone();
            assert!(
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
                    let _alias = alias;
                    panic!("native owner retirement failed");
                }))
                .is_err()
            );
        } else {
            lifetime.retire_completed_occupancy(65);
        }
        lifetime.retire_completed_occupancy(0);
        assert_eq!(
            source
                .account
                .value()
                .native_released
                .load(Ordering::Acquire),
            16
        );
        assert_eq!(
            source.validate(&source.account.value().execution),
            Err(WorkingMemoryError::ExecutionFenced)
        );
        drop((source, custody, lifetime));
        let after = pool.snapshot().unwrap();
        for (a, b) in after.domains.iter().zip(&before.domains) {
            assert_eq!(a.current_charge_bytes, b.current_charge_bytes);
            assert_eq!(a.historical_peak_bytes, b.historical_peak_bytes);
        }
    }
}

#[test]
fn poisoned_coordinator_never_refunds_completed_native_occupancy() {
    let pool = separate::device_ledger(64, 0).unwrap();
    let placement = separate::device_placement(&pool);
    let mut source = lifetime_source(&pool, &placement);
    let lifetime = source
        .claim_native()
        .unwrap()
        .bind_lifetime(64, placement)
        .unwrap();
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _locked = pool.0.usage.lock().unwrap();
            panic!("interrupted coordinator mutation");
        }))
        .is_err()
    );
    lifetime.retire_completed_occupancy(0);
    assert_eq!(
        source
            .account
            .value()
            .native_released
            .load(Ordering::Acquire),
        0
    );
    drop((source, lifetime));
    let usage = pool.0.usage.lock().unwrap_err().into_inner();
    assert_eq!(usage.domains[1].reserved, 64);
}
