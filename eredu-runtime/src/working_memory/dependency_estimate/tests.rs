use super::*;
use crate::working_memory::memory_fixture;
use eredu_core::MemoryLimit;

const SOURCE: &str = "stock parser";
const BASIS: &str = "bounded input bytes times configured parser multiplier";

#[test]
fn cumulative_dependency_inputs_admit_atomically_and_retain_each_allowance() {
    let pool = memory_fixture::host_ledger(1 << 24, 0).unwrap();
    let execution = InferenceExecutionIdentity::default();
    let initial = pool.snapshot().unwrap();
    let construction = pool
        .prepare_workspace_metadata(&execution, pool.configured_limits().clone())
        .unwrap();
    let funding = pool
        .prepare_dependency_estimates(
            &execution,
            pool.configured_limits(),
            SOURCE,
            BASIS,
            construction,
        )
        .unwrap();
    assert_eq!(
        pool.snapshot().unwrap().domains[0].estimated_overhead_bytes,
        0
    );
    funding.reserve_metadata(123).unwrap();
    funding.reserve_metadata(456).unwrap();
    assert_eq!(
        pool.snapshot().unwrap().domains[0].estimated_overhead_bytes,
        579
    );
    let before = pool.snapshot().unwrap();
    let next = pool.0.usage.lock().unwrap().next_funding;
    assert!(matches!(
        funding.reserve_metadata(1 << 24),
        Err(HostMetadataFundingError::Domain(
            MemoryDomainError::BudgetExceeded { .. }
        ))
    ));
    assert_eq!(pool.snapshot().unwrap(), before);
    assert_eq!(pool.0.usage.lock().unwrap().next_funding, next);
    let alias = funding.clone();
    drop(funding);
    assert_eq!(pool.snapshot().unwrap(), before);
    drop(alias);
    let after = pool.snapshot().unwrap();
    assert_eq!(
        after.domains[0].current_charge_bytes,
        initial.domains[0].current_charge_bytes
    );
    assert_eq!(after.reservations, initial.reservations);
}

fn estimate(pool: &MemoryLedger, upper: u64) -> DependencyEstimateFunding {
    pool.reserve_host_dependency_estimate(
        &InferenceExecutionIdentity::default(),
        pool.configured_limits(),
        SOURCE,
        FiniteMemoryEstimate::new(0, upper).unwrap(),
        BASIS,
    )
    .unwrap()
}

#[test]
fn dependency_estimates_preserve_category_alias_custody_and_independent_labels() {
    let pool = memory_fixture::host_ledger(1 << 24, 0).unwrap();
    let before = pool.snapshot().unwrap();
    let first = estimate(&pool, 1234);
    assert_eq!(first.estimate().source, SOURCE);
    assert_eq!(first.estimate().basis, BASIS);
    first.funding().reserve_metadata(1234).unwrap();
    assert!(matches!(
        first.funding().reserve_metadata(1),
        Err(HostMetadataFundingError::Capacity { .. })
    ));
    let alias = first.clone();
    let current = pool.snapshot().unwrap();
    assert_eq!(current.domains[0].estimated_overhead_bytes, 1234);
    assert!(
        current.domains[0].reservation_control_bytes > before.domains[0].reservation_control_bytes
    );
    let second = estimate(&pool, 1234);
    assert_eq!(
        pool.snapshot().unwrap().domains[0].estimated_overhead_bytes,
        2468
    );
    drop(first);
    assert_eq!(
        pool.snapshot().unwrap().domains[0].estimated_overhead_bytes,
        2468
    );
    drop(alias);
    assert_eq!(
        pool.snapshot().unwrap().domains[0].estimated_overhead_bytes,
        1234
    );
    drop(second);
    let after = pool.snapshot().unwrap();
    assert_eq!(
        after.domains[0].current_charge_bytes,
        before.domains[0].current_charge_bytes
    );
    assert_eq!(after.domains[0].estimated_overhead_bytes, 0);
    assert_eq!(after.reservations, before.reservations);
}

#[test]
fn dependency_estimate_exact_fit_and_refusals_leave_the_transaction_unchanged() {
    let pool = memory_fixture::host_ledger(1 << 24, 0).unwrap();
    let execution = InferenceExecutionIdentity::default();
    let before = pool.snapshot().unwrap();
    let host = pool.topology().host_domain();
    let constructor = HostMetadataFunding::constructor_bytes::<EstimateAccount>().unwrap() as u64;
    let charge = controls(pool.topology(), SOURCE.len(), BASIS.len(), constructor).unwrap() + 17;
    let ceiling = before.domains[0].current_charge_bytes + charge;
    for capacity in [0, ceiling - 1] {
        let limits =
            MemoryLimits::resolve(pool.topology(), [(host, MemoryLimit::Finite(capacity))])
                .unwrap();
        let next = pool.0.usage.lock().unwrap().next_funding;
        let cause = pool
            .reserve_host_dependency_estimate(
                &execution,
                &limits,
                SOURCE,
                FiniteMemoryEstimate::new(0, 17).unwrap(),
                BASIS,
            )
            .unwrap_err();
        assert!(
            matches!(cause.cause(), WorkingMemoryError::Domain(MemoryDomainError::BudgetExceeded { domain, .. }) if *domain == host)
        );
        assert_eq!(pool.snapshot().unwrap(), before);
        assert_eq!(pool.0.usage.lock().unwrap().next_funding, next);
    }
    let limits =
        MemoryLimits::resolve(pool.topology(), [(host, MemoryLimit::Finite(ceiling))]).unwrap();
    let value = pool
        .reserve_host_dependency_estimate(
            &execution,
            &limits,
            SOURCE,
            FiniteMemoryEstimate::new(0, 17).unwrap(),
            BASIS,
        )
        .unwrap();
    assert_eq!(
        pool.snapshot().unwrap().domains[0].current_charge_bytes,
        ceiling
    );
    assert_eq!(
        pool.snapshot().unwrap().domains[0].effective_limit,
        MemoryLimit::Finite(ceiling)
    );
    drop(value);
    assert_eq!(
        pool.snapshot().unwrap().domains[0].current_charge_bytes,
        before.domains[0].current_charge_bytes
    );
}

#[test]
fn unlimited_dependency_estimates_check_overflow_identity_and_descriptions() {
    let topology = memory_fixture::host_topology();
    let pool = MemoryLedger::new(
        topology.clone(),
        MemoryLimits::unlimited(&topology),
        DomainMemoryRequirements::zero(&topology),
    )
    .unwrap();
    let limits = MemoryLimits::unlimited(pool.topology());
    let execution = InferenceExecutionIdentity::default();
    let before = pool.snapshot().unwrap();
    assert_eq!(before.domains[0].configured_limit, MemoryLimit::Unlimited);
    let next = pool.0.usage.lock().unwrap().next_funding;
    let error = pool
        .reserve_host_dependency_estimate(
            &execution,
            &limits,
            SOURCE,
            FiniteMemoryEstimate::new(0, u64::MAX).unwrap(),
            BASIS,
        )
        .unwrap_err();
    assert!(matches!(error.cause(), WorkingMemoryError::Overflow));
    let foreign = memory_fixture::separate::device_ledger(1, 0).unwrap();
    let error = pool
        .reserve_host_dependency_estimate(
            &execution,
            foreign.configured_limits(),
            SOURCE,
            FiniteMemoryEstimate::new(0, 1).unwrap(),
            BASIS,
        )
        .unwrap_err();
    assert!(matches!(
        error.cause(),
        WorkingMemoryError::Domain(MemoryDomainError::ForeignTopology)
    ));
    let error = pool
        .reserve_host_dependency_estimate(
            &execution,
            &limits,
            " ",
            FiniteMemoryEstimate::new(0, 1).unwrap(),
            BASIS,
        )
        .unwrap_err();
    assert!(matches!(
        error.cause(),
        WorkingMemoryError::Domain(MemoryDomainError::InvalidEstimateDescription)
    ));
    assert_eq!(pool.snapshot().unwrap(), before);
    assert_eq!(pool.0.usage.lock().unwrap().next_funding, next);
    let zero = pool
        .reserve_host_dependency_estimate(
            &execution,
            &limits,
            SOURCE,
            FiniteMemoryEstimate::new(0, 0).unwrap(),
            BASIS,
        )
        .unwrap();
    assert!(
        pool.snapshot().unwrap().domains[0].current_charge_bytes
            > before.domains[0].current_charge_bytes
    );
    assert!(zero.funding().reserve_metadata(1).is_err());
}
