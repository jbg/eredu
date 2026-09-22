use super::*;

pub(in crate::backend::runtime::checkpoint::store) fn qualified_bytes() -> Option<u64> {
    let required = CacheHandle::required_storage_bytes();
    if std::env::var_os("EREDU_REQUIRE_QUALIFIED_CACHE_CONTEXT").is_some() {
        assert!(required.is_ok(), "{required:?}");
    }
    match required {
        Ok(bytes) => Some(bytes),
        Err(WorkingMemoryError::UnknownBound) => {
            let pool = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
            let error = CacheHandle::prepare(&pool).unwrap_err();
            assert!(matches!(
                error.0.accounting_failure(),
                Some(WorkingMemoryError::UnknownBound)
            ));
            assert!(error.0.rejected_plan().is_some());
            assert!(error.0.completed_output().is_none());
            assert_eq!(pool.fixture_host_charge().unwrap(), 0);
            None
        }
        Err(error) => panic!("unexpected cache layout refusal: {error}"),
    }
}

// Stateless ledger-only reservation for domain comparison. It does not admit
// model work, streams, source catalogs, cache rows or any native producer.
pub(in crate::backend::runtime::checkpoint::store) fn reservation_admission(
) -> eredu_core::Admission {
    use eredu_core::{
        Admission, EstimationCompleteness, ExecutionWorkspaceEstimate, InferenceGeometry,
        InputTokenCount, LayerSchedule, OutputDemand, StateMemoryLayout, WorkspaceBound,
    };
    let geometry = InferenceGeometry {
        batch_size: 1,
        cached_positions: 0,
        input_positions: 1,
        max_output_tokens: 0,
        prefill_chunk_positions: 1,
        output: OutputDemand::StateOnly,
    };
    let layout = StateMemoryLayout::new(
        LayerSchedule::empty(),
        vec![],
        1,
        1,
        EstimationCompleteness::Complete,
    )
    .unwrap();
    let zero = || WorkspaceBound::bounded(0, "stateless cache-origin comparison");
    let state = eredu_core::estimate_runtime_state(
        &layout,
        InputTokenCount::text(1),
        0,
        1,
        std::num::NonZeroU8::new(4).unwrap(),
    )
    .unwrap()
    .with_execution_workspace(crate::memory_fixture::workspace(
        ExecutionWorkspaceEstimate {
            physical_domains: None,
            geometry,
            activations: zero(),
            attention: zero(),
            vocabulary: zero(),
            state_update: zero(),
            materialization: zero(),
            retained: zero(),
        },
    ))
    .unwrap();
    crate::memory_fixture::admission(Admission {
        additional_headroom: Default::default(),
        memory_limits: Default::default(),
        requested_positions: 1,
        state,
        incremental_required_bytes: Some(0),
    })
}
fn reservation_controls() -> u64 {
    let probe = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
    crate::memory_fixture::host_total(
        &probe
            .reservation_requirements(&reservation_admission(), None)
            .unwrap(),
    )
}
fn reservation(pool: &MemoryLedger) -> WorkingMemoryReservation {
    pool.reserve(
        &eredu_runtime::working_memory::InferenceExecutionIdentity::default(),
        &reservation_admission(),
    )
    .unwrap()
}

#[test]
fn cache_capsule_exact_and_one_short_compare_before_shared_owner_construction() {
    let Some(bytes) = qualified_bytes() else {
        return;
    };
    assert!(bytes > CacheHandle::retained_storage_bytes().unwrap());
    let short = crate::memory_fixture::ledger(bytes - 1, 0).unwrap();
    let refused = CacheHandle::prepare(&short).unwrap_err();
    assert!(matches!(refused.0.accounting_failure(),
        Some(WorkingMemoryError::Domain(eredu_core::MemoryDomainError::BudgetExceeded { requested_bytes: required_bytes, limit_bytes, existing_bytes, .. }))
        if *required_bytes == bytes && limit_bytes.checked_sub(*existing_bytes).unwrap() == bytes - 1));
    assert!(refused.0.rejected_plan().is_some());
    assert!(refused.0.completed_output().is_none());
    assert_eq!(short.fixture_host_charge().unwrap(), 0);
    let exact = crate::memory_fixture::ledger(bytes, 0).unwrap();
    let cache = CacheHandle::prepare(&exact).unwrap();
    assert!(cache.try_lock().unwrap().is_empty());
    assert_eq!(exact.fixture_host_charge().unwrap(), bytes);
    let second = CacheHandle::prepare(&exact).unwrap_err();
    assert!(matches!(
        second.0.accounting_failure(),
        Some(WorkingMemoryError::Domain(
            eredu_core::MemoryDomainError::BudgetExceeded {
                limit_bytes,
                existing_bytes,
                ..
            }
        ))
    ));
    drop((refused, second, cache));
    assert_eq!(exact.fixture_host_charge().unwrap(), 0);
}

#[test]
fn cache_capsule_aliases_share_one_domain_and_never_promote_ordinary_owners() {
    let Some(bytes) = qualified_bytes() else {
        return;
    };
    let pool = crate::memory_fixture::ledger(
        (bytes * 2)
            .checked_add(reservation_controls().checked_mul(2).unwrap())
            .unwrap(),
        0,
    )
    .unwrap();
    let foreign = crate::memory_fixture::ledger(
        (bytes * 2)
            .checked_add(reservation_controls().checked_mul(2).unwrap())
            .unwrap(),
        0,
    )
    .unwrap();
    let ordinary = CacheHandle::ordinary();
    assert!(matches!(
        ordinary.validate_pool(&pool),
        Err(WorkingMemoryError::UnknownBound)
    ));
    let unquoted = pool.acquire_unquoted().unwrap();
    let refused = CacheHandle::prepare(&pool).unwrap_err();
    assert!(matches!(
        refused.0.accounting_failure(),
        Some(WorkingMemoryError::UnknownBound)
    ));
    assert_eq!(pool.fixture_host_charge().unwrap(), 0);
    drop((refused, unquoted));
    let first = CacheHandle::prepare(&pool).unwrap();
    let alias = first.clone();
    assert!(first.same(&alias));
    assert_eq!(pool.fixture_host_charge().unwrap(), bytes);
    let second = CacheHandle::prepare(&pool).unwrap();
    assert!(!first.same(&second));
    assert_eq!(pool.fixture_host_charge().unwrap(), bytes * 2);
    let a = reservation(&pool);
    let b = reservation(&pool);
    first.validate_reservation(&a).unwrap();
    alias.validate_reservation(&b).unwrap();
    let other = reservation(&foreign);
    assert!(matches!(
        first.validate_reservation(&other),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    assert!(matches!(
        ordinary.validate_reservation(&a),
        Err(WorkingMemoryError::UnknownBound)
    ));
    drop((a, b, other, first, second));
    assert_eq!(pool.fixture_host_charge().unwrap(), bytes);
    alias.validate_pool(&pool).unwrap();
    assert!(matches!(
        alias.validate_pool(&foreign),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    // Both final aliases race through the actual into_inner retirement. The
    // shared account must refund exactly once, never leak or underflow.
    let twin = alias.clone();
    let barrier = std::sync::Barrier::new(2);
    std::thread::scope(|scope| {
        let barrier = &barrier;
        scope.spawn(move || {
            barrier.wait();
            drop(alias);
        });
        scope.spawn(move || {
            barrier.wait();
            drop(twin);
        });
    });
    assert_eq!(pool.fixture_host_charge().unwrap(), 0);
    assert!(matches!(
        ordinary.validate_pool(&pool),
        Err(WorkingMemoryError::UnknownBound)
    ));
}

#[cfg(test)]
#[allow(unused_imports)]
use crate::memory_fixture::LedgerFixture;
