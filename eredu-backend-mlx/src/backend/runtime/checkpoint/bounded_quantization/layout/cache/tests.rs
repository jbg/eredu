use super::*;

#[test]
fn cleanup_refuses_before_construction_and_retains_its_account_until_reclaimed() {
    let window = crate::backend::runtime::checkpoint::bounded_quantization::pipeline::controls::required_bytes::<()>().unwrap();
    let required = BoundedAllocatorCache::required_original_bytes(window).unwrap();
    let short = crate::memory_fixture::ledger(required - 1, 0).unwrap();
    let error = BoundedAllocatorCache::prepare_original(&short, 640, window)
        .err()
        .unwrap();
    assert!(
        matches!(error.0.accounting_failure(), Some(WorkingMemoryError::Domain(eredu_core::MemoryDomainError::BudgetExceeded { requested_bytes: required_bytes, limit_bytes, existing_bytes, .. })) if *required_bytes == required && limit_bytes.checked_sub(*existing_bytes).unwrap() == required - 1)
    );
    assert!(error.0.rejected_plan().is_some());
    assert!(error.0.completed_output().is_none());
    assert_eq!(short.fixture_host_charge().unwrap(), 0);
    drop(error);

    for finish in [false, true] {
        let pool = crate::memory_fixture::ledger(required, 0).unwrap();
        let mut cache = BoundedAllocatorCache::prepare_original(&pool, 640, window).unwrap();
        assert_eq!(pool.fixture_host_charge().unwrap(), required);
        cache.begin().unwrap();
        cache.prepare_submission(320, 320).unwrap();
        assert!(cache.prepare_submission(320, 321).is_err());
        assert!(cache.prepare_submission(u64::MAX, 1).is_err());
        assert!(cache.tile_completed(641).is_err());
        if finish {
            cache.finish().unwrap();
        }
        let mut cache = Some(cache);
        loop {
            if safemlx::try_with_submission_retirement(|| {
                drop(cache.take());
                crate::backend::ordinary_retirement::reclaim();
                assert_eq!(pool.fixture_host_charge().unwrap(), required);
            })
            .is_some()
            {
                break;
            }
            std::thread::yield_now();
        }
        // The pipeline wrapper is gone, but the queued cleanup still owns the
        // original account. Only unlocked host reclamation may retire it.
        assert_eq!(pool.fixture_host_charge().unwrap(), required);
        crate::backend::ordinary_retirement::reclaim();
        assert_eq!(pool.fixture_host_charge().unwrap(), 0);
    }
}

#[test]
fn cleanup_admission_preserves_unquoted_exclusion_and_checked_geometry() {
    let pool = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
    let unquoted = pool.acquire_unquoted().unwrap();
    let error = BoundedAllocatorCache::prepare_original(&pool, 640, 128)
        .err()
        .unwrap();
    assert!(matches!(
        error.0.accounting_failure(),
        Some(WorkingMemoryError::UnknownBound)
    ));
    assert!(error.0.rejected_plan().is_some());
    assert_eq!(pool.fixture_host_charge().unwrap(), 0);
    drop(error);
    drop(unquoted);
    let error = BoundedAllocatorCache::prepare_original(&pool, 640, usize::MAX)
        .err()
        .unwrap();
    assert!(matches!(
        error.0.accounting_failure(),
        Some(WorkingMemoryError::Overflow)
    ));
    assert!(error.0.rejected_plan().is_some());
    assert_eq!(pool.fixture_host_charge().unwrap(), 0);
}

#[cfg(test)]
#[allow(unused_imports)]
use crate::memory_fixture::LedgerFixture;
