use super::*;

#[test]
fn source_stream_registration_compares_before_birth_and_keeps_process_account() {
    let required = PreparedMaterializationSourceStream::required_bytes();
    if std::env::var_os("EREDU_REQUIRE_STREAM_REGISTRATION_QUALIFICATION").is_some() {
        assert!(required.is_ok(), "{required:?}");
    }
    let bytes = match required {
        Ok(value) => value,
        Err(MaterializationSourceStreamError(Failure::Layout(
            StreamRegistrationCause::UnknownLayout,
        ))) => {
            let pool = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
            let error = PreparedMaterializationSourceStream::prepare(&pool).unwrap_err();
            assert!(matches!(
                error.0,
                Failure::Layout(StreamRegistrationCause::UnknownLayout)
            ));
            assert_eq!(pool.fixture_host_charge().unwrap(), 0);
            return;
        }
        Err(MaterializationSourceStreamError(Failure::Accounting(
            WorkingMemoryError::UnknownBound,
        ))) => {
            let pool = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
            let error = PreparedMaterializationSourceStream::prepare(&pool).unwrap_err();
            let Failure::Initialization(error) = error.0 else {
                panic!("wrong unknown path")
            };
            assert!(matches!(
                error.accounting_failure(),
                Some(WorkingMemoryError::UnknownBound)
            ));
            assert!(error.rejected_plan().is_some());
            assert_eq!(pool.fixture_host_charge().unwrap(), 0);
            return;
        }
        Err(error) => panic!("{error:?}"),
    };
    let short = crate::memory_fixture::ledger(bytes - 1, 0).unwrap();
    let error = PreparedMaterializationSourceStream::prepare(&short).unwrap_err();
    let Failure::Initialization(ref failure) = error.0 else {
        panic!("{error:?}")
    };
    assert!(matches!(failure.accounting_failure(),
        Some(WorkingMemoryError::Domain(eredu_core::MemoryDomainError::BudgetExceeded { requested_bytes: required_bytes, limit_bytes, existing_bytes, .. }))
            if *required_bytes == bytes && limit_bytes.checked_sub(*existing_bytes).unwrap() == bytes - 1));
    assert!(failure.rejected_plan().is_some());
    assert!(failure.constructor_failure().is_none());
    drop(error);
    assert_eq!(short.fixture_host_charge().unwrap(), 0);

    let pool = crate::memory_fixture::ledger(bytes, 0).unwrap();
    let registered = PreparedMaterializationSourceStream::prepare(&pool).unwrap();
    registered.validate_pool(&pool).unwrap();
    let error = registered.validate_pool(&short).unwrap_err();
    assert!(matches!(
        error.0,
        Failure::Accounting(WorkingMemoryError::IdentityMismatch)
    ));
    assert_eq!(pool.fixture_host_charge().unwrap(), bytes);
    drop(registered);
    safemlx::reclaim_allocation_owners();
    // Actual native registration survives its independent wrapper. The native
    // fresh-process fixture verifies final free precedes source retirement.
    assert_eq!(pool.fixture_host_charge().unwrap(), bytes);
}

#[cfg(test)]
#[allow(unused_imports)]
use crate::memory_fixture::LedgerFixture;
