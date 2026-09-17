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
            let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
            let error = PreparedMaterializationSourceStream::prepare(&pool).unwrap_err();
            assert!(matches!(
                error.0,
                Failure::Layout(StreamRegistrationCause::UnknownLayout)
            ));
            assert_eq!(pool.used_bytes().unwrap(), 0);
            return;
        }
        Err(MaterializationSourceStreamError(Failure::Accounting(
            WorkingMemoryError::UnknownBound,
        ))) => {
            let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
            let error = PreparedMaterializationSourceStream::prepare(&pool).unwrap_err();
            let Failure::Initialization(error) = error.0 else {
                panic!("wrong unknown path")
            };
            assert!(matches!(
                error.accounting_failure(),
                Some(WorkingMemoryError::UnknownBound)
            ));
            assert!(error.rejected_plan().is_some());
            assert_eq!(pool.used_bytes().unwrap(), 0);
            return;
        }
        Err(error) => panic!("{error:?}"),
    };
    let short = WorkingMemoryPool::new(bytes - 1, 0).unwrap();
    let error = PreparedMaterializationSourceStream::prepare(&short).unwrap_err();
    let Failure::Initialization(ref failure) = error.0 else {
        panic!("{error:?}")
    };
    assert!(matches!(failure.accounting_failure(),
        Some(WorkingMemoryError::BudgetExceeded { required_bytes, available_bytes })
            if *required_bytes == bytes && *available_bytes == bytes - 1));
    assert!(failure.rejected_plan().is_some());
    assert!(failure.constructor_failure().is_none());
    drop(error);
    assert_eq!(short.used_bytes().unwrap(), 0);

    let pool = WorkingMemoryPool::new(bytes, 0).unwrap();
    let registered = PreparedMaterializationSourceStream::prepare(&pool).unwrap();
    registered.validate_pool(&pool).unwrap();
    let error = registered.validate_pool(&short).unwrap_err();
    assert!(matches!(
        error.0,
        Failure::Accounting(WorkingMemoryError::IdentityMismatch)
    ));
    assert_eq!(pool.used_bytes().unwrap(), bytes);
    drop(registered);
    safemlx::reclaim_allocation_owners();
    // Actual native registration survives its independent wrapper. The native
    // fresh-process fixture verifies final free precedes source retirement.
    assert_eq!(pool.used_bytes().unwrap(), bytes);
}
