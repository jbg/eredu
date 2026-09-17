use super::*;
#[test]
fn exact_stream_pair_admission_and_second_refusal_preserve_real_prefix() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let required = PreparedMaterializationStreams::required_bytes(&stream, &stream);
    if std::env::var_os("EREDU_REQUIRE_PREPAID_STREAM_COPY_QUALIFICATION").is_some() {
        assert!(required.is_ok(), "{required:?}");
    }
    let bytes = match required {
        Ok(value) => value,
        Err(PreparedMaterializationStreamError(Failure::Accounting(
            WorkingMemoryError::UnknownBound,
        ))) => {
            let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
            let error =
                PreparedMaterializationStreams::prepare(&pool, &stream, &stream).unwrap_err();
            let PreparedMaterializationStreamError(Failure::Initialization(error)) = error else {
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
    let first_bytes = WorkingMemoryPool::shared_native_initialization_required_bytes(&Initializer(
        StreamCopyPlan::capture(&stream).unwrap(),
    ))
    .unwrap();
    assert_eq!(bytes, first_bytes * 2);
    let short = WorkingMemoryPool::new(bytes - 1, 0).unwrap();
    let error = PreparedMaterializationStreams::prepare(&short, &stream, &stream).unwrap_err();
    let PreparedMaterializationStreamError(Failure::Initialization(ref failure)) = error else {
        panic!("{error:?}")
    };
    assert!(
        matches!(failure.accounting_failure(), Some(WorkingMemoryError::BudgetExceeded { required_bytes, available_bytes }) if *required_bytes == first_bytes && *available_bytes == first_bytes - 1)
    );
    assert!(failure.rejected_plan().is_some());
    assert!(failure.constructor_failure().is_none());
    assert_eq!(short.used_bytes().unwrap(), first_bytes); // actual unused first copy is queued
    drop(error);
    safemlx::reclaim_allocation_owners();
    assert_eq!(short.used_bytes().unwrap(), 0);
    let exact = WorkingMemoryPool::new(bytes, 0).unwrap();
    let pair = PreparedMaterializationStreams::prepare(&exact, &stream, &stream).unwrap();
    let alias = pair.clone();
    assert_eq!(exact.used_bytes().unwrap(), bytes);
    assert!(pair.source.same(&alias.source) && pair.execution.same(&alias.execution));
    assert!(!pair.source.same(&pair.execution));
    assert!(matches!(
        pair.validate_pool(&short),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    drop(pair);
    safemlx::reclaim_allocation_owners();
    assert_eq!(exact.used_bytes().unwrap(), bytes);
    drop(alias);
    assert_eq!(exact.used_bytes().unwrap(), bytes); // queue still owns each raw account
    safemlx::reclaim_allocation_owners();
    assert_eq!(exact.used_bytes().unwrap(), 0);
}
