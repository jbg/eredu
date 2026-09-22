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
            let pool = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
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
            assert_eq!(pool.fixture_host_charge().unwrap(), 0);
            return;
        }
        Err(error) => panic!("{error:?}"),
    };
    let first_bytes = MemoryLedger::shared_native_initialization_required_bytes(&Initializer(
        StreamCopyPlan::capture(&stream).unwrap(),
    ))
    .unwrap();
    assert_eq!(bytes, first_bytes * 2);
    let short = crate::memory_fixture::ledger(bytes - 1, 0).unwrap();
    let error = PreparedMaterializationStreams::prepare(&short, &stream, &stream).unwrap_err();
    let PreparedMaterializationStreamError(Failure::Initialization(ref failure)) = error else {
        panic!("{error:?}")
    };
    assert!(
        matches!(failure.accounting_failure(), Some(WorkingMemoryError::Domain(eredu_core::MemoryDomainError::BudgetExceeded { requested_bytes: required_bytes, limit_bytes, existing_bytes, .. })) if *required_bytes == first_bytes && limit_bytes.checked_sub(*existing_bytes).unwrap() == first_bytes - 1)
    );
    assert!(failure.rejected_plan().is_some());
    assert!(failure.constructor_failure().is_none());
    assert_eq!(short.fixture_host_charge().unwrap(), first_bytes); // actual unused first copy is queued
    drop(error);
    safemlx::reclaim_allocation_owners();
    assert_eq!(short.fixture_host_charge().unwrap(), 0);
    let exact = crate::memory_fixture::ledger(bytes, 0).unwrap();
    let pair = PreparedMaterializationStreams::prepare(&exact, &stream, &stream).unwrap();
    let alias = pair.clone();
    assert_eq!(exact.fixture_host_charge().unwrap(), bytes);
    assert!(pair.source.same(&alias.source) && pair.execution.same(&alias.execution));
    assert!(!pair.source.same(&pair.execution));
    assert!(matches!(
        pair.validate_pool(&short),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    drop(pair);
    safemlx::reclaim_allocation_owners();
    assert_eq!(exact.fixture_host_charge().unwrap(), bytes);
    drop(alias);
    assert_eq!(exact.fixture_host_charge().unwrap(), bytes); // queue still owns each raw account
    safemlx::reclaim_allocation_owners();
    assert_eq!(exact.fixture_host_charge().unwrap(), 0);
}

#[cfg(test)]
#[allow(unused_imports)]
use crate::memory_fixture::LedgerFixture;
