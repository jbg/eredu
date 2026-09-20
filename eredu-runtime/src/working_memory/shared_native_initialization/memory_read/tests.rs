use super::*;

fn source() -> MemoryWeightStore {
    MemoryWeightStore::from_safetensors([
        (
            "first".into(),
            safetensors::Dtype::U8,
            vec![2],
            vec![13, 29],
        ),
        (
            "second".into(),
            safetensors::Dtype::U8,
            vec![3],
            vec![3, 7, 11],
        ),
    ])
    .unwrap()
}
fn requirement(plan: &MemoryEncodedReadInitializer<'_>) -> Option<u64> {
    let bytes = plan.required_bytes();
    if std::env::var_os("EREDU_REQUIRE_SHARED_INPUT_INITIALIZATION_QUALIFICATION").is_some() {
        assert!(bytes.is_ok(), "{bytes:?}");
    }
    match bytes {
        Ok(bytes) => Some(bytes),
        Err(WorkingMemoryError::UnknownBound) => None,
        Err(error) => panic!("unexpected qualification error: {error}"),
    }
}

#[test]
fn memory_read_exact_admission_survives_source_retirement_and_refuses_short_budget() {
    let source = source();
    let keys = ["second".into(), "first".into(), "second".into()];
    let plan = MemoryEncodedReadInitializer::new(&source, &keys).unwrap();
    let Some(bytes) = requirement(&plan) else {
        return;
    };
    let short = WorkingMemoryPool::new(bytes - 1, 0).unwrap();
    let rejected = plan.prepare(&short).unwrap_err();
    assert!(matches!(
        rejected.accounting_failure(),
        Some(WorkingMemoryError::BudgetExceeded { .. })
    ));
    assert!(rejected.rejected_plan().is_some());
    assert!(rejected.constructor_failure().is_none());
    assert_eq!(short.used_bytes().unwrap(), 0);
    drop(rejected);

    let pool = WorkingMemoryPool::new(bytes, 0).unwrap();
    let read = MemoryEncodedReadInitializer::new(&source, &keys)
        .unwrap()
        .prepare(&pool)
        .unwrap();
    assert_eq!(read.original_bytes(), bytes);
    drop((source, keys));
    assert_eq!(pool.used_bytes().unwrap(), bytes);
    assert!(read.validate_pool(&pool).is_ok());
    assert!(matches!(
        read.validate_pool(&short),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    let mut output = [0; 8];
    read.output().read_into(&mut output).unwrap();
    assert_eq!(output, [3, 7, 11, 13, 29, 3, 7, 11]);
    let mut wrong = [99; 7];
    assert!(read.output().read_into(&mut wrong).is_err());
    assert_eq!(wrong, [99; 7]);
    assert_eq!(pool.used_bytes().unwrap(), bytes);
    drop(read);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn memory_read_accounts_compete_and_keep_unquoted_exclusion() {
    let source = source();
    let keys = ["first".into()];
    let plan = || MemoryEncodedReadInitializer::new(&source, &keys).unwrap();
    let Some(bytes) = requirement(&plan()) else {
        return;
    };
    let pool = WorkingMemoryPool::new(bytes, 0).unwrap();
    let unquoted = pool.acquire_unquoted().unwrap();
    let error = plan().prepare(&pool).unwrap_err();
    assert!(matches!(
        error.accounting_failure(),
        Some(WorkingMemoryError::UnknownBound)
    ));
    assert_eq!(pool.used_bytes().unwrap(), 0);
    drop((error, unquoted));
    let first = plan().prepare(&pool).unwrap();
    let second = plan().prepare(&pool).unwrap_err();
    assert!(matches!(
        second.accounting_failure(),
        Some(WorkingMemoryError::BudgetExceeded { .. })
    ));
    drop(second);
    assert_eq!(pool.used_bytes().unwrap(), bytes);
    drop(first);
    let second = plan().prepare(&pool).unwrap();
    let mut output = [0; 2];
    second.output().read_into(&mut output).unwrap();
    assert_eq!(output, [13, 29]);
    drop(second);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[derive(Debug, thiserror::Error)]
enum FailedRead {
    #[error("read metadata construction failed: {0}")]
    Build(#[from] MemoryEncodedReadBuildError<SharedNativeInitializationCustody>),
    #[error("later producer refused after read metadata construction")]
    Later(PreparedMemoryEncodedRead<SharedNativeInitializationCustody>),
}
struct FailingProducer<'a>(MemoryEncodedReadInitializer<'a>);
impl SharedNativeInitializer for FailingProducer<'_> {
    type Output = PreparedMemoryEncodedRead<SharedNativeInitializationCustody>;
    type Error = FailedRead;
    fn required_storage_bytes(&self) -> Result<usize, WorkingMemoryError> {
        self.0.required_storage_bytes()
    }
    fn initialize(
        self,
        custody: SharedNativeInitializationCustody,
    ) -> Result<Self::Output, Self::Error> {
        let read = self.0.initialize(custody)?;
        Err(FailedRead::Later(read))
    }
}

#[test]
fn failed_producer_retains_actual_memory_read_after_borrowed_plan_retires() {
    let source = source();
    let keys = ["second".into(), "first".into()];
    let inner = MemoryEncodedReadInitializer::new(&source, &keys).unwrap();
    let Some(_) = requirement(&inner) else { return };
    let plan = FailingProducer(inner);
    let bytes = WorkingMemoryPool::shared_native_initialization_required_bytes(&plan).unwrap();
    let pool = WorkingMemoryPool::new(bytes, 0).unwrap();
    let (rejected, failure) = pool
        .initialize_shared_native(plan)
        .unwrap_err()
        .into_parts();
    assert!(rejected.is_none());
    drop(rejected);
    drop((source, keys));
    assert!(failure.accounting_failure().is_none());
    assert_eq!(pool.used_bytes().unwrap(), bytes);
    let FailedRead::Later(read) = failure.constructor_failure().unwrap() else {
        panic!("expected constructed memory read prefix")
    };
    let mut output = [0; 5];
    read.read_into(&mut output).unwrap();
    assert_eq!(output, [3, 7, 11, 13, 29]);
    drop(failure);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
