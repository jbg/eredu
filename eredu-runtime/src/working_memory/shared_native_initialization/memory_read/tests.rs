use super::*;
use eredu_checkpoint::store::{
    MemoryEncodedReadRouteError, MemoryWeightStore, RetainedCheckpointSource,
};

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
fn requirement(plan: &MemoryEncodedReadPlan<'_>) -> Option<u64> {
    let bytes = WorkingMemoryPool::shared_native_initialization_required_bytes(plan);
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
    let plan = MemoryEncodedReadPlan::new(&source, &keys).unwrap();
    let Some(bytes) = requirement(&plan) else {
        return;
    };
    let short = WorkingMemoryPool::new(bytes - 1, 0).unwrap();
    let rejected = short.initialize_shared_native(plan).unwrap_err();
    assert!(matches!(
        rejected.accounting_failure(),
        Some(WorkingMemoryError::BudgetExceeded { .. })
    ));
    assert!(rejected.rejected_plan().is_some());
    assert!(rejected.constructor_failure().is_none());
    assert_eq!(short.used_bytes().unwrap(), 0);
    drop(rejected);

    let pool = WorkingMemoryPool::new(bytes, 0).unwrap();
    let read = pool
        .initialize_shared_native(MemoryEncodedReadPlan::new(&source, &keys).unwrap())
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
    let plan = || MemoryEncodedReadPlan::new(&source, &keys).unwrap();
    let Some(bytes) = requirement(&plan()) else {
        return;
    };
    let pool = WorkingMemoryPool::new(bytes, 0).unwrap();
    let unquoted = pool.acquire_unquoted().unwrap();
    let error = pool.initialize_shared_native(plan()).unwrap_err();
    assert!(matches!(
        error.accounting_failure(),
        Some(WorkingMemoryError::UnknownBound)
    ));
    assert_eq!(pool.used_bytes().unwrap(), 0);
    drop((error, unquoted));
    let first = pool.initialize_shared_native(plan()).unwrap();
    let second = pool.initialize_shared_native(plan()).unwrap_err();
    assert!(matches!(
        second.accounting_failure(),
        Some(WorkingMemoryError::BudgetExceeded { .. })
    ));
    drop(second);
    assert_eq!(pool.used_bytes().unwrap(), bytes);
    drop(first);
    let second = pool.initialize_shared_native(plan()).unwrap();
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
struct FailingProducer<'a>(MemoryEncodedReadPlan<'a>);
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
    let inner = MemoryEncodedReadPlan::new(&source, &keys).unwrap();
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

#[test]
fn routed_memory_admission_keeps_selected_source_through_rejection_and_success() {
    use eredu_checkpoint::store::RestrictedCheckpointSource;
    use std::{collections::BTreeSet, sync::Arc};
    for short in [true, false] {
        let leaf = Arc::new(source());
        let alive = Arc::downgrade(&leaf);
        let root: RetainedCheckpointSource = Arc::new(
            RestrictedCheckpointSource::including(
                leaf.clone(),
                "first only",
                BTreeSet::from(["first".into()]),
            )
            .unwrap(),
        )
        .into();
        let denied = ["second".into()];
        assert!(matches!(
            MemoryEncodedReadPlan::from_source(&root, &denied),
            Err(MemoryEncodedReadRouteError::UnauthorizedTensor { index: 0 })
        ));
        let keys = ["first".into(), "first".into()];
        let plan = MemoryEncodedReadPlan::from_source(&root, &keys)
            .unwrap()
            .unwrap();
        let Some(bytes) = requirement(&plan) else {
            return;
        };
        drop((root, leaf));
        assert!(alive.upgrade().is_some());
        let pool = WorkingMemoryPool::new(bytes - u64::from(short), 0).unwrap();
        if short {
            let error = pool.initialize_shared_native(plan).unwrap_err();
            assert!(error.rejected_plan().is_some());
            assert!(matches!(
                error.accounting_failure(),
                Some(WorkingMemoryError::BudgetExceeded { .. })
            ));
            assert!(alive.upgrade().is_some());
            assert_eq!(pool.used_bytes().unwrap(), 0);
            drop(error);
            assert!(alive.upgrade().is_none());
        } else {
            let read = pool.initialize_shared_native(plan).unwrap();
            // The finished batch keeps payload owners, not its catalog wrapper.
            assert!(alive.upgrade().is_none());
            drop(keys);
            let mut output = [0; 4];
            read.output().read_into(&mut output).unwrap();
            assert_eq!(output, [13, 29, 13, 29]);
            assert_eq!(pool.used_bytes().unwrap(), bytes);
            drop(read);
            assert_eq!(pool.used_bytes().unwrap(), 0);
        }
    }
}
