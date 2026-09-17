use super::*;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
#[derive(Debug)]
struct Payload {
    values: Box<[u64; 3]>,
    drops: Arc<AtomicUsize>,
    _custody: SharedNativeInitializationCustody,
}
impl Drop for Payload {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::SeqCst);
    }
}
#[derive(Debug)]
struct Resource(Option<Arc<Payload>>);
impl Resource {
    fn share(&self) -> Self {
        Self(self.0.as_ref().map(Arc::clone))
    }
}
impl Drop for Resource {
    fn drop(&mut self) {
        // No Weak escapes. Free the shared shell before backing/custody.
        if let Some(inner) = self.0.take() {
            drop(Arc::into_inner(inner));
        }
    }
}
#[derive(Debug, thiserror::Error)]
#[error("controlled refusal after actual shared backing construction")]
struct PrefixFailure(Resource);
#[derive(Debug)]
struct Constructor {
    pool: WorkingMemoryPool,
    calls: Arc<AtomicUsize>,
    drops: Arc<AtomicUsize>,
    fail: bool,
}
impl SharedNativeInitializer for Constructor {
    type Output = Resource;
    type Error = PrefixFailure;
    fn required_storage_bytes(&self) -> Result<usize, WorkingMemoryError> {
        let shell = super::super::qualified_storage::shared_bytes::<Payload>()?;
        usize::try_from(shell)
            .ok()
            .and_then(|n| n.checked_add(size_of::<[u64; 3]>()))
            .and_then(|n| n.checked_add(size_of::<Resource>()))
            .ok_or(WorkingMemoryError::Overflow)
    }
    fn initialize(
        self,
        custody: SharedNativeInitializationCustody,
    ) -> Result<Resource, PrefixFailure> {
        // This public query locks Usage. Completion verifies callback unlocking.
        assert!(self.pool.used_bytes().unwrap() > 0);
        self.calls.fetch_add(1, Ordering::SeqCst);
        let resource = Resource(Some(Arc::new(Payload {
            values: Box::new([11, 37, 91]),
            drops: self.drops,
            _custody: custody,
        })));
        if self.fail {
            Err(PrefixFailure(resource))
        } else {
            Ok(resource)
        }
    }
}
fn constructor(pool: &WorkingMemoryPool, fail: bool) -> Constructor {
    Constructor {
        pool: pool.clone(),
        calls: Arc::new(AtomicUsize::new(0)),
        drops: Arc::new(AtomicUsize::new(0)),
        fail,
    }
}
fn requirement() -> Option<u64> {
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let plan = constructor(&pool, false);
    let result = WorkingMemoryPool::shared_native_initialization_required_bytes(&plan);
    if std::env::var_os("EREDU_REQUIRE_SHARED_INPUT_INITIALIZATION_QUALIFICATION").is_some() {
        assert!(result.is_ok(), "{result:?}");
    }
    match result {
        Ok(bytes) => Some(bytes),
        Err(WorkingMemoryError::UnknownBound) => {
            assert!(matches!(
                pool.initialize_shared_native(plan)
                    .unwrap_err()
                    .accounting_failure(),
                Some(WorkingMemoryError::UnknownBound)
            ));
            None
        }
        Err(error) => panic!("unexpected qualification error: {error}"),
    }
}
#[test]
fn shared_initializer_exact_compare_precedes_constructor_and_preserves_alias_custody() {
    let Some(bytes) = requirement() else {
        return;
    };
    let short = WorkingMemoryPool::new(bytes - 1, 0).unwrap();
    let plan = constructor(&short, false);
    let calls = plan.calls.clone();
    let error = short.initialize_shared_native(plan).unwrap_err();
    assert!(matches!(
        error.accounting_failure(),
        Some(WorkingMemoryError::BudgetExceeded { .. })
    ));
    assert!(error.rejected_plan().is_some());
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert_eq!(short.used_bytes().unwrap(), 0);
    let pool = WorkingMemoryPool::new(bytes, 0).unwrap();
    let plan = constructor(&pool, false);
    let calls = plan.calls.clone();
    let drops = plan.drops.clone();
    let completed = pool.initialize_shared_native(plan).unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(completed.original_bytes(), bytes);
    assert_eq!(*completed.output().0.as_ref().unwrap().values, [11, 37, 91]);
    let alias = completed.output().share();
    assert!(completed.validate_pool(&pool).is_ok());
    assert!(matches!(
        completed.validate_pool(&short),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    drop(completed);
    assert_eq!(pool.used_bytes().unwrap(), bytes);
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    drop(alias);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
#[test]
fn shared_initializer_failed_prefix_retains_exact_debit_until_storage_retirement() {
    let Some(bytes) = requirement() else {
        return;
    };
    let pool = WorkingMemoryPool::new(bytes, 0).unwrap();
    let plan = constructor(&pool, true);
    let calls = plan.calls.clone();
    let drops = plan.drops.clone();
    let error = pool.initialize_shared_native(plan).unwrap_err();
    assert!(error.constructor_failure().is_some());
    assert!(error.accounting_failure().is_none());
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    assert_eq!(pool.used_bytes().unwrap(), bytes);
    drop(error);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
#[test]
fn shared_initializer_keeps_existing_unquoted_exclusion() {
    let Some(bytes) = requirement() else {
        return;
    };
    let pool = WorkingMemoryPool::new(bytes, 0).unwrap();
    let ordinary = pool.acquire_unquoted().unwrap();
    let plan = constructor(&pool, false);
    let calls = plan.calls.clone();
    let error = pool.initialize_shared_native(plan).unwrap_err();
    assert!(matches!(
        error.accounting_failure(),
        Some(WorkingMemoryError::UnknownBound)
    ));
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert_eq!(pool.used_bytes().unwrap(), 0);
    drop(error);
    drop(ordinary);
    let completed = pool
        .initialize_shared_native(constructor(&pool, false))
        .unwrap();
    drop(completed);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
