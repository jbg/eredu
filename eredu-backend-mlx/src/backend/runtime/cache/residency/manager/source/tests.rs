use super::*;
use eredu_nn::workspace::{WorkspaceMechanisms, WorkspaceOperation, WorkspaceOperationBound};
#[derive(Debug)]
struct NoEquations;
impl WorkspaceMechanisms for NoEquations {
    fn operation_bound(
        &self,
        _: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, Error> {
        panic!("source loan performs no equations")
    }
}
#[test]
fn source_loan_refuses_busy_or_pending_manager_and_keeps_callback_error_after_unlock() {
    let manager =
        CacheResidencyManager::new(PagedCacheOptions::new(1, 4096, 4096, 1).unwrap()).unwrap();
    let context = WorkspaceContext::new(NoEquations);
    let selection = CacheBlockSelection::new(0, CacheRepresentation::KeyValue, 0, i64::MAX, 0);
    let guard = manager.inner.state.lock().unwrap();
    let error = manager
        .with_source_loan(selection, &context, |_| -> Result<(), CacheSourceFailure> {
            panic!("busy source cannot reach consumer")
        })
        .unwrap_err();
    assert!(matches!(
        error.cause(),
        CacheSourceFailureCause::Source(CacheSourceError::Busy)
    ));
    drop(guard);
    manager
        .inner
        .host_demotion_worker
        .active_payload
        .store(true, Ordering::Release);
    let error = manager
        .with_source_loan(selection, &context, |_| -> Result<(), CacheSourceFailure> {
            panic!("pending source cannot reach consumer")
        })
        .unwrap_err();
    assert!(matches!(
        error.cause(),
        CacheSourceFailureCause::Source(CacheSourceError::PendingStorage)
    ));
    manager
        .inner
        .host_demotion_worker
        .active_payload
        .store(false, Ordering::Release);
    let generation = manager
        .with_source_loan(selection, &context, |source| Ok(source.generation()))
        .unwrap();
    assert_eq!(generation, 0);
    assert!(
        manager
            .with_source_loan(selection, &context, |_| Err::<(), _>(
                CacheSourceFailure::source(CacheSourceError::Geometry, &context)
            ))
            .is_err()
    );
    assert!(manager.inner.state.try_lock().is_ok());
}
