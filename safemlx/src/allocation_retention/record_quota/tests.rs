use super::*;
use crate::{reclaim_allocation_owners, PreparedSubmissionScopeOwner, SubmissionScope};
use std::{
    panic::{catch_unwind, AssertUnwindSafe},
    sync::{
        atomic::{AtomicUsize, Ordering},
        mpsc, Arc,
    },
    time::{Duration, Instant},
};

#[derive(Debug)]
struct Owner(Arc<AtomicUsize>);
impl Drop for Owner {
    fn drop(&mut self) {
        if !std::thread::panicking() {
            assert!(crate::can_reclaim_submission_resources());
        }
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}
fn allocate<T: Send + 'static>(
    mut prepared: PreparedSubmissionRecordQuota<T>,
) -> SubmissionRecordQuota {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        match prepared.try_allocate() {
            Ok(value) => return value,
            Err(error) => {
                assert_eq!(error.cause(), SubmissionRecordQuotaCause::RuntimeBusy);
                prepared = error.into_parts().1;
                assert!(Instant::now() < deadline);
                std::thread::yield_now();
            }
        }
    }
}
fn begin<T: Send + 'static>(mut prepared: PreparedSubmissionScopeOwner<T>) -> SubmissionScope {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        match SubmissionScope::try_begin_retaining(prepared) {
            Ok(value) => return value,
            Err(error) => {
                assert_eq!(error.cause(), crate::SubmissionScopeOwnerCause::RuntimeBusy);
                prepared = error.into_parts().1;
                assert!(Instant::now() < deadline);
                std::thread::yield_now();
            }
        }
    }
}

#[test]
fn tracking_invalid_capacity_preserves_original_and_unattached_unwind_drops_it_once() {
    let count = Arc::new(AtomicUsize::new(0));
    let error = PreparedSubmissionRecordQuota::try_new(1, Owner(count.clone())).unwrap_err();
    assert_eq!(error.cause(), SubmissionRecordQuotaCause::InvalidCapacity);
    assert!(Arc::ptr_eq(&error.owner().0, &count));
    let original = error.into_parts().1;
    let result = catch_unwind(AssertUnwindSafe(|| {
        let _prepared = PreparedSubmissionRecordQuota::try_new(64 * 1024, original).unwrap();
        std::panic::panic_any(71u32);
    }));
    assert_eq!(*result.unwrap_err().downcast::<u32>().unwrap(), 71);
    assert_eq!(count.load(Ordering::SeqCst), 1);
}

#[test]
fn tracking_busy_returns_same_node_then_native_scope_outlives_all_rust_arena_aliases() {
    let count = Arc::new(AtomicUsize::new(0));
    let prepared = PreparedSubmissionRecordQuota::try_new(64 * 1024, Owner(count.clone())).unwrap();
    let identity = std::ptr::from_ref(&**prepared.node.as_ref().unwrap()) as usize;
    let (ready_tx, ready_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let worker = std::thread::spawn(move || {
        let _guard = runtime_lock::enter();
        ready_tx.send(()).unwrap();
        release_rx.recv_timeout(Duration::from_secs(5)).is_ok()
    });
    let entered = ready_rx.recv_timeout(Duration::from_secs(5)).is_ok();
    let result = prepared.try_allocate();
    let released = release_tx.send(()).is_ok();
    let held = worker.join().unwrap();
    assert!(entered && released && held);
    let error = result.unwrap_err();
    assert_eq!(error.cause(), SubmissionRecordQuotaCause::RuntimeBusy);
    assert_eq!(
        std::ptr::from_ref(&**error.owner().node.as_ref().unwrap()) as usize,
        identity
    );
    let quota = allocate(error.into_parts().1);
    let alias = quota.clone();
    assert!(quota.same_arena(&alias));
    let scope = begin(
        PreparedSubmissionScopeOwner::try_new(())
            .unwrap()
            .with_record_quota(quota.clone()),
    );
    drop((quota, alias));
    reclaim_allocation_owners();
    assert_eq!(count.load(Ordering::SeqCst), 0);
    drop(scope);
    reclaim_allocation_owners();
    assert_eq!(count.load(Ordering::SeqCst), 1);
}

#[test]
fn tracking_nonzero_graph_exhaustion_keeps_fixed_typed_native_cause() {
    let stream = crate::test_stream();
    let source = crate::Array::from_slice(&[2.0f32, 3.0, 5.0], &[3]);
    let mut graph = source.clone();
    for _ in 0..128 {
        graph = graph.add(&source, stream).unwrap();
    }
    let count = Arc::new(AtomicUsize::new(0));
    let quota =
        allocate(PreparedSubmissionRecordQuota::try_new(4096, Owner(count.clone())).unwrap());
    let mut scope = begin(
        PreparedSubmissionScopeOwner::try_new(())
            .unwrap()
            .with_record_quota(quota.clone()),
    );
    let error = graph.evaluated().err().unwrap();
    let source = std::error::Error::source(&error).unwrap();
    assert_eq!(
        source.downcast_ref::<crate::error::SubmissionTrackingFailure>(),
        Some(&crate::error::SubmissionTrackingFailure::Exhausted)
    );
    scope.seal();
    // The error is no completion certificate. Ordinary retirement uses the
    // actual native evidence; this failure occurred during graph traversal.
    let deadline = Instant::now() + Duration::from_secs(5);
    while quota.occupied_bytes() != 0 {
        scope.progress();
        crate::try_retire_completed_submissions().unwrap();
        assert!(Instant::now() < deadline);
        std::thread::yield_now();
    }
    drop(scope);
    drop((graph, quota));
    reclaim_allocation_owners();
    assert_eq!(count.load(Ordering::SeqCst), 1);
}
