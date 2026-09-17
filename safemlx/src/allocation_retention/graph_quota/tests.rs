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
    mut prepared: PreparedSubmissionGraphQuota<T>,
) -> SubmissionGraphQuota {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        match prepared.try_allocate() {
            Ok(value) => return value,
            Err(error) => {
                assert_eq!(error.cause(), SubmissionGraphQuotaCause::RuntimeBusy);
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
fn graph_metadata_invalid_capacity_preserves_original_and_unattached_unwind_drops_it_once() {
    let count = Arc::new(AtomicUsize::new(0));
    let error = PreparedSubmissionGraphQuota::try_new(1, Owner(count.clone())).unwrap_err();
    assert_eq!(error.cause(), SubmissionGraphQuotaCause::InvalidCapacity);
    assert!(Arc::ptr_eq(&error.owner().0, &count));
    let original = error.into_parts().1;
    let result = catch_unwind(AssertUnwindSafe(|| {
        let _prepared = PreparedSubmissionGraphQuota::try_new(64 * 1024, original).unwrap();
        std::panic::panic_any(71u32);
    }));
    assert_eq!(*result.unwrap_err().downcast::<u32>().unwrap(), 71);
    assert_eq!(count.load(Ordering::SeqCst), 1);
}

#[test]
fn graph_metadata_busy_returns_same_node_then_native_scope_outlives_all_rust_arena_aliases() {
    let count = Arc::new(AtomicUsize::new(0));
    let prepared = PreparedSubmissionGraphQuota::try_new(64 * 1024, Owner(count.clone())).unwrap();
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
    assert_eq!(error.cause(), SubmissionGraphQuotaCause::RuntimeBusy);
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
            .with_graph_quota(quota.clone()),
    );
    drop((quota, alias));
    reclaim_allocation_owners();
    assert_eq!(count.load(Ordering::SeqCst), 0);
    drop(scope);
    reclaim_allocation_owners();
    assert_eq!(count.load(Ordering::SeqCst), 1);
}

#[test]
fn graph_metadata_constructor_exhaustion_preserves_typed_native_cause() {
    let stream = crate::test_stream();
    let source = crate::Array::from_slice(&[2.0f32, 3.0, 5.0], &[3]);
    let count = Arc::new(AtomicUsize::new(0));
    let quota =
        allocate(PreparedSubmissionGraphQuota::try_new(4096, Owner(count.clone())).unwrap());
    let mut scope = begin(
        PreparedSubmissionScopeOwner::try_new(())
            .unwrap()
            .with_graph_quota(quota.clone()),
    );
    let mut graph = source.clone();
    let error = loop {
        match graph.add(&source, stream) {
            Ok(next) => graph = next,
            Err(error) => break error,
        }
    };
    assert_eq!(
        std::error::Error::source(&error)
            .unwrap()
            .downcast_ref::<crate::error::GraphMetadataFailure>(),
        Some(&crate::error::GraphMetadataFailure::Exhausted)
    );
    scope.seal();
    drop(graph);
    assert_eq!(quota.occupied_bytes(), 0);
    drop((scope, quota));
    reclaim_allocation_owners();
    assert_eq!(count.load(Ordering::SeqCst), 1);
}

#[test]
fn graph_metadata_empty_slice_and_scalar_errors_never_publish_null_arrays() {
    let _stream = crate::test_stream();
    assert!(crate::Array::try_from_slice(&[1u32], &[2]).is_err());
    assert!(crate::Array::try_from_slice(&[] as &[u32], &[-1]).is_err());
    let count = Arc::new(AtomicUsize::new(0));
    let quota = allocate(PreparedSubmissionGraphQuota::try_new(128, Owner(count.clone())).unwrap());
    let mut scope = begin(
        PreparedSubmissionScopeOwner::try_new(())
            .unwrap()
            .with_graph_quota(quota.clone()),
    );
    for scalar in [false, true] {
        let error = if scalar {
            crate::Array::try_from_int(73)
        } else {
            crate::Array::try_from_slice(&[2u32, 5, 7], &[3])
        }
        .unwrap_err();
        assert_eq!(
            std::error::Error::source(&error)
                .unwrap()
                .downcast_ref::<crate::error::GraphMetadataFailure>(),
            Some(&crate::error::GraphMetadataFailure::Exhausted)
        );
        assert_eq!(quota.occupied_bytes(), 0);
    }
    scope.seal();
    drop((scope, quota));
    reclaim_allocation_owners();
    assert_eq!(count.load(Ordering::SeqCst), 1);
}
