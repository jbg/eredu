use super::*;
use crate::{
    PreparedSubmissionGraphQuota, PreparedSubmissionScopeOwner, SubmissionGraphQuota,
    SubmissionScope,
};
use std::{
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

struct Original(Arc<AtomicUsize>);
impl Drop for Original {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}
fn quota(capacity: usize, original: Original) -> SubmissionGraphQuota {
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut pending = PreparedSubmissionGraphQuota::try_new(capacity, original).unwrap();
    loop {
        match pending.try_allocate() {
            Ok(arena) => return arena,
            Err(error) => {
                assert_eq!(error.cause(), crate::SubmissionGraphQuotaCause::RuntimeBusy);
                pending = error.into_parts().1;
                assert!(Instant::now() < deadline);
                std::thread::yield_now();
            }
        }
    }
}
fn begin(arena: &SubmissionGraphQuota) -> SubmissionScope {
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut pending = PreparedSubmissionScopeOwner::try_new(())
        .unwrap()
        .with_graph_quota(arena.clone());
    loop {
        match SubmissionScope::try_begin_retaining(pending) {
            Ok(scope) => return scope,
            Err(error) => {
                assert_eq!(error.cause(), crate::SubmissionScopeOwnerCause::RuntimeBusy);
                pending = error.into_parts().1;
                assert!(Instant::now() < deadline);
                std::thread::yield_now();
            }
        }
    }
}
fn exhausted(error: &Exception) {
    assert_eq!(
        std::error::Error::source(error)
            .unwrap()
            .downcast_ref::<crate::error::GraphMetadataFailure>(),
        Some(&crate::error::GraphMetadataFailure::Exhausted)
    );
}

#[test]
fn graph_refusal_retires_unpublished_c_vector_handle_and_all_partial_buffer_growth() {
    let _stream = crate::test_stream();
    let source = Array::try_from_slice(&[2u32, 5, 7], &[3]).unwrap();
    let handles = vec![source.as_ptr(); 1024];
    let count = Arc::new(AtomicUsize::new(0));
    let arena = quota(512, Original(count.clone()));
    let mut scope = begin(&arena);
    let result = unsafe { safemlx_sys::mlx_vector_array_new_data(handles.as_ptr(), handles.len()) };
    assert!(result.ctx.is_null());
    let error: Exception = crate::error::get_and_clear_last_mlx_error().unwrap().into();
    exhausted(&error);
    assert_eq!(arena.occupied_bytes(), 0);
    scope.seal();
    drop((scope, arena));
    crate::reclaim_allocation_owners();
    assert_eq!(count.load(Ordering::SeqCst), 1);
}

#[test]
fn graph_refusal_callback_append_returns_failure_without_partial_success_or_lost_source() {
    let _stream = crate::test_stream();
    let source = Array::try_from_scalar(11u32).unwrap();
    let input = VectorArray::try_from_iter(std::iter::empty::<&Array>()).unwrap();
    let count = Arc::new(AtomicUsize::new(0));
    let arena = quota(512, Original(count.clone()));
    let mut scope = begin(&arena);
    fn invoke<F: FnMut(&[Array]) -> Vec<Array>>(f: F, input: safemlx_sys::mlx_vector_array) {
        let payload = Box::into_raw(Box::new(f)).cast();
        let mut result = safemlx_sys::mlx_vector_array {
            ctx: std::ptr::null_mut(),
        };
        let status = trampoline::<F>(&mut result, input, payload);
        assert_eq!(status, FAILURE);
        assert!(result.ctx.is_null());
        exhausted(&crate::error::get_and_clear_closure_error().unwrap());
        closure_dtor::<F>(payload);
    }
    invoke(|_| vec![source.clone(); 1024], input.as_ptr());
    assert_eq!(arena.occupied_bytes(), 0);
    scope.seal();
    drop((scope, arena, input));
    crate::reclaim_allocation_owners();
    assert_eq!(count.load(Ordering::SeqCst), 1);
}

#[test]
fn graph_refusal_actual_c_closure_conversion_preserves_source_and_retires_all_local_roots() {
    let _stream = crate::test_stream();
    let source = Array::try_from_scalar(19u32).unwrap();
    let input = VectorArray::try_from_iter(std::iter::empty::<&Array>()).unwrap();
    let mut refused = 0;
    let mut accepted = 0;
    for capacity in (128..=8192).step_by(64) {
        let count = Arc::new(AtomicUsize::new(0));
        let arena = quota(capacity, Original(count.clone()));
        let mut scope = begin(&arena);
        let closure = new_mlx_fallible_closure(|_: &[Array]| Ok(vec![source.clone(); 64]));
        let result = Vec::<Array>::try_from_op(|output| unsafe {
            safemlx_sys::mlx_closure_apply(output, closure, input.as_ptr())
        })
        .map_err(|error| crate::error::get_and_clear_closure_error().unwrap_or(error));
        match result {
            Ok(values) => {
                assert_eq!(values.len(), 64);
                assert!(values.iter().all(|value| value.shape() == source.shape()));
                accepted += 1;
            }
            Err(error) => {
                exhausted(&error);
                refused += 1;
            }
        }
        unsafe { safemlx_sys::mlx_closure_free(closure) };
        assert_eq!(arena.occupied_bytes(), 0);
        scope.seal();
        drop((scope, arena));
        crate::reclaim_allocation_owners();
        assert_eq!(count.load(Ordering::SeqCst), 1);
    }
    assert!(refused > 0);
    assert!(accepted > 0);

    // The hand-written unary bridge shares the same temporary ownership rule.
    // A nonzero callback return must destroy each local handle exactly once.
    unsafe extern "C" fn fail_unary(
        _: *mut safemlx_sys::mlx_array,
        _: safemlx_sys::mlx_array,
    ) -> std::os::raw::c_int {
        FAILURE
    }
    let unary_input = VectorArray::try_from_iter([&source].into_iter()).unwrap();
    let count = Arc::new(AtomicUsize::new(0));
    let arena = quota(64 * 1024, Original(count.clone()));
    let mut scope = begin(&arena);
    let closure = unsafe { safemlx_sys::mlx_closure_new_unary(Some(fail_unary)) };
    let result = Vec::<Array>::try_from_op(|output| unsafe {
        safemlx_sys::mlx_closure_apply(output, closure, unary_input.as_ptr())
    });
    assert!(result.unwrap_err().to_string().contains("non-zero value"));
    unsafe { safemlx_sys::mlx_closure_free(closure) };
    assert_eq!(arena.occupied_bytes(), 0);
    scope.seal();
    drop((scope, arena));
    crate::reclaim_allocation_owners();
    assert_eq!(count.load(Ordering::SeqCst), 1);
}
