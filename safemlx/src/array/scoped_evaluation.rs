//! Original synchronous evaluation through the existing bound native Scope.
use super::{Array, EvaluatedArray};
use crate::{
    error::{Exception, ScopedEvaluationFailure},
    utils::runtime_lock,
    RetainedPrefillFailure,
};
use std::{
    mem::{size_of, size_of_val},
    ptr,
};

impl Array {
    /// Borrow completed host-readable storage using an exact retained original
    /// observer, even after its role is sealed. Unlike evaluation, this never
    /// submits or waits for work; unscheduled or foreign pending arrays refuse.
    /// The returned view borrows this array. Its slice access uses the existing
    /// cold data path; scalar `try_item` retains its ordinary guarded behavior.
    pub fn completed_in_original_scope(
        &self,
        observer: &crate::OriginalScopeObserver,
    ) -> crate::error::Result<EvaluatedArray<'_>> {
        observer.validate_completed_array(self)?;
        Ok(EvaluatedArray {
            storage: super::EvaluatedArrayStorage::Borrowed(self),
        })
    }
}

struct Transport {
    failure: safemlx_sys::mlx_prefill_failure,
    status: u32,
}
impl Transport {
    fn new() -> Self {
        Self {
            failure: safemlx_sys::mlx_prefill_failure {
                ctx: ptr::null_mut(),
            },
            status: 0,
        }
    }
    fn finish(self) -> crate::error::Result<()> {
        // SAFETY: the C helper returns at most one owned alias of the existing
        // current-scope carrier. No public raw ownership or source creation.
        let owner = unsafe { RetainedPrefillFailure::from_owned_raw(self.failure) };
        if self.status == 0 {
            return Ok(());
        }
        Err(Exception::from_scoped_evaluation(
            crate::error::ScopedEvaluationCause::from_status(self.status),
            owner,
        ))
    }
}
pub(super) fn evaluate(value: &Array) -> Option<crate::error::Result<()>> {
    // SAFETY: this reads this thread's existing Scope flag only. It precedes
    // ordinary error setup, owner reclamation and arbitrary housekeeping.
    if unsafe { safemlx_sys::mlx_submission_runtime_preparation_allowed() } {
        return None;
    }
    let mut transport = Transport::new();
    let Some(_guard) = runtime_lock::try_enter_for_recovery() else {
        // SAFETY: TLS-only retention of the already-bound carrier. Busy is not
        // completion; the resulting Exception keeps its actual custody alive.
        unsafe { safemlx_sys::mlx_array_eval_scoped_failure(&mut transport.failure) };
        transport.status = 10;
        return Some(transport.finish());
    };
    // SAFETY: the borrowed array and no-hooks runtime loan stay live. The
    // native helper validates current original controls before native work.
    transport.status =
        unsafe { safemlx_sys::mlx_array_eval_scoped(value.as_ptr(), &mut transport.failure) };
    Some(transport.finish())
}
// The caller already completed the same contiguous source. This path never
// turns a readiness failure into another Eval or an ordinary host conversion.
pub(super) fn deep_copy(source: &Array) -> Option<crate::error::Result<EvaluatedArray<'static>>> {
    // SAFETY: read-only current-thread original-control flag.
    if unsafe { safemlx_sys::mlx_submission_runtime_preparation_allowed() } {
        return None;
    }
    let mut transport = Transport::new();
    let Some(_guard) = runtime_lock::try_enter_for_recovery() else {
        // SAFETY: retains the already-bound carrier; no new error owner.
        unsafe { safemlx_sys::mlx_array_eval_scoped_failure(&mut transport.failure) };
        transport.status = 10;
        return Some(transport.finish().map(|_| unreachable!()));
    };
    let mut output = safemlx_sys::mlx_array {
        ctx: ptr::null_mut(),
        prepared_owner: ptr::null_mut(),
    };
    // SAFETY: source and the runtime loan stay live. Native verifies original
    // controls/readiness before using the same typed data constructor as ordinary.
    transport.status = unsafe {
        safemlx_sys::mlx_array_deep_copy_scoped(
            &mut output,
            source.as_ptr(),
            &mut transport.failure,
        )
    };
    let output = if output.ctx.is_null() {
        if transport.status == 0 {
            transport.status = 1;
        }
        None
    } else {
        // SAFETY: the closed C producer transferred this one owned handle. Adopt
        // it before propagating status so any published failure prefix retires.
        Some(unsafe { Array::from_ptr(output) })
    };
    Some(transport.finish().map(|()| EvaluatedArray {
        storage: super::EvaluatedArrayStorage::Owned(output.expect("successful original copy")),
    }))
}

/// Named fixed controls for the existing typed eager deep-copy constructor.
/// Native backing, Graph seed/handle storage and prior completion are separate.
/// Unsupported rank metadata returns None before construction.
pub fn original_scoped_deep_copy_control_bytes(maximum_rank: usize) -> Option<usize> {
    // SAFETY: pure sizeof/rank query, without TLS or native construction.
    let native = unsafe { safemlx_sys::mlx_array_deep_copy_scoped_control_bytes(maximum_rank) };
    if native == 0 {
        return None;
    }
    [
        size_of::<Transport>(),
        size_of::<ScopedEvaluationFailure>(),
        size_of::<Exception>(),
        size_of::<Option<RetainedPrefillFailure>>(),
        size_of::<safemlx_sys::mlx_array>(),
        size_of::<Option<Array>>(),
        // Same original Guarded<Array> transport used for the retained C
        // descriptor copies surrounding this typed initializer. Its native
        // array/set frames are included in the shared constructor query.
        size_of::<crate::utils::guard::MaybeUninitArray>(),
        size_of::<Result<Array, Exception>>(),
        size_of::<crate::OriginalScopeObserver>(),
        size_of::<EvaluatedArray<'static>>(),
        size_of::<Result<EvaluatedArray<'static>, Exception>>(),
        size_of::<Option<Result<EvaluatedArray<'static>, Exception>>>(),
        size_of::<Option<runtime_lock::RuntimeLockGuard>>(),
        size_of::<&Array>(),
    ]
    .into_iter()
    .try_fold(native, usize::checked_add)
}

/// Exact fixed controls for this synchronous route. Native root backing uses
/// the same Graph arena; this is not a complete graph/workspace fit claim.
pub fn original_scoped_evaluation_control_bytes() -> Option<usize> {
    // SAFETY: concrete sizeof facts only; no TLS, device, allocation or callback.
    let native = unsafe { safemlx_sys::mlx_array_eval_scoped_control_bytes() };
    let controls = [
        size_of::<Transport>(),
        size_of::<ScopedEvaluationFailure>(),
        size_of::<Exception>(),
        size_of::<Option<RetainedPrefillFailure>>(),
        size_of::<Option<crate::PrefillNativeError>>(),
        size_of::<Result<(), Exception>>(),
        size_of::<Option<Result<(), Exception>>>(),
        size_of::<EvaluatedArray<'static>>(),
        size_of::<Result<EvaluatedArray<'static>, Exception>>(),
        size_of::<runtime_lock::RuntimeLockGuard>(),
        size_of::<Option<runtime_lock::RuntimeLockGuard>>(),
        size_of::<&Array>(),
        size_of::<u32>(),
    ];
    controls.into_iter().try_fold(
        native.checked_add(size_of_val(&controls))?,
        usize::checked_add,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Device, DeviceType, PrefillRootsRuntime, PreparedPrefillFailure,
        PreparedSubmissionGraphQuota, PreparedSubmissionRecordQuota, PreparedSubmissionScopeOwner,
        ScopedSubmissionProgress, Stream, SubmissionScope,
    };
    use std::{
        cell::Cell,
        sync::{
            atomic::{AtomicUsize, Ordering},
            Arc,
        },
        time::{Duration, Instant},
    };
    thread_local! { static HOOKS: Cell<usize> = const { Cell::new(0) }; }
    fn hook() {
        HOOKS.with(|value| value.set(value.get() + 1));
    }
    struct Hook;
    impl Drop for Hook {
        fn drop(&mut self) {
            crate::unregister_thread_runtime_housekeeping(hook);
        }
    }
    #[derive(Debug)]
    struct Custody(Arc<AtomicUsize>);
    impl Drop for Custody {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }
    fn settle(scope: &mut SubmissionScope) {
        scope.seal();
        let end = Instant::now() + Duration::from_secs(5);
        while !scope.status().is_settled() {
            assert_eq!(
                scope.progress_scoped().0,
                ScopedSubmissionProgress::Observed
            );
            assert!(Instant::now() < end);
            std::thread::yield_now();
        }
    }
    fn retire() {
        let end = Instant::now() + Duration::from_secs(5);
        loop {
            match crate::try_retire_completed_submissions().unwrap() {
                crate::SubmissionRetirement::CompleteSnapshot => break,
                crate::SubmissionRetirement::Busy => {
                    assert!(Instant::now() < end);
                    std::thread::yield_now();
                }
            }
        }
        crate::reclaim_allocation_owners();
    }
    #[test]
    fn original_evaluated_and_owned_evaluated_skip_housekeeping_and_preserve_nonzero_values() {
        let guard = runtime_lock::enter();
        let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
        let _runtime = PrefillRootsRuntime::prepare_for_stream(&stream, &stream).unwrap();
        let failure = PreparedPrefillFailure::try_new(())
            .unwrap()
            .try_allocate()
            .unwrap();
        let graph = PreparedSubmissionGraphQuota::try_new(1 << 20, ())
            .unwrap()
            .try_allocate()
            .unwrap();
        let records = PreparedSubmissionRecordQuota::try_new(1 << 20, ())
            .unwrap()
            .try_allocate()
            .unwrap();
        let mut scope = SubmissionScope::try_begin_retaining(
            PreparedSubmissionScopeOwner::try_new(())
                .unwrap()
                .with_graph_quota(graph.clone())
                .with_record_quota(records.clone()),
        )
        .unwrap();
        scope.enable_scoped_observation().unwrap();
        scope.require_original_native_controls().unwrap();
        failure.bind_original_scope(&scope).unwrap();
        scope.enable_original_native_controls().unwrap();
        let source = Array::try_from_slice(&[2.0f32, -3.0, 7.0], &[3]).unwrap();
        let output = source.add(&source, &stream).unwrap();
        crate::register_thread_runtime_housekeeping(hook);
        let hook_owner = Hook;
        HOOKS.with(|value| value.set(0));
        let handler = crate::error::mlx_error_handler_state_for_test();
        let evaluated = output.evaluated().unwrap();
        assert_eq!(HOOKS.with(Cell::get), 0);
        assert_eq!(crate::error::mlx_error_handler_state_for_test(), handler);
        assert_eq!(evaluated.try_as_slice::<f32>().unwrap(), &[4.0, -6.0, 14.0]);
        drop(evaluated);
        let evaluated = output.into_evaluated().unwrap();
        assert_eq!(HOOKS.with(Cell::get), 0);
        assert_eq!(evaluated.try_as_slice::<f32>().unwrap(), &[4.0, -6.0, 14.0]);
        drop((evaluated, source, hook_owner));
        settle(&mut scope);
        drop((scope, records, graph, failure, guard));
        retire();
    }
    #[test]
    fn original_owned_evaluation_error_retains_actual_native_source_after_array_and_scope_retirement(
    ) {
        let count = Arc::new(AtomicUsize::new(0));
        let error = {
            let guard = runtime_lock::enter();
            let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
            let _runtime = PrefillRootsRuntime::prepare_for_stream(&stream, &stream).unwrap();
            let failure = PreparedPrefillFailure::try_new(Custody(count.clone()))
                .unwrap()
                .try_allocate()
                .unwrap();
            let graph = PreparedSubmissionGraphQuota::try_new(64 << 10, ())
                .unwrap()
                .try_allocate()
                .unwrap();
            let records = PreparedSubmissionRecordQuota::try_new(1 << 20, ())
                .unwrap()
                .try_allocate()
                .unwrap();
            let mut scope = SubmissionScope::try_begin_retaining(
                PreparedSubmissionScopeOwner::try_new(())
                    .unwrap()
                    .with_graph_quota(graph.clone())
                    .with_record_quota(records.clone()),
            )
            .unwrap();
            scope.enable_scoped_observation().unwrap();
            scope.require_original_native_controls().unwrap();
            failure.bind_original_scope(&scope).unwrap();
            scope.enable_original_native_controls().unwrap();
            let source = Array::try_from_slice(&[3.0f32, 9.0], &[2]).unwrap();
            let output = source.add(&source, &stream).unwrap();
            let mut filler = Vec::new(); // Fixture instrumentation, not a production bound.
            while let Ok(value) = source.add(&source, &stream) {
                filler.push(value);
            }
            while let Ok(value) = Array::try_from_scalar(1u32) {
                filler.push(value);
            }
            crate::register_thread_runtime_housekeeping(hook);
            let hook_owner = Hook;
            HOOKS.with(|value| value.set(0));
            let handler = crate::error::mlx_error_handler_state_for_test();
            let error = output.into_evaluated().unwrap_err();
            assert_eq!(
                error.scoped_evaluation_cause(),
                Some(crate::error::ScopedEvaluationCause::Failed)
            );
            assert_eq!(HOOKS.with(Cell::get), 0);
            assert_eq!(crate::error::mlx_error_handler_state_for_test(), handler);
            assert!(error.what().contains("retained native failure"));
            drop((hook_owner, filler, source));
            settle(&mut scope);
            drop((scope, graph, records, failure, guard));
            retire();
            assert_eq!(count.load(Ordering::SeqCst), 0);
            error
        };
        use std::error::Error;
        let source = error
            .source()
            .unwrap()
            .source()
            .unwrap()
            .downcast_ref::<crate::PrefillNativeError>()
            .unwrap();
        assert_eq!(source.kind(), crate::PrefillNativeFailureKind::Exception);
        assert!(!source.message_bytes().unwrap().is_empty());
        let escaped = source.clone();
        let formatted = error.to_string();
        assert!(formatted.contains("graph"));
        drop(error);
        retire();
        assert_eq!(count.load(Ordering::SeqCst), 0);
        assert!(!escaped.source_type_bytes().unwrap().is_empty());
        drop(escaped);
        retire();
        assert_eq!(count.load(Ordering::SeqCst), 1);
    }
    #[test]
    fn original_missing_authority_is_inline_and_preserves_legacy_error_constructors() {
        let guard = runtime_lock::enter();
        let value = Array::try_from_slice(&[1i32], &[1]).unwrap();
        let mut scope = SubmissionScope::try_begin().unwrap();
        scope.enable_scoped_observation().unwrap();
        scope.require_original_native_controls().unwrap();
        crate::register_thread_runtime_housekeeping(hook);
        let hook_owner = Hook;
        HOOKS.with(|calls| calls.set(0));
        let error = value.evaluated().unwrap_err();
        assert_eq!(
            error.scoped_evaluation_cause(),
            Some(crate::error::ScopedEvaluationCause::Domain)
        );
        assert_eq!(HOOKS.with(Cell::get), 0);
        assert!(std::error::Error::source(&error)
            .unwrap()
            .source()
            .is_none());
        assert!(error.what.is_empty());
        assert_eq!(error.what.capacity(), 0);
        drop((error, hook_owner, scope, value, guard));
        let legacy = Exception::custom("ordinary source");
        assert_eq!(legacy.scoped_evaluation_cause(), None);
        assert_eq!(String::from(legacy), "ordinary source");
    }
}
