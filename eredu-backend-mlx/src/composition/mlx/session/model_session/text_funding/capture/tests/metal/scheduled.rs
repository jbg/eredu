mod generated;
use super::super::super::observer::NativeScheduledCapture;
use super::*;
use crate::composition::mlx::session::{
    bounded_capture::{estimate_shape, estimate_tensor_geometry},
    model_session::{
        complete_model_operation, host_layerwise_tests, SessionOperation, SubmissionResources,
    },
};
use eredu_core::checkpoint::TensorDtype;
use eredu_runtime::capture::{FundedCaptureSession, ScheduledCaptureBackend};
use eredu_runtime::{ActivationObserver, ExpertPass};

fn observer_error(error: eredu_runtime::capture::FundedCaptureError<Error>) -> Error {
    Error::Other(Box::new(error))
}

fn memory_cause<'a>(
    mut error: &'a (dyn std::error::Error + 'static),
) -> Option<&'a WorkingMemoryError> {
    loop {
        if let Some(cause) = error.downcast_ref::<WorkingMemoryError>() {
            return Some(cause);
        }
        error = error.source()?;
    }
}

fn geometry(source: &AdmittedCapturePlan) -> CaptureTensorGeometry<'_> {
    CaptureTensorGeometry::prepare(source, 0, CapturePhase::Prefill, 0, None).unwrap()
}

#[test]
fn logical_estimate_matches_legacy_and_keeps_pre_preview_selected_extent() {
    for transform in [
        CaptureTransform::FullTensor,
        CaptureTransform::Slice,
        CaptureTransform::Preview { max_elements: 2 },
        CaptureTransform::Preview { max_elements: 0 },
    ] {
        let plan = admitted(
            vec![SymbolicDimension::Known(3), SymbolicDimension::Known(8)],
            transform,
            vec![
                CaptureSlice {
                    axis: "axis0".into(),
                    start: 0,
                    end: 3,
                    stride: 2,
                },
                CaptureSlice {
                    axis: "axis1".into(),
                    start: 1,
                    end: 8,
                    stride: 3,
                },
            ],
        );
        let selection = &plan.plan().selections[0];
        let slice = resolve_slice(&plan.points()[0], selection, &[3, 8]).unwrap();
        let old = estimate_shape(&[3, 8], selection, &slice).unwrap();
        let actual = estimate_tensor_geometry(&geometry(&plan)).unwrap();
        assert_eq!(actual, old);
        if matches!(
            selection.transform,
            CaptureTransform::Preview { max_elements: 2 }
        ) {
            assert_eq!(actual.retained_bytes, 24 * 8 + 6 * 16 + 4096 + 2 * 128);
            assert_eq!(actual.host_bytes, 32);
        }
    }
    for shape in [vec![], vec![0, 8]] {
        let plan = admitted(
            shape
                .iter()
                .copied()
                .map(SymbolicDimension::Known)
                .collect(),
            CaptureTransform::FullTensor,
            vec![],
        );
        let selection = &plan.plan().selections[0];
        let shape = shape
            .into_iter()
            .map(|n| u64::try_from(n).unwrap())
            .collect::<Vec<_>>();
        let slice = resolve_slice(&plan.points()[0], selection, &shape).unwrap();
        assert_eq!(
            estimate_tensor_geometry(&geometry(&plan)).unwrap(),
            estimate_shape(&shape, selection, &slice).unwrap()
        );
    }
    let plan = admitted(
        vec![SymbolicDimension::Known(i32::MAX as usize + 1)],
        CaptureTransform::Preview { max_elements: 1 },
        vec![],
    );
    assert!(matches!(
        estimate_tensor_geometry(&geometry(&plan)),
        Err(CaptureError::Unsupported(_))
    ));
}

thread_local! { static HOUSEKEEPING: Cell<usize> = const { Cell::new(0) }; }
fn housekeeping() {
    HOUSEKEEPING.set(HOUSEKEEPING.get() + 1);
}
struct ColdCheck;
impl ColdCheck {
    fn new() -> Self {
        safemlx::register_thread_runtime_housekeeping(housekeeping);
        HOUSEKEEPING.set(0);
        Self
    }
}
impl Drop for ColdCheck {
    fn drop(&mut self) {
        safemlx::unregister_thread_runtime_housekeeping(housekeeping);
    }
}

#[test]
fn source_precheck_and_estimate_borrow_lazy_descriptor_without_housekeeping() {
    for dtype in [Dtype::Float32, Dtype::Float16, Dtype::Bfloat16] {
        let f = fixture(dtype);
        let stream = stream();
        let source = f.input.multiply(&f.input, &stream).unwrap();
        let wrong = source.reshape(&[2, 4], &stream).unwrap();
        let integer = source.as_dtype(Dtype::Int32, &stream).unwrap();
        let native = NativeScheduledCapture::for_test(&f.work, &stream);
        let geometry = geometry(f.plan.admission());
        let shape_pointer = source.shape().as_ptr();
        let before = EVALUATIONS.get();
        {
            let _cold = ColdCheck::new();
            assert_eq!(
                native.validate_source(&source, &geometry).unwrap(),
                match dtype {
                    Dtype::Float16 => TensorDtype::F16,
                    Dtype::Bfloat16 => TensorDtype::Bf16,
                    _ => TensorDtype::F32,
                }
            );
            assert!(native.estimate(&source, &geometry).unwrap().retained_bytes > 0);
            assert!(caused_by::<
                crate::backend::array_copy::CaptureTensorNativeError,
            >(
                &native.validate_source(&wrong, &geometry).unwrap_err()
            ));
            assert!(caused_by::<
                crate::backend::array_copy::CaptureTensorNativeError,
            >(
                &native.validate_source(&integer, &geometry).unwrap_err()
            ));
            assert_eq!(source.shape().as_ptr(), shape_pointer);
            assert_eq!(HOUSEKEEPING.get(), 0);
        }
        assert_eq!(EVALUATIONS.get(), before);
        assert!(source
            .try_metadata_snapshot()
            .unwrap()
            .allocation()
            .is_none());
        assert!(f.work.roots.borrow().is_empty());
        drop(native);
        drop(geometry);
        retire_native(&f.work);
        let pool = f.pool.clone();
        drop((source, wrong, integer, f));
        settled_bytes(&pool, 0);
    }
}

struct SessionFixture {
    pool: WorkingMemoryPool,
    input: Array,
    _plan: SharedCapturePlan,
    _reservation: WorkingMemoryReservation,
    _run: Option<WorkingMemoryFundingRun>,
    work: FundedWorkOwner,
    capture: FundedCaptureSession,
    h: u64,
    n: u64,
    _sources: RetainedStoragePublication,
}
impl From<Fixture> for SessionFixture {
    fn from(f: Fixture) -> Self {
        Self {
            pool: f.pool,
            input: f.input,
            _plan: f.plan,
            _reservation: f.reservation,
            _run: Some(f.run),
            work: f.work,
            capture: f.bank.into_capture_session().unwrap(),
            h: f.h,
            n: f.n,
            _sources: f._sources,
        }
    }
}

fn observed(
    observer: &mut dyn ActivationObserver<Array, Error>,
    source: &Array,
    prediction: u64,
) -> Result<(), Error> {
    assert!(observer.requires_prepared_traversal());
    let epoch = DistributedCommitEpoch::new(prediction + 1).unwrap();
    observer.prepare_transaction(
        epoch,
        if prediction == 0 {
            ExpertPass::Prefill
        } else {
            ExpertPass::Decode
        },
    )?;
    // An unrelated hook must remain lazy even though the source is genuinely lazy.
    observer.observe("not.selected", source)?;
    observer.observe("block.output", source)?;
    observer.complete_transaction(epoch)?;
    observer.finish_transaction(epoch, true);
    Ok(())
}

#[test]
fn adapter_uses_original_scope_and_shared_frame_without_certifying_other_work() {
    for dtype in [Dtype::Float32, Dtype::Float16, Dtype::Bfloat16] {
        let mut f = SessionFixture::from(fixture(dtype));
        let stream = stream();
        let source = f.input.multiply(&f.input, &stream).unwrap();
        let unrelated = f.input.add(&f.input, &stream).unwrap();
        f.work.retain(&unrelated);
        let mut native = NativeScheduledCapture::for_test(&f.work, &stream);
        let before = EVALUATIONS.get();
        f.capture
            .with_observer(&mut native, 0, &observer_error, |o| observed(o, &source, 0))
            .unwrap()
            .unwrap();
        drop(native);
        assert_eq!(EVALUATIONS.get(), before + 1);
        assert!(unrelated
            .try_metadata_snapshot()
            .unwrap()
            .allocation()
            .is_none());
        assert!(!f.work.published.get());
        assert!(f.work.scope.borrow().is_some());
        // Completion is established independently before the host drain.
        retire_native(&f.work);
        let step = f.capture.take_shared_step().unwrap().unwrap();
        assert_eq!(step.outcome(), CaptureStepOutcome::Committed);
        let CapturePayload::SharedTensor(values) = step.records()[0].payload.as_ref().unwrap()
        else {
            panic!("shared tensor")
        };
        assert_eq!(
            values.data(),
            &TensorObservationData::F32(vec![1., 4., 9., 16., 25., 36., 49., 64.])
        );
        let alias = values.clone();
        let pool = f.pool.clone();
        // The shared F32 alias owns H, with no native source or plan payload.
        let retained = f.h;
        drop((step, source, unrelated, f));
        settled_bytes(&pool, retained);
        drop(alias);
        settled_bytes(&pool, 0);
    }
}

fn restricted_plan(quota: bool) -> SharedCapturePlan {
    let base = admitted(
        vec![SymbolicDimension::Known(8)],
        CaptureTransform::FullTensor,
        vec![],
    );
    let mut plan = base.plan().clone();
    if quota {
        plan.limits.per_step.captures = 0;
        plan.limits.on_limit = CaptureLimitPolicy::Skip;
    } else {
        plan.selections[0].schedule.decode = false;
    }
    let catalog = ObservationCatalog {
        schema_version: 1,
        points: base.points().to_vec(),
        completeness: DescriptionCompleteness::Complete,
    };
    let support = ObservationSupportReport {
        schema_version: 1,
        capture: Default::default(),
        points: vec![ObservationSupport {
            path: "block.output".into(),
            prefill: ObservationSupportStatus::Supported,
            decode: ObservationSupportStatus::Supported,
            floating_to_f32: true,
        }],
    };
    SharedCapturePlan::new(
        plan.admit(
            &catalog,
            &support,
            &CaptureCapabilities {
                transformations: vec![CaptureTransformKind::FullTensor],
                max_histogram_bins: 0,
                physical_native_limit: false,
                conditions: vec![],
            },
            base.request(),
        )
        .unwrap(),
    )
}

#[test]
fn logical_limit_skip_does_not_evaluate_or_retain_a_lazy_source() {
    let mut f = SessionFixture::from(fixture_in_pool(Dtype::Float32, restricted_plan(true), None));
    let stream = stream();
    let source = f.input.multiply(&f.input, &stream).unwrap();
    let mut native = NativeScheduledCapture::for_test(&f.work, &stream);
    let before = EVALUATIONS.get();
    f.capture
        .with_observer(&mut native, 0, &observer_error, |o| observed(o, &source, 0))
        .unwrap()
        .unwrap();
    drop(native);
    assert_eq!(EVALUATIONS.get(), before);
    assert!(source
        .try_metadata_snapshot()
        .unwrap()
        .allocation()
        .is_none());
    assert!(f.work.roots.borrow().is_empty());
    retire_native(&f.work);
    let step = f.capture.take_shared_step().unwrap().unwrap();
    assert!(matches!(
        step.records()[0].outcome,
        CaptureOutcome::Skipped {
            reason: CaptureSkipReason::Limit { .. }
        }
    ));
    assert_eq!(step.cumulative_usage().captures, 0);
    let pool = f.pool.clone();
    drop((step, source, f));
    settled_bytes(&pool, 0);
}

fn bridge_fixture(
    stream: &Stream,
) -> (
    ModelRuntime<MlxBackend<'static>>,
    tempfile::TempDir,
    SessionFixture,
) {
    // The real loaded model is pre-existing registered storage. Only the actual
    // array-capture fixture's equations/H are priced here: no model forward,
    // core text permit, or observed generation support is claimed by this test.
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let (runtime, artifact) = host_layerwise_tests::runtime(stream, &pool, None);
    let f = fixture_in_pool(Dtype::Float32, shared(), Some(pool));
    (runtime, artifact, SessionFixture::from(f))
}

fn installed_operation<'a>(
    session: &'a mut crate::composition::mlx::session::MlxModelSession,
    work: Option<&FundedWorkOwner>,
) -> SessionOperation<'a> {
    // Test-only lower-level owner seam. The production bridge has no work
    // argument and can retrieve only work installed by begin_text_submission.
    session.ensure_no_submission_in_flight().unwrap();
    let lease = session.authority.borrow_mut().begin_submission().unwrap();
    let owner = SubmissionResources::new(lease, Rc::clone(&session.poison));
    owner.funding.replace(work.cloned());
    let recovery = owner.recovery().unwrap();
    SessionOperation {
        session,
        owner,
        recovery: Some(recovery),
        handed_off: false,
        token_validations: Default::default(),
    }
}
fn finish_component(operation: SessionOperation<'_>) {
    let ((), owner, recovery) = operation.finish(Ok(())).unwrap();
    complete_model_operation((), owner, recovery).unwrap();
}
fn retire_records() {
    // Ordinary terminal-record cleanup after exact component completion; no
    // array evaluation or numerical work is submitted by this empty event.
    safemlx::transforms::async_eval_with_event(std::iter::empty::<&Array>())
        .unwrap()
        .synchronize()
        .unwrap();
}

#[test]
fn bridge_lends_exclusive_model_without_holding_native_scope_and_releases_observer_first() {
    let stream = stream();
    let (mut runtime, artifact, mut f) = bridge_fixture(&stream);
    let source = f.input.multiply(&f.input, &stream).unwrap();
    let expected_model = &runtime.session().payload.model as *const _;
    let state = runtime.session().payload.model.erased().state_snapshot();
    let backend = MlxBackend::new(&stream, &stream).with_memory_pool(f.pool.clone());
    let mut operation = installed_operation(runtime.session_mut(), Some(&f.work));
    operation
        .with_funded_capture(&backend, &mut f.capture, 0, None, |model, observer| {
            assert!(std::ptr::eq(model, expected_model));
            assert!(f.work.scope.try_borrow_mut().is_ok());
            observed(observer, &source, 0)?;
            assert!(f.work.scope.try_borrow_mut().is_ok());
            assert_eq!(model.erased().state_snapshot(), state);
            Ok(())
        })
        .unwrap();
    // The lexical observer/claim borrows have ended before whole-work finalization.
    assert!(f.capture.has_pending_step());
    assert!(f.work.scope.try_borrow_mut().is_ok());
    assert_eq!(operation.session.payload.active_owner_count(), 1);
    finish_component(operation);
    retire_native(&f.work);
    let step = f.capture.take_shared_step().unwrap().unwrap();
    assert_eq!(step.outcome(), CaptureStepOutcome::Committed);
    assert_eq!(step.cumulative_usage().captures, 1);
    assert_eq!(
        runtime.session().payload.model.erased().state_snapshot(),
        state
    );
    runtime.synchronize().unwrap();
    let pool = f.pool.clone();
    drop((step, source, f, runtime, artifact));
    retire_records();
    settled_bytes(&pool, 0);
}

#[test]
fn bridge_without_installed_funding_rejects_before_callback_or_claim() {
    let stream = stream();
    let (mut runtime, artifact, mut f) = bridge_fixture(&stream);
    let backend = MlxBackend::new(&stream, &stream).with_memory_pool(f.pool.clone());
    let mut operation = installed_operation(runtime.session_mut(), None);
    let invoked = Cell::new(false);
    let error = operation
        .with_funded_capture(&backend, &mut f.capture, 0, None, |_, _| {
            invoked.set(true);
            Ok(())
        })
        .unwrap_err();
    assert_eq!(
        memory_cause(&error),
        Some(&WorkingMemoryError::IdentityMismatch)
    );
    assert!(!invoked.get());
    assert_eq!(f.capture.spent_steps(), 0);
    assert!(!f.capture.has_pending_step());
    assert!(f.work.roots.borrow().is_empty());
    finish_component(operation);
    retire_native(&f.work);
    runtime.synchronize().unwrap();
    let pool = f.pool.clone();
    drop((error, f, runtime, artifact));
    retire_records();
    settled_bytes(&pool, 0);
}

#[test]
fn bridge_error_and_unwind_detach_aborted_frame_while_original_recovery_keeps_roots() {
    for panic in [false, true] {
        let stream = stream();
        let (mut runtime, artifact, mut f) = bridge_fixture(&stream);
        let backend = MlxBackend::new(&stream, &stream).with_memory_pool(f.pool.clone());
        let source = f.input.multiply(&f.input, &stream).unwrap();
        let mut operation = installed_operation(runtime.session_mut(), Some(&f.work));
        if panic {
            PANIC_PUBLICATION.set(true);
        } else {
            FAIL_PUBLICATION.set(true);
        }
        let result = catch_unwind(AssertUnwindSafe(|| {
            operation.with_funded_capture(&backend, &mut f.capture, 0, None, |_, observer| {
                observed(observer, &source, 0)
            })
        }));
        if panic {
            assert!(result.is_err());
        } else {
            assert!(caused_by::<InjectedIngress>(&result.unwrap().unwrap_err()));
        }
        assert!(f.capture.has_pending_step());
        assert_eq!(f.capture.spent_steps(), 1);
        assert!(!f.work.roots.borrow().is_empty());
        assert!(f.work.scope.borrow().is_some());
        assert!(!f.work.published.get());
        // Failure never invokes successful model finalization/certification.
        drop(operation);
        retire_records();
        let step = f.capture.take_shared_step().unwrap().unwrap();
        assert_eq!(step.outcome(), CaptureStepOutcome::Aborted);
        assert_eq!(step.cumulative_usage().captures, 1);
        let pool = f.pool.clone();
        let protected = f.h + f.n;
        drop((step, source, f, runtime, artifact));
        retire_records();
        settled_bytes(&pool, protected);
    }
}

#[test]
fn scheduled_decode_skip_leaves_new_lazy_source_unobserved() {
    let mut f = SessionFixture::from(fixture_in_pool(
        Dtype::Float32,
        restricted_plan(false),
        None,
    ));
    let stream = stream();
    let first = f.input.multiply(&f.input, &stream).unwrap();
    let next = f.input.add(&f.input, &stream).unwrap();
    let mut native = NativeScheduledCapture::for_test(&f.work, &stream);
    f.capture
        .with_observer(&mut native, 0, &observer_error, |o| observed(o, &first, 0))
        .unwrap()
        .unwrap();
    // Only the selected first graph ran. Its synchronous leaf and empty terminal
    // wait settle that graph before draining; the new source stays unsubmitted.
    retire_records();
    drop(f.capture.take_shared_step().unwrap().unwrap());
    let before = EVALUATIONS.get();
    let roots = f.work.roots.borrow().len();
    f.capture
        .with_observer(&mut native, 1, &observer_error, |o| observed(o, &next, 1))
        .unwrap()
        .unwrap();
    drop(native);
    assert_eq!(EVALUATIONS.get(), before);
    assert_eq!(f.work.roots.borrow().len(), roots);
    assert!(next.try_metadata_snapshot().unwrap().allocation().is_none());
    retire_native(&f.work);
    let step = f.capture.take_shared_step().unwrap().unwrap();
    assert!(matches!(
        step.records()[0].outcome,
        CaptureOutcome::Skipped {
            reason: CaptureSkipReason::Schedule
        }
    ));
    let pool = f.pool.clone();
    drop((step, first, next, f));
    retire_records();
    settled_bytes(&pool, 0);
}

#[test]
fn bridge_rejects_same_pool_foreign_bank_before_lending_model_or_spending_claim() {
    let stream = stream();
    let (mut runtime, artifact, f) = bridge_fixture(&stream);
    let source = f.capture.source().clone();
    let plan = CaptureRunHostPlan::prepare(&source).unwrap();
    let (foreign_reservation, foreign_run) = fresh(&f.pool, plan.initialization_peak_bytes());
    let mut foreign = foreign_run
        .prepare_capture_run(&foreign_reservation, plan)
        .unwrap()
        .into_capture_session()
        .unwrap();
    let backend = MlxBackend::new(&stream, &stream).with_memory_pool(f.pool.clone());
    let mut operation = installed_operation(runtime.session_mut(), Some(&f.work));
    let invoked = Cell::new(false);
    let before = (f.pool.used_bytes().unwrap(), f.pool.peak_bytes().unwrap());
    let error = operation
        .with_funded_capture(&backend, &mut foreign, 0, None, |_, _| {
            invoked.set(true);
            Ok(())
        })
        .unwrap_err();
    assert_eq!(
        memory_cause(&error),
        Some(&WorkingMemoryError::IdentityMismatch)
    );
    assert!(!invoked.get());
    assert_eq!(foreign.spent_steps(), 0);
    assert!(!foreign.has_pending_step());
    assert_eq!(
        (f.pool.used_bytes().unwrap(), f.pool.peak_bytes().unwrap()),
        before
    );
    assert!(f.work.roots.borrow().is_empty());
    finish_component(operation);
    retire_native(&f.work);
    runtime.synchronize().unwrap();
    let pool = f.pool.clone();
    drop((
        error,
        foreign,
        foreign_run,
        foreign_reservation,
        source,
        f,
        runtime,
        artifact,
    ));
    retire_records();
    settled_bytes(&pool, 0);
}

#[test]
fn bridge_rechecks_closed_or_quarantined_bank_before_model_callback() {
    for quarantine in [false, true] {
        let stream = stream();
        let (mut runtime, artifact, mut f) = bridge_fixture(&stream);
        if quarantine {
            // An abandoned sibling makes the original account unhealthy; the
            // actual work scope remains present and cannot disguise that fence.
            drop(f._run.as_ref().unwrap().scope().unwrap());
        } else {
            f._run.take().unwrap().close().unwrap();
        }
        let backend = MlxBackend::new(&stream, &stream).with_memory_pool(f.pool.clone());
        let mut operation = installed_operation(runtime.session_mut(), Some(&f.work));
        let invoked = Cell::new(false);
        let error = operation
            .with_funded_capture(&backend, &mut f.capture, 0, None, |_, _| {
                invoked.set(true);
                Ok(())
            })
            .unwrap_err();
        assert_eq!(
            memory_cause(&error),
            Some(&WorkingMemoryError::ExecutionFenced)
        );
        assert!(!invoked.get());
        assert_eq!(f.capture.spent_steps(), 0);
        assert!(!f.capture.has_pending_step());
        assert!(f.work.roots.borrow().is_empty());
        finish_component(operation);
        if !quarantine {
            retire_native(&f.work);
        }
        runtime.synchronize().unwrap();
        let pool = f.pool.clone();
        let protected = if quarantine { f.h + f.n } else { 0 };
        drop((error, f, runtime, artifact));
        retire_records();
        settled_bytes(&pool, protected);
    }
}
