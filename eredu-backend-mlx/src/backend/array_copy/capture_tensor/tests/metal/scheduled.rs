use super::*;
use eredu_runtime::working_memory::{
    CaptureRunHostError, CaptureRunHostPlan, ScheduledCaptureStep,
};
use half::{bf16, f16};

// This fixture quotes the real cumulative host schedule plus the one native
// coordinate exercised below. It is not a model or full native-run quote.
fn fresh_result(
    pool: &WorkingMemoryPool,
    bytes: u64,
) -> Result<(WorkingMemoryReservation, WorkingMemoryFundingRun), WorkingMemoryError> {
    let geometry = InferenceGeometry {
        batch_size: 1,
        cached_positions: 0,
        input_positions: 3,
        max_output_tokens: 4,
        prefill_chunk_positions: 3,
        output: OutputDemand::LastPosition,
    };
    let layout = StateMemoryLayout::new(
        LayerSchedule::new(1, vec![LayerCachePolicy::NoState]).unwrap(),
        vec![0],
        1,
        1,
        EstimationCompleteness::Complete,
    )
    .unwrap();
    let state = estimate_runtime_state(
        &layout,
        InputTokenCount::text(3),
        4,
        1,
        NonZeroU8::new(4).unwrap(),
    )
    .unwrap();
    let bound = |bytes| WorkspaceBound::bounded(bytes, "portable parent-account fixture");
    let state = state
        .with_execution_workspace(ExecutionWorkspaceEstimate {
            geometry,
            activations: bound(bytes),
            attention: bound(0),
            vocabulary: bound(0),
            state_update: bound(0),
            materialization: bound(0),
            retained: bound(0),
        })
        .unwrap();
    pool.reserve_with_capacity(
        &InferenceExecutionIdentity::default(),
        &Admission {
            state,
            requested_positions: 7,
            incremental_required_bytes: bytes,
            available_memory_bytes: None,
        },
        pool.effective_capacity().unwrap(),
    )?
    .into_funding()
}

fn all_usage() -> CaptureUsage {
    CaptureUsage {
        captures: u64::MAX,
        retained_bytes: u64::MAX,
        host_bytes: u64::MAX,
        encoded_bytes: u64::MAX,
    }
}
fn frame_finish(frame: ScheduledCaptureStep<'_>) -> SharedCapturedStep {
    frame
        .finish(CaptureStepOutcome::Committed, all_usage(), all_usage(), 0.0)
        .unwrap()
}
fn at(
    source: &AdmittedCapturePlan,
    index: usize,
    phase: CapturePhase,
    prediction: u64,
) -> CaptureTensorHostPlan<'_> {
    CaptureTensorHostPlan::prepare(
        CaptureTensorGeometry::prepare(source, index, phase, prediction, None).unwrap(),
    )
    .unwrap()
}
fn native_bound<'a>(source: &'a Array, leaf: &PreparedCaptureTensor<'a>) -> u64 {
    let context = WorkspaceContext::new(MlxMetalWorkspaceMechanisms::current_host().unwrap());
    let mut projection = ExistingArrayProjection::new(&context);
    let input = projection.project(source).unwrap();
    context.begin_state_span(&[input.clone()]).unwrap();
    let output = leaf.trace(&mut projection).unwrap();
    let report = context.report(&[input, output]).unwrap();
    assert_eq!(report.host_workspace_bytes, Some(0));
    assert!(report.unpriced_operations.is_empty());
    assert!(report.unpriced_host_operations.is_empty());
    report.total_bytes.unwrap()
}
fn root_entries(roots: &RefCell<Vec<Array>>) -> BTreeMap<StorageIdentity, u64> {
    roots
        .borrow()
        .iter()
        .filter_map(|root| {
            root.evaluated().unwrap();
            let info = root.try_metadata_snapshot().unwrap().allocation().unwrap();
            (info.bytes() != 0).then_some((
                StorageIdentity::Native(info.identity()),
                info.bytes() as u64,
            ))
        })
        .collect()
}
fn reference(array: &Array, stream: &Stream) -> Vec<u32> {
    if array.size() == 0 {
        return vec![];
    }
    crate::MlxTensor::from_array(array.clone())
        .to_f32_vec(stream)
        .unwrap()
        .into_iter()
        .map(f32::to_bits)
        .collect()
}
fn dtype(dtype: Dtype) -> eredu_core::checkpoint::TensorDtype {
    use eredu_core::checkpoint::TensorDtype;
    match dtype {
        Dtype::Float32 => TensorDtype::F32,
        Dtype::Float16 => TensorDtype::F16,
        Dtype::Bfloat16 => TensorDtype::Bf16,
        _ => unreachable!(),
    }
}
fn root(precision: usize, shape: &[i32]) -> Array {
    let count = shape.iter().map(|&n| n as usize).product::<usize>();
    let values = (0..count)
        .map(|i| i as f32 * 0.5 - 3.25)
        .collect::<Vec<_>>();
    match precision {
        0 => Array::from_slice(&values, shape),
        1 => Array::from_slice(
            &values
                .iter()
                .copied()
                .map(f16::from_f32)
                .collect::<Vec<_>>(),
            shape,
        ),
        _ => Array::from_slice(
            &values
                .iter()
                .copied()
                .map(bf16::from_f32)
                .collect::<Vec<_>>(),
            shape,
        ),
    }
}
fn run_case(
    source: &Array,
    transform: CaptureTransform,
    slices: Vec<CaptureSlice>,
    expected: &Array,
    stream: &Stream,
) {
    source.evaluated().unwrap();
    let expected = reference(expected, stream);
    let source_metadata = source.try_metadata_snapshot().unwrap();
    let source_bytes = source_metadata.allocation().unwrap().bytes() as u64;
    let shared = SharedCapturePlan::new(admission(source.shape(), transform, slices));
    let h = CaptureRunHostPlan::prepare(&shared)
        .unwrap()
        .initialization_peak_bytes();
    let leaf = PreparedCaptureTensor::new(source, host(shared.admission())).unwrap();
    let n = native_bound(source, &leaf);
    // The actual original reservation rejects the combined H+N one byte short,
    // before any claim bank, host destination or selected native work exists.
    let short = WorkingMemoryPool::new(source_bytes + h + n - 1, 0).unwrap();
    let short_source = register(&short, source);
    let before = (short.used_bytes().unwrap(), short.peak_bytes().unwrap());
    let error = fresh_result(&short, h + n).unwrap_err();
    assert!(matches!(error, WorkingMemoryError::BudgetExceeded { .. }));
    assert_eq!(
        (short.used_bytes().unwrap(), short.peak_bytes().unwrap()),
        before
    );
    drop(short_source);
    assert_eq!(short.used_bytes().unwrap(), 0);

    let pool = WorkingMemoryPool::new(source_bytes + h + n, 0).unwrap();
    let registered = register(&pool, source);
    let (reservation, run) = fresh_result(&pool, h + n).unwrap();
    let mut native = run.scope().unwrap();
    let mut bank = run
        .prepare_capture_run(&reservation, CaptureRunHostPlan::prepare(&shared).unwrap())
        .unwrap();
    assert_eq!(bank.protected_bytes(), h);
    let roots = RefCell::new(Vec::with_capacity(leaf.recovery_descriptors()));
    let capacity = roots.borrow().capacity();
    let before = pool.used_bytes().unwrap();
    let mut frame = bank
        .begin_step(CapturePhase::Prefill, 0)
        .unwrap()
        .prepare()
        .unwrap();
    let receipt = leaf
        .transfer_scheduled(frame.take_tensor(0).unwrap(), &mut native, stream, &roots)
        .unwrap();
    assert_eq!(pool.used_bytes().unwrap(), before);
    assert_eq!(roots.borrow().capacity(), capacity);
    assert_eq!(
        receipt.observation().shape(),
        host(shared.admission()).geometry().shape()
    );
    let TensorObservationData::F32(values) = receipt.observation().data() else {
        unreachable!()
    };
    assert_eq!(
        values.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        expected
    );
    let alias = receipt.observation().clone();
    frame
        .record_tensor(receipt, dtype(source.dtype()), CaptureUsage::default())
        .unwrap();
    let frame = frame_finish(frame);
    let entries = root_entries(&roots);
    assert!(entries.values().sum::<u64>() <= source_bytes + n);
    // Exact completed native allocations fit alongside all still-protected H.
    // A second child hold would have exhausted this deliberately exact account.
    let publication = native.adopt_storage_individually(entries).unwrap();
    assert_eq!(pool.used_bytes().unwrap(), before);
    settle(&roots);
    native.certify().unwrap();
    drop((publication, frame, bank, reservation, run, registered));
    assert!(pool.used_bytes().unwrap() >= h);
    drop(alias);
    assert_eq!(pool.used_bytes().unwrap(), 0);
    assert_eq!(source.try_metadata_snapshot().unwrap(), source_metadata);
}

#[test]
fn scheduled_f32_f16_bf16_full_slice_preview_use_exact_original_host_and_native_credit() {
    let stream = stream();
    for precision in 0..3 {
        let root = root(precision, &[6, 8]);
        let view = root
            .try_index_device((1..5, (1..8).stride_by(2)), &stream)
            .unwrap();
        view.evaluated().unwrap();
        assert!(view.allocation_info().unwrap().unwrap().bytes() > view.nbytes());
        run_case(&view, CaptureTransform::FullTensor, vec![], &view, &stream);
        let selected = view
            .try_index_device((1..3, (0..4).stride_by(2)), &stream)
            .unwrap();
        run_case(
            &view,
            CaptureTransform::Slice,
            vec![
                CaptureSlice {
                    axis: "axis0".into(),
                    start: 1,
                    end: 3,
                    stride: 1,
                },
                CaptureSlice {
                    axis: "axis1".into(),
                    start: 0,
                    end: 4,
                    stride: 2,
                },
            ],
            &selected,
            &stream,
        );
        let flat = view.reshape(&[16], &stream).unwrap();
        let prefix = flat.try_index_device(0..5, &stream).unwrap();
        run_case(
            &view,
            CaptureTransform::Preview { max_elements: 5 },
            vec![],
            &prefix,
            &stream,
        );
    }
}

#[test]
fn scheduled_scalar_empty_preview_and_negative_strides_preserve_existing_worker_behavior() {
    let stream = stream();
    for precision in 0..3 {
        let scalar = root(precision, &[]);
        run_case(
            &scalar,
            CaptureTransform::FullTensor,
            vec![],
            &scalar,
            &stream,
        );
        let empty = root(precision, &[0, 3]);
        run_case(
            &empty,
            CaptureTransform::FullTensor,
            vec![],
            &empty,
            &stream,
        );
        let source = root(precision, &[6]);
        let reverse = source
            .try_index_device((..).stride_by(-1), &stream)
            .unwrap();
        run_case(
            &reverse,
            CaptureTransform::FullTensor,
            vec![],
            &reverse,
            &stream,
        );
        let zero = source.try_index_device(0..0, &stream).unwrap();
        run_case(
            &source,
            CaptureTransform::Preview { max_elements: 0 },
            vec![],
            &zero,
            &stream,
        );
    }
}

fn multiple(shape: &[i32]) -> SharedCapturePlan {
    let original = admission(shape, CaptureTransform::FullTensor, vec![]);
    let mut raw = original.plan().clone();
    let mut second = raw.selections[0].clone();
    second.id.push_str("-other");
    raw.selections.push(second);
    let catalog = ObservationCatalog {
        schema_version: 1,
        points: original.points().to_vec(),
        completeness: DescriptionCompleteness::Complete,
    };
    let caps = CaptureCapabilities {
        transformations: vec![CaptureTransformKind::FullTensor],
        ..Default::default()
    };
    let support = ObservationSupportReport {
        schema_version: 1,
        capture: caps.clone(),
        points: original
            .points()
            .iter()
            .map(|p| ObservationSupport {
                path: p.path.clone(),
                prefill: ObservationSupportStatus::Supported,
                decode: ObservationSupportStatus::Supported,
                floating_to_f32: true,
            })
            .collect(),
    };
    SharedCapturePlan::new(
        raw.admit(&catalog, &support, &caps, original.request())
            .unwrap(),
    )
}

#[test]
fn scheduled_independent_admission_selection_phase_and_prediction_mismatch_stop_before_work() {
    let stream = stream();
    let source = root(0, &[4]);
    source.evaluated().unwrap();
    let snapshot = source.try_metadata_snapshot().unwrap();
    for case in 0..4 {
        let shared = multiple(&[4]);
        let independent = SharedCapturePlan::new(shared.admission().clone());
        assert_eq!(
            shared.admission().identity(),
            independent.admission().identity()
        );
        assert!(!shared.same_storage(&independent));
        let h = CaptureRunHostPlan::prepare(&shared)
            .unwrap()
            .initialization_peak_bytes();
        let bytes = snapshot.allocation().unwrap().bytes() as u64;
        let pool = WorkingMemoryPool::new(bytes + h, 0).unwrap();
        let registration = register(&pool, &source);
        let (reservation, run) = fresh_result(&pool, h).unwrap();
        let mut native = run.scope().unwrap();
        let mut bank = run
            .prepare_capture_run(&reservation, CaptureRunHostPlan::prepare(&shared).unwrap())
            .unwrap();
        let (phase, prediction) = if case == 3 {
            drop(bank.begin_step(CapturePhase::Prefill, 0).unwrap());
            (CapturePhase::Decode, 1)
        } else {
            (CapturePhase::Prefill, 0)
        };
        let mut frame = bank
            .begin_step(phase, prediction)
            .unwrap()
            .prepare()
            .unwrap();
        let geometry = match case {
            0 => at(independent.admission(), 0, phase, prediction),
            1 => at(shared.admission(), 1, phase, prediction),
            2 => at(shared.admission(), 0, CapturePhase::Decode, 1),
            _ => at(shared.admission(), 0, CapturePhase::Decode, 2),
        };
        let leaf = PreparedCaptureTensor::new(&source, geometry).unwrap();
        let roots = RefCell::new(vec![]);
        let before = pool.used_bytes().unwrap();
        FAIL_AFTER_SLICE.set(true);
        let error = leaf
            .transfer_scheduled(frame.take_tensor(0).unwrap(), &mut native, &stream, &roots)
            .unwrap_err();
        assert!(matches!(
            error,
            ScheduledCaptureTensorExecutionError::Mechanism(
                CaptureTensorNativeError::ClaimMismatch
            )
        ));
        drop(error);
        assert!(FAIL_AFTER_SLICE.replace(false));
        assert!(roots.borrow().is_empty());
        assert_eq!(pool.used_bytes().unwrap(), before);
        assert!(frame.take_tensor(0).is_err());
        assert_eq!(source.try_metadata_snapshot().unwrap(), snapshot);
        drop(frame);
        native.certify().unwrap();
        drop((bank, reservation, run, registration));
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
}

#[test]
fn scheduled_foreign_scope_and_borrowed_collector_reject_without_ops_or_second_claim() {
    let stream = stream();
    let source = root(0, &[4]);
    source.evaluated().unwrap();
    let shared = multiple(&[4]);
    let h = CaptureRunHostPlan::prepare(&shared)
        .unwrap()
        .initialization_peak_bytes();
    let bytes = source.allocation_info().unwrap().unwrap().bytes() as u64;
    let pool = WorkingMemoryPool::new(bytes + 2 * h, 0).unwrap();
    let registration = register(&pool, &source);
    let (reservation, run) = fresh_result(&pool, h).unwrap();
    let (other_reservation, other_run) = fresh_result(&pool, h).unwrap();
    let mut native = run.scope().unwrap();
    let mut wrong = other_run.scope().unwrap();
    let mut bank = run
        .prepare_capture_run(&reservation, CaptureRunHostPlan::prepare(&shared).unwrap())
        .unwrap();
    let mut frame = bank
        .begin_step(CapturePhase::Prefill, 0)
        .unwrap()
        .prepare()
        .unwrap();
    let roots = RefCell::new(vec![]);
    let before = pool.used_bytes().unwrap();
    let error =
        PreparedCaptureTensor::new(&source, at(shared.admission(), 0, CapturePhase::Prefill, 0))
            .unwrap()
            .transfer_scheduled(frame.take_tensor(0).unwrap(), &mut wrong, &stream, &roots)
            .unwrap_err();
    assert!(matches!(
        error,
        ScheduledCaptureTensorExecutionError::Mechanism(CaptureTensorNativeError::Claim(
            CaptureRunHostError::Tensor(CaptureTensorConstructionError::Memory(
                WorkingMemoryError::IdentityMismatch
            ))
        ))
    ));
    drop(error);
    {
        let _borrow = roots.borrow_mut();
        let error = PreparedCaptureTensor::new(
            &source,
            at(shared.admission(), 1, CapturePhase::Prefill, 0),
        )
        .unwrap()
        .transfer_scheduled(frame.take_tensor(1).unwrap(), &mut native, &stream, &roots)
        .unwrap_err();
        assert!(matches!(
            error,
            ScheduledCaptureTensorExecutionError::Mechanism(
                CaptureTensorNativeError::CollectorBusy
            )
        ));
        drop(error);
    }
    assert!(roots.borrow().is_empty());
    assert_eq!(pool.used_bytes().unwrap(), before);
    assert!(frame.take_tensor(0).is_err());
    assert!(frame.take_tensor(1).is_err());
    drop(frame);
    native.certify().unwrap();
    wrong.certify().unwrap();
    drop((
        bank,
        reservation,
        run,
        other_reservation,
        other_run,
        registration,
    ));
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn scheduled_cast_evaluation_failure_and_unwind_keep_roots_and_exact_scope_quarantine() {
    let stream = stream();
    for fault in 0..3 {
        let source = root(1 + fault % 2, &[8]);
        source.evaluated().unwrap();
        let snapshot = source.try_metadata_snapshot().unwrap();
        let shared = SharedCapturePlan::new(admission(&[8], CaptureTransform::FullTensor, vec![]));
        let h = CaptureRunHostPlan::prepare(&shared)
            .unwrap()
            .initialization_peak_bytes();
        let leaf = PreparedCaptureTensor::new(&source, host(shared.admission())).unwrap();
        let n = native_bound(&source, &leaf);
        let bytes = snapshot.allocation().unwrap().bytes() as u64;
        let pool = WorkingMemoryPool::new(bytes + h + n, 0).unwrap();
        let registered = register(&pool, &source);
        let (reservation, run) = fresh_result(&pool, h + n).unwrap();
        let mut native = run.scope().unwrap();
        let sibling = run.scope().unwrap();
        let mut bank = run
            .prepare_capture_run(&reservation, CaptureRunHostPlan::prepare(&shared).unwrap())
            .unwrap();
        let mut frame = bank
            .begin_step(CapturePhase::Prefill, 0)
            .unwrap()
            .prepare()
            .unwrap();
        let roots = RefCell::new(Vec::with_capacity(leaf.recovery_descriptors()));
        match fault {
            0 => FAIL_AFTER_CAST.set(true),
            1 => FAIL_AFTER_EVAL.set(true),
            _ => PANIC_AFTER_CAST.set(true),
        }
        let result = catch_unwind(AssertUnwindSafe(|| {
            let result = leaf.transfer_scheduled(
                frame.take_tensor(0).unwrap(),
                &mut native,
                &stream,
                &roots,
            );
            if fault < 2 {
                let error = result.unwrap_err();
                assert!(cause(error.source().unwrap()));
                drop(error);
            } else {
                drop(result);
            }
        }));
        if fault < 2 {
            result.unwrap();
        } else {
            assert!(result.is_err());
        }
        assert_eq!(roots.borrow().len(), 3);
        assert_eq!(roots.borrow().last().unwrap().dtype(), Dtype::Float32);
        assert!(frame.take_tensor(0).is_err());
        drop(frame);
        sibling.certify().unwrap();
        drop(registered);
        assert_eq!(pool.used_bytes().unwrap(), bytes + h + n);
        assert_eq!(source.try_metadata_snapshot().unwrap(), snapshot);
        // Actual retained work is settled independently. Deliberately abandon
        // the exact scope afterward to verify conservative quarantine, even
        // though its unrelated sibling was successfully certified.
        settle(&roots);
        drop(native);
        drop((bank, reservation, run));
        assert_eq!(pool.used_bytes().unwrap(), bytes + h + n);
        assert!(pool.acquire_unquoted().is_err());
    }
}

#[test]
fn scheduled_failed_finish_retains_borrowed_host_owner_until_explicit_retirement() {
    let stream = stream();
    let source = root(2, &[8]);
    source.evaluated().unwrap();
    let shared = SharedCapturePlan::new(admission(&[8], CaptureTransform::FullTensor, vec![]));
    let h = CaptureRunHostPlan::prepare(&shared)
        .unwrap()
        .initialization_peak_bytes();
    let leaf = PreparedCaptureTensor::new(&source, host(shared.admission())).unwrap();
    let n = native_bound(&source, &leaf);
    let bytes = source.allocation_info().unwrap().unwrap().bytes() as u64;
    let pool = WorkingMemoryPool::new(bytes + h + n, 0).unwrap();
    let registered = register(&pool, &source);
    let (reservation, run) = fresh_result(&pool, h + n).unwrap();
    let mut native = run.scope().unwrap();
    let sibling = run.scope().unwrap();
    let mut bank = run
        .prepare_capture_run(&reservation, CaptureRunHostPlan::prepare(&shared).unwrap())
        .unwrap();
    let mut frame = bank
        .begin_step(CapturePhase::Prefill, 0)
        .unwrap()
        .prepare()
        .unwrap();
    let roots = RefCell::new(Vec::with_capacity(leaf.recovery_descriptors()));
    // Close the real original parent only after selection/evaluation and full
    // direct iteration. The following finish must retain its exact error owner.
    struct ResetFinish;
    impl Drop for ResetFinish {
        fn drop(&mut self) {
            BEFORE_FINISH.with(|slot| drop(slot.borrow_mut().take()));
        }
    }
    let _reset = ResetFinish;
    BEFORE_FINISH.with(|slot| *slot.borrow_mut() = Some(Box::new(move || drop(run))));
    let error = leaf
        .transfer_scheduled(frame.take_tensor(0).unwrap(), &mut native, &stream, &roots)
        .unwrap_err();
    let ScheduledCaptureTensorExecutionError::Finish(error) = error else {
        panic!("expected borrowed finish owner")
    };
    assert!(matches!(
        error
            .source()
            .and_then(|e| e.downcast_ref::<WorkingMemoryError>()),
        Some(WorkingMemoryError::ExecutionFenced)
    ));
    assert_eq!(pool.used_bytes().unwrap(), bytes + h + n);
    sibling.certify().unwrap();
    // Retire the borrowed error before touching the exact native scope.
    drop(error);
    drop(frame);
    settle(&roots);
    native.certify().unwrap();
    drop((bank, reservation, registered));
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn source_geometry_validation_accepts_lazy_metadata_without_settling_or_authorizing_transfer() {
    let stream = stream();
    let cpu = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    PreparedCaptureTensor::validate_stream(&stream).unwrap();
    assert!(matches!(
        PreparedCaptureTensor::validate_stream(&cpu),
        Err(CaptureTensorNativeError::UnsupportedStream(DeviceType::Cpu))
    ));
    let admitted = admission(&[3, 2], CaptureTransform::FullTensor, vec![]);
    let wrong = admission(&[2, 3], CaptureTransform::FullTensor, vec![]);
    let f64_source = Array::from_slice_f64(&[1f64, -0., f64::INFINITY, -2., 3., 4.], &[3, 2]);
    for precision in 0..3 {
        let root = root(precision, &[2, 3]);
        root.evaluated().unwrap();
        let lazy = root.reshape(&[3, 2], &stream).unwrap();
        let before = lazy.try_metadata_snapshot().unwrap();
        assert!(before.allocation().is_none());
        {
            let _cold = Cold::new();
            let metadata =
                PreparedCaptureTensor::validate_source_geometry(&lazy, host(&admitted).geometry())
                    .unwrap();
            assert_eq!(metadata, before);
            assert!(matches!(
                PreparedCaptureTensor::new(&lazy, host(&admitted)),
                Err(CaptureTensorNativeError::UnsettledSource)
            ));
            assert!(matches!(
                PreparedCaptureTensor::validate_source_geometry(&lazy, host(&wrong).geometry()),
                Err(CaptureTensorNativeError::ShapeMismatch)
            ));
            assert!(matches!(
                PreparedCaptureTensor::validate_source_geometry(
                    &f64_source,
                    host(&admitted).geometry()
                ),
                Err(CaptureTensorNativeError::UnsupportedDtype(Dtype::Float64))
            ));
            assert_eq!(lazy.try_metadata_snapshot().unwrap(), before);
            assert_eq!(HOUSEKEEPING.get(), 0);
        }
        // Only the caller's subsequent explicit settlement makes the original
        // settled-source constructor available. No transfer authority is minted.
        lazy.evaluated().unwrap();
        let leaf = PreparedCaptureTensor::new(&lazy, host(&admitted)).unwrap();
        assert!(lazy.try_metadata_snapshot().unwrap().allocation().is_some());
        drop(leaf);
    }
}
