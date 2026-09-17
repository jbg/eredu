use super::*;
use crate::working_memory::*;
use eredu_core::{cache::LayerCachePolicy, *};
use std::{
    cell::{Cell, RefCell},
    num::NonZeroU8,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};
thread_local! {
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
    static PANIC_AT_RECORD: Cell<Option<usize>> = const { Cell::new(None) };
    static FENCE_AFTER_RECORD: RefCell<Option<WorkingMemoryFundingScope>> = const { RefCell::new(None) };
}
pub(super) fn before_allocate() {
    ALLOCATIONS.set(ALLOCATIONS.get() + 1);
}
pub(super) fn after_record(index: usize) {
    if index == 1 {
        let scope = FENCE_AFTER_RECORD.with(|slot| slot.borrow_mut().take());
        drop(scope);
    }
    if PANIC_AT_RECORD.get() == Some(index) {
        PANIC_AT_RECORD.set(None);
        panic!("injected after actual frame record buffers");
    }
}
fn admitted() -> AdmittedCapturePlan {
    let point = ObservationPoint {
        path: "block.output".into(),
        node_id: "block".into(),
        meaning: "actual activation".into(),
        value_type: ObservationValueType::Tensor,
        dtype: ObservationDtype::Floating,
        axes: Some(vec![
            TensorAxis {
                name: "sequence".into(),
                dimension: SymbolicDimension::Sequence,
            },
            TensorAxis {
                name: "width".into(),
                dimension: SymbolicDimension::Known(4),
            },
        ]),
        prefill: true,
        decode: true,
        requirements: vec![ObservationRequirement::ActivationHooks],
        position: ObservationPosition::BeforeIntervention,
        retained_bytes: None,
        host_bytes: None,
    };
    let support = ObservationSupportReport {
        schema_version: 1,
        capture: Default::default(),
        points: vec![ObservationSupport {
            path: point.path.clone(),
            prefill: ObservationSupportStatus::Supported,
            decode: ObservationSupportStatus::Supported,
            floating_to_f32: true,
        }],
    };
    let catalog = ObservationCatalog {
        schema_version: 1,
        points: vec![point],
        completeness: DescriptionCompleteness::Complete,
    };
    let all = unlimited();
    let selections = (0..5)
        .map(|index| CaptureSelection {
            id: format!("selection-{index}-é"),
            path: "block.output".into(),
            schedule: CaptureSchedule {
                first_prediction: if index == 2 { 3 } else { 0 },
                ..Default::default()
            },
            slices: if index == 1 || index == 2 {
                vec![CaptureSlice {
                    axis: "width".into(),
                    start: 1,
                    end: 4,
                    stride: 2,
                }]
            } else {
                vec![]
            },
            transform: if index == 1 {
                CaptureTransform::Preview { max_elements: 5 }
            } else if index == 2 {
                CaptureTransform::Slice
            } else {
                CaptureTransform::FullTensor
            },
        })
        .collect();
    CapturePlan {
        schema_version: 1,
        selections,
        limits: CaptureLimits {
            per_step: all,
            cumulative: all,
            physical_native_bytes: None,
            on_limit: CaptureLimitPolicy::Fail,
        },
    }
    .admit(
        &catalog,
        &support,
        &CaptureCapabilities {
            transformations: vec![
                CaptureTransformKind::FullTensor,
                CaptureTransformKind::Slice,
                CaptureTransformKind::Preview,
            ],
            max_histogram_bins: 1,
            physical_native_limit: false,
            conditions: vec![],
        },
        CaptureRequestShape {
            batch: 1,
            prompt_tokens: 3,
            max_predictions: 5,
        },
    )
    .unwrap()
}
fn plan(source: &AdmittedCapturePlan) -> CaptureStepHostPlan<'_> {
    CaptureStepHostPlan::prepare(source, CapturePhase::Prefill, 0, None).unwrap()
}
fn unlimited() -> CaptureUsage {
    CaptureUsage {
        captures: u64::MAX,
        retained_bytes: u64::MAX,
        host_bytes: u64::MAX,
        encoded_bytes: u64::MAX,
    }
}
fn sum(records: &[CaptureRecord]) -> CaptureUsage {
    records.iter().fold(CaptureUsage::default(), |n, r| {
        n.checked_add(r.charged).unwrap()
    })
}
fn tensor(shape: Vec<usize>) -> SharedTensorObservation {
    let count: usize = shape.iter().product();
    SharedTensorObservation::retain(
        TensorObservation::new(
            shape,
            TensorObservationData::F32((0..count).map(|n| n as f32 + 0.25).collect()),
        )
        .unwrap(),
        (),
    )
}
fn ledger(pool: &WorkingMemoryPool) -> (u64, u64, u64, usize, usize, usize) {
    let usage = pool.0.usage.lock().unwrap();
    (
        usage.reserved + usage.registered,
        usage.peak,
        usage.funding.values().map(|s| s.host_held).sum(),
        usage.funding.values().map(|s| s.scopes).sum(),
        usage.reservations,
        usage.funding.len(),
    )
}

fn fresh(
    pool: &WorkingMemoryPool,
    bytes: u64,
) -> (WorkingMemoryReservation, WorkingMemoryFundingRun) {
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
    )
    .unwrap()
    .into_funding()
    .unwrap()
}

#[test]
fn exact_and_one_short_use_actual_plan_before_any_frame_buffers() {
    let source = admitted();
    let p = plan(&source).initialization_peak_bytes();
    for bytes in [p - 1, p] {
        let pool = WorkingMemoryPool::new(2 * p, 0).unwrap();
        let (reservation, run) = fresh(&pool, bytes);
        let before = ledger(&pool);
        let allocations = ALLOCATIONS.get();
        let result = run.prepare_capture_step(&reservation, plan(&source));
        if bytes < p {
            assert!(
                matches!(result,Err(CaptureStepError::Memory(WorkingMemoryError::BudgetExceeded {required_bytes,available_bytes})) if required_bytes==p && available_bytes==p-1)
            );
            assert_eq!(ledger(&pool), before);
            assert_eq!(ALLOCATIONS.get(), allocations);
        } else {
            let builder = result.unwrap();
            assert_eq!(builder.len(), 5);
            assert!(matches!(
                builder.records()[2].outcome,
                CaptureOutcome::Skipped {
                    reason: CaptureSkipReason::Schedule
                }
            ));
            assert!(matches!(
                builder.records()[3].outcome,
                CaptureOutcome::Missing
            ));
            assert_eq!(ALLOCATIONS.get(), allocations + 1);
            assert_eq!(ledger(&pool).2, p);
            drop(builder);
            assert_eq!(ledger(&pool), before);
        }
        drop((run, reservation));
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
}

#[test]
fn fixed_capacities_pointers_preview_and_utf8_failures_survive_finish() {
    let source = admitted();
    let host_plan = plan(&source);
    let allocated = host_plan.allocated_payload_bytes();
    let p = host_plan.initialization_peak_bytes();
    let pool = WorkingMemoryPool::new(p, 0).unwrap();
    let (reservation, run) = fresh(&pool, p);
    let mut b = run.prepare_capture_step(&reservation, host_plan).unwrap();
    let actual = size_of::<CapturedStep>()
        + size_of::<Vec<RecordBuffers>>()
        + b.frame.records.capacity() * size_of::<CaptureRecord>()
        + b.buffers.capacity() * size_of::<RecordBuffers>()
        + b.frame
            .records
            .iter()
            .map(|r| r.selection_id.capacity() + r.path.capacity() + r.node_id.capacity())
            .sum::<usize>()
        + b.buffers
            .iter()
            .map(|v| {
                (v.source.capacity() + v.selected.capacity()) * size_of::<u64>()
                    + v.diagnostic.capacity()
            })
            .sum::<usize>();
    // The original plan also covers the final shared frame and custody Box,
    // which finish constructs after these initial frame/sidecar buffers.
    let shared_controls =
        UnpublishedCapturedStep::retained_control_bytes::<CaptureFrameCustody>().unwrap();
    assert!(shared_controls > 0);
    assert_eq!(
        (actual as u64).checked_add(shared_controls).unwrap(),
        allocated
    );
    let original_ledger = ledger(&pool);
    assert_eq!(original_ledger.2, p);
    let records = b.records().as_ptr();
    let id = b.records()[0].selection_id.as_ptr();
    let source_ptr = b.buffers[0].source.as_ptr();
    let selected_ptr = b.buffers[1].selected.as_ptr();
    let diagnostic_ptr = b.buffers[4].diagnostic.as_ptr();
    let values = tensor(vec![3, 4]);
    let values_alias = values.clone();
    b.record_tensor(
        0,
        TensorDtype::F16,
        values,
        CaptureUsage {
            captures: 1,
            ..Default::default()
        },
    )
    .unwrap();
    b.record_tensor(
        1,
        TensorDtype::Bf16,
        tensor(vec![5]),
        CaptureUsage {
            captures: 1,
            ..Default::default()
        },
    )
    .unwrap();
    let text = "é".repeat(127) + "€suffix";
    b.record_failure(
        4,
        CaptureFailureReason::Native,
        &text,
        Some(TensorDtype::F32),
        CaptureUsage::default(),
    )
    .unwrap();
    assert!(matches!(
        b.records()[1].outcome,
        CaptureOutcome::Truncated {
            available_elements: 6,
            emitted_elements: 5
        }
    ));
    assert_eq!(b.records()[1].selected_shape.as_deref(), Some(&[3, 2][..]));
    assert_eq!(
        b.records()[1]
            .payload
            .as_ref()
            .unwrap()
            .as_tensor()
            .unwrap()
            .shape(),
        &[5]
    );
    let CaptureOutcome::Failed { message, .. } = &b.records()[4].outcome else {
        panic!()
    };
    assert_eq!(message.len(), 254);
    assert_eq!(message.capacity(), 256);
    assert_eq!(message.as_ptr(), diagnostic_ptr);
    assert!(
        b.record_tensor(
            0,
            TensorDtype::F32,
            tensor(vec![3, 4]),
            CaptureUsage::default()
        )
        .is_err()
    );
    let usage = sum(b.records());
    let mut logical = CaptureLedger::new(&source);
    logical
        .reserve(CaptureUsage {
            host_bytes: 19,
            ..Default::default()
        })
        .unwrap();
    logical.begin_step();
    logical.reserve(usage).unwrap();
    logical
        .reserve(CaptureUsage {
            host_bytes: 32,
            ..Default::default()
        })
        .unwrap();
    let reported_step = logical.step();
    let reported_total = logical.total();
    let shared = b
        .finish(
            CaptureStepOutcome::Aborted,
            reported_step,
            reported_total,
            0.5,
        )
        .unwrap();
    assert_eq!(ledger(&pool), original_ledger);
    assert_eq!(shared.records().as_ptr(), records);
    assert_eq!(shared.records()[0].selection_id.as_ptr(), id);
    assert_eq!(
        shared.records()[0].source_shape.as_ref().unwrap().as_ptr(),
        source_ptr
    );
    assert_eq!(
        shared.records()[1]
            .selected_shape
            .as_ref()
            .unwrap()
            .as_ptr(),
        selected_ptr
    );
    let CapturePayload::SharedTensor(value) = shared.records()[0].payload.as_ref().unwrap() else {
        panic!()
    };
    assert!(value.same_storage(&values_alias));
    assert_eq!(shared.step_usage(), reported_step);
    assert_eq!(shared.cumulative_usage(), reported_total);
    assert_eq!(sum(shared.records()), usage);
    let decoded: CapturedStep =
        serde_json::from_value(serde_json::to_value(&shared).unwrap()).unwrap();
    assert_eq!(shared.as_step(), &decoded);
    let alias = shared.clone();
    drop((source, shared, run, reservation));
    assert_eq!(pool.used_bytes().unwrap(), p);
    assert_eq!(alias.records().as_ptr(), records);
    drop(alias);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn original_parent_frame_tensor_and_native_pressure_never_double_spend() {
    let source = admitted();
    let p = plan(&source).initialization_peak_bytes();
    let tensor_plan = CaptureTensorHostPlan::prepare(
        CaptureTensorGeometry::prepare(&source, 0, CapturePhase::Prefill, 0, None).unwrap(),
    )
    .unwrap();
    let h = tensor_plan.initialization_peak_bytes();
    let n = 61;
    let total = p + h + n;
    let pool = WorkingMemoryPool::new(total, 0).unwrap();
    let (reservation, run) = fresh(&pool, total);
    let native = run.scope().unwrap();
    let mut b = run
        .prepare_capture_step(&reservation, plan(&source))
        .unwrap();
    let mut values = run
        .prepare_capture_tensor(&reservation, tensor_plan)
        .unwrap();
    for i in 0..values.len() {
        values.push_f32(i as f32 + 0.5).unwrap();
    }
    let values = values.finish().unwrap();
    let escaped = values.clone();
    b.record_tensor(0, TensorDtype::F32, values, CaptureUsage::default())
        .unwrap();
    let before = ledger(&pool);
    assert_eq!(before.2, p + h);
    assert!(matches!(
        native.adopt_storage_individually([(17u32, n + 1)]),
        Err(WorkingMemoryError::BudgetExceeded { .. })
    ));
    assert_eq!(ledger(&pool), before);
    let roots = native.adopt_storage_individually([(17u32, n)]).unwrap();
    let frame = b
        .finish(CaptureStepOutcome::Committed, unlimited(), unlimited(), 0.0)
        .unwrap();
    drop((run, reservation));
    native.certify().unwrap();
    drop(frame);
    assert_eq!(ledger(&pool).2, h);
    // The native work and run are closed; only the escaped tensor's exact
    // host hold and independently registered native allocation remain.
    assert_eq!(pool.used_bytes().unwrap(), h + n);
    drop(escaped);
    assert_eq!(pool.used_bytes().unwrap(), n);
    drop(roots);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn wrong_reservation_closed_and_quarantined_parents_reject_without_record_mutation() {
    for failure in 0..3 {
        let source = admitted();
        let p = plan(&source).initialization_peak_bytes();
        let pool = WorkingMemoryPool::new(3 * p, 0).unwrap();
        let foreign = WorkingMemoryPool::new(p, 0).unwrap();
        let (r, run) = fresh(&pool, p);
        let (other, other_run) = fresh(&pool, p);
        let (foreign_r, foreign_run) = fresh(&foreign, p);
        for wrong in [&other, &foreign_r] {
            let before = ledger(&pool);
            let allocations = ALLOCATIONS.get();
            assert!(matches!(
                run.prepare_capture_step(wrong, plan(&source)),
                Err(CaptureStepError::Memory(
                    WorkingMemoryError::IdentityMismatch
                ))
            ));
            assert_eq!(ledger(&pool), before);
            assert_eq!(ALLOCATIONS.get(), allocations);
        }
        let mut b = run.prepare_capture_step(&r, plan(&source)).unwrap();
        let mut run = Some(run);
        let mut r = Some(r);
        match failure {
            0 => drop(run.take()),
            1 => drop(r.take()),
            _ => drop(run.as_ref().unwrap().scope().unwrap()),
        };
        let before = ledger(&pool);
        assert!(matches!(
            b.record_failure(
                0,
                CaptureFailureReason::Native,
                "failure",
                None,
                CaptureUsage::default()
            ),
            Err(CaptureStepError::Memory(
                WorkingMemoryError::ExecutionFenced
            ))
        ));
        assert!(matches!(b.records()[0].outcome, CaptureOutcome::Missing));
        assert_eq!(b.buffers[0].diagnostic.len(), 0);
        assert_eq!(ledger(&pool), before);
        let error = b
            .finish(CaptureStepOutcome::Aborted, unlimited(), unlimited(), 0.0)
            .unwrap_err();
        assert!(matches!(
            error.error(),
            CaptureStepError::Memory(WorkingMemoryError::ExecutionFenced)
        ));
        drop(error);
        assert_eq!(ledger(&pool).2, 0);
        drop((r, run, other, other_run, foreign_r, foreign_run));
        assert_eq!(foreign.used_bytes().unwrap(), 0);
        if failure < 2 {
            assert_eq!(pool.used_bytes().unwrap(), 0)
        } else {
            assert_eq!(pool.used_bytes().unwrap(), p)
        }
    }
}

#[test]
fn invalid_tensor_finish_and_skips_preserve_fixed_owner_and_original_outcomes() {
    let source = admitted();
    let p = plan(&source).initialization_peak_bytes();
    let pool = WorkingMemoryPool::new(p, 0).unwrap();
    let (r, run) = fresh(&pool, p);
    let mut b = run.prepare_capture_step(&r, plan(&source)).unwrap();
    let pointers = (b.records().as_ptr(), b.buffers[0].source.as_ptr());
    assert!(matches!(
        b.record_tensor(
            0,
            TensorDtype::F32,
            tensor(vec![12]),
            CaptureUsage::default()
        ),
        Err(CaptureStepError::TensorMismatch { .. })
    ));
    assert!(matches!(
        b.record_tensor(
            0,
            TensorDtype::I32,
            tensor(vec![3, 4]),
            CaptureUsage::default()
        ),
        Err(CaptureStepError::TensorMismatch { .. })
    ));
    assert!(matches!(
        b.record_failure(
            2,
            CaptureFailureReason::Native,
            "must stay skipped",
            None,
            CaptureUsage::default()
        ),
        Err(CaptureStepError::RecordNotPending { .. })
    ));
    assert!(matches!(
        b.record_skip(
            99,
            CaptureSkipReason::NotInvoked,
            None,
            CaptureUsage::default()
        ),
        Err(CaptureStepError::RecordNotPending { .. })
    ));
    b.record_skip(
        0,
        CaptureSkipReason::Limit {
            budget: CaptureBudget::Host,
            cumulative: true,
        },
        Some(TensorDtype::F16),
        CaptureUsage::default(),
    )
    .unwrap();
    assert_eq!(
        b.records()[0].source_shape.as_ref().unwrap().as_ptr(),
        pointers.1
    );
    let error = b
        .finish(
            CaptureStepOutcome::Committed,
            CaptureUsage::default(),
            unlimited(),
            0.0,
        )
        .unwrap_err();
    assert!(matches!(error.error(), CaptureStepError::InvalidCompletion));
    let b = error.into_builder();
    assert_eq!(b.records().as_ptr(), pointers.0);
    let error = b
        .finish(
            CaptureStepOutcome::Committed,
            unlimited(),
            unlimited(),
            f64::NAN,
        )
        .unwrap_err();
    let frame = error
        .into_builder()
        .finish(CaptureStepOutcome::Aborted, unlimited(), unlimited(), 0.0)
        .unwrap();
    assert!(matches!(
        frame.records()[3].outcome,
        CaptureOutcome::Missing
    ));
    drop((frame, r, run));
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn partial_construction_unwind_releases_only_host_hold_and_never_native_scope() {
    let source = admitted();
    let p = plan(&source).initialization_peak_bytes();
    let pool = WorkingMemoryPool::new(p + 9, 0).unwrap();
    let (r, run) = fresh(&pool, p + 9);
    let native = run.scope().unwrap();
    let before = ledger(&pool);
    PANIC_AT_RECORD.set(Some(1));
    assert!(
        catch_unwind(AssertUnwindSafe(
            || run.prepare_capture_step(&r, plan(&source))
        ))
        .is_err()
    );
    assert_eq!(ledger(&pool), before);
    let mut b = run.prepare_capture_step(&r, plan(&source)).unwrap();
    struct Retired {
        count: Arc<AtomicUsize>,
        pool: WorkingMemoryPool,
        held: u64,
    }
    impl Drop for Retired {
        fn drop(&mut self) {
            let usage = self
                .pool
                .0
                .usage
                .try_lock()
                .expect("payload destructor outside Usage lock");
            assert_eq!(
                usage
                    .funding
                    .values()
                    .map(|state| state.host_held)
                    .sum::<u64>(),
                self.held
            );
            self.count.fetch_add(1, Ordering::SeqCst);
        }
    }
    let retired = Arc::new(AtomicUsize::new(0));
    let value = SharedTensorObservation::retain(
        TensorObservation::new(vec![3, 4], TensorObservationData::F32(vec![1.0; 12])).unwrap(),
        Retired {
            count: retired.clone(),
            pool: pool.clone(),
            held: p,
        },
    );
    b.record_tensor(0, TensorDtype::F32, value, CaptureUsage::default())
        .unwrap();
    assert!(
        catch_unwind(AssertUnwindSafe(move || {
            let _b = b;
            panic!("consumer unwind")
        }))
        .is_err()
    );
    assert_eq!(retired.load(Ordering::SeqCst), 1);
    assert_eq!(ledger(&pool), before);
    let roots = native.adopt_storage_individually([(1u32, 9)]).unwrap();
    drop((run, r));
    assert_eq!(pool.used_bytes().unwrap(), p + 9);
    drop(native);
    drop(roots);
    assert_eq!(pool.used_bytes().unwrap(), p + 9);
    assert!(matches!(
        pool.acquire_unquoted(),
        Err(WorkingMemoryError::ReservedWorkActive)
    ));
}

fn readmit(original: CapturePlan, points: Vec<ObservationPoint>) -> AdmittedCapturePlan {
    readmit_with_invocation(original, points, None)
}
fn readmit_with_invocation(
    mut original: CapturePlan,
    points: Vec<ObservationPoint>,
    bounds: Option<CaptureInvocationBounds>,
) -> AdmittedCapturePlan {
    original.limits.per_step = unlimited();
    original.limits.cumulative = unlimited();
    let support = ObservationSupportReport {
        schema_version: 1,
        capture: Default::default(),
        points: points
            .iter()
            .map(|point| ObservationSupport {
                path: point.path.clone(),
                prefill: ObservationSupportStatus::Supported,
                decode: ObservationSupportStatus::Supported,
                floating_to_f32: true,
            })
            .collect(),
    };
    let catalog = ObservationCatalog {
        schema_version: 1,
        points,
        completeness: DescriptionCompleteness::Complete,
    };
    let capabilities = CaptureCapabilities {
        transformations: vec![
            CaptureTransformKind::FullTensor,
            CaptureTransformKind::Slice,
            CaptureTransformKind::Preview,
            CaptureTransformKind::Summary,
            CaptureTransformKind::Histogram,
        ],
        max_histogram_bins: 1,
        physical_native_limit: false,
        conditions: vec![],
    };
    match bounds {
        Some(bounds) => original.admit_invocations(&catalog, &support, &capabilities, bounds),
        None => original.admit(
            &catalog,
            &support,
            &capabilities,
            CaptureRequestShape {
                batch: 1,
                prompt_tokens: 3,
                max_predictions: 5,
            },
        ),
    }
    .unwrap()
}
#[test]
fn empty_zero_geometry_and_unsupported_unknown_overflow_remain_explicit() {
    let source = admitted();
    let mut raw = source.plan().clone();
    raw.selections.clear();
    let empty = readmit(raw, source.points()[..1].to_vec());
    let p = plan(&empty).initialization_peak_bytes();
    let pool = WorkingMemoryPool::new(p, 0).unwrap();
    let (r, run) = fresh(&pool, p);
    let b = run.prepare_capture_step(&r, plan(&empty)).unwrap();
    assert!(b.is_empty());
    assert_eq!(b.frame.records.capacity(), 0);
    assert_eq!(b.buffers.capacity(), 0);
    let frame = b
        .finish(
            CaptureStepOutcome::Committed,
            CaptureUsage::default(),
            CaptureUsage::default(),
            0.0,
        )
        .unwrap();
    assert!(frame.records().is_empty());
    drop((frame, r, run));
    assert_eq!(pool.used_bytes().unwrap(), 0);
    for kind in 0..3 {
        let mut raw = source.plan().clone();
        raw.selections.truncate(1);
        let mut points = source.points()[..1].to_vec();
        match kind {
            0 => {
                raw.selections[0].transform = CaptureTransform::Histogram {
                    edges: vec![0.0, 1.0],
                }
            }
            1 => points[0].axes.as_mut().unwrap()[1].dimension = SymbolicDimension::Unknown,
            _ => points[0].axes.as_mut().unwrap()[1].dimension = SymbolicDimension::Known(0),
        };
        let actual = readmit(raw, points);
        let before = ALLOCATIONS.get();
        let result = CaptureStepHostPlan::prepare(&actual, CapturePhase::Prefill, 0, None);
        match kind {
            0 => assert!(
                result.is_ok(),
                "admitted Histogram has exact paid frame geometry"
            ),
            1 => assert!(matches!(
                result,
                Err(CaptureStepError::Geometry(
                    CaptureTensorGeometryError::UnknownShape
                ))
            )),
            _ => {
                let host = result.unwrap();
                let p = host.initialization_peak_bytes();
                let pool = WorkingMemoryPool::new(p, 0).unwrap();
                let (r, run) = fresh(&pool, p);
                let mut b = run.prepare_capture_step(&r, host).unwrap();
                b.record_tensor(
                    0,
                    TensorDtype::F32,
                    tensor(vec![3, 0]),
                    CaptureUsage::default(),
                )
                .unwrap();
                let frame = b
                    .finish(CaptureStepOutcome::Committed, unlimited(), unlimited(), 0.0)
                    .unwrap();
                assert_eq!(
                    frame.records()[0].source_shape.as_deref(),
                    Some(&[3, 0][..])
                );
                drop((frame, r, run));
                assert_eq!(pool.used_bytes().unwrap(), 0);
            }
        }
        if kind < 2 {
            assert_eq!(ALLOCATIONS.get(), before);
        }
    }
    let invocation = CaptureInvocationShape {
        batch: 1,
        sequence: 2,
        context: Some(5),
    };
    let invoked = readmit_with_invocation(
        source.plan().clone(),
        source.points()[..1].to_vec(),
        Some(CaptureInvocationBounds {
            batch: 1,
            max_sequence: 3,
            max_context: Some(8),
            max_predictions: 5,
        }),
    );
    assert!(CaptureStepHostPlan::prepare(&invoked, CapturePhase::Decode, 2, None).is_err());
    assert!(
        CaptureStepHostPlan::prepare(&source, CapturePhase::Decode, 2, Some(invocation)).is_err()
    );
    let host =
        CaptureStepHostPlan::prepare(&invoked, CapturePhase::Decode, 2, Some(invocation)).unwrap();
    let p = host.initialization_peak_bytes();
    let pool = WorkingMemoryPool::new(p, 0).unwrap();
    let (r, run) = fresh(&pool, p);
    let mut b = run.prepare_capture_step(&r, host).unwrap();
    b.record_tensor(
        0,
        TensorDtype::Bf16,
        tensor(vec![2, 4]),
        CaptureUsage::default(),
    )
    .unwrap();
    let frame = b
        .finish(CaptureStepOutcome::Committed, unlimited(), unlimited(), 0.0)
        .unwrap();
    assert_eq!(frame.invocation(), Some(invocation));
    assert_eq!(frame.phase(), CapturePhase::Decode);
    assert_eq!(frame.prediction_index(), 2);
    assert_eq!(
        frame.records()[0].source_shape.as_deref(),
        Some(&[2, 4][..])
    );
    drop((frame, r, run));
    assert_eq!(pool.used_bytes().unwrap(), 0);
    assert!(plan::extent(usize::MAX, 2).is_err());
    assert!(plan::extent((isize::MAX as usize) / 8 + 1, 8).is_err());
    assert!(matches!(
        CaptureStepHostPlan::prepare(&source, CapturePhase::Decode, 5, None),
        Err(CaptureStepError::Capture(_))
            | Err(CaptureStepError::Geometry(
                CaptureTensorGeometryError::Inactive
            ))
    ));
}

#[test]
fn unhealthy_account_before_construction_and_poison_cleanup_never_grant_work() {
    let source = admitted();
    let p = plan(&source).initialization_peak_bytes();
    let pool = WorkingMemoryPool::new(p, 0).unwrap();
    let (r, run) = fresh(&pool, p);
    drop(run.scope().unwrap());
    let before = ledger(&pool);
    let allocations = ALLOCATIONS.get();
    assert!(matches!(
        run.prepare_capture_step(&r, plan(&source)),
        Err(CaptureStepError::Memory(
            WorkingMemoryError::ExecutionFenced
        ))
    ));
    assert_eq!(ledger(&pool), before);
    assert_eq!(ALLOCATIONS.get(), allocations);
    drop((run, r));
    assert_eq!(pool.used_bytes().unwrap(), p);
    let pool = WorkingMemoryPool::new(p, 0).unwrap();
    let (r, run) = fresh(&pool, p);
    FENCE_AFTER_RECORD.with(|slot| *slot.borrow_mut() = Some(run.scope().unwrap()));
    let allocations = ALLOCATIONS.get();
    assert!(matches!(
        run.prepare_capture_step(&r, plan(&source)),
        Err(CaptureStepError::Memory(
            WorkingMemoryError::ExecutionFenced
        ))
    ));
    assert_eq!(ALLOCATIONS.get(), allocations + 1);
    assert_eq!(ledger(&pool).2, 0);
    drop((r, run));
    assert_eq!(pool.used_bytes().unwrap(), p);

    let pool = WorkingMemoryPool::new(p, 0).unwrap();
    let (r, run) = fresh(&pool, p);
    let b = run.prepare_capture_step(&r, plan(&source)).unwrap();
    let _ = catch_unwind(AssertUnwindSafe(|| {
        let _lock = pool.0.usage.lock().unwrap();
        panic!("poison test")
    }));
    let error = b
        .finish(CaptureStepOutcome::Aborted, unlimited(), unlimited(), 0.0)
        .unwrap_err();
    assert!(matches!(
        error.error(),
        CaptureStepError::Memory(WorkingMemoryError::Poisoned)
    ));
    drop(error);
    {
        let usage = pool.0.usage.lock().unwrap_err().into_inner();
        assert_eq!(usage.funding.values().map(|s| s.host_held).sum::<u64>(), 0);
    }
    drop((r, run));
    assert!(matches!(
        pool.used_bytes(),
        Err(WorkingMemoryError::Poisoned)
    ));
}
