use super::*;
use eredu_core::{capture::*, *};
use std::{
    cell::Cell,
    panic::{AssertUnwindSafe, catch_unwind},
};

thread_local! {
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
    static PANIC_BEFORE_BUFFERS: Cell<bool> = const { Cell::new(false) };
}
pub(super) fn before_allocate() {
    ALLOCATIONS.set(ALLOCATIONS.get() + 1);
    assert!(
        !PANIC_BEFORE_BUFFERS.replace(false),
        "injected before host buffers"
    );
}

fn admitted(
    axes: Vec<SymbolicDimension>,
    transform: CaptureTransform,
    slices: Vec<CaptureSlice>,
) -> AdmittedCapturePlan {
    let point = ObservationPoint {
        path: "block.output".into(),
        node_id: "block".into(),
        meaning: "actual activation".into(),
        value_type: ObservationValueType::Tensor,
        dtype: ObservationDtype::Floating,
        axes: Some(
            axes.into_iter()
                .enumerate()
                .map(|(index, dimension)| TensorAxis {
                    name: format!("axis{index}"),
                    dimension,
                })
                .collect(),
        ),
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
    let usage = CaptureUsage {
        captures: 10,
        retained_bytes: u64::MAX,
        host_bytes: u64::MAX,
        encoded_bytes: u64::MAX,
    };
    CapturePlan {
        schema_version: 1,
        selections: vec![CaptureSelection {
            id: "tensor".into(),
            path: "block.output".into(),
            schedule: CaptureSchedule::default(),
            slices,
            transform,
        }],
        limits: CaptureLimits {
            per_step: usage,
            cumulative: usage,
            on_limit: CaptureLimitPolicy::Fail,
        },
    }
    .admit(
        &catalog,
        &support,
        &CaptureCapabilities {
            transformations: vec![
                CaptureTransformKind::Preview,
                CaptureTransformKind::Slice,
                CaptureTransformKind::FullTensor,
                CaptureTransformKind::Summary,
            ],
            max_histogram_bins: 0,
            conditions: vec![],
        },
        CaptureRequestShape {
            batch: 1,
            prompt_tokens: 3,
            max_predictions: 4,
        },
    )
    .unwrap()
}
fn plan(source: &AdmittedCapturePlan) -> CaptureTensorHostPlan<'_> {
    CaptureTensorHostPlan::prepare(
        CaptureTensorGeometry::prepare(source, 0, CapturePhase::Prefill, 0, None).unwrap(),
    )
    .unwrap()
}
fn required_charge(source: &AdmittedCapturePlan) -> u64 {
    let ledger = crate::working_memory::memory_fixture::host_ledger(u64::MAX, 0).unwrap();
    ledger
        .capture_tensor_requirements(&plan(source), &CaptureTensorLimits::default())
        .unwrap()
        .get(ledger.topology().host_domain())
        .unwrap()
        .total()
        .unwrap()
}
fn ordinary() -> AdmittedCapturePlan {
    admitted(
        vec![SymbolicDimension::Sequence, SymbolicDimension::Known(4)],
        CaptureTransform::FullTensor,
        vec![],
    )
}
fn fill(mut builder: PreparedCaptureTensor<'_>) -> SharedTensorObservation {
    for index in 0..builder.len() {
        builder.push_f32(index as f32 + 0.25).unwrap();
    }
    builder.finish().unwrap()
}

#[test]
fn exact_and_one_short_reject_before_buffers_and_preserve_all_counters() {
    let source = ordinary();
    let payload = plan(&source).initialization_peak_bytes();
    let required = required_charge(&source);
    let pool = crate::working_memory::memory_fixture::host_ledger(required, 0).unwrap();
    let limits = CaptureTensorLimits::new(crate::working_memory::memory_fixture::host_limits(
        required - 1,
    ));
    let before = (
        pool.payload_used_bytes().unwrap(),
        pool.payload_peak_bytes().unwrap(),
        ALLOCATIONS.get(),
    );
    let error = pool
        .prepare_capture_tensor(plan(&source), limits.clone())
        .unwrap_err();
    assert!(
        matches!(error, CaptureTensorConstructionError::Memory(WorkingMemoryError::Domain(eredu_core::MemoryDomainError::BudgetExceeded { requested_bytes, limit_bytes, existing_bytes, .. })) if requested_bytes == required && limit_bytes - existing_bytes == required - 1)
    );
    assert_eq!(
        (
            pool.payload_used_bytes().unwrap(),
            pool.payload_peak_bytes().unwrap(),
            ALLOCATIONS.get()
        ),
        before
    );
    let value = fill(
        pool.prepare_capture_tensor(
            plan(&source),
            CaptureTensorLimits::new(crate::working_memory::memory_fixture::host_limits(required)),
        )
        .unwrap(),
    );
    assert_eq!(pool.payload_used_bytes().unwrap(), payload);
    drop(value);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

#[test]
fn actual_nonzero_payload_aliases_outlive_source_and_keep_exact_host_account() {
    let source = ordinary();
    let retained = plan(&source).retained_payload_bytes();
    let payload = plan(&source).initialization_peak_bytes();
    let required = required_charge(&source);
    let pool = crate::working_memory::memory_fixture::host_ledger(2 * required, 0).unwrap();
    let value = fill(
        pool.prepare_capture_tensor(
            plan(&source),
            CaptureTensorLimits::new(crate::working_memory::memory_fixture::host_limits(
                2 * required,
            )),
        )
        .unwrap(),
    );
    let alias = value.clone();
    let independent = fill(
        pool.prepare_capture_tensor(
            plan(&source),
            CaptureTensorLimits::new(crate::working_memory::memory_fixture::host_limits(
                2 * required,
            )),
        )
        .unwrap(),
    );
    assert_eq!(value, independent);
    assert!(!value.same_storage(&independent));
    assert!(value.same_storage(&alias));
    assert_eq!(value.shape(), &[3, 4]);
    assert_eq!(value.retained_payload_bytes(), Some(retained));
    match (value.data(), alias.data()) {
        (TensorObservationData::F32(a), TensorObservationData::F32(b)) => {
            assert_eq!(a.as_ptr(), b.as_ptr());
            assert_eq!(a.capacity(), 12);
            assert_eq!(a, &(0..12).map(|n| n as f32 + 0.25).collect::<Vec<_>>());
        }
        _ => panic!("F32 destination"),
    }
    let wire = serde_json::to_value(&value).unwrap();
    let raw: TensorObservation = serde_json::from_value(wire.clone()).unwrap();
    assert_eq!(&raw, value.as_observation());
    assert_eq!(serde_json::to_value(&raw).unwrap(), wire);
    drop((source, raw, value, independent));
    assert_eq!(pool.payload_used_bytes().unwrap(), payload);
    assert!(matches!(
        pool.acquire_unquoted(),
        Err(WorkingMemoryError::ReservedWorkActive)
    ));
    drop(alias);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
    crate::working_memory::memory_fixture::assert_unquoted_idle(&pool);
}

#[test]
fn partial_finish_full_push_and_unwind_keep_the_same_buffers_and_custody() {
    let source = ordinary();
    let payload = plan(&source).initialization_peak_bytes();
    let required = required_charge(&source);
    let pool = crate::working_memory::memory_fixture::host_ledger(required, 0).unwrap();
    let mut builder = pool
        .prepare_capture_tensor(
            plan(&source),
            CaptureTensorLimits::new(crate::working_memory::memory_fixture::host_limits(required)),
        )
        .unwrap();
    let pointer = builder.data.as_ptr();
    builder.push_f32(7.0).unwrap();
    let error = builder.finish().unwrap_err();
    assert_eq!(pool.payload_used_bytes().unwrap(), payload);
    let mut builder = error.into_builder().unwrap();
    assert_eq!(builder.data.as_ptr(), pointer);
    for _ in 1..builder.len() {
        builder.push_f32(-2.0).unwrap();
    }
    assert_eq!(
        builder.push_f32(99.0),
        Err(WorkingMemoryError::IdentityMismatch)
    );
    assert_eq!(builder.data.as_ptr(), pointer);
    let value = builder.finish().unwrap();
    match value.data() {
        TensorObservationData::F32(v) => assert_eq!(v.as_ptr(), pointer),
        _ => unreachable!(),
    }
    drop(value);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
    for before_buffers in [true, false] {
        PANIC_BEFORE_BUFFERS.set(before_buffers);
        let result = catch_unwind(AssertUnwindSafe(|| {
            let mut partial = pool
                .prepare_capture_tensor(
                    plan(&source),
                    CaptureTensorLimits::new(crate::working_memory::memory_fixture::host_limits(
                        required,
                    )),
                )
                .unwrap();
            partial.push_f32(5.0).unwrap();
            panic!("partial destination failure");
        }));
        assert!(result.is_err());
        assert_eq!(pool.payload_used_bytes().unwrap(), 0);
        crate::working_memory::memory_fixture::assert_unquoted_idle(&pool);
    }
}

#[test]
fn unknown_overflow_and_rank_limit_never_create_an_account() {
    let pool = crate::working_memory::memory_fixture::host_ledger(u64::MAX, 0).unwrap();
    let before = (
        ALLOCATIONS.get(),
        pool.payload_used_bytes().unwrap(),
        pool.payload_peak_bytes().unwrap(),
    );
    let unknown = admitted(
        vec![SymbolicDimension::Unknown],
        CaptureTransform::FullTensor,
        vec![],
    );
    assert!(matches!(
        CaptureTensorGeometry::prepare(&unknown, 0, CapturePhase::Prefill, 0, None),
        Err(CaptureTensorGeometryError::UnknownShape)
    ));
    let huge = admitted(
        vec![SymbolicDimension::Known(usize::MAX)],
        CaptureTransform::FullTensor,
        vec![],
    );
    let geometry =
        CaptureTensorGeometry::prepare(&huge, 0, CapturePhase::Prefill, 0, None).unwrap();
    assert!(matches!(
        CaptureTensorHostPlan::prepare(geometry),
        Err(WorkingMemoryError::Overflow)
    ));
    let excessive = admitted(
        vec![SymbolicDimension::Known(1); 33],
        CaptureTransform::FullTensor,
        vec![],
    );
    assert!(matches!(
        CaptureTensorGeometry::prepare(&excessive, 0, CapturePhase::Prefill, 0, None),
        Err(CaptureTensorGeometryError::RankExceeded {
            rank: 33,
            maximum: 32
        })
    ));
    assert_eq!(
        (
            ALLOCATIONS.get(),
            pool.payload_used_bytes().unwrap(),
            pool.payload_peak_bytes().unwrap()
        ),
        before
    );
}

#[test]
fn zero_scalar_preview_and_strided_slice_preserve_existing_wire_shape() {
    let pool = crate::working_memory::memory_fixture::host_ledger(u64::MAX, 0).unwrap();
    let cases = [
        (
            admitted(
                vec![SymbolicDimension::Known(0), SymbolicDimension::Known(7)],
                CaptureTransform::FullTensor,
                vec![],
            ),
            vec![0, 7],
        ),
        (
            admitted(vec![], CaptureTransform::FullTensor, vec![]),
            vec![],
        ),
        (
            admitted(
                vec![SymbolicDimension::Sequence, SymbolicDimension::Known(4)],
                CaptureTransform::Preview { max_elements: 5 },
                vec![],
            ),
            vec![5],
        ),
        (
            admitted(
                vec![SymbolicDimension::Sequence, SymbolicDimension::Known(4)],
                CaptureTransform::Slice,
                vec![CaptureSlice {
                    axis: "axis1".into(),
                    start: 0,
                    end: 4,
                    stride: 2,
                }],
            ),
            vec![3, 2],
        ),
    ];
    for (source, shape) in cases {
        let retained = plan(&source).retained_payload_bytes();
        let output = fill(
            pool.prepare_capture_tensor(
                plan(&source),
                CaptureTensorLimits::new(crate::working_memory::memory_fixture::host_limits(
                    u64::MAX,
                )),
            )
            .unwrap(),
        );
        assert_eq!(output.shape(), shape);
        assert_eq!(output.retained_payload_bytes(), Some(retained));
        let wire = serde_json::to_value(&output).unwrap();
        assert_eq!(wire["shape"], serde_json::to_value(shape).unwrap());
        assert_eq!(wire["data"]["dtype"], "f32");
        drop(output);
        assert_eq!(pool.payload_used_bytes().unwrap(), 0);
    }
}

#[test]
fn loading_exclusion_and_other_host_accounts_remain_effective() {
    let source = ordinary();
    let payload = plan(&source).initialization_peak_bytes();
    let required = required_charge(&source);
    let pool = crate::working_memory::memory_fixture::host_ledger(required * 2 - 1, 0).unwrap();
    let loading = pool.acquire_unquoted().unwrap();
    let before = ALLOCATIONS.get();
    assert!(matches!(
        pool.prepare_capture_tensor(
            plan(&source),
            CaptureTensorLimits::new(crate::working_memory::memory_fixture::host_limits(u64::MAX))
        ),
        Err(CaptureTensorConstructionError::Memory(
            WorkingMemoryError::UnknownBound
        ))
    ));
    assert_eq!(ALLOCATIONS.get(), before);
    drop(loading);
    let first = pool
        .prepare_capture_tensor(
            plan(&source),
            CaptureTensorLimits::new(crate::working_memory::memory_fixture::host_limits(u64::MAX)),
        )
        .unwrap();
    {
        let usage = pool.0.usage.lock().unwrap();
        let state = usage.funding.values().next().unwrap();
        assert_eq!(state.host_held, payload + state.control_floor);
        assert_eq!(state.remaining, required);
        assert_eq!(state.scopes, 1);
    }
    let before = (
        ALLOCATIONS.get(),
        pool.payload_used_bytes().unwrap(),
        pool.payload_peak_bytes().unwrap(),
    );
    assert!(
        matches!(pool.prepare_capture_tensor(plan(&source), CaptureTensorLimits::new(crate::working_memory::memory_fixture::host_limits(u64::MAX))), Err(CaptureTensorConstructionError::Memory(WorkingMemoryError::Domain(eredu_core::MemoryDomainError::BudgetExceeded { requested_bytes: required_bytes, limit_bytes, existing_bytes, .. }))) if required_bytes == required && limit_bytes - existing_bytes == required - 1)
    );
    assert_eq!(
        (
            ALLOCATIONS.get(),
            pool.payload_used_bytes().unwrap(),
            pool.payload_peak_bytes().unwrap()
        ),
        before
    );
    drop(first);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

#[test]
fn source_selection_and_actual_invocation_checks_precede_host_construction() {
    let source = ordinary();
    let before = ALLOCATIONS.get();
    let decode = CaptureTensorGeometry::prepare(&source, 0, CapturePhase::Decode, 2, None).unwrap();
    assert!(std::ptr::eq(decode.admission(), &source));
    assert_eq!(decode.shape(), &[1, 4]);
    assert_eq!(decode.phase(), CapturePhase::Decode);
    assert_eq!(decode.prediction(), 2);
    assert!(matches!(
        CaptureTensorGeometry::prepare(&source, 1, CapturePhase::Decode, 2, None),
        Err(CaptureTensorGeometryError::SelectionMissing { index: 1 })
    ));
    assert!(matches!(
        CaptureTensorGeometry::prepare(&source, 0, CapturePhase::Decode, 4, None),
        Err(CaptureTensorGeometryError::Inactive)
    ));
    assert!(matches!(
        CaptureTensorGeometry::prepare(
            &source,
            0,
            CapturePhase::Prefill,
            0,
            Some(CaptureInvocationShape {
                batch: 1,
                sequence: 2,
                context: Some(2)
            })
        ),
        Err(CaptureTensorGeometryError::Capture(CaptureError::Invalid(
            _
        )))
    ));
    let summary = admitted(
        vec![SymbolicDimension::Known(3)],
        CaptureTransform::Summary,
        vec![],
    );
    assert!(matches!(
        CaptureTensorGeometry::prepare(&summary, 0, CapturePhase::Prefill, 0, None),
        Err(CaptureTensorGeometryError::Unsupported)
    ));
    let maximum_rank = admitted(
        vec![SymbolicDimension::Known(1); 32],
        CaptureTransform::FullTensor,
        vec![],
    );
    assert_eq!(
        CaptureTensorGeometry::prepare(&maximum_rank, 0, CapturePhase::Prefill, 0, None)
            .unwrap()
            .shape(),
        &[1; 32]
    );
    assert_eq!(ALLOCATIONS.get(), before);
}

mod parent_account;
