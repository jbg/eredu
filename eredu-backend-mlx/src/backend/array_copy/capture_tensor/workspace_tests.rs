//! Metadata-only tests; no device, Array, allocator or native execution.
use super::*;
use eredu_core::{capture::*, *};
use eredu_nn::workspace::{
    WorkspaceBorrowedStorage, WorkspaceExistingStorage, WorkspaceHostBound, WorkspaceMechanisms,
    WorkspaceOperation, WorkspaceOperationBound, WorkspaceOutputStorage, WorkspaceTraceReport,
};

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
            physical_native_bytes: None,
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
            physical_native_limit: false,
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
#[derive(Clone, Copy, Debug, Default)]
struct Facts {
    missing_tensor: bool,
    missing_host: bool,
}
impl WorkspaceMechanisms for Facts {
    fn operation_bound(
        &self,
        op: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, eredu_nn::Error> {
        let storage = match &op.kind {
            WorkspaceOperationKind::StaticSlice { .. } => {
                WorkspaceOutputStorage::AliasInput(0)
            }
            WorkspaceOperationKind::View("reshape") => {
                WorkspaceOutputStorage::AllocateOrAliasInputs {
                    bytes: op.outputs[0].bytes()?,
                    inputs: vec![0],
                }
            }
            WorkspaceOperationKind::Elementwise("capture_cast_f32") => {
                if self.missing_tensor {
                    return Ok(None);
                }
                WorkspaceOutputStorage::Allocate(op.outputs[0].bytes()?)
            }
            _ => return Ok(None),
        };
        Ok(Some(WorkspaceOperationBound {
            outputs: vec![storage],
            scratch_bytes: 0,
            assumptions:
                "neutral test facts: alias selection, possible reshape copy, independent cast"
                    .into(),
        }))
    }
    fn host_workspace_bound(
        &self,
        op: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceHostBound>, eredu_nn::Error> {
        if self.missing_host
            && matches!(
                op.kind,
                WorkspaceOperationKind::Elementwise("capture_cast_f32")
            )
        {
            return Ok(None);
        }
        Ok(Some(WorkspaceHostBound {
            bytes: 0,
            assumptions: "neutral test mechanism has no numerical host staging".into(),
        }))
    }
}
fn admission(
    shape: &[usize],
    transform: CaptureTransform,
    slices: Vec<CaptureSlice>,
) -> AdmittedCapturePlan {
    admitted(
        shape
            .iter()
            .copied()
            .map(SymbolicDimension::Known)
            .collect(),
        transform,
        slices,
    )
}
fn geometry(plan: &AdmittedCapturePlan) -> CaptureTensorGeometry<'_> {
    CaptureTensorGeometry::prepare(plan, 0, CapturePhase::Decode, 1, None).unwrap()
}
fn axis1(start: u64, end: u64) -> Vec<CaptureSlice> {
    vec![CaptureSlice {
        axis: "axis1".into(),
        start,
        end,
        stride: 1,
    }]
}
fn imported(
    context: &WorkspaceContext,
    shape: &[i32],
    dtype: WorkspaceDtype,
    bytes: Option<u64>,
) -> (WorkspaceTensor, WorkspaceExistingStorage) {
    let storage = WorkspaceExistingStorage::new(bytes, context);
    let value = WorkspaceTensor::existing_with_storage(
        WorkspaceLayout::new(shape, dtype).unwrap(),
        &storage,
        context,
    )
    .unwrap();
    (value, storage)
}
fn casts(report: &WorkspaceTraceReport) -> usize {
    report
        .operations
        .iter()
        .filter(|op| {
            matches!(
                op.kind,
                WorkspaceOperationKind::Elementwise("capture_cast_f32")
            )
        })
        .count()
}

#[test]
fn imported_alias_keeps_full_backing_and_existing_span_and_borrowed_identity() {
    let context = WorkspaceContext::new(Facts::default());
    let (source, storage) = imported(&context, &[2, 3], WorkspaceDtype::Float32, Some(4096));
    let borrowed = WorkspaceBorrowedStorage::new(&context, [&storage]).unwrap();
    context.set_borrowed_storage(borrowed.clone()).unwrap();
    context.begin_state_span([&source]).unwrap();
    // A preceding allocation proves selection did not reset the active span.
    let prior_plan = admission(&[], CaptureTransform::FullTensor, vec![]);
    let (scalar, _) = imported(&context, &[], WorkspaceDtype::Float32, Some(4));
    Selection::from_geometry(&geometry(&prior_plan))
        .unwrap()
        .trace_within(&scalar, &context)
        .unwrap();
    let plan = admission(&[2, 3], CaptureTransform::Slice, axis1(1, 3));
    let selection =
        Selection::with_conversion(&geometry(&plan), ConversionMode::ActualF32).unwrap();
    let output = selection.trace_within(&source, &context).unwrap();
    assert_eq!(output.shape(), [2, 2]);
    assert!(source.same_context(&output));
    drop(source);
    drop(storage);
    let report = context.report(std::slice::from_ref(&output)).unwrap();
    assert_eq!(report.operations.len(), 3);
    let WorkspaceOperationKind::StaticSlice {starts,ends,strides}=&report.operations[2].kind else {panic!("actual selection coordinates lost");};
    assert_eq!(starts,&[0,1]);assert_eq!(ends,&[2,3]);assert_eq!(strides,&[1,1]);
    assert_eq!(casts(&report), 1);
    assert_eq!(report.tensor_buffers.total_bytes, Some(4));
    let state = report.state.as_ref().unwrap();
    assert_eq!(state.retained_bytes, Some(4096));
    assert_eq!(state.displaced_bytes, Some(0));
    assert_eq!(state.transient_bytes, Some(4));
    let residual = report.residual.unwrap();
    assert!(residual.borrowed_storage.same_identity(&borrowed));
    assert_eq!(residual.total_bytes, Some(4));
}

#[test]
fn full_slice_preview_scalar_and_empty_use_one_checked_future_program() {
    let cases = [
        (
            vec![2, 3],
            CaptureTransform::FullTensor,
            vec![],
            vec![2, 3],
            24,
            2,
        ),
        (
            vec![2, 3],
            CaptureTransform::Slice,
            axis1(1, 2),
            vec![2, 1],
            8,
            2,
        ),
        (
            vec![2, 3],
            CaptureTransform::Preview { max_elements: 3 },
            axis1(1, 3),
            vec![3],
            28,
            4,
        ),
        (vec![], CaptureTransform::FullTensor, vec![], vec![], 4, 2),
        (
            vec![0, 3],
            CaptureTransform::FullTensor,
            vec![],
            vec![0, 3],
            0,
            1,
        ),
        (
            vec![2, 3],
            CaptureTransform::Preview { max_elements: 0 },
            vec![],
            vec![0],
            24,
            3,
        ),
        (
            vec![2, 3],
            CaptureTransform::Slice,
            axis1(1, 1),
            vec![2, 0],
            0,
            1,
        ),
    ];
    for (shape, transform, slices, expected, bytes, operations) in cases {
        let context = WorkspaceContext::new(Facts::default());
        let plan = admission(&shape, transform, slices);
        let source_shape: Vec<_> = shape.iter().map(|&n| n as i32).collect();
        let (source, _) = imported(&context, &source_shape, WorkspaceDtype::Float32, Some(4096));
        context.begin_state_span([&source]).unwrap();
        let selection = Selection::from_geometry(&geometry(&plan)).unwrap();
        let output = selection.trace_within(&source, &context).unwrap();
        assert_eq!(output.shape(), expected, "source {shape:?}");
        let report = context.report(&[source, output]).unwrap();
        assert_eq!(report.total_bytes, Some(bytes), "source {shape:?}");
        assert_eq!(report.operations.len(), operations, "source {shape:?}");
        assert_eq!(casts(&report), usize::from(!expected.contains(&0)));
        assert_eq!(report.host_workspace_bytes, Some(0));
    }
}

#[test]
fn future_precision_never_takes_the_actual_f32_noop_shortcut() {
    for shape in [vec![2, 3], vec![], vec![0, 3]] {
        let plan = admission(&shape, CaptureTransform::FullTensor, vec![]);
        let geometry = geometry(&plan);
        for mode in [
            ConversionMode::ActualF32,
            ConversionMode::ActualHalf,
            ConversionMode::MayRequireF32,
        ] {
            let context = WorkspaceContext::new(Facts::default());
            let dims: Vec<_> = shape.iter().map(|&n| n as i32).collect();
            let (source, _) = imported(&context, &dims, WorkspaceDtype::Float32, Some(4096));
            context.begin_state_span([&source]).unwrap();
            let output = Selection::with_conversion(&geometry, mode)
                .unwrap()
                .trace_within(&source, &context)
                .unwrap();
            let report = context.report(&[source, output]).unwrap();
            let needs_cast = !matches!(mode, ConversionMode::ActualF32) && geometry.elements() != 0;
            assert_eq!(casts(&report), usize::from(needs_cast));
            assert_eq!(
                report.total_bytes,
                Some(if needs_cast {
                    geometry.elements() as u64 * 4
                } else {
                    0
                })
            );
        }
    }
}

#[test]
fn invalid_source_context_shape_or_type_rejects_before_tracing() {
    let context = WorkspaceContext::new(Facts::default());
    let foreign = WorkspaceContext::new(Facts::default());
    let plan = admission(&[2, 3], CaptureTransform::FullTensor, vec![]);
    let selection = Selection::from_geometry(&geometry(&plan)).unwrap();
    let (source, _) = imported(&context, &[2, 3], WorkspaceDtype::Float32, Some(256));
    context.begin_state_span([&source]).unwrap();
    let (wrong_context, _) = imported(&foreign, &[2, 3], WorkspaceDtype::Float32, Some(256));
    assert!(matches!(
        selection.trace_within(&wrong_context, &context),
        Err(CaptureTensorNativeError::Workspace(_))
    ));
    let (wrong_shape, _) = imported(&context, &[3, 2], WorkspaceDtype::Float32, Some(256));
    assert!(matches!(
        selection.trace_within(&wrong_shape, &context),
        Err(CaptureTensorNativeError::ShapeMismatch)
    ));
    for dtype in [
        WorkspaceDtype::Bool,
        WorkspaceDtype::Int32,
        WorkspaceDtype::Uint32,
        WorkspaceDtype::Uint8,
    ] {
        let (wrong_type, _) = imported(&context, &[2, 3], dtype, Some(256));
        assert!(
            matches!(selection.trace_within(&wrong_type, &context), Err(CaptureTensorNativeError::UnsupportedWorkspaceDtype(actual)) if actual == dtype)
        );
    }
    let report = context.report(&[source]).unwrap();
    assert!(report.operations.is_empty());
    assert_eq!(report.total_bytes, Some(0));
    assert_eq!(report.state.unwrap().retained_bytes, Some(256));
}

#[test]
fn missing_tensor_host_or_source_bounds_remain_incomplete() {
    let plan = admission(
        &[2, 3],
        CaptureTransform::Preview { max_elements: 3 },
        vec![],
    );
    for (missing_tensor, missing_host, source_bytes) in [
        (true, false, Some(4096)),
        (false, true, Some(4096)),
        (false, false, None),
    ] {
        let context = WorkspaceContext::new(Facts {
            missing_tensor,
            missing_host,
        });
        let (source, _) = imported(&context, &[2, 3], WorkspaceDtype::Float32, source_bytes);
        context.begin_state_span([&source]).unwrap();
        let output = Selection::from_geometry(&geometry(&plan))
            .unwrap()
            .trace_within(&source, &context)
            .unwrap();
        let report = context.report(&[source, output]).unwrap();
        assert_eq!(report.operations.len(), 4);
        assert_eq!(casts(&report), 1);
        assert_eq!(report.inference_transient_bytes(), None);
        if missing_tensor {
            assert_eq!(report.tensor_buffers.total_bytes, None);
            assert!(!report.unpriced_operations.is_empty());
        }
        if missing_host {
            assert_eq!(report.host_workspace_bytes, None);
            assert!(!report.unpriced_host_operations.is_empty());
        }
        if source_bytes.is_none() {
            assert_eq!(report.state.unwrap().retained_bytes, None);
        }
    }
}

#[test]
fn source_extents_are_checked_before_any_workspace_source_is_needed() {
    let plan = admission(
        &[i32::MAX as usize + 1],
        CaptureTransform::Preview { max_elements: 0 },
        vec![],
    );
    let geometry = geometry(&plan);
    assert!(matches!(
        Selection::from_geometry(&geometry),
        Err(CaptureTensorNativeError::GeometryOverflow)
    ));
}

mod fragments;

// Shared only by this module's metadata child and its real-array cold sibling.
pub(super) fn fragment_admission(
    transform: CaptureTransform,
    slices: Vec<CaptureSlice>,
) -> (AdmittedCapturePlan, InferenceGeometry) {
    let initial = admitted(
        vec![
            SymbolicDimension::Batch,
            SymbolicDimension::Known(2),
            SymbolicDimension::Sequence,
            SymbolicDimension::Known(3),
        ],
        CaptureTransform::FullTensor,
        vec![],
    );
    let catalog = ObservationCatalog {
        schema_version: 1,
        points: initial.points().to_vec(),
        completeness: DescriptionCompleteness::Complete,
    };
    let caps = CaptureCapabilities {
        transformations: vec![
            CaptureTransformKind::FullTensor,
            CaptureTransformKind::Slice,
            CaptureTransformKind::Preview,
        ],
        ..Default::default()
    };
    let support = ObservationSupportReport {
        schema_version: 1,
        capture: caps.clone(),
        points: vec![ObservationSupport {
            path: catalog.points[0].path.clone(),
            prefill: ObservationSupportStatus::Supported,
            decode: ObservationSupportStatus::Supported,
            floating_to_f32: true,
        }],
    };
    let mut raw = initial.plan().clone();
    raw.selections[0].transform = transform;
    raw.selections[0].slices = slices;
    raw.selections[0].schedule.decode = false;
    let geometry = InferenceGeometry {
        batch_size: 2,
        cached_positions: 11,
        input_positions: 7,
        max_output_tokens: 1,
        prefill_chunk_positions: 3,
        output: OutputDemand::Sequence,
    };
    let plan = raw
        .admit_with_text_origin(
            &catalog,
            &support,
            &caps,
            CaptureRequestShape {
                batch: 2,
                prompt_tokens: 7,
                max_predictions: 1,
            },
            CaptureTextOrigin {
                cached_positions: 11,
            },
        )
        .unwrap();
    (plan, geometry)
}
