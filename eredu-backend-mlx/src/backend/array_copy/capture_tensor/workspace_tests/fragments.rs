use super::*;

#[test]
fn fragment_program_keeps_global_preview_order_and_complete_enclosing_span() {
    for transform in [
        CaptureTransform::FullTensor,
        CaptureTransform::Slice,
        CaptureTransform::Preview { max_elements: 0 },
        CaptureTransform::Preview { max_elements: 7 },
    ] {
        let preview = matches!(transform, CaptureTransform::Preview { .. });
        let slices = if matches!(transform, CaptureTransform::FullTensor) {
            vec![]
        } else {
            vec![
                CaptureSlice {
                    axis: "axis2".into(),
                    start: 1,
                    end: 7,
                    stride: 2,
                },
                CaptureSlice {
                    axis: "axis3".into(),
                    start: 0,
                    end: 3,
                    stride: 2,
                },
            ]
        };
        let (admission, inference) = fragment_admission(transform, slices);
        let assembly = CapturePrefillRowAssembly::prepare(&admission, 0, inference).unwrap();
        let context = WorkspaceContext::new(Facts::default());
        let mut roots = Vec::new();
        let mut expected_operations = 0;
        let mut expected_bytes = 0;
        context.begin_state_span([]).unwrap();
        for index in 0..assembly.chunk_count() {
            let fragment = assembly.fragment(index).unwrap();
            let shape = fragment
                .source_shape()
                .iter()
                .map(|&d| d as i32)
                .collect::<Vec<_>>();
            let (source, _) = imported(&context, &shape, WorkspaceDtype::Float32, Some(8192));
            let program = Selection::from_fragment(&fragment).unwrap();
            let before = context.report(&roots).unwrap().operations.len();
            let output = program.trace_within(&source, &context).unwrap();
            if preview {
                assert_eq!(output.shape(), [fragment.output_elements() as i32]);
            } else {
                assert_eq!(
                    output.shape(),
                    fragment
                        .selected_shape()
                        .iter()
                        .map(|&d| d as i32)
                        .collect::<Vec<_>>()
                );
            }
            roots.push(source);
            roots.push(output);
            let report = context.report(&roots).unwrap();
            let ops = &report.operations[before..];
            assert!(matches!(
                ops[0].kind,
                WorkspaceOperationKind::StaticSlice { .. }
            ));
            if preview {
                assert!(matches!(
                    ops[1].kind,
                    WorkspaceOperationKind::View("reshape")
                ));
                assert_eq!(
                    ops[1].outputs[0].shape(),
                    [fragment.selected_elements() as i32]
                );
                assert!(matches!(
                    ops[2].kind,
                    WorkspaceOperationKind::StaticSlice { .. }
                ));
                assert_eq!(
                    ops[2].outputs[0].shape(),
                    [fragment.output_elements() as i32]
                );
                expected_bytes += fragment.selected_elements() as u64 * 4;
            }
            if fragment.output_elements() > 0 {
                assert!(matches!(
                    ops.last().unwrap().kind,
                    WorkspaceOperationKind::Elementwise("capture_cast_f32")
                ));
                expected_bytes += fragment.output_elements() as u64 * 4;
            }
            expected_operations +=
                1 + usize::from(preview) * 2 + usize::from(fragment.output_elements() > 0);
            assert_eq!(report.operations.len(), expected_operations);
            assert_eq!(report.tensor_buffers.total_bytes, Some(expected_bytes));
        }
        // This diagnostic deliberately retains all three chunks in one span;
        // it proves no hook-level reset, not future canonical chunk retirement.
        assert_eq!(
            context.report(&roots).unwrap().operations.len(),
            expected_operations
        );
    }
}
#[test]
fn fragment_validation_is_cold_and_missing_source_or_operator_facts_stay_unknown() {
    let (admission, inference) =
        fragment_admission(CaptureTransform::Preview { max_elements: 7 }, vec![]);
    let assembly = CapturePrefillRowAssembly::prepare(&admission, 0, inference).unwrap();
    let fragment = assembly.fragment(0).unwrap();
    let shape = fragment
        .source_shape()
        .iter()
        .map(|&d| d as i32)
        .collect::<Vec<_>>();
    for (missing_tensor, missing_host, bytes) in [
        (true, false, Some(4096)),
        (false, true, Some(4096)),
        (false, false, None),
    ] {
        let context = WorkspaceContext::new(Facts {
            missing_tensor,
            missing_host,
        });
        let (source, _) = imported(&context, &shape, WorkspaceDtype::Float32, bytes);
        context.begin_state_span([&source]).unwrap();
        let program = Selection::from_fragment(&fragment).unwrap();
        let foreign = WorkspaceContext::new(Facts::default());
        let (other, _) = imported(&foreign, &shape, WorkspaceDtype::Float32, bytes);
        assert!(matches!(
            program.trace_within(&other, &context),
            Err(CaptureTensorNativeError::Workspace(_))
        ));
        let (wrong, _) = imported(&context, &shape, WorkspaceDtype::Uint8, bytes);
        assert!(matches!(
            program.trace_within(&wrong, &context),
            Err(CaptureTensorNativeError::UnsupportedWorkspaceDtype(_))
        ));
        let (wrong, _) = imported(&context, &[2, 2, 7, 3], WorkspaceDtype::Float32, bytes);
        assert!(matches!(
            program.trace_within(&wrong, &context),
            Err(CaptureTensorNativeError::ShapeMismatch)
        ));
        assert!(context
            .report(&[source.clone()])
            .unwrap()
            .operations
            .is_empty());
        let output = program.trace_within(&source, &context).unwrap();
        let report = context.report(&[source, output]).unwrap();
        assert_eq!(report.inference_transient_bytes(), None);
        if missing_tensor {
            assert!(!report.unpriced_operations.is_empty());
        }
        if missing_host {
            assert!(!report.unpriced_host_operations.is_empty());
        }
        if bytes.is_none() {
            assert_eq!(report.state.unwrap().retained_bytes, None);
        }
    }
}
#[cfg(target_pointer_width = "64")]
#[test]
fn signed_slice_normalization_and_preview_flatten_limits_are_checked() {
    // Every scalar fits i32, but MLX's (end-start)+(stride-1) does not.
    let initial = admitted(
        vec![
            SymbolicDimension::Sequence,
            SymbolicDimension::Known(i32::MAX as usize),
        ],
        CaptureTransform::Slice,
        vec![CaptureSlice {
            axis: "axis1".into(),
            start: 0,
            end: i32::MAX as u64,
            stride: i32::MAX as u64 - 1,
        }],
    );
    let inference = InferenceGeometry {
        batch_size: 1,
        cached_positions: 0,
        input_positions: 3,
        max_output_tokens: 4,
        prefill_chunk_positions: 1,
        output: OutputDemand::Sequence,
    };
    let assembly = CapturePrefillRowAssembly::prepare(&initial, 0, inference).unwrap();
    let fragment = assembly.fragment(0).unwrap();
    assert!(fragment.selection_axis(1).is_some());
    assert!(matches!(
        Selection::from_fragment(&fragment),
        Err(CaptureTensorNativeError::GeometryOverflow)
    ));
    // No large tensor is constructed: these are metadata-only actual admissions.
    let initial = admitted(
        vec![
            SymbolicDimension::Sequence,
            SymbolicDimension::Known(i32::MAX as usize / 2 + 1),
        ],
        CaptureTransform::Preview { max_elements: 0 },
        vec![],
    );
    let assembly = CapturePrefillRowAssembly::prepare(
        &initial,
        0,
        InferenceGeometry {
            prefill_chunk_positions: 3,
            ..inference
        },
    )
    .unwrap();
    assert!(matches!(
        Selection::from_fragment(&assembly.fragment(0).unwrap()),
        Err(CaptureTensorNativeError::GeometryOverflow)
    ));
}
