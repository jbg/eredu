use super::super::super::workspace_tests::fragment_admission;
use super::*;
use half::{bf16, f16};

#[test]
fn real_fragment_program_matches_global_selection_for_all_supported_float_types() {
    let stream = stream();
    let values = (0..2 * 2 * 7 * 3)
        .map(|i| i as f32 * 0.25 - 9.5)
        .collect::<Vec<_>>();
    for dtype in [Dtype::Float32, Dtype::Float16, Dtype::Bfloat16] {
        let full = match dtype {
            Dtype::Float32 => Array::from_slice(&values, &[2, 2, 7, 3]),
            Dtype::Float16 => Array::from_slice(
                &values.iter().map(|&v| f16::from_f32(v)).collect::<Vec<_>>(),
                &[2, 2, 7, 3],
            ),
            Dtype::Bfloat16 => Array::from_slice(
                &values
                    .iter()
                    .map(|&v| bf16::from_f32(v))
                    .collect::<Vec<_>>(),
                &[2, 2, 7, 3],
            ),
            _ => unreachable!(),
        };
        full.evaluated().unwrap();
        for transform in [
            CaptureTransform::FullTensor,
            CaptureTransform::Slice,
            CaptureTransform::Preview { max_elements: 0 },
            CaptureTransform::Preview { max_elements: 7 },
        ] {
            let sliced = !matches!(transform, CaptureTransform::FullTensor);
            let maximum = if let CaptureTransform::Preview { max_elements } = transform {
                Some(max_elements as usize)
            } else {
                None
            };
            let slices = if sliced {
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
            } else {
                vec![]
            };
            let (admission, inference) = fragment_admission(transform, slices);
            let assembly = CapturePrefillRowAssembly::prepare(&admission, 0, inference).unwrap();
            // Independent full logical reference: fixed source coordinate filter,
            // never the fragment intersection formula or local program arrays.
            let mut expected = values
                .iter()
                .enumerate()
                .filter(|(i, _)| {
                    let row = (*i / 3) % 7;
                    let column = *i % 3;
                    !sliced || (row % 2 == 1 && column % 2 == 0)
                })
                .map(|(_, &v)| v)
                .collect::<Vec<_>>();
            if let Some(n) = maximum {
                expected.truncate(n);
            }
            let mut actual = vec![None; expected.len()];
            for chunk in 0..assembly.chunk_count() {
                let fragment = assembly.fragment(chunk).unwrap();
                let source = full
                    .try_slice(
                        &[0, 0, fragment.input().start as i32, 0],
                        &[2, 2, fragment.input().end as i32, 3],
                        &[1, 1, 1, 1],
                        &stream,
                    )
                    .unwrap();
                source.evaluated().unwrap();
                let snapshot = source.try_metadata_snapshot().unwrap();
                assert!(snapshot.allocation().unwrap().bytes() > source.nbytes());
                assert_eq!(
                    snapshot.allocation().unwrap().identity(),
                    full.try_metadata_snapshot()
                        .unwrap()
                        .allocation()
                        .unwrap()
                        .identity()
                );
                let program = {
                    let _cold = Cold::new();
                    let program = Selection::from_fragment(&fragment).unwrap();
                    let context =
                        WorkspaceContext::new(MlxMetalWorkspaceMechanisms::current_host().unwrap());
                    let mut projection = ExistingArrayProjection::new(&context);
                    let value = projection.project(&source).unwrap();
                    context.begin_state_span([&value]).unwrap();
                    let output = program.trace_within(&value, &context).unwrap();
                    let report = context.report(&[value, output]).unwrap();
                    assert!(report.total_bytes.is_some());
                    assert!(report.host_workspace_bytes.is_some());
                    assert!(
                        report.state.as_ref().unwrap().retained_bytes.unwrap()
                            >= snapshot.allocation().unwrap().bytes() as u64
                    );
                    assert_eq!(
                        report
                            .operations
                            .iter()
                            .filter(|op| matches!(
                                op.kind,
                                WorkspaceOperationKind::Elementwise("capture_cast_f32")
                            ))
                            .count(),
                        usize::from(fragment.output_elements() > 0)
                    );
                    assert_eq!(source.try_metadata_snapshot().unwrap(), snapshot);
                    assert_eq!(HOUSEKEEPING.get(), 0);
                    program
                };
                // Exercise the existing shared runner, with no destination or
                // account path. This test makes no managed transfer/admission claim.
                let descriptors =
                    2 + usize::from(program.preview.is_some()) * 2 + usize::from(program.cast_f32);
                let roots = RefCell::new(Vec::with_capacity(descriptors));
                roots.borrow_mut().push(source.clone());
                let capacity = roots.borrow().capacity();
                let output = selected(
                    &program,
                    &mut Native {
                        completion: CaptureCompletion::Ordinary,
                        stream: &stream,
                        roots: &roots,
                    },
                    source.clone(),
                )
                .unwrap();
                let settled = output.evaluated().unwrap();
                assert_eq!(roots.borrow().capacity(), capacity);
                assert_eq!(roots.borrow().len(), descriptors);
                let captured = if fragment.output_elements() == 0 {
                    vec![]
                } else {
                    assert_eq!(output.dtype(), Dtype::Float32);
                    settled.try_iter::<f32>().unwrap().collect::<Vec<_>>()
                };
                assert_eq!(captured.len(), fragment.output_elements());
                for (map, value) in fragment.mappings().zip(captured) {
                    assert!(actual[map.destination_index()].replace(value).is_none());
                }
                assert_eq!(source.try_metadata_snapshot().unwrap(), snapshot);
            }
            assert_eq!(
                actual.into_iter().map(Option::unwrap).collect::<Vec<_>>(),
                expected
            );
        }
    }
}

#[test]
fn lazy_actual_fragment_source_stays_lazy_during_constructor_and_cold_trace() {
    let stream = stream();
    let (admission, inference) =
        fragment_admission(CaptureTransform::Preview { max_elements: 7 }, vec![]);
    let assembly = CapturePrefillRowAssembly::prepare(&admission, 0, inference).unwrap();
    let fragment = assembly.fragment(0).unwrap();
    let shape = fragment
        .source_shape()
        .iter()
        .map(|&n| n as i32)
        .collect::<Vec<_>>();
    let source = Array::zeros::<f32>(&shape, &stream).unwrap();
    let before = source.try_metadata_snapshot().unwrap();
    assert!(before.allocation().is_none());
    {
        let _cold = Cold::new();
        let program = Selection::from_fragment(&fragment).unwrap();
        let context = WorkspaceContext::new(MlxMetalWorkspaceMechanisms::current_host().unwrap());
        let mut projection = ExistingArrayProjection::new(&context);
        let value = projection.project(&source).unwrap();
        context.begin_state_span([&value]).unwrap();
        let output = program.trace_within(&value, &context).unwrap();
        let report = context.report(&[value, output]).unwrap();
        assert_eq!(report.inference_transient_bytes(), None);
        assert_eq!(source.try_metadata_snapshot().unwrap(), before);
        assert_eq!(HOUSEKEEPING.get(), 0);
    }
}
