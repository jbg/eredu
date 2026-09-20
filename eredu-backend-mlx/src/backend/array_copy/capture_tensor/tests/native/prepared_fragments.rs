use super::super::super::workspace_tests::fragment_admission;
use super::*;
use half::{bf16, f16};

#[test]
fn actual_fragment_precision_specialization_preserves_nonzero_global_values() {
    let stream = stream();
    let values = (0..2 * 2 * 7 * 3)
        .map(|n| n as f32 * 0.25 - 9.5)
        .collect::<Vec<_>>();
    let slices = vec![
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
    ];
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
        for maximum in [0, 7] {
            let (admission, inference) = fragment_admission(
                CaptureTransform::Preview {
                    max_elements: maximum,
                },
                slices.clone(),
            );
            let assembly = CapturePrefillRowAssembly::prepare(&admission, 0, inference).unwrap();
            let expected = values
                .iter()
                .enumerate()
                .filter(|(i, _)| (i / 3) % 7 % 2 == 1 && i % 3 % 2 == 0)
                .take(maximum as usize)
                .map(|(_, &v)| v)
                .collect::<Vec<_>>();
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
                let metadata = source.try_metadata_snapshot().unwrap();
                assert!(metadata.allocation().unwrap().bytes() > source.nbytes());
                assert_eq!(
                    metadata.allocation().unwrap().identity(),
                    full.try_metadata_snapshot()
                        .unwrap()
                        .allocation()
                        .unwrap()
                        .identity()
                );
                let prepared = {
                    let _cold = Cold::new();
                    assert_eq!(
                        PreparedCaptureFragment::validate_borrowed_source(&source, &fragment)
                            .unwrap(),
                        dtype
                    );
                    let prepared = PreparedCaptureFragment::new(&source, &fragment).unwrap();
                    let context =
                        WorkspaceContext::new(MlxMetalWorkspaceMechanisms::current_host().unwrap());
                    let mut projection = ExistingArrayProjection::new(&context);
                    let input = projection.project(&source).unwrap();
                    context.begin_state_span([&input]).unwrap();
                    let output = prepared.trace(&mut projection).unwrap();
                    let report = context.report(&[input, output]).unwrap();
                    assert!(report.total_bytes.is_some());
                    assert!(report.host_workspace_bytes.is_some());
                    assert!(
                        report.state.as_ref().unwrap().retained_bytes.unwrap()
                            >= metadata.allocation().unwrap().bytes() as u64
                    );
                    assert_eq!(
                        projection
                            .storage_roots()
                            .map(|(id, bytes, _)| (id, bytes))
                            .collect::<Vec<_>>(),
                        vec![(
                            metadata.allocation().unwrap().identity(),
                            metadata.allocation().unwrap().bytes() as u64
                        )]
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
                        usize::from(dtype != Dtype::Float32 && fragment.output_elements() != 0)
                    );
                    assert_eq!(source.try_metadata_snapshot().unwrap(), metadata);
                    assert_eq!(HOUSEKEEPING.get(), 0);
                    prepared
                };
                let roots = RefCell::new(Vec::with_capacity(prepared.recovery_descriptors()));
                roots.borrow_mut().push(source.clone());
                let capacity = roots.borrow().capacity();
                let output = selected(
                    prepared.program(),
                    &mut Native {
                        completion: CaptureCompletion::Ordinary,
                        stream: &stream,
                        roots: &roots,
                    },
                    source.clone(),
                )
                .unwrap();
                let settled = output.evaluated().unwrap();
                let captured = if fragment.output_elements() == 0 {
                    vec![]
                } else {
                    assert_eq!(output.dtype(), Dtype::Float32);
                    let cast = output.try_metadata_snapshot().unwrap();
                    assert!(cast.allocation().unwrap().bytes() >= output.size() * 4);
                    settled.try_iter::<f32>().unwrap().collect::<Vec<_>>()
                };
                assert_eq!(captured.len(), fragment.output_elements());
                for (mapping, value) in fragment.mappings().zip(captured) {
                    assert!(actual[mapping.destination_index()].replace(value).is_none());
                }
                assert_eq!(roots.borrow().len(), prepared.recovery_descriptors());
                assert_eq!(roots.borrow().capacity(), capacity);
                assert_eq!(source.try_metadata_snapshot().unwrap(), metadata);
            }
            assert_eq!(
                actual.into_iter().map(Option::unwrap).collect::<Vec<_>>(),
                expected
            );
        }
    }
}

#[test]
fn actual_fragment_preparation_rejects_logical_shape_dtype_and_unknown_backing_cold() {
    let stream = stream();
    let (admission, inference) = fragment_admission(CaptureTransform::FullTensor, vec![]);
    let assembly = CapturePrefillRowAssembly::prepare(&admission, 0, inference).unwrap();
    let fragment = assembly.fragment(0).unwrap();
    let full = Array::from_slice(&vec![1_f32; 84], &[2, 2, 7, 3]);
    let integer = Array::from_slice(&vec![7_u32; 36], &[2, 2, 3, 3]);
    let lazy = Array::zeros::<f32>(&[2, 2, 3, 3], &stream).unwrap();
    let before = lazy.try_metadata_snapshot().unwrap();
    assert!(before.allocation().is_none());
    let _cold = Cold::new();
    assert!(matches!(
        PreparedCaptureFragment::validate_borrowed_source(&full, &fragment),
        Err(CaptureTensorNativeError::ShapeMismatch)
    ));
    assert!(matches!(
        PreparedCaptureFragment::validate_borrowed_source(&integer, &fragment),
        Err(CaptureTensorNativeError::UnsupportedDtype(_))
    ));
    assert_eq!(
        PreparedCaptureFragment::validate_borrowed_source(&lazy, &fragment).unwrap(),
        Dtype::Float32
    );
    assert!(matches!(
        PreparedCaptureFragment::new(&lazy, &fragment),
        Err(CaptureTensorNativeError::UnsettledSource)
    ));
    assert_eq!(lazy.try_metadata_snapshot().unwrap(), before);
    assert_eq!(HOUSEKEEPING.get(), 0);
}
