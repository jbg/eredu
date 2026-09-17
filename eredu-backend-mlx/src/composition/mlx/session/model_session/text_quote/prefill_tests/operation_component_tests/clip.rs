//! Tensor-bound clip against the original native entry and a scalar oracle.
use super::*;
use eredu_nn::workspace::WorkspaceOperationKind;

#[test]
fn original_tensor_clip_preserves_broadcast_precision_and_reversed_bounds() {
    let mut prepared = OriginalOperationFixture::prepare(true);
    let stream = prepared.stream.clone();
    for case in 0..4 {
        let (dtype, bound_dtype) = match case {
            0 => (Dtype::Float32, Dtype::Float32),
            1 => (Dtype::Bfloat16, Dtype::Float32),
            2 => (Dtype::Float16, Dtype::Float16),
            _ => (Dtype::Bfloat16, Dtype::Bfloat16),
        };
        let grows = case == 3;
        let strided = case == 1 || case == 2;
        let input_shape = [2, if grows { 1 } else { 3 }];
        let values = if grows {
            vec![-1.5, 1.25]
        } else {
            vec![-1.5, -0.5, 0.25, 0.75, 1.25, -0.25]
        };
        let (minimum_shape, minimum): (&[i32], &[f32]) = if case == 0 {
            (&[], &[-0.75])
        } else {
            (&[1, 3], &[-0.5, 0.75, -1.0])
        };
        let (maximum_shape, maximum): (&[i32], &[f32]) = match case {
            0 => (&[], &[0.625]),
            3 => (&[3], &[0.25, 0.5, -0.125]),
            _ => (&[2, 1], &[0.25, -0.25]),
        };
        let input = if strided {
            let transposed = [
                values[0], values[3], values[1], values[4], values[2], values[5],
            ];
            Array::from_slice(&transposed, &[3, 2])
                .as_dtype(dtype, &stream)
                .unwrap()
                .transpose_axes(&[1, 0], &stream)
                .unwrap()
        } else {
            Array::from_slice(&values, &input_shape)
                .as_dtype(dtype, &stream)
                .unwrap()
        };
        let input = MlxTensor::from_array(input);
        let min = MlxTensor::from_array(
            Array::from_slice(minimum, minimum_shape)
                .as_dtype(bound_dtype, &stream)
                .unwrap(),
        );
        let max = MlxTensor::from_array(
            Array::from_slice(maximum, maximum_shape)
                .as_dtype(bound_dtype, &stream)
                .unwrap(),
        );
        let expected: Vec<f32> = (0..6)
            .map(|i| {
                let row = i / 3;
                let col = i % 3;
                let x: f32 = values[if grows { row } else { i }];
                let lo = minimum[if case == 0 { 0 } else { col }];
                let hi = maximum[match case {
                    0 => 0,
                    3 => col,
                    _ => row,
                }];
                // Deliberately not f32::clamp: reversed tensor bounds retain the
                // existing Maximum-then-Minimum numerical behavior.
                x.max(lo).min(hi)
            })
            .collect();
        let legacy =
            ops::clip(input.as_array(), (min.as_array(), max.as_array()), &stream).unwrap();
        assert_eq!(legacy.dtype(), bound_dtype);
        assert_eq!(
            legacy
                .as_dtype(Dtype::Float32, &stream)
                .unwrap()
                .evaluated()
                .unwrap()
                .try_to_vec::<f32>()
                .unwrap(),
            expected
        );
        drop(legacy);
        let invoke = || {
            let output = input.clip(&min, &max, &stream).unwrap();
            assert_eq!(output.as_array().dtype(), bound_dtype);
            output
                .into_array()
                .as_dtype(Dtype::Float32, &stream)
                .unwrap()
        };
        assert_eq!(
            invoke().evaluated().unwrap().try_to_vec::<f32>().unwrap(),
            expected
        );
        let context = WorkspaceContext::new(MlxMetalWorkspaceMechanisms::current_host().unwrap());
        let existing = |shape: &[i32], dtype, contiguous| {
            let floating = match dtype {
                Dtype::Float16 => WorkspaceFloatingType::Float16,
                Dtype::Bfloat16 => WorkspaceFloatingType::Bfloat16,
                _ => WorkspaceFloatingType::Float32,
            };
            WorkspaceTensor::existing(
                context
                    .layout(shape, WorkspaceDtype::Float32)
                    .unwrap()
                    .with_representation(Some(WorkspaceRepresentation::new(floating, contiguous))),
                &context,
            )
            .unwrap()
        };
        let source = existing(&input_shape, dtype, !strided);
        let low = existing(minimum_shape, bound_dtype, true);
        let high = existing(maximum_shape, bound_dtype, true);
        context.begin_span();
        let output = source.clip(&low, &high, &context).unwrap();
        let output = context
            .execute(
                WorkspaceOperationKind::Elementwise("cast_f32"),
                &[&output],
                vec![context.layout(&[2, 3], WorkspaceDtype::Float32).unwrap()],
            )
            .unwrap()
            .remove(0);
        let plan = OriginalComponentTestPlan::from_report(context.report(&[output]).unwrap());
        exercise_component(
            &mut prepared,
            &plan,
            &[input.as_array(), min.as_array(), max.as_array()],
            &expected,
            invoke,
        );
    }
    drop(stream);
    prepared.finish();
}
