use super::*;
use eredu_nn::Tensor;

fn layout(shape: &[i32], scalar: Option<F>) -> WorkspaceLayout {
    WorkspaceLayout::new(shape, WorkspaceDtype::Float32)
        .unwrap()
        .with_representation(scalar.map(|value| WorkspaceRepresentation::new(value, false)))
}
fn operation(values: &[Option<F>], weight: bool, bias: bool) -> WorkspaceOperation {
    WorkspaceOperation {
        kind: WorkspaceOperationKind::LayerNorm { weight, bias },
        inputs: values
            .iter()
            .enumerate()
            .map(|(i, scalar)| layout(if i == 0 { &[2, 8] } else { &[8] }, *scalar))
            .collect(),
        outputs: vec![layout(&[2, 8], None)],
    }
}
fn output_scalar(operation: &WorkspaceOperation) -> Option<F> {
    super::super::output(operation.as_view(), 0).map(WorkspaceRepresentation::dtype)
}

#[test]
fn layer_norm_preserves_exact_affine_roles_and_refuses_missing_sources() {
    let formats = [F::Float32, F::Float16, F::Bfloat16];
    for input in formats {
        assert_eq!(
            output_scalar(&operation(&[Some(input)], false, false)),
            Some(input)
        );
        for weight in formats {
            assert_eq!(
                output_scalar(&operation(&[Some(input), Some(weight)], true, false)),
                Some(promote(input, weight))
            );
            assert_eq!(
                output_scalar(&operation(&[Some(input), Some(weight)], false, true)),
                Some(input),
                "bias-only casts to input rather than widening it"
            );
            assert_eq!(
                output_scalar(&operation(&[Some(input), None], false, true)),
                Some(input)
            );
            for bias in formats {
                let values = [Some(input), Some(weight), Some(bias)];
                let op = operation(&values, true, true);
                assert_eq!(
                    output_scalar(&op),
                    Some(promote(promote(input, weight), bias))
                );
                assert!(
                    !super::super::output(op.as_view(), 0)
                        .unwrap()
                        .row_contiguous()
                );
                for missing in 0..3 {
                    let mut unknown = values;
                    unknown[missing] = None;
                    assert_eq!(output_scalar(&operation(&unknown, true, true)), None);
                }
            }
        }
    }
    let mut wrong = operation(&[Some(F::Float32); 3], true, true);
    wrong.inputs[1] = layout(&[7], Some(F::Float32));
    assert_eq!(output_scalar(&wrong), None);
    wrong = operation(&[Some(F::Float32); 3], true, true);
    wrong.outputs[0] = layout(&[1, 8], None);
    assert_eq!(output_scalar(&wrong), None);
    assert_eq!(
        output_scalar(&operation(&[Some(F::Float32); 2], false, false)),
        None
    );
    assert_eq!(
        output_scalar(&operation(&[Some(F::Float32); 2], true, true)),
        None
    );
}

#[test]
fn layer_norm_then_projection_keeps_the_actual_collective_input_scalar() {
    let mechanism = MlxMetalWorkspaceMechanisms {
        allocation: NativeAllocationFacts { page_size: 16384, cpu_header: false },
        sdpa_blocks: None,
    };
    let context = WorkspaceContext::new(mechanism);
    let value =
        |shape| WorkspaceTensor::existing(layout(shape, Some(F::Float32)), &context).unwrap();
    let input = value(&[2, 8]);
    let scale = value(&[8]);
    let bias = value(&[8]);
    let normalized =
        WorkspaceTensor::layer_norm(&input, Some(&scale), Some(&bias), 1e-6, &context).unwrap();
    let projection = value(&[16, 8]);
    let output = WorkspaceTensor::linear(&normalized, &projection, None, &context).unwrap();
    assert_eq!(
        output
            .layout()
            .representation()
            .map(WorkspaceRepresentation::dtype),
        Some(F::Float32)
    );
    // Logical F32 alone must still not authorize a physical communication type.
    let unknown = WorkspaceTensor::existing(layout(&[8], None), &context).unwrap();
    let normalized =
        WorkspaceTensor::layer_norm(&input, Some(&unknown), Some(&bias), 1e-6, &context).unwrap();
    let output = WorkspaceTensor::linear(&normalized, &projection, None, &context).unwrap();
    assert!(output.layout().representation().is_none());
}

#[test]
#[cfg(all(target_vendor = "apple", feature = "metal", not(feature = "cuda")))]
#[ignore = "executes the native CPU LayerNorm worker; run in the serialized native suite"]
fn native_layer_norm_optional_affine_dtypes_match_every_reported_fact() {
    use safemlx::{Array, Device, DeviceType, Dtype, Stream};
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let formats = [
        (F::Float32, Dtype::Float32),
        (F::Float16, Dtype::Float16),
        (F::Bfloat16, Dtype::Bfloat16),
    ];
    for (source, input_dtype) in formats {
        let input = Array::from_slice(
            &[
                -0.9_f32, 0.2, 0.4, 0.8, 1.1, -0.3, 0.7, 1.3, 0.1, 0.3, -0.8, 0.5, 1.0, 0.4, -0.4,
                0.6,
            ],
            &[2, 8],
        )
        .as_dtype(input_dtype, &stream)
        .unwrap();
        for (weight_scalar, weight_dtype) in formats {
            let weight = Array::from_slice(&[0.7_f32, 0.9, 1.1, 0.6, 1.3, 0.8, 1.2, 0.5], &[8])
                .as_dtype(weight_dtype, &stream)
                .unwrap();
            for (bias_scalar, bias_dtype) in formats {
                let bias =
                    Array::from_slice(&[0.1_f32, -0.2, 0.3, 0.1, -0.1, 0.2, -0.3, 0.4], &[8])
                        .as_dtype(bias_dtype, &stream)
                        .unwrap();
                for has_weight in [false, true] {
                    for has_bias in [false, true] {
                        let actual = safemlx::fast::layer_norm(
                            &input,
                            has_weight.then_some(&weight),
                            has_bias.then_some(&bias),
                            1e-6,
                            &stream,
                        )
                        .unwrap();
                        let mut sources = vec![Some(source)];
                        if has_weight {
                            sources.push(Some(weight_scalar));
                        }
                        if has_bias {
                            sources.push(Some(bias_scalar));
                        }
                        let declared = output_scalar(&operation(&sources, has_weight, has_bias))
                            .expect("all actual source roles and scalar types are present");
                        let native = formats
                            .iter()
                            .find(|(_, dtype)| *dtype == actual.dtype())
                            .unwrap()
                            .0;
                        assert_eq!(declared, native);
                        actual.evaluated().unwrap();
                        let values = crate::MlxTensor::from_array(actual)
                            .to_f32_vec(&stream)
                            .unwrap();
                        assert!(values.iter().all(|v| v.is_finite()));
                        assert!(values.iter().any(|v| v.abs() > 0.01));
                    }
                }
            }
        }
    }
}
