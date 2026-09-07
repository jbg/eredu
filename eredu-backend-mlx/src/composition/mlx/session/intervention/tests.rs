use super::*;

#[test]
fn native_intervention_estimates_account_for_group_masks_and_host_lookup() {
    let mut policy = InterventionRoutingPolicy {
        expert_count: 64,
        top_k: 2,
        scoring: RoutingScoring::Softmax,
        normalize_selected: true,
        normalization_epsilon: 0.,
        coefficient_scale: 1.,
        groups: 1,
        selected_groups: 1,
        learned_coefficient_scale: false,
        shared_experts: 0,
    };
    let estimate = |p: &InterventionRoutingPolicy, rows| {
        NativeInterventionEstimator
            .original_route_usage(p, rows)
            .unwrap()
    };
    let plain = estimate(&policy, 3);
    assert_eq!(plain.host_bytes, 0);
    policy.groups = 8;
    let grouped = estimate(&policy, 3);
    assert!(grouped.host_bytes > 0 && grouped.retained_bytes > plain.retained_bytes);
    policy.selected_groups = 7;
    let wider = estimate(&policy, 3);
    assert!(wider.retained_bytes > grouped.retained_bytes);
    assert_eq!(wider.host_bytes, grouped.host_bytes);
    assert!(estimate(&policy, 6).retained_bytes > wider.retained_bytes);
    assert_eq!(wider.captures, 0);
    assert_eq!(wider.encoded_bytes, 0);
    assert!(NativeInterventionEstimator
        .original_route_usage(&policy, u64::MAX)
        .is_err());
}

#[test]
fn native_activation_primitives_pass_shared_conformance() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    eredu_evaluation::intervention::activation_conformance(
        &mut NativeCapture { stream: &stream },
        |value| value.to_f32_vec(&stream).unwrap(),
    );
}

fn row() -> ResolvedCaptureSlice {
    ResolvedCaptureSlice {
        starts: vec![1, 0],
        ends: vec![2, 3],
        strides: vec![1, 1],
        shape: vec![1, 3],
    }
}

#[test]
fn native_intervention_patch_scale_mask_and_bias_preserve_other_rows() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let mut native = NativeCapture { stream: &stream };
    let input = MlxTensor::from_array(Array::from_slice(
        &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
        &[2, 3],
    ));
    let cases = [
        (
            InterventionAction::Zero {
                dtype: InterventionDtype::Float32,
            },
            vec![1.0, 2.0, 3.0, 0.0, 0.0, 0.0],
        ),
        (
            InterventionAction::Scale {
                dtype: InterventionDtype::Float32,
                factor: 2.0,
            },
            vec![1.0, 2.0, 3.0, 8.0, 10.0, 12.0],
        ),
        (
            InterventionAction::Mask {
                dtype: InterventionDtype::Float32,
                shape: vec![1, 3],
                keep: vec![true, false, true],
            },
            vec![1.0, 2.0, 3.0, 4.0, 0.0, 6.0],
        ),
        (
            InterventionAction::Replace {
                tensor: InterventionTensor {
                    shape: vec![1, 3],
                    values: InterventionValues::Float32(vec![7.0, 8.0, 9.0]),
                },
            },
            vec![1.0, 2.0, 3.0, 7.0, 8.0, 9.0],
        ),
        (
            InterventionAction::Add {
                tensor: InterventionTensor {
                    shape: vec![1, 3],
                    values: InterventionValues::Float32(vec![7.0, 8.0, 9.0]),
                },
            },
            vec![1.0, 2.0, 3.0, 11.0, 13.0, 15.0],
        ),
    ];
    for (action, expected) in cases {
        let output =
            eredu_runtime::intervention::apply_activation(&mut native, &input, &action, &row())
                .unwrap();
        assert_eq!(output.as_array().dtype(), Dtype::Float32);
        assert_eq!(output.to_f32_vec(&stream).unwrap(), expected);
        assert_eq!(
            input.to_f32_vec(&stream).unwrap(),
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );
    }
    let output = eredu_runtime::intervention::apply_activation(
        &mut native,
        &input,
        &InterventionAction::MaskLogits {
            dtype: InterventionDtype::Float32,
            token_ids: vec![1],
        },
        &row(),
    )
    .unwrap();
    assert_eq!(
        output.to_f32_vec(&stream).unwrap(),
        [1.0, 2.0, 3.0, 4.0, f32::NEG_INFINITY, 6.0]
    );
}

#[test]
fn native_intervention_half_payload_bits_are_exact_and_casts_are_rejected() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let mut native = NativeCapture { stream: &stream };
    let slice = ResolvedCaptureSlice {
        starts: vec![0],
        ends: vec![2],
        strides: vec![1],
        shape: vec![2],
    };
    for (dtype, values) in [
        (
            Dtype::Float16,
            InterventionValues::Float16(vec![0x3555, 0x8000]),
        ),
        (
            Dtype::Bfloat16,
            InterventionValues::Bfloat16(vec![0x3eab, 0x8000]),
        ),
    ] {
        let input = MlxTensor::from_array(
            Array::from_slice(&[1.0f32, 2.0], &[2])
                .as_dtype(dtype, &stream)
                .unwrap(),
        );
        let output = eredu_runtime::intervention::apply_activation(
            &mut native,
            &input,
            &InterventionAction::Replace {
                tensor: InterventionTensor {
                    shape: vec![2],
                    values: values.clone(),
                },
            },
            &slice,
        )
        .unwrap();
        let evaluated = output.as_array().evaluated().unwrap();
        match values {
            InterventionValues::Float16(bits) => assert_eq!(
                evaluated
                    .as_slice::<half::f16>()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                bits
            ),
            InterventionValues::Bfloat16(bits) => assert_eq!(
                evaluated
                    .as_slice::<half::bf16>()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                bits
            ),
            _ => unreachable!(),
        }
        assert!(eredu_runtime::intervention::apply_activation(
            &mut native,
            &input,
            &InterventionAction::Replace {
                tensor: InterventionTensor {
                    shape: vec![2],
                    values: InterventionValues::Float32(vec![1.0; 2])
                }
            },
            &slice
        )
        .is_err());
    }
}
