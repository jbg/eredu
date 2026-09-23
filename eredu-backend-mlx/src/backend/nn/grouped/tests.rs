use super::*;
use crate::{backend::ExecutionContext, module::PhysicalParam};
use safemlx::{transforms::eval, Device, DeviceType};

fn verify_packed_selector_replacement(device: DeviceType) {
    use eredu_checkpoint::{AffineQuantization, LinearFormat};
    use eredu_gguf::{Endian, GgmlType};
    let stream = Stream::new_with_device(&Device::new(device, 0));
    let config = TopKGroupSelectorConfig::new(
        3,
        3,
        32,
        TopKGroupScoring::Softmax,
        false,
        0.0,
        1.0,
        1,
        1,
        false,
        false,
        None,
        false,
        false,
    )
    .unwrap();
    let values: Vec<f32> = (0..3 * 32)
        .map(|i| ((i * 7 + i / 32 * 3) % 16) as f32 * 0.125 - 1.0)
        .collect();
    let mut bytes = Vec::new();
    let mut codes = vec![0u32; 12];
    for row in 0..3 {
        bytes.extend(0x3000u16.to_le_bytes());
        for column in 0..32 {
            let i = row * 32 + column;
            let code = ((values[i] + 1.0) * 8.0) as u32;
            codes[i / 8] |= code << ((i % 8) * 4);
            bytes.push((code as i8 - 8) as u8);
        }
    }
    let input = Array::from_slice(
        &(0..64)
            .map(|i| ((i * 3 % 11) as f32 - 5.0) * 0.03125)
            .collect::<Vec<_>>(),
        &[2, 32],
    );
    let ids = Array::from_slice(&[0i32, 1, 2, 0, 1, 2], &[2, 3]);
    let host = |array: &Array| array.evaluated().unwrap().as_slice::<f32>().to_vec();
    let close = |actual: &[f32], expected: &[f32]| {
        assert_eq!(actual.len(), expected.len());
        for (a, b) in actual.iter().zip(expected) {
            assert!((a - b).abs() < 1e-6, "{a} != {b}");
        }
    };
    for format in [
        LinearFormat::Affine(AffineQuantization::new(32, 4).unwrap()),
        LinearFormat::GgufIQuant {
            ggml_type: GgmlType::Q8_0,
            endian: Endian::Little,
        },
    ] {
        let mut selector =
            TopKGroupSelector::new_with_quantization(config, format.weight_quantization(), &stream)
                .unwrap();
        let packed = if matches!(format, LinearFormat::Affine(_)) {
            selector.scales.value = Some(Array::from_slice(&[0.125f32; 3], &[3, 1]));
            selector.biases.value = Some(Array::from_slice(&[-1.0f32; 3], &[3, 1]));
            Array::from_slice(&codes, &[3, 4])
        } else {
            Array::from_slice(&bytes, &[bytes.len() as i32])
        };
        selector.weight.value = packed.clone();
        let mut reference =
            TopKGroupSelector::new_with_quantization(config, None, &stream).unwrap();
        reference.weight.value = Array::from_slice(&values, &[3, 32]);
        let original = host(
            &selector
                .select_indices(&input, &ids, &stream)
                .unwrap()
                .scores,
        );
        close(
            &original,
            &host(
                &reference
                    .select_indices(&input, &ids, &stream)
                    .unwrap()
                    .scores,
            ),
        );
        let mut edited = values.clone();
        edited[1] += 0.75;
        edited[2] -= 0.25;
        selector.weight.value = Array::from_slice(&edited, &[3, 32]);
        reference.weight.value = Array::from_slice(&edited, &[3, 32]);
        let actual = selector.select_indices(&input, &ids, &stream).unwrap();
        let expected = reference.select_indices(&input, &ids, &stream).unwrap();
        close(&host(&actual.scores), &host(&expected.scores));
        close(&host(&actual.weights), &host(&expected.weights));
        assert_ne!(host(&actual.scores), original);
        selector.weight.value = packed;
        assert_eq!(
            host(
                &selector
                    .select_indices(&input, &ids, &stream)
                    .unwrap()
                    .scores
            ),
            original
        );
    }
}

#[test]
#[ignore = "requires local native MLX execution"]
fn packed_selector_parameter_replacement_and_restoration_cpu() {
    verify_packed_selector_replacement(DeviceType::Cpu);
}

#[test]
#[ignore = "requires local MLX Metal execution"]
fn packed_selector_parameter_replacement_and_restoration_metal() {
    verify_packed_selector_replacement(DeviceType::Gpu);
}

#[test]
fn routing_intervention_preserves_grouped_selected_softmax_epsilon_and_learned_scales() {
    use eredu_nn::routing_intervention::{
        GroupScoreStage, GroupSelectionAction, GroupSelectionControl,
    };
    let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = execution.stream();
    let mut selector = TopKGroupSelector::new_with_quantization(
        TopKGroupSelectorConfig::new(
            2,
            4,
            1,
            TopKGroupScoring::SelectedSoftmax,
            true,
            0.25,
            1.5,
            2,
            1,
            false,
            true,
            None,
            false,
            true,
        )
        .unwrap(),
        None,
        stream,
    )
    .unwrap();
    selector.weight = PhysicalParam::new(Array::from_slice(
        &[0.0f32, 2.0f32.ln(), 3.0f32.ln(), 4.0f32.ln()],
        &[4, 1],
    ));
    selector.e_score_correction_bias =
        PhysicalParam::new(Some(Array::from_slice(&[0.0f32; 4], &[4])));
    selector.learned_coefficient_scale =
        PhysicalParam::new(Some(Array::from_slice(&[2.0f32, 3.0, 4.0, 5.0], &[4])));
    let input = Array::from_slice(&[1.0f32], &[1, 1]);
    let mut control = GroupSelectionControl {
        expected: selector.selection_spec().unwrap(),
        learned_coefficient_scale: true,
        first_row: 0,
        end_row: 1,
        row_stride: 1,
        action: GroupSelectionAction::Force(vec![0, 1]),
        capture_original: true,
    };
    for raw in [false, true] {
        if raw {
            control.action = GroupSelectionAction::Bias {
                stage: GroupScoreStage::RawLogits,
                ids: vec![0, 1],
                values: vec![10.0 + 2.0f32.ln(), 10.0],
            };
        }
        let (original, effective) = selector
            .select_intervened(&input, &control, stream)
            .unwrap();
        assert!(original.is_some());
        for slot in 0..2 {
            let id = effective
                .indices
                .try_index_device((0, slot), stream)
                .unwrap()
                .item::<i32>(stream);
            let weight = effective
                .weights
                .try_index_device((0, slot), stream)
                .unwrap()
                .item::<f32>(stream);
            let expected = if raw {
                if id == 0 {
                    1.2
                } else {
                    1.8
                }
            } else if id == 0 {
                0.8
            } else {
                2.4
            };
            assert!(id == 0 || id == 1);
            assert!(
                (weight - expected).abs() < 1e-5,
                "id {id}: {weight} vs {expected}"
            );
        }
    }
    control.action = GroupSelectionAction::Force(vec![0, 2]);
    assert!(
        selector
            .select_intervened(&input, &control, stream)
            .is_err(),
        "cross-group forcing must fail"
    );
    control.action = GroupSelectionAction::Force(vec![0, 1]);
    selector.learned_coefficient_scale =
        PhysicalParam::new(Some(Array::from_slice(&[-1.0f32; 4], &[4])));
    assert!(
        selector
            .select_intervened(&input, &control, stream)
            .is_err(),
        "invalid learned coefficients must fail before dispatch"
    );
}

#[test]
#[ignore = "requires MLX runtime execution"]
fn mlx_selected_softmax_selector_applies_input_and_group_scales() {
    let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = execution.stream();
    let mut selector = TopKGroupSelector::new_with_quantization(
        TopKGroupSelectorConfig::new(
            2,
            3,
            2,
            TopKGroupScoring::SelectedSoftmax,
            false,
            0.0,
            1.0,
            1,
            1,
            false,
            false,
            Some(0.0),
            true,
            true,
        )
        .unwrap(),
        None,
        stream,
    )
    .unwrap();
    selector.weight = PhysicalParam::new(Array::from_slice(
        &[1.0_f32, 0.0, 0.0, 1.0, -1.0, 0.0],
        &[3, 2],
    ));
    selector.input_scale = PhysicalParam::new(Some(Array::from_slice(&[2.0_f32, 1.0], &[2])));
    selector.learned_coefficient_scale =
        PhysicalParam::new(Some(Array::from_slice(&[2.0_f32, 3.0, 5.0], &[3])));
    let output = selector
        .select_with_selection_bias(&Array::from_slice(&[3.0_f32, 4.0], &[1, 2]), None, stream)
        .unwrap();
    eval([&output.indices, &output.scores, &output.weights]).unwrap();
    let first = 0.4_f32.exp() / (0.4_f32.exp() + 1.0);
    let mut seen = [false; 2];
    for selection in 0..2 {
        let group = output
            .indices
            .try_index_device((0, selection), stream)
            .unwrap()
            .item::<i32>(stream);
        let score = output
            .scores
            .try_index_device((0, selection), stream)
            .unwrap()
            .item::<f32>(stream);
        let weight = output
            .weights
            .try_index_device((0, selection), stream)
            .unwrap()
            .item::<f32>(stream);
        let (expected_score, expected_weight) = match group {
            0 => (first, 2.0 * first),
            1 => (1.0 - first, 3.0 * (1.0 - first)),
            other => panic!("unexpected selected group {other}"),
        };
        seen[group as usize] = true;
        assert!((score - expected_score).abs() < 1e-5);
        assert!((weight - expected_weight).abs() < 1e-5);
    }
    assert_eq!(seen, [true, true]);
}

#[test]
#[ignore = "requires MLX runtime execution"]
fn mlx_projection_bias_affects_selected_softmax_while_correction_is_selection_only() {
    let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = execution.stream();
    let mut selector = TopKGroupSelector::new_with_quantization(
        TopKGroupSelectorConfig::new(
            2,
            3,
            1,
            TopKGroupScoring::SelectedSoftmax,
            false,
            0.0,
            1.0,
            1,
            1,
            true,
            true,
            None,
            false,
            false,
        )
        .unwrap(),
        None,
        stream,
    )
    .unwrap();
    selector.weight = PhysicalParam::new(Array::from_slice(&[0.0_f32; 3], &[3, 1]));
    selector.bias = PhysicalParam::new(Some(Array::from_slice(
        &[0.0_f32, 2.0_f32.ln(), 4.0_f32.ln()],
        &[3],
    )));
    selector.e_score_correction_bias =
        PhysicalParam::new(Some(Array::from_slice(&[10.0_f32, 0.0, 0.0], &[3])));

    let output = selector
        .select_with_selection_bias(&Array::from_slice(&[1.0_f32], &[1, 1]), None, stream)
        .unwrap();
    eval([&output.indices, &output.scores, &output.weights]).unwrap();

    let mut seen = [false; 3];
    for selection in 0..2 {
        let group = output
            .indices
            .try_index_device((0, selection), stream)
            .unwrap()
            .item::<i32>(stream);
        let score = output
            .scores
            .try_index_device((0, selection), stream)
            .unwrap()
            .item::<f32>(stream);
        let weight = output
            .weights
            .try_index_device((0, selection), stream)
            .unwrap()
            .item::<f32>(stream);
        let expected = match group {
            0 => 0.2,
            2 => 0.8,
            other => panic!("unexpected selected group {other}"),
        };
        seen[group as usize] = true;
        assert!((score - expected).abs() < 1e-5);
        assert!((weight - expected).abs() < 1e-5);
    }
    assert_eq!(seen, [true, false, true]);
}

#[test]
#[ignore = "requires MLX runtime execution"]
fn mlx_gated_product_policy_applies_projection_biases_at_exact_stages() {
    let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = execution.stream();
    let policy =
        GatedProductPolicy::new(GatedProductActivation::Silu, Some(2.0), Some(1.5), 1.7, 1.0)
            .unwrap();
    let mut bank = PackedGatedProductGroups::new(1, 1, 1, None, None, [true, true], stream)
        .unwrap()
        .with_policy(policy)
        .unwrap();
    bank.gate_up_proj = PhysicalParam::new(Array::from_slice(&[0.0_f32, 0.0], &[1, 2, 1]));
    bank.gate_up_proj_bias = PhysicalParam::new(Some(Array::from_slice(&[2.5_f32, -3.0], &[1, 2])));
    bank.down_proj = PhysicalParam::new(Array::from_slice(&[2.0_f32], &[1, 1, 1]));
    bank.down_proj_bias = PhysicalParam::new(Some(Array::from_slice(&[5.0_f32], &[1, 1])));

    let output = bank
        .forward(
            &Array::from_slice(&[7.0_f32], &[1, 1]),
            &Array::from_slice(&[0_i32], &[1, 1]),
            &Array::from_slice(&[0.25_f32], &[1, 1]),
            stream,
        )
        .unwrap();
    eval([&output]).unwrap();
    let gate = 2.0 / (1.0 + (-3.4_f32).exp());
    let expected = 0.25 * (2.0 * gate * -0.5 + 5.0);
    assert!((output.item::<f32>(stream) - expected).abs() < 1e-5);
}

#[test]
#[ignore = "requires MLX runtime construction"]
fn mlx_mxfp4_group_bank_rejects_indivisible_projection_width() {
    let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = execution.stream();
    let error = PackedGatedProductGroups::new(
        1,
        33,
        32,
        Some(WeightQuantization::MxFp4),
        Some(WeightQuantization::MxFp4),
        [false, false],
        stream,
    )
    .unwrap_err();
    assert!(error.to_string().contains("not divisible"));
}

#[test]
#[ignore = "requires MLX runtime execution"]
fn mlx_tensor_parallel_down_bias_is_route_weighted_exactly_once() {
    let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = execution.stream();
    let rank = |down_weight: f32| {
        let mut bank =
            PackedGatedProductGroups::new(1, 1, 1, None, None, [false, true], stream).unwrap();
        bank.gate_up_proj = PhysicalParam::new(Array::from_slice(&[1.0_f32, 1.0], &[1, 2, 1]));
        bank.down_proj = PhysicalParam::new(Array::from_slice(&[down_weight], &[1, 1, 1]));
        bank.down_proj_bias = PhysicalParam::new(Some(Array::from_slice(&[5.0_f32], &[1, 1])));
        bank
    };
    let input = Array::from_slice(&[1.0_f32], &[1, 1]);
    let group = Array::from_slice(&[0_i32], &[1, 1]);
    let route_weight = Array::from_slice(&[0.25_f32], &[1, 1]);
    let mut rank_zero = rank(2.0);
    let mut rank_one = rank(3.0);
    let output_zero = rank_zero
        .forward_tensor_parallel(&input, &group, &route_weight, 2, stream)
        .unwrap();
    let output_one = rank_one
        .forward_tensor_parallel(&input, &group, &route_weight, 2, stream)
        .unwrap();
    let bias = output_zero.post_reduce().unwrap();
    eval([output_zero.reducible(), output_one.reducible(), bias]).unwrap();

    let gated = 1.0 / (1.0 + (-1.0_f32).exp());
    let expected = 0.25 * (5.0 * gated + 5.0);
    let reduced = output_zero.reducible().clone().item::<f32>(stream)
        + output_one.reducible().clone().item::<f32>(stream)
        + bias.clone().item::<f32>(stream);
    assert!((reduced - expected).abs() < 1e-5);
}

#[test]
#[ignore = "explicit native BF16 routing arithmetic conformance"]
fn routing_stages_keep_input_projection_fp32_scores_and_input_coefficients() {
    use eredu_nn::{RoutingArithmetic, RoutingPrecision};
    let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = execution.stream();
    let config = TopKGroupSelectorConfig::new(
        2,
        2,
        2,
        TopKGroupScoring::Sigmoid,
        true,
        0.0,
        1.0,
        1,
        1,
        false,
        false,
        None,
        false,
        false,
    )
    .unwrap()
    .with_arithmetic(RoutingArithmetic {
        projection: RoutingPrecision::Input,
        scores: RoutingPrecision::Float32,
        coefficients: RoutingPrecision::Input,
    });
    let mut selector = TopKGroupSelector::new_with_quantization(config, None, stream).unwrap();
    selector.weight = PhysicalParam::new(
        Array::from_slice(&[1.0f32, 0.00390625, -1.0, 0.0078125], &[2, 2])
            .as_dtype(Dtype::Bfloat16, stream)
            .unwrap(),
    );
    let input = Array::from_slice(&[1.0f32, 1.0], &[1, 2])
        .as_dtype(Dtype::Bfloat16, stream)
        .unwrap();
    let ids = Array::from_slice(&[0i32, 1], &[1, 2]);
    let selected = selector.select_indices(&input, &ids, stream).unwrap();
    assert_eq!(selected.scores.dtype(), Dtype::Float32);
    assert_eq!(selected.weights.dtype(), Dtype::Bfloat16);
    let scores = selected
        .scores
        .evaluated()
        .unwrap()
        .as_slice::<f32>()
        .to_vec();
    let expected = [
        1.0 / (1.0 + (-1.0f32).exp()),
        1.0 / (1.0 + 0.9921875f32.exp()),
    ];
    for (&a, &e) in scores.iter().zip(&expected) {
        assert!((a - e).abs() < 1e-6, "{a} != {e}");
    }
    let weights = selected
        .weights
        .as_dtype(Dtype::Float32, stream)
        .unwrap()
        .evaluated()
        .unwrap()
        .as_slice::<f32>()
        .to_vec();
    let sum = expected.iter().sum::<f32>();
    for (&a, &score) in weights.iter().zip(&expected) {
        let bits = (score / sum).to_bits();
        let expected = f32::from_bits((bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000);
        assert_eq!(a, expected);
    }
    let control = eredu_nn::routing_intervention::GroupSelectionControl {
        expected: selector.selection_spec().unwrap(),
        learned_coefficient_scale: false,
        first_row: 0,
        end_row: 1,
        row_stride: 1,
        action: eredu_nn::routing_intervention::GroupSelectionAction::Force(vec![0, 1]),
        capture_original: true,
    };
    let (original, effective) = selector
        .select_intervened(&input, &control, stream)
        .unwrap();
    assert_eq!(original.unwrap().weights.dtype(), Dtype::Bfloat16);
    assert_eq!(effective.weights.dtype(), Dtype::Bfloat16);
}

#[test]
#[ignore = "explicit native BF16 routing arithmetic conformance"]
fn input_precision_sigmoid_routes_preserve_bf16_with_f32_choice_bias() {
    use eredu_nn::{RoutingArithmetic, RoutingPrecision};
    let mut devices = vec![DeviceType::Cpu];
    if cfg!(all(feature = "metal", not(feature = "cuda"))) {
        devices.push(DeviceType::Gpu);
    }
    for device in devices {
        let execution = ExecutionContext::new(Device::new(device, 0));
        let stream = execution.stream();
        let input = Array::from_slice(
            &[
                -6.0f32, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                -1.296875, -0.8828125,
            ],
            &[15],
        )
        .as_dtype(Dtype::Bfloat16, stream)
        .unwrap();
        let expected = [
            0.002_471_923_8_f32,
            0.006_683_349_6,
            0.017_944_336,
            0.047_363_28,
            0.119140625,
            0.26953125,
            0.5,
            0.73046875,
            0.87890625,
            0.953125,
            0.98046875,
            0.9921875,
            0.99609375,
            0.21484375,
            0.29296875,
        ];
        let actual = crate::backend::nn::layers::sigmoid(input, stream)
            .unwrap()
            .as_dtype(Dtype::Float32, stream)
            .unwrap()
            .evaluated()
            .unwrap()
            .as_slice::<f32>()
            .to_vec();
        assert_eq!(
            actual, expected,
            "BF16 sigmoid rounds once after the quotient"
        );
        let config = TopKGroupSelectorConfig::new(
            2,
            4,
            2,
            TopKGroupScoring::Sigmoid,
            true,
            1e-6,
            1.0,
            1,
            1,
            false,
            true,
            None,
            false,
            false,
        )
        .unwrap()
        .with_arithmetic(RoutingArithmetic::uniform(RoutingPrecision::Input));
        let mut selector = TopKGroupSelector::new_with_quantization(config, None, stream).unwrap();
        selector.weight = PhysicalParam::new(
            Array::from_slice(
                &[
                    1.0f32, 0.00390625, -1.0, 0.0078125, 0.125, 0.0078125, -0.5, -0.0078125,
                ],
                &[4, 2],
            )
            .as_dtype(Dtype::Bfloat16, stream)
            .unwrap(),
        );
        selector.e_score_correction_bias =
            PhysicalParam::new(Some(Array::from_slice(&[-0.4f32, 0.3, 0.0, 0.1], &[4])));
        let input = Array::from_slice(&[1.0f32, 1.0], &[1, 2])
            .as_dtype(Dtype::Bfloat16, stream)
            .unwrap();
        let selected = selector
            .select_with_selection_bias(&input, None, stream)
            .unwrap();
        let ids = selected
            .indices
            .as_dtype(Dtype::Int32, stream)
            .unwrap()
            .evaluated()
            .unwrap()
            .as_slice::<i32>()
            .to_vec();
        let read = |value: &Array| {
            value
                .as_dtype(Dtype::Float32, stream)
                .unwrap()
                .evaluated()
                .unwrap()
                .as_slice::<f32>()
                .to_vec()
        };
        assert_eq!(selected.scores.dtype(), Dtype::Bfloat16);
        assert_eq!(selected.weights.dtype(), Dtype::Bfloat16);
        // Independent torch BF16 linear -> sigmoid -> top-k with F32 bias ->
        // BF16 sum, epsilon and quotient. The bias changes IDs, not coefficients.
        let scores = read(&selected.scores);
        let weights = read(&selected.weights);
        let mut sorted = ids.clone();
        sorted.sort();
        assert_eq!(sorted, [1, 2]);
        for (slot, id) in ids.iter().enumerate() {
            let (score, weight) = match id {
                1 => (0.26953125, 0.3359375),
                2 => (0.53125, 0.6640625),
                _ => unreachable!(),
            };
            assert_eq!(scores[slot], score);
            assert_eq!(weights[slot], weight);
        }
        let forced = selector
            .select_indices(&input, &selected.indices, stream)
            .unwrap();
        assert_eq!(read(&forced.weights), weights);
        let control = eredu_nn::routing_intervention::GroupSelectionControl {
            expected: selector.selection_spec().unwrap(),
            learned_coefficient_scale: false,
            first_row: 0,
            end_row: 1,
            row_stride: 1,
            action: eredu_nn::routing_intervention::GroupSelectionAction::Force(
                ids.iter().map(|id| *id as u32).collect(),
            ),
            capture_original: true,
        };
        let (original, effective) = selector
            .select_intervened(&input, &control, stream)
            .unwrap();
        assert_eq!(read(&original.unwrap().weights), weights);
        assert_eq!(effective.weights.dtype(), Dtype::Bfloat16);
        assert_eq!(read(&effective.weights), weights);
    }
}

#[test]
#[ignore = "explicit native BF16 routing reduction conformance"]
fn input_precision_routing_rounds_sum_once_after_float_accumulation() {
    use eredu_nn::{RoutingArithmetic, RoutingPrecision};
    let mut devices = vec![DeviceType::Cpu];
    if cfg!(all(feature = "metal", not(feature = "cuda"))) {
        devices.push(DeviceType::Gpu);
    }
    for device in devices {
        let execution = ExecutionContext::new(Device::new(device, 0));
        let stream = execution.stream();
        let config = TopKGroupSelectorConfig::new(
            4,
            4,
            4,
            TopKGroupScoring::Sigmoid,
            true,
            1e-6,
            1.0,
            1,
            1,
            false,
            false,
            None,
            false,
            false,
        )
        .unwrap()
        .with_arithmetic(RoutingArithmetic::uniform(RoutingPrecision::Input));
        let mut selector = TopKGroupSelector::new_with_quantization(config, None, stream).unwrap();
        selector.weight = PhysicalParam::new(
            Array::from_slice(
                &[
                    1.0f32, 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.,
                ],
                &[4, 4],
            )
            .as_dtype(Dtype::Bfloat16, stream)
            .unwrap(),
        );
        let input = Array::from_slice(&[-0.8125f32, -1.28125, -1.671875, -1.7734375], &[1, 4])
            .as_dtype(Dtype::Bfloat16, stream)
            .unwrap();
        let selected = selector
            .select_with_selection_bias(&input, None, stream)
            .unwrap();
        let ids = selected
            .indices
            .as_dtype(Dtype::Int32, stream)
            .unwrap()
            .evaluated()
            .unwrap()
            .as_slice::<i32>()
            .to_vec();
        let read = |v: &Array| {
            v.as_dtype(Dtype::Float32, stream)
                .unwrap()
                .evaluated()
                .unwrap()
                .as_slice::<f32>()
                .to_vec()
        };
        // Independent BF16 torch sigmoid scores sum exactly to 0.828125.
        // BF16 additions inside the reduction lose information before its output.
        let scores = [0.306_640_63_f32, 0.217_773_44, 0.158_203_13, 0.145_507_81];
        let weights = [0.37109375f32, 0.263_671_88, 0.19140625, 0.17578125];
        assert_eq!(
            read(&selected.scores),
            ids.iter()
                .map(|id| scores[*id as usize])
                .collect::<Vec<_>>()
        );
        assert_eq!(
            read(&selected.weights),
            ids.iter()
                .map(|id| weights[*id as usize])
                .collect::<Vec<_>>()
        );
        let forced = selector
            .select_indices(&input, &selected.indices, stream)
            .unwrap();
        assert_eq!(read(&forced.weights), read(&selected.weights));
    }
}
