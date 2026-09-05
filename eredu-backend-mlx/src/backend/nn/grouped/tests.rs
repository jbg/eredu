use super::*;
use crate::{backend::ExecutionContext, module::PhysicalParam};
use safemlx::{transforms::eval, Device, DeviceType};

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
