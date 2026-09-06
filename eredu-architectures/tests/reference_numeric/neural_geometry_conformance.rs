//! Independent scalar oracles for the shared tensor-independent operator checks.

use super::*;

#[test]
fn numeric_l2_uses_additive_epsilon_including_small_norms() {
    let context = NumericContext::default();
    for (values, epsilon) in [
        ([1.0_f32, -1.0], 4.0),
        ([0.0, 0.0], 0.25),
        ([0.01, -0.02], 0.1),
    ] {
        let input = NumericTensor::new([1, 2], values.to_vec());
        let output = NumericBackend::l2_normalize(&input, epsilon, &context).unwrap();
        let denominator = (values[0] * values[0] + values[1] * values[1] + epsilon).sqrt();
        for (actual, expected) in output
            .data
            .iter()
            .zip(values.map(|value| value / denominator))
        {
            assert!((*actual - expected).abs() < 1e-7);
        }
    }
    let scalar = NumericTensor::new([], vec![1.0]);
    assert!(NumericBackend::l2_normalize(&scalar, 1e-5, &context).is_err());
    let input = NumericTensor::new([1, 2], vec![1.0, 2.0]);
    for epsilon in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        assert!(NumericBackend::l2_normalize(&input, epsilon, &context).is_err());
        assert!(NumericBackend::rms_norm_without_weight(&input, epsilon, &context).is_err());
    }
}

#[test]
fn numeric_grouped_normalization_rejects_bad_geometry_without_panics() {
    let context = NumericContext::default();
    let input = NumericTensor::new([1, 4], vec![1.0; 4]);
    let gate = NumericTensor::new([1, 4], vec![0.5; 4]);
    let weight = NumericTensor::new([4], vec![1.0; 4]);
    for (groups, epsilon) in [(0, 1e-5), (-1, 1e-5), (3, 1e-5), (2, f32::NAN)] {
        assert!(NumericBackend::gated_group_rms_norm(
            &input, &gate, &weight, groups, epsilon, &context
        )
        .is_err());
        assert!(NumericBackend::silu_gated_group_rms_norm(
            &input, &gate, &weight, groups, epsilon, &context
        )
        .is_err());
    }
    let scalar = NumericTensor::new([], vec![1.0]);
    assert!(
        NumericBackend::gated_group_rms_norm(&scalar, &scalar, &weight, 1, 1e-5, &context).is_err()
    );
}

#[test]
fn numeric_causal_mask_window_is_backward_distance_not_token_count() {
    let context = NumericContext::default();
    for distance in [None, Some(0), Some(1), Some(2), Some(i32::MAX)] {
        let mask = NumericBackend::causal_mask(3, 2, distance, &context).unwrap();
        assert_eq!(mask.shape, [3, 5]);
        for query in 0..3 {
            for key in 0..5 {
                let position = query + 2;
                let expected =
                    key <= position && distance.is_none_or(|distance| key >= position - distance);
                assert_eq!(mask.data[(query * 5 + key) as usize] == 0.0, expected);
            }
        }
    }
    for (sequence, offset, distance) in [
        (-1, 0, None),
        (1, -1, None),
        (1, 0, Some(-1)),
        (1, i32::MAX, None),
    ] {
        assert!(NumericBackend::causal_mask(sequence, offset, distance, &context).is_err());
    }
}
