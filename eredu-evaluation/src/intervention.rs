//! Reusable behavioral checks for implementations of native intervention primitives.
//! Run these from backend tests; production inference must not depend on this crate.
use eredu_core::{capture::*, intervention::*};
use eredu_nn::routing_intervention::*;

#[cfg(test)]
mod tests;

/// Runs exact F32 activation checks using the production shared driver. `read`
/// materializes test evidence only; inference never uses this callback.
pub fn activation_conformance<B: InterventionBackend>(
    backend: &mut B,
    read: impl Fn(&B::Tensor) -> Vec<f32>,
) {
    let source = InterventionTensor {
        shape: vec![2, 4],
        values: InterventionValues::Float32((1..=8).map(|v| v as f32).collect()),
    };
    let input = backend.realize_tensor(&source).unwrap();
    let slice = ResolvedCaptureSlice {
        starts: vec![0, 1],
        ends: vec![2, 4],
        strides: vec![1, 2],
        shape: vec![2, 2],
    };
    let tensor = || InterventionTensor {
        shape: vec![2, 2],
        values: InterventionValues::Float32(vec![10., 20., 30., 40.]),
    };
    let cases = [
        (
            InterventionAction::Zero {
                dtype: InterventionDtype::Float32,
            },
            vec![1., 0., 3., 0., 5., 0., 7., 0.],
        ),
        (
            InterventionAction::Scale {
                dtype: InterventionDtype::Float32,
                factor: 2.,
            },
            vec![1., 4., 3., 8., 5., 12., 7., 16.],
        ),
        (
            InterventionAction::Mask {
                dtype: InterventionDtype::Float32,
                shape: vec![2, 2],
                keep: vec![true, false, false, true],
            },
            vec![1., 2., 3., 0., 5., 0., 7., 8.],
        ),
        (
            InterventionAction::Replace { tensor: tensor() },
            vec![1., 10., 3., 20., 5., 30., 7., 40.],
        ),
        (
            InterventionAction::Add { tensor: tensor() },
            vec![1., 12., 3., 24., 5., 36., 7., 48.],
        ),
    ];
    for (action, expected) in cases {
        let value = eredu_runtime::intervention::apply_activation(backend, &input, &action, &slice)
            .unwrap();
        assert_eq!(read(&value), expected);
        assert_eq!(
            read(&input),
            vec![1., 2., 3., 4., 5., 6., 7., 8.],
            "primitive mutated its source"
        );
    }
    let row = ResolvedCaptureSlice {
        starts: vec![1, 0],
        ends: vec![2, 4],
        strides: vec![1, 1],
        shape: vec![1, 4],
    };
    let mask = InterventionAction::MaskLogits {
        dtype: InterventionDtype::Float32,
        token_ids: vec![1, 3],
    };
    let value =
        eredu_runtime::intervention::apply_activation(backend, &input, &mask, &row).unwrap();
    assert_eq!(
        read(&value),
        vec![1., 2., 3., 4., 5., f32::NEG_INFINITY, 7., f32::NEG_INFINITY]
    );
    let scale = InterventionAction::Scale {
        dtype: InterventionDtype::Float32,
        factor: 1.,
    };
    let unchanged =
        eredu_runtime::intervention::apply_activation(backend, &input, &scale, &slice).unwrap();
    assert_eq!(read(&unchanged), read(&input));
    assert!(eredu_runtime::intervention::apply_activation(backend, &input, &mask, &slice).is_err());
    for invalid in [
        InterventionAction::Replace {
            tensor: InterventionTensor {
                shape: vec![2],
                values: InterventionValues::Float32(vec![1., 2.]),
            },
        },
        InterventionAction::Scale {
            dtype: InterventionDtype::Float16,
            factor: 1.,
        },
        InterventionAction::Scale {
            dtype: InterventionDtype::Float32,
            factor: f32::INFINITY,
        },
        InterventionAction::Replace {
            tensor: InterventionTensor {
                shape: vec![2, 2],
                values: InterventionValues::Float32(vec![f32::NAN; 4]),
            },
        },
    ] {
        assert!(
            eredu_runtime::intervention::apply_activation(backend, &input, &invalid, &slice)
                .is_err()
        );
    }
}

/// Checks grouped selected-softmax routing with epsilon 0.25, scale 1.5,
/// learned multipliers `[2,3,4,5]`, and two groups of two (one selected).
/// Projection must yield one row `[0, ln(2), ln(3), ln(4)]` with no correction.
pub fn grouped_routing_conformance<M: RoutingMechanism>(
    native: &mut M,
    input: &M::Value,
    read_ids: impl Fn(&M::Value) -> Vec<u32>,
    read_values: impl Fn(&M::Value) -> Vec<f32>,
) {
    let expected =
        eredu_nn::TopKGroupSelectionSpec::new(4, 2, eredu_nn::GroupScoring::SelectedSoftmax, true)
            .unwrap()
            .with_groups(2, 1)
            .unwrap()
            .with_weight_policy(0.25, 1.5)
            .unwrap();
    assert_eq!(native.policy().unwrap(), (expected, true));
    let mut control = GroupSelectionControl {
        expected,
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
                values: vec![10. + 2f32.ln(), 10.],
            };
        }
        let result = execute_routing_intervention(native, input, &control).unwrap();
        let original = result.original.unwrap();
        let original_ids = read_ids(original.group_indices());
        assert!(original_ids.contains(&2) && original_ids.contains(&3));
        for (id, weight) in read_ids(result.effective.group_indices())
            .into_iter()
            .zip(read_values(result.effective.coefficients()))
        {
            assert!(id < 2);
            let wanted = if raw { [1.2, 1.8] } else { [0.8, 2.4] };
            assert!((weight - wanted[id as usize]).abs() < 1e-5);
        }
    }
    control.action = GroupSelectionAction::Force(vec![0, 2]);
    assert!(execute_routing_intervention(native, input, &control).is_err());
}

/// Runs all routing operations/stages through the production shared driver.
/// Fixture: four experts, top-k two, softmax, selected normalization, scale one,
/// no corrections/multipliers/groups; projection yields two rows `[0,1,2,3]`.
/// Readers materialize test-only IDs and coefficients in row-major order.
pub fn routing_conformance<M: RoutingMechanism>(
    native: &mut M,
    input: &M::Value,
    read_ids: impl Fn(&M::Value) -> Vec<u32>,
    read_values: impl Fn(&M::Value) -> Vec<f32>,
) {
    let expected =
        eredu_nn::TopKGroupSelectionSpec::new(4, 2, eredu_nn::GroupScoring::Softmax, true).unwrap();
    assert_eq!(native.policy().unwrap(), (expected, false));
    let mut control = GroupSelectionControl {
        expected,
        learned_coefficient_scale: false,
        first_row: 1,
        end_row: 2,
        row_stride: 1,
        action: GroupSelectionAction::ZeroContribution(vec![0]),
        capture_original: true,
    };
    let baseline = execute_routing_intervention(native, input, &control).unwrap();
    let ordinary_ids = read_ids(baseline.effective.group_indices());
    let ordinary_weights = read_values(baseline.effective.coefficients());
    let mut score_stage_weights = Vec::new();
    for (experiment, action) in [
        GroupSelectionAction::Exclude(vec![3]),
        GroupSelectionAction::ZeroContribution(vec![3]),
        GroupSelectionAction::Bias {
            stage: GroupScoreStage::RawLogits,
            ids: vec![0],
            values: vec![6.],
        },
        GroupSelectionAction::Bias {
            stage: GroupScoreStage::TransformedScores,
            ids: vec![0],
            values: vec![1.],
        },
        GroupSelectionAction::Bias {
            stage: GroupScoreStage::RankingScores,
            ids: vec![0],
            values: vec![6.],
        },
        GroupSelectionAction::Force(vec![0, 1]),
    ]
    .into_iter()
    .enumerate()
    {
        control.action = action;
        let result = execute_routing_intervention(native, input, &control).unwrap();
        let original = result.original.unwrap();
        assert_eq!(read_ids(original.group_indices()), ordinary_ids);
        assert_eq!(read_values(original.coefficients()), ordinary_weights);
        let ids = read_ids(result.effective.group_indices());
        let weights = read_values(result.effective.coefficients());
        assert_eq!(&ids[..2], &ordinary_ids[..2]);
        assert_eq!(&weights[..2], &ordinary_weights[..2]);
        let chosen = &ids[2..];
        let coefficients = &weights[2..];
        match experiment {
            0 => {
                assert!(!chosen.contains(&3));
                assert!(chosen.contains(&1) && chosen.contains(&2));
            }
            1 => {
                assert_eq!(ids, ordinary_ids);
                for (i, id) in chosen.iter().enumerate() {
                    assert_eq!(
                        coefficients[i],
                        if *id == 3 {
                            0.
                        } else {
                            ordinary_weights[i + 2]
                        }
                    );
                }
            }
            2..=4 => {
                assert!(chosen.contains(&0) && chosen.contains(&3));
                score_stage_weights
                    .push(coefficients[chosen.iter().position(|id| *id == 0).unwrap()]);
            }
            5 => assert_eq!(chosen, [0, 1]),
            _ => unreachable!(),
        }
        if experiment != 1 {
            assert!((coefficients.iter().sum::<f32>() - 1.).abs() < 1e-5);
        }
        // Equivalent expert-provider probe: these exact IDs/coefficients determine
        // distinguishable expert contributions, before any generated-text sampling.
        let dispatched: f32 = chosen
            .iter()
            .zip(coefficients)
            .map(|(id, w)| 10f32.powi(*id as i32) * w)
            .sum();
        assert!(dispatched.is_finite());
    }
    assert!(
        score_stage_weights[0] > score_stage_weights[1]
            && score_stage_weights[1] > score_stage_weights[2]
    );
    for action in [
        GroupSelectionAction::Force(vec![4, 1]),
        GroupSelectionAction::Force(vec![1, 1]),
        GroupSelectionAction::Force(vec![1]),
        GroupSelectionAction::Exclude(vec![0, 1, 2]),
        GroupSelectionAction::Bias {
            stage: GroupScoreStage::RawLogits,
            ids: vec![0],
            values: vec![f32::NAN],
        },
    ] {
        control.action = action;
        assert!(execute_routing_intervention(native, input, &control).is_err());
    }
    control.action = GroupSelectionAction::ZeroContribution(vec![0]);
    control.capture_original = false;
    assert!(execute_routing_intervention(native, input, &control)
        .unwrap()
        .original
        .is_none());
}
