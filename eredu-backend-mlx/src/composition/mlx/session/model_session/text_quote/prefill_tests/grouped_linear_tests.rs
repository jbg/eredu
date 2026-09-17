use super::*;
use crate::backend::nn::{shared::MlxNeuralBackend, tensor::GroupedOriginalTestPlan};
use eredu_nn::{
    GroupSelection, GroupedLinearActivation, GroupedLinearOperator, GroupedLinearSpec,
    GroupedNeuralBackend, GroupedProjectionSpec, LinearFormatSpec, ParameterSpec,
};
use safemlx::{Array, Dtype};

#[test]
fn original_bf16_grouped_projection_defers_validation_masks_invalid_ids_and_preserves_values() {
    let stream = fixture::stream();
    let _runtime = safemlx::PrefillRootsRuntime::prepare_for_stream(&stream, &stream).unwrap();
    let projection = GroupedProjectionSpec::new(
        ParameterSpec::trainable("bank.weight").unwrap(),
        None,
        LinearFormatSpec::unscaled(eredu_checkpoint::LinearFormat::Dense).unwrap(),
    )
    .unwrap();
    let spec =
        GroupedLinearSpec::new(3, 32, 4, GroupedLinearActivation::Identity, projection).unwrap();
    let mut bank = MlxNeuralBackend::grouped_linear_bank(spec, &stream).unwrap();
    let weights: Vec<f32> = (0..3 * 4 * 32)
        .map(|i| ((i * 7 % 19) as f32 - 9.0) / 32.0)
        .collect();
    let weights = Array::from_slice(&weights, &[3, 4, 32])
        .as_dtype(Dtype::Bfloat16, &stream)
        .unwrap();
    bank.bind_local_parameters(std::collections::BTreeMap::from([(
        "weight".to_owned(),
        weights.clone(),
    )]))
    .unwrap();
    let input: Vec<f32> = (0..64).map(|i| ((i * 5 % 13) as f32 - 6.0) / 8.0).collect();
    let input = crate::MlxTensor::from_array(
        Array::from_slice(&input, &[2, 32])
            .as_dtype(Dtype::Bfloat16, &stream)
            .unwrap(),
    );
    let coefficients =
        crate::MlxTensor::from_array(Array::from_slice(&[0.3f32, 0.7, 0.6, 0.4], &[2, 2]));
    let valid_ids = crate::MlxTensor::from_array(Array::from_slice(&[2i32, 0, 1, 2], &[2, 2]));
    let invalid_ids = crate::MlxTensor::from_array(Array::from_slice(&[-1i32, 3, 1, 2], &[2, 2]));
    let valid = GroupSelection::new(
        valid_ids.clone(),
        coefficients.clone(),
        coefficients.clone(),
    );
    let invalid = GroupSelection::new(
        invalid_ids.clone(),
        coefficients.clone(),
        coefficients.clone(),
    );
    let ordinary = bank.forward_grouped(&input, &valid, &stream).unwrap();
    let expected = ordinary
        .as_array()
        .evaluated()
        .unwrap()
        .try_to_vec::<f32>()
        .unwrap();
    assert!(expected.iter().all(|n| n.is_finite()));
    assert!(expected.iter().any(|n| *n != 0.0));
    let ordinary_error = bank.forward_grouped(&input, &invalid, &stream).unwrap_err();
    assert!(ordinary_error.to_string().contains("outside"));
    drop((ordinary, ordinary_error));

    // The reference has completed these leaves. Finish their ordinary waits
    // and detach the ordinary events before entering a new original domain;
    // merely completing the reference output does not mark each input available.
    for leaf in [
        input.as_array(),
        &weights,
        coefficients.as_array(),
        valid_ids.as_array(),
        invalid_ids.as_array(),
    ] {
        leaf.evaluated().unwrap();
    }

    for invalid_case in [false, true] {
        let plan = GroupedOriginalTestPlan::new();
        POINTWISE_CONTROLS.with(|slot| assert!(slot.replace(Some(plan.control_bytes())).is_none()));
        let reset = PointwiseControlsReset;
        let ids = if invalid_case {
            &invalid_ids
        } else {
            &valid_ids
        };
        let selections = if invalid_case { &invalid } else { &valid };
        let (result, pool) =
            with_registered_original_operation_controls(|controls, observer, pool| {
                safemlx::OperationEvent::validate_traversal_context(observer)
                    .expect("prepared original grouped traversal context");
                for (index, leaf) in [
                    input.as_array(),
                    &weights,
                    coefficients.as_array(),
                    ids.as_array(),
                ]
                .into_iter()
                .enumerate()
                {
                    safemlx::OperationEvent::validate_traversal_leaf(leaf, observer)
                        .unwrap_or_else(|error| {
                            panic!(
                                "grouped leaf {index} {:?} {:?}: {error:?}",
                                leaf.dtype(),
                                leaf.shape()
                            )
                        });
                }
                (
                    plan.run(controls, observer, &stream, || {
                        bank.forward_grouped(&input, selections, &stream).unwrap()
                    }),
                    pool.clone(),
                )
            });
        drop(reset);
        if invalid_case {
            let error = result.unwrap_err();
            assert!(error.to_string().contains("outside 0..3"));
            drop(error);
        } else {
            let output = result.unwrap();
            let actual = output
                .as_array()
                .evaluated()
                .unwrap()
                .try_to_vec::<f32>()
                .unwrap();
            assert_eq!(actual, expected);
            assert!(
                pool.used_bytes().unwrap() > 0,
                "published output lost original custody"
            );
            drop(output);
        }
        fixture::settle(&pool, 0);
    }
}

#[test]
fn original_grouped_chunk_destination_refuses_growth_and_retains_detached_custody() {
    let stream = fixture::stream();
    let _runtime = safemlx::PrefillRootsRuntime::prepare_for_stream(&stream, &stream).unwrap();
    let values = [
        Array::from_slice(&[1.0f32], &[1]),
        Array::from_slice(&[2.0f32], &[1]),
        Array::from_slice(&[3.0f32], &[1]),
    ];
    for value in &values {
        value.evaluated().unwrap();
    }
    POINTWISE_CONTROLS.with(|slot| {
        assert!(
            slot.replace(Some(
                GroupedOriginalTestPlan::chunk_storage_test_control_bytes(),
            ))
            .is_none()
        )
    });
    let reset = PointwiseControlsReset;
    let (output, pool) = with_registered_original_operation_controls(|controls, observer, pool| {
        (
            GroupedOriginalTestPlan::run_chunk_storage_test(controls, observer, values),
            pool.clone(),
        )
    });
    drop(reset);
    assert_eq!(output.as_slice().len(), 2);
    assert_eq!(
        output.as_slice()[0].evaluated().unwrap().as_slice::<f32>(),
        &[1.0]
    );
    assert_eq!(
        output.as_slice()[1].evaluated().unwrap().as_slice::<f32>(),
        &[2.0]
    );
    assert!(
        pool.used_bytes().unwrap() > 0,
        "detached destination lost original custody"
    );
    drop(output);
    fixture::settle(&pool, 0);
}

#[test]
fn original_attention_key_blocks_match_full_reference_and_retire_each_pass() {
    use crate::backend::nn::workspace::OriginalAttentionTestPlan;
    use safemlx::{OperationEvent, ops};
    const KEYS: usize = 8193;
    // The factory prepares real admitted streams and source families before
    // any ordinary reference work; the original callback uses this exact stream.
    let mut prepared = OriginalOperationFixture::prepare(true);
    let stream = prepared.stream.clone();
    let q = Array::from_slice(&[0.35f32, -0.2, 0.45, 0.1], &[1, 1, 1, 4]);
    let keys: Vec<f32> = (0..KEYS * 4)
        .map(|i| ((i * 7 % 37) as f32 - 18.0) / 29.0)
        .collect();
    let values: Vec<f32> = (0..KEYS * 3)
        .map(|i| ((i * 11 % 43) as f32 - 12.0) / 31.0)
        .collect();
    let allowed: Vec<bool> = (0..KEYS).map(|i| i % 11 != 0).collect();
    let k = Array::from_slice(&keys, &[1, 1, KEYS as i32, 4]);
    let v = Array::from_slice(&values, &[1, 1, KEYS as i32, 3]);
    let mask = Array::from_slice(&allowed, &[1, 1, 1, KEYS as i32]);
    let sinks = Array::from_slice(&[0.2f32], &[1]);
    let expected = {
        // Independent full-score ordinary reference, without the tiled worker.
        // Include its learned sink in the softmax denominator and remove that
        // column before the complete value product.
        let scores = ops::matmul(&q, &k.swap_axes(-1, -2, &stream).unwrap(), &stream)
            .unwrap()
            .multiply(Array::from_f32(0.5), &stream)
            .unwrap();
        let scores = ops::tanh(
            &scores
                .multiply(Array::from_f32(1.75f32.recip()), &stream)
                .unwrap(),
            &stream,
        )
        .unwrap()
        .multiply(Array::from_f32(1.75), &stream)
        .unwrap();
        let scores =
            ops::r#where(&mask, scores, Array::from_f32(f32::NEG_INFINITY), &stream).unwrap();
        let sink = sinks.reshape(&[1, 1, 1, 1], &stream).unwrap();
        let scores = ops::concatenate_axis(&[scores, sink], -1, &stream).unwrap();
        let probabilities = ops::softmax_axis(&scores, -1, true, &stream).unwrap();
        let probabilities = probabilities
            .try_slice(&[0; 4], &[1, 1, 1, KEYS as i32], &[1; 4], &stream)
            .unwrap();
        let output = ops::matmul(&probabilities, &v, &stream).unwrap();
        let result = output.evaluated().unwrap().try_to_vec::<f32>().unwrap();
        assert!(result.iter().all(|x| x.is_finite()));
        assert!(result.iter().any(|x| x.abs() > 0.01));
        result
    };
    // Detach completed ordinary events before crossing the original boundary.
    for leaf in [&q, &k, &v, &mask, &sinks] {
        leaf.evaluated().unwrap();
    }
    let plan = OriginalAttentionTestPlan::new();
    let quote_guard = plan.quote();
    POINTWISE_CONTROLS.with(|slot| assert!(slot.replace(Some(plan.control_bytes())).is_none()));
    let controls_reset = PointwiseControlsReset;
    let completion = plan.completion;
    let baseline = prepared.pool.used_bytes().unwrap();
    let baseline_unquoted = prepared.pool.unquoted_owner_count().unwrap();
    let output = with_prepared_original_operation_controls(
        None,
        &mut prepared,
        |controls, observer, _, destination| {
            assert!(destination.is_none());
            for leaf in [&q, &k, &v, &mask, &sinks] {
                OperationEvent::validate_traversal_leaf(leaf, observer).unwrap();
            }
            GroupedOriginalTestPlan::with_component_outputs(
                completion.grouped_outputs,
                controls,
                || {
                    let mut bank =
                        OperationEvent::prepare_resident_graph(completion.graph, observer).unwrap();
                    bank.configure_nested_completions(
                        &completion.nested_traversal().unwrap(),
                        completion.nested_completions,
                    )
                    .unwrap();
                    let output = crate::backend::nn::attention::attention_with_softcap(
                        &q,
                        &k,
                        &v,
                        0.5,
                        Some(&mask),
                        Some(&sinks),
                        Some(1.75),
                        eredu_nn::AttentionArithmetic::InputScores,
                        &stream,
                    )
                    .unwrap();
                    assert!(
                        observer.status().is_settled(),
                        "all real block callback frontiers retired"
                    );
                    assert!(!observer.status().failed());
                    assert!(
                        OperationEvent::validate_nested_completion(1).is_err(),
                        "all 66 declared attempts consumed"
                    );
                    drop(bank);
                    let event = safemlx::transforms::async_eval_with_original_prepared_traversal(
                        [&output].into_iter(),
                        observer,
                        &stream,
                        &completion.traversal,
                    )
                    .unwrap();
                    event.synchronize().unwrap();
                    let evaluated = output.completed_in_original_scope(observer).unwrap();
                    let actual = evaluated.try_as_slice::<f32>().unwrap();
                    assert_eq!(actual.len(), expected.len());
                    for (actual, expected) in actual.iter().zip(&expected) {
                        assert!(
                            (actual - expected).abs() <= 2e-5,
                            "blockwise {actual}, full reference {expected}"
                        );
                    }
                    drop(evaluated);
                    safemlx::try_with_submission_retirement(|| drop(event)).unwrap();
                    output
                },
            )
        },
    );
    assert!(
        quote_guard.was_quoted(),
        "component joined native capacities before admission"
    );
    drop((quote_guard, controls_reset));
    assert!(
        prepared.pool.used_bytes().unwrap() > baseline,
        "escaped output keeps actual original custody"
    );
    drop(output);
    crate::backend::submission_recovery::wait_for_retirement(|| {
        disk::reclaim();
        prepared.pool.used_bytes().unwrap() == baseline
            && prepared.pool.unquoted_owner_count().unwrap() == baseline_unquoted
    });
    drop((q, k, v, mask, sinks, stream));
    prepared.finish();
}
