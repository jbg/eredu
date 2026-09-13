//! Separate embedding and stream projections remain actual prediction inputs.
use super::*;

fn project(input: &NumericTensor, weight: &NumericTensor) -> NumericTensor {
    let width = weight.shape[1] as usize;
    let mut shape = input.shape.clone();
    *shape.last_mut().unwrap() = weight.shape[0];
    NumericTensor::new(
        shape,
        input
            .data
            .chunks_exact(width)
            .flat_map(|row| {
                weight.data.chunks_exact(width).map(move |weights| {
                    row.iter()
                        .zip(weights)
                        .map(|(a, b)| *a as f64 * *b as f64)
                        .sum::<f64>() as f32
                })
            })
            .collect(),
    )
}

#[test]
fn v4_prediction_separate_projection_fusion_and_observed_drivers_match_cached_execution() {
    let mut config = tiny_v4_config();
    config["num_nextn_predict_layers"] = 1.into();
    config["compress_ratios"] = serde_json::json!([0, 4, 128, 0]);
    let graph = eredu_architectures::configuration::MODEL_CONFIGURATIONS
        .resolve_safetensors(&config)
        .unwrap()
        .architecture_plan()
        .architecture_descriptor();
    let scope = &graph.component_scopes[0];
    let args = deepseek::parse_v4_config(&config).unwrap();
    let context = NumericContext::default();
    let parallel = NumericParallelContext::new(0, NumericParallelGroup::new(1));
    let layer =
        || deepseek::mtp::V4PredictionLayer::<NumericBackend>::new(&args, 0, &context).unwrap();
    let head = || {
        NumericBackend::linear(
            LinearSpec {
                input: args.hidden_size,
                output: args.vocab_size,
                weight: ParameterSpec::trainable("head.weight").unwrap(),
                bias: None,
                format: eredu_nn::LinearFormatSpec::unscaled(args.linear_format).unwrap(),
            },
            &context,
        )
        .unwrap()
    };
    let mut parameters = Parameters::default();
    layer().visit_parameters(&mut parameters);
    head().visit_parameters(&mut parameters);
    for mode in 0..4 {
        for masked in [false, true] {
            let mut candidate = layer();
            let mut ordinary = layer();
            let mut candidate_head = head();
            let mut ordinary_head = head();
            let mut cache = NumericPoolingCache::new(args.sliding_window, &[]);
            let mut ordinary_cache = NumericPoolingCache::new(args.sliding_window, &[]);
            for (step, ids) in [vec![1, 2, 3], vec![4], vec![5]].into_iter().enumerate() {
                let rows = ids.len();
                let tokens = NumericTensor::token_ids(&ids);
                let embedded = NumericTensor::new(
                    [1, rows as i32, 4],
                    (0..rows * 4)
                        .map(|i| ((i * 17 + step * 3) % 31) as f32 * 0.04 - 0.6)
                        .collect(),
                );
                let hidden = NumericTensor::new(
                    [1, rows as i32, 2, 4],
                    (0..rows * 8)
                        .map(|i| ((i * 13 + step * 7) % 43) as f32 * 0.03 - 0.6)
                        .collect(),
                );
                let mut capture = Components {
                    zero: masked.then_some(("mtp.0.prediction.embedding.projected", rows - 1, 1)),
                    ..Components::strict()
                };
                let pass = if step == 0 {
                    ExpertPass::Prefill
                } else {
                    ExpertPass::Decode
                };
                let actual = match mode {
                    0 => candidate.forward_observed(
                        "mtp.0",
                        &hidden,
                        &embedded,
                        &tokens,
                        &mut cache,
                        &mut candidate_head,
                        &context,
                        &mut capture,
                    ),
                    1 => candidate.forward_observed_with_provider(
                        "mtp.0",
                        &hidden,
                        &embedded,
                        &tokens,
                        &mut cache,
                        &mut candidate_head,
                        pass,
                        &mut ResidentExpertProvider,
                        &context,
                        &mut capture,
                    ),
                    2 => candidate.forward_parallel_observed(
                        "mtp.0",
                        &hidden,
                        &embedded,
                        &tokens,
                        &mut cache,
                        &mut candidate_head,
                        &parallel,
                        &context,
                        &mut capture,
                        |v, _| Ok(v),
                    ),
                    _ => candidate.forward_parallel_observed_with_provider(
                        "mtp.0",
                        &hidden,
                        &embedded,
                        &tokens,
                        &mut cache,
                        &mut candidate_head,
                        pass,
                        &mut ResidentExpertProvider,
                        &parallel,
                        &context,
                        &mut capture,
                        |v, _| Ok(v),
                    ),
                }
                .unwrap();
                let expected = ordinary
                    .forward(
                        &hidden,
                        &embedded,
                        &tokens,
                        &mut ordinary_cache,
                        &mut ordinary_head,
                        &context,
                    )
                    .unwrap();
                if !masked {
                    assert_tensor_exact(
                        &actual.logits,
                        &expected.logits,
                        "observed V4 prediction head",
                    );
                    assert_tensor_exact(
                        &actual.hidden,
                        &expected.hidden,
                        "observed V4 prediction residual",
                    );
                }
                for term in ["embedding", "hidden"] {
                    let prefix = format!("mtp.0.prediction.{term}");
                    let projected = project(
                        &capture.values[&format!("{prefix}.projection_input")],
                        &parameters.0[if term == "embedding" {
                            "mtp.0.e_proj.weight"
                        } else {
                            "mtp.0.h_proj.weight"
                        }],
                    );
                    assert_tensor_close(
                        &projected,
                        &capture.values[&format!("{prefix}.projected")],
                        "actual separate prediction projection",
                    );
                }
                let embedding = &capture.values["mtp.0.prediction.embedding.projected.effective"];
                let hidden = &capture.values["mtp.0.prediction.hidden.projected.effective"];
                let fused = &capture.values["mtp.0.prediction.fusion.output"];
                for row in 0..rows {
                    for stream in 0..2 {
                        for h in 0..4 {
                            let i = (row * 2 + stream) * 4 + h;
                            assert_eq!(fused.data[i], embedding.data[row * 4 + h] + hidden.data[i]);
                        }
                    }
                }
                assert_tensor_exact(
                    &capture.values["mtp.0.prediction.fusion.output.effective"],
                    &capture.values["mtp.0.decoder.input"],
                    "decoder consumes effective fusion",
                );
                if masked {
                    assert_eq!(embedding.data[(rows - 1) * 4 + 1], 0.0);
                    assert_ne!(actual.logits.data, expected.logits.data);
                }
                reconstruct_scope(scope, &capture, &parameters, &actual.logits);
                assert_tensor_exact(
                    &capture.values["mtp.0.output.effective"],
                    &actual.hidden,
                    "effective outer prediction state",
                );
                assert_eq!(cache.offset(), ordinary_cache.offset());
                assert_eq!(
                    capture.values["mtp.0.prediction.readout.stream_coefficients"].shape,
                    [1, rows as i32, 2]
                );
            }
        }
    }
}

pub(super) fn reconstruct_scope(
    scope: &eredu_core::component::ComponentExecutionScope,
    capture: &Components,
    parameters: &Parameters,
    scores: &NumericTensor,
) {
    use eredu_core::component::{ComponentFusionExpansion, ComponentResidualBase};
    let mixing = scope.readout.stream_residual.as_ref().unwrap();
    let streams = mixing.streams;
    let output = match &scope.residual_base {
        ComponentResidualBase::ProjectedSum { output, .. }
        | ComponentResidualBase::Source { output, .. } => output,
        _ => panic!("V4 stream base"),
    };
    let fused = &capture.values[output];
    let hidden = *fused.shape.last().unwrap() as usize;
    let rows = fused.data.len() / (streams * hidden);
    let expand = |value: &NumericTensor, expansion: &ComponentFusionExpansion| -> Vec<f64> {
        match expansion {
            ComponentFusionExpansion::Identity => value.data.iter().map(|v| *v as f64).collect(),
            ComponentFusionExpansion::BroadcastAxis {
                axis: 2, extent, ..
            } => {
                assert_eq!(*extent, streams);
                (0..rows * streams * hidden)
                    .map(|i| value.data[i / (streams * hidden) * hidden + i % hidden] as f64)
                    .collect()
            }
            _ => panic!("unexpected V4 source expansion"),
        }
    };
    let mut terms = match &scope.residual_base {
        ComponentResidualBase::ProjectedSum { inputs, .. } => inputs
            .iter()
            .map(|input| {
                assert_tensor_close(
                    &project(
                        &capture.values[&input.projection_input],
                        &parameters.0[&input.weight],
                    ),
                    &capture.values[&input.output],
                    "descriptor resolves actual separate matrix",
                );
                expand(&capture.values[&input.effective_output], &input.expansion)
            })
            .collect::<Vec<_>>(),
        ComponentResidualBase::Source {
            input, expansion, ..
        } => vec![expand(&capture.values[input], expansion)],
        _ => unreachable!(),
    };
    for i in 0..fused.data.len() {
        assert!((terms.iter().map(|t| t[i]).sum::<f64>() - fused.data[i] as f64).abs() < 2e-6);
    }
    for cycle in &mixing.cycles {
        let coefficients = &capture.values[&cycle.combination];
        for term in &mut terms {
            let mut mixed = vec![0.; term.len()];
            for row in 0..rows {
                for to in 0..streams {
                    for h in 0..hidden {
                        mixed[(row * streams + to) * hidden + h] = (0..streams)
                            .map(|from| {
                                term[(row * streams + from) * hidden + h]
                                    * coefficients.data[(row * streams + from) * streams + to]
                                        as f64
                            })
                            .sum();
                    }
                }
            }
            *term = mixed;
        }
        let post = &capture.values[&cycle.post];
        let write = &capture.values[&cycle.write];
        terms.push(
            (0..rows * streams * hidden)
                .map(|i| {
                    post.data[i / hidden] as f64
                        * write.data[i / (streams * hidden) * hidden + i % hidden] as f64
                })
                .collect(),
        );
        let actual = &capture.values[&cycle.output];
        for i in 0..actual.data.len() {
            assert!(
                (terms.iter().map(|t| t[i]).sum::<f64>() - actual.data[i] as f64).abs() < 2e-4,
                "{}",
                cycle.output
            );
        }
    }
    let readout = &scope.readout;
    let coefficients = &capture.values[&mixing.head.coefficients];
    let residual = &capture.values[&readout.residual];
    let gain = &parameters.0[readout.normalization.gain.as_ref().unwrap()];
    let head = &parameters.0[&readout.weight];
    for row in 0..rows {
        let rms = (residual.data[row * hidden..(row + 1) * hidden]
            .iter()
            .map(|v| (*v as f64).powi(2))
            .sum::<f64>()
            / hidden as f64
            + readout.normalization.epsilon.value() as f64)
            .sqrt();
        let mut reconstructed = [0.; 2];
        for term in &terms {
            for (selected, token) in [2usize, 7].into_iter().enumerate() {
                reconstructed[selected] += (0..hidden)
                    .map(|h| {
                        let collapsed = (0..streams)
                            .map(|stream| {
                                term[(row * streams + stream) * hidden + h]
                                    * coefficients.data[row * streams + stream] as f64
                            })
                            .sum::<f64>();
                        collapsed / rms * gain.data[h] as f64 * head.data[token * hidden + h] as f64
                    })
                    .sum::<f64>();
            }
        }
        for write in &readout.score_writes {
            let projected = project(
                &capture.values[&write.projection_input],
                &parameters.0[&write.weight],
            );
            assert_tensor_close(
                &projected,
                &capture.values[&write.output],
                "dynamic score projection",
            );
            assert_eq!(write.broadcast_axes, ["sequence"]);
            let effective = &capture.values[&write.effective_output];
            assert_eq!(effective.shape[1], 1);
            for (selected, token) in [2usize, 7].into_iter().enumerate() {
                reconstructed[selected] += effective.data[token] as f64;
            }
        }
        let expected = [
            scores.data[row * 16 + 2] as f64,
            scores.data[row * 16 + 7] as f64,
        ];
        for (actual, expected) in reconstructed.iter().zip(expected) {
            assert!(
                (actual - expected).abs() < 3e-4,
                "prediction score reconstruction {actual} vs {expected}"
            );
        }
        assert!(((reconstructed[0] - reconstructed[1]) - (expected[0] - expected[1])).abs() < 3e-4);
    }
}

#[test]
fn v4_prediction_outer_and_inner_boundaries_match_all_observed_traversals() {
    let mut config = tiny_v4_config();
    config["num_nextn_predict_layers"] = 1.into();
    config["compress_ratios"] = serde_json::json!([0, 4, 128, 0]);
    let args = deepseek::parse_v4_config(&config).unwrap();
    let context = NumericContext::default();
    let runtime = |parallel| {
        let model = if parallel {
            let mut groups = deepseek::parallel::v4_static_parameter_groups(&args).unwrap();
            for layer in 0..4 {
                groups.extend(deepseek::parallel::v4_layer_parameter_groups(&args, layer).unwrap());
            }
            let layout = numeric_local_layout(&groups, 1, 0).unwrap();
            let geometry = deepseek::parallel::v4_local_geometry(&args, &layout).unwrap();
            deepseek::v4::Model::<NumericBackend>::new_parallel(args.clone(), geometry, &context)
                .unwrap()
        } else {
            deepseek::v4::Model::<NumericBackend>::new(args.clone(), &context).unwrap()
        };
        LayerwiseRuntime::new(model, RebuildingUnitPolicy::default())
    };
    let state = || {
        DeviceState::<NumericBackend, _>::create(
            deepseek::v4::state_layout(&args).unwrap(),
            |layer, _| {
                let ratios = match args.attention_policy(layer).unwrap() {
                    deepseek::V4AttentionPolicy::Local => vec![],
                    deepseek::V4AttentionPolicy::Compressed { ratio: 4 } => vec![4, 4],
                    deepseek::V4AttentionPolicy::Compressed { ratio } => vec![ratio],
                };
                Ok::<_, Error>(NumericPoolingCache::new(args.sliding_window, &ratios))
            },
        )
        .unwrap()
    };
    let parallel = NumericParallelContext::new(0, NumericParallelGroup::new(1));
    for mode in 0..4 {
        for masked in [false, true] {
            let mut candidate = runtime(mode >= 2);
            let mut ordinary = runtime(false);
            let (mut state, mut original_state) = (state(), state());
            for (step, ids) in [vec![1, 2, 3], vec![4], vec![5]].into_iter().enumerate() {
                let tokens = NumericTensor::token_ids(&ids);
                let hidden = NumericTensor::new(
                    [1, ids.len() as i32, 2, 4],
                    (0..ids.len() * 8)
                        .map(|i| ((i * 11 + step * 7) % 31) as f32 * 0.04 - 0.6)
                        .collect(),
                );
                let input = || deepseek::mtp::EmbeddedInput::draft(&tokens, &hidden, 0);
                let mut capture = Components {
                    zero: masked.then_some((
                        "mtp.0.prediction.embedding.projected",
                        ids.len() - 1,
                        1,
                    )),
                    ..Components::strict()
                };
                let pass = if step == 0 {
                    ExpertPass::Prefill
                } else {
                    ExpertPass::Decode
                };
                let actual = match mode {
                    0 => {
                        candidate.forward_with_observer(input(), &mut state, &context, &mut capture)
                    }
                    1 => candidate.forward_with_provider_and_observer(
                        input(),
                        &mut state,
                        pass,
                        &mut ResidentExpertProvider,
                        &context,
                        &mut capture,
                    ),
                    2 => candidate.forward_parallel_with_observer(
                        input(),
                        &mut state,
                        &parallel,
                        &context,
                        &mut capture,
                    ),
                    _ => candidate.forward_parallel_with_provider_and_observer(
                        input(),
                        &mut state,
                        pass,
                        &mut ResidentExpertProvider,
                        &parallel,
                        &context,
                        &mut capture,
                    ),
                }
                .unwrap();
                let expected = ordinary
                    .forward(input(), &mut original_state, &context)
                    .unwrap();
                if masked {
                    assert_ne!(actual.data, expected.data);
                } else {
                    assert_tensor_exact(&actual, &expected, "observed prediction traversal");
                }
                for suffix in [
                    "input",
                    "input.effective",
                    "output",
                    "output.effective",
                    "capture",
                    "decoder.input",
                    "decoder.output",
                ] {
                    assert!(
                        capture.values.contains_key(&format!("mtp.0.{suffix}")),
                        "missing {suffix}"
                    );
                }
                assert_tensor_exact(
                    &actual,
                    &capture.values["mtp.0.prediction.readout.linear.effective"],
                    "selected prediction scores",
                );
            }
        }
    }
}
