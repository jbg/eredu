//! Prediction primitives reuse decoder hooks; public speculative admission is separate.
use super::*;
use eredu_runtime::ResidentExpertProvider;

fn projected(input: &NumericTensor, weight: &NumericTensor) -> NumericTensor {
    let width = weight.shape[1] as usize;
    NumericTensor::new(
        [1, input.shape[1], weight.shape[0]],
        input
            .data
            .chunks_exact(width)
            .flat_map(|row| {
                weight.data.chunks_exact(width).map(move |weights| {
                    row.iter()
                        .zip(weights)
                        .map(|(a, b)| f64::from(*a) * f64::from(*b))
                        .sum::<f64>() as f32
                })
            })
            .collect(),
    )
}

#[test]
fn v3_prediction_hooks_reconstruct_writes_and_change_consumed_values() {
    const PATH: &str = "model.layers.1";
    const CHANNELS: &str = "model.layers.1.attention.channels";
    const UNITS: &str = "model.layers.1.feed_forward.shared.units";
    const NORMALIZED: &str = "model.layers.1.prediction.hidden.normalized";
    const RESIDUAL: &str = "model.layers.1.prediction.readout.residual";
    let context = NumericContext::default();
    for query_rank in [None, Some(3)] {
        let mut config = v3_config(query_rank);
        config["num_nextn_predict_layers"] = 1.into();
        let descriptor = describe(&config);
        let args = deepseek::parse_v3_config(&config).unwrap();
        let layer =
            deepseek::mtp::V3PredictionLayer::<NumericBackend>::new(&args, 0, &context).unwrap();
        let mut parameters = Parameters::default();
        layer.visit_parameters(&mut parameters);
        for paged in [false, true] {
            let empty = if paged {
                NumericCompressedCache::paged(2)
            } else {
                NumericCompressedCache::resident()
            };
            // Parallel variants here use one rank: this verifies the shared head
            // and decoder dispatch, not distributed acceptance.
            for mode in 0..3 {
                for trial in 0..6 {
                    let mut candidate = layer.clone();
                    let mut ordinary = layer.clone();
                    let mut cache = empty.clone();
                    let mut ordinary_cache = empty.clone();
                    for step in 0..3 {
                        let count = if step == 0 { 3 } else { 1 };
                        let tokens = NumericTensor::token_ids(&vec![1 + step; count]);
                        let hidden = NumericTensor::new(
                            [1, count as i32, 8],
                            (0..count * 8)
                                .map(|i| (i as f32 * 0.4 - 2.0 + step as f32).sin())
                                .collect(),
                        );
                        let embedded = NumericTensor::new(
                            [1, count as i32, 8],
                            (0..count * 8)
                                .map(|i| (i as f32 * 0.7 + 0.3 - step as f32).cos())
                                .collect(),
                        );
                        let row = count - 1;
                        let mut capture = Components::default();
                        match trial {
                            1 => capture.zero = Some((CHANNELS, row, 4)),
                            2 => capture.keep = Some((CHANNELS, row, 4)),
                            3 => capture.zero = Some((UNITS, row, 2)),
                            4 => capture.zero = Some((NORMALIZED, row, 3)),
                            5 => capture.zero = Some((RESIDUAL, row, 2)),
                            _ => {}
                        }
                        let pass = if step == 0 {
                            ExpertPass::Prefill
                        } else {
                            ExpertPass::Decode
                        };
                        let output = match mode {
                            0 => candidate.forward_observed_with_provider(
                                PATH,
                                &hidden,
                                &embedded,
                                &tokens,
                                &mut cache,
                                pass,
                                &mut ResidentExpertProvider,
                                &context,
                                &mut capture,
                            ),
                            1 => candidate.forward_parallel_observed(
                                PATH,
                                &hidden,
                                &embedded,
                                &tokens,
                                &mut cache,
                                &context,
                                &mut capture,
                                |value, _| Ok(value),
                            ),
                            _ => candidate.forward_parallel_observed_with_provider(
                                PATH,
                                &hidden,
                                &embedded,
                                &tokens,
                                &mut cache,
                                pass,
                                &mut ResidentExpertProvider,
                                &context,
                                &mut capture,
                                |value, _| Ok(value),
                            ),
                        }
                        .unwrap();
                        let baseline = ordinary
                            .forward(&hidden, &embedded, &tokens, &mut ordinary_cache, &context)
                            .unwrap();
                        if trial == 0 {
                            assert_tensor_exact(
                                &output.hidden,
                                &baseline.hidden,
                                "MTP no-op hidden",
                            );
                            assert_tensor_exact(&output.logits, &baseline.logits, "MTP no-op head");
                        } else {
                            assert_ne!(
                                output.logits.data, baseline.logits.data,
                                "MTP edit affects head: mode={mode} trial={trial} step={step}"
                            );
                        }
                        for (activation, write, weight) in [
                            (
                                "attention.channels.effective",
                                "attention.write",
                                "self_attn.o_proj.weight",
                            ),
                            (
                                "feed_forward.shared.units.effective",
                                "feed_forward.shared.write",
                                "mlp.shared_experts.down_proj.weight",
                            ),
                            (
                                "prediction.readout.normalized.effective",
                                "prediction.readout.linear",
                                "shared_head.head.weight",
                            ),
                        ] {
                            assert_tensor_close(
                                &projected(
                                    &capture.values[&format!("{PATH}.{activation}")],
                                    &parameters.0[&format!("{PATH}.{weight}")],
                                ),
                                &capture.values[&format!("{PATH}.{write}")],
                                "MTP actual component/head reconstruction",
                            );
                        }
                        let effective_residual = &capture.values[&format!("{RESIDUAL}.effective")];
                        assert_tensor_exact(
                            &output.hidden,
                            effective_residual,
                            "MTP returned hidden is effective head input",
                        );
                        let gain = &parameters.0[&format!("{PATH}.shared_head.norm.weight")];
                        let normalized = NumericTensor::new(
                            effective_residual.shape.clone(),
                            effective_residual
                                .data
                                .as_chunks::<8>()
                                .0
                                .iter()
                                .flat_map(|row| {
                                    let scale =
                                        (row.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>()
                                            / 8.0
                                            + f64::from(args.rms_norm_eps))
                                        .sqrt();
                                    row.iter().zip(&gain.data).map(move |(x, gain)| {
                                        (f64::from(*x) / scale * f64::from(*gain)) as f32
                                    })
                                })
                                .collect(),
                        );
                        assert_tensor_close(
                            &normalized,
                            &capture.values[&format!("{PATH}.prediction.readout.normalized")],
                            "MTP head renormalizes edited residual",
                        );
                        verify_scope_reconstruction(
                            &descriptor,
                            &capture,
                            &parameters,
                            &output.logits,
                        );
                        if let Some((target, position, component)) = capture.zero.or(capture.keep) {
                            let original = &capture.values[target];
                            let effective = &capture.values[&format!("{target}.effective")];
                            let width = *original.shape.last().unwrap() as usize;
                            for (index, (before, after)) in
                                original.data.iter().zip(&effective.data).enumerate()
                            {
                                let selected = index / width == position
                                    && ((index % width == component) != capture.keep.is_some());
                                assert_eq!(*after, if selected { 0.0 } else { *before });
                            }
                        }
                        assert_eq!(cache.offset(), 3 + step as i32);
                        assert_eq!(output.tokens.data, tokens.data);
                    }
                }
            }
        }
    }
}

#[test]
fn v3_prediction_observed_traversal_preserves_depth_state_and_siblings() {
    let context = NumericContext::default();
    let mut config = v3_config(Some(3));
    config["num_nextn_predict_layers"] = 2.into();
    let args = deepseek::parse_v3_config(&config).unwrap();
    let runtime = |parallel| {
        let model = if parallel {
            let mut groups = deepseek::parallel::v3_static_parameter_groups(&args).unwrap();
            for layer in 0..3 {
                groups.extend(deepseek::parallel::v3_layer_parameter_groups(&args, layer).unwrap());
            }
            let layout = numeric_local_layout(&groups, 1, 0).unwrap();
            let geometry = deepseek::parallel::v3_local_geometry(&args, &layout).unwrap();
            deepseek::v3::Model::<NumericBackend>::new_parallel(args.clone(), geometry, &context)
                .unwrap()
        } else {
            deepseek::v3::Model::<NumericBackend>::new(args.clone(), &context).unwrap()
        };
        let units = (0..3)
            .map(|group| model.construct_unit(group, 0, &context).unwrap())
            .collect();
        LayerwiseRuntime::new(model, ResidentUnitWindow::new(units))
    };
    let mut parent_state = DeviceState::<NumericBackend, _>::create(
        deepseek::v3::state_layout(&args).unwrap(),
        |_, _| Ok::<_, Error>(NumericCompressedCache::paged(2)),
    )
    .unwrap();
    let target_tokens = NumericTensor::token_ids(&[1, 2, 3]);
    let (_, target) = runtime(false)
        .forward_with_context_hook(
            deepseek::mtp::EmbeddedInput::target(&target_tokens, None),
            &mut parent_state,
            &context,
            |_, _, _| Ok(()),
        )
        .unwrap();
    let prior = target.target_capture().unwrap();
    for mode in 0..4 {
        for depth in 0..2 {
            for masked in [false, true] {
                let mut candidate = runtime(mode >= 2);
                let mut ordinary = runtime(false);
                let mut state = parent_state.clone();
                let mut ordinary_state = parent_state.clone();
                let parallel = NumericParallelContext::new(0, NumericParallelGroup::new(1));
                let mut first = None;
                for step in 0..3 {
                    let ids = if step == 0 {
                        vec![4, 5, 6]
                    } else {
                        vec![7 + step]
                    };
                    let tokens = NumericTensor::token_ids(&ids);
                    let hidden = if step == 0 {
                        prior.clone()
                    } else {
                        prior.axis_slice(1, 2, 3)
                    };
                    let input = || deepseek::mtp::EmbeddedInput::draft(&tokens, &hidden, depth);
                    let mut capture = Components {
                        zero: masked.then_some((
                            if depth == 0 {
                                "model.layers.1.attention.channels"
                            } else {
                                "model.layers.2.attention.channels"
                            },
                            ids.len() - 1,
                            2,
                        )),
                        ..Default::default()
                    };
                    let pass = if step == 0 {
                        ExpertPass::Prefill
                    } else {
                        ExpertPass::Decode
                    };
                    let actual = match mode {
                        0 => candidate.forward_with_observer(
                            input(),
                            &mut state,
                            &context,
                            &mut capture,
                        ),
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
                        .forward(input(), &mut ordinary_state, &context)
                        .unwrap();
                    if step == 0 {
                        first = Some(expected.clone());
                    }
                    if masked {
                        assert_ne!(
                            actual.data, expected.data,
                            "selected MTP depth changes its actual draft scores"
                        );
                    } else {
                        assert_tensor_exact(
                            &actual,
                            &expected,
                            "observed MTP traversal matches ordinary draft scores",
                        );
                    }
                    assert_tensor_exact(
                        &actual,
                        &capture.values[&format!(
                            "model.layers.{}.prediction.readout.linear.effective",
                            depth + 1
                        )],
                        "traversal returns the selected prediction head",
                    );
                    assert!(capture.values.contains_key(&format!("mtp.{depth}.capture")));
                    assert!(!capture
                        .values
                        .contains_key("model.layers.0.attention.channels"));
                    assert_eq!(state.layer(0).unwrap().offset(), 3);
                    for other in 0..2 {
                        assert_eq!(
                            state.layer(other + 1).unwrap().offset(),
                            if other == depth { 3 + step as i32 } else { 0 }
                        );
                    }
                }
                let mut sibling_state = parent_state.clone();
                let tokens = NumericTensor::token_ids(&[4, 5, 6]);
                let sibling = runtime(false)
                    .forward(
                        deepseek::mtp::EmbeddedInput::draft(&tokens, prior, depth),
                        &mut sibling_state,
                        &context,
                    )
                    .unwrap();
                assert_tensor_exact(
                    &sibling,
                    first.as_ref().unwrap(),
                    "fresh sibling retains original prediction state",
                );
                assert_eq!(parent_state.layer(0).unwrap().offset(), 3);
                assert_eq!(parent_state.layer(1).unwrap().offset(), 0);
                assert_eq!(parent_state.layer(2).unwrap().offset(), 0);
            }
        }
    }
}

/// Uses declared scope membership and parameter coordinates, including the whole
/// sparse write, without deriving checkpoint keys from family-specific names.
fn verify_scope_reconstruction(
    graph: &eredu_core::ArchitectureDescriptor,
    capture: &Components,
    parameters: &Parameters,
    logits: &NumericTensor,
) {
    use eredu_core::component::*;
    let scope = &graph.component_scopes[0];
    let ComponentResidualBase::LinearFusion {
        inputs,
        weight,
        projection_input,
        output,
        effective_output,
        bias,
        ..
    } = &scope.residual_base
    else {
        panic!("V3 fixture declares concatenated linear fusion")
    };
    assert!(bias.is_none());
    let multiplication_input = &capture.values[projection_input];
    let input_width = *multiplication_input.shape.last().unwrap() as usize;
    // The two separately normalized sources occupy the declared fusion columns.
    for input in inputs {
        let actual = &capture.values[&input.output];
        for (row, values) in actual.data.chunks_exact(input.columns.len()).enumerate() {
            assert_eq!(
                values,
                &multiplication_input.data[row * input_width + input.columns.start
                    ..row * input_width + input.columns.end]
            );
        }
    }
    assert_tensor_close(
        &projected(multiplication_input, &parameters.0[weight]),
        &capture.values[output],
        "declared fusion equation",
    );
    let base = &capture.values[effective_output];
    let width = *base.shape.last().unwrap() as usize;
    let rows = base.data.len() / width;
    let readout = &scope.readout;
    let mut terms = vec![base.data.iter().map(|v| f64::from(*v)).collect::<Vec<_>>()];
    for group in &scope.components {
        // Constituents of an explicitly declared whole write are not added twice.
        let mut node = Some(graph.node(&group.node_id).unwrap());
        let mut included = false;
        while let Some(current) = node {
            included |= readout
                .other_writes
                .iter()
                .any(|write| write.node_id == current.id);
            node = current
                .parent
                .as_ref()
                .and_then(|parent| graph.node(parent));
        }
        if included {
            continue;
        }
        assert!(group.write_bias.is_none());
        assert!(group.output_normalization.is_none());
        let activation = &capture.values[group.write_input.as_ref().unwrap()];
        let weight = &parameters.0[&group.write_weight];
        for component in 0..group.count {
            let mut contribution = vec![0.0; rows * width];
            for row in 0..rows {
                for hidden in 0..width {
                    contribution[row * width + hidden] =
                        f64::from(activation.data[row * group.count + component])
                            * f64::from(weight.data[hidden * group.count + component])
                            * f64::from(group.residual_scale.value());
                }
            }
            terms.push(contribution);
        }
    }
    for write in &readout.other_writes {
        terms.push(
            capture.values[&write.effective_output]
                .data
                .iter()
                .map(|value| f64::from(*value) * f64::from(write.residual_scale.value()))
                .collect(),
        );
    }
    let summed = (0..rows * width)
        .map(|i| terms.iter().map(|term| term[i]).sum::<f64>())
        .collect::<Vec<_>>();
    assert_tensor_close(
        &NumericTensor::new(
            base.shape.clone(),
            summed.iter().map(|v| *v as f32).collect(),
        ),
        &capture.values[&readout.residual],
        "scope fusion plus declared component writes reconstructs pre-head residual",
    );
    let effective = &capture.values[&format!("{}.effective", readout.residual)];
    // A direct residual edit is its own term, not attributed to earlier units.
    terms.push(
        effective
            .data
            .iter()
            .zip(&capture.values[&readout.residual].data)
            .map(|(after, before)| f64::from(*after) - f64::from(*before))
            .collect(),
    );
    assert_eq!(readout.normalization.kind, ComponentNormalizationKind::Rms);
    assert_eq!(readout.output_transform, ComponentOutputTransform::Identity);
    assert!(readout.bias.is_none());
    let gain = &parameters.0[readout.normalization.gain.as_ref().unwrap()];
    let head = &parameters.0[&readout.weight];
    let vocabulary = head.shape[0] as usize;
    assert_tensor_close(
        &projected(
            &capture.values[readout.projection_input.as_ref().unwrap()],
            head,
        ),
        &capture.values[&readout.linear_scores],
        "scope selects its own effective head parameters",
    );
    assert_tensor_exact(
        logits,
        &capture.values[&readout.logits],
        "scope logits publication",
    );
    for row in 0..rows {
        let denominator = (effective.data[row * width..(row + 1) * width]
            .iter()
            .map(|v| f64::from(*v).powi(2))
            .sum::<f64>()
            / width as f64
            + f64::from(readout.normalization.epsilon.value()))
        .sqrt();
        let score = |token: usize| {
            terms
                .iter()
                .map(|term| {
                    (0..width)
                        .map(|hidden| {
                            term[row * width + hidden] * f64::from(gain.data[hidden]) / denominator
                                * f64::from(head.data[token * width + hidden])
                        })
                        .sum::<f64>()
                })
                .sum::<f64>()
        };
        let target = score(3);
        let alternative = score(7);
        let expected_target = f64::from(logits.data[row * vocabulary + 3]);
        let expected_alternative = f64::from(logits.data[row * vocabulary + 7]);
        for (actual, expected) in [
            (target, expected_target),
            (alternative, expected_alternative),
            (target - alternative, expected_target - expected_alternative),
        ] {
            assert!(
                (actual - expected).abs() < 2e-4 * (1.0 + expected.abs()),
                "signed scope score/difference {actual} != {expected}"
            );
        }
    }
}
