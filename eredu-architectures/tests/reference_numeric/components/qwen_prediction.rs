//! Shared fusion and normalized hidden state in Qwen hybrid prediction units.
use super::*;

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
                        .map(|(a, b)| *a as f64 * *b as f64)
                        .sum::<f64>() as f32
                })
            })
            .collect(),
    )
}

#[test]
fn qwen_prediction_fusion_hooks_preserve_cached_execution_and_effective_hidden() {
    const PATH: &str = "mtp.layers.0";
    const CHANNEL: &str = "mtp.layers.0.attention.channels";
    const HIDDEN: &str = "mtp.layers.0.prediction.hidden.normalized";
    const FINAL: &str = "mtp.layers.0.prediction.readout.normalized";
    for config in heterogeneous_replicated_configs()
        .into_iter()
        .filter(|c| {
            matches!(
                c["model_type"].as_str(),
                Some("qwen3_next" | "qwen3_5_text")
            )
        })
        .flat_map(|config| {
            let mut routed = config.clone();
            if routed["model_type"] == "qwen3_5_text" {
                routed["model_type"] = "qwen3_5_moe_text".into();
            }
            routed["num_experts"] = 4.into();
            routed["num_experts_per_tok"] = 2.into();
            routed["moe_intermediate_size"] = 6.into();
            routed["shared_expert_intermediate_size"] = 8.into();
            routed["norm_topk_prob"] = true.into();
            [config, routed]
        })
    {
        let mut args = qwen::hybrid::model_args_from_config_value(&config)
            .unwrap()
            .text;
        args.mtp_num_hidden_layers = 2;
        let context = NumericContext::default();
        let unit = qwen::hybrid::PredictionUnit::<NumericBackend>::new(&args, 0, &context).unwrap();
        let shared =
            qwen::hybrid::PredictionShared::<NumericBackend>::new(&args, &context).unwrap();
        let state = || {
            DeviceState::<NumericBackend, _>::create(
                qwen::hybrid::state_layout(&args).unwrap(),
                |_, policy| Ok::<_, Error>(NumericHybridLayerState::new(policy)),
            )
            .unwrap()
        };
        let unit_path = if args.num_experts == 0 {
            "mtp.layers.0.feed_forward.units"
        } else {
            "mtp.layers.0.mlp.shared_expert.feed_forward.units"
        };
        for tensor_parallel in [false, true] {
            let parallel = NumericParallelContext::new(0, NumericParallelGroup::new(1));
            for mode in 0..6 {
                let mut candidate = unit.clone();
                let mut ordinary = unit.clone();
                let mut candidate_shared = shared.clone();
                let mut ordinary_shared = shared.clone();
                let (mut actual_state, mut ordinary_state) = (state(), state());
                for step in 0..3 {
                    let rows = if step == 0 { 3 } else { 1 };
                    let hidden = NumericTensor::new(
                        [1, rows, 8],
                        (0..rows * 8)
                            .map(|i| (i as f32 * 0.4 - 1.7 + step as f32).sin())
                            .collect(),
                    );
                    let embedded = NumericTensor::new(
                        [1, rows, 8],
                        (0..rows * 8)
                            .map(|i| (i as f32 * 0.7 + 0.2 - step as f32).cos())
                            .collect(),
                    );
                    let mask = (rows > 1)
                        .then(|| NumericBackend::causal_mask(rows, 0, None, &context).unwrap());
                    let mut capture = Components::strict();
                    match mode {
                        1 => capture.zero = Some((CHANNEL, rows as usize - 1, 1)),
                        2 => capture.keep = Some((unit_path, rows as usize - 1, 2)),
                        3 => capture.zero = Some((HIDDEN, rows as usize - 1, 3)),
                        4 => capture.zero = Some((FINAL, rows as usize - 1, 4)),
                        5 => {
                            capture.zero = Some((CHANNEL, rows as usize - 1, 1));
                            capture.keep = Some((unit_path, rows as usize - 1, 2));
                        }
                        _ => {}
                    }
                    let points = eredu_runtime::RoutedObservationPoints::new(
                        eredu_runtime::RoutedBankId::new(0),
                        "mtp.layers.0.mlp".to_owned(),
                        args.num_experts,
                    );
                    let actual_lane = actual_state.layer(args.num_hidden_layers as usize).unwrap();
                    let output = if tensor_parallel {
                        candidate.forward_parallel_observed_with_provider(
                            &mut candidate_shared,
                            PATH,
                            points,
                            &hidden,
                            &embedded,
                            mask.as_ref(),
                            actual_lane,
                            &parallel,
                            &context,
                            &mut eredu_runtime::ResidentExpertProvider,
                            &mut capture,
                        )
                    } else {
                        candidate.forward_observed_with_provider(
                            &mut candidate_shared,
                            PATH,
                            points,
                            &hidden,
                            &embedded,
                            mask.as_ref(),
                            actual_lane,
                            &context,
                            &mut eredu_runtime::ResidentExpertProvider,
                            &mut capture,
                        )
                    }
                    .unwrap();
                    let baseline = ordinary
                        .forward(
                            &mut ordinary_shared,
                            &hidden,
                            &embedded,
                            mask.as_ref(),
                            ordinary_state
                                .layer(args.num_hidden_layers as usize)
                                .unwrap(),
                            &context,
                        )
                        .unwrap();
                    if mode == 0 {
                        assert_tensor_exact(&output, &baseline, "Qwen MTP no-op cached execution");
                    } else {
                        assert_ne!(
                            output.data, baseline.data,
                            "causal Qwen MTP mode={mode} step={step}"
                        );
                    }
                    assert_tensor_exact(
                        &output,
                        &capture.values[&format!("{FINAL}.effective")],
                        "the next prediction consumes effective normalized hidden state",
                    );
                    assert_tensor_close(
                        &projected(
                            &capture.values[&format!("{PATH}.prediction.fusion.input")],
                            &shared.fusion.weight,
                        ),
                        &capture.values[&format!("{PATH}.prediction.fusion.output")],
                        "shared fusion matrix consumes actual normalized inputs",
                    );
                    let qwen::hybrid::TokenMixer::Attention(attention) = &unit.block.mixer else {
                        panic!("prediction uses full attention");
                    };
                    let (ffn, ffn_path) = match &unit.block.feed_forward {
                        qwen::hybrid::FeedForward::Dense(ffn) => (ffn, PATH.to_owned()),
                        qwen::hybrid::FeedForward::Routed(moe) => {
                            (&moe.shared_expert, format!("{PATH}.mlp.shared_expert"))
                        }
                    };
                    for (activation, write, weight) in [
                        (
                            format!("{PATH}.attention.channels.effective"),
                            format!("{PATH}.attention.write"),
                            &attention.output.weight,
                        ),
                        (
                            format!("{ffn_path}.feed_forward.units.effective"),
                            format!("{ffn_path}.feed_forward.write"),
                            &ffn.down.weight,
                        ),
                    ] {
                        assert_tensor_close(
                            &projected(&capture.values[&activation], weight),
                            &capture.values[&write],
                            "actual scalar component sum",
                        );
                    }
                    let residual =
                        &capture.values[&format!("{PATH}.prediction.readout.residual.effective")];
                    let expected = NumericTensor::new(
                        residual.shape.clone(),
                        residual
                            .data
                            .chunks_exact(8)
                            .flat_map(|row| {
                                let rms = (row.iter().map(|v| (*v as f64).powi(2)).sum::<f64>()
                                    / 8.
                                    + args.rms_norm_eps as f64)
                                    .sqrt();
                                row.iter().zip(&shared.final_norm.weight.data).map(
                                    move |(v, gain)| (*v as f64 / rms * (*gain as f64 + 1.)) as f32,
                                )
                            })
                            .collect(),
                    );
                    assert_tensor_close(
                        &expected,
                        &capture.values[FINAL],
                        "learned-offset final RMS",
                    );
                    for (selection, keep) in [(capture.zero, false), (capture.keep, true)] {
                        if let Some((path, row, component)) = selection {
                            let before = &capture.values[path];
                            let after = &capture.values[&format!("{path}.effective")];
                            let width = *before.shape.last().unwrap() as usize;
                            for (i, (before, after)) in
                                before.data.iter().zip(&after.data).enumerate()
                            {
                                let selected =
                                    i / width == row && ((i % width == component) != keep);
                                assert_eq!(*after, if selected { 0. } else { *before });
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn qwen_shared_units_include_the_measured_post_projection_gate() {
    const UNIT: &str = "model.layers.1.mlp.shared_expert.feed_forward.units";
    const GATE: &str = "model.layers.1.mlp.shared_expert.gate";
    let mut config = heterogeneous_replicated_configs()
        .into_iter()
        .find(|c| c["model_type"] == "qwen3_next")
        .unwrap();
    config["num_experts"] = 4.into();
    config["num_experts_per_tok"] = 2.into();
    config["moe_intermediate_size"] = 6.into();
    config["shared_expert_intermediate_size"] = 8.into();
    config["norm_topk_prob"] = true.into();
    let args = qwen::hybrid::model_args_from_config_value(&config)
        .unwrap()
        .text;
    let graph = eredu_architectures::configuration::MODEL_CONFIGURATIONS
        .resolve_safetensors(&config)
        .unwrap()
        .architecture_plan()
        .architecture_descriptor();
    let group = graph
        .components
        .iter()
        .find(|g| g.activation == UNIT)
        .unwrap();
    let gate = group.output_gate.as_ref().unwrap();
    assert_eq!(
        gate.activation,
        eredu_core::component::ComponentNonlinearity::Sigmoid
    );
    assert_eq!(
        gate.read.role,
        eredu_core::component::ComponentReadRole::OutputGate
    );
    for component in 0..group.count {
        assert_eq!(gate.read.rows.row_range(component), Some(0..1));
    }
    let context = NumericContext::default();
    let block = qwen::hybrid::Block::<NumericBackend>::new(&args, 1, &context).unwrap();
    let qwen::hybrid::FeedForward::Routed(ffn) = &block.feed_forward else {
        panic!("routed fixture");
    };
    let mut baseline = Vec::new();
    for mode in 0..4 {
        let model =
            qwen::hybrid::LayeredModel::<NumericBackend>::new(args.clone(), &context).unwrap();
        let mut runtime = ResidentRuntime::new(model, &context).unwrap();
        let mut state = DeviceState::<NumericBackend, _>::create(
            qwen::hybrid::state_layout(&args).unwrap(),
            |_, policy| Ok::<_, Error>(NumericHybridLayerState::new(policy)),
        )
        .unwrap();
        for (step, ids) in [vec![1, 3, 2], vec![4], vec![5]].into_iter().enumerate() {
            let mut capture = Components::default();
            match mode {
                1 => capture.zero = Some((UNIT, ids.len() - 1, 2)),
                2 => capture.keep = Some((UNIT, ids.len() - 1, 2)),
                3 => capture.zero = Some((GATE, ids.len() - 1, 0)),
                _ => {}
            }
            let tokens = NumericTensor::token_ids(&ids);
            let (output, _) = runtime
                .forward_with_traversal_hook(
                    qwen::hybrid::EmbeddedInput::target(&tokens, None),
                    &mut state,
                    &context,
                    &mut capture,
                )
                .unwrap();
            if mode == 0 {
                baseline.push(output.clone());
            } else {
                assert_ne!(
                    output.data, baseline[step].data,
                    "shared gate/unit affects target scores"
                );
            }
            let units = &capture.values[&group.effective_activation];
            let write = &capture.values[group.write_output.as_ref().unwrap()];
            assert_tensor_close(
                &projected(units, &ffn.shared_expert.down.weight),
                write,
                "shared units reconstruct the pre-gate affine write",
            );
            let projected_gate = projected(
                &capture.values[&gate.projection_input],
                &ffn.shared_expert_gate.weight,
            );
            let original_gate = &capture.values[&gate.output];
            for (logit, value) in projected_gate.data.iter().zip(&original_gate.data) {
                assert!((1. / (1. + (-*logit as f64).exp()) - *value as f64).abs() < 1e-6);
            }
            let effective_gate = &capture.values[&gate.effective_output];
            let scaled = NumericTensor::new(
                write.shape.clone(),
                write
                    .data
                    .iter()
                    .enumerate()
                    .map(|(i, value)| *value * effective_gate.data[i / args.hidden_size as usize])
                    .collect(),
            );
            assert_tensor_close(
                &scaled,
                &capture.values[group.output.as_ref().unwrap()],
                "actual post-projection gate scales every shared unit contribution",
            );
            assert!(units.data.iter().any(|v| v.abs() > 1e-7));
            assert!(original_gate.data.iter().all(|v| *v > 0. && *v < 1.));
            if mode == 3 {
                assert_eq!(effective_gate.data[ids.len() - 1], 0.);
                assert!(capture.values[group.output.as_ref().unwrap()].data
                    [(ids.len() - 1) * args.hidden_size as usize..]
                    .iter()
                    .all(|v| *v == 0.));
            }
        }
    }
}

fn qwen_component_config(model_type: &str, routed: bool) -> serde_json::Value {
    serde_json::json!({
        "model_type":if routed && model_type == "qwen3_5_text" {"qwen3_5_moe_text"} else {model_type},"vocab_size":7,"hidden_size":8,
        "num_hidden_layers":2,"num_attention_heads":4,"num_key_value_heads":2,
        "head_dim":2,"max_position_embeddings":64,"linear_conv_kernel_dim":2,
        "linear_key_head_dim":2,"linear_value_head_dim":2,
        "linear_num_key_heads":2,"linear_num_value_heads":2,
        "intermediate_size":8,"moe_intermediate_size":6,
        "shared_expert_intermediate_size":8,"num_experts_per_tok":2,
        "num_experts":if routed {4} else {0},"norm_topk_prob":true,
        "layer_types":["linear_attention","full_attention"],"tie_word_embeddings":false
    })
}

fn qwen_target_trial(
    args: &qwen::hybrid::HybridConfig,
    descriptor: &eredu_core::ArchitectureDescriptor,
    layout: Option<LocalModelLayout>,
    parallel: Option<&NumericParallelContext>,
    mode: usize,
) -> Vec<(NumericTensor, BTreeMap<String, NumericTensor>)> {
    const CHANNEL: &str = "model.layers.1.attention.channels";
    let unit_path = if args.is_moe() {
        "model.layers.1.mlp.shared_expert.feed_forward.units"
    } else {
        "model.layers.1.feed_forward.units"
    };
    let context = layout
        .clone()
        .map(NumericContext::with_local_layout)
        .unwrap_or_default();
    let geometry = layout
        .as_ref()
        .map(|layout| qwen::hybrid::local_geometry(args, layout).unwrap());
    let state_layout = geometry
        .as_ref()
        .map(|g| g.state_layout().clone())
        .unwrap_or_else(|| qwen::hybrid::state_layout(args).unwrap());
    let model = match geometry {
        Some(geometry) => qwen::hybrid::LayeredModel::<NumericBackend>::new_parallel(
            args.clone(),
            geometry,
            &context,
        ),
        None => qwen::hybrid::LayeredModel::<NumericBackend>::new(args.clone(), &context),
    }
    .unwrap();
    struct RecurrentMagnitude<'c>(&'c NumericContext);
    impl<'a> ParameterVisitorMut<'a, NumericTensor> for RecurrentMagnitude<'_> {
        fn visit_mut(&mut self, metadata: ParameterMetadata, value: &'a mut NumericTensor) {
            let name = metadata.id.as_str();
            if name.contains(".linear_attn.") {
                if name.ends_with(".norm.weight") {
                    // Raw architecture parameters start unloaded, unlike this
                    // backend's deterministic linear operators. Bind a real gain.
                    value.data.fill(1.0);
                } else if name.ends_with(".A_log") {
                    value.data.fill(-0.5);
                } else if name.ends_with(".dt_bias") {
                    value.data.fill(0.1);
                } else if name.ends_with(".weight") {
                    *value = local_parameter(
                        &ParameterSpec::trainable(name).unwrap(),
                        value.shape.clone(),
                        false,
                        self.0,
                    )
                    .unwrap()
                    .map(|value| value * 10.0);
                }
            }
        }
    }
    let units = (0..args.num_hidden_layers as usize)
        .map(|i| {
            let mut unit = model.construct_unit(0, i, &context).unwrap();
            unit.visit_parameters_mut(&mut RecurrentMagnitude(&context));
            unit
        })
        .collect();
    let mut runtime = LayerwiseRuntime::new(model, ResidentUnitWindow::new(units));
    let mut state = DeviceState::<NumericBackend, _>::create(state_layout, |_, policy| {
        Ok::<_, Error>(NumericHybridLayerState::new(policy))
    })
    .unwrap();
    let local = |path: &str, index: usize| {
        layout.as_ref().map_or(Some(index), |layout| {
            let group = descriptor
                .components
                .iter()
                .find(|g| g.activation == path)
                .unwrap();
            eredu_architectures::component_partition::derive_component_coordinates(
                group,
                layout.tensor(&group.write_weight).unwrap(),
            )
            .unwrap()
            .global_to_local(index)
        })
    };
    [vec![1, 3, 2], vec![4], vec![5]]
        .into_iter()
        .map(|ids| {
            let position = ids.len() - 1;
            let mut capture = Components::strict();
            if mode == 1 || mode == 3 {
                capture.zero = local(CHANNEL, 1).map(|index| (CHANNEL, position, index));
            }
            if mode == 2 || mode == 3 {
                // A keep-only set can have no surviving units on one rank.
                capture.keep = Some((
                    unit_path,
                    position,
                    local(unit_path, 5).unwrap_or(usize::MAX),
                ));
            }
            if mode == 4 {
                capture.zero = Some(("readout.normalized", position, 3));
            }
            if mode == 5 {
                const RECURRENT: &str = "model.layers.0.mixer.channels";
                capture.zero = local(RECURRENT, 1).map(|index| (RECURRENT, position, index));
            }
            if mode == 6 {
                const RECURRENT: &str = "model.layers.0.mixer.channels";
                capture.keep = Some((
                    RECURRENT,
                    position,
                    local(RECURRENT, 3).unwrap_or(usize::MAX),
                ));
            }
            let tokens = NumericTensor::token_ids(&ids);
            let input = qwen::hybrid::EmbeddedInput::target(&tokens, None);
            let (output, _) = match parallel {
                Some(parallel) => runtime.forward_parallel_with_traversal_hook(
                    input,
                    &mut state,
                    parallel,
                    &context,
                    &mut capture,
                ),
                None => {
                    runtime.forward_with_traversal_hook(input, &mut state, &context, &mut capture)
                }
            }
            .unwrap();
            (output, capture.values)
        })
        .collect()
}

#[test]
fn qwen_target_components_preserve_sharded_masks_shared_gate_and_collective_order() {
    for model_type in ["qwen3_next", "qwen3_5_text"] {
        for routed in [false, true] {
            let config = qwen_component_config(model_type, routed);
            let args = qwen::hybrid::model_args_from_config_value(&config)
                .unwrap()
                .text;
            assert_eq!(
                args.is_moe(),
                routed,
                "fixture selects the actual family variant"
            );
            let descriptor = eredu_architectures::configuration::MODEL_CONFIGURATIONS
                .resolve_safetensors(&config)
                .unwrap()
                .architecture_plan()
                .architecture_descriptor();
            let context = NumericContext::default();
            let model =
                qwen::hybrid::LayeredModel::<NumericBackend>::new(args.clone(), &context).unwrap();
            let modules = model.static_modules();
            let mut groups = decoder::static_parallel_parameter_groups::<NumericBackend>(
                &modules.embeddings,
                &modules.norm,
                modules.lm_head.as_ref(),
                "model",
            )
            .unwrap();
            for layer in 0..2 {
                groups.extend(
                    qwen::hybrid::unit_parallel_parameter_groups(
                        &model.construct_unit(0, layer, &context).unwrap(),
                        &args,
                        0,
                        layer,
                    )
                    .unwrap(),
                );
            }
            let mut baseline = Vec::new();
            let mut no_op_trace = Vec::new();
            for mode in 0..7 {
                let expected = qwen_target_trial(&args, &descriptor, None, None, mode);
                if mode == 0 {
                    baseline = expected.iter().map(|v| v.0.clone()).collect();
                } else {
                    for (step, (value, _)) in expected.iter().enumerate() {
                        assert_ne!(
                            value.data, baseline[step].data,
                            "causal Qwen target mode={mode}"
                        );
                    }
                }
                let group = NumericParallelGroup::new(2);
                let actual = std::thread::scope(|scope| {
                    let handles = (0..2)
                        .map(|rank| {
                            let layout = numeric_local_layout(&groups, 2, rank).unwrap();
                            let parallel = NumericParallelContext::new(rank, group.clone());
                            let (args, descriptor) = (&args, &descriptor);
                            scope.spawn(move || {
                                let values = qwen_target_trial(
                                    args,
                                    descriptor,
                                    Some(layout),
                                    Some(&parallel),
                                    mode,
                                );
                                (values, parallel.trace())
                            })
                        })
                        .collect::<Vec<_>>();
                    handles
                        .into_iter()
                        .map(|h| h.join().unwrap())
                        .collect::<Vec<_>>()
                });
                for (rank, (steps, trace)) in actual.iter().enumerate() {
                    if mode == 0 {
                        no_op_trace.push(trace.clone());
                    } else {
                        assert_eq!(
                            trace, &no_op_trace[rank],
                            "instrumentation preserves collectives"
                        );
                    }
                    for (step, (score, capture)) in steps.iter().enumerate() {
                        assert_tensor_close(score, &expected[step].0, "global masked TP2 scores");
                        for path in [
                            "readout.embedding.effective",
                            "model.layers.1.attention.input.effective",
                            "model.layers.1.feed_forward.input.effective",
                            "readout.residual.effective",
                            "readout.normalized.effective",
                            "readout.linear.effective",
                        ] {
                            assert_tensor_close(&capture[path], &expected[step].1[path], path);
                        }
                    }
                }
                if routed {
                    for step in 0..3 {
                        let path = "model.layers.1.mlp.shared_expert";
                        let write = format!("{path}.feed_forward.output.effective");
                        let sum = actual[0].0[step].1[&write]
                            .add(&actual[1].0[step].1[&write], &context)
                            .unwrap();
                        assert_tensor_close(
                            &sum,
                            &expected[step].1[&write],
                            "sum of actual gated shared TP writes",
                        );
                        for (steps, _) in &actual {
                            let gate = format!("{path}.gate.effective");
                            assert_tensor_close(
                                &steps[step].1[&gate],
                                &expected[step].1[&gate],
                                "replicated post-projection scalar gate",
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn prepared_qwen_hybrid_components_and_scalar_gates_cross_tp_ep_pp_and_residency() {
    struct Populate<'a>(&'a BTreeMap<String, NumericTensor>);
    impl<'a> ParameterVisitorMut<'a, NumericTensor> for Populate<'_> {
        fn visit_mut(&mut self, metadata: ParameterMetadata, value: &'a mut NumericTensor) {
            let source = self
                .0
                .get(metadata.id.as_str())
                .unwrap_or_else(|| panic!("missing reference parameter {}", metadata.id.as_str()));
            // Official depthwise convolution storage includes a singleton input
            // channel; the neutral operator stores the same coefficients as C×K.
            assert!(
                source.shape == value.shape
                    || (value.shape.len() == 2
                        && source.shape == [value.shape[0], 1, value.shape[1]]),
                "{}",
                metadata.id.as_str()
            );
            value.data.clone_from(&source.data);
        }
    }
    for model_type in ["qwen3_next", "qwen3_5_text"] {
        for routed in [false, true] {
            eprintln!("prepared Qwen hybrid components: {model_type}, routed={routed}");
            let config = qwen_component_config(model_type, routed);
            let args = qwen::hybrid::model_args_from_config_value(&config)
                .unwrap()
                .text;
            let (root, fixture) = prepared_adapter::payload_fixture_config(&config, 1.0);
            let inspection =
                eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
            let context = NumericContext {
                bind_checkpoint_values: true,
                ..Default::default()
            };
            let store = eredu_checkpoint::store::SafetensorsWeightStore::open(root.path()).unwrap();
            let mut recipes = qwen::hybrid::static_recipes(&store).unwrap();
            for layer in 0..2 {
                recipes.extend(qwen::hybrid::unit_recipes(&store, &args, layer).unwrap());
            }
            let mut fixture = fixture
                .into_iter()
                .map(|(name, (shape, bits))| {
                    (
                        name,
                        NumericTensor::new(shape, bits.into_iter().map(f32::from_bits).collect()),
                    )
                })
                .collect::<BTreeMap<_, _>>();
            // Interpret family-owned source recipes before the serial reference;
            // partitioned execution below independently binds the selected bytes.
            for (target, recipe) in recipes {
                fixture.insert(
                    target,
                    payload::recipe_value(&recipe, &store, &context).unwrap(),
                );
            }
            let model = || {
                qwen::hybrid::LayeredModel::<NumericBackend>::new(args.clone(), &context).unwrap()
            };
            let parameters = model().parameter_description(&context).unwrap();
            let inputs = [
                NumericTensor::token_ids(&[1, 2, 3]),
                NumericTensor::token_ids(&[4]),
                NumericTensor::token_ids(&[5]),
            ];
            let unit = if routed {
                "model.layers.1.mlp.shared_expert.feed_forward.units"
            } else {
                "model.layers.1.feed_forward.units"
            };
            let channel = "model.layers.1.attention.channels";
            let targets = vec![
                (None, vec![]),
                (
                    None,
                    vec![("model.layers.0.mixer.channels", vec![1, 3], false)],
                ),
                (
                    None,
                    vec![("model.layers.0.mixer.channels", vec![1, 3], true)],
                ),
                (None, vec![(channel, vec![1, 5], false)]),
                (None, vec![(unit, vec![2, 5], true)]),
                (
                    None,
                    vec![(channel, vec![1, 5], true), (unit, vec![2, 5], true)],
                ),
            ];
            let expected = targets
                .iter()
                .map(|(target, masks)| {
                    let mut model = model();
                    model
                        .static_modules_mut()
                        .visit_parameters_mut(&mut Populate(&fixture));
                    let mut state = DeviceState::<NumericBackend, _>::create(
                        model.state_layout().unwrap(),
                        |_, policy| Ok::<_, Error>(NumericHybridLayerState::new(policy)),
                    )
                    .unwrap();
                    let units = (0..2)
                        .map(|layer| {
                            let mut unit = model.construct_unit(0, layer, &context).unwrap();
                            unit.visit_parameters_mut(&mut Populate(&fixture));
                            unit
                        })
                        .collect();
                    let mut runtime = LayerwiseRuntime::new(model, ResidentUnitWindow::new(units));
                    inputs
                        .iter()
                        .enumerate()
                        .map(|(step, tokens)| {
                            let mut observer = GlobalComponentObserver {
                                inner: NumericLifecycleObserver {
                                    zero_path: target.map(str::to_owned),
                                    ..Default::default()
                                },
                                values: BTreeMap::new(),
                                layout: None,
                                masks,
                                position: usize::from(step == 0),
                                projection_plan: None,
                                producer_receipts: None,
                                prediction: step as u64,
                            };
                            let output = runtime
                                .forward_with_observer(
                                    qwen::hybrid::EmbeddedInput::target(tokens, None),
                                    &mut state,
                                    &context,
                                    &mut observer,
                                )
                                .unwrap();
                            // Match the session's final observation seam and
                            // selected last-row score publication.
                            let output =
                                eredu_runtime::observe_model_logits(&mut observer, &output)
                                    .unwrap();
                            let vocabulary = *output.shape.last().unwrap() as usize;
                            (
                                NumericTensor::new(
                                    vec![1, 1, vocabulary as i32],
                                    output.data[output.data.len() - vocabulary..].to_vec(),
                                ),
                                observer.values,
                            )
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            for (trial_index, trial) in expected.iter().enumerate().skip(1) {
                for (step, (score, capture)) in trial.iter().enumerate() {
                    let logits = &capture[eredu_core::MODEL_LOGITS_OBSERVATION_PATH];
                    let baseline = &expected[0][step].1[eredu_core::MODEL_LOGITS_OBSERVATION_PATH];
                    assert!(
                        logits
                            .data
                            .iter()
                            .zip(&baseline.data)
                            .any(|(a, b)| (a - b).abs() > 1e-5),
                        "nonzero prepared hybrid causal fixture"
                    );
                    if step == 0 {
                        let vocabulary = logits.shape[2] as usize;
                        assert_eq!(
                            &logits.data[..vocabulary],
                            &baseline.data[..vocabulary],
                            "middle-position masks preserve preceding predictions"
                        );
                        if targets[trial_index]
                            .1
                            .iter()
                            .any(|(path, _, _)| *path == "model.layers.0.mixer.channels")
                        {
                            assert_ne!(
                                score.data, expected[0][step].0.data,
                                "later attention consumes the edited recurrent residual"
                            );
                        } else {
                            // Final-layer masks occur after its KV update and
                            // affect only the selected middle prefill row.
                            assert_tensor_close(
                                score,
                                &expected[0][step].0,
                                "prefill position isolation",
                            );
                        }
                    }
                }
            }
            let mut topologies = vec![
                ParallelTopology::new(2, 1, 1, 1).unwrap(),
                ParallelTopology::new(1, 2, 1, 1).unwrap(),
                ParallelTopology::new(2, 2, 1, 1).unwrap(),
            ];
            if routed {
                topologies.extend([
                    ParallelTopology::new(1, 1, 2, 1).unwrap(),
                    ParallelTopology::new(2, 1, 2, 1).unwrap(),
                    ParallelTopology::new(1, 2, 2, 1).unwrap(),
                    ParallelTopology::new(2, 2, 2, 1).unwrap(),
                ]);
            }
            verify_prepared_component_execution_with_topologies(
                &inspection,
                parameters,
                &inputs,
                &targets,
                &expected,
                if routed {
                    |sources, context| {
                        partitioned_adapter::routed(
                            sources,
                            context,
                            Arc::new(AtomicUsize::new(0)),
                            None,
                        )
                    }
                } else {
                    partitioned_adapter::dense
                },
                &topologies,
            );
        }
    }
}
