//! Consumed KDA/MLA channels retain their exact projection and cache semantics.
use super::*;
use eredu_architectures::kimi_linear::{Block, ModelArgs, TokenMixer};

struct Initialize;
impl<'a> ParameterVisitorMut<'a, NumericTensor> for Initialize {
    fn visit_mut(&mut self, metadata: ParameterMetadata, value: &'a mut NumericTensor) {
        let name = metadata.id.as_str();
        let seed = name
            .bytes()
            .fold(0_u32, |a, b| a.wrapping_mul(31).wrapping_add(b.into()));
        for (i, element) in value.data.iter_mut().enumerate() {
            let delta = ((i * 7 + (seed % 97) as usize) % 41) as f32 - 20.0;
            *element = if name.contains("norm") && name.ends_with("weight") {
                1.0 + delta * 0.003
            } else if name.ends_with("A_log") {
                -0.5 + delta * 0.01
            } else {
                delta * 0.02
            };
        }
    }
}

#[derive(Default)]
struct Parameters(BTreeMap<String, NumericTensor>);
impl<'a> ParameterVisitor<'a, NumericTensor> for Parameters {
    fn visit(&mut self, metadata: ParameterMetadata, value: &'a NumericTensor) {
        self.0.insert(metadata.id.to_string(), value.clone());
    }
}

fn config_value(low_rank: bool) -> serde_json::Value {
    serde_json::json!({
            "model_type":"kimi_linear", "vocab_size":16, "hidden_size":12,
            "num_hidden_layers":2, "num_attention_heads":3, "num_key_value_heads":3,
            "intermediate_size":17, "head_dim":4, "model_max_length":64,
            "linear_attn_config":{
                "kda_layers":[1], "full_attn_layers":[2], "num_heads":3,
                "head_dim":4, "short_conv_kernel_size":3
            },
            "num_experts":2, "moe_intermediate_size":9, "kv_lora_rank":6,
            "q_lora_rank": if low_rank { Some(5) } else { None },
            "qk_nope_head_dim":4, "qk_rope_head_dim":2, "v_head_dim":4,
            "mla_use_nope":true, "num_experts_per_token":1, "num_shared_experts":1,
            "routed_scaling_factor":1.0, "first_k_dense_replace":1,
            "num_expert_group":1, "topk_group":1, "tie_word_embeddings":false
    })
}

fn config(low_rank: bool, split: bool) -> ModelArgs {
    let mut args =
        eredu_architectures::kimi_linear::model_args_from_config_value(&config_value(low_rank))
            .unwrap();
    // Physical KV-B layout is inferred from checkpoint keys during admission.
    args.split_kv_b = split;
    args
}

fn run(
    mixer: &mut TokenMixer<NumericBackend>,
    input: &NumericTensor,
    state: &mut NumericHybridLayerState,
    observer: Option<&mut Components>,
    context: &NumericContext,
) -> NumericTensor {
    match observer {
        Some(observer) => {
            let mut instrumentation = ComponentInstrumentation::new("unit", observer);
            match mixer {
                TokenMixer::Kda(mixer) => {
                    mixer.forward_instrumented(input, state, None, context, &mut instrumentation)
                }
                TokenMixer::Mla(mixer) => mixer.forward_instrumented(
                    input,
                    None,
                    Some(state),
                    None,
                    context,
                    &mut instrumentation,
                ),
            }
        }
        None => match mixer {
            TokenMixer::Kda(mixer) => mixer.forward(input, state, context),
            TokenMixer::Mla(mixer) => mixer.forward(input, None, Some(state), context),
        },
    }
    .unwrap()
}

#[test]
fn kimi_channel_components_reconstruct_writes_and_preserve_cached_replay() {
    let context = NumericContext::default();
    for (low_rank, split) in [(false, false), (true, false), (true, true)] {
        let args = config(low_rank, split);
        assert_eq!(args.split_kv_b, split);
        let layout = eredu_architectures::kimi_linear::state_layout(&args).unwrap();
        for layer in 0..2 {
            let mut block = Block::<NumericBackend>::new(&args, layer, &context).unwrap();
            block.visit_parameters_mut(&mut Initialize);
            let mut parameters = Parameters::default();
            block.visit_parameters(&mut parameters);
            let weight = &parameters.0[&format!("model.layers.{layer}.self_attn.o_proj.weight")];
            for paged in [false, true] {
                if paged && layer == 0 {
                    continue;
                }
                let mut state = NumericHybridLayerState::new(layout.layer(layer).unwrap());
                if paged {
                    state.compressed.as_mut().unwrap().block_size = Some(2);
                }
                for (step, sequence) in [3, 1, 1].into_iter().enumerate() {
                    let input = NumericTensor::new(
                        [1, sequence, 12],
                        (0..sequence * 12)
                            .map(|i| ((i * 11 + step as i32 * 7) % 31) as f32 / 13.0 - 1.0)
                            .collect(),
                    );
                    let before = state.clone();
                    let mut observed_state = before.clone();
                    let mut observed = Components::strict();
                    let ordinary = run(&mut block.mixer, &input, &mut state, None, &context);
                    let actual = run(
                        &mut block.mixer,
                        &input,
                        &mut observed_state,
                        Some(&mut observed),
                        &context,
                    );
                    assert_tensor_exact(&actual, &ordinary, "Kimi primitive observation parity");
                    let channels = &observed.values["unit.attention.channels"];
                    assert!(channels.data.iter().any(|value| value.abs() > 1e-4));
                    assert_tensor_exact(
                        channels,
                        &observed.values["unit.attention.channels.effective"],
                        "Kimi unchanged effective channels",
                    );
                    let width = channels.shape[2] as usize;
                    let output_width = weight.shape[0] as usize;
                    let reconstructed = (0..sequence as usize * output_width)
                        .map(|i| {
                            let row = i / output_width;
                            let output = i % output_width;
                            (0..width)
                                .map(|channel| {
                                    channels.data[row * width + channel] as f64
                                        * weight.data[output * width + channel] as f64
                                })
                                .sum::<f64>() as f32
                        })
                        .collect();
                    assert_tensor_close(
                        &NumericTensor::new(ordinary.shape.clone(), reconstructed),
                        &ordinary,
                        "Kimi signed channel reconstruction",
                    );
                    for keep in [false, true] {
                        let selection = Some(("unit.attention.channels", sequence as usize - 1, 2));
                        let mut observer = Components {
                            zero: (!keep).then_some(selection.unwrap()),
                            keep: keep.then_some(selection.unwrap()),
                            ..Default::default()
                        };
                        let mut edited_state = before.clone();
                        let edited = run(
                            &mut block.mixer,
                            &input,
                            &mut edited_state,
                            Some(&mut observer),
                            &context,
                        );
                        let effective = &observer.values["unit.attention.channels.effective"];
                        let selected_start = (sequence as usize - 1) * width;
                        for (i, (&a, &b)) in channels.data.iter().zip(&effective.data).enumerate() {
                            let removed =
                                i >= selected_start && ((i - selected_start == 2) != keep);
                            assert_eq!(b, if removed { 0.0 } else { a });
                        }
                        assert!(edited
                            .data
                            .iter()
                            .zip(&ordinary.data)
                            .any(|(a, b)| (a - b).abs() > 1e-6));
                        // Channel edits happen after recurrent/KV updates. They change
                        // this write without retroactively changing retained history.
                        let next = NumericTensor::new([1, 1, 12], vec![0.3; 12]);
                        let expected =
                            run(&mut block.mixer, &next, &mut state.clone(), None, &context);
                        let replay =
                            run(&mut block.mixer, &next, &mut edited_state, None, &context);
                        assert_tensor_exact(
                            &replay,
                            &expected,
                            "Kimi channel mask preserves preceding cache update",
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn kimi_dense_and_shared_units_use_the_same_observed_residual_driver() {
    let context = NumericContext::default();
    let args = config(true, false);
    let layout = eredu_architectures::kimi_linear::state_layout(&args).unwrap();
    for (layer, path, target) in [
        (0, "model.layers.0", "model.layers.0.feed_forward.units"),
        (
            1,
            "model.layers.1",
            "model.layers.1.mlp.shared_experts.feed_forward.units",
        ),
    ] {
        let mut block = Block::<NumericBackend>::new(&args, layer, &context).unwrap();
        block.visit_parameters_mut(&mut Initialize);
        let input = NumericTensor::new(
            [1, 3, 12],
            (0..36).map(|i| (i as f32 * 0.3 - 2.0).sin()).collect(),
        );
        let empty = NumericHybridLayerState::new(layout.layer(layer).unwrap());
        let ordinary = block
            .forward(&input, None, &mut empty.clone(), &context)
            .unwrap();
        let mut run = |observer: &mut Components| {
            block.forward_partition_instrumented_with_feed_forward(
                &input, None, &mut empty.clone(), None, &context,
                &mut ComponentInstrumentation::new(path, observer),
                |policy, input, context, instrumentation| {
                    eredu_architectures::decoder::DecoderProjectionOperator::forward_feed_forward_observed(
                        policy, input, context, instrumentation,
                    )
                },
            ).unwrap()
        };
        let mut observed = Components::strict();
        let actual = run(&mut observed);
        assert_tensor_exact(&actual, &ordinary, "Kimi observed residual parity");
        assert!(observed.values[target]
            .data
            .iter()
            .any(|value| value.abs() > 1e-4));
        let attention = &observed.values[&format!("{path}.attention.output.effective")];
        let ffn = &observed.values[&format!(
            "{path}.feed_forward.{}.effective",
            if layer == 0 { "output" } else { "contribution" }
        )];
        let sum = input
            .add(attention, &context)
            .unwrap()
            .add(ffn, &context)
            .unwrap();
        assert_tensor_exact(&sum, &actual, "Kimi complete residual writes");
        for keep in [false, true] {
            let mut observer = Components {
                zero: (!keep).then_some((target, 2, 1)),
                keep: keep.then_some((target, 2, 1)),
                ..Components::strict()
            };
            let edited = run(&mut observer);
            assert_eq!(&edited.data[..24], &ordinary.data[..24]);
            assert!(edited.data[24..]
                .iter()
                .zip(&ordinary.data[24..])
                .any(|(a, b)| (a - b).abs() > 1e-6));
        }
    }
}

#[test]
fn kimi_observed_pipeline_cut_preserves_components_and_cached_trials() {
    use eredu_architectures::kimi_linear as family;
    type State = DeviceState<NumericBackend, NumericHybridLayerState>;
    type Partition = family::PartitionedLayeredModel<NumericBackend>;
    let context = NumericContext::default();
    for tied in [false, true] {
        let mut args = config(true, false);
        args.tie_word_embeddings = tied;
        let model = || {
            let mut model =
                family::LayeredModel::<NumericBackend>::new(args.clone(), &context).unwrap();
            model
                .static_modules_mut()
                .visit_parameters_mut(&mut Initialize);
            model
        };
        let units = || {
            (0..2)
                .map(|layer| {
                    let mut unit = Block::<NumericBackend>::new(&args, layer, &context).unwrap();
                    unit.visit_parameters_mut(&mut Initialize);
                    unit
                })
                .collect::<Vec<_>>()
        };
        let state = || {
            State::create(family::state_layout(&args).unwrap(), |_, policy| {
                Ok::<_, Error>(NumericHybridLayerState::new(policy))
            })
            .unwrap()
        };
        let description = model().parameter_description(&context).unwrap();
        let topology = ParallelTopology::new(1, 2, 1, 1).unwrap();
        for masked in [false, true] {
            let mut ordinary = LayerwiseRuntime::new(model(), ResidentUnitWindow::new(units()));
            let mut reference = LayerwiseRuntime::new(model(), ResidentUnitWindow::new(units()));
            let (mut ordinary_state, mut reference_state) = (state(), state());
            let mut stages = (0..2).map(|layer| {
                let rank = ParallelRankTopology::new(topology, layer).unwrap();
                let layout = eredu_architectures::partitioned_execution::derive_partitioned_local_layout(
                    &description, rank,
                ).unwrap();
                let realization = family::expert_realization_plan(&model(), rank).unwrap().unwrap();
                let geometry = family::partition_local_routed_geometry(
                    &args, &layout, layer..layer + 1, rank, &realization,
                ).unwrap();
                let complete = geometry.complete_state_layout().clone();
                let partition = ArchitecturePartition::from_description(
                    &description, [(decoder::TARGET_EXECUTION_GROUP, layer..layer + 1)],
                    if layer == 0 {
                        PartitionOwnership::new(true, false, ["embedding"]).unwrap()
                    } else {
                        PartitionOwnership::new(false, true, ["norm", "output"]).unwrap()
                    },
                    &complete,
                    &ArchitectureStatePartitionPlan::new([
                        ArchitectureStatePartitionRule::group_units(0, 0..2),
                    ]),
                    geometry, NoAuxiliaryBoundarySchema::new(args.hidden_size),
                ).unwrap();
                let mut architecture = Partition::from_partition(
                    args.clone(), &description, &partition, &context,
                ).unwrap();
                <Partition as LayeredArchitecture<NumericBackend, State>>::static_modules_mut(
                    &mut architecture,
                ).visit_parameters_mut(&mut Initialize);
                let mut unit = architecture.construct_unit(layer, &context).unwrap();
                unit.visit_parameters_mut(&mut Initialize);
                let state = State::create(partition.state().unwrap().layout().clone(), |_, policy| {
                    Ok::<_, Error>(NumericHybridLayerState::new(policy))
                }).unwrap();
                let driver = LayeredPartitionDriver::new(&partition, 0, layer..layer + 1).unwrap();
                (architecture, unit, state, driver)
            }).collect::<Vec<_>>();
            let mut position = 0;
            for ids in [vec![1, 3, 2], vec![4], vec![5]] {
                let tokens = NumericTensor::token_ids(&ids);
                let capture = || Components {
                    zero: masked.then_some(("model.layers.0.attention.channels", ids.len() - 1, 1)),
                    keep: masked.then_some((
                        "model.layers.1.mlp.shared_experts.feed_forward.units",
                        ids.len() - 1,
                        2,
                    )),
                    ..Components::strict()
                };
                let (mut expected_capture, mut actual_capture) = (capture(), capture());
                let input = || decoder::LayeredInput {
                    tokens: &tokens,
                    mask: None,
                };
                let expected = reference
                    .forward_with_observer(
                        input(),
                        &mut reference_state,
                        &context,
                        &mut expected_capture,
                    )
                    .unwrap();
                if !masked {
                    let baseline = ordinary
                        .forward(input(), &mut ordinary_state, &context)
                        .unwrap();
                    assert_tensor_exact(&expected, &baseline, "Kimi complete no-op observation");
                }
                let mut input = LayeredPartitionInput::Tokens(&tokens);
                let mut actual = None;
                for (layer, (architecture, unit, state, driver)) in stages.iter_mut().enumerate() {
                    let mut forward = driver
                        .begin_observed(
                            architecture,
                            input,
                            None,
                            state,
                            None,
                            &context,
                            &mut actual_capture,
                        )
                        .unwrap();
                    forward.hidden = architecture
                        .forward_unit_observed(
                            0,
                            layer,
                            unit,
                            &forward.hidden,
                            state,
                            &mut forward.context,
                            &context,
                            &mut actual_capture,
                        )
                        .unwrap();
                    match driver
                        .finish_observed(
                            architecture,
                            &forward.hidden,
                            state,
                            &mut forward.context,
                            None,
                            &context,
                            &mut actual_capture,
                        )
                        .unwrap()
                    {
                        LayeredPartitionOutput::StateOnly { .. } => {
                            panic!("sequence readout omitted scores")
                        }
                        LayeredPartitionOutput::Boundary { hidden, .. } => {
                            input = LayeredPartitionInput::Hidden {
                                hidden,
                                auxiliary: NoAuxiliaryBoundary,
                            };
                        }
                        LayeredPartitionOutput::Final { output, .. } => {
                            actual = Some(output);
                            input = LayeredPartitionInput::Tokens(&tokens);
                        }
                    }
                }
                assert_tensor_exact(
                    &actual.unwrap(),
                    &expected,
                    "Kimi observed recurrent/MLA pipeline cut",
                );
                for (path, expected) in &expected_capture.values {
                    if !(path.starts_with("readout.")
                        || path.contains(".attention.")
                        || path.contains(".feed_forward.")
                        || path.contains(".mlp."))
                    {
                        continue;
                    }
                    let actual = actual_capture
                        .values
                        .get(path)
                        .unwrap_or_else(|| panic!("missing partition component: {path}"));
                    assert_tensor_exact(expected, actual, path);
                }
                position += ids.len() as i32;
                for (layer, (_, _, state, _)) in stages.iter_mut().enumerate() {
                    assert_eq!(state.layer(0).unwrap().position(), position);
                    assert_eq!(reference_state.layer(layer).unwrap().position(), position);
                }
            }
        }
    }
}

fn affine(
    input: &NumericTensor,
    weight: &NumericTensor,
    rows: std::ops::Range<usize>,
) -> NumericTensor {
    let width = weight.shape[1] as usize;
    let data = input
        .data
        .chunks_exact(width)
        .flat_map(|input| {
            rows.clone().map(move |row| {
                input
                    .iter()
                    .zip(&weight.data[row * width..(row + 1) * width])
                    .map(|(a, b)| *a as f64 * *b as f64)
                    .sum::<f64>() as f32
            })
        })
        .collect();
    NumericTensor::new([input.shape[0], input.shape[1], rows.len() as i32], data)
}

fn normalized(
    input: &NumericTensor,
    norm: &eredu_core::component::ComponentNormalization,
    parameters: &Parameters,
) -> NumericTensor {
    use eredu_core::component::ComponentNormalizationKind;
    assert_eq!(norm.kind, ComponentNormalizationKind::Rms);
    assert_eq!(norm.groups, 1);
    let width = input.shape[2] as usize;
    let gain = norm.gain.as_ref().map(|name| &parameters.0[name]);
    let data = input
        .data
        .chunks_exact(width)
        .flat_map(|row| {
            let denom = (row.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / width as f64
                + norm.epsilon.value() as f64)
                .sqrt();
            row.iter().enumerate().map(move |(i, v)| {
                (*v as f64 / denom
                    * gain.map_or(1.0, |g| g.data[i] as f64 + norm.gain_offset.value() as f64))
                    as f32
            })
        })
        .collect();
    NumericTensor::new(input.shape.clone(), data)
}

#[test]
fn kimi_discovered_reads_writes_and_convolutions_match_consumed_values() {
    use eredu_architectures::kimi_linear as family;
    use eredu_core::component::*;
    let context = NumericContext::default();
    for low_rank in [false, true] {
        let args = config(low_rank, false);
        let descriptor = eredu_architectures::configuration::MODEL_CONFIGURATIONS
            .resolve_safetensors(&config_value(low_rank))
            .unwrap()
            .architecture_plan()
            .architecture_descriptor();
        assert_eq!(descriptor.components.len(), 4);
        assert_eq!(descriptor.routed_components.len(), 1);
        assert_eq!(descriptor.component_transforms.len(), 3);
        assert_eq!(
            descriptor,
            serde_json::from_value(serde_json::to_value(&descriptor).unwrap()).unwrap()
        );
        let mut model =
            family::LayeredModel::<NumericBackend>::new(args.clone(), &context).unwrap();
        model
            .static_modules_mut()
            .visit_parameters_mut(&mut Initialize);
        let mut parameters = Parameters::default();
        model.static_modules().visit_parameters(&mut parameters);
        let units = (0..2)
            .map(|layer| {
                let mut unit = Block::<NumericBackend>::new(&args, layer, &context).unwrap();
                unit.visit_parameters_mut(&mut Initialize);
                unit.visit_parameters(&mut parameters);
                unit
            })
            .collect();
        let mut runtime = LayerwiseRuntime::new(model, ResidentUnitWindow::new(units));
        let mut state = DeviceState::<NumericBackend, _>::create(
            family::state_layout(&args).unwrap(),
            |_, p| Ok::<_, Error>(NumericHybridLayerState::new(p)),
        )
        .unwrap();
        let mut histories: BTreeMap<String, Vec<f32>> = BTreeMap::new();
        for ids in [vec![1, 3, 2], vec![4], vec![5]] {
            let tokens = NumericTensor::token_ids(&ids);
            let mut capture = Components::strict();
            let logits = runtime
                .forward_with_observer(
                    decoder::LayeredInput {
                        tokens: &tokens,
                        mask: None,
                    },
                    &mut state,
                    &context,
                    &mut capture,
                )
                .unwrap();
            for group in &descriptor.components {
                let input = &capture.values[&group.input];
                for path in [&group.activation, &group.effective_activation] {
                    assert!(descriptor.observations.get(path).is_some());
                    assert_eq!(capture.values[path].shape[2] as usize, group.count);
                }
                let write = &parameters.0[&group.write_weight];
                let expected = affine(
                    &capture.values[group.write_input.as_ref().unwrap()],
                    write,
                    0..write.shape[0] as usize,
                );
                assert_tensor_close(
                    &expected,
                    &capture.values[group.write_output.as_ref().unwrap()],
                    "declared Kimi write columns",
                );
                for read in &group.reads {
                    assert!(descriptor
                        .parameter_groups
                        .iter()
                        .any(|g| g.id == read.parameter_group));
                    let weight = &parameters.0[&read.weight];
                    for component in 0..group.count {
                        assert!(
                            read.rows.row_range(component).unwrap().end <= weight.shape[0] as usize
                        );
                    }
                    let mut projected = input.clone();
                    for stage in &read.input_projections {
                        projected =
                            affine(&projected, &parameters.0[&stage.weight], stage.rows.clone());
                        if let Some(norm) = &stage.normalization {
                            projected = normalized(&projected, norm, &parameters);
                        }
                        assert_tensor_close(
                            &projected,
                            &capture.values[&stage.output],
                            "declared Kimi latent read chain",
                        );
                    }
                    if let Some(output) = &read.projection_output {
                        projected = affine(&projected, weight, 0..weight.shape[0] as usize);
                        assert_tensor_close(
                            &projected,
                            &capture.values[output],
                            "declared Kimi pre-convolution read",
                        );
                    }
                }
                if let ComponentActivation::GatedDeltaAttention {
                    value_heads: heads,
                    key_head_width: head_width,
                    decay_rate,
                    decay_bias,
                    channel_normalization,
                    output_gate,
                    query_scale,
                    key_scale,
                    ..
                } = &group.activation_equation
                {
                    assert_eq!(
                        parameters.0[&decay_rate.parameter].shape,
                        [1, 1, *heads as i32, 1]
                    );
                    assert_eq!(
                        parameters.0[&decay_bias.parameter].data.len(),
                        heads * head_width
                    );
                    assert_eq!(
                        parameters.0[channel_normalization.normalization.gain.as_ref().unwrap()]
                            .shape,
                        [*head_width as i32]
                    );
                    assert_eq!(*output_gate, ComponentNonlinearity::Sigmoid);
                    assert_eq!(query_scale.value(), 1.0 / *head_width as f32);
                    assert_eq!(key_scale.value(), (*head_width as f32).sqrt().recip());
                    assert!(group
                        .reads
                        .iter()
                        .any(|r| r.role == ComponentReadRole::Decay));
                    assert!(group
                        .reads
                        .iter()
                        .any(|r| r.role == ComponentReadRole::Update));
                }
            }
            for transform in &descriptor.component_transforms {
                let ComponentTensorTransformEquation::CausalDepthwiseConvolution {
                    kernel,
                    channels,
                    taps,
                    residual,
                    activation,
                } = &transform.equation
                else {
                    panic!("Kimi causal read transform")
                };
                assert!(!residual);
                assert_eq!(
                    *activation,
                    Some(ComponentNonlinearity::Silu {
                        multiplier: ComponentScalar::new(1.0)
                    })
                );
                let input = &capture.values[&transform.input];
                let kernel = &parameters.0[&kernel.parameter];
                let history = histories.entry(transform.id.clone()).or_default();
                let offset = history.len() / channels;
                history.extend_from_slice(&input.data);
                let expected = (0..ids.len() * channels)
                    .map(|i| {
                        let time = offset + i / channels;
                        let channel = i % channels;
                        let value = (0..*taps)
                            .filter_map(|tap| {
                                (time + tap + 1).checked_sub(*taps).map(|position| {
                                    history[position * channels + channel] as f64
                                        * kernel.data[channel * taps + tap] as f64
                                })
                            })
                            .sum::<f64>();
                        (value / (1.0 + (-value).exp())) as f32
                    })
                    .collect();
                assert_tensor_close(
                    &NumericTensor::new(input.shape.clone(), expected),
                    &capture.values[&transform.output],
                    "declared causal SiLU transform with preceding history",
                );
            }
            let readout = descriptor.component_readout.as_ref().unwrap();
            let mut residual = capture.values[&format!("{}.effective", readout.embedding)].clone();
            for path in [
                "model.layers.0.attention.output.effective",
                "model.layers.0.feed_forward.output.effective",
                "model.layers.1.attention.output.effective",
                "model.layers.1.feed_forward.contribution.effective",
            ] {
                residual = residual.add(&capture.values[path], &context).unwrap();
            }
            assert_tensor_close(
                &residual,
                &capture.values[&readout.residual],
                "Kimi residual accounting without double-counting shared writes",
            );
            let normalized = normalized(&residual, &readout.normalization, &parameters);
            assert_tensor_close(
                &normalized,
                &capture.values[&readout.normalized],
                "Kimi declared readout normalization",
            );
            let head = &parameters.0[&readout.weight];
            let scores = affine(&normalized, head, 0..head.shape[0] as usize);
            assert_tensor_close(&scores, &logits, "Kimi signed affine score reconstruction");
            for (expected, actual) in scores
                .data
                .chunks_exact(args.vocab_size as usize)
                .zip(logits.data.chunks_exact(args.vocab_size as usize))
            {
                assert!(((expected[1] - expected[7]) - (actual[1] - actual[7])).abs() < 2e-5);
            }
        }
    }
}

#[test]
fn kimi_prepared_transforms_retain_source_layout_across_residency_and_parallelism() {
    use eredu_architectures::kimi_linear as family;
    for routed in [false, true] {
        let mut config = config_value(true);
        for (field, value) in [
            ("hidden_size", 32),
            ("num_attention_heads", 2),
            ("num_key_value_heads", 2),
            ("head_dim", 16),
            ("intermediate_size", 64),
            ("moe_intermediate_size", 64),
            ("kv_lora_rank", 16),
            ("q_lora_rank", 16),
            ("qk_nope_head_dim", 16),
            ("qk_rope_head_dim", 2),
            ("v_head_dim", 16),
        ] {
            config[field] = value.into();
        }
        config["linear_attn_config"]["num_heads"] = 2.into();
        config["linear_attn_config"]["head_dim"] = 16.into();
        config["first_k_dense_replace"] = if routed { 1 } else { 2 }.into();
        let (artifact, _) =
            prepared_adapter::payload_fixture_config_with(&config, 1.0, |name, shape| {
                let seed = name
                    .bytes()
                    .fold(0_u32, |a, b| a.wrapping_mul(31).wrapping_add(b.into()));
                Some(NumericTensor::new(
                    shape.to_vec(),
                    (0..shape.iter().product::<i32>() as usize)
                        .map(|index| {
                            let delta = ((index * 7 + (seed % 97) as usize) % 41) as f32 - 20.0;
                            if name.contains("norm") && name.ends_with("weight") {
                                1.0 + delta * 0.003
                            } else if name.ends_with("A_log") {
                                -0.5 + delta * 0.01
                            } else {
                                delta * 0.01
                            }
                        })
                        .collect(),
                ))
            });
        let inspection =
            eredu_architectures::configuration::inspect_artifact(artifact.path()).unwrap();
        let args = family::model_args_from_config_value(&config).unwrap();
        let parameters =
            family::LayeredModel::<NumericBackend>::new(args.clone(), &NumericContext::default())
                .unwrap()
                .parameter_description(&NumericContext::default())
                .unwrap();
        let mut topologies = vec![(2, 1, 1), (1, 2, 1), (2, 2, 1)];
        if routed {
            topologies.extend([(1, 1, 2), (2, 1, 2), (1, 2, 2), (2, 2, 2)]);
        }
        let mut reference: Option<Vec<NumericTensor>> = None;
        for (tp, pp, ep) in topologies {
            let topology = ParallelTopology::new(tp, pp, ep, 1).unwrap();
            for residency in [
                eredu_core::ResidencyPlan::FullyResident,
                eredu_core::ResidencyPlan::LayerwiseHost {
                    device_layer_window: 1,
                    device_budget_bytes: Some(1 << 20),
                    host_budget_bytes: Some(1 << 20),
                },
                eredu_core::ResidencyPlan::DenseDiskStream {
                    device_budget_bytes: 1 << 20,
                    host_budget_bytes: 1 << 20,
                    host_lookahead: 1,
                    background_queue: 1,
                },
            ] {
                let plan = prepared_adapter::plan(Some(eredu_core::QuantizationRequest::Affine {
                    group_size: 16,
                    bits: 4,
                }))
                .with_topology(topology)
                .with_residency(residency);
                let world = Arc::new(NumericPartitionWorld::default());
                let results = std::thread::scope(|scope| {
                    let handles = (0..topology.world_size()).map(|rank| {
                        let (inspection, plan, args, parameters) = (&inspection, &plan, &args, &parameters);
                        let world = world.clone();
                        scope.spawn(move || {
                            let sources = partitioned_adapter::prepare_plan(inspection, plan, rank, std::time::Duration::from_secs(30)).unwrap();
                            let selected = sources.selected().execution().text_realization();
                            assert!(selected.parameters().iter().any(|p| matches!(p.lowering(), eredu_runtime::WeightLoweringKind::Transform | eredu_runtime::WeightLoweringKind::DerivedTransform)));
                            let target_args = family::with_checkpoint_formats(args, selected.parameters().iter().filter_map(|p| p.executable().weight_quantization().map(|format| (p.name().to_owned(), format))).collect()).unwrap();
                            let target = family::LayeredModel::<NumericBackend>::new(target_args, &NumericContext::default()).unwrap();
                            let target_parameters = target.parameter_description(&NumericContext::default()).unwrap();
                            let rank_topology = ParallelRankTopology::new(topology, rank).unwrap();
                            let mut layout = eredu_architectures::partitioned_execution::derive_partitioned_local_layout(parameters, rank_topology).unwrap();
                            let encoded = eredu_architectures::partitioned_execution::derive_partitioned_local_layout(&target_parameters, rank_topology).unwrap();
                            for (name, companion) in encoded.tensors() { if !layout.contains(name) { layout.insert(name.to_owned(), companion.clone()); } }
                            let mut context = NumericContext::with_partition(layout, rank, world);
                            context.bind_checkpoint_values = true;
                            let mut executable = if routed {
                                partitioned_adapter::routed(sources, &context, Arc::new(AtomicUsize::new(0)), None)
                            } else { partitioned_adapter::dense(sources, &context) }.unwrap();
                            [&[1_usize, 3, 2][..], &[4][..], &[5][..]].into_iter().enumerate().map(|(step, tokens)| {
                                executable.forward_observed(&NumericTensor::token_ids(tokens), step == 0, &mut Components::strict()).unwrap()
                            }).collect::<Vec<_>>()
                        })
                    }).collect::<Vec<_>>();
                    handles
                        .into_iter()
                        .map(|handle| handle.join().unwrap())
                        .collect::<Vec<_>>()
                });
                if reference.is_none() {
                    reference = Some(results[0].clone());
                }
                for result in results {
                    for (actual, expected) in result.iter().zip(reference.as_ref().unwrap()) {
                        assert_tensor_close(
                            actual,
                            expected,
                            "Kimi exact transform-source placement and cached execution",
                        );
                    }
                }
            }
        }
    }
}
