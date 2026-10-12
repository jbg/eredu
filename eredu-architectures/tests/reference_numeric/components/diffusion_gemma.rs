//! Independent scalar reconstruction of feedback and shared invocation scales.
use super::*;
use eredu_architectures::diffusion_gemma::{text::SelfConditioning, FamilyConfig};

fn config() -> FamilyConfig {
    FamilyConfig::from_hf_json(
        br#"{
        "model_type":"diffusion_gemma", "canvas_length":3, "text_config": {
        "model_type":"diffusion_gemma_text", "hidden_size":4, "num_hidden_layers":1,
        "intermediate_size":6, "num_attention_heads":1, "num_key_value_heads":1,
        "head_dim":4, "rms_norm_eps":0.00001, "vocab_size":11,
        "max_position_embeddings":64, "num_experts":3, "top_k_experts":2,
        "moe_intermediate_size":3, "layer_types":["full_attention"]}}
    "#,
    )
    .unwrap()
}

#[test]
fn diffusion_feedback_normalizes_the_combined_embedding_with_and_without_prior_scores() {
    let context = NumericContext::default();
    let config = config();
    let mut module = SelfConditioning::<NumericBackend>::new(&config.text, &context).unwrap();
    let input = NumericTensor::new([1, 2, 4], vec![0.3, -0.7, 1.1, 0.2, -0.4, 0.8, 0.1, -1.3]);
    let prior = NumericTensor::new([1, 2, 4], vec![0.1, 0.5, -0.2, 0.7, -0.6, 0.2, 0.4, -0.1]);
    // Nonzero, explicitly supplied weights avoid constructor-dependent answers.
    module.pre_norm.weight.data = vec![0.8, 1.1, 0.9, 1.2];
    for (phase, linear) in [
        (0.2f32, &mut module.gate),
        (0.4, &mut module.up),
        (0.6, &mut module.down),
    ] {
        for (i, value) in linear.weight.data.iter_mut().enumerate() {
            *value = ((i as f32 * 0.37 + phase).sin()) * 0.2;
        }
    }
    let project = |row: &[f64], weight: &NumericTensor| -> Vec<f64> {
        weight
            .data
            .chunks_exact(row.len())
            .map(|w| w.iter().zip(row).map(|(w, x)| f64::from(*w) * x).sum())
            .collect()
    };
    for feedback in [None, Some(&prior)] {
        let actual = module.forward(&input, feedback, &context).unwrap();
        let mut expected = Vec::new();
        for token in 0..2 {
            let signal: Vec<f64> = feedback.map_or(vec![0.0; 4], |t| {
                t.data[token * 4..token * 4 + 4]
                    .iter()
                    .map(|x| f64::from(*x))
                    .collect()
            });
            let rms = (signal.iter().map(|x| x * x).sum::<f64>() / 4.0 + 1e-5).sqrt();
            let normalized: Vec<_> = signal
                .iter()
                .zip(&module.pre_norm.weight.data)
                .map(|(x, w)| x / rms * f64::from(*w))
                .collect();
            let gate = project(&normalized, &module.gate.weight);
            let up = project(&normalized, &module.up.weight);
            let activated: Vec<_> = gate
                .iter()
                .zip(up)
                .map(|(x, u)| {
                    let gelu = 0.5
                        * x
                        * (1.0
                            + ((2.0 / std::f64::consts::PI).sqrt() * (x + 0.044715 * x * x * x))
                                .tanh());
                    gelu * u
                })
                .collect();
            let transformed = project(&activated, &module.down.weight);
            let combined: Vec<_> = input.data[token * 4..token * 4 + 4]
                .iter()
                .zip(transformed)
                .map(|(x, s)| f64::from(*x) + s)
                .collect();
            let rms = (combined.iter().map(|x| x * x).sum::<f64>() / 4.0 + 1e-5).sqrt();
            expected.extend(combined.into_iter().map(|x| (x / rms) as f32));
        }
        assert_tensor_close(
            &actual,
            &NumericTensor::new([1, 2, 4], expected),
            "independent diffusion feedback equation",
        );
        assert!(actual.data.iter().any(|value| value.abs() > 0.1));
    }
}

#[test]
fn invocation_scale_does_not_change_shared_decoder_parameters() {
    let context = NumericContext::default();
    let config = config();
    let mut block = gemma4::DenseBlock::<NumericBackend>::new_at(
        &config.text,
        0,
        "model.decoder.layers",
        &context,
    )
    .unwrap();
    block
        .layer_scalar
        .replace(NumericTensor::new([1], vec![0.7]));
    let input = NumericTensor::new([1, 3, 4], (0..12).map(|i| (i as f32 * 0.3).sin()).collect());
    let run = |block: &mut gemma4::DenseBlock<NumericBackend>, scalar: Option<&NumericTensor>| {
        let mut shared = gemma4::SharedAttentionStates::new();
        let input = gemma4::BlockInput {
            hidden: &input,
            mask: None,
            cache: None::<&mut NumericCache>,
            shared: &mut shared,
            per_layer_input: None,
            rotary_position: None,
        };
        match scalar {
            Some(scalar) => block
                .forward_with_layer_scalar(
                    input,
                    scalar,
                    ExpertPass::Prefill,
                    &mut eredu_runtime::ResidentExpertProvider,
                    &context,
                    &mut ComponentInstrumentation::disabled(),
                )
                .unwrap(),
            None => block.forward(input, &context).unwrap(),
        }
    };
    let ordinary = run(&mut block, None);
    let encoder = run(&mut block, Some(&NumericTensor::new([1], vec![1.4])));
    let expected = NumericTensor::new(
        ordinary.shape.clone(),
        ordinary.data.iter().map(|x| 2.0 * x).collect(),
    );
    assert_tensor_close(
        &encoder,
        &expected,
        "independent encoder scale over shared residual",
    );
    assert_tensor_exact(
        &run(&mut block, None),
        &ordinary,
        "encoder invocation preserves decoder parameters",
    );
}

impl eredu_nn::ReadOnlyAttentionCache<NumericTensor> for NumericCache {
    fn attention_with_transient(
        &mut self,
        request: AttentionRequest<'_, NumericTensor>,
        context: &NumericContext,
    ) -> Result<NumericTensor, Error> {
        let retained = self
            .window
            .map_or(self.retained(), |window| self.retained().min(window - 1));
        let merge = |history: &Option<NumericTensor>,
                     transient: &NumericTensor|
         -> Result<NumericTensor, Error> {
            match history {
                Some(history) if retained > 0 => NumericTensor::concatenate(
                    &[
                        history.axis_slice(
                            2,
                            (history.shape[2] - retained) as usize,
                            history.shape[2] as usize,
                        ),
                        transient.clone(),
                    ],
                    2,
                    context,
                ),
                _ => Ok(transient.clone()),
            }
        };
        let keys = merge(&self.keys, &request.keys)?;
        let values = merge(&self.values, &request.values)?;
        match request.softcap {
            Some(cap) => attention_with_softcap(
                &request.queries,
                &keys,
                &values,
                request.scale,
                request.mask,
                request.sinks,
                Some(cap),
            ),
            None => attention_with_sinks(
                &request.queries,
                &keys,
                &values,
                request.scale,
                request.mask,
                request.sinks,
            ),
        }
    }
}

struct Nonzero;
impl eredu_nn::ReadOnlyAttentionCache<NumericTensor> for NumericHybridLayerState {
    fn attention_with_transient(
        &mut self,
        request: AttentionRequest<'_, NumericTensor>,
        context: &NumericContext,
    ) -> Result<NumericTensor, Error> {
        self.attention
            .as_mut()
            .ok_or_else(|| Error::backend("fixed layer has no retained attention context"))?
            .attention_with_transient(request, context)
    }
}

impl<'a> ParameterVisitorMut<'a, NumericTensor> for Nonzero {
    fn visit_mut(&mut self, metadata: ParameterMetadata, value: &'a mut NumericTensor) {
        let name = metadata.id.as_str();
        let phase = name.bytes().map(u32::from).sum::<u32>() as f32 * 0.013;
        let scalar = name.ends_with("layer_scalar");
        let norm = name.contains("norm") && name.ends_with("weight");
        for (i, slot) in value.data.iter_mut().enumerate() {
            *slot = if scalar || norm {
                1.0
            } else {
                ((i as f32 * 0.37 + phase).sin()) * 0.15
            };
        }
    }
}

#[test]
fn diffusion_shared_runtime_refines_without_publishing_and_restores_authoritative_context() {
    use eredu_architectures::diffusion_gemma::model::{TextInput, TextModel};
    type State = DeviceState<NumericBackend, NumericCache>;
    let config = config();
    let context = NumericContext::default();
    let mut state = State::create(
        eredu_architectures::diffusion_gemma::state_layout(&config).unwrap(),
        |_, _| Ok::<_, Error>(NumericCache::new(None)),
    )
    .unwrap();
    let mut runtime = ResidentRuntime::new(
        TextModel::<NumericBackend>::new(config, &context).unwrap(),
        &context,
    )
    .unwrap();
    <TextModel<NumericBackend> as eredu_runtime::LayeredArchitecture<NumericBackend, State>>::static_modules_mut(runtime.architecture_mut()).visit_parameters_mut(&mut Nonzero);
    for unit in runtime.units_mut().iter_mut().flatten() {
        unit.visit_parameters_mut(&mut Nonzero);
    }
    runtime
        .forward(
            TextInput::Context {
                tokens: &NumericTensor::token_ids(&[2, 3, 5]),
                mask: None,
            },
            &mut state,
            &context,
        )
        .unwrap();
    let before = state.clone();
    let canvas = NumericTensor::token_ids(&[1, 4]);
    let first = runtime
        .forward(
            TextInput::Refinement {
                tokens: &canvas,
                feedback: None,
                mask: None,
            },
            &mut state,
            &context,
        )
        .unwrap();
    assert!(first.data.iter().any(|v| v.abs() > 1e-4));
    let other = runtime
        .forward(
            TextInput::Refinement {
                tokens: &NumericTensor::token_ids(&[6, 8]),
                feedback: Some(&first),
                mask: None,
            },
            &mut state,
            &context,
        )
        .unwrap();
    assert!(first
        .data
        .iter()
        .zip(&other.data)
        .any(|(a, b)| (a - b).abs() > 1e-4));
    assert_eq!(state.layer(0).unwrap().offset(), 3);
    assert_tensor_exact(
        state.layer(0).unwrap().keys.as_ref().unwrap(),
        before.clone().layer(0).unwrap().keys.as_ref().unwrap(),
        "refinement preserves authoritative keys",
    );
    assert_tensor_exact(
        state.layer(0).unwrap().values.as_ref().unwrap(),
        before.clone().layer(0).unwrap().values.as_ref().unwrap(),
        "refinement preserves authoritative values",
    );
    let replay = runtime
        .forward(
            TextInput::Refinement {
                tokens: &canvas,
                feedback: None,
                mask: None,
            },
            &mut state,
            &context,
        )
        .unwrap();
    assert_tensor_exact(&first, &replay, "repeated canvases read identical context");
    runtime
        .forward(
            TextInput::Context {
                tokens: &NumericTensor::token_ids(&[7, 9]),
                mask: None,
            },
            &mut state,
            &context,
        )
        .unwrap();
    assert_eq!(state.layer(0).unwrap().offset(), 5);
    let appended = runtime
        .forward(
            TextInput::Refinement {
                tokens: &canvas,
                feedback: None,
                mask: None,
            },
            &mut state,
            &context,
        )
        .unwrap();
    assert!(first
        .data
        .iter()
        .zip(&appended.data)
        .any(|(a, b)| (a - b).abs() > 1e-5));
    let mut restored = before;
    let replay = runtime
        .forward(
            TextInput::Refinement {
                tokens: &canvas,
                feedback: None,
                mask: None,
            },
            &mut restored,
            &context,
        )
        .unwrap();
    assert_tensor_exact(
        &first,
        &replay,
        "restored context reproduces refinement after a divergent append",
    );
}

#[test]
fn diffusion_bounded_shared_units_match_resident_context_feedback_and_append() {
    use eredu_architectures::diffusion_gemma::model::{TextInput, TextModel};
    type State = DeviceState<NumericBackend, NumericCache>;
    type Model = TextModel<NumericBackend>;
    let mut json = serde_json::json!({
        "model_type":"diffusion_gemma", "canvas_length":3, "text_config": {
        "model_type":"diffusion_gemma_text", "hidden_size":4, "num_hidden_layers":2,
        "intermediate_size":6, "num_attention_heads":1, "num_key_value_heads":1,
        "head_dim":4, "rms_norm_eps":0.00001, "vocab_size":11,
        "max_position_embeddings":64, "num_experts":3, "top_k_experts":2,
        "moe_intermediate_size":3, "layer_types":["full_attention", "full_attention"]}}
    );
    json["text_config"]["tie_word_embeddings"] = true.into();
    let config = FamilyConfig::from_hf_json(&serde_json::to_vec(&json).unwrap()).unwrap();
    let context = NumericContext::default();
    let make_state = || {
        State::create(
            eredu_architectures::diffusion_gemma::state_layout(&config).unwrap(),
            |_, _| Ok::<_, Error>(NumericCache::new(None)),
        )
        .unwrap()
    };
    let mut resident_state = make_state();
    let mut bounded_state = make_state();
    let mut resident =
        ResidentRuntime::new(Model::new(config.clone(), &context).unwrap(), &context).unwrap();
    let mut bounded = LayerwiseRuntime::new(
        Model::new(config, &context).unwrap(),
        RebuildingUnitPolicy::default(),
    );
    for architecture in [resident.architecture_mut(), bounded.architecture_mut()] {
        <Model as LayeredArchitecture<NumericBackend, State>>::static_modules_mut(architecture)
            .visit_parameters_mut(&mut Nonzero);
    }
    for unit in resident.units_mut().iter_mut().flatten() {
        unit.visit_parameters_mut(&mut Nonzero);
    }
    let mut feedback = None;
    for (context_tokens, canvas_tokens) in
        [(&[2, 3, 5][..], &[1, 4][..]), (&[7, 9][..], &[6, 8][..])]
    {
        for refine in [false, true, true] {
            let tokens = NumericTensor::token_ids(if refine {
                canvas_tokens
            } else {
                context_tokens
            });
            let input = || {
                if refine {
                    TextInput::Refinement {
                        tokens: &tokens,
                        feedback: feedback.as_ref(),
                        mask: None,
                    }
                } else {
                    TextInput::Context {
                        tokens: &tokens,
                        mask: None,
                    }
                }
            };
            let expected = resident
                .forward(input(), &mut resident_state, &context)
                .unwrap();
            let actual = bounded
                .forward_with_unit_executor(
                    input(),
                    &mut bounded_state,
                    &context,
                    |architecture, group, index, unit, hidden, state, forward, context| {
                        unit.visit_parameters_mut(&mut Nonzero);
                        architecture
                            .forward_unit(group, index, unit, hidden, state, forward, context)
                    },
                )
                .unwrap();
            assert_tensor_exact(
                &actual,
                &expected,
                "rebuilt diffusion layers preserve context and feedback equations",
            );
            if refine {
                feedback = Some(expected);
            }
            for layer in 0..2 {
                let a = resident_state.layer(layer).unwrap();
                let b = bounded_state.layer(layer).unwrap();
                assert_eq!(a.offset(), b.offset());
                assert_tensor_exact(
                    a.keys.as_ref().unwrap(),
                    b.keys.as_ref().unwrap(),
                    "bounded authoritative keys",
                );
                assert_tensor_exact(
                    a.values.as_ref().unwrap(),
                    b.values.as_ref().unwrap(),
                    "bounded authoritative values",
                );
            }
        }
    }
    let expected: Vec<_> = (0..6)
        .flat_map(|_| expected_stream_events(&[(0, 0), (0, 1)]))
        .collect();
    assert_eq!(bounded.policy().events, expected);
}

fn parallel_config() -> FamilyConfig {
    FamilyConfig::from_hf_json(
        br#"{
        "model_type":"diffusion_gemma", "canvas_length":3, "text_config": {
        "model_type":"diffusion_gemma_text", "hidden_size":8, "num_hidden_layers":2,
        "intermediate_size":6, "num_attention_heads":2, "num_key_value_heads":2,
        "head_dim":4, "rms_norm_eps":0.00001, "vocab_size":11,
        "max_position_embeddings":64, "num_experts":3, "top_k_experts":2,
        "moe_intermediate_size":4, "layer_types":["full_attention","full_attention"]}}
    "#,
    )
    .unwrap()
}

struct InvocationScales;
impl<'a> ParameterVisitorMut<'a, NumericTensor> for InvocationScales {
    fn visit_mut(&mut self, metadata: ParameterMetadata, value: &'a mut NumericTensor) {
        if metadata.id.as_str().ends_with("layer_scalar") {
            value.data.fill(1.0);
        }
    }
}

#[test]
fn diffusion_tp2_preserves_shared_context_and_full_canvas_feedback() {
    use eredu_architectures::diffusion_gemma::{
        model::{TextInput, TextModel},
        parallel,
    };
    type State = DeviceState<NumericBackend, NumericCache>;
    type Model = TextModel<NumericBackend>;
    let config = parallel_config();
    let context = NumericContext::default();
    let mut resident =
        ResidentRuntime::new(Model::new(config.clone(), &context).unwrap(), &context).unwrap();
    for unit in resident.units_mut().iter_mut().flatten() {
        unit.visit_parameters_mut(&mut InvocationScales);
    }
    let mut state = State::create(
        eredu_architectures::diffusion_gemma::state_layout(&config).unwrap(),
        |_, _| Ok::<_, Error>(NumericCache::new(None)),
    )
    .unwrap();
    let mut expected = Vec::new();
    let mut feedback = None;
    let invocations = [
        (false, vec![2, 3, 5]),
        (true, vec![1, 4]),
        (true, vec![6, 8]),
        (false, vec![7, 9]),
        (true, vec![1, 4]),
    ];
    for (refine, ids) in &invocations {
        let tokens = NumericTensor::token_ids(ids);
        let input = if *refine {
            TextInput::Refinement {
                tokens: &tokens,
                feedback: feedback.as_ref(),
                mask: None,
            }
        } else {
            TextInput::Context {
                tokens: &tokens,
                mask: None,
            }
        };
        let output = resident.forward(input, &mut state, &context).unwrap();
        if *refine {
            feedback = Some(output.clone());
        }
        expected.push(output);
    }
    assert!(expected[1].data.iter().any(|v| v.abs() > 1e-4));
    let mut groups = parallel::static_parameter_groups(&config).unwrap();
    for index in 0..2 {
        groups.extend(parallel::layer_parameter_groups(&config, index).unwrap());
    }
    let world = NumericParallelGroup::new(2);
    let outputs = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..2)
            .map(|rank| {
                let layout = numeric_local_layout(&groups, 2, rank).unwrap();
                let config = config.clone();
                let invocations = invocations.clone();
                let world = Arc::clone(&world);
                scope.spawn(move || {
                    let context = NumericContext::with_local_layout(layout.clone());
                    let geometry = parallel::local_geometry(&config, &layout, 0..2).unwrap();
                    let mut state = State::create(geometry.state_layout().clone(), |_, _| {
                        Ok::<_, Error>(NumericCache::new(None))
                    })
                    .unwrap();
                    let model = Model::new_parallel(config, geometry, &context).unwrap();
                    let units = (0..2)
                        .map(|index| {
                            let mut unit =
                                <Model as LayeredArchitecture<NumericBackend, State>>::build_unit(
                                    &model, 0, index, &context,
                                )
                                .unwrap();
                            unit.visit_parameters_mut(&mut InvocationScales);
                            unit
                        })
                        .collect();
                    let mut runtime = LayerwiseRuntime::new(model, ResidentUnitWindow::new(units));
                    let parallel = NumericParallelContext::new(rank, world);
                    #[derive(Default)]
                    struct Observer {
                        unit: bool,
                        full_canvas: bool,
                    }
                    impl eredu_runtime::ActivationObserver<NumericTensor, Error> for Observer {
                        fn observe(
                            &mut self,
                            path: &str,
                            value: &NumericTensor,
                        ) -> Result<(), Error> {
                            self.unit |= path == "model.decoder.layers.1.residual.before_scale";
                            if path == "readout.scores" {
                                assert_eq!(value.shape, [1, 2, 11]);
                                self.full_canvas = true;
                            }
                            Ok(())
                        }
                    }
                    let mut observer = Observer::default();
                    let mut feedback = None;
                    let mut outputs = Vec::new();
                    for (refine, ids) in invocations {
                        let tokens = NumericTensor::token_ids(&ids);
                        let input = if refine {
                            TextInput::Refinement {
                                tokens: &tokens,
                                feedback: feedback.as_ref(),
                                mask: None,
                            }
                        } else {
                            TextInput::Context {
                                tokens: &tokens,
                                mask: None,
                            }
                        };
                        let output = runtime
                            .forward_parallel_with_observer(
                                input,
                                &mut state,
                                &parallel,
                                &context,
                                &mut observer,
                            )
                            .unwrap();
                        if refine {
                            feedback = Some(output.clone());
                        }
                        outputs.push(output);
                    }
                    assert!(observer.unit && observer.full_canvas);
                    (outputs, state)
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect::<Vec<_>>()
    });
    for (rank, (outputs, mut local_state)) in outputs.into_iter().enumerate() {
        for (actual, expected) in outputs.iter().zip(&expected) {
            assert_tensor_close(actual, expected, &format!("diffusion TP2 rank {rank}"));
        }
        for layer in 0..2 {
            assert_eq!(local_state.layer(layer).unwrap().offset(), 5);
            let full = state.layer(layer).unwrap();
            let local = local_state.layer(layer).unwrap();
            for (a, b) in [
                (local.keys.as_ref().unwrap(), full.keys.as_ref().unwrap()),
                (
                    local.values.as_ref().unwrap(),
                    full.values.as_ref().unwrap(),
                ),
            ] {
                assert_tensor_close(
                    a,
                    &b.axis_slice(1, rank, rank + 1),
                    "diffusion rank-owned authoritative projections",
                );
            }
        }
    }
}

#[test]
fn diffusion_pipeline_phase_transport_matches_resident_with_and_without_tp() {
    use eredu_architectures::diffusion_gemma::{
        boundary::TextBoundarySchema,
        model::{TextInput, TextModel},
        parallel,
    };
    use eredu_runtime::ArchitectureBoundary;
    type State = DeviceState<NumericBackend, NumericCache>;
    type Model = TextModel<NumericBackend>;
    let config = parallel_config();
    let context = NumericContext::default();
    let model = Model::new(config.clone(), &context).unwrap();
    let description = model.parameter_description(&context).unwrap();
    let complete = eredu_architectures::diffusion_gemma::state_layout(&config).unwrap();
    let mut resident = ResidentRuntime::new(model, &context).unwrap();
    for unit in resident.units_mut().iter_mut().flatten() {
        unit.visit_parameters_mut(&mut InvocationScales);
    }
    let mut state = State::create(complete.clone(), |_, _| {
        Ok::<_, Error>(NumericCache::new(None))
    })
    .unwrap();
    let invocations = [
        (false, vec![2, 3, 5]),
        (true, vec![1, 4]),
        (true, vec![6, 8]),
        (false, vec![7, 9]),
        (true, vec![1, 4]),
    ];
    let mut expected = Vec::new();
    let mut feedback = None;
    for (step, (refine, ids)) in invocations.iter().enumerate() {
        let tokens = NumericTensor::token_ids(ids);
        let mask = (step == 1).then(|| {
            NumericTensor::new(
                [1, 1, 2, 5],
                vec![-1e9, 0.0, 0.0, 0.0, 0.0, -1e9, 0.0, 0.0, 0.0, 0.0],
            )
        });
        let input = if *refine {
            TextInput::Refinement {
                tokens: &tokens,
                feedback: feedback.as_ref(),
                mask: mask.as_ref(),
            }
        } else {
            TextInput::Context {
                tokens: &tokens,
                mask: mask.as_ref(),
            }
        };
        let output = resident.forward(input, &mut state, &context).unwrap();
        if *refine {
            feedback = Some(output.clone());
        }
        expected.push(output);
    }
    for tp in [1, 2] {
        let worlds = [NumericParallelGroup::new(tp), NumericParallelGroup::new(tp)];
        let outputs = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..tp)
                .map(|rank| {
                    let config = config.clone();
                    let description = description.clone();
                    let invocations = invocations.clone();
                    let worlds = worlds.clone();
                    scope.spawn(move || {
                        let make_stage = |stage: usize| {
                            let topology = ParallelTopology::new(tp, 2, 1, 1).unwrap();
                            let topology =
                                ParallelRankTopology::new(topology, stage * tp + rank).unwrap();
                            let layout = eredu_runtime::derive_partitioned_local_layout(
                                &description,
                                topology,
                            )
                            .unwrap();
                            let context = NumericContext::with_local_layout(layout.clone());
                            let range = stage..stage + 1;
                            let ownership = if stage == 0 {
                                PartitionOwnership::new(true, false, ["embedding", "conditioning"])
                                    .unwrap()
                            } else {
                                PartitionOwnership::new(false, true, ["norm", "output"]).unwrap()
                            };
                            let partition = parallel::partition(
                                &config,
                                &description,
                                &layout,
                                range.clone(),
                                ownership,
                            )
                            .unwrap();
                            let geometry = partition.local_geometry().clone();
                            let model =
                                Model::new_parallel(config.clone(), geometry, &context).unwrap();
                            let mut unit =
                                <Model as LayeredArchitecture<NumericBackend, State>>::build_unit(
                                    &model, 0, stage, &context,
                                )
                                .unwrap();
                            unit.visit_parameters_mut(&mut InvocationScales);
                            let state = State::create(
                                partition.state().unwrap().layout().clone(),
                                |_, _| Ok::<_, Error>(NumericCache::new(None)),
                            )
                            .unwrap();
                            let driver = LayeredPartitionDriver::new(&partition, 0, range).unwrap();
                            let parallel =
                                NumericParallelContext::new(rank, Arc::clone(&worlds[stage]));
                            (model, unit, state, driver, parallel, context)
                        };
                        let (
                            mut first_model,
                            mut first_unit,
                            mut first_state,
                            first_driver,
                            first_parallel,
                            first_context,
                        ) = make_stage(0);
                        let (
                            mut last_model,
                            mut last_unit,
                            mut last_state,
                            last_driver,
                            last_parallel,
                            last_context,
                        ) = make_stage(1);
                        let mut feedback = None;
                        let mut outputs = Vec::new();
                        for (step, (refine, ids)) in invocations.into_iter().enumerate() {
                            let tokens = NumericTensor::token_ids(&ids);
                            let mask = (step == 1).then(|| {
                                NumericTensor::new(
                                    [1, 1, 2, 5],
                                    vec![-1e9, 0.0, 0.0, 0.0, 0.0, -1e9, 0.0, 0.0, 0.0, 0.0],
                                )
                            });
                            let input = if refine {
                                TextInput::Refinement {
                                    tokens: &tokens,
                                    feedback: feedback.as_ref(),
                                    mask: mask.as_ref(),
                                }
                            } else {
                                TextInput::Context {
                                    tokens: &tokens,
                                    mask: mask.as_ref(),
                                }
                            };
                            let input = first_model
                                .prepare_partition_input(
                                    input,
                                    &mut first_state,
                                    Some(&first_parallel),
                                    &first_context,
                                )
                                .unwrap();
                            let mut first = first_driver
                                .begin(
                                    &mut first_model,
                                    input.input,
                                    input.mask.as_ref(),
                                    &mut first_state,
                                    Some(&first_parallel),
                                    &first_context,
                                )
                                .unwrap();
                            first.hidden = first_model
                                .forward_unit_parallel(
                                    0,
                                    0,
                                    &mut first_unit,
                                    &first.hidden,
                                    &mut first_state,
                                    &mut first.context,
                                    &first_parallel,
                                    &first_context,
                                )
                                .unwrap();
                            let LayeredPartitionOutput::Boundary { hidden, auxiliary } =
                                first_driver
                                    .finish(
                                        &mut first_model,
                                        &first.hidden,
                                        &mut first_state,
                                        &mut first.context,
                                        Some(&first_parallel),
                                        &first_context,
                                    )
                                    .unwrap()
                            else {
                                panic!("input stage must transport a boundary");
                            };
                            let schema = TextBoundarySchema::new(config.text.hidden_size);
                            let wire = schema.encode(auxiliary).unwrap();
                            let auxiliary = schema
                                .decode(
                                    wire.into_iter().map(|value| value.into_parts().1).collect(),
                                )
                                .unwrap();
                            let mut last = last_driver
                                .begin(
                                    &mut last_model,
                                    LayeredPartitionInput::Hidden { hidden, auxiliary },
                                    mask.as_ref(),
                                    &mut last_state,
                                    Some(&last_parallel),
                                    &last_context,
                                )
                                .unwrap();
                            last.hidden = last_model
                                .forward_unit_parallel(
                                    0,
                                    1,
                                    &mut last_unit,
                                    &last.hidden,
                                    &mut last_state,
                                    &mut last.context,
                                    &last_parallel,
                                    &last_context,
                                )
                                .unwrap();
                            let LayeredPartitionOutput::Final { output, .. } = last_driver
                                .finish(
                                    &mut last_model,
                                    &last.hidden,
                                    &mut last_state,
                                    &mut last.context,
                                    Some(&last_parallel),
                                    &last_context,
                                )
                                .unwrap()
                            else {
                                panic!("last stage must read out all positions");
                            };
                            if refine {
                                feedback = Some(output.clone());
                            }
                            outputs.push(output);
                        }
                        assert_eq!(first_state.layer(0).unwrap().offset(), 5);
                        assert_eq!(last_state.layer(0).unwrap().offset(), 5);
                        (outputs, first_state, last_state)
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect::<Vec<_>>()
        });
        for (rank, (outputs, mut first, mut last)) in outputs.into_iter().enumerate() {
            for (a, b) in outputs.iter().zip(&expected) {
                assert_tensor_close(a, b, &format!("diffusion TP{tp}/PP2 scores"));
            }
            for (layer, local) in [(0, &mut first), (1, &mut last)] {
                let owned = local.layer(0).unwrap();
                let full = state.layer(layer).unwrap();
                let range = balanced_rank_range(2, tp, rank);
                assert_tensor_close(
                    owned.keys.as_ref().unwrap(),
                    &full
                        .keys
                        .as_ref()
                        .unwrap()
                        .axis_slice(1, range.start, range.end),
                    "pipeline owned context keys",
                );
                assert_tensor_close(
                    owned.values.as_ref().unwrap(),
                    &full
                        .values
                        .as_ref()
                        .unwrap()
                        .axis_slice(1, range.start, range.end),
                    "pipeline owned context values",
                );
            }
        }
    }
}

#[test]
fn diffusion_routed_provider_preserves_request_pass_and_read_only_frontier() {
    use eredu_architectures::diffusion_gemma::model::{TextInput, TextModel};
    type State = DeviceState<NumericBackend, NumericCache>;
    type Model = TextModel<NumericBackend>;
    let config = config();
    let context = NumericContext::default();
    let make_state = || {
        State::create(
            eredu_architectures::diffusion_gemma::state_layout(&config).unwrap(),
            |_, _| Ok::<_, Error>(NumericCache::new(None)),
        )
        .unwrap()
    };
    let mut ordinary_state = make_state();
    let mut routed_state = make_state();
    let mut resident =
        ResidentRuntime::new(Model::new(config.clone(), &context).unwrap(), &context).unwrap();
    for unit in resident.units_mut().iter_mut().flatten() {
        unit.visit_parameters_mut(&mut InvocationScales);
    }
    let units = resident.units_mut().iter().flatten().cloned().collect();
    let mut routed = LayerwiseRuntime::new(
        Model::new(config, &context).unwrap(),
        ResidentUnitWindow::new(units),
    );
    let mut provider = RecordingNumericExpertProvider::default();
    #[derive(Default)]
    struct Observer {
        paths: std::collections::BTreeSet<String>,
        zero_residual: bool,
    }
    impl eredu_runtime::ActivationObserver<NumericTensor, Error> for Observer {
        fn observe(&mut self, path: &str, _: &NumericTensor) -> Result<(), Error> {
            self.paths.insert(path.to_owned());
            Ok(())
        }
        fn intervene(
            &mut self,
            path: &str,
            value: &NumericTensor,
        ) -> Result<Option<NumericTensor>, Error> {
            Ok(
                (self.zero_residual && path.ends_with("residual.before_scale"))
                    .then(|| NumericTensor::zeros(value.shape.clone())),
            )
        }
    }
    let mut observer = Observer::default();
    let tokens = NumericTensor::token_ids(&[2, 3]);
    let canvas = NumericTensor::token_ids(&[5, 7]);
    for (refine, pass) in [
        (false, ExpertPass::Prefill),
        (true, ExpertPass::Decode),
        (true, ExpertPass::Prefill),
    ] {
        let input = || {
            if refine {
                TextInput::Refinement {
                    tokens: &canvas,
                    feedback: None,
                    mask: None,
                }
            } else {
                TextInput::Context {
                    tokens: &tokens,
                    mask: None,
                }
            }
        };
        let expected = resident
            .forward(input(), &mut ordinary_state, &context)
            .unwrap();
        let actual = routed
            .forward_with_provider_and_observer(
                input(),
                &mut routed_state,
                pass,
                &mut provider,
                &context,
                &mut observer,
            )
            .unwrap();
        assert_tensor_exact(
            &actual,
            &expected,
            "provider preserves shared invocation equations",
        );
        assert_eq!(routed_state.layer(0).unwrap().offset(), 2);
    }
    assert!(observer
        .paths
        .contains("model.decoder.layers.0.routed_feed_forward.input"));
    assert!(observer
        .paths
        .contains("model.decoder.layers.0.residual.before_scale.effective"));
    observer.zero_residual = true;
    let intervened = routed
        .forward_with_provider_and_observer(
            TextInput::Refinement {
                tokens: &canvas,
                feedback: None,
                mask: None,
            },
            &mut routed_state,
            ExpertPass::Decode,
            &mut provider,
            &context,
            &mut observer,
        )
        .unwrap();
    assert!(intervened.data.iter().all(|x| *x == 0.0));
    assert_eq!(routed_state.layer(0).unwrap().offset(), 2);
    assert_eq!(
        provider.calls.iter().map(|call| call.1).collect::<Vec<_>>(),
        [
            ExpertPass::Prefill,
            ExpertPass::Decode,
            ExpertPass::Prefill,
            ExpertPass::Decode
        ]
    );
}

#[test]
fn diffusion_composite_prepared_partitions_preserve_padded_media_context_across_layer_cuts() {
    use eredu_architectures::{
        composite_execution::{CompositeArchitecture, PreparedCompositeInput},
        diffusion_gemma as family,
    };
    type Model = family::composite::Model<NumericBackend>;
    type State = DeviceState<NumericBackend, NumericHybridLayerState>;
    for sparse in [false, true] {
        let config = serde_json::json!({
            "model_type":"diffusion_gemma", "canvas_length":3, "image_token_id":5,
            "text_config": { "model_type":"diffusion_gemma_text", "hidden_size":8,
                "num_hidden_layers":4, "intermediate_size":12, "num_attention_heads":2,
                "num_key_value_heads":2, "head_dim":4, "rms_norm_eps":0.00001,
                "vocab_size":19, "max_position_embeddings":64, "enable_moe_block":sparse,
                "num_experts":sparse.then_some(4), "top_k_experts":sparse.then_some(2),
                "moe_intermediate_size":sparse.then_some(6), "sliding_window":4,
                "layer_types":["sliding_attention","full_attention","sliding_attention","full_attention"] },
            "vision_config": { "hidden_size":8, "intermediate_size":12, "num_hidden_layers":4,
                "num_attention_heads":2, "num_key_value_heads":2, "head_dim":4, "patch_size":2,
                "pooling_kernel_size":2, "position_embedding_size":4, "rms_norm_eps":0.00001 }
        });
        let (artifact, bits) =
            prepared_adapter::payload_fixture_config_with(&config, 1.0, |name, shape| {
                let phase = name.bytes().map(u32::from).sum::<u32>() as f32 * 0.017;
                Some(NumericTensor::new(
                    shape.to_vec(),
                    (0..shape.iter().product::<i32>())
                        .map(|i| {
                            if name.ends_with("layer_scalar") {
                                1.17
                            } else if (name.contains("norm") && name.ends_with("weight"))
                                || name.contains("std")
                            {
                                1.0 + (i as f32 * 0.13 + phase).sin() * 0.1
                            } else {
                                (i as f32 * 0.13 + phase).sin() * 0.12
                            }
                        })
                        .collect(),
                ))
            });
        let args =
            family::FamilyConfig::from_hf_json(&serde_json::to_vec(&config).unwrap()).unwrap();
        let context = NumericContext {
            bind_checkpoint_values: true,
            ..Default::default()
        };
        let store = eredu_checkpoint::store::SafetensorsWeightStore::open(artifact.path()).unwrap();
        let mut values: BTreeMap<_, _> = bits
            .into_iter()
            .map(|(name, (shape, bits))| {
                (
                    family::canonical_safetensors_target(&name),
                    NumericTensor::new(shape, bits.into_iter().map(f32::from_bits).collect()),
                )
            })
            .collect();
        for unit in 0..4 {
            for (name, recipe) in family::unit_recipes(&store, &args, unit).unwrap() {
                values.insert(
                    name,
                    payload::recipe_value(&recipe, &store, &context).unwrap(),
                );
            }
        }
        struct Populate<'a>(&'a BTreeMap<String, NumericTensor>);
        impl<'a> ParameterVisitorMut<'a, NumericTensor> for Populate<'_> {
            fn visit_mut(&mut self, metadata: ParameterMetadata, value: &'a mut NumericTensor) {
                let source = self
                    .0
                    .get(metadata.id.as_str())
                    .unwrap_or_else(|| panic!("missing {}", metadata.id));
                assert_eq!(value.shape, source.shape, "{}", metadata.id);
                value.data.clone_from(&source.data);
            }
        }
        let mut model = Model::new(args.clone(), &context).unwrap();
        <Model as LayeredArchitecture<NumericBackend, State>>::static_modules_mut(&mut model)
            .visit_parameters_mut(&mut Populate(&values));
        let units = (0..2)
            .flat_map(|group| (0..4).map(move |unit| (group, unit)))
            .map(|(group, index)| {
                let mut unit = <Model as LayeredArchitecture<NumericBackend, State>>::build_unit(
                    &model, group, index, &context,
                )
                .unwrap();
                unit.visit_parameters_mut(&mut Populate(&values));
                unit
            })
            .collect();
        let mut resident = LayerwiseRuntime::new(model, ResidentUnitWindow::new(units));
        let mut state = State::create(family::state_layout(&args).unwrap(), |_, policy| {
            Ok::<_, Error>(NumericHybridLayerState::new(policy))
        })
        .unwrap();
        let mut parts = super::gemma4_placement::image_input().parts().to_vec();
        parts.extend(numeric_text_prepared_input(&[2, 5]).parts().iter().cloned());
        parts.push(
            eredu_runtime::PreparedInputPart::new_with_extents(
                eredu_core::InputModality::Image,
                eredu_runtime::PreparedInputPayload::Tensor(NumericTensor::new(
                    [1, 8, 12],
                    (0..96).map(|i| (i as f32 * 0.021).cos() * 0.2).collect(),
                )),
                [
                    (
                        eredu_core::InputMetadataKey::PatchGrid,
                        NumericTensor::new([1, 3], vec![1.0, 2.0, 4.0]),
                    ),
                    (
                        eredu_core::InputMetadataKey::PatchPositions,
                        NumericTensor::new(
                            [1, 8, 2],
                            (0..8)
                                .flat_map(|i| [(i % 4) as f32, (i / 4) as f32])
                                .collect(),
                        ),
                    ),
                ],
                [eredu_core::InputExtent::PatchGrid {
                    time: 1,
                    height: 2,
                    width: 4,
                }],
            )
            .unwrap(),
        );
        let media = eredu_runtime::PreparedModelInput::new(parts, |tensor| {
            eredu_runtime::PreparedInputInspector::identity(&NumericInputInspector, tensor)
        })
        .unwrap();
        let inputs = [
            media,
            numeric_text_prepared_input(&[3]),
            numeric_text_prepared_input(&[4]),
        ];
        let canvas = numeric_text_prepared_input(&[1, 4, 7]);
        let mut feedback = None;
        let expected: Vec<_> = inputs
            .iter()
            .map(|input| {
                let admitted =
                    <Model as CompositeArchitecture<NumericBackend, State>>::admit_prepared_input(
                        &args,
                        input,
                        &NumericInputInspector,
                    )
                    .unwrap();
                let output = resident
                    .forward(
                        PreparedCompositeInput::new(input, &admitted).unwrap(),
                        &mut state,
                        &context,
                    )
                    .unwrap();
                let encoded = NumericTensor::new(
                    [1, 1, args.text.hidden_size],
                    output.data[output.data.len() - args.text.hidden_size as usize..].to_vec(),
                );
                let admitted =
                    <Model as CompositeArchitecture<NumericBackend, State>>::admit_prepared_input(
                        &args,
                        &canvas,
                        &NumericInputInspector,
                    )
                    .unwrap();
                let first = resident
                    .forward(
                        PreparedCompositeInput::new(&canvas, &admitted)
                            .unwrap()
                            .with_refinement(feedback.as_ref()),
                        &mut state,
                        &context,
                    )
                    .unwrap();
                let second = resident
                    .forward(
                        PreparedCompositeInput::new(&canvas, &admitted)
                            .unwrap()
                            .with_refinement(Some(&first)),
                        &mut state,
                        &context,
                    )
                    .unwrap();
                feedback = Some(second.clone());
                (encoded, first, second)
            })
            .collect();
        let inspection =
            eredu_architectures::configuration::inspect_artifact(artifact.path()).unwrap();
        let parameters = numeric_composite_parameter_description(&config);
        for (tp, pp, ep) in [(1, 4, 1), (2, 2, 1), (1, 2, 2), (2, 2, 2)] {
            if !sparse && ep > 1 {
                continue;
            }
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
                let topology = ParallelTopology::new(tp, pp, ep, 1).unwrap();
                let plan = prepared_adapter::plan(None)
                    .with_topology(topology)
                    .with_residency(residency);
                let world = Arc::new(NumericPartitionWorld::default());
                let outputs = std::thread::scope(|scope| {
                    let workers:Vec<_>=(0..topology.world_size()).map(|rank| {
                        let (inspection,parameters,inputs,plan,canvas)=(&inspection,&parameters,&inputs,&plan,&canvas);let world=world.clone();
                        std::thread::Builder::new().stack_size(8 << 20).spawn_scoped(scope, move || {
                            let sources=partitioned_adapter::prepare_plan_with_banks_and_sequence_maximum(inspection,plan,rank,std::time::Duration::from_secs(30),None,8).unwrap();
                            let layout=eredu_runtime::derive_partitioned_local_layout(parameters,ParallelRankTopology::new(topology,rank).unwrap()).unwrap();
                            let mut context=NumericContext::with_partition(layout,rank,world);context.bind_checkpoint_values=true;context.cache_owned_attention=true;
                            let mut executable=partitioned_adapter::composite(sources,&context).unwrap();
                            let mut feedback=None;
                            inputs.iter().enumerate().map(|(step,input)| {
                                let encoded=executable.forward(input,step==0).unwrap();
                                let before=executable.positions().unwrap();
                                let first=executable.refine(canvas,feedback.as_ref()).unwrap();
                                let second=executable.refine(canvas,Some(&first)).unwrap();
                                assert_eq!(executable.positions().unwrap(),before,"refinement preserves every rank-owned frontier");
                                feedback=Some(second.clone());
                                (encoded,first,second)
                            }).collect::<Vec<_>>()
                        }).unwrap()
                    }).collect();
                    workers
                        .into_iter()
                        .map(|worker| worker.join().unwrap())
                        .collect::<Vec<_>>()
                });
                for (rank, actual) in outputs.into_iter().enumerate() {
                    for (step, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
                        for (actual, expected) in [
                            (&actual.0, &expected.0),
                            (&actual.1, &expected.1),
                            (&actual.2, &expected.2),
                        ] {
                            assert_tensor_close(actual,expected,&format!("diffusion composite sparse={sparse} TP{tp}/PP{pp}/EP{ep} rank={rank} step={step}"));
                        }
                    }
                }
            }
        }
    }
}
