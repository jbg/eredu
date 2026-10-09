//! Coupled channel ingress and readout through the ordinary cached Nemotron driver.
use super::*;
use eredu_architectures::voicechat::{
    backbone_execution::{execute, BackboneRequest, OrdinaryBackbone, ParallelBackbone},
    channels::{ChannelFusion, FunctionHead, PromptPolicy},
    config::VoiceChatConfig,
    frontend::FrontendOptions,
    function_head_layered::ParallelFunctionHead,
    input::InputState,
    progress::{observe, ProgressError},
};
use serde_json::Value;

fn tensor(value: &Value) -> NumericTensor {
    NumericTensor::new(
        value["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap() as i32)
            .collect::<Vec<_>>(),
        value["values"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect(),
    )
}

struct BindStatic<'a>(&'a Value);
impl<'a> ParameterVisitorMut<'a, NumericTensor> for BindStatic<'_> {
    fn visit_mut(&mut self, metadata: ParameterMetadata, slot: &'a mut NumericTensor) {
        let name = match metadata.id.as_str() {
            "model.embeddings.weight" => "stt_model.embed_tokens.weight",
            "model.norm_f.weight" => "stt_model.llm.norm_f.weight",
            "lm_head.weight" => "stt_model.lm_head.weight",
            "stt_model.function_head.weight" => "stt_model.function_head.weight",
            name => panic!("unexpected static parameter {name}"),
        };
        let value = tensor(&self.0["parameters"][name]);
        assert_eq!(slot.shape(), value.shape());
        *slot = value;
    }
}

fn args() -> nemotron_h::ModelArgs {
    nemotron_h::model_args_from_config_value(&serde_json::json!({
        "model_type":"nemotron_h", "vocab_size":24, "hidden_size":4,
        "intermediate_size":8, "num_hidden_layers":3,
        "hybrid_override_pattern":"M*-", "num_attention_heads":2,
        "num_key_value_heads":2, "head_dim":2, "mamba_num_heads":2,
        "n_groups":2, "mamba_head_dim":2, "ssm_state_size":3,
        "conv_kernel":3, "chunk_size":2, "num_nextn_predict_layers":0,
        "tie_word_embeddings":false, "residual_in_fp32":true,
        "layer_norm_epsilon":1e-5
    }))
    .unwrap()
}

pub(super) fn model(
    fixture: &Value,
    context: &NumericContext,
) -> nemotron_h::LayeredModel<NumericBackend> {
    let mut model = nemotron_h::LayeredModel::new(args(), context).unwrap();
    model
        .static_modules_mut()
        .visit_parameters_mut(&mut BindStatic(fixture));
    model
}

pub(super) fn function(fixture: &Value, context: &NumericContext) -> FunctionHead<NumericBackend> {
    let mut head = FunctionHead::new(4, 24, dense_linear_format(), context).unwrap();
    head.visit_parameters_mut(&mut BindStatic(fixture));
    head
}

pub(super) fn state() -> DeviceState<NumericBackend, NumericHybridLayerState> {
    DeviceState::create(nemotron_h::state_layout(&args()).unwrap(), |_, policy| {
        Ok::<_, Error>(NumericHybridLayerState::new(policy))
    })
    .unwrap()
}

/// The same compact sharded backbone feeds component and coupled-session tests.
pub(super) fn parallel_components(
    fixture: &Value,
    rank: usize,
) -> (
    NumericContext,
    LayerwiseRuntime<
        nemotron_h::LayeredModel<NumericBackend>,
        NumericBackend,
        DeviceState<NumericBackend, NumericHybridLayerState>,
        RebuildingUnitPolicy,
    >,
    FunctionHead<NumericBackend>,
    DeviceState<NumericBackend, NumericHybridLayerState>,
) {
    let context = NumericContext::default();
    let architecture = model(fixture, &context);
    let description = architecture.parameter_description(&context).unwrap();
    let groups = description
        .groups()
        .iter()
        .map(|g| g.group().clone())
        .collect::<Vec<_>>();
    let layout = numeric_local_layout(&groups, 2, rank).unwrap();
    struct BindShard<'a> {
        fixture: &'a Value,
        range: std::ops::Range<usize>,
    }
    impl<'a> ParameterVisitorMut<'a, NumericTensor> for BindShard<'_> {
        fn visit_mut(&mut self, metadata: ParameterMetadata, slot: &'a mut NumericTensor) {
            let full_shape = if slot.shape().len() == 2 {
                vec![24, 4]
            } else {
                vec![4]
            };
            let mut full = NumericTensor::zeros(full_shape);
            BindStatic(self.fixture).visit_mut(metadata, &mut full);
            *slot = if full.shape().len() == 2 {
                full.axis_slice(0, self.range.start, self.range.end)
            } else {
                full
            };
        }
    }
    let context = NumericContext::with_local_layout(layout.clone());
    let geometry = nemotron_h::local_geometry(&args(), &layout).unwrap();
    let state_layout = geometry.state_layout().clone();
    let range = geometry.embedding_range().clone();
    let mut architecture =
        nemotron_h::LayeredModel::<NumericBackend>::new_parallel(args(), geometry, &context)
            .unwrap();
    architecture
        .static_modules_mut()
        .visit_parameters_mut(&mut BindShard {
            fixture,
            range: range.local.clone(),
        });
    let mut head =
        FunctionHead::new_parallel(4, 24, dense_linear_format(), range.clone(), &context).unwrap();
    head.visit_parameters_mut(&mut BindShard {
        fixture,
        range: range.local,
    });
    let runtime = LayerwiseRuntime::new(architecture, RebuildingUnitPolicy::default());
    let state = DeviceState::<NumericBackend, _>::create(state_layout, |_, policy| {
        Ok::<_, Error>(NumericHybridLayerState::new(policy))
    })
    .unwrap();
    (context, runtime, head, state)
}

// Independent released F32 equation, using the NVIDIA fixture's shared embedding
// table instead of ChannelFusion or the runtime's embedding dispatch.
fn fused(fixture: &Value, audio: &[f32], agent: &[i32], function: &[i32]) -> NumericTensor {
    let embedding = tensor(&fixture["parameters"]["stt_model.embed_tokens.weight"]);
    assert_eq!(audio.len(), agent.len() * 4);
    let mut data = Vec::with_capacity(audio.len());
    for row in 0..agent.len() {
        for column in 0..4 {
            data.push(
                (embedding.data[agent[row] as usize * 4 + column] + audio[row * 4 + column])
                    + 2.0 * embedding.data[function[row] as usize * 4 + column],
            );
        }
    }
    NumericTensor::new(vec![1, agent.len() as i32, 4], data)
}

// The fixture head is deliberately independent of the agent head. Projecting the
// expected normalized result directly catches swapped heads and a second RMSNorm.
fn expected_function(fixture: &Value, normalized: &NumericTensor) -> NumericTensor {
    let weights = tensor(&fixture["parameters"]["stt_model.function_head.weight"]);
    let data = normalized
        .data
        .chunks_exact(4)
        .flat_map(|hidden| {
            weights.data.chunks_exact(4).map(move |row| {
                hidden
                    .iter()
                    .zip(row)
                    .map(|(value, weight)| value * weight)
                    .sum()
            })
        })
        .collect();
    NumericTensor::new(vec![1, normalized.dim(1), 24], data)
}

#[test]
fn voicechat_backbone_composition_matches_ordinary_prefill_cached_and_bounded_execution() {
    let fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::BACKBONE_REFERENCE_JSON).unwrap();
    let config = VoiceChatConfig::from_json(include_str!(
        "../../src/voicechat/fixtures/official-inference-config.json"
    ))
    .unwrap();
    let fusion = ChannelFusion::from_config(&config.channels).unwrap();
    let context = NumericContext::default();
    let mut ordinary = ResidentRuntime::new(model(&fixture, &context), &context).unwrap();
    let mut normalized_reference =
        ResidentRuntime::new(model(&fixture, &context), &context).unwrap();
    let mut resident = ResidentRuntime::new(model(&fixture, &context), &context).unwrap();
    let mut bounded =
        LayerwiseRuntime::new(model(&fixture, &context), RebuildingUnitPolicy::default());
    let mut ordinary_state = state();
    let mut normalized_state = state();
    let mut resident_state = state();
    let mut bounded_state = state();
    let mut resident_function = function(&fixture, &context);
    let mut bounded_function = super::voicechat_function_head_residency::bounded(&fixture);
    let input = InputState::<NumericTensor>::new(
        &config.perception.preprocessor,
        FrontendOptions::default(),
    )
    .unwrap();
    let speech_fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::TTS_TRANSITION_REFERENCE_JSON).unwrap();
    let tts = super::voicechat_tts_generator::model(&speech_fixture, &context);
    let tts_caches = [
        super::voicechat_transformer::CausalCache(NumericCache::new(Some(3))),
        super::voicechat_transformer::CausalCache(NumericCache::new(None)),
    ];

    let embedding = tensor(&fixture["parameters"]["stt_model.embed_tokens.weight"]);
    let prompt_ids = [1, 5, 7, 2];
    let prompt_audio = prompt_ids
        .iter()
        .flat_map(|&id| embedding.data[id * 4..id * 4 + 4].iter().copied())
        .collect::<Vec<_>>();
    let cases = [
        (vec![1, 5, 7, 2], vec![12; 4], vec![12; 4], prompt_audio),
        (vec![3], vec![3], vec![9], vec![0.12, -0.31, 0.08, 0.42]),
        (vec![6], vec![6], vec![2], vec![-0.24, 0.17, 0.35, -0.09]),
    ];
    for (step, (ids, agent, function, audio)) in cases.iter().enumerate() {
        let tokens = NumericTensor::token_ids(ids);
        let embedded = fused(&fixture, audio, agent, function);
        let expected_agent = ordinary
            .forward(
                nemotron_h::EmbeddedInput::target_embedded(
                    &tokens,
                    &embedded,
                    None,
                    nemotron_h::TargetReadout::Logits,
                ),
                &mut ordinary_state,
                &context,
            )
            .unwrap();
        let expected_hidden = normalized_reference
            .forward(
                nemotron_h::EmbeddedInput::target_embedded(
                    &tokens,
                    &embedded,
                    None,
                    nemotron_h::TargetReadout::NormalizedHidden,
                ),
                &mut normalized_state,
                &context,
            )
            .unwrap();
        let expected_function = expected_function(&fixture, &expected_hidden);
        let audio = NumericTensor::new(vec![1, agent.len() as i32, 4], audio.clone());
        let agent =
            NumericTensor::from_i32_slice(agent, &[1, agent.len() as i32], &context).unwrap();
        let function =
            NumericTensor::from_i32_slice(function, &[1, function.len() as i32], &context).unwrap();
        let request = || {
            if step == 0 {
                BackboneRequest::Prompt {
                    tokens: &[5, 7],
                    policy: PromptPolicy {
                        repeat: 1,
                        max_tokens: 4,
                    },
                }
            } else {
                BackboneRequest::Audio {
                    projected_audio: &audio,
                    agent_tokens: &agent,
                    function_tokens: &function,
                }
            }
        };
        let actual = execute(
            &mut OrdinaryBackbone::new(&mut resident),
            &mut resident_function,
            &fusion,
            request(),
            &mut resident_state,
            &context,
        )
        .unwrap();
        let streamed = execute(
            &mut OrdinaryBackbone::new(&mut bounded),
            &mut bounded_function,
            &fusion,
            request(),
            &mut bounded_state,
            &context,
        )
        .unwrap();
        for output in [&actual, &streamed] {
            assert_tensor_close(
                &output.normalized,
                &expected_hidden,
                "single normalized hidden",
            );
            assert_tensor_close(
                &output.logits.agent,
                &expected_agent,
                "ordinary agent logits",
            );
            assert_tensor_close(
                &output.logits.function,
                &expected_function,
                "independent function head",
            );
        }
        assert_state_exact(
            &resident_state,
            &ordinary_state,
            3,
            "one resident backbone pass",
        );
        assert_state_exact(
            &bounded_state,
            &ordinary_state,
            3,
            "one bounded backbone pass",
        );
        let observed = observe::<NumericBackend, _, _, _, _>(
            &input,
            &OrdinaryBackbone::new(&mut bounded),
            &mut bounded_state,
            &tts.transformer,
            &tts_caches[..],
            None::<&eredu_architectures::voicechat::speech::SpeechState<NumericTensor>>,
        )
        .unwrap();
        assert_eq!(observed.backbone_positions, 4 + step as u64);
        assert_eq!(observed.audio_frames, 0);
        assert_eq!(observed.tts_conditional_positions, 0);
    }
    let mut ahead = tts_caches.clone();
    for cache in &mut ahead {
        cache.0.offset = 1;
    }
    assert!(matches!(
        observe::<NumericBackend, _, _, _, _>(
            &input,
            &OrdinaryBackbone::new(&mut bounded),
            &mut bounded_state,
            &tts.transformer,
            &ahead[..],
            None::<&eredu_architectures::voicechat::speech::SpeechState<NumericTensor>>
        ),
        Err(ProgressError::Inconsistent(_))
    ));
    assert_eq!(
        bounded.policy().events,
        expected_stream_events(&[(0, 0), (0, 1), (0, 2)]).repeat(3),
        "both heads use exactly one bounded block traversal per invocation",
    );
    assert_eq!(
        bounded_function.runtime().policy().inner.events,
        expected_stream_events(&[(0, 0)]).repeat(3),
        "each shared normalized result leases its independent function head once",
    );
    assert_eq!(
        context.embedding_forward_calls.get(),
        12,
        "prompt and each audio step look up feedback twice, without target re-embedding",
    );
}

#[test]
fn voicechat_backbone_composition_tp2_matches_complete_cached_invocations() {
    let fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::BACKBONE_REFERENCE_JSON).unwrap();
    let config = VoiceChatConfig::from_json(include_str!(
        "../../src/voicechat/fixtures/official-inference-config.json"
    ))
    .unwrap();
    let fusion = ChannelFusion::from_config(&config.channels).unwrap();
    let context = NumericContext::default();
    let architecture = model(&fixture, &context);
    let mut ordinary = ResidentRuntime::new(architecture, &context).unwrap();
    let mut ordinary_state = state();
    let mut ordinary_head = function(&fixture, &context);
    let audio = NumericTensor::new(vec![1, 1, 4], vec![0.12, -0.31, 0.08, 0.42]);
    let feedback = [(3, 9), (6, 2)];
    let expected = (0..3usize)
        .map(|step| {
            let (a, f) = feedback[step.saturating_sub(1)];
            let agent = NumericTensor::token_ids(&[a]);
            let function = NumericTensor::token_ids(&[f]);
            let request = if step == 0 {
                BackboneRequest::Prompt {
                    tokens: &[5, 7],
                    policy: PromptPolicy {
                        repeat: 1,
                        max_tokens: 4,
                    },
                }
            } else {
                BackboneRequest::Audio {
                    projected_audio: &audio,
                    agent_tokens: &agent,
                    function_tokens: &function,
                }
            };
            execute(
                &mut OrdinaryBackbone::new(&mut ordinary),
                &mut ordinary_head,
                &fusion,
                request,
                &mut ordinary_state,
                &context,
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let group = NumericParallelGroup::new(2);
    let outputs = std::thread::scope(|scope| {
        let handles = (0..2)
            .map(|rank| {
                let group = Arc::clone(&group);
                let fixture = &fixture;
                let audio = &audio;
                scope.spawn(move || {
                    let (context, mut runtime, mut head, mut state) =
                        parallel_components(fixture, rank);
                    let parallel = NumericParallelContext::new(rank, group);
                    let outputs = (0..3usize)
                        .map(|step| {
                            let (a, f) = feedback[step.saturating_sub(1)];
                            let agent = NumericTensor::token_ids(&[a]);
                            let function = NumericTensor::token_ids(&[f]);
                            let request = if step == 0 {
                                BackboneRequest::Prompt {
                                    tokens: &[5, 7],
                                    policy: PromptPolicy {
                                        repeat: 1,
                                        max_tokens: 4,
                                    },
                                }
                            } else {
                                BackboneRequest::Audio {
                                    projected_audio: audio,
                                    agent_tokens: &agent,
                                    function_tokens: &function,
                                }
                            };
                            let output = execute(
                                &mut ParallelBackbone::new(&mut runtime, &parallel, None),
                                &mut ParallelFunctionHead::new(&mut head, &parallel),
                                &fusion,
                                request,
                                &mut state,
                                &context,
                            )
                            .unwrap();
                            assert_eq!(state.layer(0).unwrap().position(), (4 + step) as i32);
                            assert_eq!(state.layer(1).unwrap().position(), (4 + step) as i32);
                            assert_eq!(state.layer(2).unwrap().position(), 0);
                            output
                        })
                        .collect::<Vec<_>>();
                    assert_eq!(
                        runtime.policy().events,
                        expected_stream_events(&[(0, 0), (0, 1), (0, 2)]).repeat(3)
                    );
                    outputs
                })
            })
            .collect::<Vec<_>>();
        handles
            .into_iter()
            .map(|h| h.join().unwrap())
            .collect::<Vec<_>>()
    });
    for rank in outputs {
        for (actual, expected) in rank.iter().zip(&expected) {
            assert_tensor_close(
                &actual.normalized,
                &expected.normalized,
                "TP2 shared normalized hidden",
            );
            assert_tensor_close(
                &actual.logits.agent,
                &expected.logits.agent,
                "TP2 agent head",
            );
            assert_tensor_close(
                &actual.logits.function,
                &expected.logits.function,
                "TP2 function head",
            );
        }
    }
}
