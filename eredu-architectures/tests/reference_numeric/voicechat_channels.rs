//! Independent fusion and dual-head fixtures through ordinary and TP mechanisms.
use super::*;
use eredu_architectures::voicechat::function_head_layered::FunctionHeadExecution;
use eredu_architectures::voicechat::{
    channels::{ChannelError, ChannelFusion, FunctionHead, PromptPolicy},
    config::VoiceChatConfig,
};
use eredu_nn::LinearFormatSpec;
use serde_json::Value;

fn tensor(value: &Value) -> NumericTensor {
    NumericTensor::new(
        value["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_i64().unwrap() as i32)
            .collect::<Vec<_>>(),
        value["values"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap() as f32)
            .collect(),
    )
}
fn ids(value: &Value, context: &NumericContext) -> NumericTensor {
    let rows = value.as_array().unwrap();
    let width = rows[0].as_array().unwrap().len();
    let values = rows
        .iter()
        .flat_map(|row| row.as_array().unwrap())
        .map(|x| x.as_i64().unwrap() as i32)
        .collect::<Vec<_>>();
    NumericTensor::from_i32_slice(&values, &[rows.len() as i32, width as i32], context).unwrap()
}
struct Bind<'v> {
    parameters: &'v Value,
    range: Option<std::ops::Range<usize>>,
}
impl<'a> ParameterVisitorMut<'a, NumericTensor> for Bind<'_> {
    fn visit_mut(&mut self, metadata: ParameterMetadata, target: &'a mut NumericTensor) {
        let full = tensor(&self.parameters[metadata.id.as_str()]);
        let replacement = match &self.range {
            Some(range) if full.shape().len() == 2 => full.axis_slice(0, range.start, range.end),
            _ => full,
        };
        assert_eq!(target.shape(), replacement.shape());
        *target = replacement;
    }
}
fn specification(fixture: &Value) -> decoder::StaticModuleSpec {
    decoder::StaticModuleSpec {
        normalization_groups: None,
        embedding_weight: "stt_model.embed_tokens.weight".into(),
        normalization_weight: "stt_model.llm.norm_f.weight".into(),
        head_weight: "stt_model.lm_head.weight".into(),
        vocabulary: fixture["geometry"]["vocab_size"].as_i64().unwrap() as i32,
        hidden_size: fixture["geometry"]["width"].as_i64().unwrap() as i32,
        normalization_epsilon: fixture["geometry"]["norm_epsilon"].as_f64().unwrap() as f32,
        normalization_offset: 0.0,
        embedding_quantization: None,
        head_format: eredu_checkpoint::LinearFormat::Dense,
        tied_head: false,
    }
}
fn compare(actual: &NumericTensor, expected: &Value, tolerance: &Value) {
    let expected = tensor(expected);
    assert_eq!(actual.shape(), expected.shape());
    let absolute = tolerance["absolute"].as_f64().unwrap() as f32;
    let relative = tolerance["relative"].as_f64().unwrap() as f32;
    for (actual, expected) in actual.data.iter().zip(expected.data) {
        assert!(
            (actual - expected).abs() <= absolute + relative * expected.abs(),
            "{actual} versus NVIDIA {expected}"
        );
    }
}

#[test]
fn voicechat_channels_match_nvidia_fusion_dual_heads_prompts_and_tp2() {
    let fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::BACKBONE_REFERENCE_JSON).unwrap();
    let configuration = VoiceChatConfig::from_json(include_str!(
        "../../src/voicechat/fixtures/official-inference-config.json"
    ))
    .unwrap();
    let fusion = ChannelFusion::from_config(&configuration.channels).unwrap();
    let context = NumericContext::default();
    let spec = specification(&fixture);
    let mut modules =
        decoder::StaticModules::<NumericBackend>::from_spec(spec.clone(), &context).unwrap();
    let dense = || LinearFormatSpec::unscaled(eredu_checkpoint::LinearFormat::Dense).unwrap();
    let mut function =
        FunctionHead::<NumericBackend>::new(spec.hidden_size, spec.vocabulary, dense(), &context)
            .unwrap();
    modules.visit_parameters_mut(&mut Bind {
        parameters: &fixture["parameters"],
        range: None,
    });
    function.visit_parameters_mut(&mut Bind {
        parameters: &fixture["parameters"],
        range: None,
    });
    let case = &fixture["f32"];
    let agent = ids(&case["agent_tokens"], &context);
    let function_tokens = ids(&case["function_tokens"], &context);
    let audio = tensor(&case["audio_input_f32"]);
    let fused = fusion
        .fuse(&mut modules, &audio, &agent, &function_tokens, &context)
        .unwrap();
    compare(&fused, &case["fused_input"], &fixture["tolerance"]);
    let normalized = modules
        .boundary
        .norm
        .forward(&tensor(&case["residual_hidden"]), &context)
        .unwrap();
    compare(
        &normalized,
        &case["normalized_hidden"],
        &fixture["tolerance"],
    );
    let logits = function
        .project_channels(&normalized, &mut modules, &context)
        .unwrap();
    compare(&logits.agent, &case["text_logits"], &fixture["tolerance"]);
    compare(
        &logits.function,
        &case["function_logits"],
        &fixture["tolerance"],
    );
    assert_eq!(
        logits.greedy_function(&context).unwrap().data,
        ids(&case["function_argmax"], &context).data
    );
    for prompt in fixture["prompts"].as_array().unwrap() {
        let repeat = prompt["repeat_n"].as_u64().unwrap() as usize;
        let prepared = fusion
            .prefill(
                &mut modules,
                &[5, 7],
                PromptPolicy {
                    repeat,
                    max_tokens: 8,
                },
                &context,
            )
            .unwrap();
        compare(
            &prepared.embedded,
            &prompt["fused_input"],
            &fixture["tolerance"],
        );
        assert_eq!(
            prepared.tokens.to_i32_vec(&context).unwrap(),
            prompt["prompt_tokens"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_i64().unwrap() as i32)
                .collect::<Vec<_>>()
        );
    }
    let lookups = context.embedding_forward_calls.get();
    assert!(matches!(
        fusion.prefill(
            &mut modules,
            &[5, 7],
            PromptPolicy {
                repeat: 2,
                max_tokens: 7
            },
            &context
        ),
        Err(ChannelError::PromptCapacity {
            required: 8,
            capacity: 7
        })
    ));
    assert_eq!(
        context.embedding_forward_calls.get(),
        lookups,
        "prompt capacity rejects before embedding dispatch"
    );

    let group = NumericParallelGroup::new(2);
    let outputs = std::thread::scope(|scope| {
        let handles = (0..2)
            .map(|rank| {
                let group = Arc::clone(&group);
                let fixture = &fixture;
                let spec = spec.clone();
                scope.spawn(move || {
                    let context = NumericContext::default();
                    let parallel = NumericParallelContext::new(rank, group);
                    let range = VocabularyParallelRange {
                        global_vocabulary: spec.vocabulary as usize,
                        local: rank * 12..(rank + 1) * 12,
                    };
                    let mut modules = decoder::StaticModules::<NumericBackend>::from_parallel_spec(
                        spec.clone(),
                        range.clone(),
                        Some(range.clone()),
                        &context,
                    )
                    .unwrap();
                    let mut function = FunctionHead::<NumericBackend>::new_parallel(
                        spec.hidden_size,
                        spec.vocabulary,
                        dense(),
                        range.clone(),
                        &context,
                    )
                    .unwrap();
                    modules.visit_parameters_mut(&mut Bind {
                        parameters: &fixture["parameters"],
                        range: Some(range.local.clone()),
                    });
                    function.visit_parameters_mut(&mut Bind {
                        parameters: &fixture["parameters"],
                        range: Some(range.local),
                    });
                    let case = &fixture["f32"];
                    let fused = fusion
                        .fuse_parallel(
                            &mut modules,
                            &tensor(&case["audio_input_f32"]),
                            &ids(&case["agent_tokens"], &context),
                            &ids(&case["function_tokens"], &context),
                            &parallel,
                            &context,
                        )
                        .unwrap();
                    let normalized = modules
                        .boundary
                        .norm
                        .forward(&tensor(&case["residual_hidden"]), &context)
                        .unwrap();
                    let logits = function
                        .project_channels_parallel(&normalized, &mut modules, &parallel, &context)
                        .unwrap();
                    let decisions = logits.greedy_function(&context).unwrap();
                    let prompts = fixture["prompts"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|prompt| {
                            fusion
                                .prefill_parallel(
                                    &mut modules,
                                    &[5, 7],
                                    PromptPolicy {
                                        repeat: prompt["repeat_n"].as_u64().unwrap() as usize,
                                        max_tokens: 8,
                                    },
                                    &parallel,
                                    &context,
                                )
                                .unwrap()
                                .embedded
                        })
                        .collect::<Vec<_>>();
                    (fused, logits, decisions, prompts)
                })
            })
            .collect::<Vec<_>>();
        handles
            .into_iter()
            .map(|h| h.join().unwrap())
            .collect::<Vec<_>>()
    });
    for (fused, logits, decisions, prompts) in outputs {
        compare(&fused, &case["fused_input"], &fixture["tolerance"]);
        compare(&logits.agent, &case["text_logits"], &fixture["tolerance"]);
        compare(
            &logits.function,
            &case["function_logits"],
            &fixture["tolerance"],
        );
        assert_eq!(decisions.data, ids(&case["function_argmax"], &context).data);
        for (actual, prompt) in prompts.iter().zip(fixture["prompts"].as_array().unwrap()) {
            compare(actual, &prompt["fused_input"], &fixture["tolerance"]);
        }
    }
}

#[test]
fn voicechat_channel_ingress_guarded_failures_stop_before_peer_lookup_and_retry() {
    use super::voicechat_rnnt_coordination::coordinator_pair;
    use eredu_architectures::voicechat::backbone_execution::{
        BackboneExecution, BackboneInput, BackboneRequest, ParallelBackbone,
    };
    use eredu_core::BackendFailure;
    use eredu_runtime::replicated_execution::ReplicatedCoordinator;

    let (coordinators, timeouts) = coordinator_pair();
    let group = NumericParallelGroup::new(2);
    std::thread::scope(|scope| {
        let workers = coordinators
            .into_iter()
            .enumerate()
            .map(|(rank, mut coordinator)| {
                let group = group.clone();
                scope.spawn(move || {
                    let fixture: Value =
                        serde_json::from_str(eredu_evaluation::voicechat::BACKBONE_REFERENCE_JSON)
                            .unwrap();
                    let configuration = VoiceChatConfig::from_json(include_str!(
                        "../../src/voicechat/fixtures/official-inference-config.json"
                    ))
                    .unwrap();
                    let fusion = ChannelFusion::from_config(&configuration.channels).unwrap();
                    let context = NumericContext::default();
                    let parallel = NumericParallelContext::new(rank, group);
                    let spec = specification(&fixture);
                    let range = VocabularyParallelRange {
                        global_vocabulary: spec.vocabulary as usize,
                        local: rank * 12..(rank + 1) * 12,
                    };
                    let mut modules = decoder::StaticModules::<NumericBackend>::from_parallel_spec(
                        spec,
                        range.clone(),
                        Some(range.clone()),
                        &context,
                    )
                    .unwrap();
                    modules.visit_parameters_mut(&mut Bind {
                        parameters: &fixture["parameters"],
                        range: Some(range.local),
                    });
                    let case = &fixture["f32"];
                    let audio = tensor(&case["audio_input_f32"]);
                    let agent = ids(&case["agent_tokens"], &context);
                    let function = ids(&case["function_tokens"], &context);
                    let policy = PromptPolicy {
                        repeat: 1,
                        max_tokens: 8,
                    };
                    let fail_after_lookup = Cell::new(None);
                    let mut agree = |descriptor, success, context: &NumericContext| {
                        let injected = rank == 0
                            && fail_after_lookup.get()
                                == Some(context.embedding_forward_calls.get());
                        let agreed =
                            coordinator.agree(descriptor, success && !injected, context)?;
                        if injected {
                            fail_after_lookup.set(None);
                            Err(ChannelError::Backend(BackendFailure::from_error(
                                std::io::Error::other(
                                    "injected agreement failure after first embedding lookup",
                                ),
                            )))
                        } else {
                            Ok(agreed)
                        }
                    };
                    let request = if rank == 0 {
                        BackboneRequest::Prompt {
                            tokens: &[5, 7],
                            policy,
                        }
                    } else {
                        BackboneRequest::Audio {
                            projected_audio: &audio,
                            agent_tokens: &agent,
                            function_tokens: &function,
                        }
                    };
                    assert!(BackboneInput::prepare_parallel_guarded(
                        &fusion,
                        &mut modules,
                        request,
                        &parallel,
                        &context,
                        &mut agree
                    )
                    .is_err());
                    assert_eq!(context.embedding_forward_calls.get(), 0);
                    assert!(parallel.trace().is_empty());

                    let request = BackboneRequest::Prompt {
                        tokens: &[5, 7],
                        policy: PromptPolicy {
                            repeat: 1,
                            max_tokens: if rank == 0 { 3 } else { 8 },
                        },
                    };
                    let error = BackboneInput::prepare_parallel_guarded(
                        &fusion,
                        &mut modules,
                        request,
                        &parallel,
                        &context,
                        &mut agree,
                    )
                    .unwrap_err();
                    if rank == 0 {
                        assert!(matches!(error, ChannelError::PromptCapacity { .. }));
                    }
                    assert_eq!(context.embedding_forward_calls.get(), 0);
                    assert!(parallel.trace().is_empty());

                    // Individually valid prompts still need identical token content.
                    let request = BackboneRequest::Prompt {
                        tokens: if rank == 0 { &[5, 7] } else { &[5, 8] },
                        policy,
                    };
                    assert!(BackboneInput::prepare_parallel_guarded(
                        &fusion,
                        &mut modules,
                        request,
                        &parallel,
                        &context,
                        &mut agree
                    )
                    .is_err());
                    assert_eq!(context.embedding_forward_calls.get(), 0);
                    assert!(parallel.trace().is_empty());

                    let malformed = NumericTensor::from_i32_slice(&[12], &[1], &context).unwrap();
                    let request = BackboneRequest::Audio {
                        projected_audio: &audio,
                        agent_tokens: &agent,
                        function_tokens: if rank == 0 { &malformed } else { &function },
                    };
                    let error = BackboneInput::prepare_parallel_guarded(
                        &fusion,
                        &mut modules,
                        request,
                        &parallel,
                        &context,
                        &mut agree,
                    )
                    .unwrap_err();
                    if rank == 0 {
                        assert!(matches!(error, ChannelError::Shape(_)));
                    }
                    assert_eq!(context.embedding_forward_calls.get(), 0);
                    assert!(parallel.trace().is_empty());

                    // The mechanism reports a failed agreement immediately after the
                    // completed agent lookup; neither rank may enter function lookup.
                    // This injects transport agreement failure, not a numerical fault.
                    fail_after_lookup.set(Some(1));
                    let request = BackboneRequest::Audio {
                        projected_audio: &audio,
                        agent_tokens: &agent,
                        function_tokens: &function,
                    };
                    let error = BackboneInput::prepare_parallel_guarded(
                        &fusion,
                        &mut modules,
                        request,
                        &parallel,
                        &context,
                        &mut agree,
                    )
                    .unwrap_err();
                    if rank == 0 {
                        let mut source: &(dyn std::error::Error + 'static) = &error;
                        let mut original = false;
                        loop {
                            original |=
                                source
                                    .downcast_ref::<std::io::Error>()
                                    .is_some_and(|error| {
                                        error.to_string().contains(
                                        "injected agreement failure after first embedding lookup",
                                    )
                                    });
                            match source.source() {
                                Some(next) => source = next,
                                None => break,
                            }
                        }
                        assert!(original, "agreement failure retains its original source");
                    }
                    assert_eq!(context.embedding_forward_calls.get(), 1);
                    assert_eq!(parallel.trace().len(), 1);

                    let request = BackboneRequest::Audio {
                        projected_audio: &audio,
                        agent_tokens: &agent,
                        function_tokens: &function,
                    };
                    let result = BackboneInput::prepare_parallel_guarded(
                        &fusion,
                        &mut modules,
                        request,
                        &parallel,
                        &context,
                        &mut agree,
                    )
                    .unwrap();
                    compare(
                        result.embedded(),
                        &case["fused_input"],
                        &fixture["tolerance"],
                    );
                    assert_eq!(
                        result.tokens().to_i32_vec(&context).unwrap(),
                        agent.to_i32_vec(&context).unwrap()
                    );
                    assert_eq!(context.embedding_forward_calls.get(), 3);
                    assert_eq!(parallel.trace().len(), 3);

                    for prompt in fixture["prompts"].as_array().unwrap() {
                        let request = BackboneRequest::Prompt {
                            tokens: &[5, 7],
                            policy: PromptPolicy {
                                repeat: prompt["repeat_n"].as_u64().unwrap() as usize,
                                max_tokens: 8,
                            },
                        };
                        let result = BackboneInput::prepare_parallel_guarded(
                            &fusion,
                            &mut modules,
                            request,
                            &parallel,
                            &context,
                            &mut agree,
                        )
                        .unwrap();
                        compare(
                            result.embedded(),
                            &prompt["fused_input"],
                            &fixture["tolerance"],
                        );
                        assert_eq!(
                            result.tokens().to_i32_vec(&context).unwrap(),
                            prompt["prompt_tokens"]
                                .as_array()
                                .unwrap()
                                .iter()
                                .map(|value| value.as_i64().unwrap() as i32)
                                .collect::<Vec<_>>()
                        );
                    }
                    // Selected runtime composition retains the same ingress
                    // guard before any bounded layer is acquired or advanced.
                    let (selected_context, mut runtime, _, mut state) =
                        super::voicechat_backbone_execution::parallel_components(&fixture, rank);
                    let before = parallel.trace().len();
                    let mut selected = ParallelBackbone::<NumericBackend, _>::new(
                        &mut runtime,
                        &parallel,
                        Some(&mut agree),
                    );
                    let request = BackboneRequest::Audio {
                        projected_audio: &audio,
                        agent_tokens: &agent,
                        function_tokens: if rank == 0 { &malformed } else { &function },
                    };
                    assert!(selected
                        .execute_hidden(&fusion, request, &mut state, &selected_context)
                        .is_err());
                    assert_eq!(selected.position(&mut state).unwrap(), 0);
                    assert_eq!(selected_context.embedding_forward_calls.get(), 0);
                    assert_eq!(parallel.trace().len(), before);
                    assert!(runtime.policy().events.is_empty());
                })
            })
            .collect::<Vec<_>>();
        for worker in workers {
            worker.join().unwrap();
        }
    });
    assert_eq!(timeouts.load(Ordering::SeqCst), 0);
}
