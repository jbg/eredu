use super::*;

use crate::api::{LoadedModel, PreparedChatInput, PreparedChatMtpBatchRequest, SpeculativeDraft};
use crate::backend::mlx::architectures::gemma4::model as gemma4;
use crate::{
    backend::mlx::architectures::{
        gpt_oss::model as gpt_oss,
        llama::model as llama,
        qwen::dense as dense_qwen,
        qwen::{hybrid::qwen3_5, vl::model as qwen3_vl},
    },
    backend::mlx::error::Error,
    backend::mlx::runtime::checkpoint::quantization::{
        AffineQuantization, CheckpointQuantizationOptions, WeightQuantization,
    },
    backend::mlx::runtime::execution::inspection::ActivationRecorder,
    backend::mlx::runtime::generation::sampler::{ConstrainedSampler, DefaultSampler},
    backend::mlx::runtime::media::input,
    backend::mlx::{
        resolve_model_config, validate_gguf_quantization_source, Model, ModelLoadOptions,
        ResolvedModelConfig,
    },
    core::generation::MtpSchedulerOptions,
    core::{ModelKind, SpeculativeExecutionTopology},
    runtime::chat::constraints::{ConstraintController, ConstraintError},
    runtime::chat::PreparedChat,
};
use eredu_gguf::{GgmlType, MetadataValue as GgufWriterMetadata, TensorInput, Writer};
use safemlx::{
    argmax_axis,
    module::ModuleParameters,
    ops::{indexing::TryIndexOp, zeros_dtype, GgufMetadataArray, GgufMetadataValue},
    Array, Device, DeviceType, ExecutionContext, Stream,
};

const GEMMA4_LARGE_FIXTURE: &str =
    include_str!("../../../tests/fixtures/chat_templates/gemma-4-26b-a4b-it-4d7ae498.jinja");

fn constrained_sampler<S>(
    policy: S,
    plan: &crate::runtime::chat::GenerationRuntimePlan,
) -> Result<ConstrainedSampler<S, ConstraintController>, ConstraintError> {
    Ok(ConstrainedSampler::new(
        policy,
        ConstraintController::from_generation_plan(plan)?,
    ))
}

fn load_test_model(
    path: impl AsRef<std::path::Path>,
    options: ModelLoadOptions,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<crate::PreparedModel<crate::backend::mlx::MlxModel>, crate::ModelLoadError<Error>> {
    let backend = crate::backend::mlx::MlxBackend::new(stream, weights_stream);
    crate::load_model(&backend, path, options)
}

#[test]
#[ignore = "requires MLX runtime execution and SAFEMLX_INSPECTION_MODEL_DIR"]
fn observer_forward_reports_attention_and_residual_hooks() {
    let model_dir = std::env::var("SAFEMLX_INSPECTION_MODEL_DIR")
        .expect("set SAFEMLX_INSPECTION_MODEL_DIR to a local model directory");
    let ctx = safemlx::ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
    let weights_ctx =
        safemlx::ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let mut model = LoadedModel::load(
        crate::backend::mlx::MlxBackend::new(ctx.stream(), weights_ctx.stream()),
        model_dir,
        ModelLoadOptions::default(),
    )
    .unwrap();
    let ids = model.encode("hello", true).unwrap();
    let input = safemlx::Array::from(ids.as_slice())
        .try_index_device(safemlx::ops::indexing::NewAxis, ctx.stream())
        .unwrap();
    let mut recorder = ActivationRecorder::new();
    let parts = [crate::backend::mlx::runtime::media::input::InputPart::text_token_ids(&input)];
    let input = crate::backend::mlx::MlxModelInput::from(
        crate::backend::mlx::runtime::media::input::ModelInput::new(&parts),
    );
    let (backend, session) = model.runtime_mut().parts_mut();
    session
        .submit_prefill_with_observer(backend, input, &mut recorder)
        .unwrap()
        .wait()
        .unwrap();

    let names = recorder
        .activations()
        .iter()
        .map(|activation| activation.name.as_str())
        .collect::<Vec<_>>();
    assert!(
        names.iter().any(|name| name.ends_with(".attention_probs")),
        "{names:?}"
    );
    assert!(
        names
            .iter()
            .any(|name| name.ends_with(".residual_delta_attention")),
        "{names:?}"
    );
    assert!(
        names
            .iter()
            .any(|name| name.ends_with(".residual_delta_mlp"))
            || names
                .iter()
                .any(|name| name.ends_with(".residual_delta_moe")),
        "{names:?}"
    );
}

#[test]
fn chat_preparation_contracts_are_public_and_default_conservatively() {
    let request = ChatTemplateRequest::default();
    assert_eq!(request.tool_choice, ToolChoice::Auto);
    assert_eq!(
        request.parallel_tool_calls,
        ParallelToolCallPolicy::Disabled
    );
    let _: fn(
        &mut LoadedModel<crate::backend::mlx::MlxBackend<'static>>,
        ChatTemplateRequest,
    ) -> Result<PreparedChat, TextModelError> = LoadedModel::prepare_chat;
}

#[test]
fn prepared_chat_input_keeps_its_semantic_owner_with_an_opaque_backend_prompt() {
    let prepared = PreparedChat {
        rendered_prompt: "prompt must not be prefetched".into(),
        generation_prompt: String::new(),
        template_identity: ChatTemplateIdentity::Single,
        format_profile_identity: None,
        native_tool_support: NativeToolSupport::Unsupported {
            reason: "synthetic unsupported profile".into(),
        },
        semantic_support: crate::runtime::chat::SemanticSupport::Unsupported {
            reason: "synthetic unsupported profile".into(),
        },
        capabilities: crate::runtime::chat::ChatCapabilities {
            reasoning_parser: crate::runtime::chat::CapabilitySupport::Unsupported {
                reason: "synthetic unsupported profile".into(),
            },
            visible_text_parser: crate::runtime::chat::CapabilitySupport::Unsupported {
                reason: "synthetic unsupported profile".into(),
            },
            tool_output_parser: crate::runtime::chat::CapabilitySupport::Unsupported {
                reason: "synthetic unsupported profile".into(),
            },
            tool_input_rendering: crate::runtime::chat::CapabilitySupport::Unsupported {
                reason: "synthetic unsupported profile".into(),
            },
            mapping_tool_arguments: crate::runtime::chat::CapabilitySupport::Unsupported {
                reason: "synthetic unsupported profile".into(),
            },
            string_tool_arguments: crate::runtime::chat::CapabilitySupport::Unsupported {
                reason: "synthetic unsupported profile".into(),
            },
            constrained_tool_generation: crate::runtime::chat::CapabilitySupport::Unsupported {
                reason: "synthetic unsupported profile".into(),
            },
        },
        generation_runtime_plan: None,
        eos_token_ids: vec![2],
        preserved_structural_token_ids: Vec::new(),
        profile_stop_sequences: Vec::new(),
    };
    let input: PreparedChatInput<'_, crate::backend::mlx::MlxBackend<'static>> =
        PreparedChatInput::rendered_prompt(&prepared);
    assert!(std::ptr::eq(input.prepared_chat(), &prepared));
    assert!(input.backend_prompt().is_none());
    let model_input = crate::backend::mlx::runtime::media::prepared_model_input(vec![
        crate::backend::mlx::runtime::media::PreparedInputPart::text_token_ids(&[7]),
    ])
    .unwrap();
    let model_input =
        model_input.with_model_input(|input| crate::backend::mlx::MlxModelInput::from(input));
    let input: PreparedChatInput<'_, crate::backend::mlx::MlxBackend<'static>> =
        PreparedChatInput::prepared_backend_input(&prepared, model_input);
    assert!(std::ptr::eq(input.prepared_chat(), &prepared));
    assert!(input.backend_prompt().is_some());
}

#[test]
fn prepared_chat_embedded_mtp_batch_dispatches_qwen_without_a_drafter() {
    let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = context.stream();
    let compiler = Ok(ConstraintCompiler::synthetic_for_tests());
    let mut tokenizer = gemma4_chat_tokenizer(23);
    let mut prepared = prepare_chat_from_parts(
        &mut tokenizer,
        ModelChatTemplate::Single(GEMMA4_EDGE_FIXTURE.into()),
        "prepared-embedded-batch-test",
        &[2],
        Some(&compiler),
        ChatTemplateRequest {
            messages: vec![json!({"role": "user", "content": "lookup"})],
            tools: vec![production_tool("lookup")],
            tool_choice: ToolChoice::Required,
            add_generation_prompt: true,
            ..ChatTemplateRequest::default()
        },
    )
    .unwrap();
    prepared.rendered_prompt.clear();
    let config = json!({
        "model_type": "qwen3_5_text",
        "vocab_size": 8,
        "hidden_size": 8,
        "num_hidden_layers": 1,
        "mtp_num_hidden_layers": 1,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "max_position_embeddings": 16,
        "intermediate_size": 16,
        "num_experts": 0,
        "tie_word_embeddings": true,
        "layer_types": ["full_attention"]
    });
    let args = qwen3_5::model_args_from_config_value(&config).unwrap();
    let qwen = qwen3_5::Model::new(args, None, None, None, stream).unwrap();
    let directory = temp_model_dir(&config.to_string());
    save_zero_checkpoint(&qwen, &directory, stream);
    let Model::Qwen35(qwen) =
        load_test_model(&directory, ModelLoadOptions::default(), stream, stream)
            .unwrap()
            .into_inner()
            .into_complete()
            .unwrap()
    else {
        panic!("expected canonical Qwen3.5 model");
    };
    let runtime = eredu_core::ModelRuntime::from_prepared(
        crate::backend::mlx::MlxBackend::new(stream, stream),
        eredu_core::PreparedModel::new(crate::backend::mlx::MlxModel::complete(Model::Qwen35(
            qwen,
        ))),
    )
    .unwrap();
    let mut model = LoadedModel::from_runtime(
        runtime,
        tokenizer,
        crate::api::LoadedTextModelConfig {
            chat_template: None,
            model_type: "qwen3_5_text".into(),
            model_id: "prepared-embedded-batch-test".into(),
            eos_token_ids: vec![2],
            checkpoint_generation_config: None,
        },
    );

    let capabilities = model.capabilities().unwrap();
    assert_eq!(capabilities.model_type, "qwen3_5_text");
    let prompt = <crate::backend::mlx::MlxBackend<'static> as eredu_core::TextGenerationBackend>::prepare_text_prompt(
        model.runtime().backend(),
        vec![1, 2],
    )
    .unwrap();
    let input = model.count_prepared_input(&prompt).unwrap();
    assert_eq!(input.text_tokens, 2);
    assert_eq!(input.model_positions, 2);
    assert_eq!(
        model
            .estimate_runtime_state(input, 2, 1)
            .unwrap()
            .assumptions
            .requested_positions,
        4
    );
    assert!(matches!(
        model
            .admit(
                eredu_core::AdmissionRequest {
                    input,
                    max_output_tokens: 2,
                    batch_size: 1,
                    safety_reserve_bytes: 0,
                    application_memory_budget_bytes: None,
                    require_complete_estimate: true,
                },
                None,
            )
            .unwrap(),
        eredu_core::AdmissionResult::Admitted(_)
    ));
    model.static_memory().unwrap();
    let session = model.runtime().session();
    assert!(!session
        .prompt_cache_architecture_fingerprint()
        .unwrap()
        .is_empty());
    let cache_layout = session.prompt_cache_layer_layout().unwrap();
    assert_eq!(
        session.prompt_cache_layer_prefix_offsets().unwrap().len(),
        cache_layout.len()
    );
    assert!(session.native_quantization_stats().is_none());

    let output = model
        .generate_prepared_chat_mtp_batch(PreparedChatMtpBatchRequest {
            drafting: SpeculativeDraft::Embedded,
            lanes: Vec::new(),
            scheduler: MtpSchedulerOptions::default(),
        })
        .unwrap();
    fs::remove_dir_all(directory).unwrap();

    assert!(output.requests.is_empty());
    assert_eq!(
        output.scheduler.execution_topology,
        SpeculativeExecutionTopology::Single
    );
    assert_eq!(output.scheduler.turns, 0);
}

#[test]
fn production_deepseek_tool_boundaries_are_constrained_and_stream_incrementally() {
    fn prepare(
        template: &str,
        tools: Vec<serde_json::Value>,
        tool_choice: ToolChoice,
        parallel_tool_calls: ParallelToolCallPolicy,
        preceding_tokens: usize,
    ) -> Result<PreparedChat, TextModelError> {
        prepare_chat_from_parts(
            &mut deepseek_chat_tokenizer(preceding_tokens),
            ModelChatTemplate::Single(template.into()),
            "architecture-metadata-must-not-select-deepseek",
            &[],
            Some(&Ok(ConstraintCompiler::synthetic_for_tests())),
            ChatTemplateRequest {
                messages: vec![json!({"role": "user", "content": "Use a tool."})],
                tools,
                tool_choice,
                parallel_tool_calls,
                enable_thinking: Some(false),
                add_generation_prompt: true,
                ..ChatTemplateRequest::default()
            },
        )
    }

    let one_v31 = concat!(
        "<｜tool▁calls▁begin｜>",
        "<｜tool▁call▁begin｜>lookup<｜tool▁sep｜>{\"value\":7}",
        "<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
    );
    let two_v31 = concat!(
        "<｜tool▁calls▁begin｜>",
        "<｜tool▁call▁begin｜>lookup<｜tool▁sep｜>{\"value\":7}<｜tool▁call▁end｜>",
        "<｜tool▁call▁begin｜>lookup<｜tool▁sep｜>{\"value\":8}<｜tool▁call▁end｜>",
        "<｜tool▁calls▁end｜>",
    );
    let three_v31 = concat!(
        "<｜tool▁calls▁begin｜>",
        "<｜tool▁call▁begin｜>lookup<｜tool▁sep｜>{\"value\":7}<｜tool▁call▁end｜>",
        "<｜tool▁call▁begin｜>lookup<｜tool▁sep｜>{\"value\":8}<｜tool▁call▁end｜>",
        "<｜tool▁call▁begin｜>lookup<｜tool▁sep｜>{\"value\":9}<｜tool▁call▁end｜>",
        "<｜tool▁calls▁end｜>",
    );
    let one_v3 = concat!(
        "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>lookup\n",
        "```json\n{\"value\":7}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
    );

    let required = prepare(
        DEEPSEEK_V31_TOOL_FIXTURE,
        vec![production_tool("lookup")],
        ToolChoice::Required,
        ParallelToolCallPolicy::Disabled,
        71,
    )
    .unwrap();
    let required_plan = required
        .tool_runtime_plan()
        .unwrap_or_else(|| panic!("registered DeepSeek V3.1 template must prepare native tools"));
    assert_eq!(required_plan.auto_activation_trigger(), None);
    assert!(plan_accepts(required_plan, one_v31));
    assert!(plan_accepts(
        required_plan,
        &format!("Need 東京 🦀{one_v31}")
    ));
    assert!(!plan_accepts(required_plan, two_v31));
    assert!(constrained_sampler(DefaultSampler, required_plan)
        .unwrap()
        .controller()
        .constraint_is_active());

    let parallel = prepare(
        DEEPSEEK_V31_TOOL_FIXTURE,
        vec![production_tool("lookup")],
        ToolChoice::Required,
        ParallelToolCallPolicy::Enabled {
            max_calls: std::num::NonZeroUsize::new(2),
        },
        81,
    )
    .unwrap();
    let parallel_plan = parallel
        .tool_runtime_plan()
        .unwrap_or_else(|| panic!("registered DeepSeek V3.1 template must prepare parallel tools"));
    assert!(plan_accepts(parallel_plan, two_v31));
    assert!(!plan_accepts(parallel_plan, three_v31));

    let auto = prepare(
        DEEPSEEK_V31_TOOL_FIXTURE,
        vec![production_tool("lookup")],
        ToolChoice::Auto,
        ParallelToolCallPolicy::Disabled,
        91,
    )
    .unwrap();
    let auto_plan = auto
        .tool_runtime_plan()
        .unwrap_or_else(|| panic!("registered DeepSeek V3.1 template must prepare Auto tools"));
    assert_eq!(
        auto_plan.auto_activation_trigger(),
        Some("<｜tool▁calls▁begin｜>")
    );
    assert!(!constrained_sampler(DefaultSampler, auto_plan)
        .unwrap()
        .controller()
        .constraint_is_active());
    assert!(plan_accepts(auto_plan, one_v31));

    let old_surface = prepare(
        DEEPSEEK_V3_TOOL_FIXTURE,
        vec![production_tool("lookup")],
        ToolChoice::Required,
        ParallelToolCallPolicy::Disabled,
        101,
    )
    .unwrap();
    let old_plan = old_surface
        .tool_runtime_plan()
        .unwrap_or_else(|| panic!("registered DeepSeek V3 template must prepare native tools"));
    assert!(plan_accepts(old_plan, one_v3));
    assert!(!plan_accepts(old_plan, one_v31));

    for malformed in [
            "<｜tool▁calls▁begin｜>",
            "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>",
            "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>lookup:{\"value\":7}<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
            "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>missing<｜tool▁sep｜>{\"value\":7}<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
            "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>lookup<｜tool▁sep｜>{\"value\":\"seven\"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
            "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>lookup<｜tool▁sep｜>[7]<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
            "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>lookup<｜tool▁sep｜>{\"value\":7,}<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
            "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>lookup<｜tool▁sep｜>{\"value\":7}<｜tool▁call▁end｜>",
            "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>lookup<｜tool▁sep｜>{\"value\":7}<｜tool▁call▁end｜> <｜tool▁calls▁end｜>",
        ] {
            assert!(!plan_accepts(parallel_plan, malformed), "{malformed:?}");
        }

    let city_tool = json!({
        "type": "function",
        "function": {
            "name": "city_lookup",
            "description": "Look up a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
                "additionalProperties": false
            }
        }
    });
    let unicode = prepare(
        DEEPSEEK_V31_TOOL_FIXTURE,
        vec![city_tool],
        ToolChoice::Required,
        ParallelToolCallPolicy::Disabled,
        111,
    )
    .unwrap();
    let unicode_plan = unicode.tool_runtime_plan().unwrap_or_else(|| {
        panic!("registered DeepSeek V3.1 template must prepare Unicode arguments")
    });
    assert!(plan_accepts(
        unicode_plan,
        concat!(
            "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>city_lookup<｜tool▁sep｜>",
            "{\"city\":\"東京 🦀\"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
        )
    ));

    for invalid_name in [
        "lookup.city".to_owned(),
        "herramienta_ñ".to_owned(),
        "x".repeat(65),
    ] {
        let error = prepare(
            DEEPSEEK_V31_TOOL_FIXTURE,
            vec![production_tool(&invalid_name)],
            ToolChoice::Required,
            ParallelToolCallPolicy::Disabled,
            121,
        )
        .unwrap_err();
        assert!(
            matches!(error, TextModelError::ToolConstraint(_)),
            "{invalid_name:?}"
        );
    }
    prepare(
        DEEPSEEK_V31_TOOL_FIXTURE,
        vec![production_tool(&"x".repeat(64))],
        ToolChoice::Required,
        ParallelToolCallPolicy::Disabled,
        131,
    )
    .unwrap();
    let too_many = (0..129)
        .map(|index| production_tool(&format!("tool_{index}")))
        .collect();
    assert!(matches!(
        prepare(
            DEEPSEEK_V31_TOOL_FIXTURE,
            too_many,
            ToolChoice::Required,
            ParallelToolCallPolicy::Disabled,
            141,
        ),
        Err(TextModelError::ToolConstraint(message)) if message.contains("at most 128 tools")
    ));
    assert!(matches!(
        prepare(
            DEEPSEEK_V31_TOOL_FIXTURE,
            vec![json!({"type": "function", "function": {"name": "broken"}})],
            ToolChoice::Required,
            ParallelToolCallPolicy::Disabled,
            151,
        ),
        Err(TextModelError::ToolConstraint(_))
    ));

    let streamed = format!("Need 東京 🦀{two_v31}<｜end▁of▁sentence｜>ignored");
    for split in (0..=streamed.len()).filter(|index| streamed.is_char_boundary(*index)) {
        let mut parser = parallel_plan.create_parser().unwrap();
        parser.push(&streamed[..split]).unwrap();
        parser.push(&streamed[split..]).unwrap();
        assert_eq!(
            tool_argument_events(parser.events()).concat(),
            "{\"value\":7}{\"value\":8}",
            "split {split}"
        );
        let visible = parser
            .events()
            .iter()
            .filter_map(|event| match event {
                SemanticEvent::TextDelta(text) => Some(text.as_str()),
                _ => None,
            })
            .collect::<String>();
        assert_eq!(visible, "Need 東京 🦀", "split {split}");
        assert_eq!(
            parser
                .events()
                .iter()
                .filter(|event| matches!(event, SemanticEvent::ToolCallEnd))
                .count(),
            2,
            "split {split}"
        );
        assert_eq!(
            parser.events().last(),
            Some(&SemanticEvent::Finished {
                reason: FinishReason::StopSequence,
            }),
            "split {split}"
        );
    }

    let mut incremental = parallel_plan.create_parser().unwrap();
    incremental
        .push(concat!(
            "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>",
            "lookup<｜tool▁sep｜>{"
        ))
        .unwrap();
    assert!(incremental
        .events()
        .contains(&SemanticEvent::ToolCallStart {
            index: 0,
            id: "call_0".into(),
            name: "lookup".into(),
        }));
    assert_eq!(tool_argument_events(incremental.events()), ["{"]);

    let mut incomplete = parallel_plan.create_parser().unwrap();
    incomplete
        .push(concat!(
            "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>",
            "lookup<｜tool▁sep｜>{\"value\":7}"
        ))
        .unwrap();
    incomplete.finish(FinishReason::MaxTokens).unwrap();
    assert!(!incomplete
        .events()
        .iter()
        .any(|event| matches!(event, SemanticEvent::ToolCallEnd)));
    assert_eq!(
        incomplete.events().last(),
        Some(&SemanticEvent::Finished {
            reason: FinishReason::MaxTokens,
        })
    );

    let mut malformed = parallel_plan.create_parser().unwrap();
    assert!(malformed
        .push(concat!(
            "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>",
            "lookup<｜tool▁sep｜>{\"value\":]}"
        ))
        .is_err());

    let mut caller_stop = parallel_plan
        .create_parser_with_stops(["CALLER_STOP"])
        .unwrap();
    caller_stop.push("VisibleCALLER_STOPmust not leak").unwrap();
    assert!(caller_stop
        .events()
        .contains(&SemanticEvent::TextDelta("Visible".into())));
    assert_eq!(
        caller_stop.events().last(),
        Some(&SemanticEvent::Finished {
            reason: FinishReason::StopSequence,
        })
    );

    let mut raw = Tokenizer::new(WordLevel::default());
    raw.add_special_tokens(
        [
            "<｜tool▁calls▁begin｜>",
            "<｜tool▁calls▁end｜>",
            "<｜tool▁call▁begin｜>",
            "<｜tool▁call▁end｜>",
            "<｜tool▁sep｜>",
        ]
        .map(|token| AddedToken::from(token, true).normalized(false)),
    )
    .unwrap();
    let mut missing_structural = ChatTokenizer::from_tokenizer(raw);
    missing_structural.set_template_kwargs(serde_json::Map::from_iter([(
        "bos_token".into(),
        json!("<｜begin▁of▁sentence｜>"),
    )]));
    let missing = prepare_chat_from_parts(
        &mut missing_structural,
        ModelChatTemplate::Single(DEEPSEEK_V31_TOOL_FIXTURE.into()),
        "unrelated",
        &[],
        Some(&Ok(ConstraintCompiler::synthetic_for_tests())),
        ChatTemplateRequest {
            messages: vec![json!({"role": "user", "content": "Use a tool."})],
            tools: vec![production_tool("lookup")],
            tool_choice: ToolChoice::Required,
            enable_thinking: Some(false),
            add_generation_prompt: true,
            ..ChatTemplateRequest::default()
        },
    )
    .unwrap();
    assert_eq!(missing.format_profile_identity(), None);
    assert!(missing.tool_runtime_plan().is_none());
    assert!(matches!(
        missing.native_tool_support(),
        NativeToolSupport::Unsupported { .. }
    ));
}

#[test]
fn inkling_template_recognition_routes_reasoning_and_visible_text() {
    let compiler = Ok(ConstraintCompiler::synthetic_for_tests());
    let mut tokenizer = inkling_chat_tokenizer(41);
    let prepared = prepare_chat_from_parts(
        &mut tokenizer,
        ModelChatTemplate::Single(INKLING_SMALL_FIXTURE.into()),
        "architecture-and-repository-are-not-support-keys",
        &[],
        Some(&compiler),
        ChatTemplateRequest {
            messages: vec![json!({"role": "user", "content": "Why is the sky blue?"})],
            enable_thinking: Some(true),
            add_generation_prompt: true,
            ..ChatTemplateRequest::default()
        },
    )
    .unwrap();

    assert_eq!(
        prepared.format_profile_identity(),
        Some("inkling.messages.v1")
    );
    assert!(matches!(
        prepared.semantic_support(),
        crate::runtime::chat::SemanticSupport::Supported
    ));
    assert!(prepared.capabilities().reasoning_parser.is_supported());
    assert!(prepared.capabilities().visible_text_parser.is_supported());
    assert!(prepared.capabilities().tool_output_parser.is_supported());
    assert!(matches!(
        prepared.native_tool_support(),
        NativeToolSupport::Supported
    ));
    assert_eq!(prepared.generation_prompt(), "<|message_model|>");
    assert!(prepared.rendered_prompt().contains(concat!(
        "<|message_system|><|content_text|>",
        "Thinking effort level: 0.9<|end_message|>"
    )));
    assert_eq!(
        prepared.profile_stop_sequences(),
        ["<|content_model_end_sampling|>"]
    );
    let generation_plan = prepared
        .generation_runtime_plan()
        .expect("recognized Inkling chat must compile its generation grammar");
    assert!(!generation_plan.has_tool_surface());
    assert!(constrained_sampler(DefaultSampler, generation_plan)
        .unwrap()
        .controller()
        .constraint_is_active());

    let plan = prepared
        .semantic_runtime_plan()
        .expect("recognized Inkling template must prepare a semantic parser");
    let structural = plan.structural_tokens().collect::<Vec<_>>();
    let token_id = |spelling: &str| {
        structural
            .iter()
            .find_map(|(token_id, candidate)| (*candidate == spelling).then_some(*token_id))
            .unwrap_or_else(|| panic!("missing Inkling structural token {spelling}"))
    };
    let reasoning = "private 東京 🦀 thought";
    let visible = "The sky is blue because of Rayleigh scattering.";
    for reasoning_split in (0..=reasoning.len()).filter(|index| reasoning.is_char_boundary(*index))
    {
        for visible_split in (0..=visible.len()).filter(|index| visible.is_char_boundary(*index)) {
            let mut parser = plan.create_parser_with_stops(std::iter::empty()).unwrap();
            parser
                .push_structural(token_id("<|content_thinking|>"), "<|content_thinking|>")
                .unwrap();
            parser.push(&reasoning[..reasoning_split]).unwrap();
            parser.push(&reasoning[reasoning_split..]).unwrap();
            parser
                .push_structural(token_id("<|end_message|>"), "<|end_message|>")
                .unwrap();
            parser
                .push_structural(token_id("<|message_model|>"), "<|message_model|>")
                .unwrap();
            parser
                .push_structural(token_id("<|content_text|>"), "<|content_text|>")
                .unwrap();
            parser.push(&visible[..visible_split]).unwrap();
            parser.push(&visible[visible_split..]).unwrap();
            parser
                .push_structural(token_id("<|end_message|>"), "<|end_message|>")
                .unwrap();
            assert!(parser
                .push_structural(
                    token_id("<|content_model_end_sampling|>"),
                    "<|content_model_end_sampling|>",
                )
                .unwrap());

            let parsed_reasoning = parser
                .events()
                .iter()
                .filter_map(|event| match event {
                    SemanticEvent::ReasoningDelta(text) => Some(text.as_str()),
                    _ => None,
                })
                .collect::<String>();
            let parsed_visible = parser
                .events()
                .iter()
                .filter_map(|event| match event {
                    SemanticEvent::TextDelta(text) => Some(text.as_str()),
                    _ => None,
                })
                .collect::<String>();
            assert_eq!(parsed_reasoning, reasoning);
            assert_eq!(parsed_visible, visible);
            assert_eq!(
                parser.events().last(),
                Some(&SemanticEvent::Finished {
                    reason: FinishReason::StopSequence,
                })
            );
        }
    }
}

#[test]
fn production_gemma4_templates_render_exact_thinking_tool_history_and_prompts() {
    let compiler = Ok(ConstraintCompiler::synthetic_for_tests());
    let messages = vec![
        json!({"role": "user", "content": "first"}),
        json!({
            "role": "assistant",
            "content": "",
            "reasoning_content": "inspect",
            "tool_calls": [
                {
                    "id": "call_a",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": {"value": 1}}
                },
                {
                    "id": "call_b",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": {"value": 2}}
                }
            ]
        }),
        json!({"role": "tool", "tool_call_id": "call_a", "content": "{\"result\":1}"}),
        json!({"role": "tool", "tool_call_id": "call_b", "content": "{\"result\":2}"}),
        json!({"role": "user", "content": "again"}),
    ];
    let tool_declaration = concat!(
        "<|tool>declaration:lookup{description:<|\"|>Look up one integer.<|\"|>,",
        "parameters:{properties:{value:{description:<|\"|>The integer to look up.<|\"|>,",
        "type:<|\"|>INTEGER<|\"|>}},required:[<|\"|>value<|\"|>],",
        "type:<|\"|>OBJECT<|\"|>}}<tool|>",
    );
    let history = concat!(
        "<|turn>user\nfirst<turn|>\n",
        "<|turn>model\n<|channel>thought\ninspect\n<channel|>",
        "<|tool_call>call:lookup{value:1}<tool_call|>",
        "<|tool_call>call:lookup{value:2}<tool_call|>",
        "<|tool_response>response:lookup{value:<|\"|>{\"result\":1}<|\"|>}",
        "<tool_response|><|tool_response>response:lookup{value:<|\"|>{\"result\":2}",
        "<|\"|>}<tool_response|><turn|>\n",
        "<|turn>user\nagain<turn|>\n",
    );
    for (template, identity, disabled_generation_prompt) in [
        (GEMMA4_EDGE_FIXTURE, "gemma.channels.v1", "<|turn>model\n"),
        (
            GEMMA4_LARGE_FIXTURE,
            "gemma.channels.v1",
            "<|turn>model\n<|channel>thought\n<channel|>",
        ),
    ] {
        for enable_thinking in [false, true] {
            for tools in [Vec::new(), vec![production_tool("lookup")]] {
                let tool_choice = if tools.is_empty() {
                    ToolChoice::Auto
                } else {
                    ToolChoice::Required
                };
                for add_generation_prompt in [false, true] {
                    let mut tokenizer = gemma4_chat_tokenizer(10);
                    let prepared = prepare_chat_from_parts(
                        &mut tokenizer,
                        ModelChatTemplate::Single(template.into()),
                        "model-type-is-not-a-support-key",
                        &[],
                        Some(&compiler),
                        ChatTemplateRequest {
                            messages: messages.clone(),
                            tools: tools.clone(),
                            tool_choice,
                            parallel_tool_calls: ParallelToolCallPolicy::Enabled {
                                max_calls: std::num::NonZeroUsize::new(2),
                            },
                            enable_thinking: Some(enable_thinking),
                            reasoning_effort: None,
                            allow_unparsed_reasoning: false,
                            add_generation_prompt,
                            extra_template_kwargs: serde_json::Map::from_iter([(
                                "preserve_thinking".into(),
                                json!(true),
                            )]),
                        },
                    )
                    .unwrap();
                    let system = if enable_thinking || !tools.is_empty() {
                        format!(
                            "<|turn>system\n{}{}<turn|>\n",
                            if enable_thinking { "<|think|>\n" } else { "" },
                            if tools.is_empty() {
                                ""
                            } else {
                                tool_declaration
                            },
                        )
                    } else {
                        String::new()
                    };
                    let generation_prompt = if enable_thinking {
                        "<|turn>model\n"
                    } else {
                        disabled_generation_prompt
                    };
                    let expected = format!(
                        "<bos>{system}{history}{}",
                        if add_generation_prompt {
                            generation_prompt
                        } else {
                            ""
                        }
                    );

                    assert_eq!(prepared.rendered_prompt(), expected, "{identity}");
                    assert_eq!(prepared.generation_prompt(), generation_prompt);
                    assert_eq!(prepared.format_profile_identity(), Some(identity));
                    assert_eq!(
                        prepared.preserved_structural_token_ids(),
                        if tools.is_empty() {
                            &[10, 11, 15, 16][..]
                        } else {
                            &[10, 11, 12, 13, 14, 15, 16][..]
                        }
                    );
                    assert_eq!(
                        prepared.profile_stop_sequences(),
                        ["<|tool_response>", "<turn|>"]
                    );
                    if tools.is_empty() {
                        assert!(prepared.tool_runtime_plan().is_none());
                        assert!(matches!(
                            prepared.semantic_support(),
                            crate::runtime::chat::SemanticSupport::Supported
                        ));
                    } else {
                        let plan = prepared
                            .tool_runtime_plan()
                            .expect("recognized Gemma tool protocol must be supported");
                        assert_eq!(plan.auto_activation_trigger(), None);
                        assert!(constrained_sampler(DefaultSampler, plan)
                            .unwrap()
                            .controller()
                            .constraint_is_active());
                    }
                }
            }
        }
    }
}

#[test]
fn gguf_eos_ids_merge_with_sidecars_and_loader_metadata() {
    let dir = temp_model_dir(r#"{"model_type":"llama","eos_token_id":1}"#);
    fs::write(
        dir.join("generation_config.json"),
        r#"{"eos_token_id":[2,3]}"#,
    )
    .unwrap();
    let metadata = std::collections::HashMap::from([(
        "tokenizer.ggml.eos_token_id".into(),
        GgufMetadataValue::Array(GgufMetadataArray::Uint32(vec![3, 4])),
    )]);

    let merged = merge_eos_token_id_sources([
        eos_token_ids_from_sidecar_dir(&dir).unwrap(),
        vec![4, 5],
        gguf_eos_token_ids(&metadata).unwrap(),
    ]);

    assert_eq!(merged, [1, 2, 3, 4, 5]);
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn eos_loading_rejects_invalid_json_and_gguf_values() {
    for value in ["-1", "4294967296", "1.5", r#"[1,"2"]"#] {
        let dir = temp_model_dir(&format!(
            r#"{{"model_type":"llama","eos_token_id":{value}}}"#
        ));
        assert!(
            eos_token_ids_from_sidecar_dir(&dir).is_err(),
            "accepted invalid JSON EOS value {value}"
        );
        fs::remove_dir_all(dir).unwrap();
    }

    for value in [
        GgufMetadataValue::Int64(-1),
        GgufMetadataValue::Uint64(u64::from(u32::MAX) + 1),
        GgufMetadataValue::Array(GgufMetadataArray::Int64(vec![1, -2])),
        GgufMetadataValue::String("3".into()),
    ] {
        let metadata =
            std::collections::HashMap::from([("tokenizer.ggml.eos_token_id".into(), value)]);
        assert!(
            gguf_eos_token_ids(&metadata).is_err(),
            "accepted invalid GGUF EOS value"
        );
    }
}

#[test]
fn load_time_quantization_accepts_only_unquantized_gguf_sources() {
    let dense_metadata = std::collections::HashMap::from([(
        "general.file_type".into(),
        GgufMetadataValue::Uint32(1),
    )]);
    validate_gguf_quantization_source(
        &std::collections::HashMap::new(),
        &dense_metadata,
        Some(WeightQuantization::MxFp4),
    )
    .unwrap();

    let error = validate_gguf_quantization_source(
        &std::collections::HashMap::new(),
        &std::collections::HashMap::new(),
        Some(WeightQuantization::MxFp4),
    )
    .unwrap_err();
    assert!(error.to_string().contains("general.file_type"));

    let quantized_metadata = std::collections::HashMap::from([(
        "general.file_type".into(),
        GgufMetadataValue::Uint32(7),
    )]);
    let error = validate_gguf_quantization_source(
        &std::collections::HashMap::new(),
        &quantized_metadata,
        Some(WeightQuantization::MxFp4),
    )
    .unwrap_err();
    assert!(error.to_string().contains("already quantized"));

    let packed_arrays = std::collections::HashMap::from([(
        "blk.0.attn_q.scales".into(),
        Array::from_slice(&[1.0f32], &[1]),
    )]);
    let error = validate_gguf_quantization_source(
        &packed_arrays,
        &dense_metadata,
        Some(WeightQuantization::MxFp4),
    )
    .unwrap_err();
    assert!(error.to_string().contains("packed GGUF tensors"));
}

fn write_zero_llama_gguf(path: &std::path::Path) {
    let metadata = BTreeMap::from([
        (
            "general.architecture".into(),
            GgufWriterMetadata::String("llama".into()),
        ),
        ("general.file_type".into(), GgufWriterMetadata::Uint32(0)),
        ("llama.block_count".into(), GgufWriterMetadata::Uint32(1)),
        (
            "llama.embedding_length".into(),
            GgufWriterMetadata::Uint32(32),
        ),
        (
            "llama.attention.head_count".into(),
            GgufWriterMetadata::Uint32(4),
        ),
        (
            "llama.attention.head_count_kv".into(),
            GgufWriterMetadata::Uint32(2),
        ),
        (
            "llama.attention.key_length".into(),
            GgufWriterMetadata::Uint32(8),
        ),
        (
            "llama.feed_forward_length".into(),
            GgufWriterMetadata::Uint32(64),
        ),
        (
            "llama.attention.layer_norm_rms_epsilon".into(),
            GgufWriterMetadata::Float32(0.00001),
        ),
        (
            "llama.context_length".into(),
            GgufWriterMetadata::Uint32(128),
        ),
        ("llama.vocab_size".into(), GgufWriterMetadata::Uint32(32)),
    ]);
    let specs = [
        ("token_embd.weight", vec![32, 32]),
        ("output_norm.weight", vec![32]),
        ("blk.0.attn_norm.weight", vec![32]),
        ("blk.0.ffn_norm.weight", vec![32]),
        ("blk.0.attn_q.weight", vec![32, 32]),
        ("blk.0.attn_k.weight", vec![32, 16]),
        ("blk.0.attn_v.weight", vec![32, 16]),
        ("blk.0.attn_output.weight", vec![32, 32]),
        ("blk.0.ffn_gate.weight", vec![32, 64]),
        ("blk.0.ffn_up.weight", vec![32, 64]),
        ("blk.0.ffn_down.weight", vec![64, 32]),
    ];
    let payloads = specs
        .iter()
        .map(|(_, dimensions)| vec![0u8; dimensions.iter().product::<u64>() as usize * 4])
        .collect::<Vec<_>>();
    let tensors = specs
        .iter()
        .zip(&payloads)
        .map(|((name, dimensions), data)| TensorInput {
            name,
            dimensions,
            ggml_type: GgmlType::F32,
            data,
        })
        .collect::<Vec<_>>();
    Writer::default()
        .write(std::fs::File::create(path).unwrap(), &metadata, &tensors)
        .unwrap();
}

#[test]
fn dense_gguf_uses_shared_packed_overlay_for_nonresident_execution() {
    let context = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
    let weights_context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = context.stream();
    let weights_stream = weights_context.stream();
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("llama-f32.gguf");
    write_zero_llama_gguf(&path);
    let quantization = WeightQuantization::Affine(AffineQuantization::new(32, 4).unwrap());
    let policies = [
        crate::backend::mlx::runtime::execution::layerwise::WeightResidency::layerwise_host(
            crate::backend::mlx::runtime::execution::layerwise::LayerwiseLoadOptions::default(),
        ),
        crate::backend::mlx::runtime::execution::layerwise::WeightResidency::dense_disk_stream(
            crate::backend::mlx::runtime::residency::dense_stream::DenseDiskStreamLoadOptions::new(
                1 << 20,
                1 << 20,
                1,
                1,
            )
            .unwrap(),
        ),
    ];

    let mut tokens_by_policy = Vec::new();
    for residency in policies {
        let options =
            ModelLoadOptions::with_quantization(quantization).with_weight_residency(residency);
        let mut loaded = load_test_model(&path, options, stream, weights_stream)
            .unwrap()
            .into_inner()
            .into_complete()
            .unwrap();
        let Model::Llama(model) = &loaded else {
            panic!("expected Llama GGUF model");
        };
        let materialization = model.metadata().materialization().unwrap();
        assert!(materialization.transformed_weights > 0);
        assert!(materialization.output_bytes < materialization.source_bytes_read);
        let diagnostics = model.checkpoint_store().diagnostics().unwrap();
        assert!(diagnostics.physical_reads > 0);
        assert!(
            diagnostics.physical_read_bytes
                <= materialization
                    .source_bytes_read
                    .saturating_add(materialization.output_bytes)
                    .saturating_add(4096)
        );

        let tokens = Array::from_slice(&[1u32, 2], &[1, 2]);
        let parts = [input::InputPart::text_token_ids(&tokens)];
        let mut cache = loaded.new_cache();
        let logits = loaded
            .submit_prefill(input::ModelInput::new(&parts), &mut cache, stream)
            .unwrap()
            .wait()
            .unwrap();
        tokens_by_policy.push(
            argmax_axis!(&logits, -1, stream = stream)
                .unwrap()
                .item::<u32>(stream),
        );
    }
    assert_eq!(tokens_by_policy[0], tokens_by_policy[1]);
}

fn save_zero_checkpoint<M: ModuleParameters>(model: &M, dir: &std::path::Path, stream: &Stream) {
    let parameters = model.parameters().flatten();
    let arrays = parameters
        .iter()
        .map(|(name, parameter)| {
            (
                name.to_string(),
                zeros_dtype(parameter.shape(), parameter.dtype(), stream).unwrap(),
            )
        })
        .collect::<Vec<_>>();
    Array::save_safetensors(
        arrays.iter().map(|(name, array)| (name.as_str(), array)),
        None,
        dir.join("model.safetensors"),
    )
    .unwrap();
}

#[test]
fn tiny_text_families_quantize_through_high_level_dispatch() {
    let context = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
    let weights_context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = context.stream();
    let weights_stream = weights_context.stream();
    let fixtures = [
        (
            r#"{
                  "model_type":"llama","hidden_size":32,"num_hidden_layers":1,
                  "intermediate_size":64,"num_attention_heads":4,"num_key_value_heads":2,
                  "head_dim":8,"rms_norm_eps":0.00001,"vocab_size":32,
                  "max_position_embeddings":128,"tie_word_embeddings":true,
                  "rope_scaling":null
                }"#,
            "llama",
        ),
        (
            r#"{
                  "model_type":"mistral","hidden_size":32,"num_hidden_layers":1,
                  "intermediate_size":64,"num_attention_heads":4,"num_key_value_heads":2,
                  "head_dim":8,"rms_norm_eps":0.00001,"vocab_size":32,
                  "max_position_embeddings":128,"sliding_window":16,
                  "tie_word_embeddings":true,"rope_scaling":null
                }"#,
            "mistral",
        ),
        (
            r#"{
                  "model_type":"qwen3","hidden_size":32,"num_hidden_layers":1,
                  "intermediate_size":64,"num_attention_heads":4,"num_key_value_heads":2,
                  "head_dim":8,"rms_norm_eps":0.00001,"vocab_size":32,
                  "max_position_embeddings":128,"rope_theta":10000.0,
                  "tie_word_embeddings":true,"rope_scaling":null
                }"#,
            "qwen3",
        ),
        (
            r#"{
                  "model_type":"qwen3_5_text","vocab_size":32,"hidden_size":32,
                  "num_hidden_layers":1,"num_attention_heads":4,"num_key_value_heads":2,
                  "head_dim":8,"max_position_embeddings":128,"rms_norm_eps":0.000001,
                  "tie_word_embeddings":true,"attention_bias":false,"hidden_act":"silu",
                  "intermediate_size":64,"layer_types":["full_attention"]
                }"#,
            "qwen3_5",
        ),
        (
            r#"{
                  "model_type":"gemma4",
                  "tie_word_embeddings":true,
                  "text_config":{
                    "model_type":"gemma4_text","hidden_size":32,"num_hidden_layers":1,
                    "intermediate_size":64,"num_attention_heads":4,"num_key_value_heads":2,
                    "head_dim":8,"rms_norm_eps":0.00001,"vocab_size":32,
                    "max_position_embeddings":128,"tie_word_embeddings":true,
                    "attention_k_eq_v":false,"layer_types":["full_attention"]
                  }
                }"#,
            "gemma4",
        ),
    ];

    for (config, family) in fixtures {
        let dir = temp_model_dir(config);
        match family {
            "llama" | "mistral" => {
                let args = llama::get_llama_model_args(&dir).unwrap();
                save_zero_checkpoint(
                    &llama::ResidentModel::new(args, stream).unwrap(),
                    &dir,
                    stream,
                );
            }
            "qwen3" => {
                let args = dense_qwen::load_config(&dir).unwrap();
                save_zero_checkpoint(&dense_qwen::Model::new(args, stream).unwrap(), &dir, stream);
            }
            "qwen3_5" => {
                let (args, image_token_id, video_token_id, vision_config) =
                    qwen3_5::get_qwen3_5_model_args(&dir).unwrap();
                save_zero_checkpoint(
                    &qwen3_5::Model::new(
                        args,
                        image_token_id,
                        video_token_id,
                        vision_config,
                        stream,
                    )
                    .unwrap(),
                    &dir,
                    stream,
                );
            }
            "gemma4" => {
                let args = gemma4::get_gemma4_model_args(&dir).unwrap();
                save_zero_checkpoint(&gemma4::Model::new(args, stream).unwrap(), &dir, stream);
            }
            _ => unreachable!(),
        }

        for quantization in [
            WeightQuantization::Affine(AffineQuantization::new(32, 4).unwrap()),
            WeightQuantization::MxFp4,
        ] {
            let mut dense =
                load_test_model(&dir, ModelLoadOptions::default(), stream, weights_stream)
                    .unwrap()
                    .into_inner()
                    .into_complete()
                    .unwrap();
            let mut quantized = load_test_model(
                &dir,
                ModelLoadOptions::with_quantization(quantization),
                stream,
                weights_stream,
            )
            .unwrap_or_else(|error| panic!("{family} {quantization:?}: {error}"))
            .into_inner()
            .into_complete()
            .unwrap();
            let suffix = if quantization == WeightQuantization::MxFp4 {
                "mxfp4"
            } else {
                "q4"
            };
            let saved_dir = dir.with_extension(suffix);
            crate::backend::mlx::runtime::checkpoint::quantization::quantize_checkpoint(
                &dir,
                &saved_dir,
                &CheckpointQuantizationOptions {
                    quantization,
                    ..Default::default()
                },
                stream,
            )
            .unwrap();
            let mut saved_quantized = load_test_model(
                &saved_dir,
                ModelLoadOptions::with_quantization(quantization),
                stream,
                weights_stream,
            )
            .unwrap()
            .into_inner()
            .into_complete()
            .unwrap();
            let tokens = Array::from_slice(&[1u32, 2], &[1, 2]);
            let parts = [input::InputPart::text_token_ids(&tokens)];
            let input = input::ModelInput::new(&parts);
            let mut dense_cache = dense.new_cache();
            let dense_logits = dense
                .submit_prefill(input, &mut dense_cache, stream)
                .unwrap()
                .wait()
                .unwrap();
            let mut quantized_cache = quantized.new_cache();
            let quantized_logits = quantized
                .submit_prefill(input, &mut quantized_cache, stream)
                .unwrap()
                .wait()
                .unwrap();
            assert_eq!(dense_logits.shape(), quantized_logits.shape());
            let dense_token = argmax_axis!(&dense_logits, -1, stream = stream)
                .unwrap()
                .item::<u32>(stream);
            let quantized_token = argmax_axis!(&quantized_logits, -1, stream = stream)
                .unwrap()
                .item::<u32>(stream);
            assert_eq!(dense_token, quantized_token, "{family} {quantization:?}");
            let mut saved_cache = saved_quantized.new_cache();
            let saved_logits = saved_quantized
                .submit_prefill(input, &mut saved_cache, stream)
                .unwrap()
                .wait()
                .unwrap();
            let saved_token = argmax_axis!(&saved_logits, -1, stream = stream)
                .unwrap()
                .item::<u32>(stream);
            assert_eq!(
                quantized_token, saved_token,
                "saved {family} {quantization:?}"
            );

            let decode_input = Array::from_slice(&[dense_token], &[1, 1]);
            let dense_decode = dense
                .submit_decode(decode_input.clone(), &mut dense_cache, stream)
                .unwrap()
                .wait()
                .unwrap();
            let quantized_decode = quantized
                .submit_decode(decode_input.clone(), &mut quantized_cache, stream)
                .unwrap()
                .wait()
                .unwrap();
            let saved_decode = saved_quantized
                .submit_decode(decode_input, &mut saved_cache, stream)
                .unwrap()
                .wait()
                .unwrap();
            assert_eq!(dense_decode.shape(), quantized_decode.shape());
            assert_eq!(quantized_decode.shape(), saved_decode.shape());
            assert_eq!(
                argmax_axis!(&dense_decode, -1, stream = stream)
                    .unwrap()
                    .item::<u32>(stream),
                argmax_axis!(&quantized_decode, -1, stream = stream)
                    .unwrap()
                    .item::<u32>(stream),
                "decode {family} {quantization:?}"
            );
            if family == "llama" && matches!(quantization, WeightQuantization::Affine(_)) {
                struct FixedTokenController;

                impl eredu_core::TokenFilterController for FixedTokenController {
                    type Error = std::convert::Infallible;

                    fn current_filter(&mut self) -> Result<eredu_core::TokenFilter, Self::Error> {
                        let mut allowed = vec![false; 32];
                        allowed[5] = true;
                        Ok(eredu_core::TokenFilter::allowed(allowed).unwrap())
                    }

                    fn commit_token(&mut self, token_id: u32) -> Result<(), Self::Error> {
                        assert_eq!(token_id, 5);
                        Ok(())
                    }

                    fn is_complete(&mut self) -> Result<bool, Self::Error> {
                        Ok(false)
                    }
                }

                let mut runtime = eredu_core::ModelRuntime::from_prepared(
                    crate::backend::mlx::MlxBackend::new(stream, stream),
                    eredu_core::PreparedModel::new(crate::backend::mlx::MlxModel::complete(dense)),
                )
                .unwrap();
                let sampling =
                    eredu_core::resolve_generation_config(None, Default::default()).unwrap();
                let mut generation = eredu_core::ControlledTextGeneration::new(
                    &mut runtime,
                    vec![1, 2],
                    eredu_core::TextGenerationConfig::new(sampling),
                    FixedTokenController,
                )
                .unwrap();
                for _ in 0..3 {
                    let token = generation.next().unwrap().unwrap();
                    assert_eq!(token.token_id(), 5);
                }
            }
            fs::remove_dir_all(saved_dir).unwrap();
        }
        fs::remove_dir_all(dir).unwrap();
    }
}

#[test]
fn tiny_gpt_oss_preserves_native_experts_and_quantizes_dense_matrices_to_mxfp4() {
    let context = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
    let weights_context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = context.stream();
    let weights_stream = weights_context.stream();
    let dir = temp_model_dir(
        r#"{
              "model_type":"gpt_oss","hidden_size":32,"intermediate_size":32,
              "num_hidden_layers":1,"num_attention_heads":4,"num_key_value_heads":2,
              "head_dim":8,"vocab_size":32,"num_local_experts":2,
              "num_experts_per_tok":1,"rms_norm_eps":0.00001,"sliding_window":16,
              "max_position_embeddings":128,"rope_scaling":null,
              "quantization_config":{"quant_method":"mxfp4"}
            }"#,
    );
    let args = gpt_oss::get_model_args(&dir).unwrap();
    save_zero_checkpoint(&gpt_oss::Model::new(args, stream).unwrap(), &dir, stream);

    let model = load_test_model(
        &dir,
        ModelLoadOptions::with_quantization(WeightQuantization::MxFp4),
        stream,
        weights_stream,
    )
    .unwrap()
    .into_inner()
    .into_complete()
    .unwrap();
    let Model::GptOss(model) = model else {
        panic!("expected GPT-OSS model")
    };
    assert_eq!(
        model.residency_metadata().quantization(),
        Some(WeightQuantization::MxFp4)
    );
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn tiny_qwen3_vl_mxfp4_on_load_quantizes_only_language_model() {
    let context = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
    let weights_context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = context.stream();
    let weights_stream = weights_context.stream();
    let dir = temp_model_dir(
        r#"{
              "model_type":"qwen3_vl","image_token_id":30,"video_token_id":31,
              "text_config":{
                "model_type":"qwen3_vl_text","hidden_size":32,"num_hidden_layers":1,
                "intermediate_size":64,"num_attention_heads":4,"num_key_value_heads":2,
                "head_dim":8,"rms_norm_eps":0.000001,"vocab_size":32,
                "max_position_embeddings":128,"rope_theta":10000.0,
                "tie_word_embeddings":true,
                "rope_scaling":{"mrope_section":[2,1,1],"mrope_interleaved":true}
              },
              "vision_config":{
                "depth":1,"hidden_size":8,"hidden_act":"gelu_pytorch_tanh",
                "intermediate_size":16,"num_heads":2,"num_position_embeddings":16,
                "in_channels":3,"patch_size":2,"spatial_merge_size":2,
                "temporal_patch_size":2,"out_hidden_size":32,
                "deepstack_visual_indexes":[0]
              }
            }"#,
    );
    let args = qwen3_vl::get_qwen3_vl_model_args(&dir).unwrap();
    save_zero_checkpoint(&qwen3_vl::Model::new(args, stream).unwrap(), &dir, stream);

    let quantization = WeightQuantization::MxFp4;
    let mut quantized = load_test_model(
        &dir,
        ModelLoadOptions::with_quantization(quantization),
        stream,
        weights_stream,
    )
    .unwrap()
    .into_inner()
    .into_complete()
    .unwrap();
    let Model::Qwen3Vl(model) = &quantized else {
        panic!("expected Qwen3-VL model");
    };
    assert_eq!(
        model.residency_metadata().quantization(),
        Some(WeightQuantization::MxFp4)
    );

    let tokens = Array::from_slice(&[1u32, 2], &[1, 2]);
    let pixels = Array::zeros::<f32>(&[4, 24], stream).unwrap();
    let grid = Array::from_slice(&[1i32, 2, 2], &[1, 3]);
    let parts = [
        input::InputPart::text_token_ids(&tokens),
        input::InputPart::image_tensor(&pixels, input::InputMetadata::qwen_grid_thw(&grid)),
    ];
    let mut cache = quantized.new_cache();
    let logits = quantized
        .submit_prefill(input::ModelInput::new(&parts), &mut cache, stream)
        .unwrap()
        .wait()
        .unwrap();
    assert_eq!(logits.shape(), &[1, 32]);

    let saved_dir = dir.with_extension("mxfp4");
    crate::backend::mlx::runtime::checkpoint::quantization::quantize_checkpoint(
        &dir,
        &saved_dir,
        &CheckpointQuantizationOptions {
            quantization,
            exclude: vec!["model.visual.".into()],
            ..Default::default()
        },
        stream,
    )
    .unwrap();
    let saved_quantized = load_test_model(
        &saved_dir,
        ModelLoadOptions::with_quantization(quantization),
        stream,
        weights_stream,
    )
    .unwrap()
    .into_inner()
    .into_complete()
    .unwrap();
    let Model::Qwen3Vl(saved_model) = &saved_quantized else {
        panic!("expected saved Qwen3-VL model");
    };
    assert_eq!(
        saved_model.residency_metadata().quantization(),
        Some(WeightQuantization::MxFp4)
    );

    fs::remove_dir_all(dir).unwrap();
    fs::remove_dir_all(saved_dir).unwrap();
}

#[test]
fn load_policy_admits_fully_resident_inkling_and_nemotron_materialization() {
    for quantization in [
        WeightQuantization::Affine(AffineQuantization::default()),
        WeightQuantization::MxFp4,
    ] {
        let options = ModelLoadOptions::with_quantization(quantization);
        for kind in [
            crate::core::ModelKind::Inkling,
            crate::core::ModelKind::NemotronH,
        ] {
            options
                .validate_preparation(kind, None, eredu_core::ArtifactFormat::SafeTensors)
                .unwrap();
        }
    }
}

#[test]
fn tiny_qwen35_moe_mxfp4_quantizes_packed_experts_through_high_level_dispatch() {
    let context = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
    let weights_context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = context.stream();
    let weights_stream = weights_context.stream();
    let config = r#"{
          "model_type":"qwen3_5_moe",
          "tie_word_embeddings":false,
          "text_config":{
            "model_type":"qwen3_5_moe_text","vocab_size":32,"hidden_size":32,
            "num_hidden_layers":1,"num_attention_heads":4,"num_key_value_heads":2,
            "head_dim":8,"max_position_embeddings":128,"rms_norm_eps":0.000001,
            "tie_word_embeddings":false,"attention_bias":false,"hidden_act":"silu",
            "moe_intermediate_size":32,"shared_expert_intermediate_size":32,
            "num_experts_per_tok":2,"num_experts":4,"norm_topk_prob":true,
            "layer_types":["full_attention"]
          }
        }"#;
    let dir = temp_model_dir(config);
    let (args, image_token_id, video_token_id, vision_config) =
        qwen3_5::get_qwen3_5_model_args(&dir).unwrap();
    save_zero_checkpoint(
        &qwen3_5::Model::new(args, image_token_id, video_token_id, vision_config, stream).unwrap(),
        &dir,
        stream,
    );

    let mut dense = load_test_model(&dir, ModelLoadOptions::default(), stream, weights_stream)
        .unwrap()
        .into_inner()
        .into_complete()
        .unwrap();
    let mut quantized = load_test_model(
        &dir,
        ModelLoadOptions::with_quantization(WeightQuantization::MxFp4),
        stream,
        weights_stream,
    )
    .unwrap()
    .into_inner()
    .into_complete()
    .unwrap();
    let Model::Qwen35(quantized_model) = &quantized else {
        panic!("expected Qwen3.5-MoE model");
    };
    assert_eq!(
        quantized_model.residency_metadata().quantization(),
        Some(WeightQuantization::MxFp4)
    );

    let tokens = Array::from_slice(&[1u32, 2], &[1, 2]);
    let parts = [input::InputPart::text_token_ids(&tokens)];
    let input = input::ModelInput::new(&parts);
    let mut dense_cache = dense.new_cache();
    let dense_logits = dense
        .submit_prefill(input, &mut dense_cache, stream)
        .unwrap()
        .wait()
        .unwrap();
    let mut quantized_cache = quantized.new_cache();
    let quantized_logits = quantized
        .submit_prefill(input, &mut quantized_cache, stream)
        .unwrap()
        .wait()
        .unwrap();
    assert_eq!(dense_logits.shape(), quantized_logits.shape());
    assert_eq!(
        argmax_axis!(&dense_logits, -1, stream = stream)
            .unwrap()
            .item::<u32>(stream),
        argmax_axis!(&quantized_logits, -1, stream = stream)
            .unwrap()
            .item::<u32>(stream)
    );
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn resolve_model_config_reports_supported_llama() {
    let support = resolve_model_config(&json!({
        "model_type": "llama",
        "hidden_size": 8,
        "num_hidden_layers": 1,
        "intermediate_size": 16,
        "num_attention_heads": 2,
        "rms_norm_eps": 0.00001,
        "vocab_size": 32,
        "num_key_value_heads": 2,
        "max_position_embeddings": 128,
        "head_dim": 4
    }));

    assert!(support.is_ok(), "{support:?}");
}

#[test]
fn resolve_model_config_recognizes_exact_qwen2_identity() {
    let config = json!({
        "architectures": ["Qwen2ForCausalLM"],
        "model_type": "qwen2",
        "hidden_size": 16,
        "num_hidden_layers": 2,
        "intermediate_size": 32,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-6,
        "vocab_size": 64,
        "max_position_embeddings": 128,
        "rope_theta": 1000000.0,
        "tie_word_embeddings": false,
        "use_sliding_window": false,
        "sliding_window": 64,
        "max_window_layers": 2
    });
    assert_eq!(
        resolve_model_config(&config).ok(),
        Some(ResolvedModelConfig {
            kind: ModelKind::Qwen2,
            model_type: "qwen2".into(),
            effective_model_type: "qwen2".into(),
        })
    );

    for nearby in ["qwen", "qwen2_moe", "qwen2_vl", "qwen2_5_vl"] {
        let mut unsupported = config.clone();
        unsupported["model_type"] = json!(nearby);
        assert!(
            !resolve_model_config(&unsupported).is_ok(),
            "nearby architecture {nearby:?} must fail closed"
        );
    }
    let mut disguised_vl = config;
    disguised_vl["architectures"] = json!(["Qwen2VLForConditionalGeneration"]);
    assert!(!resolve_model_config(&disguised_vl).is_ok());
}

#[test]
fn resolve_model_config_validates_qwen2_sliding_window() {
    let config = json!({
        "model_type": "qwen2", "hidden_size": 16, "num_hidden_layers": 4,
        "intermediate_size": 32, "num_attention_heads": 4,
        "num_key_value_heads": 2, "rms_norm_eps": 1e-6, "vocab_size": 64,
        "max_position_embeddings": 128, "rope_theta": 10000.0,
        "tie_word_embeddings": true, "use_sliding_window": true,
        "sliding_window": 8, "max_window_layers": 2
    });
    assert!(resolve_model_config(&config).is_ok());

    let mut missing_window = config.clone();
    missing_window["sliding_window"] = serde_json::Value::Null;
    assert!(resolve_model_config(&missing_window)
        .unwrap_err()
        .to_string()
        .contains("sliding_window must be a positive integer"));

    for invalid_window in [json!(0), json!(-1), json!(u64::MAX)] {
        let mut invalid = config.clone();
        invalid["sliding_window"] = invalid_window;
        assert_eq!(
            resolve_model_config(&invalid).is_ok(),
            crate::backend::mlx::architectures::qwen::dense::config_from_hf_value(&invalid).is_ok(),
            "inspection and load normalization diverged for {invalid}"
        );
    }
}

#[test]
fn resolve_model_config_reports_supported_kimi_linear() {
    let support = resolve_model_config(&json!({
        "model_type": "kimi_linear",
        "vocab_size": 163840,
        "hidden_size": 2304,
        "num_hidden_layers": 27,
        "num_attention_heads": 32,
        "num_key_value_heads": 1,
        "intermediate_size": 9216,
        "head_dim": 72,
        "model_max_length": 1048576,
        "rms_norm_eps": 0.00001,
        "rope_theta": 10000.0,
        "linear_attn_config": {
            "kda_layers": [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 21, 22, 23, 25, 26],
            "full_attn_layers": [4, 8, 12, 16, 20, 24, 27],
            "num_heads": 32,
            "head_dim": 128,
            "short_conv_kernel_size": 4
        },
        "num_experts": 256,
        "moe_intermediate_size": 1024,
        "kv_lora_rank": 512,
        "q_lora_rank": null,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "mla_use_nope": true,
        "num_experts_per_token": 8,
        "num_shared_experts": 1,
        "moe_router_activation_func": "sigmoid",
        "moe_renormalize": true,
        "routed_scaling_factor": 2.446,
        "first_k_dense_replace": 1,
        "moe_layer_freq": 1,
        "use_grouped_topk": true,
        "num_expert_group": 1,
        "topk_group": 1,
        "tie_word_embeddings": false,
        "num_nextn_predict_layers": 0
    }));

    assert_eq!(
        support.ok(),
        Some(ResolvedModelConfig {
            kind: ModelKind::KimiLinear,
            model_type: "kimi_linear".into(),
            effective_model_type: "kimi_linear".into(),
        })
    );
}

#[test]
fn resolve_model_config_reports_supported_dense_mistral() {
    let support = resolve_model_config(&json!({
        "architectures": ["MistralForCausalLM"],
        "model_type": "mistral",
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "intermediate_size": 14336,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "rms_norm_eps": 0.00001,
        "vocab_size": 32032,
        "max_position_embeddings": 32768,
        "rope_theta": 10000.0,
        "sliding_window": 4096,
        "tie_word_embeddings": false
    }));

    assert_eq!(
        support.ok(),
        Some(ResolvedModelConfig {
            kind: ModelKind::Llama,
            model_type: "mistral".to_string(),
            effective_model_type: "mistral".to_string(),
        })
    );
}

#[test]
fn resolve_model_config_reports_supported_lfm2_families() {
    let dense = json!({
        "model_type": "lfm2",
        "vocab_size": 32,
        "hidden_size": 16,
        "intermediate_size": 24,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "max_position_embeddings": 128,
        "layer_types": ["conv", "full_attention"]
    });
    assert_eq!(
        resolve_model_config(&dense).ok(),
        Some(ResolvedModelConfig {
            kind: ModelKind::Lfm2,
            model_type: "lfm2".into(),
            effective_model_type: "lfm2".into(),
        })
    );
    let mut moe = dense;
    moe["model_type"] = json!("lfm2_moe");
    moe["moe_intermediate_size"] = json!(8);
    moe["num_dense_layers"] = json!(1);
    moe["num_experts"] = json!(4);
    moe["num_experts_per_tok"] = json!(2);
    assert!(resolve_model_config(&moe).is_ok());
}

#[test]
fn resolve_model_config_reports_supported_gpt_oss() {
    let support = resolve_model_config(&json!({
        "model_type": "gpt_oss",
        "hidden_size": 2880,
        "intermediate_size": 2880,
        "num_hidden_layers": 24,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "head_dim": 64,
        "vocab_size": 201088,
        "num_local_experts": 32,
        "num_experts_per_tok": 4,
        "rms_norm_eps": 1e-5,
        "sliding_window": 128,
        "max_position_embeddings": 131072,
        "rope_scaling": {
            "rope_type": "yarn",
            "factor": 32.0,
            "original_max_position_embeddings": 4096,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "truncate": false
        },
        "quantization_config": {"quant_method": "mxfp4"}
    }));

    assert_eq!(
        support.ok(),
        Some(ResolvedModelConfig {
            kind: ModelKind::GptOss,
            model_type: "gpt_oss".to_string(),
            effective_model_type: "gpt_oss".to_string(),
        })
    );
}

#[test]
fn resolve_model_config_reports_supported_full_attention_mistral_small() {
    let support = resolve_model_config(&json!({
        "architectures": ["MistralForCausalLM"],
        "model_type": "mistral",
        "hidden_size": 5120,
        "num_hidden_layers": 40,
        "intermediate_size": 32768,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "rms_norm_eps": 0.00001,
        "vocab_size": 131072,
        "max_position_embeddings": 32768,
        "rope_theta": 100000000.0,
        "sliding_window": null,
        "tie_word_embeddings": false
    }));

    assert!(support.is_ok(), "{support:?}");
}

#[test]
fn resolve_model_config_reports_unsupported_model_type() {
    let support = resolve_model_config(&json!({
        "model_type": "not_a_model"
    }));

    assert!(!support.is_ok());
    assert_eq!(
        support.unwrap_err().to_string(),
        "unsupported model type: not_a_model"
    );
}

#[test]
fn resolve_model_config_reports_qwen3_5_moe_missing_text_config() {
    let support = resolve_model_config(&json!({
        "model_type": "qwen3_5_moe"
    }));

    assert!(!support.is_ok());
    assert_eq!(
        support.unwrap_err().to_string(),
        "unsupported model architecture: qwen3_5_moe config is missing text_config"
    );
}

#[test]
fn resolve_model_config_reports_supported_qwen3_5_moe() {
    let support = resolve_model_config(&json!({
        "model_type": "qwen3_5_moe",
        "image_token_id": 248056,
        "video_token_id": 248057,
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "vocab_size": 128,
            "hidden_size": 16,
            "num_hidden_layers": 4,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "max_position_embeddings": 128
        }
    }));

    assert_eq!(
        support.ok(),
        Some(ResolvedModelConfig {
            kind: ModelKind::Qwen35,
            model_type: "qwen3_5_moe".to_string(),
            effective_model_type: "qwen3_5_moe_text".to_string(),
        })
    );
}

#[test]
fn resolve_model_config_reports_supported_qwen3_next() {
    let support = resolve_model_config(&json!({
        "model_type":"qwen3_next","vocab_size":128,"hidden_size":16,
        "num_hidden_layers":4,"num_attention_heads":2,"num_key_value_heads":1,
        "head_dim":8,"max_position_embeddings":128,"intermediate_size":32,
        "moe_intermediate_size":8,"shared_expert_intermediate_size":8,
        "num_experts_per_tok":2,"num_experts":4,"tie_word_embeddings":false,
        "rope_theta":10000000,"partial_rotary_factor":0.25
    }));

    assert_eq!(
        support.ok(),
        Some(ResolvedModelConfig {
            kind: ModelKind::Qwen3Next,
            model_type: "qwen3_next".to_string(),
            effective_model_type: "qwen3_next".to_string(),
        })
    );
}

#[test]
fn resolve_model_config_reports_supported_qwen3_vl_moe() {
    let support = resolve_model_config(&json!({
        "model_type":"qwen3_vl_moe","image_token_id":30,"video_token_id":31,
        "tie_word_embeddings":false,
        "text_config":{
            "model_type":"qwen3_vl_moe_text","hidden_size":16,"num_hidden_layers":2,
            "intermediate_size":32,"num_attention_heads":2,"rms_norm_eps":0.000001,
            "vocab_size":32,"num_key_value_heads":1,"max_position_embeddings":128,
            "rope_theta":10000.0,"head_dim":8,"moe_intermediate_size":8,
            "num_experts":4,"num_experts_per_tok":2,"norm_topk_prob":true,
            "rope_scaling":{"mrope_section":[2,1,1]}
        },
        "vision_config":{
            "depth":1,"hidden_size":8,"hidden_act":"gelu_pytorch_tanh",
            "intermediate_size":16,"num_heads":2,"num_position_embeddings":16,
            "in_channels":3,"patch_size":2,"spatial_merge_size":2,
            "temporal_patch_size":2,"out_hidden_size":16,
            "deepstack_visual_indexes":[0]
        }
    }));

    assert_eq!(
        support.ok(),
        Some(ResolvedModelConfig {
            kind: ModelKind::Qwen3VlMoe,
            model_type: "qwen3_vl_moe".to_string(),
            effective_model_type: "qwen3_vl_moe_text".to_string(),
        })
    );
}

#[test]
fn resolve_model_config_reports_supported_dense_qwen3_5() {
    let support = resolve_model_config(&json!({
        "model_type": "qwen3_5",
        "image_token_id": 248056,
        "text_config": {
            "model_type": "qwen3_5_text",
            "vocab_size": 128,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 4,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "max_position_embeddings": 128
        }
    }));

    assert_eq!(
        support.ok(),
        Some(ResolvedModelConfig {
            kind: ModelKind::Qwen35,
            model_type: "qwen3_5".to_string(),
            effective_model_type: "qwen3_5_text".to_string(),
        })
    );
}

#[test]
fn resolve_model_config_reports_supported_dense_qwen3_5_text() {
    let support = resolve_model_config(&json!({
        "model_type": "qwen3_5_text",
        "vocab_size": 128,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "max_position_embeddings": 128,
        "layer_types": ["full_attention"]
    }));

    assert_eq!(
        support.ok(),
        Some(ResolvedModelConfig {
            kind: ModelKind::Qwen35,
            model_type: "qwen3_5_text".to_string(),
            effective_model_type: "qwen3_5_text".to_string(),
        })
    );
}

#[test]
fn resolve_model_config_reports_supported_qwen3_vl() {
    let support = resolve_model_config(&json!({
        "model_type": "qwen3_vl",
        "image_token_id": 151655,
        "video_token_id": 151656,
        "text_config": {
            "model_type": "qwen3_vl_text",
            "vocab_size": 151936,
            "hidden_size": 2048,
            "num_hidden_layers": 28,
            "intermediate_size": 6144,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "rms_norm_eps": 0.000001,
            "max_position_embeddings": 262144,
            "rope_theta": 5000000.0,
            "tie_word_embeddings": true,
            "rope_scaling": {
                "rope_type": "default",
                "mrope_interleaved": true,
                "mrope_section": [24, 20, 20]
            }
        },
        "vision_config": {
            "depth": 24,
            "hidden_size": 1024,
            "hidden_act": "gelu_pytorch_tanh",
            "intermediate_size": 4096,
            "num_heads": 16,
            "num_position_embeddings": 2304,
            "in_channels": 3,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "out_hidden_size": 2048,
            "deepstack_visual_indexes": [5, 11, 17]
        }
    }));
    assert_eq!(
        support.ok(),
        Some(ResolvedModelConfig {
            kind: ModelKind::Qwen3Vl,
            model_type: "qwen3_vl".to_string(),
            effective_model_type: "qwen3_vl_text".to_string(),
        })
    );
}

#[test]
fn resolve_model_config_reports_supported_nemotron_h() {
    let support = resolve_model_config(&json!({
        "model_type": "nemotron_h",
        "vocab_size": 131072,
        "hidden_size": 2688,
        "intermediate_size": 1856,
        "num_hidden_layers": 6,
        "hybrid_override_pattern": "MEMEM*",
        "num_attention_heads": 32,
        "num_key_value_heads": 2,
        "head_dim": 128,
        "max_position_embeddings": 262144,
        "mlp_hidden_act": "relu2",
        "mamba_hidden_act": "silu",
        "ssm_state_size": 128,
        "mamba_num_heads": 64,
        "mamba_head_dim": 64,
        "conv_kernel": 4,
        "chunk_size": 128,
        "n_groups": 8,
        "n_routed_experts": 128,
        "n_shared_experts": 1,
        "num_experts_per_tok": 6,
        "torch_dtype": "bfloat16"
    }));

    assert_eq!(
        support.ok(),
        Some(ResolvedModelConfig {
            kind: ModelKind::NemotronH,
            model_type: "nemotron_h".to_string(),
            effective_model_type: "nemotron_h".to_string(),
        })
    );
}

#[test]
fn resolve_model_config_reports_supported_gemma4_moe() {
    let support = resolve_model_config(&json!({
        "model_type": "gemma4",
        "text_config": {
            "model_type": "gemma4_text",
            "hidden_size": 8,
            "num_hidden_layers": 1,
            "intermediate_size": 16,
            "num_attention_heads": 2,
            "rms_norm_eps": 0.00001,
            "vocab_size": 32,
            "num_key_value_heads": 2,
            "max_position_embeddings": 128,
            "head_dim": 4,
            "enable_moe_block": true,
            "num_experts": 4,
            "top_k_experts": 2,
            "moe_intermediate_size": 8
        }
    }));

    assert_eq!(
        support.ok(),
        Some(ResolvedModelConfig {
            kind: ModelKind::Gemma4,
            model_type: "gemma4".to_string(),
            effective_model_type: "gemma4_text".to_string(),
        })
    );
}

#[test]
fn resolve_model_config_reports_supported_gemma4_unified_text() {
    let support = resolve_model_config(&json!({
        "model_type": "gemma4_unified",
        "text_config": {
            "model_type": "gemma4_unified_text",
            "hidden_size": 8,
            "num_hidden_layers": 1,
            "intermediate_size": 16,
            "num_attention_heads": 2,
            "rms_norm_eps": 0.00001,
            "vocab_size": 32,
            "num_key_value_heads": 2,
            "max_position_embeddings": 128,
            "head_dim": 4,
            "enable_moe_block": false
        }
    }));

    assert_eq!(
        support.ok(),
        Some(ResolvedModelConfig {
            kind: ModelKind::Gemma4,
            model_type: "gemma4_unified".to_string(),
            effective_model_type: "gemma4_unified_text".to_string(),
        })
    );
}

#[test]
fn resolve_model_config_reports_supported_gemma4_unified_moe() {
    let support = resolve_model_config(&json!({
        "model_type": "gemma4_unified",
        "text_config": {
            "model_type": "gemma4_unified_text",
            "hidden_size": 8,
            "num_hidden_layers": 1,
            "intermediate_size": 16,
            "num_attention_heads": 2,
            "rms_norm_eps": 0.00001,
            "vocab_size": 32,
            "num_key_value_heads": 2,
            "max_position_embeddings": 128,
            "head_dim": 4,
            "enable_moe_block": true,
            "num_experts": 4,
            "top_k_experts": 2,
            "expert_intermediate_size": 8
        }
    }));

    assert_eq!(
        support.ok(),
        Some(ResolvedModelConfig {
            kind: ModelKind::Gemma4,
            model_type: "gemma4_unified".to_string(),
            effective_model_type: "gemma4_unified_text".to_string(),
        })
    );
}
