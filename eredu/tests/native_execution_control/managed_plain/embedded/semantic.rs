//! Actual public prepared-controller sources and shared semantic output parity.
use super::*;
const CASE: &str = "managed_plain::embedded::semantic::native_original_prepared_chat_forbidden_semantics_match_ordinary_and_controlled";

fn fixture(auto_tool: bool) -> Fixture {
    let fixture = managed_fixture(super::super::super::speculative::captured_prefill::source(
        "qwen",
    ));
    let path = fixture.0.join("tokenizer.json");
    let mut tokenizer: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    let vocab = tokenizer["model"]["vocab"].as_object_mut().unwrap();
    // A real model ID represents the complete forbidden trigger. The ByteLevel
    // alphabet's Ċ is decoded as newline; the controller sees those exact bytes.
    assert_eq!(vocab.remove("i"), Some(serde_json::json!(8)));
    // The selected Auto request sees a real, complete tool token through the
    // ordinary decoder and activation worker. No callback injects output.
    let first = if auto_tool {
        "<tool_call>Ċ{\"name\":\"read\",\"arguments\":{}}Ċ</tool_call>"
    } else {
        "<tool_call>Ċ"
    };
    vocab.insert(first.into(), 8.into());
    assert_eq!(vocab.remove("?"), Some(serde_json::json!(63)));
    vocab.insert("<|im_end|>".into(), 63.into());
    tokenizer["added_tokens"] = serde_json::json!([{
        "id":63,"content":"<|im_end|>","single_word":false,"lstrip":false,
        "rstrip":false,"normalized":false,"special":true
    }]);
    std::fs::write(path, serde_json::to_vec(&tokenizer).unwrap()).unwrap();
    let config_path = fixture.0.join("config.json");
    let mut config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&config_path).unwrap()).unwrap();
    config["max_position_embeddings"] = 2048.into();
    if auto_tool {
        // The released tool prompt differs from the plain fixture prompt, so
        // its first logit rank must come from this actual numerical fixture.
        // A positive, varied residual stream and a dominant output row give
        // token 8 a stable first rank. Every layer and MTP weight retains the
        // original nonzero fixture values; no sampler or event is replaced.
        let hidden = usize::try_from(config["hidden_size"].as_u64().unwrap()).unwrap();
        let resolved = eredu_architectures::configuration::resolve_model_config(&config).unwrap();
        write_tensor_plan_with_values(
            &fixture.0,
            resolved.architecture.checkpoint(),
            |key, index| {
                if key == "model.embed_tokens.weight" {
                    Some(32.0 + (index % hidden) as f32 * 0.01)
                } else if key == "lm_head.weight" && index / hidden == 8 {
                    Some(2.0)
                } else {
                    None
                }
            },
        );
    }
    std::fs::write(config_path, serde_json::to_vec(&config).unwrap()).unwrap();
    // Released template; profile selection is the production behavioral recognizer.
    std::fs::write(
        fixture.0.join("chat_template.jinja"),
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/chat_templates/qwen3-0.6b-7e4ae267.jinja"
        )),
    )
    .unwrap();
    fixture
}
fn run(mode: &str) -> serde_json::Value {
    run_controller(mode, false)
}
fn run_controller(mode: &str, active: bool) -> serde_json::Value {
    run_request(mode, active, false)
}
fn run_request(mode: &str, active: bool, auto_tool: bool) -> serde_json::Value {
    let fixture = fixture(auto_tool);
    let execution =
        ExecutionPlan::fully_resident(eredu_core::DevicePlan::new("mlx", "metal:0").unwrap())
            .with_required_session_capabilities(SessionCapabilities::new(true, true, true))
            .with_drafting(DraftingPlan::Embedded {
                max_draft_tokens: 2,
                lookahead: false,
                adaptive_lookahead: false,
            });
    let mut loaded =
        LoadedModel::load_execution_plan(&MlxBackendFactory::default(), &fixture.0, &execution)
            .unwrap();
    let options = loaded.speculative_generation_options().unwrap().unwrap();
    let (model, drafting) = loaded.parts_mut();
    let chat = model.source_chat_with_capacity(ChatTemplateRequest {
        messages: vec![serde_json::json!({"role":"user","content":"abcde"})],
        // No tool surface selects the actual active grammar in the shared
        // production plan; ToolChoice::None with a surface selects forbidden.
        tools: if active { Vec::new() } else { vec![serde_json::json!({"type":"function","function":{"name":"read", "parameters":{"type":"object","properties":{},"additionalProperties":false}}})] },
        tool_choice: if auto_tool { ToolChoice::Auto } else { ToolChoice::None }, add_generation_prompt: true,
        ..Default::default()
    }, 8 * 1024 * 1024 * 1024).unwrap();
    assert!(matches!(
        chat.semantic_support(),
        eredu::runtime::chat::SemanticSupport::Supported
    ));
    let mut settings = settings(0.0);
    settings.overrides.max_new_tokens = Some(6);
    settings.inference.prefill_chunk_positions = std::num::NonZeroU64::new(64);
    let mut events = Vec::new();
    let mut committed = Vec::new();
    let request = PreparedChatSpeculativeRequest {
        chat: &chat,
        input: PreparedChatPrompt::Rendered,
        output_mode: PreparedChatOutputMode::Semantic,
        skip_special_tokens: true,
        drafting: drafting.as_speculative_draft().unwrap(),
        settings: chat_settings(&chat, settings.clone()),
        options,
        caller_stop_sequences: &[],
        cancellation: Default::default(),
        on_event: |event| events.push(event),
    };
    let output = if mode == "controlled" {
        model
            .with_controlled_prepared_chat_speculative(
                request,
                ControlledSpeculativeOptions::default(),
                |session| {
                    while let Some(step) = session.step()? {
                        committed.extend(step.committed_token_ids);
                    }
                    assert_eq!(committed, session.token_ids());
                    Ok(())
                },
            )
            .unwrap_or_else(report_failure)
    } else {
        assert_eq!(mode, "managed");
        model
            .generate_prepared_chat_speculative(request)
            .unwrap_or_else(report_failure)
    };
    assert!(!output.token_ids().is_empty());
    if auto_tool {
        assert_eq!(
            output.token_ids().first(),
            Some(&8),
            "actual Auto trigger must be sampled first; ids={:?}, events={events:?}",
            output.token_ids()
        );
        assert!(events.iter().any(|event| matches!(event, SemanticEvent::ToolCallStart { index:0, name, .. } if name.as_str() == "read")));
        let arguments = events
            .iter()
            .filter_map(|event| match event {
                SemanticEvent::ToolArgumentsDelta {
                    index: 0,
                    json_fragment,
                } => Some(json_fragment.as_str()),
                _ => None,
            })
            .collect::<String>();
        assert_eq!(arguments, "{}");
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, SemanticEvent::ToolCallEnd))
                .count(),
            1,
            "the shared tool callback must validate and complete exactly one call"
        );
    } else {
        assert!(
            !output.token_ids().contains(&8),
            "actual forbidden trigger or absent-tool grammar ID must be filtered"
        );
    }
    if auto_tool {
        // The complete tool token terminates at the shared prefill commit,
        // before any speculative verification round can be submitted.
        assert_eq!(output.token_ids(), &[8]);
        assert_eq!(output.stats().rounds(), 0);
        assert!(
            matches!(
                output.finish_reason(),
                eredu_core::FinishReason::GrammarComplete | eredu_core::FinishReason::StopSequence
            ),
            "atomic tool completion must be terminal: {:?}",
            output.finish_reason()
        );
    } else {
        assert!(output.stats().rounds() > 0);
    }
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, SemanticEvent::Finished { .. }))
            .count(),
        1
    );
    if !auto_tool {
        assert!(events
            .iter()
            .any(|event| matches!(event, SemanticEvent::TextDelta(text) if !text.is_empty())));
    }
    if mode == "controlled" {
        assert_eq!(committed, output.token_ids());
    }
    let pointer = output.token_ids().as_ptr();
    drop((chat, loaded, fixture));
    assert_eq!(output.token_ids().as_ptr(), pointer);
    // Both the native output and immutable semantic payloads escape all sources.
    serde_json::json!({"ids":output.token_ids(),"events":events,"finish":output.finish_reason()})
}
#[test]
#[ignore = "requires an accessible Metal device"]
fn native_original_prepared_chat_forbidden_semantics_match_ordinary_and_controlled() {
    super::parity_with(
        "semantic-forbidden",
        CASE,
        &["managed", "controlled"],
        |_, mode| run(mode),
    );
}

#[test]
#[ignore = "requires an accessible Metal device"]
fn native_original_prepared_chat_active_semantics_match_ordinary_and_controlled() {
    const ACTIVE_CASE: &str = "managed_plain::embedded::semantic::native_original_prepared_chat_active_semantics_match_ordinary_and_controlled";
    super::parity_with(
        "semantic-active",
        ACTIVE_CASE,
        &["managed", "controlled"],
        |_, mode| run_controller(mode, true),
    );
}

#[test]
#[ignore = "requires an accessible Metal device"]
fn native_original_prepared_chat_auto_tool_semantics_match_ordinary_and_controlled() {
    const AUTO_CASE: &str = "managed_plain::embedded::semantic::native_original_prepared_chat_auto_tool_semantics_match_ordinary_and_controlled";
    super::parity_with(
        "semantic-auto-tool",
        AUTO_CASE,
        &["managed", "controlled"],
        |_, mode| run_request(mode, false, true),
    );
}
