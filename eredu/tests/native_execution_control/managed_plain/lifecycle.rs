//! Public composition at the two new boundaries: chat rendering and admitted reset.
use super::*;

#[test]
#[ignore = "requires an accessible Metal device and complete managed native qualification"]
fn native_managed_reset_reuses_model_without_refunding_retained_output() {
    reset_fixture(fixture(false));
}

#[test]
#[ignore = "requires an accessible Metal device and complete managed native qualification"]
fn native_managed_hybrid_reset_reuses_all_state_tables() {
    reset_fixture(super::super::components::qwen_35_fixture());
}

fn reset_fixture(fixture: Fixture) {
    reset_fixture_with_state(fixture, eredu_runtime::CacheResidencyPolicy::Device)
}

pub(super) fn reset_fixture_with_state(
    fixture: Fixture,
    state: eredu_runtime::CacheResidencyPolicy,
) {
    let root = managed_fixture(fixture);
    let execution =
        ExecutionPlan::fully_resident(eredu_core::DevicePlan::new("mlx", "metal:0").unwrap());
    let (mut model, _) = LoadedModel::load_execution_plan(
        &MlxBackendFactory::default().with_state_residency(state),
        &root.0,
        &execution,
    )
    .unwrap()
    .into_parts();
    let source = model
        .compile_managed_plain_text_source(
            std::fs::File::open(root.0.join("tokenizer.json")).unwrap(),
        )
        .unwrap();
    let cancellation = GenerationCancellationToken::new();
    let request = ManagedPlainTextRequest::new(PROMPT, settings(0.0));
    let first = model
        .generate_managed_plain_text(&source, request, &cancellation, &mut |_| {})
        .unwrap_or_else(report_failure)
        .unwrap();
    let address = first.text.as_str().as_ptr();

    // A refused reset must leave both the old state and its output custody valid.
    let mut short = eredu_core::SessionResetLimits::new(8 * 1024 * 1024 * 1024);
    short.application_memory_budget_bytes = Some(1);
    let error = model
        .prepare_reset_ordinary()
        .unwrap_or_else(report_failure)
        .reset_admitted(short)
        .unwrap_err();
    assert!(
        std::iter::successors(Some(&error as &(dyn Error + 'static)), |e| (*e).source()).any(
            |e| matches!(
                e.downcast_ref::<eredu_core::SessionResetRejection>(),
                Some(eredu_core::SessionResetRejection::ApplicationBudgetExceeded { .. })
            )
        ),
        "{error}"
    );
    drop(error);
    model
        .prepare_reset_ordinary()
        .unwrap_or_else(report_failure)
        .reset_admitted(eredu_core::SessionResetLimits::new(8 * 1024 * 1024 * 1024))
        .unwrap_or_else(report_failure);
    let second = model
        .generate_managed_plain_text(&source, request, &cancellation, &mut |_| {})
        .unwrap_or_else(report_failure)
        .unwrap();
    assert_eq!(first.finish_reason, FinishReason::MaxTokens);
    assert_eq!(second.finish_reason, first.finish_reason);
    assert_eq!(second.token_ids.as_ref(), first.token_ids.as_ref());
    assert_eq!(second.text.as_str(), first.text.as_str());
    drop((source, model));
    assert_eq!(first.text.as_str().as_ptr(), address);
    assert_eq!(second.text.as_str(), first.text.as_str());
    println!(
        "PUBLIC_MANAGED_RESET_RESULT:{}",
        serde_json::json!({
            "ids": second.token_ids.as_ref(), "text": second.text.as_str(),
            "retained_outputs": 2
        })
    );
    // Physical cache counters can reach zero before nested host publications
    // release their request custody. Retire the escaped outputs on this owner
    // thread and prove that no old request still excludes a fresh model load.
    drop((first, second));
    eredu_backend_mlx::backend::nn::shared::MlxNeuralBackend::reclaim_retired_resources();
    let fresh = LoadedModel::load_execution_plan(
        &MlxBackendFactory::default(),
        &root.0,
        &execution,
    )
    .unwrap_or_else(report_failure);
    drop(fresh);
    eredu_backend_mlx::backend::nn::shared::MlxNeuralBackend::reclaim_retired_resources();
}

#[test]
#[ignore = "requires an accessible Metal device and complete managed native qualification"]
fn native_managed_chat_matches_rendered_ordinary_and_controlled_generation() {
    let template_config: serde_json::Value = serde_json::from_str(include_str!(
        "../../../src/api/portable/original_token_input/tests/chat_fixtures/tokenizer_config.json"
    ))
    .unwrap();
    chat_fixture(
        "managed_plain::lifecycle::native_managed_chat_matches_rendered_ordinary_and_controlled_generation",
        "EREDU_PUBLIC_MANAGED_CHAT_MODE",
        template_config["chat_template"].as_str().unwrap(),
        "<|im_start|>system\nabcde<|im_end|>\n<|im_start|>assistant\n",
    );
}

#[test]
#[ignore = "requires an accessible Metal device and complete managed native qualification"]
fn native_managed_source_chat_matches_ordinary_and_controlled_generation() {
    chat_fixture(
        "managed_plain::lifecycle::native_managed_source_chat_matches_ordinary_and_controlled_generation",
        "EREDU_PUBLIC_MANAGED_SOURCE_CHAT_MODE",
        "{% for turn in messages %}{{ turn.content }}{% endfor %}{% if add_generation_prompt %} assistant:{% endif %}",
        "abcde assistant:",
    );
}

fn chat_fixture(case: &str, mode_variable: &str, template: &str, rendered: &str) {
    const MARKER: &str = "PUBLIC_MANAGED_CHAT_RESULT:";
    let Ok(mode) = std::env::var(mode_variable) else {
        let mut expected = None;
        for mode in ["managed", "ordinary", "controlled"] {
            let result = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", case, "--ignored", "--nocapture"])
                .env(mode_variable, mode)
                .output()
                .unwrap();
            let stdout = String::from_utf8_lossy(&result.stdout);
            assert!(
                result.status.success(),
                "{mode}: {stdout}\n{}",
                String::from_utf8_lossy(&result.stderr)
            );
            let actual: serde_json::Value = serde_json::from_str(
                stdout
                    .lines()
                    .find_map(|line| line.strip_prefix(MARKER))
                    .unwrap_or_else(|| panic!("missing execution marker: {mode}: {stdout}")),
            )
            .unwrap();
            if let Some(expected) = &expected {
                assert_eq!(&actual, expected, "{mode}");
            } else {
                expected = Some(actual);
            }
        }
        return;
    };
    let alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ<>|_ \n!?:.01";
    assert_eq!(alphabet.chars().count(), 64);
    let root = managed_fixture_with_vocabulary(fixture(false), alphabet);
    std::fs::write(root.0.join("chat_template.jinja"), template).unwrap();
    std::fs::write(
        root.0.join("tokenizer_config.json"),
        serde_json::to_vec(&serde_json::json!({"chat_template": template})).unwrap(),
    )
    .unwrap();
    let execution =
        ExecutionPlan::fully_resident(eredu_core::DevicePlan::new("mlx", "metal:0").unwrap());
    let (mut model, _) =
        LoadedModel::load_execution_plan(&MlxBackendFactory::default(), &root.0, &execution)
            .unwrap()
            .into_parts();
    let chat = ChatTemplateRequest {
        messages: vec![serde_json::json!({"role":"system", "content":"abcde"})],
        tool_choice: ToolChoice::None,
        add_generation_prompt: true,
        ..Default::default()
    };
    let mut settings = settings(0.0);
    settings.inference.prefill_chunk_positions = NonZeroU64::new(8);
    if mode == "ordinary" {
        let mut oracle = eredu_text::tokenizer::Tokenizer::from_bytes(
            &std::fs::read(root.0.join("tokenizer.json")).unwrap(),
        )
        .unwrap();
        let actual = oracle
            .apply_chat_template_json(
                eredu_text::tokenizer::ModelChatTemplate::Single(template.into()),
                [chat.messages.clone()],
                None,
                "fixture",
                true,
                None,
            )
            .unwrap()
            .pop()
            .unwrap();
        assert_eq!(actual, rendered);
        let prompt = model.encode(&actual, false).unwrap();
        assert_eq!(prompt.len(), rendered.len());
        let config = model.resolve_generation_config(settings.overrides).unwrap();
        let ids: Vec<u32> = model
            .generate_tokens(
                prompt,
                TextGenerationConfig::new(config).with_seed(settings.seed),
            )
            .unwrap()
            .map(|token| token.unwrap().token_id().unwrap())
            .collect();
        let text = model.decode(&ids, true).unwrap();
        println!("{MARKER}{}", serde_json::json!({"ids": ids, "text": text}));
        return;
    }
    assert!(matches!(mode.as_str(), "managed" | "controlled"));
    let tokenizer = model
        .compile_managed_plain_text_source(
            std::fs::File::open(root.0.join("tokenizer.json")).unwrap(),
        )
        .unwrap();
    let cancellation = GenerationCancellationToken::new();
    let source = model
        .compile_managed_chat_source(
            &tokenizer,
            std::fs::File::open(root.0.join("tokenizer_config.json")).unwrap(),
            false,
            &cancellation,
        )
        .unwrap_or_else(report_failure)
        .unwrap();
    let cancelled = GenerationCancellationToken::new();
    cancelled.cancel();
    assert!(
        model
            .prepare_chat(
                &source,
                &chat,
                settings.inference.managed_memory_capacity_bytes.unwrap(),
                &cancelled
            )
            .unwrap()
            .is_none()
    );
    let error = model
        .prepare_chat(&source, &chat, 1, &cancellation)
        .unwrap_err();
    assert!(budget_failure(&error), "{error}");
    drop(error);
    let prepared = model
        .prepare_chat(
            &source,
            &chat,
            settings.inference.managed_memory_capacity_bytes.unwrap(),
            &cancellation,
        )
        .unwrap_or_else(report_failure)
        .unwrap();
    assert_eq!(prepared.rendered_prompt(), rendered);
    let mut request = PreparedChatRequest::new(&prepared, settings);
    request.output_mode = PreparedChatOutputMode::Text;
    let mut visible = String::new();
    let mut finishes = Vec::new();
    let mut emit = |event: SemanticEvent| match event {
        SemanticEvent::TextDelta(text) => visible.push_str(&text),
        SemanticEvent::Finished { reason } => finishes.push(reason),
        _ => panic!("literal output produced a semantic protocol event"),
    };
    let mut session = model
        .start_prepared_chat(request, &cancellation)
        .unwrap_or_else(report_failure)
        .unwrap();
    let output = if mode == "controlled" {
        while session.finish_reason().is_none() {
            session = session
                .advance(&cancellation, &mut emit)
                .unwrap_or_else(report_failure);
        }
        session
            .into_output()
            .unwrap_or_else(|_| panic!("terminal chat session"))
    } else {
        session
            .run(&cancellation, &mut emit)
            .unwrap_or_else(report_failure)
    };
    assert_eq!(output.finish_reason, FinishReason::MaxTokens);
    assert_eq!(finishes, [FinishReason::MaxTokens]);
    assert_eq!(output.token_ids.as_ref().len(), 4);
    assert_eq!(
        model.decode(output.token_ids.as_ref(), true).unwrap(),
        visible
    );
    drop((prepared, source, tokenizer, model));
    println!(
        "{MARKER}{}",
        serde_json::json!({
            "ids": output.token_ids.as_ref(), "text": visible
        })
    );
}
