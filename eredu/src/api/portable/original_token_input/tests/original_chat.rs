use super::super::chat::compile_original_chat_file;
use super::*;
use crate::memory_fixture::{LedgerFixture as _, StorageFixture as _};
use crate::runtime::chat::ChatTemplateRequest;
use eredu_core::generation::SemanticEvent;
use eredu_runtime::working_memory::OriginalTextSourceBudget;
use eredu_text::chat_storage::{ChatRenderContext, ChatTemplatePlan};
use std::io::Write as _;

impl<M: Clone + 'static> OriginalChatBackend for Backend<M> {
    fn compile_original_forbidden_source(
        runtime: &ModelRuntime<Self>,
        plan: eredu_core::speculative::PreparedForbiddenInputCopy<'_>,
    ) -> Result<OriginalForbiddenSource, OriginalForbiddenSourceError> {
        runtime.backend().pool.compile_forbidden_source(plan)
    }
    fn compile_original_forbidden_tokenizer_source(
        runtime: &ModelRuntime<Self>,
        tokenizer: &OriginalTokenizer,
        trigger: &[u8],
    ) -> Result<OriginalForbiddenSource, OriginalForbiddenSourceError> {
        runtime
            .backend()
            .pool
            .compile_forbidden_tokenizer_source(tokenizer, trigger)
    }
    fn prepare_original_chat_profile(
        runtime: &ModelRuntime<Self>,
        template: &OriginalChatTemplate,
        tokenizer: &OriginalTokenizer,
        capacity: &MemoryLimitDeclarations,
    ) -> Result<OriginalChatProfilePreparation, OriginalChatProfileError> {
        OriginalChatProfilePreparation::new(
            template,
            tokenizer,
            &runtime.backend().execution,
            capacity.resolve(runtime.backend().pool.topology()).unwrap(),
        )
    }

    fn validate_original_chat_sources(
        runtime: &ModelRuntime<Self>,
        template: &OriginalChatTemplate,
        tokenizer: &OriginalTokenizer,
    ) -> Result<(), TokenInputRejection> {
        template
            .validate_pool(&runtime.backend().pool)
            .map_err(|_| TokenInputRejection::IdentityMismatch)?;
        tokenizer
            .validate_pool(&runtime.backend().pool)
            .map_err(|_| TokenInputRejection::IdentityMismatch)
    }
    fn validate_original_chat_render(
        runtime: &ModelRuntime<Self>,
        render: &OriginalRenderedChat,
    ) -> Result<(), TokenInputRejection> {
        render
            .validate_pool(&runtime.backend().pool)
            .map_err(|_| TokenInputRejection::IdentityMismatch)
    }
    fn compile_original_chat_template(
        runtime: &ModelRuntime<Self>,
        plan: ChatTemplatePlan<'_>,
    ) -> Result<OriginalChatTemplate, OriginalChatSourceError> {
        runtime.backend().facts.borrow_mut().chat_sources += 1;
        runtime
            .backend()
            .pool
            .compile_chat_template(plan)
            .map_err(Into::into)
    }
    fn compile_original_chat_template_file(
        runtime: &ModelRuntime<Self>,
        read: eredu_checkpoint::artifact::PreparedArtifactFileRead,
        model_id: &str,
        has_tools: bool,
    ) -> Result<OriginalChatTemplate, OriginalChatSourceError> {
        runtime.backend().facts.borrow_mut().chat_sources += 1;
        runtime
            .backend()
            .pool
            .compile_chat_template_file(read, model_id, has_tools)
            .map_err(Into::into)
    }
    fn render_original_chat(
        runtime: &ModelRuntime<Self>,
        template: &OriginalChatTemplate,
        tokenizer: &OriginalTokenizer,
        context: ChatRenderContext<'_>,
    ) -> Result<OriginalRenderedChat, OriginalChatRenderOperationError> {
        runtime.backend().facts.borrow_mut().chat_renders += 1;
        let rendered = runtime
            .backend()
            .pool
            .render_original_chat(template, tokenizer, context)?;
        if let Some(cancel) = &runtime.backend().facts.borrow().cancel_after_render {
            cancel.cancel();
        }
        Ok(rendered)
    }
}

const CHAT_CONFIG: &str = include_str!("chat_fixtures/tokenizer_config.json");
// Actual small HF model with an explicit unknown token. The released full-file
// oracle uses its complete vocabulary; this neutral driver fixture preserves
// the existing h / Ġhi predictions while exercising full rendered prompt bytes.
const TOKENIZER: &str = include_str!("chat_fixtures/tokenizer.json");
fn source_budget_bytes() -> u64 {
    OriginalTextSourceBudget::storage_bytes().unwrap()
}
fn request() -> ChatTemplateRequest {
    ChatTemplateRequest {
        messages: vec![
            serde_json::json!({"role":"system","content":"hi"}),
            serde_json::json!({"role":"user","content":"hi 世界"}),
        ],
        add_generation_prompt: true,
        ..ChatTemplateRequest::default()
    }
}
fn loaded_sources(
    runtime: ModelRuntime<Backend>,
    tokenizer: &[u8],
    config: &str,
    model_id: &str,
    family: ModelKind,
    cancellation: &GenerationCancellationToken,
) -> (
    LoadedModel<Backend>,
    crate::api::ManagedPlainTextSource,
    crate::api::ManagedChatSource,
) {
    let selected = eredu_text::tokenizer::load_model_chat_template_from_str(config)
        .unwrap()
        .unwrap();
    let model = LoadedModel::from_runtime(
        runtime,
        ChatTokenizer::from_bytes(tokenizer).unwrap(),
        LoadedTextModelConfig {
            model_family: family,
            effective_model_type: model_id.into(),
            model_id: model_id.into(),
            chat_template: Some(selected),
            eos_token_ids: vec![],
            checkpoint_generation_config: None,
        },
    )
    .unwrap();
    let mut file = tempfile::tempfile().unwrap();
    file.write_all(tokenizer).unwrap();
    let tokenizer = model.compile_managed_plain_text_source(file).unwrap();
    let mut file = tempfile::tempfile().unwrap();
    file.write_all(config.as_bytes()).unwrap();
    let source = model
        .compile_managed_chat_source(&tokenizer, file, false, cancellation)
        .unwrap()
        .unwrap();
    (model, tokenizer, source)
}
fn chat_settings() -> crate::api::PreparedChatGenerationSettings {
    crate::api::PreparedChatGenerationSettings {
        overrides: GenerationConfigOverrides {
            max_new_tokens: Some(3),
            ..Default::default()
        },
        inference: TextInferencePolicy {
            memory_limits: eredu_core::MemoryLimitDeclarations::new([(
                "host".into(),
                eredu_core::MemoryLimit::Finite(u64::MAX),
            )]),
            ..Default::default()
        },
        ..Default::default()
    }
}
fn literal_request<P>(
    prepared: &crate::runtime::chat::PreparedChat,
    settings: crate::api::PreparedChatGenerationSettings,
) -> crate::api::PreparedChatRequest<'_, P> {
    let mut request = crate::api::PreparedChatRequest::new(prepared, settings);
    request.output_mode = crate::api::PreparedChatOutputMode::Text;
    request
}

#[test]
fn public_chat_file_render_encode_and_shared_cursor_preserve_cached_outputs() {
    for manual in [false, true] {
        let (runtime, facts, pool) = bare_runtime_with_capacity(u64::MAX);
        facts.borrow_mut().shared_admission_capacity = Some(u64::MAX);
        let cancellation = GenerationCancellationToken::new();
        let (mut model, tokenizer, source) = loaded_sources(
            runtime,
            TOKENIZER.as_bytes(),
            CHAT_CONFIG,
            "smol",
            ModelKind::Llama,
            &cancellation,
        );
        let cold = pool.live_charge_bytes().unwrap();
        let reference = tokenizers::Tokenizer::from_bytes(TOKENIZER.as_bytes()).unwrap();
        let mut ordinary = ChatTokenizer::from_tokenizer(reference.clone());
        let selected = eredu_text::tokenizer::load_model_chat_template_from_str(CHAT_CONFIG)
            .unwrap()
            .unwrap();
        let mut previous = None;
        for generation in [false, true] {
            let mut chat = request();
            chat.add_generation_prompt = generation;
            let expected = ordinary
                .apply_chat_template_json(
                    selected.clone(),
                    [chat.messages.clone()],
                    None,
                    "smol",
                    generation,
                    None,
                )
                .unwrap()
                .pop()
                .unwrap();
            let expected_ids = reference.encode(expected.as_str(), false).unwrap();
            let prepared = model
                .prepare_chat(
                    &source,
                    &chat,
                    &crate::memory_fixture::limits(u64::MAX),
                    &cancellation,
                )
                .unwrap()
                .unwrap();
            assert_eq!(prepared.rendered_prompt(), expected);
            let retained_render = prepared.render().clone();
            facts.borrow_mut().ids.clear();
            let mut session = model
                .start_prepared_chat(literal_request(&prepared, chat_settings()), &cancellation)
                .unwrap()
                .unwrap();
            assert_eq!(facts.borrow().ids, expected_ids.get_ids());
            let mut visible = String::new();
            let mut event = |event: SemanticEvent| {
                if let SemanticEvent::TextDelta(text) = event {
                    visible.push_str(&text);
                }
            };
            let output = if manual {
                while session.finish_reason().is_none() {
                    session = session.advance(&cancellation, &mut event).unwrap();
                }
                session
                    .into_output()
                    .unwrap_or_else(|_| panic!("terminal chat"))
            } else {
                session.run(&cancellation, &mut event).unwrap()
            };
            assert_eq!(visible, "h hih");
            assert_eq!(output.token_ids.as_ref(), &[0, 8, 0]);
            let ids = output.token_ids.clone();
            drop(output);
            drop(prepared);
            assert_eq!(retained_render.prompt(generation), expected);
            let with_alias = pool.live_charge_bytes().unwrap();
            drop(retained_render);
            assert!(
                pool.live_charge_bytes().unwrap() < with_alias,
                "render and execution token ownership retire independently"
            );
            assert!(pool.live_charge_bytes().unwrap() > cold);
            if let Some(previous) = previous.replace(ids) {
                assert_eq!(previous.as_ref(), &[0, 8, 0]);
            }
        }
        drop((source, tokenizer, model));
        assert!(pool.live_charge_bytes().unwrap() > 0);
        assert_eq!(previous.as_ref().unwrap().as_ref(), &[0, 8, 0]);
        drop(previous);
        assert_eq!(pool.live_charge_bytes().unwrap(), 0);
    }
}

#[test]
fn literal_chat_preserves_special_spellings_and_caller_stop_output() {
    for skip in [false, true] {
        for stop in [false, true] {
            for manual in [false, true] {
                let (runtime, facts, pool) = bare_runtime_with_capacity(u64::MAX);
                facts.borrow_mut().shared_admission_capacity = Some(u64::MAX);
                facts.borrow_mut().prediction_ids = Some([0, 4, 8]);
                let cancellation = GenerationCancellationToken::new();
                let (mut model, tokenizer, source) = loaded_sources(
                    runtime,
                    TOKENIZER.as_bytes(),
                    CHAT_CONFIG,
                    "smol",
                    ModelKind::Llama,
                    &cancellation,
                );
                let prepared = model
                    .prepare_chat(
                        &source,
                        &request(),
                        &crate::memory_fixture::limits(u64::MAX),
                        &cancellation,
                    )
                    .unwrap()
                    .unwrap();
                let stops = ["hi".to_owned()];
                let mut invocation = literal_request(&prepared, chat_settings());
                invocation.skip_special_tokens = skip;
                if stop {
                    invocation.stop_sequences = &stops;
                }
                let mut session = model
                    .start_prepared_chat(invocation, &cancellation)
                    .unwrap()
                    .unwrap();
                let mut visible = String::new();
                let mut finishes = Vec::new();
                let mut emit = |event| match event {
                    SemanticEvent::TextDelta(text) => visible.push_str(&text),
                    SemanticEvent::Finished { reason } => finishes.push(reason),
                    _ => panic!("literal policy must not publish protocol events"),
                };
                let output = if manual {
                    while session.finish_reason().is_none() {
                        session = session.advance(&cancellation, &mut emit).unwrap();
                    }
                    session
                        .into_output()
                        .unwrap_or_else(|_| panic!("terminal literal chat"))
                } else {
                    session.run(&cancellation, &mut emit).unwrap()
                };
                let raw = tokenizers::Tokenizer::from_bytes(TOKENIZER.as_bytes())
                    .unwrap()
                    .decode(&[0, 4, 8], skip)
                    .unwrap();
                let expected = if stop {
                    raw.split_once("hi").unwrap().0
                } else {
                    &raw
                };
                assert_eq!(
                    visible, expected,
                    "skip={skip}, stop={stop}, manual={manual}"
                );
                assert_eq!(output.token_ids.as_ref(), &[0, 4, 8]);
                let reason = if stop {
                    FinishReason::StopSequence
                } else {
                    FinishReason::MaxTokens
                };
                assert_eq!(output.finish_reason, reason);
                assert_eq!(finishes, [reason]);
                drop((output, prepared, source, tokenizer, model));
                assert_eq!(pool.live_charge_bytes().unwrap(), 0);
            }
        }
    }
}

#[test]
fn public_chat_cancellation_and_late_startup_failures_keep_exact_owner_lifetimes() {
    let (runtime, facts, pool) = bare_runtime_with_capacity(u64::MAX);
    let cancelled = GenerationCancellationToken::new();
    cancelled.cancel();
    assert!(compile_original_chat_file(
        &runtime,
        tempfile::tempfile().unwrap(),
        "smol",
        false,
        &cancelled
    )
    .unwrap()
    .is_none());
    assert_eq!(facts.borrow().chat_sources, 0);
    assert_eq!(pool.live_charge_bytes().unwrap(), 0);
    drop(runtime);
    for phase in ["render", "stop", "encode", "short", "changed_capacity"] {
        let (runtime, facts, pool) = bare_runtime_with_capacity(u64::MAX);
        facts.borrow_mut().shared_admission_capacity = Some(u64::MAX);
        let cancellation = GenerationCancellationToken::new();
        let (mut model, tokenizer, source) = loaded_sources(
            runtime,
            TOKENIZER.as_bytes(),
            CHAT_CONFIG,
            "smol",
            ModelKind::Llama,
            &cancellation,
        );
        let cold = pool.live_charge_bytes().unwrap();
        if phase == "render" {
            facts.borrow_mut().cancel_after_render = Some(cancellation.clone());
        }
        let prepared = model
            .prepare_chat(
                &source,
                &request(),
                &crate::memory_fixture::limits(u64::MAX),
                &cancellation,
            )
            .unwrap();
        if phase == "render" {
            assert!(prepared.is_none());
            assert_eq!(pool.live_charge_bytes().unwrap(), cold);
            assert_eq!(facts.borrow().stops, 0);
            assert_eq!(facts.borrow().encodes, 0);
            continue;
        }
        let prepared = prepared.unwrap();
        let prepared_bytes = pool.live_charge_bytes().unwrap();
        if phase == "stop" {
            facts.borrow_mut().cancel_after_stops = Some(cancellation.clone());
        }
        if phase == "encode" {
            facts.borrow_mut().cancel_after_encode = Some(cancellation.clone());
        }
        if phase == "short" {
            facts.borrow_mut().short = true;
        }
        let mut settings = chat_settings();
        if phase == "changed_capacity" {
            settings.inference.memory_limits = eredu_core::MemoryLimitDeclarations::new([(
                "host".into(),
                eredu_core::MemoryLimit::Finite(u64::MAX - 1),
            )]);
        }
        let stops = ["hi".to_owned()];
        let mut invocation = literal_request(&prepared, settings);
        invocation.stop_sequences = &stops;
        let result = model.start_prepared_chat(invocation, &cancellation);
        match phase {
            "stop" | "encode" => {
                assert!(result.unwrap().is_none());
                assert_eq!(facts.borrow().stops, 1);
                assert_eq!(facts.borrow().encodes, usize::from(phase == "encode"));
                assert_eq!(pool.live_charge_bytes().unwrap(), prepared_bytes);
            }
            "short" | "changed_capacity" => {
                let error = match result {
                    Err(error) => error,
                    Ok(_) => panic!("expected startup rejection"),
                };
                if phase == "changed_capacity" {
                    assert_eq!(
                        error.input_rejection(),
                        Some(TokenInputRejection::IdentityMismatch)
                    );
                    assert_eq!(facts.borrow().stops, 0);
                    assert_eq!(facts.borrow().encodes, 0);
                    assert_eq!(pool.live_charge_bytes().unwrap(), prepared_bytes);
                } else {
                    assert_eq!(facts.borrow().stops, 1);
                    assert_eq!(facts.borrow().encodes, 1);
                    assert!(pool.live_charge_bytes().unwrap() > prepared_bytes);
                }
                assert!(!facts.borrow().order.contains(&"submit"));
                drop(error);
                assert_eq!(pool.live_charge_bytes().unwrap(), prepared_bytes);
            }
            _ => unreachable!(),
        }
        drop(prepared);
        assert_eq!(pool.live_charge_bytes().unwrap(), cold);
        drop((source, tokenizer, model));
        assert_eq!(pool.live_charge_bytes().unwrap(), 0);
    }
}

#[test]
fn invalid_chat_eos_refuses_before_native_submission() {
    let (runtime, facts, pool) = bare_runtime_with_capacity(u64::MAX);
    facts.borrow_mut().shared_admission_capacity = Some(u64::MAX);
    let cancellation = GenerationCancellationToken::new();
    let (mut model, tokenizer, source) = loaded_sources(
        runtime,
        TOKENIZER.as_bytes(),
        CHAT_CONFIG,
        "smol",
        ModelKind::Llama,
        &cancellation,
    );
    let cold = pool.live_charge_bytes().unwrap();
    model.eos_token_ids = vec![u32::MAX];
    match model.prepare_chat(
        &source,
        &request(),
        &crate::memory_fixture::limits(u64::MAX),
        &cancellation,
    ) {
        Err(error) => drop(error),
        Ok(Some(prepared)) => {
            let error = match model
                .start_prepared_chat(literal_request(&prepared, chat_settings()), &cancellation)
            {
                Err(error) => error,
                Ok(_) => panic!("foreign EOS must reject before native submission"),
            };
            drop((error, prepared));
        }
        Ok(None) => panic!("request was not cancelled"),
    }
    assert!(!facts.borrow().order.contains(&"submit"));
    assert_eq!(pool.live_charge_bytes().unwrap(), cold);
    drop((source, tokenizer, model));
    assert_eq!(pool.live_charge_bytes().unwrap(), 0);
}

#[test]
fn chat_caller_ceiling_precedes_render_and_survives_preparation_failure() {
    let (runtime, facts, pool) = bare_runtime_with_capacity(u64::MAX);
    let cancellation = GenerationCancellationToken::new();
    let (model, tokenizer, source) = loaded_sources(
        runtime,
        TOKENIZER.as_bytes(),
        CHAT_CONFIG,
        "smol",
        ModelKind::Llama,
        &cancellation,
    );
    let cold = pool.live_charge_bytes().unwrap();
    for capacity in [1, cold + source_budget_bytes()] {
        let error = model
            .prepare_chat(
                &source,
                &request(),
                &crate::memory_fixture::limits(capacity),
                &cancellation,
            )
            .unwrap_err();
        assert_eq!(facts.borrow().chat_renders, 0);
        assert_eq!(facts.borrow().stops, 0);
        assert_eq!(facts.borrow().encodes, 0);
        assert!(facts.borrow().order.is_empty());
        if capacity != 1 {
            assert_eq!(pool.live_charge_bytes().unwrap(), capacity);
            assert!(Backend::prepare_original_text_source_budget(
                &model.runtime,
                tokenizer.original(),
                &crate::memory_fixture::limits(u64::MAX)
            )
            .is_err());
        } else {
            assert_eq!(pool.live_charge_bytes().unwrap(), cold);
        }
        drop(error);
        assert_eq!(pool.live_charge_bytes().unwrap(), cold);
    }
    drop((source, tokenizer, model));
    assert_eq!(pool.live_charge_bytes().unwrap(), 0);
}

#[test]
fn invalid_chat_request_and_equal_content_foreign_sources_reject_before_execution() {
    let (runtime, facts, pool) = bare_runtime_with_capacity(u64::MAX);
    let cancellation = GenerationCancellationToken::new();
    let (mut model, tokenizer, source) = loaded_sources(
        runtime,
        TOKENIZER.as_bytes(),
        CHAT_CONFIG,
        "smol",
        ModelKind::Llama,
        &cancellation,
    );
    let cold = pool.live_charge_bytes().unwrap();
    for case in [0, 1, 2, 6, 7] {
        let mut input = request();
        match case {
            0 => input.tool_choice = crate::runtime::chat::ToolChoice::Required,
            1 => {
                input.tool_choice = crate::runtime::chat::ToolChoice::None;
                input.tools.push(serde_json::json!({"name":"tool"}));
            }
            2 => input.enable_thinking = Some(true),
            6 => {
                input
                    .extra_template_kwargs
                    .insert("messages".into(), serde_json::json!([null]));
            }
            7 => input.messages[0] = serde_json::json!("not a message object"),
            _ => unreachable!(),
        }
        let renders = facts.borrow().chat_renders;
        if matches!(case, 0 | 1) {
            let prepared = model
                .prepare_chat(
                    &source,
                    &input,
                    &crate::memory_fixture::limits(u64::MAX),
                    &cancellation,
                )
                .unwrap()
                .unwrap();
            assert!(!prepared.text_generation_support().is_supported());
            let error = match model
                .start_prepared_chat(literal_request(&prepared, chat_settings()), &cancellation)
            {
                Err(error) => error,
                Ok(_) => panic!("literal mode must reject declarations"),
            };
            assert_eq!(
                error.text_output_rejection().as_ref(),
                Some(prepared.text_generation_support())
            );
            assert_eq!(facts.borrow().stops, 0);
            assert_eq!(facts.borrow().encodes, 0);
            assert!(!facts.borrow().order.contains(&"submit"));
            drop((error, prepared));
            assert_eq!(pool.live_charge_bytes().unwrap(), cold);
            continue;
        }
        let error = model
            .prepare_chat(
                &source,
                &input,
                &crate::memory_fixture::limits(u64::MAX),
                &cancellation,
            )
            .unwrap_err();
        if case != 2 {
            assert_eq!(facts.borrow().chat_renders, renders);
        }
        assert_eq!(facts.borrow().stops, 0);
        assert_eq!(facts.borrow().encodes, 0);
        assert!(!facts.borrow().order.contains(&"submit"));
        drop(error);
        assert_eq!(pool.live_charge_bytes().unwrap(), cold, "case {case}");
    }
    let prepared = model
        .prepare_chat(
            &source,
            &request(),
            &crate::memory_fixture::limits(u64::MAX),
            &cancellation,
        )
        .unwrap()
        .unwrap();
    let (other_runtime, other_facts, other_pool) = bare_runtime_with_capacity(u64::MAX);
    let (mut other, other_tokenizer, other_source) = loaded_sources(
        other_runtime,
        TOKENIZER.as_bytes(),
        CHAT_CONFIG,
        "smol",
        ModelKind::Llama,
        &cancellation,
    );
    let other_cold = other_pool.live_charge_bytes().unwrap();
    let error = match other
        .start_prepared_chat(literal_request(&prepared, chat_settings()), &cancellation)
    {
        Err(error) => error,
        Ok(_) => panic!("equal-content foreign source must reject"),
    };
    assert_eq!(
        error.input_rejection(),
        Some(TokenInputRejection::IdentityMismatch)
    );
    assert_eq!(other_facts.borrow().stops, 0);
    assert_eq!(other_facts.borrow().encodes, 0);
    assert!(other_facts.borrow().order.is_empty());
    drop(error);
    assert_eq!(other_pool.live_charge_bytes().unwrap(), other_cold);
    drop((other_source, other_tokenizer, other));
    assert_eq!(other_pool.live_charge_bytes().unwrap(), 0);
    drop((prepared, source, tokenizer, model));
    assert_eq!(pool.live_charge_bytes().unwrap(), 0);
}

#[test]
#[ignore = "requires exact pinned complete tokenizer and config paths; required serial chat oracle"]
fn released_complete_config_and_tokenizer_use_public_prepared_chat_request() {
    use sha2::{Digest, Sha256};
    #[derive(serde::Deserialize)]
    struct Input {
        origin: String,
        sha256: String,
    }
    #[derive(serde::Deserialize)]
    struct Inputs {
        tokenizer: Input,
        config: Input,
    }
    let manifest = std::env::var_os("EREDU_ORIGINAL_CHAT_ARTIFACTS").expect("pinned chat inputs");
    let inputs: Inputs = serde_json::from_slice(&std::fs::read(manifest).unwrap()).unwrap();
    let read = |input: &Input| {
        let bytes = std::fs::read(&input.origin).unwrap();
        let digest = Sha256::digest(&bytes)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>();
        assert_eq!(digest, input.sha256);
        bytes
    };
    let tokenizer_bytes = read(&inputs.tokenizer);
    let config_bytes = read(&inputs.config);
    let reference = tokenizers::Tokenizer::from_bytes(&tokenizer_bytes).unwrap();
    let selected = eredu_text::tokenizer::load_model_chat_template_from_str(
        std::str::from_utf8(&config_bytes).unwrap(),
    )
    .unwrap()
    .unwrap();
    let mut ordinary = eredu_text::tokenizer::Tokenizer::from_tokenizer(reference.clone());
    let (runtime, facts, pool) = bare_runtime_with_capacity(u64::MAX);
    facts.borrow_mut().shared_admission_capacity = Some(u64::MAX);
    facts.borrow_mut().maximum_context = Some(1024);
    let cancellation = GenerationCancellationToken::new();
    let (mut model, source, template) = loaded_sources(
        runtime,
        &tokenizer_bytes,
        std::str::from_utf8(&config_bytes).unwrap(),
        "smol",
        ModelKind::Llama,
        &cancellation,
    );
    let cold = pool.live_charge_bytes().unwrap();
    let a = source.original().token_id("a").unwrap();
    let b = source.original().token_id("b").unwrap();
    assert!(!source.original().is_special("a") && !source.original().is_special("b"));
    let domain = source
        .original()
        .generation_domain()
        .unwrap()
        .allowed_mask()
        .unwrap();
    facts.borrow_mut().prediction_ids = Some([a, b, a]);
    facts.borrow_mut().output_width = Some(domain.len());
    let scalar_bytes = (0..=255)
        .map(|n| char::from_u32(n).unwrap())
        .collect::<String>();
    let cases = [
        vec![],
        vec![serde_json::json!({"role":"system","content":""})],
        vec![serde_json::json!({"role":"user","content":" \n e\u{301} 世界🙂\0\t<|im_end|> "})],
        vec![
            serde_json::json!({"role":"system","content":"Rules"}),
            serde_json::json!({"role":"user","content":"Question"}),
            serde_json::json!({"role":"assistant","content":"Answer"}),
        ],
        vec![
            serde_json::json!({"role":"","content":""}),
            serde_json::json!({"role":"custom rôle","content":"nonempty"}),
        ],
        vec![serde_json::json!({"role":"user","content":scalar_bytes})],
    ];
    let mut rows = Vec::new();
    for (case, messages) in cases.into_iter().enumerate() {
        let mut request = ChatTemplateRequest {
            messages,
            ..ChatTemplateRequest::default()
        };
        let mut last_ids = None;
        for manual in [false, true] {
            for generation in [false, true] {
                for skip in [false, true] {
                    let expected = ordinary
                        .apply_chat_template_json(
                            selected.clone(),
                            [request.messages.clone()],
                            None,
                            "smol",
                            generation,
                            None,
                        )
                        .unwrap()
                        .pop()
                        .unwrap();
                    let expected_ids = reference.encode(expected.as_str(), false).unwrap();
                    request.add_generation_prompt = generation;
                    let prepared = model
                        .prepare_chat(
                            &template,
                            &request,
                            &crate::memory_fixture::limits(u64::MAX),
                            &cancellation,
                        )
                        .unwrap()
                        .unwrap();
                    assert_eq!(prepared.rendered_prompt(), expected);
                    assert_eq!(
                        prepared.render().generation_suffix(),
                        "<|im_start|>assistant\n"
                    );
                    facts.borrow_mut().ids.clear();
                    let prepared_used = pool.live_charge_bytes().unwrap();
                    let prior_actions = facts.borrow().order.len();
                    let mut invocation = literal_request(&prepared, chat_settings());
                    invocation.skip_special_tokens = skip;
                    let started = model.start_prepared_chat(invocation, &cancellation);
                    if expected_ids.get_ids().is_empty() {
                        assert!(expected.is_empty());
                        let error = match started {
                            Err(error) => error,
                            Ok(_) => panic!("empty encoded prompt must reject before I"),
                        };
                        assert!(matches!(
                            error.input_rejection(),
                            Some(TokenInputRejection::Empty)
                        ));
                        assert!(pool.live_charge_bytes().unwrap() > prepared_used);
                        assert_eq!(facts.borrow().order.len(), prior_actions);
                        drop(error);
                        assert_eq!(pool.live_charge_bytes().unwrap(), prepared_used);
                        drop(prepared);
                        assert_eq!(pool.live_charge_bytes().unwrap(), cold);
                        last_ids = Some(Vec::new());
                        rows.push(serde_json::json!({"case":case,"manual":manual,"generation_prompt":generation,"skip_special":skip,"prompt_bytes":0,"input_ids":0,"outcome":"typed_empty_before_I"}));
                        continue;
                    }
                    let mut session = started.unwrap().unwrap();
                    assert_eq!(facts.borrow().ids, expected_ids.get_ids());
                    let mut visible = String::new();
                    let mut emit = |event: SemanticEvent| {
                        if let SemanticEvent::TextDelta(text) = event {
                            visible.push_str(&text);
                        }
                    };
                    let output = if manual {
                        while session.finish_reason().is_none() {
                            session = session.advance(&cancellation, &mut emit).unwrap();
                        }
                        session
                            .into_output()
                            .unwrap_or_else(|_| panic!("terminal chat"))
                    } else {
                        session.run(&cancellation, &mut emit).unwrap()
                    };
                    assert_eq!(output.token_ids.as_ref(), &[a, b, a]);
                    assert_eq!(visible, reference.decode(&[a, b, a], skip).unwrap());
                    let ids = output.token_ids.clone();
                    let retained = pool.live_charge_bytes().unwrap();
                    drop(output);
                    assert_eq!(pool.live_charge_bytes().unwrap(), retained);
                    drop(prepared);
                    assert!(pool.live_charge_bytes().unwrap() > cold);
                    assert_eq!(ids.as_ref(), &[a, b, a]);
                    drop(ids);
                    assert_eq!(pool.live_charge_bytes().unwrap(), cold);
                    last_ids = Some(expected_ids.get_ids().to_vec());
                    rows.push(serde_json::json!({"case":case,"manual":manual,"generation_prompt":generation,"skip_special":skip,"prompt_bytes":expected.len(),"input_ids":expected_ids.get_ids().len(),"output_ids":3}));
                }
            }
        }
        assert!(last_ids.is_some());
    }
    assert_eq!(rows.len(), 48);
    assert_eq!(
        rows.iter()
            .filter(|row| row.get("outcome").is_some())
            .count(),
        4
    );
    assert_eq!(facts.borrow().chat_renders, 48);
    assert_eq!(facts.borrow().encodes, 48);
    assert_eq!(facts.borrow().stops, 48);
    drop((template, source, model));
    assert_eq!(pool.live_charge_bytes().unwrap(), 0);
    println!(
        "ORIGINAL_CHAT_ORACLE_JSON={}",
        serde_json::json!({"scope":"complete pinned config/tokenizer original public prepared chat","config_sha256":inputs.config.sha256,"tokenizer_sha256":inputs.tokenizer.sha256,"public_requests":48,"completed_requests":44,"empty_rejections":4,"rows":rows})
    );
}

mod public_managed;

mod qwen_tools;

mod inkling;
mod released_template;

mod muse_nemotron;

mod deepseek;

mod lfm;

mod llama;

mod gemma;

mod mistral;

mod clock_templates;

mod nanbeige_kimi;

#[path = "original_chat/profile.rs"]
mod profile;
