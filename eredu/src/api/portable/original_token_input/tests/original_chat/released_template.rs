use super::*;
use crate::api::{ManagedChatRequest, PreparedChatGenerationSettings};

pub(super) fn run_case(
    template: &str,
    family: ModelKind,
    model_id: &str,
    chat: ChatTemplateRequest,
    manual: bool,
    verify: impl FnOnce(&str),
) {
    let (runtime, facts, pool) = bare_runtime_with_capacity(u64::MAX);
    facts.borrow_mut().shared_admission_capacity = Some(u64::MAX);
    let config = serde_json::json!({"chat_template":template}).to_string();
    let selected = eredu_text::tokenizer::load_model_chat_template_from_str(&config)
        .unwrap()
        .unwrap();
    let mut model = LoadedModel::from_runtime(
        runtime,
        ChatTokenizer::from_bytes(TOKENIZER.as_bytes()).unwrap(),
        LoadedTextModelConfig {
            model_family: family,
            effective_model_type: model_id.into(),
            model_id: model_id.into(),
            chat_template: Some(selected.clone()),
            eos_token_ids: vec![],
            checkpoint_generation_config: None,
        },
    )
    .unwrap();
    let cancellation = GenerationCancellationToken::new();
    let mut tokenizer_file = tempfile::tempfile().unwrap();
    tokenizer_file.write_all(TOKENIZER.as_bytes()).unwrap();
    let tokenizer = model
        .compile_managed_plain_text_source(tokenizer_file)
        .unwrap();
    let mut template_file = tempfile::tempfile().unwrap();
    template_file.write_all(config.as_bytes()).unwrap();
    let source = model
        .compile_managed_chat_source(&tokenizer, template_file, &cancellation)
        .unwrap()
        .unwrap();
    let cold = pool.used_bytes().unwrap();
    let mut oracle = ChatTokenizer::from_bytes(TOKENIZER.as_bytes()).unwrap();
    let prompt = oracle
        .apply_chat_template_json(
            selected,
            [chat.messages.clone()],
            None,
            model_id,
            true,
            Some(&chat.extra_template_kwargs),
        )
        .unwrap()
        .pop()
        .unwrap();
    verify(&prompt);
    let expected = tokenizers::Tokenizer::from_bytes(TOKENIZER.as_bytes())
        .unwrap()
        .encode(prompt.as_str(), false)
        .unwrap();
    facts.borrow_mut().maximum_context = Some(u64::try_from(expected.len()).unwrap() + 3);
    let settings = PreparedChatGenerationSettings {
        overrides: GenerationConfigOverrides {
            max_new_tokens: Some(3),
            ..Default::default()
        },
        inference: TextInferencePolicy {
            managed_memory_capacity_bytes: Some(u64::MAX),
            ..Default::default()
        },
        ..Default::default()
    };
    // A refusal after rendering must retain the actual J/C/H; retries
    // use the same source and leave no previous render allowance alive.
    facts.borrow_mut().short = true;
    let error = match model.start_managed_chat(
        &source,
        ManagedChatRequest::new(&chat, settings),
        &cancellation,
    ) {
        Err(error) => error,
        Ok(_) => panic!("exact short admission must refuse"),
    };
    assert!(pool.used_bytes().unwrap() > cold);
    drop(error);
    assert_eq!(pool.used_bytes().unwrap(), cold);
    facts.borrow_mut().short = false;
    facts.borrow_mut().ids.clear();
    let mut visible = String::new();
    let mut emit = |event: GenerationPlainTextEvent<'_>| {
        if let GenerationPlainTextEvent::TextDelta(value) = event {
            visible.push_str(value);
        }
    };
    let request = ManagedChatRequest::new(&chat, settings);
    let output = if manual {
        let mut session = model
            .start_managed_chat(&source, request, &cancellation)
            .unwrap()
            .unwrap();
        while session.finish_reason().is_none() {
            session = session.advance(&cancellation, &mut emit).unwrap();
        }
        session
            .into_output()
            .unwrap_or_else(|_| panic!("terminal released-template chat"))
    } else {
        model
            .generate_managed_chat(&source, request, &cancellation, &mut emit)
            .unwrap()
            .unwrap()
    };
    assert_eq!(facts.borrow().ids, expected.get_ids());
    assert_eq!(output.text.as_str(), "h hih");
    assert_eq!(output.text.as_str(), visible);
    assert_eq!(output.token_ids.as_ref(), &[0, 8, 0]);
    assert_eq!(facts.borrow().chat_sources, 1);
    assert_eq!(facts.borrow().chat_renders, 2);
    let escaped = output.text.clone();
    drop((output, chat, source, tokenizer, model));
    assert!(pool.used_bytes().unwrap() > 0);
    assert_eq!(escaped.as_str(), "h hih");
    drop(escaped);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
