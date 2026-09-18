use super::*;
use eredu::api::{PreparedChatGenerationSettings, PreparedChatRequest};
use eredu::runtime::chat::{ChatTemplateRequest, SemanticSupport, ToolChoice};
use eredu_runtime::working_memory::{WorkingMemoryError, WorkingMemoryPool};
use serde_json::json;
use std::{cell::RefCell, rc::Rc};
use tokenizers::{decoders::byte_level::ByteLevel, models::bpe::BPE};

fn fixture(
    pool: &WorkingMemoryPool,
    template: bool,
) -> (
    original_sources::Fixture<MockBackend>,
    Rc<RefCell<BackendCalls>>,
    Vec<u32>,
) {
    let vocabulary: tokenizers::models::bpe::Vocab = ByteLevel::alphabet()
        .into_iter()
        .enumerate()
        .map(|(id, character)| (character.to_string(), id as u32))
        .collect();
    let mut tokenizer = Tokenizer::new(
        BPE::builder()
            .vocab_and_merges(vocabulary, Vec::new())
            .build()
            .unwrap(),
    );
    tokenizer.with_pre_tokenizer(Some(ByteLevel::new(false, false, false)));
    tokenizer.with_decoder(Some(ByteLevel::default()));
    tokenizer
        .add_special_tokens([AddedToken::from("<|im_end|>", true).normalized(false)])
        .unwrap();
    let eos = tokenizer.token_to_id("<|im_end|>").unwrap();
    let mut answer = tokenizer.encode("okay", false).unwrap().get_ids().to_vec();
    answer.push(eos);
    let backend = MockBackend::default();
    let calls = backend.calls.clone();
    calls.borrow_mut().pool = Some(pool.clone());
    calls.borrow_mut().scripted_tokens = answer.iter().copied().collect();
    let model = original_sources::Fixture::from_runtime(
        ModelRuntime::prepare(backend, ()).unwrap(),
        ChatTokenizer::from_tokenizer(tokenizer),
        LoadedTextModelConfig {
            model_family: ModelKind::Qwen2,
            effective_model_type: "qwen2".into(),
            model_id: "frozen-recipe-host-custody".into(),
            chat_template: template.then(|| {
                include_str!("../fixtures/chat_templates/qwen2.5-7b-instruct-acbd9653.jinja").into()
            }),
            eos_token_ids: vec![eos],
            checkpoint_generation_config: None,
        },
    )
    .unwrap();
    (model, calls, answer)
}

fn request() -> ChatTemplateRequest {
    ChatTemplateRequest {
        messages: vec![json!({"role": "user", "content": "Say okay."})],
        tool_choice: ToolChoice::None,
        add_generation_prompt: true,
        ..Default::default()
    }
}

fn settings() -> PreparedChatGenerationSettings {
    original_sources::settings(PreparedChatGenerationSettings {
        overrides: GenerationConfigOverrides {
            max_new_tokens: Some(16),
            ..Default::default()
        },
        ..Default::default()
    })
}

pub(super) fn cause<'a, T: std::error::Error + 'static>(
    mut error: &'a (dyn std::error::Error + 'static),
) -> Option<&'a T> {
    loop {
        if let Some(cause) = error.downcast_ref::<T>() {
            return Some(cause);
        }
        error = error.source()?;
    }
}

#[test]
fn source_preparation_refuses_before_prompt_sampling_or_submission() {
    let pool = WorkingMemoryPool::new(8 * 1024 * 1024 * 1024, 0).unwrap();
    let (mut model, calls, _) = fixture(&pool, true);
    let cancel = eredu_core::GenerationCancellationToken::new();
    let source = model.chat_source(false, &cancel).unwrap().unwrap();
    let before = pool.used_bytes().unwrap();
    let error = model
        .prepare_chat(&source, &request(), 1, &cancel)
        .unwrap_err();
    assert!(
        cause::<WorkingMemoryError>(&error).is_some()
            || cause::<eredu_core::HostMetadataFundingError>(&error).is_some(),
        "{error:?}"
    );
    assert_eq!(calls.borrow().prompts, 0);
    assert!(calls.borrow().configs.is_empty());
    assert!(calls.borrow().filters.is_empty());
    assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
    drop(error);
    assert_eq!(pool.used_bytes().unwrap(), before);
    let prepared = model
        .prepare_chat(&source, &request(), original_sources::CAPACITY, &cancel)
        .unwrap()
        .unwrap();
    assert!(matches!(
        prepared.semantic_support(),
        SemanticSupport::Supported
    ));
}

#[test]
fn recognized_prepared_chat_clones_keep_exact_source_charge_after_model_drop() {
    let pool = WorkingMemoryPool::new(8 * 1024 * 1024 * 1024, 0).unwrap();
    let (mut model, _, _) = fixture(&pool, true);
    let cancel = eredu_core::GenerationCancellationToken::new();
    let source = model.chat_source(false, &cancel).unwrap().unwrap();
    let prepared = model
        .prepare_chat(&source, &request(), original_sources::CAPACITY, &cancel)
        .unwrap()
        .unwrap();
    let bytes = pool.used_bytes().unwrap();
    assert!(bytes > 0);
    let alias = prepared.clone();
    let last = alias.clone();
    assert_eq!(pool.used_bytes().unwrap(), bytes);
    drop((model, source, prepared));
    let retained = pool.used_bytes().unwrap();
    assert!(retained > 0 && retained <= bytes);
    drop(alias);
    assert_eq!(pool.used_bytes().unwrap(), retained);
    assert!(!last.rendered_prompt().is_empty());
    drop(last);
    assert_eq!(pool.used_bytes().unwrap(), 0);
    assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
}

#[test]
fn cancelled_and_foreign_preparation_refuse_before_backend_work() {
    let pool = WorkingMemoryPool::new(8 * 1024 * 1024 * 1024, 0).unwrap();
    let (mut model, calls, _) = fixture(&pool, true);
    let cancel = eredu_core::GenerationCancellationToken::new();
    let source = model.chat_source(false, &cancel).unwrap().unwrap();
    cancel.cancel();
    let before = pool.used_bytes().unwrap();
    assert!(
        model
            .prepare_chat(&source, &request(), original_sources::CAPACITY, &cancel)
            .unwrap()
            .is_none()
    );
    assert_eq!(pool.used_bytes().unwrap(), before);
    let foreign = WorkingMemoryPool::new(8 * 1024 * 1024 * 1024, 0).unwrap();
    let (mut other, other_calls, _) = fixture(&foreign, true);
    let cancel = eredu_core::GenerationCancellationToken::new();
    assert!(
        other
            .prepare_chat(&source, &request(), original_sources::CAPACITY, &cancel)
            .is_err()
    );
    assert_eq!(calls.borrow().prompts, 0);
    assert_eq!(other_calls.borrow().prompts, 0);
    assert!(calls.borrow().configs.is_empty());
    assert!(other_calls.borrow().configs.is_empty());
}

#[test]
fn manual_and_uninterrupted_semantics_share_original_sources_and_output_custody() {
    let pool = WorkingMemoryPool::new(8 * 1024 * 1024 * 1024, 0).unwrap();
    let (mut model, calls, answer) = fixture(&pool, true);
    let cancel = eredu_core::GenerationCancellationToken::new();
    let source = model.chat_source(false, &cancel).unwrap().unwrap();
    let chat = model
        .prepare_chat(&source, &request(), original_sources::CAPACITY, &cancel)
        .unwrap()
        .unwrap();
    let alias = chat.clone();
    let baseline = pool.used_bytes().unwrap();
    let mut outputs = Vec::new();
    let mut all_events = Vec::new();
    for manual in [false, true] {
        calls.borrow_mut().scripted_tokens = answer.iter().copied().collect();
        let mut events = Vec::new();
        let mut session = model
            .start_prepared_chat(PreparedChatRequest::new(&chat, settings()), &cancel)
            .unwrap()
            .unwrap();
        let output = if manual {
            while session.finish_reason().is_none() {
                session = session
                    .advance(&cancel, &mut |event| events.push(event))
                    .unwrap();
            }
            session.into_output().unwrap_or_else(|_| panic!("terminal"))
        } else {
            session
                .run(&cancel, &mut |event| events.push(event))
                .unwrap()
        };
        assert_eq!(output.token_ids.as_ref(), answer);
        assert!(
            pool.used_bytes().unwrap() > baseline,
            "output retains its actual funded token destination"
        );
        outputs.push(output);
        all_events.push(events);
    }
    assert_eq!(all_events[0], all_events[1]);
    assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
    drop((model, source, chat, alias));
    assert!(pool.used_bytes().unwrap() > 0);
    drop(outputs);
    drop(calls);
    drop(all_events);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[path = "frozen_recipe/error_conversion.rs"]
mod error_conversion;
