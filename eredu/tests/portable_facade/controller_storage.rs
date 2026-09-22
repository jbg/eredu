use super::*;
use eredu::api::{PreparedChatGenerationSettings, PreparedChatRequest};
use eredu::runtime::chat::{ChatTemplateRequest, ParallelToolCallPolicy, ToolChoice};
use serde_json::json;
use tokenizers::{decoders::byte_level::ByteLevel, models::bpe::BPE};

#[test]
fn grammar_source_is_reused_without_registration_or_prompt_work_before_admission() {
    for choice in [ToolChoice::None, ToolChoice::Auto] {
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
        let answer = tokenizer.encode("okay", false).unwrap().get_ids().to_vec();
        let backend = MockBackend::default();
        let calls = backend.calls.clone();
        calls.borrow_mut().scripted_tokens = answer.iter().copied().chain([eos]).collect();
        let mut model = original_sources::Fixture::from_runtime(
            ModelRuntime::prepare(backend, ()).unwrap(),
            ChatTokenizer::from_tokenizer(tokenizer),
            LoadedTextModelConfig {
                model_family: ModelKind::Qwen2,
                effective_model_type: "qwen2".into(),
                model_id: "shared-vocabulary-hook".into(),
                chat_template: Some(
                    include_str!("../fixtures/chat_templates/qwen2.5-7b-instruct-acbd9653.jinja")
                        .into(),
                ),
                eos_token_ids: vec![eos],
                checkpoint_generation_config: None,
            },
        )
        .unwrap();
        let request = ChatTemplateRequest {
            messages: vec![json!({"role": "user", "content": "Say okay."})],
            tools: vec![json!({"type": "function", "function": {
                "name": "ping", "parameters": {"type": "object", "properties": {}, "additionalProperties": false}
            }})],
            tool_choice: choice,
            parallel_tool_calls: ParallelToolCallPolicy::Disabled,
            add_generation_prompt: true,
            ..Default::default()
        };
        let cancel = eredu_core::GenerationCancellationToken::new();
        let source = model.chat_source(true, &cancel).unwrap().unwrap();
        let prepared = model
            .prepare_chat(
                &source,
                &request,
                &crate::memory::limits(original_sources::CAPACITY),
                &cancel,
            )
            .unwrap()
            .unwrap();
        assert_eq!(calls.borrow().prompts, 0);
        assert!(calls.borrow().configs.is_empty());
        assert!(calls.borrow().filters.is_empty());
        let settings = original_sources::settings(PreparedChatGenerationSettings {
            overrides: GenerationConfigOverrides {
                max_new_tokens: Some(16),
                ..Default::default()
            },
            ..Default::default()
        });
        let mut stale = settings.clone();
        stale.inference.memory_limits = crate::memory::limits(original_sources::CAPACITY - 1);
        let error = model
            .start_prepared_chat(PreparedChatRequest::new(&prepared, stale), &cancel)
            .err()
            .expect("different accepted capacity must refuse");
        assert_eq!(
            error.input_rejection(),
            Some(eredu_core::TokenInputRejection::IdentityMismatch)
        );
        assert_eq!(calls.borrow().prompts, 0);
        assert!(calls.borrow().configs.is_empty());
        assert_eq!(calls.borrow().scripted_tokens.len(), answer.len() + 1);
        let output = model
            .start_prepared_chat(
                PreparedChatRequest::new(&prepared, settings.clone()),
                &cancel,
            )
            .unwrap()
            .unwrap()
            .run(&cancel, &mut |_| {})
            .unwrap();
        assert_eq!(
            output.token_ids.as_ref(),
            answer.iter().copied().chain([eos]).collect::<Vec<_>>()
        );
        assert_eq!(calls.borrow().shared_bytes_attempts, 0);
        assert_eq!(calls.borrow().shared_bytes_factories, 0);
        assert_eq!(calls.borrow().prompts, 1);
        assert_eq!(calls.borrow().configs.len(), 1);
    }
}
