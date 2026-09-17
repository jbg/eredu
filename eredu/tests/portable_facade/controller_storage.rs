use super::*;
use eredu::api::{
    PreparedChatError, PreparedChatGenerationRequest, PreparedChatGenerationSettings,
    PreparedChatInput,
};
use eredu::runtime::chat::{ChatTemplateRequest, ParallelToolCallPolicy, ToolChoice};
use serde_json::json;
use tokenizers::{decoders::byte_level::ByteLevel, models::bpe::BPE};

#[test]
fn grammar_vocabulary_factory_is_admitted_before_prompt_sampler_and_controller_work() {
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
        let mut model = LoadedModel::from_runtime(
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
        let prepared = model.prepare_chat(ChatTemplateRequest {
            messages: vec![json!({"role": "user", "content": "Say okay."})],
            tools: vec![json!({"type": "function", "function": {
                "name": "ping", "parameters": {"type": "object", "properties": {}, "additionalProperties": false}
            }})],
            tool_choice: choice,
            parallel_tool_calls: ParallelToolCallPolicy::Disabled,
            add_generation_prompt: true,
            ..Default::default()
        }).unwrap();
        let (attempts, factories, payload) = {
            let calls = calls.borrow();
            (
                calls.shared_bytes_attempts,
                calls.shared_bytes_factories,
                calls.shared_bytes_payload,
            )
        };
        calls.borrow_mut().reject_shared_bytes = true;
        let settings = PreparedChatGenerationSettings {
            overrides: GenerationConfigOverrides {
                max_new_tokens: Some(16),
                ..Default::default()
            },
            ..Default::default()
        };
        let error = model
            .generate_prepared_chat(PreparedChatGenerationRequest {
                input: PreparedChatInput::token_ids(&prepared, vec![answer[0]]),
                settings,
                caller_stop_sequences: &[],
                cancellation: Default::default(),
                on_event: |_| panic!("rejected vocabulary must not emit an event"),
            })
            .unwrap_err();
        let PreparedChatError::Backend(backend) = error else {
            panic!("source preparation must retain its typed backend error: {error:?}");
        };
        assert!(std::error::Error::source(&backend)
            .unwrap()
            .is::<MockError>());
        {
            let calls = calls.borrow();
            assert_eq!(calls.shared_bytes_attempts, attempts + 1);
            assert_eq!(calls.shared_bytes_factories, factories);
            assert_eq!(calls.shared_bytes_payload, payload);
            assert_eq!(calls.prompts, 0);
            assert!(calls.configs.is_empty());
            assert!(calls.filters.is_empty());
            assert_eq!(calls.scripted_tokens.len(), answer.len() + 1);
        }
        calls.borrow_mut().reject_shared_bytes = false;
        let output = model
            .generate_prepared_chat(PreparedChatGenerationRequest {
                input: PreparedChatInput::token_ids(&prepared, vec![answer[0]]),
                settings,
                caller_stop_sequences: &[],
                cancellation: Default::default(),
                on_event: |_| {},
            })
            .unwrap();
        assert_eq!(
            output.token_ids,
            answer.iter().copied().chain([eos]).collect::<Vec<_>>()
        );
        let calls = calls.borrow();
        assert_eq!(calls.shared_bytes_attempts, attempts + 2);
        assert_eq!(calls.shared_bytes_factories, factories + 1);
        // Offsets and all 257 token entries live in the same owned buffer.
        assert!(calls.shared_bytes_payload - payload > (257 + 1) * 8);
        assert_eq!(calls.prompts, 1);
        assert_eq!(calls.configs.len(), 1);
        assert!(!calls.filters.is_empty());
    }
}
