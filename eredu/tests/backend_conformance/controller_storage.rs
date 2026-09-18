use super::*;
use eredu::api::TraceLimits;
use eredu::runtime::chat::{ParallelToolCallPolicy, ToolChoice};
use eredu_core::{capture::CapturePlan, BackendFailure, SharedControllerBytes};
use std::{cell::Cell, ops::ControlFlow};
use tokenizers::models::bpe::BPE;

thread_local! {
    static REJECT: Cell<bool> = const { Cell::new(false) };
    static ATTEMPTS: Cell<usize> = const { Cell::new(0) };
    static FACTORIES: Cell<usize> = const { Cell::new(0) };
}

pub(super) fn prepare(
    factory: impl FnOnce() -> Vec<u8>,
) -> Result<SharedControllerBytes, BackendFailure> {
    ATTEMPTS.set(ATTEMPTS.get() + 1);
    if REJECT.get() {
        return Err(BackendFailure::from_error(MockError::Capture(
            "shared vocabulary rejected before packing".into(),
        )));
    }
    FACTORIES.set(FACTORIES.get() + 1);
    Ok(SharedControllerBytes::new(
        factory(),
        eredu_core::HostPreparationAuthority::unmanaged(),
    ))
}

struct RejectVocabulary;
impl Drop for RejectVocabulary {
    fn drop(&mut self) {
        REJECT.set(false);
    }
}

#[test]
fn controlled_semantic_setup_uses_original_receipt_and_refuses_mismatched_capacity() {
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
        let mut model = original_sources::Fixture::from_runtime(
            ModelRuntime::prepare(MockBackend, ()).unwrap(),
            ChatTokenizer::from_tokenizer(tokenizer),
            LoadedTextModelConfig {
                model_family: ModelKind::Qwen2,
                effective_model_type: "qwen2".into(),
                model_id: "controlled-shared-vocabulary".into(),
                chat_template: Some(ModelChatTemplate::Single(QWEN_TEMPLATE.into())),
                eos_token_ids: vec![eos],
                checkpoint_generation_config: None,
            },
        )
        .unwrap();
        let chat = {
            let request = ChatTemplateRequest {
                messages: vec![serde_json::json!({"role": "user", "content": "Say hello."})],
                tools: vec![serde_json::json!({"type": "function", "function": {
                    "name": "ping", "parameters": {"type": "object", "properties": {}, "additionalProperties": false}
                }})],
                tool_choice: choice,
                parallel_tool_calls: ParallelToolCallPolicy::Disabled,
                add_generation_prompt: true,
                ..Default::default()
            };
            let cancellation = eredu_core::GenerationCancellationToken::new();
            let source = model
                .chat_source(!request.tools.is_empty(), &cancellation)
                .unwrap()
                .unwrap();
            model
                .prepare_chat(&source, &request, original_sources::CAPACITY, &cancellation)
                .unwrap()
                .unwrap()
        };
        let settings = PreparedChatGenerationSettings {
            overrides: GenerationConfigOverrides {
                max_new_tokens: Some(1),
                ..Default::default()
            },
            ..Default::default()
        };
        let pool = model.original_pool().clone();
        let before = pool.used_bytes().unwrap();
        let mut configured = original_sources::settings(settings);
        configured.inference.managed_memory_capacity_bytes = Some(1);
        let mut request = eredu::api::PreparedChatRequest::new(&chat, configured);
        request.input = eredu::api::PreparedChatPrompt::TokenIds(&[0]);
        let error = model
            .start_controlled_chat(
                request,
                TraceLimits {
                    per_record_bytes: 65536,
                    total_bytes: 1048576,
                },
                Default::default(),
                |_| panic!("unadmitted controller delivery"),
            )
            .err()
            .unwrap();
        assert!(error.session_failure().unwrap().input_rejection().is_some());
        drop(error);
        // Attempted cold controls spend their original account until its source retires.
        assert!(pool.used_bytes().unwrap() >= before);
        let mut request =
            eredu::api::PreparedChatRequest::new(&chat, original_sources::settings(settings));
        request.input = eredu::api::PreparedChatPrompt::TokenIds(&[0]);
        let run = model
            .start_controlled_chat(
                request,
                TraceLimits {
                    per_record_bytes: 65536,
                    total_bytes: 1048576,
                },
                Default::default(),
                |_| ControlFlow::Continue(()),
            )
            .unwrap()
            .unwrap();
        assert!(run.token_ids().is_empty());
        drop(run);
        drop(chat);
        drop(model);
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
}
