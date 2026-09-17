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
    Ok(SharedControllerBytes::new(factory()))
}

struct RejectVocabulary;
impl Drop for RejectVocabulary {
    fn drop(&mut self) {
        REJECT.set(false);
    }
}

#[test]
fn controlled_semantic_setup_preserves_vocabulary_rejection_before_factory() {
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
        let mut model = LoadedModel::from_runtime(
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
        let chat = model.prepare_chat(ChatTemplateRequest {
            messages: vec![serde_json::json!({"role": "user", "content": "Say hello."})],
            tools: vec![serde_json::json!({"type": "function", "function": {
                "name": "ping", "parameters": {"type": "object", "properties": {}, "additionalProperties": false}
            }})],
            tool_choice: choice,
            parallel_tool_calls: ParallelToolCallPolicy::Disabled,
            add_generation_prompt: true,
            ..Default::default()
        }).unwrap();
        let settings = PreparedChatGenerationSettings {
            overrides: GenerationConfigOverrides {
                max_new_tokens: Some(1),
                ..Default::default()
            },
            ..Default::default()
        };
        let prepared = model
            .prepare_observed_token_ids(
                &chat,
                vec![0],
                settings,
                CapturePlan::none(),
                TraceLimits {
                    per_record_bytes: 65536,
                    total_bytes: 1048576,
                },
            )
            .unwrap();
        let attempts = ATTEMPTS.get();
        let factories = FACTORIES.get();
        REJECT.set(true);
        let guard = RejectVocabulary;
        let error = model
            .start_controlled_chat(prepared, &[], Default::default(), |_| -> ControlFlow<()> {
                panic!("rejected shared vocabulary must not emit controlled records")
            })
            .err()
            .expect("controlled host preparation must preserve backend rejection");
        assert_eq!(ATTEMPTS.get(), attempts + 1);
        assert_eq!(FACTORIES.get(), factories);
        let mut cause: &(dyn std::error::Error + 'static) = &error;
        loop {
            if let Some(MockError::Capture(message)) = cause.downcast_ref::<MockError>() {
                assert_eq!(message, "shared vocabulary rejected before packing");
                break;
            }
            cause = cause
                .source()
                .expect("original backend source must survive controlled mapping");
        }
        drop(guard);
    }
}
