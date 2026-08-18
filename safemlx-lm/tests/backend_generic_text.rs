use std::convert::Infallible;

use safemlx_lm::{
    api::{
        ChatTemplateRequest, ChatTokenizer, LoadedModel, LoadedTextModelConfig, ModelChatTemplate,
        PreparedChatGenerationRequest, PreparedChatGenerationSettings, PreparedChatInput,
    },
    Backend, BackendCapabilities, BackendDescriptor, BackendSession, DeviceDescriptor,
    FinishReason, GenerationConfigOverrides, ModelRuntime, PreparedModel, SemanticEvent,
    Submission, TextGenerationBackend, TextGenerationConfig, TokenFilter, TokenOutput,
};
use safemlx_lm_core::Completion;
use tokenizers::{
    decoders::byte_level::ByteLevel, models::wordlevel::WordLevel,
    pre_tokenizers::whitespace::Whitespace, AddedToken, Tokenizer,
};

const QWEN_TEMPLATE: &str =
    include_str!("fixtures/chat_templates/qwen2.5-7b-instruct-acbd9653.jinja");

struct MockBackend;
struct MockSession;
struct Done;

impl Completion for Done {
    type Error = Infallible;

    fn is_complete(&self) -> Result<bool, Self::Error> {
        Ok(true)
    }

    fn wait(&self) -> Result<(), Self::Error> {
        Ok(())
    }
}

impl Backend for MockBackend {
    type ModelConfig = ();
    type Model = ();
    type Session = MockSession;
    type Error = Infallible;

    fn descriptor(&self) -> BackendDescriptor {
        BackendDescriptor {
            name: "mock".into(),
            version: "test".into(),
        }
    }

    fn devices(&self) -> Result<Vec<(DeviceDescriptor, BackendCapabilities)>, Self::Error> {
        Ok(Vec::new())
    }

    fn prepare_model(
        &self,
        _: Self::ModelConfig,
    ) -> Result<PreparedModel<Self::Model>, Self::Error> {
        Ok(PreparedModel::new(()))
    }

    fn create_session(&self, _: PreparedModel<Self::Model>) -> Result<Self::Session, Self::Error> {
        Ok(MockSession)
    }
}

impl BackendSession<MockBackend> for MockSession {
    type PrefillInput = Vec<u32>;
    type DecodeInput = u32;
    type Output = u32;
    type Completion = Done;

    fn prefill(
        &mut self,
        _: &MockBackend,
        input: Self::PrefillInput,
    ) -> Result<Submission<Self::Output, Self::Completion>, Infallible> {
        Ok(Submission {
            output: input.len() as u32,
            completion: Done,
        })
    }

    fn decode(
        &mut self,
        _: &MockBackend,
        input: Self::DecodeInput,
    ) -> Result<Submission<Self::Output, Self::Completion>, Infallible> {
        Ok(Submission {
            output: input + 1,
            completion: Done,
        })
    }
}

impl TextGenerationBackend for MockBackend {
    type Prompt = Vec<u32>;
    type Token = u32;
    type TextGenerationState = ();
    type TextCompletion = Done;

    fn start_text_generation(
        _: &Self,
        _: TextGenerationConfig,
    ) -> Result<Self::TextGenerationState, Self::Error> {
        Ok(())
    }

    fn prepare_text_prompt(
        _: &Self,
        prompt_token_ids: Vec<u32>,
    ) -> Result<Self::Prompt, Self::Error> {
        Ok(prompt_token_ids)
    }

    fn submit_text_prefill(
        runtime: &mut ModelRuntime<Self>,
        prompt: Self::Prompt,
        filter: &TokenFilter,
        _: &mut Self::TextGenerationState,
    ) -> Result<Submission<Self::Token, Self::TextCompletion>, Self::Error> {
        let submission = runtime.prefill(prompt)?;
        Ok(Submission {
            output: apply_filter(submission.output, filter),
            completion: submission.completion,
        })
    }

    fn submit_text_decode(
        runtime: &mut ModelRuntime<Self>,
        token: Self::Token,
        filter: &TokenFilter,
        _: &mut Self::TextGenerationState,
    ) -> Result<Submission<Self::Token, Self::TextCompletion>, Self::Error> {
        let submission = runtime.decode(token)?;
        Ok(Submission {
            output: apply_filter(submission.output, filter),
            completion: submission.completion,
        })
    }
}

fn apply_filter(candidate: u32, filter: &TokenFilter) -> u32 {
    let Some(allowed) = filter.allowed_mask() else {
        return candidate;
    };
    allowed
        .get(candidate as usize)
        .copied()
        .unwrap_or(false)
        .then_some(candidate)
        .or_else(|| {
            allowed
                .iter()
                .position(|allowed| *allowed)
                .map(|token| token as u32)
        })
        .unwrap()
}

fn client_code<B: TextGenerationBackend>(model: &mut LoadedModel<B>) -> Vec<u32> {
    let prompt = model.encode("hello", false).unwrap();
    let sampling = model
        .resolve_generation_config(GenerationConfigOverrides {
            max_new_tokens: Some(3),
            ..Default::default()
        })
        .unwrap();
    model
        .generate_tokens(prompt, TextGenerationConfig::new(sampling))
        .unwrap()
        .map(|token| token.unwrap().token_id().unwrap())
        .collect()
}

#[test]
fn downstream_text_client_is_generic_over_the_selected_backend() {
    let vocabulary = [("[UNK]".to_owned(), 0), ("hello".to_owned(), 1)]
        .into_iter()
        .collect();
    let tokenizer = WordLevel::builder()
        .vocab(vocabulary)
        .unk_token("[UNK]".into())
        .build()
        .unwrap();
    let tokenizer = ChatTokenizer::from_tokenizer(Tokenizer::new(tokenizer));
    let runtime = ModelRuntime::prepare(MockBackend, ()).unwrap();
    let mut model = LoadedModel::from_runtime(
        runtime,
        tokenizer,
        LoadedTextModelConfig {
            model_type: "mock_text".into(),
            model_id: "mock".into(),
            chat_template: None,
            eos_token_ids: Vec::new(),
            checkpoint_generation_config: None,
        },
    );

    assert_eq!(client_code(&mut model), vec![1, 2, 3]);
    assert_eq!(model.model_type(), "mock_text");
}

#[test]
fn prepared_chat_constraints_and_semantics_use_the_same_generic_client_api() {
    let vocabulary = std::iter::once(("[UNK]".to_owned(), 0))
        .chain((0..64).map(|index| (format!("ordinary_{index}"), index + 1)))
        .collect();
    let word_level = WordLevel::builder()
        .vocab(vocabulary)
        .unk_token("[UNK]".into())
        .build()
        .unwrap();
    let mut tokenizer = Tokenizer::new(word_level);
    tokenizer.with_pre_tokenizer(Some(Whitespace));
    tokenizer
        .add_special_tokens([AddedToken::from("<|im_end|>", true).normalized(false)])
        .unwrap();
    tokenizer.with_decoder(Some(ByteLevel::default()));
    let eos_token_id = tokenizer.token_to_id("<|im_end|>").unwrap();
    let mut tokenizer = ChatTokenizer::from_tokenizer(tokenizer);
    tokenizer.set_template_kwargs(serde_json::Map::from_iter([(
        "bos_token".into(),
        serde_json::json!("<|begin_of_text|>"),
    )]));

    let runtime = ModelRuntime::prepare(MockBackend, ()).unwrap();
    let mut model = LoadedModel::from_runtime(
        runtime,
        tokenizer,
        LoadedTextModelConfig {
            model_type: "qwen2".into(),
            model_id: "mock-qwen".into(),
            chat_template: Some(ModelChatTemplate::Single(QWEN_TEMPLATE.into())),
            eos_token_ids: vec![eos_token_id],
            checkpoint_generation_config: None,
        },
    );
    let prepared = model
        .prepare_chat(ChatTemplateRequest {
            messages: vec![serde_json::json!({"role": "user", "content": "hello"})],
            add_generation_prompt: true,
            ..Default::default()
        })
        .unwrap();
    let mut events = Vec::new();
    let output = model
        .generate_prepared_chat(PreparedChatGenerationRequest {
            input: PreparedChatInput::rendered_prompt(&prepared),
            settings: PreparedChatGenerationSettings {
                overrides: GenerationConfigOverrides {
                    max_new_tokens: Some(2),
                    ..Default::default()
                },
                seed: 7,
            },
            caller_stop_sequences: &[],
            cancellation: Default::default(),
            on_event: |event| events.push(event),
        })
        .unwrap();

    assert_eq!(output.token_ids.len(), 2);
    assert_eq!(output.finish_reason, FinishReason::MaxTokens);
    assert_eq!(
        events.last(),
        Some(&SemanticEvent::Finished {
            reason: FinishReason::MaxTokens
        })
    );
}
