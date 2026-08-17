use std::convert::Infallible;

use safemlx_lm::{
    api::{ChatTokenizer, LoadedModel, LoadedTextModelConfig},
    Backend, BackendCapabilities, BackendDescriptor, BackendSession, DeviceDescriptor,
    GenerationConfigOverrides, ModelRuntime, PreparedModel, Submission, TextGenerationBackend,
    TextGenerationConfig, TokenOutput,
};
use safemlx_lm_core::Completion;
use tokenizers::{models::wordlevel::WordLevel, Tokenizer};

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
    type Token = u32;
    type TextGenerationState = ();
    type TextCompletion = Done;

    fn start_text_generation(
        _: &Self,
        _: TextGenerationConfig,
    ) -> Result<Self::TextGenerationState, Self::Error> {
        Ok(())
    }

    fn submit_text_prefill(
        runtime: &mut ModelRuntime<Self>,
        prompt_token_ids: Vec<u32>,
        _: &mut Self::TextGenerationState,
    ) -> Result<Submission<Self::Token, Self::TextCompletion>, Self::Error> {
        runtime.prefill(prompt_token_ids)
    }

    fn submit_text_decode(
        runtime: &mut ModelRuntime<Self>,
        token: Self::Token,
        _: &mut Self::TextGenerationState,
    ) -> Result<Submission<Self::Token, Self::TextCompletion>, Self::Error> {
        runtime.decode(token)
    }
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
