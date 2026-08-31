use eredu::api::{
    inspect_text_model, load_tokenizer, LoadedModel, LoadedTextModelConfig, TextInspectionOptions,
};
use eredu_architectures::ModelKind;
use eredu_core::{
    ArtifactFormat, AutomaticPlanRequest, BackendDescriptor, BackendProvider, BackendSession,
    Completion, DeviceCapabilities, DeviceDescriptor, DevicePlan, GenerationConfigOverrides,
    InspectionReadiness, ModelInspectionReport, ModelRuntime, ObservationSet, ObservationValue,
    PreparedModel, SessionCapabilities, Submission, TextGenerationBackend, TextGenerationConfig,
    TokenFilter, TokenOutput,
};
use eredu_text::tokenizer::Tokenizer as ChatTokenizer;
use tokenizers::{models::wordlevel::WordLevel, AddedToken, Tokenizer};

struct MockBackend;
struct MockSession;

#[derive(Clone)]
struct MockToken(u32);

#[derive(Debug, thiserror::Error)]
#[error("mock backend failed")]
struct MockError;

struct Complete;

impl Completion for Complete {
    type Error = MockError;

    fn is_complete(&self) -> Result<bool, Self::Error> {
        Ok(true)
    }

    fn wait(&self) -> Result<(), Self::Error> {
        Ok(())
    }
}

impl TokenOutput for MockToken {
    type Error = MockError;

    fn token_id(&self) -> Result<u32, Self::Error> {
        Ok(self.0)
    }
}

impl BackendProvider for MockBackend {
    type ModelConfig = ();
    type Model = ();
    type Session = MockSession;
    type Error = MockError;

    fn descriptor(&self) -> BackendDescriptor {
        BackendDescriptor {
            name: "portable-mock".into(),
            version: "1".into(),
        }
    }

    fn devices(&self) -> Result<Vec<(DeviceDescriptor, DeviceCapabilities)>, Self::Error> {
        Ok(Vec::new())
    }

    fn prepare_model(
        &self,
        _: Self::ModelConfig,
    ) -> Result<PreparedModel<Self::Model>, Self::Error> {
        Ok(PreparedModel::new((), SessionCapabilities::default()))
    }

    fn create_session(&self, _: PreparedModel<Self::Model>) -> Result<Self::Session, Self::Error> {
        Ok(MockSession)
    }
}

impl BackendSession<MockBackend> for MockSession {
    type PrefillInput = Vec<u32>;
    type DecodeInput = u32;
    type Output = u32;
    type Completion = Complete;

    fn capabilities(&self) -> SessionCapabilities {
        SessionCapabilities::default()
    }

    fn prefill(
        &mut self,
        _: &MockBackend,
        input: Self::PrefillInput,
    ) -> Result<Submission<Self::Output, Self::Completion>, MockError> {
        Ok(Submission {
            output: input.len() as u32,
            completion: Complete,
        })
    }

    fn decode(
        &mut self,
        _: &MockBackend,
        input: Self::DecodeInput,
    ) -> Result<Submission<Self::Output, Self::Completion>, MockError> {
        Ok(Submission {
            output: input + 1,
            completion: Complete,
        })
    }

    fn observe_output(
        &self,
        _: &MockBackend,
        output: &Self::Output,
    ) -> Result<ObservationSet, MockError> {
        let mut observations = ObservationSet::new();
        observations
            .insert(
                "mock.output",
                ObservationValue::Unsigned(u64::from(*output)),
            )
            .unwrap();
        Ok(observations)
    }
}

impl TextGenerationBackend for MockBackend {
    type Prompt = Vec<u32>;
    type Token = MockToken;
    type TextGenerationState = ();
    type TextCompletion = Complete;

    fn start_text_generation(
        _: &Self,
        _: TextGenerationConfig,
    ) -> Result<Self::TextGenerationState, Self::Error> {
        Ok(())
    }

    fn prepare_text_prompt(_: &Self, ids: Vec<u32>) -> Result<Self::Prompt, Self::Error> {
        Ok(ids)
    }

    fn submit_text_prefill(
        runtime: &mut ModelRuntime<Self>,
        prompt: Self::Prompt,
        _: &TokenFilter,
        _: &mut Self::TextGenerationState,
    ) -> Result<Submission<Self::Token, Self::TextCompletion>, Self::Error> {
        let submission = runtime.prefill(prompt)?;
        Ok(Submission {
            output: MockToken(submission.output),
            completion: submission.completion,
        })
    }

    fn submit_text_decode(
        runtime: &mut ModelRuntime<Self>,
        token: Self::Token,
        _: &TokenFilter,
        _: &mut Self::TextGenerationState,
    ) -> Result<Submission<Self::Token, Self::TextCompletion>, Self::Error> {
        let submission = runtime.decode(token.0)?;
        Ok(Submission {
            output: MockToken(submission.output),
            completion: submission.completion,
        })
    }
}

#[test]
fn loaded_model_generates_without_an_mlx_dependency() {
    let runtime = ModelRuntime::prepare(MockBackend, ()).unwrap();
    let mut tokenizer = Tokenizer::new(WordLevel::default());
    tokenizer
        .add_tokens([
            AddedToken::from("hello".to_owned(), false),
            AddedToken::from("world".to_owned(), false),
        ])
        .unwrap();
    let mut model = LoadedModel::from_runtime(
        runtime,
        ChatTokenizer::from_tokenizer(tokenizer),
        LoadedTextModelConfig {
            model_family: ModelKind::Qwen35,
            effective_model_type: "qwen3_5_text".into(),
            model_id: "mock/model".into(),
            chat_template: None,
            eos_token_ids: vec![99],
            checkpoint_generation_config: None,
        },
    );
    let sampling = model
        .resolve_generation_config(GenerationConfigOverrides {
            max_new_tokens: Some(3),
            ..Default::default()
        })
        .unwrap();
    let prompt = model.encode("hello", false).unwrap();
    let tokens = model
        .generate_tokens(prompt, TextGenerationConfig::new(sampling))
        .unwrap()
        .map(|token| token.unwrap().token_id().unwrap())
        .collect::<Vec<_>>();

    assert_eq!(tokens, vec![1, 2, 3]);
    assert_eq!(model.model_family(), ModelKind::Qwen35);
    assert_eq!(model.effective_model_type(), "qwen3_5_text");
}

#[test]
fn automatic_planning_documents_are_available_without_mlx() {
    let request = AutomaticPlanRequest::new(
        "model",
        DevicePlan::new("mock", "gpu:0").expect("portable device identity is valid"),
    );
    assert_eq!(
        serde_json::from_slice::<AutomaticPlanRequest>(&serde_json::to_vec(&request).unwrap())
            .unwrap(),
        request
    );
}

#[test]
fn tokenizer_and_text_inspection_are_available_without_mlx() {
    let directory = std::env::temp_dir().join(format!(
        "eredu-portable-inspection-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    std::fs::create_dir_all(&directory).unwrap();
    std::fs::write(
        directory.join("tokenizer.json"),
        br#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"<unk>":0,"hello":1},"unk_token":"<unk>"}}"#,
    )
    .unwrap();
    std::fs::write(directory.join("config.json"), br#"{"model_type":"llama"}"#).unwrap();

    assert_eq!(load_tokenizer(&directory).unwrap().get_vocab_size(false), 2);
    let mut structural = ModelInspectionReport::unverified(&directory, ArtifactFormat::SafeTensors);
    structural.model_loadability = InspectionReadiness::Ready;
    structural.requested_load = InspectionReadiness::Ready;
    let report = inspect_text_model(structural, TextInspectionOptions::default());
    assert_eq!(report.tokenizer, InspectionReadiness::Ready);
    assert_eq!(report.text_generation, InspectionReadiness::Ready);

    std::fs::remove_dir_all(directory).unwrap();
}
