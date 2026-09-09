use eredu::api::{
    inspect_text_model, load_tokenizer, LoadedModel, LoadedTextModelConfig, TextInspectionOptions,
};
use eredu_architectures::ModelKind;
use eredu_core::{
    ArtifactFormat, AutomaticPlanRequest, BackendDescriptor, BackendProvider, BackendSession,
    BoundedCompletion, BoundedCompletionOutcome, BoundedCompletionWait, Completion,
    DeviceCapabilities, DeviceDescriptor, DevicePlan, GenerationConfigOverrides,
    InspectionReadiness, ModelInspectionReport, ModelRuntime, ObservationSet, ObservationValue,
    PreparedModel, SessionCapabilities, Submission, TextGenerationBackend, TextGenerationConfig,
    TokenFilter, TokenOutput,
};
use eredu_text::tokenizer::Tokenizer as ChatTokenizer;
use tokenizers::{models::wordlevel::WordLevel, AddedToken, Tokenizer};

#[path = "portable_facade/sampling.rs"]
mod sampling;

#[path = "portable_facade/lifecycle.rs"]
mod lifecycle;

#[path = "portable_facade/nanbeige.rs"]
mod nanbeige;

#[derive(Default)]
struct BackendCalls {
    configs: Vec<TextGenerationConfig>,
    filters: Vec<TokenFilter>,
    scripted_tokens: std::collections::VecDeque<u32>,
    prompts: usize,
    speculative: usize,
}

#[derive(Default)]
struct MockBackend {
    logits: Vec<f32>,
    calls: std::rc::Rc<std::cell::RefCell<BackendCalls>>,
}

impl MockBackend {
    fn sample(&self, filter: &TokenFilter) -> Result<MockToken, MockError> {
        self.calls.borrow_mut().filters.push(filter.clone());
        if let Some(token) = self.calls.borrow_mut().scripted_tokens.pop_front() {
            assert!(filter.allows(token), "scripted token {token} was masked");
            return Ok(MockToken(token));
        }
        let allowed = filter
            .allowed_mask_for(self.logits.len())
            .map_err(|_| MockError)?;
        self.logits
            .iter()
            .enumerate()
            .filter(|(id, _)| allowed.as_ref().is_none_or(|mask| mask[*id]))
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(id, _)| MockToken(id as u32))
            .ok_or(MockError)
    }
}
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

impl BoundedCompletion for Complete {
    fn wait_bounded(
        self,
        _policy: BoundedCompletionWait,
    ) -> Result<BoundedCompletionOutcome, Self::Error> {
        Ok(BoundedCompletionOutcome::Completed)
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
        BackendDescriptor::new("portable-mock", "1")
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
    fn reset_session(_: &Self, _: &mut Self::Session) -> Result<(), eredu_core::BackendFailure> {
        Ok(())
    }

    fn synchronize_session(_: &Self, _: &Self::Session) -> Result<(), eredu_core::BackendFailure> {
        Ok(())
    }

    type Prompt = Vec<u32>;
    type Token = MockToken;
    type TextGenerationState = ();
    type TextCompletion = Complete;

    fn start_text_generation(
        backend: &Self,
        config: TextGenerationConfig,
    ) -> Result<Self::TextGenerationState, Self::Error> {
        backend.calls.borrow_mut().configs.push(config);
        Ok(())
    }

    fn prepare_text_prompt(backend: &Self, ids: Vec<u32>) -> Result<Self::Prompt, Self::Error> {
        backend.calls.borrow_mut().prompts += 1;
        Ok(ids)
    }

    fn submit_text_prefill(
        runtime: &mut ModelRuntime<Self>,
        prompt: Self::Prompt,
        filter: &TokenFilter,
        _: &mut Self::TextGenerationState,
    ) -> Result<Submission<Self::Token, Self::TextCompletion>, Self::Error> {
        let submission = runtime.prefill(prompt)?;
        Ok(Submission {
            output: runtime.backend().sample(filter)?,
            completion: submission.completion,
        })
    }

    fn submit_text_decode(
        runtime: &mut ModelRuntime<Self>,
        token: Self::Token,
        filter: &TokenFilter,
        _: &mut Self::TextGenerationState,
    ) -> Result<Submission<Self::Token, Self::TextCompletion>, Self::Error> {
        let submission = runtime.decode(token.0)?;
        Ok(Submission {
            output: runtime.backend().sample(filter)?,
            completion: submission.completion,
        })
    }
}

#[test]
fn loaded_model_generates_without_an_mlx_dependency() {
    let runtime = ModelRuntime::prepare(
        MockBackend {
            logits: vec![0.0, 1.0, 100.0, 200.0],
            ..Default::default()
        },
        (),
    )
    .unwrap();
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

    assert_eq!(tokens, vec![1, 1, 1]);
    assert_eq!(model.model_family(), ModelKind::Qwen35);
    assert_eq!(model.effective_model_type(), "qwen3_5_text");
}

#[test]
#[ignore = "requires EREDU_LFM2_TOOL_CHECKPOINT; tokenizer/template only, no native backend"]
fn lfm_checkpoint_native_tool_activation_with_portable_backend() {
    use eredu::api::{
        PreparedChatGenerationRequest, PreparedChatGenerationSettings, PreparedChatInput,
    };
    use eredu::runtime::chat::{ChatTemplateRequest, ParallelToolCallPolicy, ToolChoice};
    use eredu_core::{FinishReason, SemanticEvent};
    use serde_json::json;

    let path = std::path::PathBuf::from(std::env::var("EREDU_LFM2_TOOL_CHECKPOINT").unwrap());
    let mut tokenizer = ChatTokenizer::from_tokenizer(load_tokenizer(&path).unwrap());
    let metadata: serde_json::Value =
        serde_json::from_slice(&std::fs::read(path.join("tokenizer_config.json")).unwrap())
            .unwrap();
    tokenizer.set_template_kwargs(serde_json::Map::from_iter([(
        "bos_token".into(),
        metadata["bos_token"].clone(),
    )]));
    let start = tokenizer.token_to_id("<|tool_call_start|>").unwrap();
    let eos = tokenizer.token_to_id("<|im_end|>").unwrap();
    let extent = eredu_text::tokenizer::token_id_vocabulary(&tokenizer)
        .last_key_value()
        .map(|(&id, _)| id as usize + 1)
        .unwrap();
    let mut logits = vec![0.0; extent];
    // Make the decision to call a tool deterministic while exercising the
    // facade's real activation filter and all subsequent grammar masks.
    logits[start as usize] = 1.0;
    let runtime = ModelRuntime::prepare(
        MockBackend {
            logits,
            ..Default::default()
        },
        (),
    )
    .unwrap();
    let mut model = LoadedModel::from_runtime(
        runtime,
        tokenizer,
        LoadedTextModelConfig {
            model_family: ModelKind::Lfm2,
            effective_model_type: "lfm2".into(),
            model_id: path.display().to_string(),
            chat_template: Some(
                std::fs::read_to_string(path.join("chat_template.jinja"))
                    .unwrap()
                    .into(),
            ),
            eos_token_ids: vec![eos],
            checkpoint_generation_config: None,
        },
    );
    let prepared = model
        .prepare_chat(ChatTemplateRequest {
            messages: vec![json!({"role": "user", "content": "Look up value 7."})],
            tools: vec![json!({"type": "function", "function": {
                "name": "lookup",
                "parameters": {
                    "type": "object", "properties": {"value": {"type": "integer", "enum": [7]}},
                    "required": ["value"], "additionalProperties": false
                }
            }})],
            tool_choice: ToolChoice::Auto,
            parallel_tool_calls: ParallelToolCallPolicy::Disabled,
            add_generation_prompt: true,
            ..Default::default()
        })
        .unwrap();
    assert_eq!(
        prepared.format_profile_identity(),
        Some("lfm2.python-tools.v1")
    );
    let mut events = Vec::new();
    let output = model
        .generate_prepared_chat(PreparedChatGenerationRequest {
            input: PreparedChatInput::rendered_prompt(&prepared),
            settings: PreparedChatGenerationSettings {
                overrides: GenerationConfigOverrides {
                    max_new_tokens: Some(64),
                    ..Default::default()
                },
                ..Default::default()
            },
            caller_stop_sequences: &[],
            cancellation: Default::default(),
            on_event: |event| events.push(event),
        })
        .unwrap();
    assert_eq!(output.token_ids.first(), Some(&start));
    assert!(matches!(
        output.finish_reason,
        FinishReason::GrammarComplete | FinishReason::StopSequence
    ));
    assert!(
        matches!(events.first(), Some(SemanticEvent::ToolCallStart { index: 0, name, .. }) if name == "lookup")
    );
    let arguments = events
        .iter()
        .filter_map(|event| match event {
            SemanticEvent::ToolArgumentsDelta {
                index: 0,
                json_fragment,
            } => Some(json_fragment.as_str()),
            _ => None,
        })
        .collect::<String>();
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&arguments).unwrap(),
        json!({"value": 7})
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, SemanticEvent::ToolCallEnd))
            .count(),
        1
    );
    assert_eq!(
        events.last(),
        Some(&SemanticEvent::Finished {
            reason: output.finish_reason
        })
    );
}

#[test]
fn qwen_tool_generation_stops_at_eos_or_the_requested_call_limit() {
    use eredu::api::{
        PreparedChatGenerationRequest, PreparedChatGenerationSettings, PreparedChatInput,
    };
    use eredu::runtime::chat::{ChatTemplateRequest, ParallelToolCallPolicy, ToolChoice};
    use eredu_core::{FinishReason, SemanticEvent};
    use serde_json::json;
    use std::num::NonZeroUsize;
    use tokenizers::{decoders::byte_level::ByteLevel, models::bpe::BPE};

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
    let call = |value| {
        format!(
            "<tool_call>\n{{\"name\":\"ping\",\"arguments\":{{\"value\":{value}}}}}\n</tool_call>"
        )
    };
    let first = call(1);
    let both = format!("{first}\n{}", call(2));
    let parallel = |limit: Option<usize>| ParallelToolCallPolicy::Enabled {
        max_calls: limit.and_then(NonZeroUsize::new),
    };
    for template in [
        include_str!("fixtures/chat_templates/qwen2.5-7b-instruct-acbd9653.jinja"),
        include_str!("fixtures/chat_templates/qwen3-0.6b-7e4ae267.jinja"),
    ] {
        for tool_choice in [ToolChoice::Auto, ToolChoice::Required] {
            for (policy, scripted, expected, count, reason) in [
                (
                    ParallelToolCallPolicy::Disabled,
                    &both,
                    &first,
                    1,
                    FinishReason::GrammarComplete,
                ),
                (
                    parallel(Some(1)),
                    &both,
                    &first,
                    1,
                    FinishReason::GrammarComplete,
                ),
                (
                    parallel(Some(2)),
                    &both,
                    &both,
                    2,
                    FinishReason::GrammarComplete,
                ),
                (
                    parallel(Some(3)),
                    &both,
                    &both,
                    2,
                    FinishReason::StopSequence,
                ),
                (parallel(None), &both, &both, 2, FinishReason::StopSequence),
                (
                    parallel(Some(2)),
                    &first,
                    &first,
                    1,
                    FinishReason::StopSequence,
                ),
            ] {
                let backend = MockBackend::default();
                let calls = backend.calls.clone();
                calls.borrow_mut().scripted_tokens = tokenizer
                    .encode(format!("{scripted}<|im_end|>"), false)
                    .unwrap()
                    .get_ids()
                    .iter()
                    .copied()
                    .collect();
                let mut model = LoadedModel::from_runtime(
                    ModelRuntime::prepare(backend, ()).unwrap(),
                    ChatTokenizer::from_tokenizer(tokenizer.clone()),
                    LoadedTextModelConfig {
                        model_family: ModelKind::Qwen2,
                        effective_model_type: "qwen2".into(),
                        model_id: "scripted-qwen-tools".into(),
                        chat_template: Some(template.into()),
                        eos_token_ids: vec![eos],
                        checkpoint_generation_config: None,
                    },
                );
                let prepared = model
                    .prepare_chat(ChatTemplateRequest {
                        messages: vec![json!({"role": "user", "content": "Ping twice."})],
                        tools: vec![json!({"type": "function", "function": {
                            "name": "ping",
                            "parameters": {"type": "object", "properties": {
                                "value": {"type": "integer"}
                            }, "required": ["value"], "additionalProperties": false}
                        }})],
                        tool_choice,
                        parallel_tool_calls: policy,
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
                                max_new_tokens: Some(256),
                                ..Default::default()
                            },
                            ..Default::default()
                        },
                        caller_stop_sequences: &[],
                        cancellation: Default::default(),
                        on_event: |event| events.push(event),
                    })
                    .unwrap();
                let mut expected_tokens = tokenizer
                    .encode(expected.as_str(), false)
                    .unwrap()
                    .get_ids()
                    .to_vec();
                if reason == FinishReason::StopSequence {
                    expected_tokens.push(eos);
                }
                assert_eq!(output.token_ids, expected_tokens);
                assert_eq!(output.finish_reason, reason);
                assert_eq!(calls.borrow().filters.len(), expected_tokens.len());
                let starts: Vec<_> = events
                    .iter()
                    .filter_map(|event| match event {
                        SemanticEvent::ToolCallStart { index, name, .. } => {
                            Some((*index, name.as_str()))
                        }
                        _ => None,
                    })
                    .collect();
                assert_eq!(
                    starts,
                    (0..count).map(|index| (index, "ping")).collect::<Vec<_>>()
                );
                assert_eq!(
                    events
                        .iter()
                        .filter(|event| matches!(event, SemanticEvent::ToolCallEnd))
                        .count(),
                    count
                );
                assert!(!events
                    .iter()
                    .any(|event| matches!(event, SemanticEvent::TextDelta(_))));
                assert_eq!(events.last(), Some(&SemanticEvent::Finished { reason }));
            }
        }
    }
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
    for model_type in ["llama", "gemma4"] {
        std::fs::write(
            directory.join("config.json"),
            serde_json::json!({"model_type": model_type}).to_string(),
        )
        .unwrap();

        assert_eq!(load_tokenizer(&directory).unwrap().get_vocab_size(false), 2);
        let mut structural =
            ModelInspectionReport::unverified(&directory, ArtifactFormat::SafeTensors);
        structural.model_loadability = InspectionReadiness::Ready;
        structural.requested_load = InspectionReadiness::Ready;
        let report = inspect_text_model(
            structural,
            &eredu::api::TextModelOptions::default(),
            TextInspectionOptions::default(),
        );
        assert_eq!(report.tokenizer, InspectionReadiness::Ready);
        assert_eq!(report.text_generation, InspectionReadiness::Ready);
        assert_eq!(report.chat_template, InspectionReadiness::Missing);
    }

    std::fs::remove_dir_all(directory).unwrap();
}

fn sparse_vocabulary_model(logits: Vec<f32>) -> LoadedModel<MockBackend> {
    sparse_vocabulary_model_with_backend(
        MockBackend {
            logits,
            ..Default::default()
        },
        None,
    )
}

fn sparse_vocabulary_model_with_backend(
    backend: MockBackend,
    checkpoint_generation_config: Option<eredu_core::generation::CheckpointGenerationConfig>,
) -> LoadedModel<MockBackend> {
    let words = WordLevel::builder()
        .vocab(
            [
                ("[UNK]".to_owned(), 0),
                ("a".into(), 2),
                ("<|im_end|>".into(), 5),
            ]
            .into_iter()
            .collect(),
        )
        .unk_token("[UNK]".into())
        .build()
        .unwrap();
    let mut tokenizer = Tokenizer::new(words);
    tokenizer.with_decoder(Some(tokenizers::decoders::byte_level::ByteLevel::default()));
    tokenizer.with_pre_tokenizer(Some(tokenizers::pre_tokenizers::whitespace::Whitespace));
    tokenizer
        .add_special_tokens([AddedToken::from("<|im_end|>", true).normalized(false)])
        .unwrap();
    LoadedModel::from_runtime(
        ModelRuntime::prepare(backend, ()).unwrap(),
        ChatTokenizer::from_tokenizer(tokenizer),
        LoadedTextModelConfig {
            model_family: ModelKind::Qwen2,
            effective_model_type: "qwen2".into(),
            model_id: "sparse-vocabulary".into(),
            chat_template: Some(
                include_str!("fixtures/chat_templates/qwen2.5-7b-instruct-acbd9653.jinja").into(),
            ),
            eos_token_ids: vec![5],
            checkpoint_generation_config,
        },
    )
}

#[test]
fn ordinary_and_semantic_generation_exclude_holes_and_padded_logits() {
    use eredu::api::{
        PreparedChatGenerationRequest, PreparedChatGenerationSettings, PreparedChatInput,
    };
    use eredu::runtime::chat::ChatTemplateRequest;
    for (logits, expected) in [
        (vec![0.0, 100.0, 10.0, 200.0, 300.0, 1.0, 400.0, 500.0], 2),
        // A shorter output prefix still intersects the sparse tokenizer domain.
        (vec![0.0, 100.0, 10.0], 2),
        // A mapped EOS remains eligible despite larger invalid logits.
        (vec![0.0, 100.0, 1.0, 200.0, 300.0, 10.0, 400.0], 5),
    ] {
        let mut model = sparse_vocabulary_model(logits);
        let overrides = GenerationConfigOverrides {
            max_new_tokens: Some(2),
            ..Default::default()
        };
        let config = TextGenerationConfig::new(model.resolve_generation_config(overrides).unwrap());
        let ordinary: Vec<_> = model
            .generate_tokens(vec![2], config)
            .unwrap()
            .map(|token| token.unwrap().token_id().unwrap())
            .collect();
        assert_eq!(ordinary, [expected, expected]);
        let chat = model
            .prepare_chat(ChatTemplateRequest {
                messages: vec![serde_json::json!({"role": "user", "content": "a"})],
                add_generation_prompt: true,
                ..Default::default()
            })
            .unwrap();
        let semantic = model
            .generate_prepared_chat(PreparedChatGenerationRequest {
                input: PreparedChatInput::rendered_prompt(&chat),
                settings: PreparedChatGenerationSettings {
                    overrides,
                    seed: 0,
                    ..Default::default()
                },
                caller_stop_sequences: &[],
                cancellation: Default::default(),
                on_event: |_| {},
            })
            .unwrap();
        if expected == 5 {
            assert_eq!(semantic.token_ids, [5]);
            // The Qwen parser also recognizes this EOS spelling as a profile stop.
            assert_eq!(
                semantic.finish_reason,
                eredu_core::FinishReason::StopSequence
            );
        } else {
            assert_eq!(semantic.token_ids, [expected, expected]);
        }
    }
}

#[test]
fn ordinary_generation_fails_if_no_mapped_id_is_executable() {
    let words = WordLevel::builder()
        .vocab([("a".to_owned(), 4)].into_iter().collect())
        .build()
        .unwrap();
    let mut model = LoadedModel::from_runtime(
        ModelRuntime::prepare(
            MockBackend {
                logits: vec![100.0, 200.0],
                ..Default::default()
            },
            (),
        )
        .unwrap(),
        ChatTokenizer::from_tokenizer(Tokenizer::new(words)),
        LoadedTextModelConfig {
            model_family: ModelKind::Qwen2,
            effective_model_type: "qwen2".into(),
            model_id: "empty-intersection".into(),
            chat_template: None,
            eos_token_ids: vec![],
            checkpoint_generation_config: None,
        },
    );
    let config =
        TextGenerationConfig::new(model.resolve_generation_config(Default::default()).unwrap());
    assert!(model
        .generate_tokens(vec![0], config)
        .unwrap()
        .next()
        .unwrap()
        .is_err());
}
