//! End-to-end conformance suite for a backend implemented without MLX.
//!
//! The client probes are generic over the facade contracts so new backend
//! implementations can reuse the same loading, generation, capability, media,
//! speculative, and realtime call shapes.

use std::{
    convert::Infallible,
    io::Write,
    num::{NonZeroU8, NonZeroUsize},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

use safemlx_lm::{
    api::{
        ChatTemplateRequest, ChatTokenizer, LoadedModel, LoadedTextModelConfig, Media,
        ModelChatTemplate, MultimodalRequest, MultimodalSegment, PreparedChat, PreparedChatError,
        PreparedChatGenerationRequest, PreparedChatGenerationSettings, PreparedChatInput,
        PreparedChatMtpBatchLane, PreparedChatMtpBatchRequest, PreparedChatMtpGenerationOptions,
        PreparedChatMtpGenerationRequest, RgbImage,
    },
    core::{MtpSchedulerStats, MtpStats},
    load_realtime_model_with_options, AdmissionRequest, AdmissionResult, Backend,
    BackendCapabilities, BackendDescriptor, BackendSession, CacheStateStrategy, CapabilityError,
    DeviceDescriptor, EstimationCompleteness, FinishReason, GenerationConfigOverrides,
    GrowingState, InputModalities, InputTokenCount, ModelCapabilities, ModelCapabilityBackend,
    ModelLoadingBackend, ModelRuntime, MtpCapability, MtpCheckpointKind,
    MultimodalPreparationBackend, Observed, PhysicalMemorySemantics, PreparedModel,
    RealtimeBackend, RealtimeModelLoadingBackend, RealtimeSampling, RealtimeScheduler,
    RealtimeSpeechConfig, RequestId, RuntimeStateEstimate, SchedulerLimits, SemanticEvent,
    SemanticStateTransaction, SpeculativeDraft, SpeculativeGenerationBackend,
    SpeculativeGenerationBatchOutput, SpeculativeGenerationBatchRequest,
    SpeculativeGenerationOutput, SpeculativeGenerationRequest, SpeculativeTokenFilterController,
    StateLayout, StaticMemoryReport, Submission, TextGenerationBackend, TextGenerationConfig,
    TokenFilter, TokenOutput, WorkDescriptor,
};
use safemlx_lm_core::Completion;
use tokenizers::{
    decoders::byte_level::ByteLevel, models::wordlevel::WordLevel,
    pre_tokenizers::whitespace::Whitespace, AddedToken, Tokenizer,
};

const QWEN_TEMPLATE: &str =
    include_str!("fixtures/chat_templates/qwen2.5-7b-instruct-acbd9653.jinja");

struct TestDirectory(PathBuf);

impl TestDirectory {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "safemlx-lm-backend-conformance-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TestDirectory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

struct MockBackend;
struct MockSession;
struct Done;
#[derive(Clone)]
struct MockToken(u32);

#[derive(Debug, thiserror::Error)]
enum MockError {
    #[error("synthetic token extraction failure for token {0}")]
    Token(u32),
}

impl TokenOutput for MockToken {
    type Error = MockError;

    fn token_id(&self) -> Result<u32, Self::Error> {
        if self.0 == 999 {
            Err(MockError::Token(self.0))
        } else {
            Ok(self.0)
        }
    }
}

impl Completion for Done {
    type Error = MockError;

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
    type Error = MockError;

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
    ) -> Result<Submission<Self::Output, Self::Completion>, MockError> {
        Ok(Submission {
            output: input.len() as u32,
            completion: Done,
        })
    }

    fn decode(
        &mut self,
        _: &MockBackend,
        input: Self::DecodeInput,
    ) -> Result<Submission<Self::Output, Self::Completion>, MockError> {
        Ok(Submission {
            output: input + 1,
            completion: Done,
        })
    }
}

impl TextGenerationBackend for MockBackend {
    type Prompt = Vec<u32>;
    type Token = MockToken;
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
            output: MockToken(if submission.output == 999 {
                999
            } else {
                apply_filter(submission.output, filter)
            }),
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
            output: MockToken(apply_filter(submission.output, filter)),
            completion: submission.completion,
        })
    }
}

impl MultimodalPreparationBackend for MockBackend {
    fn prepare_multimodal_input<E>(
        _: &ModelRuntime<Self>,
        request: &safemlx_lm::TokenizedMultimodalRequest,
        encode_backend_text: &mut dyn FnMut(&str) -> Result<Vec<u32>, E>,
    ) -> Result<Self::Prompt, safemlx_lm::MultimodalPreparationFailure<Self::Error, E>>
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        let mut prompt = Vec::new();
        for segment in request.segments() {
            match segment {
                safemlx_lm::TokenizedMultimodalSegment::TokenIds(ids) => {
                    prompt.extend_from_slice(ids);
                }
                safemlx_lm::TokenizedMultimodalSegment::Media(_) => prompt.push(2_001),
            }
        }
        prompt.extend(
            encode_backend_text("hello").map_err(safemlx_lm::MultimodalPreparationFailure::Text)?,
        );
        Ok(prompt)
    }
}

impl ModelCapabilityBackend for MockBackend {
    fn model_capabilities(_: &ModelRuntime<Self>) -> Result<ModelCapabilities, CapabilityError> {
        Ok(ModelCapabilities {
            model_type: "llama".into(),
            native_max_context: Observed::exact(32, "mock configuration"),
            effective_max_context: Observed::exact(32, "mock configuration"),
            state_strategy: CacheStateStrategy::FullKv,
            modalities: InputModalities {
                image: true,
                ..InputModalities::TEXT
            },
            estimation: EstimationCompleteness::Complete,
        })
    }

    fn count_prepared_input(
        _: &ModelRuntime<Self>,
        input: &Self::Prompt,
    ) -> Result<InputTokenCount, CapabilityError> {
        Ok(InputTokenCount::text(input.len() as u64))
    }

    fn estimate_runtime_state(
        _: &ModelRuntime<Self>,
        input: InputTokenCount,
        max_output_tokens: u64,
        batch_size: u64,
    ) -> Result<RuntimeStateEstimate, CapabilityError> {
        safemlx_lm::core::estimate_runtime_state(
            &StateLayout {
                fixed_scalars_per_batch: 0,
                growing: vec![GrowingState {
                    layers: 1,
                    scalars_per_position: 2,
                    window: None,
                }],
                hidden_size: 1,
                allocation_granularity: 1,
                completeness: EstimationCompleteness::Complete,
            },
            input,
            max_output_tokens,
            batch_size,
            NonZeroU8::new(4).unwrap(),
        )
    }

    fn static_memory(_: &ModelRuntime<Self>) -> Result<StaticMemoryReport, CapabilityError> {
        let unavailable = || Observed::unavailable("mock counter is unavailable");
        Ok(StaticMemoryReport {
            logical_parameter_bytes: Observed::exact(128, "mock model"),
            current_host_resident_bytes: unavailable(),
            current_device_resident_bytes: unavailable(),
            planned_disk_backed_bytes: unavailable(),
            backend_active_allocation_bytes: unavailable(),
            backend_allocator_cache_bytes: unavailable(),
            physical_semantics: PhysicalMemorySemantics::Unknown,
            currently_mapped_shards: unavailable(),
        })
    }
}

impl ModelLoadingBackend for MockBackend {
    type LoadOptions = ();

    fn preparation_policy(
        &self,
        _: &Self::LoadOptions,
    ) -> Result<safemlx_lm_core::PreparationPolicy, Self::Error> {
        Ok(safemlx_lm_core::PreparationPolicy::default())
    }

    fn model_config(
        &self,
        plan: safemlx_lm_core::ModelPreparationPlan,
        _: Self::LoadOptions,
    ) -> Result<Self::ModelConfig, Self::Error> {
        assert_eq!(
            plan.inspection().configuration().kind,
            safemlx_lm_core::ModelKind::Llama
        );
        Ok(())
    }
}

struct MockDrafter;

impl SpeculativeGenerationBackend for MockBackend {
    type Drafter = MockDrafter;

    fn mtp_capability(_: &ModelRuntime<Self>) -> MtpCapability {
        MtpCapability::Ready {
            checkpoint: MtpCheckpointKind::Embedded,
        }
    }

    fn execute_speculative<C, F>(
        _: &mut ModelRuntime<Self>,
        request: SpeculativeGenerationRequest<'_, Self, Self::Drafter, C, F>,
    ) -> Result<SpeculativeGenerationOutput, MockError>
    where
        C: SpeculativeTokenFilterController,
        F: FnMut(SemanticEvent),
    {
        assert!(matches!(request.drafting, SpeculativeDraft::Embedded));
        assert!(!request.prompt.is_empty());
        assert_eq!(request.config.max_draft_tokens, 4);
        assert_eq!(request.generation.seed(), 0);
        request.constraint.filter_at(&[]).unwrap();
        let mut on_event = request.on_event;
        let mut semantic = request.semantic;
        semantic.fork_box().unwrap();
        semantic.finish(FinishReason::MaxTokens).unwrap();
        for event in semantic.take_events() {
            on_event(event);
        }
        Ok(SpeculativeGenerationOutput {
            token_ids: vec![7, 11],
            finish_reason: FinishReason::MaxTokens,
            stats: MtpStats::default(),
        })
    }

    fn execute_speculative_batch<C>(
        _: &mut ModelRuntime<Self>,
        request: SpeculativeGenerationBatchRequest<'_, Self, Self::Drafter, C>,
    ) -> Result<SpeculativeGenerationBatchOutput, MockError>
    where
        C: SpeculativeTokenFilterController,
    {
        assert!(matches!(request.drafting, SpeculativeDraft::Embedded));
        let requests = request
            .lanes
            .into_iter()
            .map(|mut lane| {
                assert!(!lane.prompt.is_empty());
                assert_eq!(lane.config.max_draft_tokens, 2);
                lane.semantic.finish(FinishReason::MaxTokens).unwrap();
                for event in lane.semantic.take_events() {
                    (lane.on_event)(event);
                }
                SpeculativeGenerationOutput {
                    token_ids: vec![13],
                    finish_reason: FinishReason::MaxTokens,
                    stats: MtpStats::default(),
                }
            })
            .collect();
        Ok(SpeculativeGenerationBatchOutput {
            requests,
            scheduler: MtpSchedulerStats::default(),
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

fn multimodal_client_code<B: MultimodalPreparationBackend>(model: &LoadedModel<B>) -> B::Prompt {
    let request = MultimodalRequest::new(vec![
        MultimodalSegment::Text("hello".into()),
        MultimodalSegment::Media(Media::Image(RgbImage::new(vec![10, 20, 30], 1, 1).unwrap())),
        MultimodalSegment::TokenIds(vec![7]),
    ])
    .unwrap();
    model.prepare_multimodal_input(&request).unwrap()
}

fn capability_client_code<B: ModelCapabilityBackend>(model: &LoadedModel<B>, prepared: &B::Prompt) {
    let capabilities = model.capabilities().unwrap();
    assert_eq!(capabilities.model_type, model.model_type());
    assert_eq!(model.count_token_ids(&[1, 2]).unwrap().model_positions, 2);
    assert_eq!(model.count_text("hello", false).unwrap().text_tokens, 1);
    let input = model.count_prepared_input(prepared).unwrap();
    let state = model.estimate_runtime_state(input, 3, 1).unwrap();
    assert_eq!(
        state.assumptions.requested_positions,
        input.model_positions + 3
    );
    assert_eq!(
        model
            .static_memory()
            .unwrap()
            .logical_parameter_bytes
            .value(),
        Some(&128)
    );
    assert!(matches!(
        model
            .admit(
                AdmissionRequest {
                    input,
                    max_output_tokens: 3,
                    batch_size: 1,
                    safety_reserve_bytes: 0,
                    application_memory_budget_bytes: None,
                    require_complete_estimate: true,
                },
                None,
            )
            .unwrap(),
        AdmissionResult::Admitted(_)
    ));
}

fn speculative_client_code<B: SpeculativeGenerationBackend>(
    model: &mut LoadedModel<B>,
    prepared: &PreparedChat,
) -> (SpeculativeGenerationOutput, Vec<SemanticEvent>) {
    assert!(matches!(
        model.mtp_capability(),
        MtpCapability::Ready { .. }
    ));
    let mut events = Vec::new();
    let output = model
        .generate_prepared_chat_mtp(PreparedChatMtpGenerationRequest {
            input: PreparedChatInput::rendered_prompt(prepared),
            drafting: SpeculativeDraft::Embedded,
            settings: PreparedChatGenerationSettings::default(),
            options: PreparedChatMtpGenerationOptions::default(),
            caller_stop_sequences: &[],
            cancellation: Default::default(),
            on_event: |event| events.push(event),
        })
        .unwrap();
    (output, events)
}

fn speculative_batch_client_code<B: SpeculativeGenerationBackend>(
    model: &mut LoadedModel<B>,
    prepared: &PreparedChat,
) -> (SpeculativeGenerationBatchOutput, Vec<SemanticEvent>) {
    let mut events = Vec::new();
    let output = model
        .generate_prepared_chat_mtp_batch(PreparedChatMtpBatchRequest {
            drafting: SpeculativeDraft::Embedded,
            lanes: vec![PreparedChatMtpBatchLane {
                input: PreparedChatInput::rendered_prompt(prepared),
                settings: PreparedChatGenerationSettings::default(),
                max_draft_tokens: NonZeroUsize::new(2).unwrap(),
                caller_stop_sequences: &[],
                cancellation: Default::default(),
                on_event: Box::new(|event| events.push(event)),
            }],
            scheduler: Default::default(),
        })
        .unwrap();
    (output, events)
}

fn write_loadable_text_artifact(root: &std::path::Path) {
    std::fs::write(
        root.join("config.json"),
        r#"{"model_type":"llama","eos_token_id":0}"#,
    )
    .unwrap();
    let header = br#"{"token_embd.weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
    let mut weights = std::fs::File::create(root.join("model.safetensors")).unwrap();
    weights
        .write_all(&(header.len() as u64).to_le_bytes())
        .unwrap();
    weights.write_all(header).unwrap();
    weights.write_all(&[0; 4]).unwrap();

    let vocabulary = [("[UNK]".to_owned(), 0), ("hello".to_owned(), 1)]
        .into_iter()
        .collect();
    let tokenizer = WordLevel::builder()
        .vocab(vocabulary)
        .unk_token("[UNK]".into())
        .build()
        .unwrap();
    Tokenizer::new(tokenizer)
        .save(root.join("tokenizer.json"), false)
        .unwrap();
}

fn assert_loading_generation_capability_and_multimodal_conformance() {
    let artifact = TestDirectory::new();
    write_loadable_text_artifact(artifact.path());

    let mut model = LoadedModel::load(MockBackend, artifact.path(), ()).unwrap();

    assert_eq!(model.model_type(), "llama");
    assert_eq!(model.eos_token_ids(), &[0]);
    let prepared = multimodal_client_code(&model);
    assert_eq!(prepared, vec![1, 2_001, 7, 1]);
    capability_client_code(&model, &prepared);
    assert_eq!(client_code(&mut model), vec![1, 2, 3]);
}

fn assert_prepared_generation_and_speculative_conformance() {
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
    assert!(
        model
            .count_prepared_chat(&prepared)
            .unwrap()
            .model_positions
            > 0
    );
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

    let error = model
        .generate_prepared_chat(PreparedChatGenerationRequest {
            input: PreparedChatInput::prepared_backend_input(&prepared, vec![0; 999]),
            settings: PreparedChatGenerationSettings {
                overrides: GenerationConfigOverrides {
                    max_new_tokens: Some(1),
                    ..Default::default()
                },
                seed: 9,
            },
            caller_stop_sequences: &[],
            cancellation: Default::default(),
            on_event: |_| {},
        })
        .unwrap_err();
    assert!(matches!(
        error,
        PreparedChatError::Backend(MockError::Token(999))
    ));

    let (speculative, speculative_events) = speculative_client_code(&mut model, &prepared);
    assert_eq!(speculative.token_ids, vec![7, 11]);
    assert_eq!(speculative.finish_reason, FinishReason::MaxTokens);
    assert_eq!(
        speculative_events,
        vec![SemanticEvent::Finished {
            reason: FinishReason::MaxTokens
        }]
    );

    let (batch, batch_events) = speculative_batch_client_code(&mut model, &prepared);
    assert_eq!(batch.requests.len(), 1);
    assert_eq!(batch.requests[0].token_ids, vec![13]);
    assert_eq!(
        batch_events,
        vec![SemanticEvent::Finished {
            reason: FinishReason::MaxTokens
        }]
    );
}

#[derive(Clone)]
struct MockRealtimeSession {
    step: u32,
    sampling: RealtimeSampling,
}

impl SemanticStateTransaction for MockRealtimeSession {
    type Branch = Self;
    type Error = Infallible;

    fn branch(&self) -> Result<Self::Branch, Self::Error> {
        Ok(self.clone())
    }

    fn commit_branch(&mut self, branch: Self::Branch) -> Result<(), Self::Error> {
        *self = branch;
        Ok(())
    }
}

struct MockFrame(Vec<u32>);

impl WorkDescriptor for MockFrame {
    type Error = Infallible;

    fn encode_descriptor(&self, output: &mut Vec<u32>) -> Result<(), Self::Error> {
        output.extend_from_slice(&self.0);
        Ok(())
    }
}

struct RealtimeDone;

impl Completion for RealtimeDone {
    type Error = Infallible;

    fn is_complete(&self) -> Result<bool, Self::Error> {
        Ok(true)
    }

    fn wait(&self) -> Result<(), Self::Error> {
        Ok(())
    }
}

struct MockRealtimeBackend;

impl RealtimeBackend for MockRealtimeBackend {
    type Model = u64;
    type ModelIdentity = u64;
    type Input = MockFrame;
    type Output = u32;
    type Session = MockRealtimeSession;
    type Completion = RealtimeDone;
    type Error = Infallible;

    fn name(&self) -> &str {
        "portable-mock-realtime"
    }

    fn model_identity(&self, model: &Self::Model) -> Self::ModelIdentity {
        *model
    }

    fn speech_config(&self, _: &Self::Model) -> RealtimeSpeechConfig {
        RealtimeSpeechConfig::new(2, 1, 1, 1, 0, 0, vec![0, 1]).unwrap()
    }

    fn create_session(
        &self,
        _: &Self::Model,
        sampling: RealtimeSampling,
    ) -> Result<Self::Session, Self::Error> {
        Ok(MockRealtimeSession { step: 0, sampling })
    }

    fn validate_session(&self, _: &Self::Model, _: &Self::Session) -> Result<(), Self::Error> {
        Ok(())
    }

    fn validate_input(&self, _: &Self::Model, _: &Self::Input) -> Result<(), Self::Error> {
        Ok(())
    }

    fn input_batch_size(&self, input: &Self::Input) -> usize {
        input.0.len()
    }

    fn set_sampling(
        &self,
        session: &mut Self::Session,
        sampling: RealtimeSampling,
    ) -> Result<(), Self::Error> {
        session.sampling = sampling;
        Ok(())
    }

    fn submit_step(
        &self,
        model: &mut Self::Model,
        session: &mut Self::Session,
        input: &Self::Input,
    ) -> Result<Submission<Self::Output, Self::Completion>, Self::Error> {
        session.step += 1;
        Ok(Submission {
            output: *model as u32 + session.step + input.0.iter().sum::<u32>(),
            completion: RealtimeDone,
        })
    }
}

impl RealtimeModelLoadingBackend for MockRealtimeBackend {
    type LoadOptions = u64;

    fn prepare_realtime_model(
        &self,
        _: &Path,
        options: Self::LoadOptions,
    ) -> Result<Self::Model, Self::Error> {
        Ok(options)
    }
}

fn assert_realtime_conformance() {
    let mut model =
        load_realtime_model_with_options(MockRealtimeBackend, "mock-realtime-artifact", 23)
            .unwrap();
    assert_eq!(model.backend().name(), "portable-mock-realtime");
    assert_eq!(*model.model(), 23);
    assert_eq!(model.speech_config().generated_audio_codebooks(), 1);

    let limits = SchedulerLimits::with_execution_bounds(1, 2, 1, 1, 1, usize::MAX).unwrap();
    let mut scheduler = RealtimeScheduler::new(&model, limits).unwrap();
    let request = RequestId::new(7);
    scheduler
        .register_request(&model, request, RealtimeSampling::greedy())
        .unwrap();
    scheduler
        .enqueue(&model, request, MockFrame(vec![2]))
        .unwrap();
    let completed = scheduler.run_queued(&mut model).unwrap();
    assert_eq!(completed.len(), 1);
    assert_eq!(completed.into_iter().next().unwrap().into_parts().1, 26);

    let sampling = RealtimeSampling::new(0.5, 0.75, 11).unwrap();
    scheduler
        .set_request_sampling(&model, request, sampling)
        .unwrap();
    let session = scheduler.release_request(request).unwrap();
    assert_eq!(session.state().sampling, sampling);
}

#[test]
fn non_mlx_backend_conforms_to_the_complete_generic_facade() {
    assert_loading_generation_capability_and_multimodal_conformance();
    assert_prepared_generation_and_speculative_conformance();
    assert_realtime_conformance();
}
