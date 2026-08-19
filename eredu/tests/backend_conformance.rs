//! End-to-end conformance suite for a backend implemented without MLX.
//!
//! The client probes are generic over the facade contracts so new backend
//! implementations can reuse the same loading, generation, capability, media,
//! speculative, realtime, distributed, automatic-planning, and residency
//! realization call shapes.

use std::{
    collections::{BTreeMap, BTreeSet},
    convert::Infallible,
    io::Write,
    num::{NonZeroU8, NonZeroUsize},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

use eredu::{
    api::{
        ChatTemplateRequest, ChatTokenizer, LoadedModel, LoadedTextModelConfig, Media,
        ModelChatTemplate, MultimodalRequest, MultimodalSegment, PreparedChat, PreparedChatError,
        PreparedChatGenerationRequest, PreparedChatGenerationSettings, PreparedChatInput,
        PreparedChatMtpBatchLane, PreparedChatMtpBatchRequest, PreparedChatMtpGenerationOptions,
        PreparedChatMtpGenerationRequest, RgbImage,
    },
    core::{
        checkpoint::TensorDtype, BoundedResidencyRequirement, CandidateAdmission,
        MtpSchedulerStats, MtpStats,
    },
    load_realtime_model_with_options, AdmissionRequest, AdmissionResult, ArtifactFormat,
    AutomaticPlanRequest, AutomaticPlanner, AutomaticPlanningBackend, AutomaticPlanningError,
    Backend, BackendCapabilities, BackendDescriptor, BackendId, BackendSession, CacheStateStrategy,
    CapabilityError, CollectiveScope, DeviceDescriptor, DevicePlan, DistributedBackend,
    DistributedCapabilities, DistributedSession, DistributedSessionDescriptor, DraftPlacementPlan,
    DraftingPlan, EstimationCompleteness, ExecutionPlan, ExecutionPlanBackendFactory,
    ExecutionPlanTarget, ExternalDraftArtifact, FinishReason, GenerationConfigOverrides,
    GrowingState, HardwareBackendProfile, HardwareDeviceProfile, HardwareMemorySemantics,
    HardwareProfile, InputModalities, InputTokenCount, MemoryTier, ModelCapabilities,
    ModelCapabilityBackend, ModelKind, ModelLoadingBackend, ModelResourceProfile, ModelRuntime,
    MtpCapability, MtpCheckpointKind, MultimodalPreparationBackend, Observed, OffloadConfig,
    OffloadPlan, OffloadUnitId, OffloadUnitSpec, ParallelAxis, ParallelTopology,
    PhysicalMemorySemantics, PreparedModel, RealizedDrafting, RealtimeBackend,
    RealtimeModelLoadingBackend, RealtimeSampling, RealtimeScheduler, RealtimeSpeechConfig,
    RequestId, ResidencyLedger, ResidencyPlan, ResidencyPolicy, RuntimeStateEstimate,
    SchedulerLimits, SemanticEvent, SemanticStateTransaction, SpeculativeDraft,
    SpeculativeGenerationBackend, SpeculativeGenerationBatchOutput,
    SpeculativeGenerationBatchRequest, SpeculativeGenerationOutput, SpeculativeGenerationRequest,
    SpeculativeTokenFilterController, StateLayout, StaticMemoryReport, Submission,
    TextGenerationBackend, TextGenerationConfig, TokenFilter, TokenOutput, ValueDescriptor,
    WorkDescriptor, AUTOMATIC_SCHEMA_VERSION,
};
use eredu_core::Completion;
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
            "eredu-backend-conformance-{}-{}",
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
struct MockSession {
    distributed: MockDistributedSession,
}
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
        Ok(vec![(
            DeviceDescriptor {
                id: "gpu:0".into(),
                name: "mock accelerator".into(),
                family: "mock-accelerator".into(),
                memory_bytes: Some(8 * 1024),
            },
            BackendCapabilities {
                exact_completion: true,
                transfers: true,
                collectives: true,
                persistent_cache: true,
            },
        )])
    }

    fn prepare_model(
        &self,
        _: Self::ModelConfig,
    ) -> Result<PreparedModel<Self::Model>, Self::Error> {
        Ok(PreparedModel::new(()))
    }

    fn create_session(&self, _: PreparedModel<Self::Model>) -> Result<Self::Session, Self::Error> {
        Ok(MockSession {
            distributed: MockDistributedSession {
                descriptor: DistributedSessionDescriptor::new(
                    ParallelTopology::new(2, 1, 1, 1).unwrap(),
                    0,
                )
                .unwrap(),
            },
        })
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

#[derive(Debug, Clone, Eq, PartialEq)]
struct MockDistributedValue(Vec<u32>);

#[derive(Clone, Copy)]
struct MockDistributedSession {
    descriptor: DistributedSessionDescriptor,
}

impl DistributedSession for MockDistributedSession {
    type Value = MockDistributedValue;
    type Completion = Done;
    type Error = MockError;

    fn descriptor(&self) -> DistributedSessionDescriptor {
        self.descriptor
    }

    fn capabilities(&self) -> DistributedCapabilities {
        DistributedCapabilities {
            world_collectives: true,
            collective_axes: vec![ParallelAxis::Tensor],
            point_to_point: true,
            variable_all_to_all: true,
            exact_completion: true,
        }
    }

    fn all_reduce_sum(
        &self,
        scope: CollectiveScope,
        input: &Self::Value,
    ) -> Result<Submission<Self::Value, Self::Completion>, Self::Error> {
        assert_eq!(scope, CollectiveScope::Axis(ParallelAxis::Tensor));
        Ok(Submission {
            output: MockDistributedValue(input.0.iter().map(|value| value * 2).collect()),
            completion: Done,
        })
    }

    fn all_gather(
        &self,
        scope: CollectiveScope,
        input: &Self::Value,
    ) -> Result<Submission<Self::Value, Self::Completion>, Self::Error> {
        assert_eq!(scope, CollectiveScope::World);
        let mut output = input.0.clone();
        output.extend_from_slice(&input.0);
        Ok(Submission {
            output: MockDistributedValue(output),
            completion: Done,
        })
    }

    fn all_to_all_v(
        &self,
        scope: CollectiveScope,
        input: &Self::Value,
        send_counts: &[usize],
        receive_counts: &[usize],
    ) -> Result<Submission<Self::Value, Self::Completion>, Self::Error> {
        assert_eq!(scope, CollectiveScope::World);
        assert_eq!(send_counts, [1, 1]);
        assert_eq!(receive_counts, [1, 1]);
        Ok(Submission {
            output: input.clone(),
            completion: Done,
        })
    }

    fn send(
        &self,
        scope: CollectiveScope,
        peer: usize,
        input: &Self::Value,
    ) -> Result<Submission<Self::Value, Self::Completion>, Self::Error> {
        assert_eq!(scope, CollectiveScope::World);
        assert_eq!(peer, 1);
        Ok(Submission {
            output: input.clone(),
            completion: Done,
        })
    }

    fn receive(
        &self,
        scope: CollectiveScope,
        peer: usize,
        value: &ValueDescriptor,
    ) -> Result<Submission<Self::Value, Self::Completion>, Self::Error> {
        assert_eq!(scope, CollectiveScope::World);
        assert_eq!(peer, 1);
        assert_eq!(value.dtype, TensorDtype::U32);
        Ok(Submission {
            output: MockDistributedValue(vec![peer as u32; value.shape.iter().product()]),
            completion: Done,
        })
    }

    fn all_gather_words(&self, local: &[u32]) -> Result<Vec<u32>, Self::Error> {
        let mut gathered = local.to_vec();
        gathered.extend_from_slice(local);
        Ok(gathered)
    }
}

impl DistributedBackend for MockBackend {
    type DistributedSession = MockDistributedSession;

    fn distributed_session<'a>(session: &'a Self::Session) -> Option<&'a Self::DistributedSession> {
        Some(&session.distributed)
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
        request: &eredu::TokenizedMultimodalRequest,
        encode_backend_text: &mut dyn FnMut(&str) -> Result<Vec<u32>, E>,
    ) -> Result<Self::Prompt, eredu::MultimodalPreparationFailure<Self::Error, E>>
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        let mut prompt = Vec::new();
        for segment in request.segments() {
            match segment {
                eredu::TokenizedMultimodalSegment::TokenIds(ids) => {
                    prompt.extend_from_slice(ids);
                }
                eredu::TokenizedMultimodalSegment::Media(_) => prompt.push(2_001),
            }
        }
        prompt.extend(
            encode_backend_text("hello").map_err(eredu::MultimodalPreparationFailure::Text)?,
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
        eredu::core::estimate_runtime_state(
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
    ) -> Result<eredu_core::PreparationPolicy, Self::Error> {
        Ok(eredu_core::PreparationPolicy::default())
    }

    fn model_config(
        &self,
        plan: eredu_core::ModelPreparationPlan,
        _: Self::LoadOptions,
    ) -> Result<Self::ModelConfig, Self::Error> {
        assert_eq!(
            plan.inspection().configuration().kind,
            eredu_core::ModelKind::Llama
        );
        Ok(())
    }
}

impl AutomaticPlanningBackend for MockBackend {
    fn backend_id(&self) -> BackendId {
        BackendId::new("mock").unwrap()
    }

    fn discover_hardware(&self) -> Result<HardwareProfile, AutomaticPlanningError> {
        Ok(HardwareProfile {
            schema_version: AUTOMATIC_SCHEMA_VERSION,
            operating_system: "portable-test".into(),
            architecture: "mock".into(),
            logical_cpu_count: Observed::exact(4, "conformance fixture"),
            physical_memory_bytes: Observed::exact(64 * 1024, "conformance fixture"),
            available_memory_bytes: Observed::exact(64 * 1024, "conformance fixture"),
            physical_memory_semantics: HardwareMemorySemantics::SeparateTiers,
            backends: vec![HardwareBackendProfile {
                backend: self.backend_id(),
                available: true,
                detail: None,
                devices: vec![HardwareDeviceProfile {
                    id: "gpu:0".into(),
                    family: "mock-accelerator".into(),
                    index: 0,
                    total_memory_bytes: Observed::exact(8 * 1024, "conformance fixture"),
                    available_memory_bytes: Observed::exact(8 * 1024, "conformance fixture"),
                }],
            }],
        })
    }

    fn inspect_resources(
        &self,
        model_path: &Path,
    ) -> Result<ModelResourceProfile, AutomaticPlanningError> {
        assert!(model_path.join("model.safetensors").is_file());
        let mut profile =
            ModelResourceProfile::unmeasured(model_path.into(), ArtifactFormat::SafeTensors);
        profile.model_kind = Some(ModelKind::Llama);
        profile.architecture = Some("llama".into());
        profile.tensor_count = Some(1);
        profile.checkpoint_shards = Some(1);
        profile.stored_tensor_bytes = Observed::exact(4, "conformance fixture");
        profile.largest_stored_tensor_bytes = Observed::exact(4, "conformance fixture");
        profile.materialized_parameter_bytes = Observed::exact(16 * 1024, "conformance fixture");
        Ok(profile)
    }

    fn admit_candidate(
        &self,
        _: &Path,
        plan: &ExecutionPlan,
    ) -> Result<CandidateAdmission, AutomaticPlanningError> {
        assert_eq!(plan.device.backend, self.backend_id());
        let supported = plan.expert_cache.is_none();
        Ok(CandidateAdmission {
            supported,
            rejection: (!supported).then(|| "mock model has no routed experts".into()),
        })
    }

    fn bounded_residency_requirement(
        &self,
        _: &Path,
        plan: &ExecutionPlan,
    ) -> Result<BoundedResidencyRequirement, AutomaticPlanningError> {
        assert!(matches!(
            plan.residency,
            ResidencyPlan::LayerwiseHost { .. } | ResidencyPlan::DenseDiskStream { .. }
        ));
        Ok(BoundedResidencyRequirement {
            static_bytes: 1024,
            window_bytes: 2048,
            required_bytes: 3072,
            depth: 1,
        })
    }

    fn embedded_draft_layers(
        &self,
        _: &Path,
        model_kind: Option<ModelKind>,
    ) -> Result<Option<usize>, AutomaticPlanningError> {
        assert_eq!(model_kind, Some(ModelKind::Llama));
        Ok(Some(0))
    }
}

impl ExecutionPlanBackendFactory for MockBackend {
    type Backend = Self;
    type Drafter = MockDrafter;

    fn realize_target(
        &self,
        _: &ExecutionPlan,
    ) -> Result<ExecutionPlanTarget<Self::Backend>, AutomaticPlanningError> {
        Ok(ExecutionPlanTarget::new(MockBackend, ()))
    }

    fn realize_drafting(
        &self,
        plan: &ExecutionPlan,
        _: &ModelRuntime<Self::Backend>,
        external_artifact: Option<ExternalDraftArtifact>,
    ) -> Result<RealizedDrafting<MockDrafter>, AutomaticPlanningError> {
        Ok(match plan.drafting {
            DraftingPlan::Disabled => {
                assert!(external_artifact.is_none());
                RealizedDrafting::Disabled
            }
            DraftingPlan::Embedded { .. } => {
                assert!(external_artifact.is_none());
                RealizedDrafting::Embedded
            }
            DraftingPlan::External { .. } => {
                let artifact = external_artifact.expect("external drafting carries identities");
                assert_eq!(
                    artifact.target_tokenizer_fingerprint,
                    artifact.draft_tokenizer_fingerprint
                );
                RealizedDrafting::External(MockDrafter)
            }
        })
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

type DistributedValue<B> =
    <<B as DistributedBackend>::DistributedSession as DistributedSession>::Value;

struct DistributedProbe<V> {
    reduced: V,
    gathered: V,
    exchanged: V,
    sent: V,
    received: V,
    scheduler_words: Vec<u32>,
}

fn distributed_client_code<B: DistributedBackend>(
    runtime: &ModelRuntime<B>,
    input: &DistributedValue<B>,
) -> DistributedProbe<DistributedValue<B>> {
    let session = B::distributed_session(runtime.session())
        .expect("the conformance backend must expose its selected distributed session");
    let descriptor = session.descriptor();
    assert_eq!(descriptor.topology.world_size(), 2);
    assert_eq!(descriptor.rank, 0);
    let capabilities = session.capabilities();
    assert!(capabilities.world_collectives);
    assert_eq!(capabilities.collective_axes, vec![ParallelAxis::Tensor]);
    assert!(capabilities.point_to_point);
    assert!(capabilities.variable_all_to_all);
    assert!(capabilities.exact_completion);

    DistributedProbe {
        reduced: session
            .all_reduce_sum(CollectiveScope::Axis(ParallelAxis::Tensor), input)
            .unwrap()
            .wait()
            .unwrap(),
        gathered: session
            .all_gather(CollectiveScope::World, input)
            .unwrap()
            .wait()
            .unwrap(),
        exchanged: session
            .all_to_all_v(CollectiveScope::World, input, &[1, 1], &[1, 1])
            .unwrap()
            .wait()
            .unwrap(),
        sent: session
            .send(CollectiveScope::World, 1, input)
            .unwrap()
            .wait()
            .unwrap(),
        received: session
            .receive(
                CollectiveScope::World,
                1,
                &ValueDescriptor {
                    shape: vec![2],
                    dtype: TensorDtype::U32,
                },
            )
            .unwrap()
            .wait()
            .unwrap(),
        scheduler_words: session.all_gather_words(&[7, 9]).unwrap(),
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

fn automatic_planning_client_code<B: AutomaticPlanningBackend>(
    backend: &B,
    model_path: &Path,
    device: &str,
) -> eredu::ExecutionPlanReport {
    let request = AutomaticPlanRequest::new(
        model_path,
        DevicePlan::new(backend.backend_id().as_str(), device).unwrap(),
    );
    let report = AutomaticPlanner::default().plan(backend, &request).unwrap();
    assert_eq!(report.schema_version, AUTOMATIC_SCHEMA_VERSION);
    assert_eq!(report.plan.device.backend, backend.backend_id());
    assert_eq!(report.plan.device.device, device);
    report
}

fn planned_loading_client_code<F>(
    factory: &F,
    model_path: &Path,
    device: &str,
) -> (
    eredu::api::PlannedModel<F::Backend, F::Drafter>,
    eredu::ExecutionPlanReport,
)
where
    F: ExecutionPlanBackendFactory,
    F::Backend: TextGenerationBackend,
{
    let request = AutomaticPlanRequest::new(
        model_path,
        DevicePlan::new(factory.backend_id().as_str(), device).unwrap(),
    );
    LoadedModel::plan_and_load(factory, &AutomaticPlanner::default(), &request).unwrap()
}

#[derive(Debug, Eq, PartialEq)]
struct MockResidentCopy {
    id: OffloadUnitId,
    tier: MemoryTier,
    bytes: u64,
}

fn realize_residency_copy<R, E>(
    ledger: &mut ResidencyLedger,
    resources: &mut BTreeMap<(OffloadUnitId, MemoryTier), R>,
    id: &OffloadUnitId,
    tier: MemoryTier,
    materialize: impl FnOnce(&OffloadUnitSpec, MemoryTier) -> Result<R, E>,
    mut release: impl FnMut(R),
) -> Result<(), E> {
    let spec = ledger.spec(id).unwrap().clone();
    let evicted = ledger
        .reserve_copy(id, tier, spec.bytes(), &BTreeSet::new())
        .unwrap();
    for evicted in evicted {
        let resource = resources
            .remove(&(evicted.id, evicted.tier))
            .expect("every ledger copy must have one backend resource");
        release(resource);
    }

    let generation = ledger.next_transfer_generation().unwrap();
    let resource = match materialize(&spec, tier) {
        Ok(resource) => resource,
        Err(error) => {
            ledger.rollback_reserved(id, tier).unwrap();
            return Err(error);
        }
    };
    ledger
        .publish_reserved(id, tier, spec.bytes(), Some(generation))
        .unwrap();
    assert!(resources.insert((id.clone(), tier), resource).is_none());
    assert!(ledger
        .resolve_transfer(&[id.clone()], tier, generation, true)
        .unwrap()
        .is_empty());
    Ok(())
}

fn assert_automatic_planning_conformance() {
    let artifact = TestDirectory::new();
    write_loadable_text_artifact(artifact.path());
    let backend = MockBackend;
    let report = automatic_planning_client_code(&backend, artifact.path(), "gpu:0");
    assert!(matches!(
        &report.plan.residency,
        ResidencyPlan::LayerwiseHost { .. }
    ));
    assert_eq!(report.resources.pinned_parameter_bytes.value(), Some(&1024));
    assert_eq!(
        report.resources.largest_execution_group_bytes.value(),
        Some(&2048)
    );

    let requirement = backend
        .bounded_residency_requirement(artifact.path(), &report.plan)
        .unwrap();
    assert_eq!(requirement.required_bytes, 3072);
    assert_eq!(
        requirement.required_bytes,
        requirement.static_bytes + requirement.window_bytes
    );

    let (mut planned, realized_report) =
        planned_loading_client_code(&backend, artifact.path(), "gpu:0");
    assert_eq!(realized_report.plan, report.plan);
    let model = planned.model_mut();
    assert_eq!(model.runtime().backend().descriptor().name, "mock");
    assert_eq!(client_code(model), vec![1, 2, 3]);

    let mut wrong_backend = report.plan.clone();
    wrong_backend.device = DevicePlan::new("other", "gpu:0").unwrap();
    assert!(matches!(
        eredu::core::realize_execution_plan_target(&backend, &wrong_backend),
        Err(AutomaticPlanningError::Invalid(message))
            if message.contains("factory owns mock")
    ));

    let mut missing_device = report.plan.clone();
    missing_device.device = DevicePlan::new("mock", "gpu:1").unwrap();
    assert!(matches!(
        eredu::core::realize_execution_plan_target(&backend, &missing_device),
        Err(AutomaticPlanningError::Invalid(message))
            if message.contains("does not expose selected device gpu:1")
    ));

    let mut external = report.plan.clone();
    external.drafting = DraftingPlan::External {
        model: artifact.path().display().to_string(),
        placement: DraftPlacementPlan::Target,
        max_draft_tokens: 4,
        lookahead: true,
        adaptive_lookahead: true,
    };
    let mut planned = LoadedModel::load_execution_plan(&backend, artifact.path(), &external)
        .expect("generic plan loading realizes the external assistant");
    assert!(planned.drafting().is_external());
    let (_, drafting) = planned.parts_mut();
    assert!(matches!(
        drafting.as_speculative_draft(),
        Some(SpeculativeDraft::External(_))
    ));
}

fn assert_residency_realization_conformance() {
    let first = OffloadUnitId::new("layer.0").unwrap();
    let second = OffloadUnitId::new("layer.1").unwrap();
    let plan = OffloadPlan::new(
        OffloadConfig::new(Some(4), Some(8), 1).unwrap(),
        [
            OffloadUnitSpec::new(
                first.clone(),
                4,
                ResidencyPolicy::Cacheable,
                MemoryTier::Disk,
            )
            .unwrap(),
            OffloadUnitSpec::new(
                second.clone(),
                4,
                ResidencyPolicy::Cacheable,
                MemoryTier::Disk,
            )
            .unwrap(),
        ],
    )
    .unwrap();
    let mut ledger = ResidencyLedger::new(plan);
    let mut resources = BTreeMap::new();
    let mut released = Vec::new();
    for id in [&first, &second] {
        realize_residency_copy(
            &mut ledger,
            &mut resources,
            id,
            MemoryTier::Host,
            |spec, tier| {
                Ok::<_, &'static str>(MockResidentCopy {
                    id: spec.id().clone(),
                    tier,
                    bytes: spec.bytes(),
                })
            },
            |resource| released.push(resource),
        )
        .unwrap();
    }
    ledger.mark_initialized();
    ledger.require_initialized().unwrap();

    let error = realize_residency_copy(
        &mut ledger,
        &mut resources,
        &first,
        MemoryTier::Device,
        |_, _| Err("synthetic materialization failure"),
        |resource| released.push(resource),
    )
    .unwrap_err();
    assert_eq!(error, "synthetic materialization failure");
    assert!(!ledger.is_resident(&first, MemoryTier::Device).unwrap());

    for id in [&first, &second] {
        realize_residency_copy(
            &mut ledger,
            &mut resources,
            id,
            MemoryTier::Device,
            |spec, tier| {
                Ok::<_, &'static str>(MockResidentCopy {
                    id: spec.id().clone(),
                    tier,
                    bytes: spec.bytes(),
                })
            },
            |resource| released.push(resource),
        )
        .unwrap();
    }

    assert!(ledger.is_resident(&first, MemoryTier::Host).unwrap());
    assert!(ledger.is_resident(&second, MemoryTier::Host).unwrap());
    assert!(!ledger.is_resident(&first, MemoryTier::Device).unwrap());
    assert!(ledger.is_resident(&second, MemoryTier::Device).unwrap());
    assert_eq!(resources.len(), 3);
    assert_eq!(
        released,
        vec![MockResidentCopy {
            id: first,
            tier: MemoryTier::Device,
            bytes: 4,
        }]
    );
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

fn assert_distributed_conformance() {
    let runtime = ModelRuntime::prepare(MockBackend, ()).unwrap();
    let distributed = distributed_client_code(&runtime, &MockDistributedValue(vec![2, 3]));
    assert_eq!(distributed.reduced, MockDistributedValue(vec![4, 6]));
    assert_eq!(distributed.gathered, MockDistributedValue(vec![2, 3, 2, 3]));
    assert_eq!(distributed.exchanged, MockDistributedValue(vec![2, 3]));
    assert_eq!(distributed.sent, MockDistributedValue(vec![2, 3]));
    assert_eq!(distributed.received, MockDistributedValue(vec![1, 1]));
    assert_eq!(distributed.scheduler_words, vec![7, 9, 7, 9]);
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

#[test]
fn non_mlx_backend_conforms_to_distributed_sessions() {
    assert_distributed_conformance();
}

#[test]
fn non_mlx_backend_conforms_to_automatic_planning() {
    assert_automatic_planning_conformance();
}

#[test]
fn non_mlx_backend_conforms_to_residency_realization() {
    assert_residency_realization_conformance();
}
