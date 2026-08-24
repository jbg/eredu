//! Backend-neutral model loading, generation, and realtime facade.
//!
//! [`api`] and [`runtime`] remain available without an execution backend. The
//! default `mlx` feature adds the MLX model implementations and runtime.

#![warn(missing_docs)]
#![cfg_attr(test, allow(dead_code))]
// Backend execution boundaries intentionally pass complete runtime context;
// concrete models remain unboxed; builder helpers can return typed builders;
// explicit drops delimit provider borrows; and MLX completions are process-local.
#![allow(
    clippy::arc_with_non_send_sync,
    clippy::drop_non_drop,
    clippy::large_enum_variant,
    clippy::new_ret_no_self,
    clippy::too_many_arguments,
    clippy::type_complexity
)]

/// High-level model loading, dispatch, and request APIs.
pub mod api;
/// Backend-independent chat and committed-generation orchestration.
pub mod runtime;
pub use api::{inspect_text_model, PlannedModelLoadError, TextInspectionOptions};
pub use eredu_architectures::configuration::inspect_artifact;
pub use eredu_architectures::moshi::{
    prepare_realtime_model, RealtimePreparationError, RealtimePreparationPlan,
};
pub use eredu_architectures::{
    prepare_external_assistant, ExternalAssistantCheckpoint, ExternalAssistantPreparationPlan,
};
pub use eredu_architectures::{GgufArchitecture, ModelKind};
pub use eredu_checkpoint::{
    store::{WeightStoreBackend, WeightStoreDiagnostics},
    AffineQuantization, WeightQuantization,
};
pub use eredu_core::artifact::{
    plan_model_preparation, ArtifactFormat, ArtifactInspection, LoadingProtocol,
    MaterializationRoute, ModelArtifact, ModelConfiguration, ModelPreparationPlan,
    PreparationPolicy, QuantizationRequest, ResidencyRequest,
};
pub use eredu_core::attention::{AttentionPolicy, LayerSchedule, LayerScheduleError};
pub use eredu_core::cache::{
    CacheBlockId, CachePolicyError, CacheRankIdentity, CacheRepresentation, CacheTier,
    LayerCachePolicy, MutableStateResidency, PoolingStateComponent, PromptCacheBlock,
    PromptCacheDescriptor, PromptCacheError, PromptCacheManifest, PromptCacheModelIdentity,
    PromptCacheOptions, PromptCacheStateTensor, PromptCacheTopology, StateResidencyClass,
    StateTensorDimension, StateTensorDtype, StateTensorOwner, StateTensorPolicy,
    StateTensorPresence, StateTensorRole, PROMPT_CACHE_SCHEMA_VERSION,
};
pub use eredu_core::generation::{
    CheckpointGenerationConfig, FinishReason, GenerationCancellationToken,
    GenerationConfigOverrides, MtpConfig, MtpRequestId, MtpRequestPhase, MtpSchedulerOptions,
    ResolvedGenerationConfig, SemanticEvent,
};
pub use eredu_core::residency::{
    AllocatorMemoryMetrics, BackgroundPrefetchReport, CacheEvictionPolicy, EvictionMetrics,
    MemoryTier, OffloadConfig, OffloadError, OffloadPlan, OffloadReport, OffloadTelemetry,
    OffloadUnitId, OffloadUnitSpec, PrefetchAdmission, PrefetchCompletion,
    PrefetchDemandObservation, PrefetchDemandResolution, PrefetchExecutionState, PrefetchMetrics,
    PrefetchOutcome, PrefetchStateError, PrefetchWork, ProcessMetrics, ResidencyBlocker,
    ResidencyLedger, ResidencyLedgerError, ResidencyPolicy, TierByteTotals, TierUnitTotals,
    TransferDirection, TransferMetrics, UnitResidencyReport, OFFLOAD_PLAN_SCHEMA_VERSION,
};
pub use eredu_core::scheduler::{
    CancellationCause, RequestId, RequestStatus, Scheduler, SchedulerCapabilities, SchedulerError,
    SchedulerLimits, SchedulerProgress, SchedulerReport, SemanticStateTransaction,
    TransitionOutput, WorkDescriptor, WorkId, WorkLifecycle,
};
pub use eredu_core::{
    load_model, load_realtime_model, load_realtime_model_with_options, Admission,
    AdmissionRejection, AdmissionRequest, AdmissionResult, AllocatorTelemetry, ArtifactModality,
    ArtifactTensorEncoding, Audio, AutomaticPlanRequest, AutomaticPlanner, AutomaticPlannerPolicy,
    AutomaticPlanningBackend, AutomaticPlanningError, AvailableMemory, BackendCapabilities,
    BackendDescriptor, BackendError, BackendId, BackendProvider, BackendSession,
    CacheStateStrategy, CapabilityError, CollectiveScope, Completion, ControlledTextGeneration,
    ControlledTextGenerationError, ControlledToken, DeviceDescriptor, DevicePlan,
    DistributedBackend, DistributedCapabilities, DistributedSession, DistributedSessionDescriptor,
    DraftPlacementPlan, DraftingPlan, DurationSeconds, EstimationCompleteness, ExecutionPlan,
    ExecutionPlanBackendFactory, ExecutionPlanReport, ExecutionPlanTarget, ExecutionTelemetry,
    ExpertCachePlan, ExpertCacheTelemetry, ExternalDraftArtifact, GrowingState,
    HardwareBackendProfile, HardwareDeviceProfile, HardwareMemorySemantics, HardwareProfile,
    InputModalities, InputTokenCount, InspectableBackendSession, InspectedOutput, InspectionIssue,
    InspectionIssueCode, InspectionReadiness, InspectionRequirement, InspectionSeverity, Media,
    MediaBinding, MediaRequestError, ModelCapabilities, ModelCapabilityBackend,
    ModelInspectionReport, ModelLoadError, ModelLoadingBackend, ModelResourceProfile, ModelRuntime,
    MtpCapability, MtpCheckpointKind, MtpTelemetry, MultimodalPreparationBackend,
    MultimodalPreparationFailure, MultimodalRequest, MultimodalSegment, ObservationError,
    ObservationKind, ObservationRequest, ObservationSelector, ObservationSet, ObservationValue,
    Observed, ParallelAxis, ParallelCoordinates, ParallelRankTopology, ParallelTopology,
    PhysicalMemorySemantics, PlanExplanation, PlanExplanationEntry, PlanExplanationLevel,
    PreparedModel, RealizedDrafting, RealtimeBackend, RealtimeCompletedStep, RealtimeConfigError,
    RealtimeDecisionDiagnostics, RealtimeError, RealtimeFrameConvention, RealtimeFrameForcing,
    RealtimeFrameScheduleState, RealtimeFrameSlot, RealtimeFrameTransition, RealtimeInputFrame,
    RealtimeModel, RealtimeModelLoadingBackend, RealtimeOutputFrame, RealtimeSampling,
    RealtimeScheduleError, RealtimeScheduler, RealtimeSession, RealtimeSlotCoordinate,
    RealtimeSlotOccupancy, RealtimeSpeechConfig, RealtimeTargetDecision, RealtimeTargetSource,
    RealtimeTemporalSource, ResidencyPlan, ResidencyTelemetry, RgbImage, RuntimeStateEstimate,
    SlidingWindowLayerCount, SpeculativeDraft, SpeculativeGenerationBackend,
    SpeculativeGenerationBatchOutput, SpeculativeGenerationBatchRequest, SpeculativeGenerationLane,
    SpeculativeGenerationOutput, SpeculativeTokenFilterController, StateLayout,
    StateMemoryAssumptions, StaticMemoryReport, SubgroupMembership, Submission, TensorObservation,
    TensorObservationData, TextGeneration, TextGenerationBackend, TextGenerationConfig,
    TextSamplingStrategy, TimingTelemetry, TokenFilter, TokenFilterController, TokenFilterError,
    TokenOutput, TokenizedMultimodalRequest, TokenizedMultimodalSegment,
    TokenizerCompatibilityError, TokenizerCompatibilityProof, TopologyPreflightReport,
    TransferTelemetry, ValueDescriptor, Video, VideoSampling, WeightTransformationPlan,
    AUTOMATIC_SCHEMA_VERSION, EXECUTION_PLAN_SCHEMA_VERSION,
};
