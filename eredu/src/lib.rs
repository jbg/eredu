//! Backend-neutral language-model loading and generation facade.
//!
//! [`api`] and [`core`] remain available without an execution backend. The
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
/// Canonical backend-neutral runtime types.
pub use eredu_core as core;
pub use eredu_core::artifact::{
    plan_model_preparation, ArtifactFormat, ArtifactInspection, LoadingProtocol,
    MaterializationRoute, ModelArtifact, ModelConfiguration, ModelPreparationPlan,
    PreparationPolicy, QuantizationRequest, ResidencyRequest,
};
pub use eredu_core::generation::{
    CheckpointGenerationConfig, FinishReason, GenerationCancellationToken,
    GenerationConfigOverrides, MtpConfig, MtpRequestId, MtpRequestPhase, MtpSchedulerOptions,
    ResolvedGenerationConfig, SemanticEvent,
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
    InputModalities, InputTokenCount, InspectionIssue, InspectionIssueCode, InspectionReadiness,
    InspectionRequirement, InspectionSeverity, Media, MediaBinding, MediaRequestError,
    ModelCapabilities, ModelCapabilityBackend, ModelInspectionReport, ModelLoadError,
    ModelLoadingBackend, ModelResourceProfile, ModelRuntime, MtpCapability, MtpCheckpointKind,
    MtpTelemetry, MultimodalPreparationBackend, MultimodalPreparationFailure, MultimodalRequest,
    MultimodalSegment, ObservationKind, Observed, ParallelAxis, ParallelCoordinates,
    ParallelRankTopology, ParallelTopology, PhysicalMemorySemantics, PlanExplanation,
    PlanExplanationEntry, PlanExplanationLevel, PreparedModel, RealizedDrafting, RealtimeBackend,
    RealtimeCompletedStep, RealtimeConfigError, RealtimeError, RealtimeFrameConvention,
    RealtimeFrameForcing, RealtimeFrameScheduleState, RealtimeFrameSlot, RealtimeFrameTransition,
    RealtimeModel, RealtimeModelLoadingBackend, RealtimeSampling, RealtimeScheduleError,
    RealtimeScheduler, RealtimeSession, RealtimeSlotCoordinate, RealtimeSlotOccupancy,
    RealtimeSpeechConfig, RealtimeTargetDecision, RealtimeTargetSource, RealtimeTemporalSource,
    ResidencyPlan, ResidencyTelemetry, RgbImage, RuntimeStateEstimate, SlidingWindowLayerCount,
    SpeculativeDraft, SpeculativeGenerationBackend, SpeculativeGenerationBatchOutput,
    SpeculativeGenerationBatchRequest, SpeculativeGenerationLane, SpeculativeGenerationOutput,
    SpeculativeGenerationRequest, SpeculativeTokenFilterController, StateLayout,
    StateMemoryAssumptions, StaticMemoryReport, SubgroupMembership, Submission, TextGeneration,
    TextGenerationBackend, TextGenerationConfig, TextSamplingStrategy, TimingTelemetry,
    TokenFilter, TokenFilterController, TokenFilterError, TokenOutput, TokenizedMultimodalRequest,
    TokenizedMultimodalSegment, TopologyPreflightReport, TransferTelemetry, ValueDescriptor, Video,
    VideoSampling, WeightTransformationPlan, AUTOMATIC_SCHEMA_VERSION,
    EXECUTION_PLAN_SCHEMA_VERSION,
};
#[cfg(all(test, feature = "mlx-test-support"))]
mod test_utils;
#[cfg(all(test, feature = "mlx-test-support"))]
extern crate self as eredu;
#[cfg(all(test, feature = "mlx-test-support"))]
#[path = "../tests/distributed_cartesian_ring.rs"]
mod distributed_cartesian_ring;
#[cfg(all(test, feature = "mlx-test-support"))]
#[path = "../tests/distributed_expert_exchange_ring.rs"]
mod distributed_expert_exchange_ring;
#[cfg(all(test, feature = "mlx-test-support"))]
#[path = "../tests/distributed_partition_ring.rs"]
mod distributed_partition_ring;
#[cfg(all(test, feature = "mlx-test-support"))]
#[path = "../tests/distributed_pipeline_ring.rs"]
mod distributed_pipeline_ring;

pub use eredu_core::attention::{AttentionPolicy, LayerSchedule, LayerScheduleError};
pub use eredu_core::cache::{
    CacheBlockId, CachePolicyError, CacheRankIdentity, CacheRepresentation, CacheTier,
    LayerCachePolicy, MutableStateResidency, PoolingStateComponent, PromptCacheBlock,
    PromptCacheDescriptor, PromptCacheError, PromptCacheManifest, PromptCacheModelIdentity,
    PromptCacheOptions, PromptCacheStateTensor, PromptCacheTopology, StateResidencyClass,
    StateTensorDimension, StateTensorDtype, StateTensorOwner, StateTensorPolicy,
    StateTensorPresence, StateTensorRole, PROMPT_CACHE_SCHEMA_VERSION,
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
pub use eredu_runtime::{
    finalize_prompt_cache_shard, hash_prompt_cache_shard_payload, inspect_prompt_cache,
    resolve_prompt_cache_root, safe_prompt_cache_shard_path, validate_prompt_cache_manifest,
    CacheBlockLifecycle, CacheBlockStorage, CacheHostDemotionOperation, CacheHostPromotion,
    CacheIoAdmission, CacheIoCompletionDisposition, CacheIoExecutionState,
    CacheIoExecutionStateError, CacheIoOperation, CacheIoOperationKey, CacheIoOperationKind,
    CacheIoPreparation, CacheIoStartDisposition, CacheIoSubmission, CacheIoSubmissionOutcome,
    CacheIoTicket, CacheIoWorker, CacheIoWorkerError, CacheLayerResidencyReport,
    CacheLayerResidencyStats, CacheLifecycleError, CachePoolError, CachePoolLimits,
    CachePoolReport, CachePoolResource, CacheResidencyConfigurationError, CacheResidencyPolicy,
    CacheResidencyPool, CacheResidencyReport, CacheResidencyTelemetry, CacheStorageError,
    CacheStoragePhase, DenseDiskStreamLoadOptions, DenseDiskStreamReport, ExecutionResidency,
    ExpertCacheLoadOptions, ExpertIdentity, ExpertPass, ExpertWeightResidency,
    LayerWeightResidency, LayerwiseLoadOptions, LayerwiseModelMetadata, LiveCacheBlockPublication,
    LiveCacheDiskPolicy, LiveCachePublicationError, MutableCacheTail, NonExpertWeightResidency,
    PagedCacheOptions, ParallelModelInfo, PromptCachePersistenceError, PromptCachePublication,
    StaticUnitBindings, WeightResidency, WeightResidencyPolicyError,
    CACHE_RESIDENCY_LAYER_REPORT_LIMIT, DENSE_TRANSFER_WINDOW, MAX_PROMPT_CACHE_SHARD_HEADER_BYTES,
    PROMPT_CACHE_CURRENT_FILE, PROMPT_CACHE_GENERATIONS_DIRECTORY,
};
