//! Backend-neutral language-model loading and generation facade.
//!
//! [`api`] and [`core`] remain available without an execution backend. The
//! default `mlx` feature adds the MLX model implementations and runtime.

#![warn(missing_docs)]

/// High-level model loading, dispatch, and request APIs.
pub mod api;
/// Execution-backend implementations selected by crate features.
pub mod backend;
/// Backend-independent chat and committed-generation orchestration.
pub mod runtime;
pub use api::{inspect_text_model, TextInspectionOptions};
/// Canonical backend-neutral runtime types.
pub use safemlx_lm_core as core;
pub use safemlx_lm_core::artifact::{
    inspect_artifact, plan_model_preparation, ArtifactFormat, ArtifactInspection, GgufArchitecture,
    MaterializationRoute, ModelArtifact, ModelConfiguration, ModelKind, ModelPreparationPlan,
    PreparationPolicy, QuantizationRequest, ResidencyRequest,
};
pub use safemlx_lm_core::generation::{
    CheckpointGenerationConfig, FinishReason, GenerationCancellationToken,
    GenerationConfigOverrides, MtpConfig, MtpRequestId, MtpRequestPhase, MtpSchedulerOptions,
    ResolvedGenerationConfig, SemanticEvent,
};
pub use safemlx_lm_core::{
    load_model, load_realtime_model, load_realtime_model_with_options, Admission,
    AdmissionRejection, AdmissionRequest, AdmissionResult, AllocatorTelemetry, ArtifactModality,
    ArtifactTensorEncoding, Audio, AutomaticPlanRequest, AutomaticPlanner, AutomaticPlannerPolicy,
    AutomaticPlanningBackend, AutomaticPlanningError, AvailableMemory, Backend,
    BackendCapabilities, BackendDescriptor, BackendError, BackendId, BackendSession,
    CacheStateStrategy, CapabilityError, CollectiveScope, Completion, ControlledTextGeneration,
    ControlledTextGenerationError, ControlledToken, DeviceDescriptor, DevicePlan,
    DistributedBackend, DistributedCapabilities, DistributedSession, DistributedSessionDescriptor,
    DraftPlacementPlan, DraftingPlan, DurationSeconds, EstimationCompleteness, ExecutionPlan,
    ExecutionPlanReport, ExecutionTelemetry, ExpertCachePlan, ExpertCacheTelemetry, GrowingState,
    HardwareBackendProfile, HardwareDeviceProfile, HardwareMemorySemantics, HardwareProfile,
    InputModalities, InputTokenCount, InspectionIssue, InspectionIssueCode, InspectionReadiness,
    InspectionRequirement, InspectionSeverity, Media, MediaBinding, MediaRequestError,
    ModelCapabilities, ModelCapabilityBackend, ModelInspectionReport, ModelLoadError,
    ModelLoadingBackend, ModelResourceProfile, ModelRuntime, MtpCapability, MtpCheckpointKind,
    MtpTelemetry, MultimodalPreparationBackend, MultimodalPreparationFailure, MultimodalRequest,
    MultimodalSegment, ObservationKind, Observed, ParallelAxis, ParallelCoordinates,
    ParallelRankTopology, ParallelTopology, PhysicalMemorySemantics, PlanExplanation,
    PlanExplanationEntry, PlanExplanationLevel, PreparedModel, RealtimeBackend,
    RealtimeCompletedStep, RealtimeConfigError, RealtimeError, RealtimeModel,
    RealtimeModelLoadingBackend, RealtimeSampling, RealtimeScheduler, RealtimeSession,
    RealtimeSpeechConfig, ResidencyPlan, ResidencyTelemetry, RgbImage, RuntimeStateEstimate,
    SlidingWindowLayerCount, SpeculativeTokenFilterController, StateLayout, StateMemoryAssumptions,
    StaticMemoryReport, SubgroupMembership, Submission, TextGeneration, TextGenerationBackend,
    TextGenerationConfig, TimingTelemetry, TokenFilter, TokenFilterController, TokenFilterError,
    TokenOutput, TokenizedMultimodalRequest, TokenizedMultimodalSegment, TopologyPreflightReport,
    TransferTelemetry, ValueDescriptor, Video, VideoSampling, WeightTransformationPlan,
    AUTOMATIC_SCHEMA_VERSION,
};
#[cfg(all(test, feature = "mlx"))]
mod test_utils;
#[cfg(all(test, feature = "mlx"))]
extern crate self as safemlx_lm;
#[cfg(all(test, feature = "mlx"))]
#[path = "../tests/distributed_cartesian_ring.rs"]
mod distributed_cartesian_ring;
#[cfg(all(test, feature = "mlx"))]
#[path = "../tests/distributed_expert_exchange_ring.rs"]
mod distributed_expert_exchange_ring;
#[cfg(all(test, feature = "mlx"))]
#[path = "../tests/distributed_expert_parallel_ring.rs"]
mod distributed_expert_parallel_ring;
#[cfg(all(test, feature = "mlx"))]
#[path = "../tests/distributed_gemma4_multimodal_pipeline_ring.rs"]
mod distributed_gemma4_multimodal_pipeline_ring;
#[cfg(all(test, feature = "mlx"))]
#[path = "../tests/distributed_partition_ring.rs"]
mod distributed_partition_ring;
#[cfg(all(test, feature = "mlx"))]
#[path = "../tests/distributed_pipeline_ring.rs"]
mod distributed_pipeline_ring;
#[cfg(all(test, feature = "mlx"))]
#[path = "../tests/distributed_qwen3_vl_pipeline_ring.rs"]
mod distributed_qwen3_vl_pipeline_ring;
#[cfg(all(test, feature = "mlx"))]
#[path = "../tests/distributed_tensor_parallel_ring.rs"]
mod distributed_tensor_parallel_ring;

pub use safemlx_lm_core::attention::{AttentionPolicy, LayerSchedule, LayerScheduleError};
pub use safemlx_lm_core::cache::{
    CacheBlockId, CacheBlockLifecycle, CacheBlockStorage, CacheHostDemotionOperation,
    CacheHostPromotion, CacheIoAdmission, CacheIoCompletionDisposition, CacheIoExecutionState,
    CacheIoExecutionStateError, CacheIoOperation, CacheIoOperationKey, CacheIoOperationKind,
    CacheIoPreparation, CacheIoStartDisposition, CacheLifecycleError, CachePolicyError,
    CachePoolError, CachePoolLimits, CachePoolReport, CachePoolResource, CacheRankIdentity,
    CacheRepresentation, CacheResidencyPool, CacheStorageError, CacheStoragePhase, CacheTier,
    LayerCachePolicy, MutableCacheTail, MutableStateResidency, PoolingStateComponent,
    PromptCacheBlock, PromptCacheDescriptor, PromptCacheError, PromptCacheManifest,
    PromptCacheModelIdentity, PromptCacheOptions, PromptCacheStateTensor, PromptCacheTopology,
    StateResidencyClass, StateTensorDimension, StateTensorDtype, StateTensorOwner,
    StateTensorPolicy, StateTensorPresence, StateTensorRole, PROMPT_CACHE_SCHEMA_VERSION,
};
pub use safemlx_lm_core::residency::{
    AllocatorMemoryMetrics, BackgroundPrefetchReport, CacheEvictionPolicy, EvictionMetrics,
    MemoryTier, OffloadConfig, OffloadError, OffloadPlan, OffloadReport, OffloadTelemetry,
    OffloadUnitId, OffloadUnitSpec, PrefetchAdmission, PrefetchCompletion,
    PrefetchDemandObservation, PrefetchDemandResolution, PrefetchExecutionState, PrefetchMetrics,
    PrefetchOutcome, PrefetchStateError, PrefetchWork, ProcessMetrics, ResidencyBlocker,
    ResidencyLedger, ResidencyLedgerError, ResidencyPolicy, TierByteTotals, TierUnitTotals,
    TransferDirection, TransferMetrics, UnitResidencyReport, OFFLOAD_PLAN_SCHEMA_VERSION,
};
pub use safemlx_lm_core::scheduler::{
    CancellationCause, RequestId, RequestStatus, Scheduler, SchedulerCapabilities, SchedulerError,
    SchedulerLimits, SchedulerProgress, SchedulerReport, SemanticStateTransaction,
    TransitionOutput, WorkDescriptor, WorkId, WorkLifecycle,
};
pub use safemlx_lm_core::scheduler::{
    SchedulerCapabilities as RealtimeSchedulerCapabilities,
    SchedulerReport as RealtimeSchedulerReport,
};
