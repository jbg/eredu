//! Backend-neutral language-model loading and generation facade.
//!
//! [`api`] and [`core`] remain available without an execution backend. The
//! default `mlx` feature adds the MLX model implementations and runtime.

#![warn(missing_docs)]

/// High-level model loading, dispatch, and request APIs.
pub mod api;
/// Model-family implementations and architecture-specific adapters.
#[cfg(feature = "mlx")]
pub mod architectures;
/// Execution-backend implementations selected by crate features.
pub mod backend;
/// Reusable MLX neural-network building blocks.
#[cfg(feature = "mlx")]
pub mod nn;
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
    SlidingWindowLayerCount, StateLayout, StateMemoryAssumptions, StaticMemoryReport,
    SubgroupMembership, Submission, TextGeneration, TextGenerationBackend, TextGenerationConfig,
    TimingTelemetry, TokenFilter, TokenFilterController, TokenFilterError, TokenOutput,
    TokenizedMultimodalRequest, TokenizedMultimodalSegment, TopologyPreflightReport,
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

#[cfg(feature = "mlx")]
use safemlx::Array;

#[cfg(feature = "mlx")]
use crate::architectures::qwen::dense as resident_dense_qwen;

/// Builder passed to [`ModelInput`] implementations during generic generation.
#[cfg(feature = "mlx")]
pub struct ModelInputBuilder<'a, C, T> {
    /// Token ids or prompt ids for the current model step.
    pub y: &'a Array,
    /// Mutable per-layer cache used by the model implementation.
    pub cache: &'a mut Vec<Option<C>>,
    /// Caller-owned generation state carried across steps.
    pub state: &'a mut T,
}

/// Converts generic generation state into a model-specific input value.
#[cfg(feature = "mlx")]
pub trait ModelInput<'a, C, T> {
    /// Builds the concrete model input expected by a [`safemlx::module::Module`].
    fn from_model_input_builder(builder: ModelInputBuilder<'a, C, T>) -> Self;
}

#[cfg(feature = "mlx")]
impl<'a, C> ModelInput<'a, C, Option<Array>> for resident_dense_qwen::ModelInput<'a, C> {
    fn from_model_input_builder(builder: ModelInputBuilder<'a, C, Option<Array>>) -> Self {
        let ModelInputBuilder { y, cache, state } = builder;

        Self {
            inputs: y,
            mask: state.as_ref(),
            cache,
        }
    }
}

/// Output type that exposes logits for token sampling.
#[cfg(feature = "mlx")]
pub trait ModelOutput {
    /// Returns the logits tensor for the current generation step.
    fn logits(&self) -> &Array;
}

#[cfg(feature = "mlx")]
impl ModelOutput for Array {
    fn logits(&self) -> &Array {
        self
    }
}
