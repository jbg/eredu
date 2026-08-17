//! Language-model loading and generation with the MLX backend enabled by default.
//!
//! [`api`] provides high-level loading and generation. Model-family
//! implementations live under [`architectures`], reusable neural components
//! under [`nn`], and execution infrastructure under [`runtime`]. Portable
//! contracts and orchestration are defined by [`core`].

#![warn(missing_docs)]

/// High-level model loading, dispatch, and request APIs.
pub mod api;
/// Model-family implementations and architecture-specific adapters.
pub mod architectures;
/// MLX implementation of the backend-neutral execution contract.
pub mod backend;
/// Error types returned by the language-model runtime.
pub mod error;
/// Reusable MLX neural-network building blocks.
pub mod nn;
/// Facade execution infrastructure, including MLX-specific implementations.
pub mod runtime;
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
    Backend, BackendSession, CollectiveScope, DistributedBackend, DistributedCapabilities,
    DistributedSession, DistributedSessionDescriptor, ModelRuntime, ValueDescriptor,
};
#[cfg(test)]
mod test_utils;
#[cfg(test)]
extern crate self as safemlx_lm;
#[cfg(test)]
#[path = "../tests/distributed_cartesian_ring.rs"]
mod distributed_cartesian_ring;
#[cfg(test)]
#[path = "../tests/distributed_expert_exchange_ring.rs"]
mod distributed_expert_exchange_ring;
#[cfg(test)]
#[path = "../tests/distributed_expert_parallel_ring.rs"]
mod distributed_expert_parallel_ring;
#[cfg(test)]
#[path = "../tests/distributed_gemma4_multimodal_pipeline_ring.rs"]
mod distributed_gemma4_multimodal_pipeline_ring;
#[cfg(test)]
#[path = "../tests/distributed_partition_ring.rs"]
mod distributed_partition_ring;
#[cfg(test)]
#[path = "../tests/distributed_pipeline_ring.rs"]
mod distributed_pipeline_ring;
#[cfg(test)]
#[path = "../tests/distributed_qwen3_vl_pipeline_ring.rs"]
mod distributed_qwen3_vl_pipeline_ring;
#[cfg(test)]
#[path = "../tests/distributed_tensor_parallel_ring.rs"]
mod distributed_tensor_parallel_ring;

pub use api::realtime::{
    load_model as load_realtime_model, load_model_with_options as load_realtime_model_with_options,
    LoadedRealtimeModel, RealtimeCompletedStep, RealtimeInferenceScheduler, RealtimeModelKind,
    RealtimeSampling, RealtimeSchedulerCapabilities, RealtimeSchedulerReport, RealtimeSession,
    RealtimeSpeechConfig, RealtimeStepInput, RealtimeStepOutput,
};
pub use api::{
    discover_hardware, execution_plan_load_options, inspect_model, load_model,
    load_model_with_options, plan_automatic_execution, AllocatorTelemetry, ArtifactKind,
    ArtifactModality, ArtifactTensorEncoding, AutomaticPlanRequest, AutomaticPlanner,
    AutomaticPlannerPolicy, BackendKind, DevicePlan, DraftPlacementPlan, DraftingPlan,
    DurationSeconds, ExecutionPlan, ExecutionPlanReport, ExecutionTelemetry, ExpertCachePlan,
    ExpertCacheTelemetry, HardwareBackendProfile, HardwareDeviceProfile, HardwareMemorySemantics,
    HardwareProfile, InspectionIssue, InspectionIssueCode, InspectionReadiness,
    InspectionRequirement, InspectionSeverity, ModelInspectionOptions, ModelInspectionReport,
    ModelLoadOptions, ModelResourceProfile, ObservationKind, Observed, ParallelismPlan,
    PlanExplanation, PlanExplanationEntry, PlanExplanationLevel, ResidencyPlan, ResidencyTelemetry,
    TimingTelemetry, TransferTelemetry, WeightTransformationPlan, AUTOMATIC_SCHEMA_VERSION,
};
pub use architectures::llama::layerwise::{LlamaCache, LlamaModel};
pub use backend::mlx::{
    MlxBackend, MlxDistributedSession, MlxModel, MlxModelInput, MlxModelOutput, MlxModelSession,
    MlxSessionCompletion,
};
pub use runtime::attention::{AttentionPolicy, LayerSchedule, LayerScheduleError};
pub use runtime::cache::residency::{
    inspect_prompt_cache, CacheLayerResidencyReport, CacheLayerResidencyStats, CacheResidencyError,
    CacheResidencyManager, CacheResidencyPolicy, CacheResidencyReport, LiveCacheDiskPolicy,
    PagedCacheOptions, CACHE_RESIDENCY_LAYER_REPORT_LIMIT,
};
pub use runtime::cache::PagedKeyValueCache;
pub use runtime::distributed::completion::DistributedCompletion;
pub use runtime::distributed::parallel::{
    LocalModelLayout, LocalTensorLayout, MemberSharding, ParallelBuildContext,
    ParallelExecutionContext, ParallelPlanBuilder, ParameterGroupSpec, ParameterMemberSpec,
    ParameterRole, ShardingPolicy, SynchronizedToken,
};
pub use runtime::distributed::topology::{
    DeviceAssignment, ParallelAxis, ParallelCoordinates, ParallelTopology, PlacementPlan,
    RankPartition, SubgroupMembership, TensorPlacement, TopologyPreflightReport,
};
pub use runtime::execution::layerwise::{
    load_layerwise_model, ArchitectureAdapter, DenseCacheMetrics, DenseDiskStreamReport,
    DenseExecutionGroupReport, DensePassReport, DenseTierResidencyReport, ExecutionResidency,
    ExpertWeightResidency, LayerWeightResidency, LayerwiseForwardState, LayerwiseLoadOptions,
    LayerwiseModel, LayerwiseModelMetadata, NonExpertWeightResidency, ParallelModelInfo,
    SharedWeightStore, WeightResidency,
};
pub use runtime::residency::dense_stream::{DenseDiskStreamLoadOptions, DenseStreamError};
pub use runtime::residency::expert_cache::{ExpertCacheLoadOptions, ExpertCacheReport};
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

use safemlx::Array;

use crate::architectures::qwen::dense as resident_dense_qwen;

/// Builder passed to [`ModelInput`] implementations during generic generation.
pub struct ModelInputBuilder<'a, C, T> {
    /// Token ids or prompt ids for the current model step.
    pub y: &'a Array,
    /// Mutable per-layer cache used by the model implementation.
    pub cache: &'a mut Vec<Option<C>>,
    /// Caller-owned generation state carried across steps.
    pub state: &'a mut T,
}

/// Converts generic generation state into a model-specific input value.
pub trait ModelInput<'a, C, T> {
    /// Builds the concrete model input expected by a [`safemlx::module::Module`].
    fn from_model_input_builder(builder: ModelInputBuilder<'a, C, T>) -> Self;
}

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
pub trait ModelOutput {
    /// Returns the logits tensor for the current generation step.
    fn logits(&self) -> &Array;
}

impl ModelOutput for Array {
    fn logits(&self) -> &Array {
        self
    }
}
