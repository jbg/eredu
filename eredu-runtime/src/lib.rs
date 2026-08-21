//! Backend-neutral model execution contracts and algorithms.
//!
//! This crate orchestrates opaque backend-native values. It deliberately has
//! no dependency on an architecture implementation or execution backend.

#![warn(missing_docs)]

/// Backend execution, parameter, transfer, and collective capabilities.
pub mod backend;
/// Backend-neutral mutable-cache ownership, storage, and admission algorithms.
pub mod cache;
/// Typed multimodal component graphs and residency accounting.
pub mod component;
/// Backend-neutral sequential prediction decisions and layered handoff.
pub mod decision;
/// Backend-neutral dense-stream residency telemetry.
pub mod dense;
/// Backend-neutral speculative-state fork, commit, and rollback ownership.
pub mod draft;
/// Portable execution-group topology and scheduling state.
pub mod execution;
pub mod expert;
/// Backend-neutral causal-model and token-sampling contracts.
pub mod generation;
/// Backend-neutral ownership of prepared multimodal tensors.
pub mod input;
pub mod inspection;
/// Statically dispatched layered architecture lifecycle and resident policy.
pub mod layered;
/// Architecture-declared parallel parameter semantics and local layouts.
pub mod parallel;
/// Neutral checkpoint materialization and stable parameter binding.
pub mod parameter;
/// Backend-neutral rank-local architecture ownership.
pub mod partition;
/// Backend-neutral bounded background weight-prefetch execution.
pub mod prefetch;
/// Atomic realtime model, schedule, sampler, and random-state transactions.
pub mod realtime;
/// Backend-neutral immutable-weight residency declarations and orchestration.
pub mod residency;
/// Architecture-declared mutable state and concrete runtime realizations.
pub mod state;
mod weight_residency;

pub use backend::{CollectiveBackend, ParameterBackend, SubmissionBackend, TransferBackend};
pub use cache::{
    finalize_prompt_cache_shard, hash_prompt_cache_shard_payload, inspect_prompt_cache,
    resolve_prompt_cache_root, safe_prompt_cache_shard_path, validate_prompt_cache_manifest,
    CacheBlockLifecycle, CacheBlockStorage, CacheHostDemotionOperation, CacheHostPromotion,
    CacheIoAdmission, CacheIoCompletionDisposition, CacheIoExecutionState,
    CacheIoExecutionStateError, CacheIoOperation, CacheIoOperationKey, CacheIoOperationKind,
    CacheIoPreparation, CacheIoStartDisposition, CacheIoSubmission, CacheIoSubmissionOutcome,
    CacheIoTicket, CacheIoWorker, CacheIoWorkerError, CacheLayerResidencyReport,
    CacheLayerResidencyStats, CacheLifecycleError, CachePoolError, CachePoolLimits,
    CachePoolMembership, CachePoolReport, CachePoolReservation, CachePoolResource, CachePoolUsage,
    CacheResidencyConfigurationError, CacheResidencyPolicy, CacheResidencyPool,
    CacheResidencyReport, CacheResidencyTelemetry, CacheStorageError, CacheStoragePhase,
    LiveCacheBlockPublication, LiveCacheDiskPolicy, LiveCachePublicationError, MutableCacheTail,
    PagedCacheOptions, PromptCachePersistenceError, PromptCachePublication,
    CACHE_RESIDENCY_LAYER_REPORT_LIMIT, MAX_PROMPT_CACHE_SHARD_HEADER_BYTES,
    PROMPT_CACHE_CURRENT_FILE, PROMPT_CACHE_GENERATIONS_DIRECTORY,
};
pub use component::{
    ComponentDomain, ComponentGraph, ComponentGraphError, ComponentKind, ComponentResidencyClass,
    ComponentSpec,
};
pub use decision::{
    FullyForcedTailDecision, PredictionDirective, SequentialDecision, SequentialDecisionBoundary,
    SequentialDecisionDiagnostic, SequentialDecisionDriver, SequentialDecisionError,
    SequentialDecisionMode, SequentialDecisionPlan, SequentialDecisionPlanError,
    SequentialDecisionSource, SequentialDecisionTraversal, SequentialSamplingState,
};
pub use dense::{
    DenseCacheMetrics, DenseDiskStreamReport, DenseExecutionGroupReport, DensePassCounterSnapshot,
    DensePassReport, DenseStreamTelemetry, DenseStreamTelemetryError, DenseTierResidencyReport,
};
pub use draft::DraftStateTransaction;
pub use execution::{
    ExecutionGraph, ExecutionGraphError, ExecutionGroupId, ExecutionGroupReadySet,
    ExecutionGroupSchedule, ExecutionGroupSpec, ExecutionScheduleError, ExecutionUnitAddress,
    ExecutionUnitLayout, ExecutionUnitLayoutError, ReadyGroupState,
};
pub use expert::{
    combine_routed_expert_tensor_parallel, combine_tensor_parallel_expert_outputs,
    reduce_routed_expert_tensor_parallel, reduce_tensor_parallel_expert_output,
    ObservedExpertProvider, ObservedExpertProviderError, ResidentExpertProvider,
    RoutedExpertProvider, RoutedExpertRequest, RoutedExpertTensorParallelOutput,
    RoutedObservationPoint,
};
pub use generation::{
    CausalModel, ConstrainedSampler, DefaultSampler, GenerationSampler, MirostatV2Sampler,
    PenaltyConfig, Sampler, SamplingBackend, SamplingConfigurationError, SpeculativeSampler,
    TokenDomain,
};
pub use input::{PreparedInputPart, PreparedInputPayload, PreparedModelInput};
pub use inspection::{
    observe_and_intervene, ActivationObserver, NoopObserver, RoutingObservation,
    TargetStateCapture, TargetStateCaptureError, TargetStateTap,
};
pub use layered::{
    ArchitectureGroupKind, ArchitectureGroupPlacement, ArchitectureGroupTransport,
    ArchitectureMergeDestination, ArchitectureParallelSubgroup, CompositeLayeredTraversalHook,
    LayeredArchitecture, LayeredForwardState, LayeredTraversalHook, LayeredTraversalPoint,
    LayeredUnitAction, LayerwiseAcquireError, LayerwisePolicy, LayerwiseRuntime,
    LayerwiseRuntimeError, ParallelLayeredArchitecture, ParallelRoutedLayeredArchitecture,
    ResidentRuntime, ResidentUnitWindow, ResidentUnitWindowError, RoutedLayeredArchitecture,
};
pub use parallel::{
    aligned_partition_units, expand_linear_format_parameter_groups, module_parameter_group,
    partitioned_module_parameter_group, partitioned_projection_group, projection_parameter_group,
    segmented_projection_group, LocalModelLayout, LocalTensorLayout, MemberSharding,
    ParallelModelInfo, ParallelPlanError, ParameterGroupSpec, ParameterMemberSpec, ParameterRole,
    ProjectionSharding, ShardingPolicy, TensorPlacement,
};
pub use parameter::{
    bind_materialized_unit, bindings_from_recipe_set, materialize_bindings, MaterializedUnit,
    ParameterOrchestrationError, RecipeBindingError,
};
pub use partition::{
    validate_boundary_tensor_count, ArchitectureBoundary, ArchitectureBoundaryError,
    ArchitectureParameterDescription, ArchitectureParameterError, ArchitecturePartition,
    ArchitecturePartitionError, NoAuxiliaryBoundary, OwnedParameterGroupSpec, ParameterGroupOwner,
    PartitionGroup, PartitionOwnership, PartitionState,
};
pub use prefetch::{BackgroundPrefetchWorker, BackgroundPrefetchWorkerError};
pub use realtime::{
    RealtimeCompletionAttachmentError, RealtimeGenerationBranch, RealtimeGenerationState,
    RealtimeGenerationTransactionError,
};
pub use residency::{
    DeviceLayerWindow, OffloadUnit, ResidencyAcquisition, ResidencyController,
    ResidencyControllerError, ResidencyDeclarationError, ResidencyLease, ResidencyLeaseOwner,
    ResidencyLeaseStorage, ResidencyReport, ResidencyTransfer, ResidencyTransferOwner,
    ResidencyWindowError, ResidencyWindowManager, ResidentLayerGroup, ResidentLayerGroupReport,
    WeightBinding, WeightBindingPlan, WeightBindingSelectionError, WeightMaterializationReport,
};
pub use state::{
    DeviceState, LayerRuntimeState, ModelStateIdentity, PagedStatePlan,
    ResettableRuntimeLayerState, ResettableRuntimeState, RuntimeLayerState, RuntimeState,
    RuntimeStateComponents, StateError, StateLayout, StateResidencyPlan, StateSegmentId,
    StateSegmentLifetime, StateSegmentSpec, DEFAULT_STATE_SEGMENT_ID,
};
pub use weight_residency::{
    DenseDiskStreamLoadOptions, DenseTransferSchedule, DenseTransferScheduleError,
    ExecutionResidency, ExpertCacheLoadOptions, ExpertIdentity, ExpertPass, ExpertWeightResidency,
    LayerWeightResidency, LayerwiseLoadOptions, LayerwiseModelMetadata, NonExpertWeightResidency,
    StaticUnitBindings, WeightResidency, WeightResidencyPolicyError, DENSE_TRANSFER_WINDOW,
};

/// Inspectable architecture/runtime topology without backend-native values.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct RuntimeMetadata {
    model_identity: String,
    execution_graph: ExecutionGraph,
    execution_unit_count: usize,
}

impl RuntimeMetadata {
    /// Creates validated runtime metadata for one concrete architecture instance.
    pub fn new(
        model_identity: impl Into<String>,
        execution_graph: ExecutionGraph,
        execution_unit_count: usize,
    ) -> Result<Self, RuntimeMetadataError> {
        let model_identity = model_identity.into();
        if model_identity.trim().is_empty() {
            return Err(RuntimeMetadataError::EmptyModelIdentity);
        }
        if execution_unit_count == 0 {
            return Err(RuntimeMetadataError::EmptyExecution);
        }
        Ok(Self {
            model_identity,
            execution_graph,
            execution_unit_count,
        })
    }

    /// Returns the architecture-provided compatibility identity.
    pub fn model_identity(&self) -> &str {
        &self.model_identity
    }

    /// Returns the validated architecture execution graph.
    pub const fn execution_graph(&self) -> &ExecutionGraph {
        &self.execution_graph
    }

    /// Returns the total number of ordered execution units across all groups.
    pub const fn execution_unit_count(&self) -> usize {
        self.execution_unit_count
    }
}

/// Invalid backend-neutral runtime metadata.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
pub enum RuntimeMetadataError {
    /// The architecture supplied no compatibility identity.
    #[error("runtime model identity must not be empty")]
    EmptyModelIdentity,
    /// The architecture supplied no executable units.
    #[error("runtime must contain at least one execution unit")]
    EmptyExecution,
}

/// Error produced by backend-neutral runtime orchestration.
#[derive(Debug, thiserror::Error)]
pub enum RuntimeError {
    /// Invalid execution-group topology or scheduling transition.
    #[error(transparent)]
    ExecutionGraph(#[from] ExecutionGraphError),
    /// Invalid runtime metadata.
    #[error(transparent)]
    Metadata(#[from] RuntimeMetadataError),
    /// A concrete backend capability failed.
    #[error("runtime backend operation failed: {0}")]
    Backend(String),
}
