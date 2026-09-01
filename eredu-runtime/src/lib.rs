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
/// Selection contracts for replicated text architectures.
pub mod replicated_text;
/// Backend-neutral immutable-weight residency declarations and orchestration.
pub mod residency;
/// Backend-neutral speculative request lifecycle and fair scheduling.
pub mod speculative;
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
    ExecutionGraph, ExecutionGraphError, ExecutionGroupId, ExecutionGroupSchedule,
    ExecutionGroupSpec, ExecutionScheduleError, ExecutionUnitAddress, ExecutionUnitLayout,
    ExecutionUnitLayoutError, ReadyGroupState,
};
pub use expert::{
    combine_routed_expert_tensor_parallel, combine_tensor_parallel_expert_outputs,
    reduce_routed_expert_tensor_parallel, reduce_tensor_parallel_expert_output,
    ObservedExpertProvider, ObservedExpertProviderError, ResidentExpertProvider,
    RoutedExpertProvider, RoutedExpertRequest, RoutedExpertTensorParallelOutput,
    RoutedObservationPoint, TensorParallelRoutedExpertProvider,
};
pub use generation::{
    CausalModel, ConstrainedSampler, DefaultSampler, GenerationSampler, MirostatV2Sampler,
    PenaltyConfig, Sampler, SamplingBackend, SamplingConfigurationError, SpeculativeSampler,
    TokenDomain,
};
pub use input::{PreparedInputPart, PreparedInputPayload, PreparedModelInput};
pub use inspection::{
    observe_and_intervene, observe_model_logits, ActivationObserver, NoopObserver,
    RoutingObservation, TargetStateCapture, TargetStateCaptureError, TargetStateTap,
};
pub use layered::{
    ArchitectureGroupKind, ArchitectureGroupPlacement, ArchitectureGroupTransport,
    ArchitectureMergeDestination, ArchitectureParallelSubgroup, ArchitectureParameters,
    CompositeLayeredTraversalHook, LayeredArchitecture, LayeredForwardState, LayeredPartitionInput,
    LayeredPartitionOutput, LayeredPipelineSchedule, LayeredPipelineScheduleError,
    LayeredTraversalHook, LayeredTraversalPoint, LayeredUnitAction, LayerwiseAcquireError,
    LayerwisePolicy, LayerwiseRuntime, LayerwiseRuntimeError, ParallelLayeredArchitecture,
    ParallelRoutedLayeredArchitecture, PartitionedLayeredArchitecture, ResidentRuntime,
    ResidentUnitWindow, ResidentUnitWindowError, RoutedLayeredArchitecture, StaticParameterVisitor,
    StaticParameterVisitorMut,
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
    ArchitecturePartitionError, BoundaryTensorDimension, BoundaryTensorDtype, BoundaryTensorSpec,
    BoundaryWireSchema, LayeredPartitionDriver, LayeredPartitionError, NoAuxiliaryBoundary,
    NoAuxiliaryBoundarySchema, OwnedParameterGroupSpec, ParameterGroupOwner, PartitionGroup,
    PartitionOwnership, PartitionState, PipelineActivationDtype, PipelineWireContract,
    ResolvedBoundaryTensorSpec, ResolvedBoundaryWireSchema,
};
pub use prefetch::{BackgroundPrefetchWorker, BackgroundPrefetchWorkerError};
pub use realtime::{
    RealtimeCompletionAttachmentError, RealtimeGenerationBranch, RealtimeGenerationState,
    RealtimeGenerationTransactionError,
};
pub use replicated_text::{
    select_replicated_text_realization, BackendMechanismCapabilities, GroupedOperationRequirement,
    ParameterTransformConstraint, ParameterTransformTarget, ReplicatedTextArchitecture,
    ReplicatedTextContractError, ReplicatedTextParameterOwner, ReplicatedTextParameterPresence,
    ReplicatedTextParameterRequirement, ReplicatedTextParameterRole, ReplicatedTextRequirements,
    ReplicatedTextSelectionError, ReplicatedTextSelectionRequest, SelectedParameterRealization,
    SelectedReplicatedTextRealization, StateResidencyMechanism, WeightLoweringCapability,
    WeightLoweringDescriptor, WeightLoweringKind, WeightResidencyMechanism,
};
pub use residency::{
    DeviceLayerWindow, OffloadUnit, ResidencyAcquisition, ResidencyController,
    ResidencyControllerError, ResidencyDeclarationError, ResidencyLease, ResidencyLeaseOwner,
    ResidencyLeaseStorage, ResidencyReport, ResidencyTransfer, ResidencyTransferOwner,
    ResidencyWindowError, ResidencyWindowManager, ResidentLayerGroup, ResidentLayerGroupReport,
    WeightBinding, WeightBindingPlan, WeightBindingSelectionError, WeightMaterializationReport,
};
pub use speculative::{RunSpeculativeGeneration, SpeculativeScheduler};
pub use state::{
    ArchitectureStatePartitionError, ArchitectureStatePartitionPlan,
    ArchitectureStatePartitionRule, ArchitectureStatePlacement, DeviceState, LayerRuntimeState,
    ModelStateIdentity, ResettableRuntimeLayerState, ResettableRuntimeState, RuntimeLayerState,
    RuntimeState, RuntimeStateComponents, StateError, StateLayout, StateSegmentId,
    StateSegmentLifetime, StateSegmentSpec, DEFAULT_STATE_SEGMENT_ID,
};
pub use weight_residency::{
    DenseDiskStreamLoadOptions, DenseTransferSchedule, DenseTransferScheduleError,
    ExecutionResidency, ExpertCacheLoadOptions, ExpertIdentity, ExpertPass, ExpertWeightResidency,
    LayerWeightResidency, LayerwiseLoadOptions, LayerwiseModelMetadata, NonExpertWeightResidency,
    StaticUnitBindings, WeightResidency, WeightResidencyPolicyError, DENSE_TRANSFER_WINDOW,
};
