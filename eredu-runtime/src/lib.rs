//! Backend-neutral model execution contracts and algorithms.
//!
//! This crate orchestrates opaque backend-native values. It deliberately has
//! no dependency on an architecture implementation or execution backend.

#![warn(missing_docs)]

/// Backend execution, parameter, transfer, and collective capabilities.
pub mod backend;
/// Backend-neutral mutable-cache ownership, storage, and admission algorithms.
pub mod cache;
/// Opaque groups, routes, and capability contracts for distributed mechanisms.
pub mod communication;
/// Typed multimodal component graphs and residency accounting.
pub mod component;
/// Mechanism-only selection contracts for composite model input.
pub mod composite;
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
/// Rank-local graph execution over opaque communication resources.
pub mod partitioned_execution;
/// Backend-neutral bounded background weight-prefetch execution.
pub mod prefetch;
/// Atomic realtime model, schedule, sampler, and random-state transactions.
pub mod realtime;
/// Complete family-blind realtime frame coordination.
pub mod realtime_executor;
/// Portable realtime input validation before opaque token materialization.
pub mod realtime_ingress;
/// Neutral delayed-frame interpretation over opaque token mechanisms.
pub mod realtime_interpreter;
/// Family-blind construction of selected layered realtime models.
pub mod realtime_model;
/// Backend-neutral payload retention for delayed realtime coordinates.
pub mod realtime_payload;
/// Atomic model-state and delayed-payload-history transactions.
pub mod realtime_payload_state;
/// Family-blind realtime requirements, selection, and construction gating.
pub mod realtime_selection;
/// Backend-neutral realtime session ownership over the singular fair scheduler.
pub mod realtime_session;
/// Backend-neutral replicated-text execution and session ownership.
pub mod replicated_session;
/// Selection contracts for replicated text architectures.
pub mod replicated_text;
/// Backend-neutral immutable-weight residency declarations and orchestration.
pub mod residency;
/// Backend-neutral speculative request lifecycle and fair scheduling.
pub mod speculative;
/// Family-blind speculative requirements, selection, and construction gating.
pub mod speculative_selection;
/// Architecture-declared mutable state and concrete runtime realizations.
pub mod state;
mod weight_residency;

pub use backend::{
    BarrierBackend, BroadcastBackend, CollectiveBackend, CommunicationBackend, EvenGatherBackend,
    FailureAgreementBackend, ParameterBackend, PointToPointBackend, RoleExactBoundaryValue,
    SubmissionBackend, SumReductionBackend, TransferBackend, UnevenGatherBackend,
    VariableAllToAllBackend,
};
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
    ReversiblePromptCachePublication, CACHE_RESIDENCY_LAYER_REPORT_LIMIT,
    MAX_PROMPT_CACHE_SHARD_HEADER_BYTES, PROMPT_CACHE_CURRENT_FILE,
    PROMPT_CACHE_GENERATIONS_DIRECTORY,
};
pub use communication::validate_communication_manifest_consensus;
pub use communication::{
    project_all_communication_manifests, project_communication_manifest,
    validate_compatible_communication_manifests, BoundaryDimensionContract,
    BoundaryFramingProtocol, BoundaryRoleContract, CommunicationCapabilities,
    CommunicationCapabilityError, CommunicationCompletionCapabilities,
    CommunicationCompletionPolicy, CommunicationGroupDescriptor, CommunicationGroupId,
    CommunicationGroupRequirements, CommunicationManifest, CommunicationManifestConsensusError,
    CommunicationManifestError, CommunicationOperation, CommunicationOperationRequirement,
    CommunicationPeerCounts, CommunicationRouteDescriptor, CommunicationRouteId,
    CommunicationTensorLimits, RoleExactBoundaryContract, TopologyCommunicationPlan,
};
pub use component::{
    ComponentDomain, ComponentGraph, ComponentGraphError, ComponentKind, ComponentResidencyClass,
    ComponentSpec,
};
pub use composite::{
    select_composite_realization, select_processor_execution, CompositeSelectionError,
    MediaPrimitiveCapabilities, ModalityProcessorRequirements, ProcessorExecutionRequirements,
    ProcessorPrimitive, ProcessorSelectionError, ProcessorSelectionRequest,
    SelectedCompositeRealization, SelectedProcessorExecution,
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
pub use draft::{execute_draft_group, DraftGroupExecutionError, DraftStateTransaction};
pub use execution::{
    ExecutionGraph, ExecutionGraphError, ExecutionGroupId, ExecutionGroupSchedule,
    ExecutionGroupSpec, ExecutionScheduleError, ExecutionUnitAddress, ExecutionUnitLayout,
    ExecutionUnitLayoutError, ReadyGroupState,
};
pub use expert::{
    combine_routed_expert_tensor_parallel, combine_tensor_parallel_expert_outputs,
    reduce_routed_expert_tensor_parallel, reduce_tensor_parallel_expert_output,
    AddressableBankMember, AddressableBankMemberError, AddressableExpertRouteProvider,
    AddressableExpertRouteRequest, AddressableGatedProductBank, AddressableGroupedBank,
    ExpertRouteCombination, ExpertRouteExchange, ExpertRouteTensorMovement, IndexedMovement,
    ObservedExpertProvider, ObservedExpertProviderError, ParameterBankAcquisition,
    ResidentExpertProvider, RoutedExpertProvider, RoutedExpertRequest,
    RoutedExpertTensorParallelOutput, RoutedObservationPoint, TensorParallelRoutedExpertProvider,
};
pub use generation::{
    CausalModel, ConstrainedSampler, DefaultSampler, GenerationSampler, MirostatV2Sampler,
    PenaltyConfig, Sampler, SamplingBackend, SamplingConfigurationError, SpeculativeSampler,
    TokenDomain,
};
pub use input::{
    PreparedInputCacheIdentity, PreparedInputCacheIdentityError, PreparedInputInspector,
    PreparedInputPart, PreparedInputPayload, PreparedModelInput,
};
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
    LayerwisePolicy, LayerwisePolicyForward, LayerwiseRuntime, LayerwiseRuntimeError,
    ParallelLayeredArchitecture, ParallelRoutedLayeredArchitecture, PartitionedLayeredArchitecture,
    ResidentRuntime, ResidentUnitWindow, ResidentUnitWindowError, RoutedLayeredArchitecture,
    StaticParameterVisitor, StaticParameterVisitorMut,
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
    ArchitectureBoundaryValue, ArchitectureParameterDescription, ArchitectureParameterError,
    ArchitecturePartition, ArchitecturePartitionError, BoundaryTensorDimension,
    BoundaryTensorDtype, BoundaryTensorSpec, BoundaryWireSchema, LayeredPartitionBeginError,
    LayeredPartitionDriver, LayeredPartitionError, NoAuxiliaryBoundary, NoAuxiliaryBoundarySchema,
    OwnedParameterGroupSpec, ParameterGroupOwner, PartitionGroup, PartitionOwnership,
    PartitionState, PipelineActivationDtype, PipelineWireContract, ResolvedBoundaryTensorSpec,
    ResolvedBoundaryWireSchema,
};
pub use partitioned_execution::{
    CommunicationTensorMetadata, DistributedExecutionPhase, LayerwiseTraversalPartitionExecutor,
    LayerwiseTraversalRuntime, NoBoundaryTransport, NoCommitAgreement, NoOutputPublisher,
    OpaqueBoundaryTransport, OpaqueCommitAgreement, OpaqueFailureAgreement, OpaqueOutputPublisher,
    PartitionBoundaryRoute, PartitionBoundaryTransport, PartitionCommitAgreement,
    PartitionCommunication, PartitionCommunicationAuthority, PartitionExecutionError,
    PartitionOutputAuthority, PartitionOutputPublication, PartitionOutputPublisher,
    PartitionedExecutionPlan, PartitionedGroupExecutor, PartitionedTextExecution,
    PartitionedTextRuntime, PartitionedTraversalError, PartitionedTraversalResult,
    RealizedCommunicationGroup, RealizedCommunicationRoute,
};
pub use prefetch::{BackgroundPrefetchWorker, BackgroundPrefetchWorkerError};
pub use realtime::{
    RealtimeCompletionAttachmentError, RealtimeFrameExecutionError, RealtimeFrameTransition,
    RealtimeGenerationBranch, RealtimeGenerationState, RealtimeGenerationTransactionError,
};
pub use realtime_executor::{
    execute_realtime_frame, PreparedRealtimeFrameExecutor, PrepublicationRealtimeFrame,
    RealtimeCompletionCreationError, RealtimeDecisionExecution, RealtimeFrameCompletionMechanism,
    RealtimeFrameCoordinatorError, RealtimeFrameHostObserver, RealtimeHostOutputUnavailable,
    RealtimePrepublicationError, SubmittedRealtimeFrame,
};
pub use realtime_ingress::{
    MaterializedRealtimeInput, RealtimeHostTokenMaterializer, RealtimeIngressContract,
    RealtimeIngressError, RealtimePayloadKind, RealtimeTokenKind, ValidatedRealtimeInput,
};
pub use realtime_interpreter::{
    complete_realtime_frame, prepare_realtime_frame, CompletedRealtimeFrame, PreparedRealtimeFrame,
    RealtimeFrameInterpretationError, RealtimeFrameTensorMechanisms,
};
pub use realtime_model::{
    construct_realtime_model, ConstructedRealtimeExecution, ConstructedRealtimeModel,
    PreparedRealtimeModelContract, RealizedRealtimePolicy, RealizedRealtimeState,
    RealtimeArchitectureConstructionIdentity, RealtimeArchitectureIdentity,
    RealtimeLayerwiseRuntime, RealtimeMaterializationComponent, RealtimeMaterializationTask,
    RealtimeModelConstructionError, RealtimeModelConstructionMechanisms,
    RealtimeModelContractError,
};
pub use realtime_payload::{
    RealtimePayloadContract, RealtimePayloadContractError, RealtimePayloadEnvelope,
    RealtimePayloadGeneration, RealtimePayloadHistory, RealtimePayloadHistoryError,
    RealtimePayloadOwnerIdentity,
};
pub use realtime_payload_state::{
    RealtimePayloadBranch, RealtimePayloadState, RealtimePayloadStateTransactionError,
};
pub use realtime_selection::{
    select_and_prepare_realtime_realization, select_realtime_realization,
    ConstructedRealtimeResources, PreparedRealtimeRealization, RealtimeArchitectureProof,
    RealtimeArchitectureRequirements, RealtimeContractError, RealtimeExecutionRequirements,
    RealtimeIdentity, RealtimeMechanism, RealtimeMechanismCapabilities,
    RealtimeMechanismRequirements, RealtimeObservationRequirements, RealtimePreparationError,
    RealtimeSelectionError, RealtimeSelectionIssue, RealtimeSelectionRequest,
    RealtimeTopologyPolicy, RealtimeWeightComponentRequirement, RealtimeWeightComponentRole,
    RealtimeWeightLoweringRequirement, SelectedRealtimeRealization,
    SelectedRealtimeStateComponentRealization, SelectedRealtimeStateRealization,
};
pub use realtime_session::{
    RealtimeHistoryGeneration, RealtimeModelOwnerIdentity, RealtimeModelSessionIdentity,
    RealtimeSamplingReplacementError, RealtimeSamplingUpdateError, RealtimeSessionBranch,
    RealtimeSessionError, RealtimeSessionExecutionError, RealtimeSessionIncarnation,
    RealtimeSessionResumeError, RealtimeSessionScheduler, RealtimeSessionState,
    RealtimeSessionTransactionError, ReleasedRealtimeSession,
};
pub use replicated_session::{
    construct_replicated_text_session, construct_replicated_text_session_with_execution,
    construct_replicated_text_session_with_runtime, prepare_layered_text_contract,
    prepare_layered_text_contract_with_addressable_parameters, prepare_partitioned_session_runtime,
    prepare_partitioned_session_runtime_with_exclusions, prepare_replicated_text_contract,
    prepare_replicated_text_contract_with_addressable_parameters, DirectReplicatedTextExecution,
    DistributedSessionCheckpoint, DistributedStateCheckpoint, PartitionedSessionFactoryInput,
    PartitionedSessionPreparationError, PredictionTargetOperation,
    PreparedPartitionedSessionRuntime, PreparedReplicatedTextContract,
    ReplicatedRuntimeExecutionStrategy, ReplicatedTextExecutionStrategy, ReplicatedTextRuntime,
    ReplicatedTextSession, ReplicatedTextSessionCheckpoint, ReplicatedTextSessionError,
    ReplicatedTextSessionMechanisms, ReplicatedTextSessionReport, RoutedReplicatedTextExecution,
    SessionStateRealization, TransactionalPromptCacheMechanisms,
};
pub use replicated_text::{
    partitioned_replicated_text_materialization_tasks, replicated_text_materialization_tasks,
    select_replicated_text_realization, AddressableStorageCapabilities, AddressableStorageTiers,
    BackendMechanismCapabilities, GroupedOperationRequirement, ParameterTransformConstraint,
    ParameterTransformTarget, ReplicatedTextArchitecture, ReplicatedTextContractError,
    ReplicatedTextMaterializationTask, ReplicatedTextOutputCompanion,
    ReplicatedTextOutputSelection, ReplicatedTextParameterOwner, ReplicatedTextParameterPresence,
    ReplicatedTextParameterRequirement, ReplicatedTextParameterRole, ReplicatedTextPhysicalSource,
    ReplicatedTextRequirements, ReplicatedTextSelectionError, ReplicatedTextSelectionRequest,
    ReplicatedTextStateAccess, SelectedParameterRealization, SelectedReplicatedTextRealization,
    SelectedStateComponentRealization, SelectedStateRealization, StateComponentMechanism,
    StateComponentPlacement, StateMechanismCapabilities, WeightLoweringCapability,
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
pub use speculative_selection::{
    select_and_prepare_speculative_realization,
    select_and_prepare_speculative_realization_observed, select_speculative_realization,
    ConstructedSpeculativeResources, PreparedSpeculativeRealization, SelectedSpeculativeCompletion,
    SelectedSpeculativePlacement, SelectedSpeculativeRealization, SelectedSpeculativeSampling,
    SelectedSpeculativeState, SpeculativeArchitectureCompatibilityProof, SpeculativeCaptureEntry,
    SpeculativeCaptureEnvelope, SpeculativeCaptureError, SpeculativeCaptureMetadata,
    SpeculativeCaptureSchema, SpeculativeContractError, SpeculativeIdentity,
    SpeculativeLaneIdentity, SpeculativeMechanism, SpeculativeMechanismCapabilities,
    SpeculativeMechanismRequirements, SpeculativePlacementRequest, SpeculativePreparationError,
    SpeculativeRealizationRequirements, SpeculativeSelectionError, SpeculativeSelectionRequest,
    SpeculativeStateCacheIdentityIngredients, SpeculativeStrategyClass,
    SpeculativeStrategyRequirements,
};
pub use state::{
    realize_architecture_state, ArchitectureStateFactory, ArchitectureStatePartitionError,
    ArchitectureStatePartitionPlan, ArchitectureStatePartitionRule, ArchitectureStatePlacement,
    ArchitectureStateRealizationError, DeviceState, LayerRuntimeState, ModelStateIdentity,
    ResettableRuntimeLayerState, ResettableRuntimeState, RuntimeLayerState, RuntimeState,
    RuntimeStateComponents, StateError, StateLayout, StateSegmentId, StateSegmentLifetime,
    StateSegmentSpec, DEFAULT_STATE_SEGMENT_ID,
};
pub use weight_residency::{
    DenseDiskStreamLoadOptions, DenseTransferSchedule, DenseTransferScheduleError,
    ExecutionResidency, ExpertPass, LayerWeightResidency, LayerwiseLoadOptions,
    LayerwiseModelMetadata, OrdinaryWeightResidency, ParameterBankAccess, ParameterBankKey,
    ParameterBankLoadOptions, ParameterBankResidency, StaticUnitBindings, WeightResidency,
    WeightResidencyPolicyError, DENSE_TRANSFER_WINDOW,
};
