//! Backend-neutral model execution contracts and algorithms.
//!
//! This crate orchestrates opaque backend-native values. It deliberately has
//! no dependency on an architecture implementation or execution backend.

#![warn(missing_docs)]

/// Portable automatic-plan resource sizing and telemetry projections.
pub mod automatic_support;
/// Backend execution, parameter, transfer, and collective capabilities.
pub mod backend;
/// Backend-neutral mutable-cache ownership, storage, and admission algorithms.
pub mod cache;
/// Bounded observation admission and collection.
pub mod capture;
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
/// Completed-token lifecycle, snapshot reservation and branch accounting policy.
pub mod execution_control;
pub mod execution_plan;
pub mod expert;
/// Backend-neutral causal-model and token-sampling contracts.
pub mod generation;
/// Closed immutable host-metadata ownership and accounting attachments.
pub mod host_metadata;
/// Backend-neutral ownership of prepared multimodal tensors.
pub mod input;
pub mod inspection;
/// Immutable intervention scheduling and evidence using the shared capture ledger.
pub mod intervention;
/// Statically dispatched layered architecture lifecycle and resident policy.
pub mod layered;
/// Normalized portable policy for cold model preparation.
pub mod load_request;
/// Exact mechanism capability synthesis from neutral requirements.
pub mod mechanism_synthesis;
/// Retained media ingress through the same selected prefill transaction.
pub mod media_prefill;
/// Architecture-declared parallel parameter semantics and local layouts.
pub mod parallel;
/// Neutral checkpoint materialization and stable parameter binding.
pub mod parameter;
pub mod parameter_operations;
/// Backend-neutral rank-local architecture ownership.
pub mod partition;
/// Rank-local graph execution over opaque communication resources.
pub mod partitioned_execution;
pub mod placement;
/// Backend-neutral bounded background weight-prefetch execution.
pub mod prefetch;
/// Completion-gated scheduling for bounded prompt execution.
pub mod prefill;
/// Atomic realtime model, schedule, sampler, and random-state transactions.
pub mod realtime;
/// Complete family-blind realtime frame coordination.
pub mod realtime_executor;
/// Portable realtime input validation before opaque token materialization.
pub mod realtime_ingress;
/// Neutral delayed-frame interpretation over opaque token mechanisms.
pub mod realtime_interpreter;
pub mod realtime_mechanism_synthesis;
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
/// Bounded all-rank readiness and lifetime accounting before text execution.
pub mod run_preparation;
/// Backend-neutral speculative request lifecycle and fair scheduling.
pub mod speculative;
/// Family-blind speculative requirements, selection, and construction gating.
pub mod speculative_selection;
/// Architecture-declared mutable state and concrete runtime realizations.
pub mod state;
mod weight_residency;
/// Shared request-bound inference working-memory accounting.
pub mod working_memory;

pub use automatic_support::{
    BoundedResidencySizingError, SelectedParameterResources, placed_recipe_peak_bytes,
    placed_source_recipe, residency_telemetry, selected_parameter_resources,
    selected_parameter_resources_for_layout, selected_text_bounded_requirement,
};
pub use backend::{
    BarrierBackend, BroadcastBackend, CollectiveBackend, CommunicationBackend, EvenGatherBackend,
    FailureAgreementBackend, ParameterBackend, PointToPointBackend, RoleExactBoundaryValue,
    SubmissionBackend, SumReductionBackend, TerminalCommunicationBackend, TransferBackend,
    UnevenGatherBackend, VariableAllToAllBackend,
};
pub use cache::{
    CACHE_RESIDENCY_LAYER_REPORT_LIMIT, CacheBlockLifecycle, CacheBlockSelection,
    CacheBlockStorage, CacheHostDemotionOperation, CacheHostPromotion, CacheIoAdmission,
    CacheIoCompletionDisposition, CacheIoExecutionState, CacheIoExecutionStateError,
    CacheIoOperation, CacheIoOperationKey, CacheIoOperationKind, CacheIoPreparation,
    CacheIoStartDisposition, CacheIoSubmission, CacheIoSubmissionOutcome, CacheIoTicket,
    CacheIoWorker, CacheIoWorkerError, CacheLayerResidencyReport, CacheLayerResidencyStats,
    CacheLifecycleError, CachePoolError, CachePoolLimits, CachePoolMembership, CachePoolReport,
    CachePoolReservation, CachePoolResource, CachePoolUsage, CacheResidencyConfigurationError,
    CacheResidencyPolicy, CacheResidencyPool, CacheResidencyReport, CacheResidencyTelemetry,
    CacheStorageError, CacheStoragePhase, LiveCacheBlockPublication, LiveCacheDiskPolicy,
    LiveCachePublicationError, MAX_PROMPT_CACHE_SHARD_HEADER_BYTES, MutableCacheTail,
    PROMPT_CACHE_CURRENT_FILE, PROMPT_CACHE_GENERATIONS_DIRECTORY, PagedCacheOptions,
    PromptCachePersistenceError, PromptCachePublication, ReversiblePromptCachePublication,
    finalize_prompt_cache_shard, hash_prompt_cache_shard_payload, inspect_prompt_cache,
    prompt_cache_rank_path, resolve_prompt_cache_root, safe_prompt_cache_shard_path,
    validate_prompt_cache_manifest,
};
pub use communication::validate_communication_manifest_consensus;
pub use communication::{CommunicationGroupOperation, CommunicationGroupOperationError,
    CommunicationTensorContractError};
pub use communication::{
    AgreedCommunicationSession, BoundaryDimensionContract, BoundaryFramingProtocol,
    BoundaryRoleContract, CommunicationCapabilities, CommunicationCapabilityError,
    CommunicationCompletionCapabilities, CommunicationCompletionPolicy,
    CommunicationGroupDescriptor, CommunicationGroupRequirements, CommunicationManifest,
    CommunicationManifestConsensusError, CommunicationManifestError, CommunicationOperation,
    CommunicationOperationRequirement, CommunicationPeerCounts, CommunicationPeerMatrix, PreparedPeerCountLoan, CommunicationRealizationError,
    CommunicationRouteDescriptor, CommunicationRouteId, CommunicationSessionIdentity,
    CommunicationTensorLimits, CommunicationTopologyCapabilities, PreparedCommunicationRealization,
    RetainedCommunicationSource, RetainedCommunicationSourceError,
    PreparedBoundarySource, PreparedBoundaryFrames, PreparedBoundaryFrameError, PreparedBoundaryFrameCause,
    RoleExactBoundaryContract, TopologyCommunicationPlan, establish_communication_session,
    prepare_communication_realization, project_all_communication_manifests,
    project_communication_manifest, validate_compatible_communication_manifests,
};
pub use component::{
    ComponentDomain, ComponentGraph, ComponentGraphError, ComponentKind, ComponentResidencyClass,
    ComponentSpec,
};
pub use composite::{
    CompositeSelectionError, MediaPrimitiveCapabilities, ModalityProcessorRequirements,
    ProcessorExecutionRequirements, ProcessorPrimitive, ProcessorSelectionError,
    ProcessorSelectionRequest, SelectedCompositeRealization, SelectedProcessorExecution,
    select_composite_realization, select_processor_execution,
};
pub use decision::{
    FullyForcedTailDecision, PredictionDirective, SequentialDecision, SequentialDecisionBoundary,
    SequentialDecisionDiagnostic, SequentialDecisionDriver, SequentialDecisionError,
    SequentialDecisionMode, SequentialDecisionObservation, SequentialDecisionPlan,
    SequentialDecisionPlanError, SequentialDecisionSource, SequentialDecisionTraversal,
    SequentialSamplingState, merge_output_demand,
};
pub use dense::{
    DenseCacheMetrics, DenseDiskStreamReport, DenseExecutionGroupReport, DensePassCounterSnapshot,
    DensePassReport, DenseStreamTelemetry, DenseStreamTelemetryError, DenseStreamTelemetryPlan,
    DenseTelemetryPreparationError, DenseTierResidencyReport,
};
pub use draft::{DraftGroupExecutionError, DraftStateTransaction, execute_draft_group};
pub use execution::{
    ArchitectureExecutionGraph, ExecutionGraph, ExecutionGraphError, ExecutionGroupId,
    ExecutionGroupSchedule, ExecutionGroupSpec, ExecutionScheduleError, ExecutionUnitAddress,
    ExecutionUnitLayout, ExecutionUnitLayoutError, GroupSubmissionMechanism, ReadyGroupState,
};
pub use execution_plan::{
    ExecutionPlanLoadError, ResidencyDiagnostics, execution_plan_quantization,
};
pub use expert::{
    AddressableBankBindingPlan, AddressableBankDistribution, AddressableBankMember,
    AddressableBankMemberError, AddressableBankMemberPlacement, AddressableBankParameter,
    AddressableBankTask, AddressableBindingTransform, AddressableExpertRouteProvider,
    AddressableExpertRouteRequest, AddressableGatedProductBank, AddressableGroupedBank,
    ExpertRouteCombination, ExpertRouteExchange, ExpertRouteInvocation, ExpertRouteTensorMovement, PreparedExpertMovementLoan, ExpertRouteMovementSourceError,
    IndexedMovement, ObservedExpertProvider, ObservedExpertProviderError, ParameterBankAcquisition,
    ProviderUnitObserver, ResidentExpertProvider, RoutedBankId, RoutedBankProviderError,
    BorrowedRoutedBankProviders, RoutedBankProviders, RoutedExpertProvider, RoutedExpertRequest,
    RoutedExpertTensorParallelOutput, RoutedObservationPoints, RoutedUnitBatch,
    RoutedUnitInvocation, RoutedUnitObserver, TensorParallelRoutedExpertProvider,
    combine_routed_expert_tensor_parallel, combine_tensor_parallel_expert_outputs,
    plan_addressable_bank_bindings, reduce_routed_expert_tensor_parallel,
    reduce_tensor_parallel_expert_output, selected_addressable_parameter_bytes,
    with_borrowed_resident_unit_coordinates, with_provider_unit_observer,
    with_resident_unit_coordinates, with_routed_unit_invocation, with_routed_unit_observer,
};
pub use expert::{
    RoutedUnitOrigin, RoutedUnitOrigins, select_routes_with_observer, select_routes_with_provider,
    with_exchanged_unit_observer, with_partition_unit_observer,
};
pub use generation::{
    CausalModel, ConfiguredTextSampler, ConstrainedSampler, DefaultSampler, GenerationSampler,
    MirostatV2Sampler, PenaltyConfig, Sampler, SamplingBackend, SamplingConfigurationError,
    SpeculativeSampler, TokenDomain,
};
pub use host_metadata::{
    DenseHostSlotFinishError, DenseHostSlotInitialization, DenseHostSlotInitializationBuilder,
    HostMetadataIdentity, HostMetadataKey, HostSlotAttachmentError, HostSlotFinishError,
    HostSlotInitialization, HostSlotInitializationBuilder, HostSlotInitializationError,
    HostSlotMetadata, HostSlotPushError, HostSlotTable, InitializedDenseHostSlots,
    InitializedHostSlots, PreparedDenseHostCopyError, SharedHostMetadata,
};
pub use input::{
    PreparedInputCacheIdentity, PreparedInputCacheIdentityError, PreparedInputInspector,
    PreparedInputPart, PreparedInputPayload, PreparedModelInput, SharedPreparedInputCacheIdentity,
};
pub use inspection::{
    ActivationObserver, BorrowedActivationObserver, NoopObserver, RoutingDecision,
    RoutingObservation, RoutingUnmodifiedInterest, TargetStateCapture, TargetStateCaptureError,
    TargetStateTap, observe_and_intervene, observe_model_logits,
};
pub use layered::{
    ordinary_addressed_units, ArchitectureGroupKind, ArchitectureGroupPlacement, ArchitectureGroupTransport,
    ArchitectureGroupTransportDeclaration, ArchitectureMergeDestination,
    ArchitectureParallelSubgroup, ArchitectureParameters, CompositeLayeredTraversalHook,
    LayeredArchitecture, LayeredForwardState, LayeredPartitionInput, LayeredPartitionOutput,
    LayeredPipelineSchedule, LayeredPipelineScheduleError, LayeredTraversalHook,
    LayeredTraversalPoint, LayeredUnitAction, LayerwiseAcquireError, LayerwisePolicy,
    LayerwisePolicyForward, LayerwiseRuntime, LayerwiseRuntimeError, OrderedLayerwiseCompletion,
    ParallelLayeredArchitecture, ParallelRoutedLayeredArchitecture, PartitionedLayeredArchitecture,
    LayeredObservationBinding, PreparedLayeredObservationError, PreparedLayeredObservationPaths,
    PreparedObservationBindingIdentity, ResidentRuntime, ResidentUnitWindow,
    ResidentUnitWindowError, RoutedLayeredArchitecture, SharedLayeredObservationPaths,
    StaticParameterVisitor, StaticParameterVisitorMut,
};
pub use load_request::{
    DraftingLoadRequest, NormalizedLoadRequest, NormalizedLoadRequestError, ParallelLoadRequest,
    ValidatedModelLoadRequest,
};
pub use mechanism_synthesis::{
    BackendMechanismFacts, ReplicatedTextMechanismSupport, StateLifecycleCapabilities,
    synthesize_replicated_text_capabilities,
};
pub use parallel::{
    LocalModelLayout, LocalTensorLayout, MemberSharding, ParallelModelInfo, ParallelPlanError, PartitionChunkRangeError, ParallelLayoutStorageError,
    ParameterGroupSpec, ParameterMemberSpec, ParameterRole, ProjectionSharding, ShardingPolicy,
    TensorPlacement, aligned_partition_units, aligned_partition_units_with_metadata,
    aligned_partition_units_with_tail, derive_transform_source_layout,
    expand_linear_format_parameter_groups, expand_linear_format_parameter_groups_with_metadata, module_parameter_group,
    module_parameter_group_with_metadata, partition_chunk_range, partition_parameter_group_chunks,
    partition_parameter_group_chunks_with_metadata, partition_parameter_group_chunks_with_source,
    partitioned_module_parameter_group, partitioned_module_parameter_group_with_metadata,
    partitioned_projection_group, partitioned_projection_group_with_metadata,
    projection_parameter_group, projection_parameter_group_with_metadata,
    segmented_projection_group, segmented_projection_group_with_metadata,
};
pub use parameter::{
    BindingPlan, BindingPlanError, MaterializedUnit, ModuleBindingPlan, ModuleBindingPlanError,
    ParameterBatchBudget, ParameterBindingTarget, ParameterOrchestrationError, PlannedBinding,
    PreparedParameterBinding, PreparedParameterBindingError, RecipeBindingError,
    SelectedBindingPlan, bind_materialized_unit, bind_materialized_unit_excluding,
    bind_prepared_parameter_values, bindings_from_recipe_set, build_exact_replicated_text_bindings,
    build_exact_replicated_text_bindings_for_targets, build_module_binding_plan,
    materialize_bindings, materialize_selected_bindings, preflight_bindings,
    prepared_parameter_binding_control_bytes, resolve_replicated_text_transform_source,
    select_bindings, TransformSourceError,
};
pub use partition::{
    ArchitectureBoundary, ArchitectureBoundaryError, ArchitectureBoundaryValue,
    ArchitectureParameterDescription, ArchitectureParameterError, ArchitecturePartition,
    ArchitecturePartitionError, BoundaryTensorDimension, BoundaryTensorDtype, BoundaryTensorSpec,
    BoundaryWireSchema, LayeredPartitionBeginError, LayeredPartitionDriver, LayeredPartitionError,
    NoAuxiliaryBoundary, NoAuxiliaryBoundarySchema, OwnedParameterGroupSpec, ParameterGroupOwner,
    PartitionGroup, PartitionOwnership, PartitionState, PipelineActivationDtype,
    PipelineWireContract, ResolvedBoundaryTensorSpec, ResolvedBoundaryWireSchema,
    validate_boundary_tensor_count,
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
pub use placement::{
    BindingPlacementError, PlacementPlan, PlacementPlanError, PlacementRank,
    ResolvedTensorPlacement, TensorSlice, place_addressable_member_bindings, place_weight_bindings,
    placement_selection,
};
pub use prefetch::{
    BackgroundPrefetchFailure, BackgroundPrefetchPanic, BackgroundPrefetchWorker,
    BackgroundPrefetchWorkerError, BackgroundThreadFinishError, PrefetchStoragePreparationError,
    PrefetchUnit, PreparedPrefetchStorage,
};
pub use realtime::{
    RealtimeCompletionAttachmentError, RealtimeFrameExecutionError, RealtimeFrameTransition,
    RealtimeGenerationBranch, RealtimeGenerationState, RealtimeGenerationTransactionError,
};
pub use realtime_executor::{
    PreparedRealtimeFrameExecutor, PrepublicationRealtimeFrame, RealtimeCompletionCreationError,
    RealtimeCoordinatorHostSource,
    RealtimeDecisionExecution, RealtimeFrameCompletionMechanism, RealtimeFrameCoordinatorError,
    RealtimeFrameHostObserver, RealtimeHostOutputUnavailable, RealtimePrepublicationError,
    SubmittedRealtimeFrame, execute_realtime_frame, execute_realtime_frame_view,
    RealtimeFrameExecutionView, RealtimeFrameExecutionUpdates,
};
pub use realtime_ingress::{
    MaterializedRealtimeInput, RealtimeHostTokenMaterializer, RealtimeIngressContract,
    RealtimeIngressError, RealtimeIngressSource, RealtimeInputMatrix, RealtimePayloadKind, RealtimeTokenKind, ValidatedRealtimeInput,
};
pub use realtime_interpreter::{
    CompletedRealtimeFrame, PreparedRealtimeFrame, RealtimeFrameInterpretationError,
    RealtimeFrameTensorMechanisms, NeuralRealtimeFrameTensorMechanisms, complete_realtime_frame, prepare_realtime_frame,
};
pub use realtime_mechanism_synthesis::{
    RealtimeMechanismFacts, RealtimeMechanismSupport, synthesize_realtime_capabilities,
};
pub use realtime_model::{
    ConstructedRealtimeExecution, ConstructedRealtimeModel, PreparedRealtimeModelContract,
    PreparedRealtimeTaskBindingPlan, RealizedRealtimePolicy, RealizedRealtimeState,
    RealtimeArchitectureConstructionIdentity, RealtimeArchitectureIdentity,
    RealtimeLayerwiseRuntime, RealtimeMaterializationComponent, RealtimeMaterializationTask,
    RealtimeModelConstructionError, RealtimeModelConstructionMechanisms,
    RealtimeModelContractError, RealtimeTaskBindingPlan, construct_realtime_model,
    preflight_realtime_materialization_tasks, realtime_task_binding_plan,
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
    ConstructedRealtimeResources, PreparedRealtimeRealization, RealtimeArchitectureProof,
    RealtimeArchitectureRequirements, RealtimeContractError, RealtimeExecutionRequirements,
    RealtimeIdentity, RealtimeMechanism, RealtimeMechanismCapabilities,
    RealtimeMechanismRequirements, RealtimeObservationRequirements, RealtimePreparationError,
    RealtimeSelectionError, RealtimeSelectionIssue, RealtimeSelectionRequest,
    RealtimeTopologyPolicy, RealtimeWeightComponentRequirement, RealtimeWeightComponentRole,
    RealtimeWeightLoweringRequirement, SelectedRealtimeRealization,
    SelectedRealtimeStateComponentRealization, SelectedRealtimeStateRealization,
    select_and_prepare_realtime_realization, select_realtime_realization,
};
pub use realtime_session::{
    RealtimeHistoryGeneration, RealtimeModelOwnerIdentity, RealtimeModelSessionIdentity,
    RealtimeSamplingReplacementError, RealtimeSamplingUpdateError, RealtimeSessionBranch,
    RealtimeSessionError, RealtimeSessionExecutionError, RealtimeSessionIncarnation,
    RealtimeSessionScheduler, RealtimeSessionState, RealtimeSessionTransactionError,
    ReleasedRealtimeSession,
};
pub use replicated_session::{
    DirectReplicatedTextExecution, DistributedSessionCheckpoint, DistributedStateCheckpoint,
    PartitionedRuntimeConstructionError, PartitionedSessionFactoryInput,
    PartitionedSessionPreparationError, PartitionedUnitScope, PredictionTargetOperation,
    PredictionStateLoanError,
    PreparedContractMaterialization, PreparedPartitionedRuntimeComponents,
    PreparedPartitionedSessionRuntime, PreparedReplicatedTextContract,
    PreparedReplicatedTextExecutionGeometry, PreparedSessionObservationError,
    PreparedTextContractError, ReplicatedRuntimeExecutionStrategy, ReplicatedTextExecutionStrategy,
    ReplicatedTextRuntime, ReplicatedTextSession, ReplicatedTextSessionCheckpoint,
    ReplicatedTextSessionError, ReplicatedTextSessionMechanisms, ReplicatedTextSessionReport,
    RoutedReplicatedTextExecution, SessionStateRealization, TransactionalPromptCacheMechanisms,
    construct_replicated_text_session, construct_replicated_text_session_with_execution,
    construct_replicated_text_session_with_runtime, prepare_default_partitioned_runtime,
    partitioned_materialization_addresses, partitioned_materialization_unit_layout,
    prepare_layered_text_contract, prepare_layered_text_contract_with_addressable_parameters,
    prepare_layered_text_contract_with_materialization,
    prepare_layered_text_contract_with_metadata, prepare_partitioned_session_runtime,
    prepare_partitioned_session_runtime_with_exclusions, prepare_replicated_text_contract,
    prepare_replicated_text_contract_with_addressable_parameters,
};
pub use replicated_text::{
    AddressableStorageCapabilities, AddressableStorageTiers, BackendMechanismCapabilities,
    GroupedOperationRequirement, ParameterTransformConstraint, ParameterTransformTarget,
    ReplicatedTextArchitecture, ReplicatedTextContractError,
    ReplicatedTextMaterializationPartitionPlan, ReplicatedTextMaterializationTask,
    ReplicatedTextOutputCompanion, ReplicatedTextOutputSelection, ReplicatedTextParameterOwner,
    ReplicatedTextParameterPresence, ReplicatedTextParameterRequirement,
    ReplicatedTextParameterRole, ReplicatedTextPhysicalSource, ReplicatedTextRequirements,
    ReplicatedTextSelectionError, ReplicatedTextSelectionRequest, ReplicatedTextStateAccess,
    ReplicatedTextTransformGroup, SelectedParameterRealization, SelectedReplicatedTextRealization,
    SelectedStateComponentRealization, SelectedStateRealization, StateComponentMechanism,
    StateComponentPlacement, StateMechanismCapabilities, StateStorageDtype,
    WeightLoweringCapability, WeightLoweringDescriptor, WeightLoweringKind,
    WeightResidencyMechanism, group_replicated_text_transform_tasks,
    locally_materialized_replicated_text_outputs,
    partition_selected_replicated_text_materialization_tasks,
    partitioned_replicated_text_materialization_tasks,
    plan_local_replicated_text_materialization_tasks, plan_replicated_text_materialization_tasks,
    replicated_text_materialization_tasks, select_replicated_text_realization,
    selected_materialization_task_bytes,
};
pub use residency::{
    DeviceLayerWindow, OffloadUnit, QuantizationCompanionBindings, ResidencyAcquisition,
    ResidencyController, ResidencyControllerError, ResidencyDeclarationError, ResidencyLease,
    ResidencyLeaseOwner, ResidencyLeaseStorage, ResidencyReport, ResidencyTransfer,
    ResidencyTransferOwner, ResidencyWindowError, ResidencyWindowManager, ResidentLayerGroup,
    ResidentLayerGroupReport, WeightBinding, WeightBindingPlan, WeightBindingSelectionError,
    WeightMaterializationReport,
};
pub use speculative::{RunSpeculativeGeneration, SpeculativeScheduler};
pub use speculative_selection::{
    ConstructedSpeculativeResources, PreparedSpeculativeRealization, SelectedSpeculativeCompletion,
    SelectedSpeculativePlacement, SelectedSpeculativeRealization, SelectedSpeculativeSampling,
    SelectedSpeculativeState, SpeculativeArchitectureCompatibilityProof, SpeculativeCaptureEntry,
    SpeculativeCaptureEnvelope, SpeculativeCaptureError, SpeculativeCaptureMetadata,
    SpeculativeCaptureSchema, SpeculativeContractError, SpeculativeIdentity,
    SpeculativeLaneIdentity, SpeculativeLaneIdentityRef, SpeculativeLaneIdentityView,
    SpeculativeMechanism, SpeculativeMechanismCapabilities, SpeculativeMechanismRequirements,
    SpeculativePlacementRequest, SpeculativePreparationError, SpeculativeRealizationRequirements,
    SpeculativeSelectionError, SpeculativeSelectionRequest,
    SpeculativeStateCacheIdentityIngredients, SpeculativeStrategyClass,
    SpeculativeStrategyRequirements, select_and_prepare_speculative_realization,
    select_and_prepare_speculative_realization_observed, select_speculative_realization,
};
pub use state::{
    ArchitectureStateFactory, ArchitectureStatePartitionError, ArchitectureStatePartitionPlan,
    ArchitectureStatePartitionRule, ArchitectureStatePlacement, ArchitectureStateRealizationError,
    DEFAULT_STATE_SEGMENT_ID, DeviceState, LayerRuntimeState, ModelStateIdentity,
    ResettableRuntimeLayerState, ResettableRuntimeState, RuntimeLayerState, RuntimeState,
    RuntimeStateComponents, SharedStateLayout, StateError, StateLayout, StateSegmentId,
    StateSegmentLifetime, StateSegmentSpec, realize_architecture_state,
};
pub use weight_residency::{
    AuxiliaryModuleResidency, AuxiliaryWeightRequirements, DENSE_TRANSFER_WINDOW,
    DenseDiskStreamLoadOptions, DenseTransferSchedule, DenseTransferScheduleError,
    ExecutionResidency, ExpertPass, LayerWeightResidency, LayerwiseLoadOptions,
    LayerwiseModelMetadata, OrdinaryWeightResidency, ParameterBankAccess, ParameterBankKey,
    ParameterBankLoadOptions, ParameterBankResidency, StaticUnitBindings, WeightResidency,
    WeightResidencyPolicyError,
};

pub use partitioned_execution::PartitionedMediaGroupExecutor;
