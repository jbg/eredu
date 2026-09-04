//! Backend-neutral language-model contracts and orchestration.
//!
//! This crate deliberately contains no tensor runtime. Backends own tensors,
//! streams, executable models, caches, and completion primitives; core owns
//! validation, lifecycle state, scheduling, and portable schemas.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

/// Portable artifact inspection and model-preparation planning.
pub mod artifact;
/// Validated decoder attention schedules.
pub mod attention;
/// Backend-neutral automatic execution planning.
pub mod automatic;
/// High-level execution-backend contract.
pub mod backend;
/// Aggregate ownership and admission for backend-managed live caches.
pub mod cache;
/// Portable model capabilities, runtime-state accounting, and admission policy.
pub mod capability;
/// Neutral checkpoint tensor descriptions and validation.
pub mod checkpoint;
/// Backend-neutral distributed scheduler consensus.
pub mod consensus;
/// Portable execution plans, capabilities, and telemetry.
pub mod execution;
/// Backend-independent generation lifecycle and output events.
pub mod generation;
/// Portable identity for ordered, prepared model input.
pub mod input;
/// Portable model-artifact inspection results.
pub mod inspection;
/// Portable decoded-media requests and backend preparation inputs.
pub mod media;
/// Portable, explicitly requested execution observations.
pub mod observation;
/// Backend-generic realtime token-session execution and scheduling.
pub mod realtime;
/// Weight-residency ownership, capacity, and resource planning.
pub mod residency;
/// Transactional fair work scheduler.
pub mod scheduler;
/// High-level speculative execution contracts and orchestration.
pub mod speculative;
/// Parallel topology and placement planning.
pub mod topology;

pub use artifact::{
    gguf_u32_metadata_values, inspect_artifact, plan_model_preparation, resolve_gguf_companions,
    validate_preparation_policy, ArtifactFormat, ArtifactInspection, GgufCompanionEncoding,
    GgufCompanionRequirement, GgufCompanionRole, LoadingProtocol, MaterializationRoute,
    ModelArtifact, ModelConfiguration, ModelConfigurationResolver, ModelPreparationPlan,
    PreparationPolicy, QuantizationRequest, ResidencyRequest, ResolvedModelConfiguration,
    ValidatedGguf, ValidatedGgufCompanion,
};
pub use attention::{AttentionPolicy, LayerSchedule, LayerScheduleError};
pub use automatic::{
    realize_execution_plan_drafting, realize_execution_plan_target, AllocatorTelemetry,
    AutomaticPlanRequest, AutomaticPlanner, AutomaticPlannerPolicy, AutomaticPlanningBackend,
    AutomaticPlanningError, BoundedResidencyRequirement, CandidateAdmission, DurationSeconds,
    ExecutionPlanBackendFactory, ExecutionPlanReport, ExecutionPlanTarget, ExecutionTelemetry,
    ExpertCacheTelemetry, ExternalDraftArtifact, HardwareBackendProfile, HardwareDeviceProfile,
    HardwareMemorySemantics, HardwareProfile, ModelResourceProfile, ObservationKind, Observed,
    PlanExplanation, PlanExplanationEntry, PlanExplanationLevel, RealizedDrafting,
    ResidencyTelemetry, SpeculativeDecodingTelemetry, TimingTelemetry, TokenizerCompatibilityError,
    TokenizerCompatibilityProof, TransferTelemetry, AUTOMATIC_SCHEMA_VERSION,
};
pub use backend::{
    load_model, prepare_inspected_model, BackendDescriptor, BackendError, BackendProvider,
    BackendSession, BoundedCompletion, BoundedCompletionOutcome, BoundedCompletionWait,
    BoundedCompletionWaitError, BoundedSubmissionOutcome, CollectiveGroupDescriptor,
    CollectiveGroupId, CollectiveScope, Completion, CompletionCancellationMode,
    ControlledTextGeneration, ControlledTextGenerationError, ControlledToken, DeviceCapabilities,
    DeviceDescriptor, DistributedBackend, DistributedCapabilities, DistributedCommitEpoch,
    DistributedCommitOutcome, DistributedCommitPhase, DistributedSession,
    DistributedSessionDescriptor, InspectableBackendSession, ModelCapabilityBackend,
    ModelLoadError, ModelLoadingBackend, ModelRuntime, MultimodalPreparationBackend,
    MultimodalPreparationFailure, PreparedModel, SessionCapabilities, SessionCapabilityError,
    SpeculativeTokenFilterController, Submission, TextGeneration, TextGenerationBackend,
    TextGenerationConfig, TextSamplingStrategy, TokenFilter, TokenFilterController,
    TokenFilterError, TokenOutput, ValueDescriptor,
};
pub use capability::{
    apply_admission_policy, estimate_runtime_state, Admission, AdmissionRejection,
    AdmissionRequest, AdmissionResult, AvailableMemory, CacheStateStrategy, CapabilityError,
    EstimationCompleteness, InputModalities, InputTokenCount, ModelCapabilities,
    PhysicalMemorySemantics, RuntimeStateEstimate, SlidingWindowLayerCount, StateMemoryAssumptions,
    StateMemoryLayout, StaticMemoryReport,
};
pub use execution::{
    BackendId, DevicePlan, DraftPlacementPlan, DraftingPlan, ExecutionPlan, ExecutionPlanError,
    ExpertCachePlan, ResidencyPlan, WeightTransformationPlan, DEFAULT_MAX_CACHED_SHARDS,
    EXECUTION_PLAN_SCHEMA_VERSION,
};
pub use generation::{
    resolve_generation_config, resolve_optimistic_reuse, CheckpointGenerationConfig, FinishReason,
    GenerationCancellationToken, GenerationConfigOverrides, GenerationError, GenerationSequence,
    OptimisticReuseDecision, ResolvedGenerationConfig, SemanticEvent,
    SpeculativeCancellationDisposition, SpeculativeCommitPlan, SpeculativeConfig,
    SpeculativeRequestId, SpeculativeRequestLifecycle, SpeculativeRequestStatus, SpeculativeRound,
    SpeculativeSchedulerOptions, SpeculativeTail, TokenCommit, TokenTerminalSignals,
};
pub use input::{
    InputExtent, InputMetadataKey, InputModality, InputPartDescriptor, InputPayloadKind,
    InputTensorIdentity, PreparedInputError, PreparedInputIdentity,
};
pub use inspection::{
    ArtifactModality, ArtifactTensorEncoding, InspectionIssue, InspectionIssueCode,
    InspectionReadiness, InspectionRequirement, InspectionSeverity, ModelInspectionReport,
};
pub use media::{
    Audio, Media, MediaBinding, MediaRequestError, MultimodalRequest, MultimodalSegment, RgbImage,
    TokenizedMultimodalRequest, TokenizedMultimodalSegment, Video, VideoSampling,
};
pub use observation::{
    InspectedOutput, ObservationError, ObservationRequest, ObservationSelector, ObservationSet,
    ObservationValue, TensorObservation, TensorObservationData,
    AUDIO_PROJECTOR_OUTPUT_OBSERVATION_PATH, MODALITY_MERGE_OUTPUT_OBSERVATION_PATH,
    MODEL_LOGITS_OBSERVATION_PATH, PROCESSOR_OUTPUT_OBSERVATION_PATH,
    VISION_PROJECTOR_OUTPUT_OBSERVATION_PATH,
};
pub use realtime::{
    load_realtime_model, load_realtime_model_with_options, RealtimeBackend, RealtimeCompletedStep,
    RealtimeConfigError, RealtimeDecisionDiagnostics, RealtimeError, RealtimeFrameConvention,
    RealtimeFrameForcing, RealtimeFrameScheduleState, RealtimeFrameSlot, RealtimeFrameTransition,
    RealtimeInputFrame, RealtimeModel, RealtimeModelLoadingBackend, RealtimeOutputFrame,
    RealtimeSampling, RealtimeScheduleError, RealtimeScheduler, RealtimeSession,
    RealtimeSlotCoordinate, RealtimeSlotOccupancy, RealtimeSpeechConfig, RealtimeTargetDecision,
    RealtimeTargetSource, RealtimeTemporalSource, MAX_REALTIME_FRAME_DELAY,
};
pub use residency::{
    BackgroundPrefetchReport, PrefetchAdmission, PrefetchCompletion, PrefetchDemandObservation,
    PrefetchDemandResolution, PrefetchExecutionState, PrefetchStateError, PrefetchWork,
};
pub use speculative::{
    cancel_pending_verification, decide_speculative_proposal, propose_block,
    resolve_commit_and_publish, resolve_optimistic_branch, resolve_round,
    speculative_acceptance_probability, submit_verification_transaction,
    CompletedSpeculativeRequest, CompletedSpeculativeSchedule, PendingSpeculativeVerification,
    PreparedSpeculativeLane, ProposalDecision, PublishedSpeculativeResult,
    PublishedSpeculativeVerification, ResolvedSpeculativeRound, SamplingPlacement,
    SpeculativeAction, SpeculativeCallbackPublisher, SpeculativeCandidate, SpeculativeCapability,
    SpeculativeCommit, SpeculativeConstraint, SpeculativeContinuation, SpeculativeDraft,
    SpeculativeDraftBlock, SpeculativeDraftRandomPosition, SpeculativeDraftSource,
    SpeculativeDriverError, SpeculativeExecutionTopology, SpeculativeExecutor,
    SpeculativeGenerationBackend, SpeculativeGenerationBatchOutput,
    SpeculativeGenerationBatchRequest, SpeculativeGenerationLane, SpeculativeGenerationOutput,
    SpeculativeGenerationVisitor, SpeculativeLifecycleObserver, SpeculativeLifecycleStage,
    SpeculativeOptimisticBranch, SpeculativeOutputError, SpeculativeOutputRuntime,
    SpeculativePrefill, SpeculativeProposal, SpeculativePublicationStatus, SpeculativePublisher,
    SpeculativeRandomness, SpeculativeRequest, SpeculativeRequestTable, SpeculativeSampling,
    SpeculativeSchedule, SpeculativeSchedulerStats, SpeculativeSemanticConstraint,
    SpeculativeSemanticState, SpeculativeStats, SpeculativeTelemetry,
};
pub use topology::{
    balanced_contiguous_range, ParallelAxis, ParallelCoordinates, ParallelRankTopology,
    ParallelTopology, SubgroupMembership, TopologyError, TopologyPreflightReport,
};
