//! Backend-neutral language-model contracts and orchestration.
//!
//! This crate deliberately contains no tensor runtime. Backends own tensors,
//! streams, executable models, caches, and completion primitives; core owns
//! validation, lifecycle state, scheduling, and portable schemas.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

/// Portable architecture, artifact, request, and backend mechanism admission.
pub mod admission;
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
/// Bounded, admitted execution capture and portable transport records.
pub mod capture;
/// Neutral checkpoint tensor descriptions and validation.
pub mod checkpoint;
/// Portable scalar component topology and effective parameter relationships.
pub mod component;
/// Backend-neutral distributed scheduler consensus.
pub mod consensus;
/// Logical architecture and capture discovery contracts.
pub mod discovery;
pub mod parameters;
pub use discovery::*;
/// Portable execution plans, capabilities, and telemetry.
pub mod execution;
/// Completed-token execution control, discovery and snapshot resource contracts.
pub mod execution_control;
/// Backend-independent generation lifecycle and output events.
pub mod generation;
/// Exact output demand and geometry for bounded inference.
pub mod inference;
/// Portable identity for ordered, prepared model input.
pub mod input;
pub use inference::{
    InferenceGeometry, OutputDemand, TextControllerWorkspace, TextFilterWorkspace,
    TextInferencePolicy, TextPreparationReport,
};
/// Portable model-artifact inspection results.
pub mod inspection;
/// Validated portable activation and pre-dispatch routing interventions.
pub mod intervention;
/// Portable decoded-media requests and backend preparation inputs.
pub mod media;
/// Portable, explicitly requested execution observations.
pub mod observation;
/// Backend-generic realtime token-session execution and scheduling.
pub mod realtime;
/// Weight-residency ownership, capacity, and resource planning.
pub mod residency;
/// Portable text-run preparation status and cumulative reservation reports.
pub mod run_preparation;
/// Transactional fair work scheduler.
pub mod scheduler;
/// Exact session admission and unresolved-submission ownership.
pub mod session_authority;
/// High-level speculative execution contracts and orchestration.
pub mod speculative;
/// Parallel topology and placement planning.
pub mod topology;

pub use admission::{
    ArchitecturePreparationCapabilities, PreparationAdmission, PreparationAdmissionError,
    PreparationAdmissionRequest, PreparationMechanismCapabilities, admit_preparation,
};
pub use artifact::{
    ArtifactFormat, ArtifactInspection, GgufCompanionEncoding, GgufCompanionRequirement,
    GgufCompanionRole, LoadingProtocol, MaterializationRoute, ModelArtifact, ModelConfiguration,
    ModelConfigurationResolver, ModelPreparationPlan, PreparationPolicy, QuantizationRequest,
    ResidencyRequest, ResolvedModelConfiguration, ValidatedGguf, ValidatedGgufCompanion,
    gguf_u32_metadata_values, inspect_artifact, inspect_artifact_with_prepared_gguf_headers,
    inspect_artifact_with_safetensors_admission,
    plan_model_preparation, resolve_gguf_companions, validate_preparation_policy,
};
pub use attention::{AttentionPolicy, LayerSchedule, LayerScheduleError};
pub use automatic::{
    AUTOMATIC_SCHEMA_VERSION, AllocatorTelemetry, AutomaticPlanRequest, AutomaticPlanner,
    AutomaticPlannerPolicy, AutomaticPlanningBackend, AutomaticPlanningError,
    BoundedResidencyRequirement, CandidateAdmission, DurationSeconds, ExecutionPlanBackendFactory,
    ExecutionPlanReport, ExecutionPlanTarget, ExecutionPlanTargetLoadError,
    ExecutionPlanTargetSelection, ExecutionTelemetry, ExpertCacheTelemetry, ExternalDraftArtifact,
    HardwareBackendProfile, HardwareDeviceProfile, HardwareMemorySemantics, HardwareProfile,
    ModelResourceProfile, ObservationKind, Observed, ParameterMaterializationWorkspace,
    PlanExplanation, PlanExplanationEntry, PlanExplanationLevel, PreparedExecutionPlanTarget,
    RealizedDrafting, ResidencyTelemetry, RetainedAutomaticPlan, SelectedExecutionPlanDrafting,
    SelectedExecutionPlanTarget, SelectedRankResourceProfile, SpeculativeDecodingTelemetry,
    TimingTelemetry, TokenizerCompatibilityError, TokenizerCompatibilityProof, TransferTelemetry,
    realize_execution_plan_drafting, realize_execution_plan_target, select_execution_plan_drafting,
    select_execution_plan_target, speculative_decoding_telemetry,
};
pub use execution_control::{SamplingOverride, SamplingOverrideError, SamplingStateFacts, TextSamplingControlBackend};
pub use backend::{
    ProspectiveTokenController, TextTokenChoiceBoundary, TextSamplingBoundary,
    BackendDescriptor, BackendError, BackendFailure, BackendFailureKind, BackendProvider,
    BackendSession, BoundedCompletion, BoundedCompletionOutcome, BoundedCompletionWait,
    BoundedCompletionWaitError, BoundedSubmissionOutcome, CollectiveGroupDescriptor,
    CollectiveGroupId, CollectiveScope, Completion, CompletionCancellationMode,
    ControlledTextGeneration, ControlledTextGenerationError, ControlledToken,
    ControllerDeclarationData, DeviceCapabilities, DeviceDescriptor, DistributedBackend,
    DistributedCapabilities, DistributedCommitEpoch, DistributedCommitOutcome,
    DistributedCommitPhase, DistributedSession, DistributedSessionDescriptor,
    ErasedSharedStorageOwner, GenerationDecoderError, GenerationDecoderInput,
    GenerationDecoderOutput, GenerationPlainText, GenerationPlainTextEvent,
    GenerationPlainTextEvents, GenerationPlainTextProjection, GenerationSequenceAdmissionError,
    GenerationSequenceBankRejection, GenerationSequenceConsumerLayout,
    GenerationSequencePreparation, GenerationSequenceRequest, HostMetadataAccount, HostMetadataFunding, HostMetadataFundingError,
    HostPreparationAuthority, InspectableBackendSession, ModelCapabilityBackend, ModelLoadError,
    ModelLoadingBackend, ModelRuntime, MultimodalPreparationBackend, MultimodalPreparationFailure,
    OriginalTextResumeKind, OriginalTextResumeOptions, TextResumeFacts, TextResumeSourceFacts, OriginalSourceWitness, PreparedControllerSource, PREPARED_PROMPT_ATTRIBUTION_VERSION,
    PackedTokenFilter, PackedTokenFilterError, PendingTextInput, PreparedControlInput, PreparedControlInputBackend, PreparedControlInputError,
    PreparedModel, PreparedPromptAttribution, PreparedPromptSegment, PreparedPromptSegmentPlan,
    PreparedRequestRejection, PreparedSessionReset, PromptTokenAttribution,
    SelectedModelPreparation, SessionCapabilities, SessionCapabilityError,
    SessionResetPreparationBackend, SessionResetReadiness, SharedBackendFailure,
    SharedControllerBytes, SharedControllerDeclaration, SharedControllerSource,
    SharedPromptAttribution, SharedStorageAttachmentError, SharedStorageDomain,
    SharedStorageIdentity, SharedStorageOwner, SharedStorageRetirement, SharedTokenFilter,
    SpeculativeTokenFilterController, Submission, TextContextError,
    TextContinuationBoundary, TextContinuationError, TextContinuationIdentity,
    TextControllerContract, TextControllerContractError, TextControllerStorage, TextDriverIdentity, TextBranchSource, TextGenerationBranch, TextBranchFenced,
    TextGeneration, TextGenerationBackend, TextGenerationConfig, TextGenerationContinuation,
    TextGenerationDriver, TextGenerationInput, TextPolicyIdentity, TextPreparationInput,
    TextPreparationOptions, TextResumeBackend, TextRunIdentity, TextSamplingStrategy,
    TextSnapshotSource, TextStepContext, TokenFilter, TokenFilterController, TokenFilterError,
    TokenIdsInputPlan, TokenInputRejection, TokenOutput, TokenSamplingDecision, ValueDescriptor,
    load_model, prepare_inspected_model, prepare_inspected_model_config,
    text_generation_control_bytes, text_resume_control_bytes,
};
pub use capability::{
    Admission, AdmissionRejection, AdmissionRequest, AdmissionResult, AvailableMemory,
    CacheStateStrategy, CapabilityError, EstimationCompleteness, InputModalities, InputTokenCount,
    ModelCapabilities, PhysicalMemorySemantics, RuntimeStateEstimate, RuntimeStateFacts,
    SelectedStateBacking, SlidingWindowLayerCount, StateMemoryAssumptions, StateMemoryLayout,
    StateWindowDestinationError, StateWindowPlan, StaticMemoryReport, apply_admission_policy,
    apply_admission_policy_with_incremental, check_admission_context, estimate_runtime_state,
    estimate_runtime_state_facts,
};
pub use capability::{
    AdmissionObservation, AdmissionPolicyDecision, AdmissionPolicyError, AdmissionRequirements,
    AdmissionStateRequirements, BorrowedAdmissionRejection, BorrowedAdmissionResult,
    ExecutionWorkspaceRequirements, SelectedStateRequirements, apply_admission_requirements,
    check_admission_context_borrowed,
};
pub use capability::{ExecutionWorkspaceEstimate, WorkspaceBound};
pub use execution::{
    BackendId, DEFAULT_MAX_CACHED_SHARDS, DevicePlan, DraftPlacementPlan, DraftingPlan,
    EXECUTION_PLAN_SCHEMA_VERSION, ExecutionPlan, ExecutionPlanError, ExpertCachePlan,
    ResidencyPlan, WeightTransformationPlan,
};
pub use generation::{
    CheckpointGenerationConfig, FinishReason, GenerationCancellationToken,
    GenerationConfigOverrides, GenerationError, GenerationOutput, GenerationPlainTextOutput,
    GenerationSequence, GenerationSequenceStorage, GenerationText, GenerationTiming,
    GenerationTokenIdStorage, GenerationTokenIds, GenerationTokenIdsIntoIter,
    OwnedGenerationStorage, OptimisticReuseDecision, ResolvedGenerationConfig,
    RetainedGenerationSequence, RetainedGenerationSequenceCopy, RetainedGenerationSequenceStorage,
    RetainedGenerationStorage, RetainedGenerationStorageOwner, RetainedSequenceConstructionError,
    RetainedSequenceCopyMismatch, RetainedSequencePreparationError, SemanticEvent,
    SpeculativeCancellationDisposition, SpeculativeCommitPlan, SpeculativeConfig,
    SpeculativeRequestId, SpeculativeRequestLifecycle, SpeculativeRequestStatus, SpeculativeRound,
    SpeculativeSchedulerOptions, SpeculativeTail, TokenCommit, TokenTerminalSignals,
    resolve_generation_config, resolve_optimistic_reuse,
};
pub use input::{
    InputExtent, InputIdentityMap, InputMetadataKey, InputModality, InputPartDescriptor,
    InputPayloadKind, InputTensorIdentity, PreparedInputError, PreparedInputIdentity,
};
pub use inspection::{
    ArtifactModality, ArtifactTensorEncoding, InspectionIssue, InspectionIssueCode,
    InspectionReadiness, InspectionRequirement, InspectionSeverity, MediaFeatureAvailability,
    MediaProjectorRequirement, ModelInspectionReport, RealizedInspectionOutcomes,
    assemble_portable_model_inspection, finalize_realized_model_inspection,
    media_feature_readiness, record_media_projector_inspection, record_processor_inspection,
    reject_portable_artifact_inspection,
};
pub use media::{
    Audio, Media, MediaBinding, MediaRequestError, MultimodalRequest, MultimodalSegment, RgbImage,
    TokenizedMultimodalRequest, TokenizedMultimodalSegment, Video, VideoSampling,
};
pub use observation::{
    AUDIO_PROJECTOR_OUTPUT_OBSERVATION_PATH, InspectedOutput,
    MODALITY_MERGE_OUTPUT_OBSERVATION_PATH, MODEL_LOGITS_OBSERVATION_PATH, ObservationError,
    ObservationRequest, ObservationSelector, ObservationSet, ObservationValue,
    PROCESSOR_OUTPUT_OBSERVATION_PATH, SharedTensorObservation, TensorObservation,
    TensorObservationData, VISION_PROJECTOR_OUTPUT_OBSERVATION_PATH,
};
pub use realtime::{
    MAX_REALTIME_FRAME_DELAY, RealtimeConfigError, RealtimeDecisionDiagnostics, RealtimeError,
    RealtimeForcedSource, RealtimeFrameConvention, RealtimeFrameForcing,
    RealtimeFrameScheduleState, RealtimeFrameSlot, RealtimeFrameTransition,
    RealtimeInputDescriptorError, RealtimeInputFrame, RealtimeOutputFrame, RealtimeSampling,
    RealtimeScheduleError, RealtimeSlotCoordinate, RealtimeSlotOccupancy, RealtimeSpeechConfig,
    RealtimeTargetDecision, RealtimeTargetSource, RealtimeTemporalSource,
};
pub use residency::{
    BackgroundPrefetchReport, PrefetchAdmission, PrefetchCompletion, PrefetchDemandObservation,
    PrefetchDemandResolution, PrefetchExecutionState, PrefetchStateError, PrefetchWork,
};
pub use session_authority::{
    SessionAdmission, SessionAdmissionError, SessionAuthority, SessionAuthorityError,
    SubmissionLease,
};
pub use speculative::{
    CompletedSpeculativeRequest, CompletedSpeculativeSchedule, PendingSpeculativeVerification,
    PreparedSpeculativeLane, ProposalDecision, PublishedSpeculativeResult,
    PublishedSpeculativeVerification, ResolvedSpeculativeRound, SamplingPlacement,
    SpeculativeAction, SpeculativeBuffer, SpeculativeBufferAllocationError,
    SpeculativeBufferIntoIter, SpeculativeCallbackPublisher, SpeculativeCandidate,
    SpeculativeCapability, SpeculativeCommit, SpeculativeConfiguration,
    SpeculativeConfigurationError, SpeculativeConstraint, SpeculativeContinuation,
    SpeculativeDraft, SpeculativeDraftBlock, SpeculativeDraftRandomPosition,
    SpeculativeDraftSource, SpeculativeDriverError, SpeculativeEventCallback,
    SpeculativeExecutionTopology, SpeculativeExecutor, SpeculativeGenerationBackend,
    SpeculativeGenerationBatchOutput, SpeculativeGenerationBatchRequest, SpeculativeGenerationLane,
    SpeculativeGenerationOutput, SpeculativeGenerationVisitor, SpeculativeLifecycleObserver,
    SpeculativeLifecycleStage, SpeculativeOptimisticBranch, SpeculativeOutputError,
    SpeculativeOutputRuntime, SpeculativePrefill, SpeculativePrefillOutcome, SpeculativeProposal,
    SpeculativePublicationStatus, SpeculativePublisher, SpeculativeRandomness, SpeculativeRequest,
    SpeculativeRequestIdentity, SpeculativeRequestTable, SpeculativeSampling, SpeculativeSchedule,
    SpeculativeScheduleState, SpeculativeSchedulerStats, SpeculativeSemanticConstraint,
    SpeculativeSequence,
    SpeculativeSequenceAllocationError, SpeculativeSequenceRef, SpeculativeStats,
    SpeculativeStatsCounters, SpeculativeTelemetry, SpeculativeTokenIds,
    SpeculativeTokenIdsIntoIter, SpeculativeValues, SpeculativeValuesIntoIter,
    cancel_pending_verification, decide_speculative_proposal, propose_block,
    resolve_commit_and_publish, resolve_optimistic_branch, resolve_round,
    speculative_acceptance_probability, submit_verification_transaction,
};
pub use topology::{
    ParallelAxis, ParallelCoordinates, ParallelRankTopology, ParallelTopology, SubgroupMembership,
    TopologyError, TopologyPreflightReport, balanced_contiguous_range,
};

mod session_reset;
pub use session_reset::{
    SessionResetAcceptance, SessionResetClaim, SessionResetLimits, SessionResetRejection,
};

// Semantic payloads preserve their actual host owner through caller retention.
pub use generation::{SemanticText, SemanticTextAllocationError};

pub use generation::{SemanticState, SemanticStateOwner};
