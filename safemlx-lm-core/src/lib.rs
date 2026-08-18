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
/// High-level execution-backend contract.
pub mod backend;
/// Aggregate ownership and admission for backend-managed live caches.
pub mod cache;
/// Neutral checkpoint tensor descriptions and validation.
pub mod checkpoint;
/// Backend-neutral distributed scheduler consensus.
pub mod consensus;
/// Portable execution plans, capabilities, and telemetry.
pub mod execution;
/// Backend-independent generation lifecycle and output events.
pub mod generation;
/// Stable model and artifact identities.
pub mod model;
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
    inspect_artifact, plan_model_preparation, validate_preparation_policy, ArtifactFormat,
    ArtifactInspection, GgufArchitecture, MaterializationRoute, ModelArtifact, ModelConfiguration,
    ModelKind, ModelPreparationPlan, PreparationPolicy, QuantizationRequest, ResidencyRequest,
};
pub use attention::{AttentionPolicy, LayerSchedule, LayerScheduleError};
pub use backend::{
    Backend, BackendCapabilities, BackendDescriptor, BackendError, BackendSession, CollectiveScope,
    Completion, ControlledTextGeneration, ControlledTextGenerationError, ControlledToken,
    DeviceDescriptor, DistributedBackend, DistributedCapabilities, DistributedSession,
    DistributedSessionDescriptor, ModelRuntime, PreparedModel, Submission, TextGeneration,
    TextGenerationBackend, TextGenerationConfig, TokenFilter, TokenFilterController,
    TokenFilterError, TokenOutput, ValueDescriptor,
};
pub use generation::{
    resolve_generation_config, resolve_optimistic_reuse, CheckpointGenerationConfig, FinishReason,
    GenerationCancellationToken, GenerationConfigOverrides, GenerationError, GenerationPhase,
    GenerationSequence, MtpCancellationDisposition, MtpConfig, MtpRequestId, MtpRequestLifecycle,
    MtpRequestPhase, MtpSchedulerOptions, OptimisticReuseDecision, ResolvedGenerationConfig,
    SemanticEvent, SpeculativeCommitPlan, SpeculativeRound, SpeculativeTail, TokenCommit,
    TokenTerminalSignals,
};
pub use realtime::{
    RealtimeBackend, RealtimeCompletedStep, RealtimeConfigError, RealtimeError, RealtimeModel,
    RealtimeSampling, RealtimeScheduler, RealtimeSession, RealtimeSpeechConfig,
};
pub use residency::{
    BackgroundPrefetchReport, PrefetchAdmission, PrefetchCompletion, PrefetchDemandObservation,
    PrefetchDemandResolution, PrefetchExecutionState, PrefetchStateError, PrefetchWork,
};
pub use speculative::{
    cancel_pending_verification, propose_block, resolve_commit_and_publish,
    resolve_optimistic_branch, resolve_round, submit_verification_transaction,
    CompletedSpeculativeRequest, CompletedSpeculativeSchedule, MtpBatchOutput, MtpCapability,
    MtpCheckpointKind, MtpSchedulerStats, MtpStats, PendingSpeculativeVerification,
    ProposalDecision, PublishedSpeculativeVerification, ResolvedSpeculativeRound,
    SamplingPlacement, SpeculativeAction, SpeculativeCandidate, SpeculativeCommit,
    SpeculativeConstraint, SpeculativeContinuation, SpeculativeDraftBlock, SpeculativeDriverError,
    SpeculativeExecutionTopology, SpeculativeExecutor, SpeculativeOptimisticBranch,
    SpeculativeOutputRuntime, SpeculativePrefill, SpeculativeProposal,
    SpeculativePublicationStatus, SpeculativePublisher, SpeculativeRandomness, SpeculativeRequest,
    SpeculativeRequestTable, SpeculativeSampling, SpeculativeSchedule, SpeculativeTelemetry,
};
