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
    Backend, BackendCapabilities, BackendDescriptor, BackendError, BackendSession, Completion,
    DeviceDescriptor, PreparedModel, Submission,
};
pub use generation::{
    resolve_generation_config, resolve_optimistic_reuse, CheckpointGenerationConfig, FinishReason,
    GenerationCancellationToken, GenerationConfigOverrides, GenerationError, GenerationPhase,
    GenerationSequence, MtpCancellationDisposition, MtpConfig, MtpRequestId, MtpRequestLifecycle,
    MtpRequestPhase, MtpSchedulerOptions, OptimisticReuseDecision, ResolvedGenerationConfig,
    SemanticEvent, SpeculativeCommitPlan, SpeculativeRound, SpeculativeTail, TokenCommit,
    TokenTerminalSignals,
};
pub use speculative::{
    propose_block, resolve_round, ProposalDecision, ResolvedSpeculativeRound, SamplingPlacement,
    SpeculativeAction, SpeculativeCandidate, SpeculativeCommit, SpeculativeConstraint,
    SpeculativeDriverError, SpeculativeExecutionTopology, SpeculativeExecutor, SpeculativePrefill,
    SpeculativeProposal, SpeculativeRandomness, SpeculativeSampling, SpeculativeSchedule,
};
