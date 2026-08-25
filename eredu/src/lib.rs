//! Backend-neutral model loading, generation, and realtime facade.
//!
//! [`api`] and [`runtime`] remain available without an execution backend. The
//! `mlx` or `cuda` feature adds the MLX model implementations and runtime.
//!
//! Backend implementation contracts are imported from their owning crates:
//!
//! ```compile_fail
//! use eredu::BackendProvider;
//! ```
//!
//! ```compile_fail
//! use eredu::BackendSession;
//! ```
//!
//! ```compile_fail
//! use eredu::PreparedModel;
//! ```
//!
//! ```compile_fail
//! use eredu::Completion;
//! ```

#![warn(missing_docs)]
#![cfg_attr(test, allow(dead_code))]
// Backend execution boundaries intentionally pass complete runtime context;
// concrete models remain unboxed; builder helpers can return typed builders;
// explicit drops delimit provider borrows; and MLX completions are process-local.
#![allow(
    clippy::arc_with_non_send_sync,
    clippy::drop_non_drop,
    clippy::large_enum_variant,
    clippy::new_ret_no_self,
    clippy::too_many_arguments,
    clippy::type_complexity
)]

/// High-level model loading, dispatch, and request APIs.
pub mod api;
/// Backend-independent chat and committed-generation orchestration.
pub mod runtime;
pub use api::{inspect_text_model, PlannedModelLoadError, TextInspectionOptions};
pub use eredu_architectures::configuration::inspect_artifact;
pub use eredu_architectures::moshi::{
    prepare_realtime_model, RealtimePreparationError, RealtimePreparationPlan,
};
pub use eredu_architectures::{
    prepare_external_assistant, ExternalAssistantCheckpoint, ExternalAssistantPreparationPlan,
};
pub use eredu_architectures::{GgufArchitecture, ModelKind};
pub use eredu_checkpoint::{AffineQuantization, WeightQuantization};
pub use eredu_core::artifact::{
    plan_model_preparation, ArtifactFormat, ArtifactInspection, LoadingProtocol,
    MaterializationRoute, ModelArtifact, ModelConfiguration, ModelPreparationPlan,
    PreparationPolicy, QuantizationRequest, ResidencyRequest,
};
pub use eredu_core::cache::{
    CachePolicyError, PromptCacheDescriptor, PromptCacheError, PromptCacheManifest,
    PromptCacheOptions, PromptCacheStateSegment, PROMPT_CACHE_SCHEMA_VERSION,
};
pub use eredu_core::generation::{
    CheckpointGenerationConfig, FinishReason, GenerationCancellationToken,
    GenerationConfigOverrides, MtpConfig, MtpRequestId, MtpRequestPhase, MtpSchedulerOptions,
    ResolvedGenerationConfig, SemanticEvent,
};
pub use eredu_core::{
    Admission, AdmissionRejection, AdmissionRequest, AdmissionResult, AllocatorTelemetry,
    ArtifactModality, ArtifactTensorEncoding, Audio, AutomaticPlanRequest, AutomaticPlanner,
    AutomaticPlannerPolicy, AutomaticPlanningError, AvailableMemory, BackendCapabilities,
    BackendDescriptor, BackendId, CacheStateStrategy, CapabilityError, DeviceDescriptor,
    DevicePlan, DistributedCapabilities, DraftPlacementPlan, DraftingPlan, DurationSeconds,
    EstimationCompleteness, ExecutionPlan, ExecutionPlanReport, ExecutionTelemetry,
    ExpertCachePlan, ExpertCacheTelemetry, GrowingState, HardwareBackendProfile,
    HardwareDeviceProfile, HardwareMemorySemantics, HardwareProfile, InputModalities,
    InputTokenCount, InspectedOutput, InspectionIssue, InspectionIssueCode, InspectionReadiness,
    InspectionRequirement, InspectionSeverity, Media, MediaBinding, MediaRequestError,
    ModelCapabilities, ModelInspectionReport, ModelResourceProfile, MtpCapability,
    MtpCheckpointKind, MtpTelemetry, MultimodalRequest, MultimodalSegment, ObservationError,
    ObservationKind, ObservationRequest, ObservationSelector, ObservationSet, ObservationValue,
    Observed, ParallelAxis, ParallelTopology, PhysicalMemorySemantics, PlanExplanation,
    PlanExplanationEntry, PlanExplanationLevel, RealtimeCompletedStep, RealtimeConfigError,
    RealtimeDecisionDiagnostics, RealtimeError, RealtimeFrameConvention, RealtimeFrameForcing,
    RealtimeFrameScheduleState, RealtimeFrameSlot, RealtimeFrameTransition, RealtimeInputFrame,
    RealtimeOutputFrame, RealtimeSampling, RealtimeScheduleError, RealtimeScheduler,
    RealtimeSlotCoordinate, RealtimeSlotOccupancy, RealtimeSpeechConfig, RealtimeTargetDecision,
    RealtimeTargetSource, RealtimeTemporalSource, ResidencyPlan, ResidencyTelemetry, RgbImage,
    RuntimeStateEstimate, SlidingWindowLayerCount, SpeculativeDraft,
    SpeculativeGenerationBatchOutput, SpeculativeGenerationBatchRequest, SpeculativeGenerationLane,
    SpeculativeGenerationOutput, StateMemoryAssumptions, StaticMemoryReport, TensorObservation,
    TensorObservationData, TextGenerationConfig, TextSamplingStrategy, TimingTelemetry,
    TokenFilter, TokenFilterError, TokenOutput, TokenizedMultimodalRequest,
    TokenizedMultimodalSegment, TokenizerCompatibilityError, TokenizerCompatibilityProof,
    TransferTelemetry, ValueDescriptor, Video, VideoSampling, WeightTransformationPlan,
    AUTOMATIC_SCHEMA_VERSION, EXECUTION_PLAN_SCHEMA_VERSION,
};
