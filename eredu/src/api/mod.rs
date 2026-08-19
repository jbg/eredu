//! Backend-neutral language-model facade.
//!
//! This module is available without an execution backend. Enabling the
//! default `mlx` feature adds the concrete MLX loader, model implementations,
//! prepared-chat execution, and native runtime diagnostics.
//! MLX executable, cache, load-policy, and generation types live under
//! `backend::mlx`, not in this namespace.

mod media;
mod metadata;
mod portable;
mod request;
mod tokenizer;

pub use crate::runtime::chat::constraints::ConstraintError;
pub use crate::runtime::chat::{
    CapabilitySupport, ChatCapabilities, ChatTemplateIdentity, ChatTemplateRequest,
    NativeToolSupport, ParallelToolCallPolicy, PreparedChat, SemanticSupport, ToolChoice,
};
pub use request::{
    PreparedChatError, PreparedChatGenerationOutput, PreparedChatGenerationRequest,
    PreparedChatGenerationSettings, PreparedChatInput, PreparedChatMtpBatchLane,
    PreparedChatMtpBatchRequest, PreparedChatMtpError, PreparedChatMtpGenerationOptions,
    PreparedChatMtpGenerationRequest, PreparedChatSpeculativeConstraint,
};
pub use tokenizer::{chat_template_kwargs, load_tokenizer, TextMetadataError};

mod capability;
mod inspection;
mod loaded;
pub use inspection::{inspect_text_model, TextInspectionOptions};
pub use loaded::{LoadedModelLoadError, PlannedModelLoadError};
pub use media::MultimodalPreparationError;

pub use eredu_core::{
    Admission, AdmissionRejection, AdmissionRequest, AdmissionResult, AllocatorTelemetry,
    ArtifactModality, ArtifactTensorEncoding, Audio, AutomaticPlanRequest, AutomaticPlanner,
    AutomaticPlannerPolicy, AutomaticPlanningBackend, AutomaticPlanningError, AvailableMemory,
    BackendId, CacheStateStrategy, CapabilityError, DevicePlan, DraftPlacementPlan, DraftingPlan,
    DurationSeconds, EstimationCompleteness, ExecutionPlan, ExecutionPlanReport,
    ExecutionTelemetry, ExpertCachePlan, ExpertCacheTelemetry, HardwareBackendProfile,
    HardwareDeviceProfile, HardwareMemorySemantics, HardwareProfile, InputModalities,
    InputTokenCount, InspectionIssue, InspectionIssueCode, InspectionReadiness,
    InspectionRequirement, InspectionSeverity, Media, MediaBinding, MediaRequestError,
    ModelCapabilities, ModelCapabilityBackend, ModelInspectionReport, ModelKind,
    ModelResourceProfile, MtpTelemetry, MultimodalPreparationBackend, MultimodalPreparationFailure,
    MultimodalRequest, MultimodalSegment, ObservationKind, Observed, PhysicalMemorySemantics,
    PlanExplanation, PlanExplanationEntry, PlanExplanationLevel, ResidencyPlan, ResidencyTelemetry,
    RgbImage, RuntimeStateEstimate, SlidingWindowLayerCount, SpeculativeDraft,
    SpeculativeGenerationBackend, SpeculativeGenerationBatchOutput, SpeculativeGenerationOutput,
    StateMemoryAssumptions, StaticMemoryReport, TimingTelemetry, TokenizedMultimodalRequest,
    TokenizedMultimodalSegment, TransferTelemetry, Video, VideoSampling, WeightTransformationPlan,
    AUTOMATIC_SCHEMA_VERSION,
};
pub use eredu_text::tokenizer::{ModelChatTemplate, Tokenizer as ChatTokenizer};
pub use portable::{
    LoadedModel, LoadedTextModelConfig, PlannedModel, TextDecoder, TextDecoderError, TextModelError,
};

#[cfg(test)]
mod tests;
