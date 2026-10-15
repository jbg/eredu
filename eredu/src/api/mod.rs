//! Backend-neutral language-model facade.
//!
//! This module is available without an execution backend. Enabling the
//! `mlx` feature adds MLX runtime and diagnostic helpers. Models, requests,
//! generation controls, and realtime mechanisms retain their generic parameters;
//! import a concrete backend factory from its owning crate to select execution.
//!
//! Application errors are backend-independent. Native failures are retained as
//! sources in `eredu_core::BackendFailure`; backend implementation traits remain
//! in `eredu-core`.

mod errors;
pub use errors::{DevicePlanError, ExpertCacheBenchmarkError};

mod chat_input;
mod media;
pub use chat_input::*;
mod metadata;
mod portable;
mod session;
pub use session::*;
pub mod realtime;
mod request;
mod tokenizer;

#[cfg(feature = "mlx")]
mod selected;

pub use request::{
    GenerationRequest, OutputMode, PreparedChatError, PreparedChatGenerationSettings,
    PreparedChatInput, PreparedChatSpeculativeBatchLane, PreparedChatSpeculativeBatchRequest,
    PreparedChatSpeculativeConstraint,
};
#[cfg(feature = "mlx")]
pub use selected::*;
pub use tokenizer::{chat_template_kwargs, load_tokenizer, TextMetadataError};

mod capability;
mod control;
mod controlled_speculative;
pub use controlled_speculative::*;
mod inspection;
mod prepared;
pub use prepared::*;
mod metadata_inspection;
pub use metadata_inspection::*;
mod memory_estimation;
pub use memory_estimation::*;
mod forecast;
pub use forecast::*;
mod draft;
mod loaded;
mod observed;
mod parameters;
pub use control::{
    ControlledGenerationBranch, ControlledGenerationError, ControlledGenerationRecord,
    ControlledGenerationSession, ControlledGenerationSnapshot, GenerationBranchMetadata,
    GenerationBranchOptions, GenerationOutputCheckpoint, GenerationSnapshotMetadata,
};
pub use eredu_core::decoding::{
    DecodingCapabilities, DecodingMode, DecodingModeError, DecodingState,
    ModeChangeAcknowledgement, SpeculationOptions, SpeculationOptionsRequest,
};
pub use eredu_core::DraftAdapterSource;
pub use eredu_core::{GenerationTiming, PrefillChunkPolicy, TextSamplingStrategy};
pub use eredu_runtime::execution_control::{SamplingOverride, SamplingStateFacts};
pub use inspection::{
    inspect_architecture, ArtifactMetadata, CheckpointMetadata, GgufCompanionMetadata, GgufHeader,
    MetadataProvenance, ModelInspectionOutcome, SafetensorsHeader, TextInspectionOptions,
};
pub use loaded::{LoadedModelLoadError, PlannedModelLoadError};
pub use media::MultimodalPreparationError;
pub use observed::{
    ObservedGenerationEvent, ObservedGenerationRecord, PreparedObservedGeneration, TraceLimits,
    GENERATION_SCHEMA_VERSION,
};

pub use portable::{
    GeneratedToken, LoadedModel, LoadedTextModelConfig, TextDecoder, TextDecoderError,
    TextGeneration, TextModelError, TextModelOptions,
};

/// Terminal metadata shared by target-only and speculative generation.
pub type GenerationOutput<S = GenerationStats> = eredu_core::GenerationOutput<S>;

/// Portable failure reported by prepared-chat constraint state.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{message}")]
pub struct ConstraintError {
    message: String,
}

impl ConstraintError {
    pub(crate) fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

#[cfg(test)]
mod tests;
