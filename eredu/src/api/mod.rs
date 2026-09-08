//! Backend-neutral language-model facade.
//!
//! This module is available without an execution backend. Enabling the
//! `mlx` feature adds MLX runtime and diagnostic helpers. Models, requests,
//! generation controls, and realtime mechanisms retain their generic parameters;
//! import a concrete backend factory from its owning crate to select execution.
//!
//! Backend implementation traits and their errors are imported from
//! `eredu-core`, not this facade namespace.

mod media;
mod metadata;
mod portable;
pub mod realtime;
mod request;
mod tokenizer;

#[cfg(feature = "mlx")]
mod selected;

pub use request::{
    PreparedChatError, PreparedChatGenerationOutput, PreparedChatGenerationRequest,
    PreparedChatGenerationSettings, PreparedChatInput, PreparedChatSpeculativeBatchLane,
    PreparedChatSpeculativeBatchRequest, PreparedChatSpeculativeConstraint,
    PreparedChatSpeculativeError, PreparedChatSpeculativeGenerationOptions,
    PreparedChatSpeculativeGenerationRequest,
};
#[cfg(feature = "mlx")]
pub use selected::*;
pub use tokenizer::{chat_template_kwargs, load_tokenizer, TextMetadataError};

mod capability;
mod control;
mod inspection;
mod loaded;
mod observed;
pub use control::{
    ControlledGenerationBranch, ControlledGenerationError, ControlledGenerationRecord,
    ControlledGenerationSession, ControlledGenerationSnapshot, GenerationBranchMetadata,
    GenerationBranchOptions, GenerationOutputCheckpoint, GenerationSnapshotMetadata,
};
pub use eredu_core::{GenerationOutput, GenerationTiming, TextSamplingStrategy};
pub use eredu_runtime::execution_control::{SamplingOverride, SamplingStateFacts};
pub use inspection::{inspect_architecture, inspect_text_model, TextInspectionOptions};
pub use loaded::{LoadedModelLoadError, PlannedModelLoadError};
pub use media::MultimodalPreparationError;
pub use observed::{
    ObservedGenerationEvent, ObservedGenerationRecord, PreparedObservedGeneration, TraceLimits,
};

pub use portable::{
    LoadedModel, LoadedTextModelConfig, LoadedTextModelOptions, PlannedModel, TextDecoder,
    TextDecoderError, TextModelError,
};

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
