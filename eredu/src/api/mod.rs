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
pub use tokenizer::{TextMetadataError, chat_template_kwargs, load_tokenizer};

mod capability;
mod control;
mod controlled_speculative;
pub use controlled_speculative::*;
mod inspection;
mod loaded;
mod observed;
mod parameters;
pub use control::{
    ControlRecordMode, ControlledGenerationBranch, ControlledGenerationError,
    ControlledGenerationRecord, ControlledGenerationSession, ControlledGenerationSnapshot,
    GenerationBranchMetadata, GenerationBranchOptions, GenerationOutputCheckpoint,
    GenerationSnapshotMetadata, LegacyText, PREPARED_EXECUTION_CONTROL_SCHEMA_VERSION,
    PreparedControlledGenerationBranch, PreparedControlledGenerationEvent,
    PreparedControlledGenerationRecord, PreparedControlledGenerationSession,
    PreparedControlledGenerationSnapshot, PreparedControlledInput, PreparedControlledRecordData,
    PreparedControlledWireRecord, PreparedGenerationBranchMetadata, PreparedGenerationSnapshotData,
    PreparedGenerationSnapshotMetadata, PreparedInputInstrumentation, PreparedInputV2,
    PreparedInstrumentationRecord,
};
pub use eredu_core::{
    GenerationOutput, GenerationPlainTextEvent, GenerationPlainTextOutput, GenerationTiming,
    TextInferencePolicy, TextSamplingStrategy,
};
pub use eredu_runtime::execution_control::{SamplingOverride, SamplingStateFacts};
pub use inspection::{TextInspectionOptions, inspect_architecture, inspect_text_model};
pub use loaded::{LoadedModelLoadError, PlannedModelLoadError};
pub use media::MultimodalPreparationError;
pub use observed::{
    ObservedGenerationEvent, ObservedGenerationRecord, PreparedObservedGeneration, TraceLimits,
};

pub use portable::{
    GeneratedToken, LoadedModel, LoadedTextModelConfig, LoadedTokenizerView, ManagedChatError,
    ManagedChatPolicyRejection, ManagedChatRequest, ManagedChatSource, ManagedPlainTextError,
    ManagedPlainTextRequest, ManagedPlainTextSession, ManagedPlainTextSnapshot, ManagedPreparedInputRequest, ManagedModelInputError,
    ManagedPlainTextSnapshotError, ManagedPlainTextSource, ManagedPlainTextSpeculativeBatchLane,
    ManagedPlainTextSpeculativeBatchRequest, ManagedPlainTextSpeculativeError,
    ManagedPlainTextSpeculativeRequest, ManagedPreparedChatSpeculativeError,
    ManagedPreparedChatSpeculativeRequest, PlannedModel, TextDecoder, TextDecoderError,
    TextGeneration, TextModelError, TextModelOptions,
};

/// Portable failure reported by prepared-chat constraint state.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{message}")]
pub struct ConstraintError {
    message: std::borrow::Cow<'static, str>,
}

impl ConstraintError {
    pub(crate) const fn fixed(message: &'static str) -> Self {
        Self {
            message: std::borrow::Cow::Borrowed(message),
        }
    }
    pub(crate) fn new(message: impl Into<String>) -> Self {
        Self {
            message: std::borrow::Cow::Owned(message.into()),
        }
    }
}

#[cfg(test)]
mod tests;
