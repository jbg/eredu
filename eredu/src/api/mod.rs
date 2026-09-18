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
pub(crate) use request::policy::CompiledChatPolicy;
mod tokenizer;

#[cfg(feature = "mlx")]
mod selected;

pub use request::{
    PreparedChatGenerationSettings, PreparedChatSpeculativeConstraint,
    PreparedChatSpeculativeGenerationOptions,
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
    ControlledGenerationBranch, ControlledGenerationError, ControlledGenerationEvent,
    ControlledGenerationRecord, ControlledGenerationSession, ControlledGenerationSnapshot,
    ControlledRecordData, ControlledWireRecord,
    GenerationBranchMetadata, GenerationBranchOptions, GenerationOutputCheckpoint, GenerationOutputCheckpointData,
    GenerationSnapshotData, GenerationSnapshotMetadata, PREPARED_EXECUTION_CONTROL_SCHEMA_VERSION,
    PreparedInstrumentationRecord, ControlledSessionFailure,
    RecordConstructionCause, RecordConstructionError,
};
pub use eredu_core::{
    GenerationOutput, GenerationPlainTextEvent, GenerationPlainTextOutput, GenerationTiming,
    TextInferencePolicy, TextSamplingStrategy,
};
pub use eredu_runtime::execution_control::{SamplingOverride, SamplingStateFacts, TokenChoiceError};
pub use inspection::{TextInspectionOptions, inspect_architecture, inspect_text_model};
pub use loaded::{LoadedModelLoadError, PlannedModelLoadError};
pub use media::MultimodalPreparationError;
pub use observed::{
    ObservedGenerationEvent, TraceLimits,
};

pub use portable::{
    PreparedChatOutputMode, PreparedChatPrompt, PreparedChatRequest, PreparedChatSession, PreparedChatSessionError, PreparedChatResumeSettings, PreparedChatSnapshot, PreparedChatBranch,
    GeneratedToken, LoadedModel, LoadedTextModelConfig, LoadedTokenizerView, ManagedChatError,
    ChatSourceInput, ManagedChatSource, ManagedChatSourceError, ManagedPlainTextError, ManagedPlainTextSourceError,
    ManagedPlainTextRequest, ManagedPlainTextSession, ManagedPlainTextSnapshot, ManagedPreparedInputRequest, ManagedModelInputError,
    GenerationSnapshotError, ManagedPlainTextSource, TokenizerSourceInput, ManagedPlainTextSpeculativeBatchLane,
    ManagedPlainTextSpeculativeBatchRequest, ManagedPlainTextSpeculativeError,
    ManagedPlainTextSpeculativeRequest, PreparedChatSpeculativeBatchLane, PreparedChatSpeculativeBatchRequest, PreparedChatSpeculativeError,
    PreparedChatSpeculativeRequest, PlannedModel, TextDecoder, TextDecoderError,
    TextGeneration, TextModelError, TextModelOptions,
};

/// Portable failure reported by prepared-chat constraint state.
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
pub struct ConstraintError {
    #[source]
    cause: ConstraintErrorCause,
}

#[derive(Debug, thiserror::Error)]
enum ConstraintErrorCause {
    #[error("{0}")]
    Message(std::borrow::Cow<'static, str>),
    #[error(transparent)]
    Funding(eredu_core::HostMetadataFundingError),
    #[error(transparent)]
    Forbidden(eredu_core::speculative::ForbiddenControllerError),
    #[error(transparent)]
    Plain(eredu_core::speculative::PlainControllerError),
    #[error(transparent)]
    Filter(eredu_core::TokenFilterError),
    #[error(transparent)]
    Prepared(crate::runtime::chat::constraints::prepared::Failure),
    #[error(transparent)]
    Preparation(crate::runtime::chat::constraints::PreparationFailure),
}

impl ConstraintError {
    pub(crate) fn preparation(cause: crate::runtime::chat::constraints::PreparationFailure) -> Self {
        Self { cause: ConstraintErrorCause::Preparation(cause) }
    }
    pub(crate) const fn fixed(message: &'static str) -> Self {
        Self {
            cause: ConstraintErrorCause::Message(std::borrow::Cow::Borrowed(message)),
        }
    }
    pub(crate) fn new(message: impl Into<String>) -> Self {
        Self {
            cause: ConstraintErrorCause::Message(std::borrow::Cow::Owned(message.into())),
        }
    }
    pub(crate) fn funding(cause: eredu_core::HostMetadataFundingError) -> Self {
        Self { cause: ConstraintErrorCause::Funding(cause) }
    }
    pub(crate) fn prepared(cause: crate::runtime::chat::constraints::prepared::Failure) -> Self {
        Self { cause: ConstraintErrorCause::Prepared(cause) }
    }
    pub(crate) fn forbidden(cause: eredu_core::speculative::ForbiddenControllerError) -> Self {
        Self { cause: ConstraintErrorCause::Forbidden(cause) }
    }
    pub(crate) fn plain(cause: eredu_core::speculative::PlainControllerError) -> Self {
        Self { cause: ConstraintErrorCause::Plain(cause) }
    }
    pub(crate) fn filter(cause: eredu_core::TokenFilterError) -> Self {
        Self { cause: ConstraintErrorCause::Filter(cause) }
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
pub(crate) use tokenizer::tokenizer_token_filter as tokenizer_token_filter_for_tests;
