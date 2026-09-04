//! Backend-neutral language-model facade.
//!
//! This module is available without an execution backend. Enabling the
//! `mlx` feature adds the selected local execution adapter. The `metal`,
//! `cuda`, `image`, and `audio` features only configure that adapter when it is
//! selected.
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
use crate::runtime::chat::PreparedChat;
#[cfg(feature = "mlx")]
use eredu_architectures::ModelKind;
#[cfg(feature = "mlx")]
use eredu_text::tokenizer::Tokenizer as ChatTokenizer;

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
mod inspection;
mod loaded;
pub use inspection::{inspect_text_model, TextInspectionOptions};
pub use loaded::{LoadedModelLoadError, PlannedModelLoadError};
pub use media::MultimodalPreparationError;

pub use portable::{
    LoadedModel, LoadedTextModelConfig, PlannedModel, TextDecoder, TextDecoderError, TextModelError,
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
