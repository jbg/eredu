//! Backend-neutral loaded-model ownership and text generation.

use safemlx_lm_core::generation::{
    resolve_generation_config, CheckpointGenerationConfig, GenerationConfigOverrides,
    ResolvedGenerationConfig,
};
use safemlx_lm_core::{ModelRuntime, TextGeneration, TextGenerationBackend, TextGenerationConfig};
use safemlx_lm_utils::tokenizer::{ModelChatTemplate, Tokenizer as ChatTokenizer};

/// Backend-independent failure from tokenizer-aware text facade operations.
#[derive(Debug, thiserror::Error)]
pub enum TextModelError {
    /// Portable generation configuration was invalid.
    #[error(transparent)]
    Generation(#[from] safemlx_lm_core::generation::GenerationError),
    /// Chat-template selection, inspection, or rendering failed.
    #[error(transparent)]
    Template(#[from] safemlx_lm_utils::error::Error),
    /// Tokenizer encoding or decoding failed.
    #[error(transparent)]
    Tokenizer(#[from] Box<dyn std::error::Error + Send + Sync>),
    /// The loaded checkpoint does not provide a chat template.
    #[error("the loaded model does not provide a chat template")]
    MissingChatTemplate,
    /// A native tool definition or its generation grammar is invalid.
    #[error("native tool constraint error: {0}")]
    ToolConstraint(String),
}

/// Failure while incrementally decoding tokenizer output.
#[derive(Debug, thiserror::Error)]
pub enum TextDecoderError {
    /// The checkpoint tokenizer rejected the token stream.
    #[error(transparent)]
    Tokenizer(#[from] Box<dyn std::error::Error + Send + Sync>),
    /// Decoding ended with a partial byte-fallback sequence.
    #[error("generated token stream ended with an incomplete tokenizer byte sequence")]
    IncompleteByteSequence,
}

/// Stateful tokenizer decoder for incrementally generated token ids.
#[derive(Clone)]
pub struct TextDecoder {
    pub(crate) tokenizer: tokenizers::Tokenizer,
    pub(crate) skip_special_tokens: bool,
    pub(crate) ids: Vec<u32>,
    pub(crate) prefix: String,
    pub(crate) prefix_index: usize,
}

impl TextDecoder {
    /// Decodes one token, returning text only when the token completes a chunk.
    pub fn step(&mut self, id: u32) -> Result<Option<String>, TextDecoderError> {
        tokenizers::tokenizer::step_decode_stream(
            &self.tokenizer,
            vec![id],
            self.skip_special_tokens,
            &mut self.ids,
            &mut self.prefix,
            &mut self.prefix_index,
        )
        .map_err(TextDecoderError::Tokenizer)
    }
}

/// Backend-neutral tokenizer and chat metadata attached to a prepared runtime.
pub struct LoadedTextModelConfig {
    /// Normalized architecture identity reported to clients.
    pub model_type: String,
    /// Model identity supplied to chat-template rendering.
    pub model_id: String,
    /// Optional checkpoint chat template or named-template collection.
    pub chat_template: Option<ModelChatTemplate>,
    /// Checkpoint EOS vocabulary ids.
    pub eos_token_ids: Vec<u32>,
    /// Optional checkpoint sampling recommendations.
    pub checkpoint_generation_config: Option<CheckpointGenerationConfig>,
}

/// A prepared backend model together with its portable text metadata.
///
/// The type contains no backend tensor, device, stream, or completion type.
/// Backend-specific state remains owned by [`ModelRuntime`].
pub struct LoadedModel<B: TextGenerationBackend> {
    pub(crate) runtime: ModelRuntime<B>,
    pub(crate) tokenizer: ChatTokenizer,
    pub(crate) tokenizer_fingerprint: [u8; 32],
    pub(crate) chat_template: Option<ModelChatTemplate>,
    pub(crate) model_type: String,
    pub(crate) model_id: String,
    pub(crate) eos_token_ids: Vec<u32>,
    pub(crate) checkpoint_generation_config: Option<CheckpointGenerationConfig>,
}

impl<B: TextGenerationBackend> LoadedModel<B> {
    /// Combines any prepared backend runtime with portable tokenizer metadata.
    pub fn from_runtime(
        runtime: ModelRuntime<B>,
        tokenizer: ChatTokenizer,
        config: LoadedTextModelConfig,
    ) -> Self {
        let tokenizer_fingerprint = safemlx_lm_utils::tokenizer::vocabulary_fingerprint(&tokenizer);
        Self {
            runtime,
            tokenizer,
            tokenizer_fingerprint,
            chat_template: config.chat_template,
            model_type: config.model_type,
            model_id: config.model_id,
            eos_token_ids: config.eos_token_ids,
            checkpoint_generation_config: config.checkpoint_generation_config,
        }
    }

    /// Borrows the selected backend and its session owner.
    pub const fn runtime(&self) -> &ModelRuntime<B> {
        &self.runtime
    }

    /// Mutably borrows the selected backend and its session owner.
    pub fn runtime_mut(&mut self) -> &mut ModelRuntime<B> {
        &mut self.runtime
    }

    /// Returns the effective runtime model type.
    pub fn model_type(&self) -> &str {
        &self.model_type
    }

    /// Borrows the tokenizer attached to the prepared model.
    pub const fn tokenizer(&self) -> &ChatTokenizer {
        &self.tokenizer
    }

    /// Returns the stable token-id vocabulary fingerprint.
    pub const fn tokenizer_fingerprint(&self) -> &[u8; 32] {
        &self.tokenizer_fingerprint
    }

    /// Encodes text to tokenizer ids.
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>, TextModelError> {
        Ok(self
            .tokenizer
            .encode(text, add_special_tokens)?
            .get_ids()
            .to_vec())
    }

    /// Decodes tokenizer ids back to text.
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String, TextModelError> {
        self.tokenizer
            .decode(ids, skip_special_tokens)
            .map_err(TextModelError::Tokenizer)
    }

    /// Creates an independent stateful decoder for streaming generated tokens.
    pub fn text_decoder(&self, skip_special_tokens: bool) -> TextDecoder {
        TextDecoder {
            tokenizer: (*self.tokenizer).clone(),
            skip_special_tokens,
            ids: Vec::new(),
            prefix: String::new(),
            prefix_index: 0,
        }
    }

    /// Returns sampling values declared by `generation_config.json`, if present.
    pub fn checkpoint_generation_config(&self) -> Option<&CheckpointGenerationConfig> {
        self.checkpoint_generation_config.as_ref()
    }

    /// Resolves request overrides over checkpoint recommendations and fallbacks.
    pub fn resolve_generation_config(
        &self,
        overrides: GenerationConfigOverrides,
    ) -> Result<ResolvedGenerationConfig, safemlx_lm_core::generation::GenerationError> {
        resolve_generation_config(self.checkpoint_generation_config.as_ref(), overrides)
    }

    /// Starts asynchronous text generation from tokenizer ids.
    pub fn generate_tokens(
        &mut self,
        prompt_token_ids: Vec<u32>,
        config: TextGenerationConfig,
    ) -> Result<TextGeneration<'_, B>, B::Error> {
        TextGeneration::new(&mut self.runtime, prompt_token_ids, config)
    }

    /// Returns the model id passed to chat-template rendering.
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Returns whether a chat template is attached to the model.
    pub fn has_chat_template(&self) -> bool {
        self.chat_template.is_some()
    }

    /// Returns the configured EOS token ids.
    pub fn eos_token_ids(&self) -> &[u32] {
        &self.eos_token_ids
    }

    /// Returns true when `id` is a configured EOS token id.
    pub fn is_eos_token(&self, id: u32) -> bool {
        self.eos_token_ids.contains(&id)
    }
}
