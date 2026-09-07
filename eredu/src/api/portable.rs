//! Backend-neutral loaded-model ownership and text generation.

use eredu_architectures::ModelKind;
use eredu_core::generation::{
    resolve_generation_config, CheckpointGenerationConfig, GenerationConfigOverrides,
    ResolvedGenerationConfig,
};
use eredu_core::{
    ModelRuntime, RealizedDrafting, TextGeneration, TextGenerationBackend, TextGenerationConfig,
};
use eredu_text::tokenizer::{ModelChatTemplate, Tokenizer as ChatTokenizer};

/// Backend-independent failure from tokenizer-aware text facade operations.
#[derive(Debug, thiserror::Error)]
pub enum TextModelError {
    /// Portable generation configuration was invalid.
    #[error(transparent)]
    Generation(#[from] eredu_core::generation::GenerationError),
    /// Chat-template selection, inspection, or rendering failed.
    #[error(transparent)]
    Template(#[from] eredu_text::error::Error),
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
    // Immutable tokenizer configuration is shared between exact decoder forks;
    // only the incremental IDs/prefix below are independently copied.
    pub(crate) tokenizer: std::sync::Arc<tokenizers::Tokenizer>,
    pub(crate) skip_special_tokens: bool,
    pub(crate) ids: Vec<u32>,
    pub(crate) prefix: String,
    pub(crate) prefix_index: usize,
}

impl TextDecoder {
    /// Bounds decoder retention and all text it can hand to an incremental
    /// parser during a finite generated-token run. Decoder chains may expand
    /// strings; price each stage, including empty-pattern replacements. Counting
    /// the maximum full decode once per decision is conservative for incremental
    /// lookbehind and avoids assuming prefix-stable tokenizer cleanup.
    pub(crate) fn continuation_storage_bounds(&self, max_tokens: u64) -> Option<(u64, u64)> {
        use tokenizers::decoders::DecoderWrapper as D;
        fn decoded_bound(decoder: &D, bytes: u64, count: u64) -> Option<u64> {
            match decoder {
                D::Sequence(sequence) => sequence
                    .get_decoders()
                    .iter()
                    .try_fold(bytes, |bytes, decoder| decoded_bound(decoder, bytes, count)),
                D::ByteLevel(_) | D::ByteFallback(_) => bytes.checked_mul(3),
                D::BPE(bpe) if bpe.suffix.is_empty() => bytes.checked_mul(2)?.checked_add(count),
                D::BPE(_) | D::Metaspace(_) | D::Fuse(_) | D::Strip(_) => Some(bytes),
                D::WordPiece(_) => bytes.checked_add(count),
                D::CTC(_) => bytes.checked_mul(2)?.checked_add(count),
                D::Replace(replace) => bytes.checked_add(
                    bytes
                        .checked_add(count)?
                        .checked_mul(replace.content.len() as u64)?,
                ),
            }
        }
        let token_bytes = self
            .tokenizer
            .get_vocab(true)
            .keys()
            .map(String::len)
            .max()? as u64;
        let raw = token_bytes.checked_mul(max_tokens)?;
        let decoded = match self.tokenizer.get_decoder() {
            Some(decoder) => decoded_bound(decoder, raw, max_tokens)?,
            None => raw.checked_add(max_tokens)?,
        };
        let storage = max_tokens.checked_mul(4)?.checked_add(decoded)?;
        // Include tokenizer-confirmed structural spellings in the same bound.
        let emitted = decoded.checked_mul(max_tokens)?.checked_add(raw)?;
        Some((storage, emitted))
    }

    pub(crate) fn snapshot_storage_bytes(&self) -> Option<u64> {
        let Self {
            tokenizer: _,
            skip_special_tokens: _,
            ids,
            prefix,
            prefix_index: _,
        } = self;
        // The immutable tokenizer is shared by Arc; only mutable decoding state
        // is copied, including text not yet emitted as a complete UTF-8 chunk.
        (std::mem::size_of::<Self>() as u64)
            .checked_add((ids.len() as u64).checked_mul(4)?)?
            .checked_add(prefix.len() as u64)
    }
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
    /// Canonical architecture family reported to clients.
    pub model_family: ModelKind,
    /// Parsed implementation or nested text-model type.
    pub effective_model_type: String,
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
    pub(crate) session_identity: String,
    pub(crate) runtime: ModelRuntime<B>,
    pub(crate) tokenizer: ChatTokenizer,
    pub(crate) tokenizer_fingerprint: [u8; 32],
    pub(crate) chat_template: Option<ModelChatTemplate>,
    pub(crate) model_family: ModelKind,
    pub(crate) effective_model_type: String,
    pub(crate) model_id: String,
    pub(crate) eos_token_ids: Vec<u32>,
    pub(crate) checkpoint_generation_config: Option<CheckpointGenerationConfig>,
}

/// One completely realized execution plan: target model plus drafting resources.
///
/// The backend factory owns both target and assistant placement. Callers can
/// execute the same code for any backend without constructing a backend-native
/// assistant, stream, or device.
pub struct PlannedModel<B: TextGenerationBackend, D> {
    model: LoadedModel<B>,
    drafting: RealizedDrafting<D>,
    drafting_plan: eredu_core::DraftingPlan,
}

impl<B: TextGenerationBackend, D> PlannedModel<B, D> {
    pub(crate) fn new(
        model: LoadedModel<B>,
        drafting: RealizedDrafting<D>,
        drafting_plan: eredu_core::DraftingPlan,
    ) -> Self {
        Self {
            model,
            drafting,
            drafting_plan,
        }
    }

    /// Borrows the loaded target model.
    pub const fn model(&self) -> &LoadedModel<B> {
        &self.model
    }

    /// Mutably borrows the loaded target model.
    pub fn model_mut(&mut self) -> &mut LoadedModel<B> {
        &mut self.model
    }

    /// Borrows the backend-owned drafting realization.
    pub const fn drafting(&self) -> &RealizedDrafting<D> {
        &self.drafting
    }

    /// Returns the portable drafting policy that produced these resources.
    pub const fn drafting_plan(&self) -> &eredu_core::DraftingPlan {
        &self.drafting_plan
    }

    /// Resolves the execution plan's speculative request controls.
    ///
    /// This prevents automatic-plan proposal and lookahead limits from being
    /// silently replaced by unrelated request defaults.
    pub fn speculative_generation_options(
        &self,
    ) -> Result<Option<super::PreparedChatSpeculativeGenerationOptions>, eredu_core::GenerationError>
    {
        let (maximum, lookahead, adaptive) = match &self.drafting_plan {
            eredu_core::DraftingPlan::Disabled => return Ok(None),
            eredu_core::DraftingPlan::Embedded {
                max_draft_tokens,
                lookahead,
                adaptive_lookahead,
            }
            | eredu_core::DraftingPlan::External {
                max_draft_tokens,
                lookahead,
                adaptive_lookahead,
                ..
            } => (*max_draft_tokens, *lookahead, *adaptive_lookahead),
            _ => return Ok(None),
        };
        let max_draft_tokens = std::num::NonZeroUsize::new(maximum)
            .ok_or(eredu_core::GenerationError::ZeroDraftTokens)?;
        let mut scheduler =
            eredu_core::SpeculativeSchedulerOptions::default().with_lookahead(lookahead);
        scheduler.adaptive_lookahead = adaptive;
        Ok(Some(super::PreparedChatSpeculativeGenerationOptions {
            max_draft_tokens,
            scheduler: scheduler.validate()?,
        }))
    }

    /// Mutably borrows the target and drafting resources as one planned session.
    pub fn parts_mut(&mut self) -> (&mut LoadedModel<B>, &mut RealizedDrafting<D>) {
        (&mut self.model, &mut self.drafting)
    }

    /// Consumes the planned session into its target model and drafting resources.
    pub fn into_parts(self) -> (LoadedModel<B>, RealizedDrafting<D>) {
        (self.model, self.drafting)
    }
}

impl<B: TextGenerationBackend> LoadedModel<B> {
    /// Combines any prepared backend runtime with portable tokenizer metadata.
    pub fn from_runtime(
        runtime: ModelRuntime<B>,
        tokenizer: ChatTokenizer,
        config: LoadedTextModelConfig,
    ) -> Self {
        let tokenizer_fingerprint = eredu_text::tokenizer::vocabulary_fingerprint(&tokenizer);
        Self {
            session_identity: super::observed::new_identity("session"),
            runtime,
            tokenizer,
            tokenizer_fingerprint,
            chat_template: config.chat_template,
            model_family: config.model_family,
            effective_model_type: config.effective_model_type,
            model_id: config.model_id,
            eos_token_ids: config.eos_token_ids,
            checkpoint_generation_config: config.checkpoint_generation_config,
        }
    }

    pub(crate) const fn runtime(&self) -> &ModelRuntime<B> {
        &self.runtime
    }

    /// Returns portable metadata for the backend that owns this model.
    pub fn backend_descriptor(&self) -> eredu_core::BackendDescriptor {
        self.runtime.backend().descriptor()
    }

    /// Returns the canonical architecture family.
    pub const fn model_family(&self) -> ModelKind {
        self.model_family
    }

    /// Returns the parsed implementation or nested text-model type.
    pub fn effective_model_type(&self) -> &str {
        &self.effective_model_type
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
            tokenizer: std::sync::Arc::new((*self.tokenizer).clone()),
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
    ) -> Result<ResolvedGenerationConfig, eredu_core::generation::GenerationError> {
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
