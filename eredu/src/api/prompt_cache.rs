//! Automatic input reuse through the ordinary prompt-cache store operations.
use super::{
    GenerationOutput, GenerationRequest, LoadedModel, PreparedChatError, PreparedChatInput,
};
use eredu_core::{
    cache::{
        prompt_cache_token_fingerprint, PromptCacheDescriptor, PromptCacheOptions, PromptCacheStore,
    },
    FinishReason, GenerationCancellationToken, PrefillChunkPolicy, SemanticEvent,
    TextGenerationBackend, TextGenerationConfig, TextGenerationDriver, TokenFilter,
    TokenFilterController,
};
use std::{
    convert::Infallible,
    time::{Duration, Instant},
};

/// Metadata for ordinary generation using a configured prompt-cache store.
#[derive(Debug, Clone, Default)]
pub struct PromptCacheStats {
    /// Complete input extent before cache reuse.
    pub input_tokens: usize,
    /// Exact prefix restored without replay.
    pub reused_tokens: usize,
    /// Restore, completed prefix prefill and publication time.
    pub prefill_time: Duration,
    /// Store bytes retained after publication.
    pub retained_bytes: u64,
    /// Conservative export allowance, including staging and metadata.
    pub capture_estimate_bytes: Option<u64>,
}
struct PrefillOnly;
impl TokenFilterController for PrefillOnly {
    type Error = Infallible;
    fn current_filter(&mut self) -> Result<TokenFilter, Infallible> {
        Ok(TokenFilter::All)
    }
    fn commit_token(&mut self, _: u32) -> Result<(), Infallible> {
        Ok(())
    }
    fn is_complete(&mut self) -> Result<bool, Infallible> {
        Ok(false)
    }
}
fn backend(error: impl std::error::Error + Send + Sync + 'static) -> PreparedChatError {
    PreparedChatError::Backend(eredu_core::BackendFailure::from_error(error))
}
impl<B: TextGenerationBackend> LoadedModel<B> {
    /// Uses the same prompt-cache format and import/export path with any store.
    /// The fingerprint must identify the exact admitted checkpoint. It is also
    /// used for directory stores across loads; a model path alone is insufficient
    /// if its contents can change. Full inputs are still supplied to `generate`.
    pub fn set_prompt_cache_store(
        &mut self,
        checkpoint_fingerprint: impl Into<String>,
        store: Box<dyn PromptCacheStore>,
    ) -> Result<(), PreparedChatError> {
        let checkpoint = checkpoint_fingerprint.into();
        if checkpoint.trim().is_empty() {
            return Err(eredu_core::cache::PromptCacheStoreError::Cache(
                eredu_core::cache::PromptCacheError::Malformed(
                    "empty checkpoint fingerprint".into(),
                ),
            )
            .into());
        }
        B::prompt_cache_model_identity(&self.runtime).map_err(PreparedChatError::Backend)?;
        self.prompt_cache_store = Some((checkpoint, store));
        Ok(())
    }
    /// Configured store, including its explicit capture allowance and retained bytes.
    pub fn prompt_cache_store(&self) -> Option<&dyn PromptCacheStore> {
        self.prompt_cache_store.as_ref().map(|(_, s)| &**s)
    }
    /// Disables automatic reuse and releases this model's store owner.
    pub fn remove_prompt_cache_store(&mut self) {
        self.prompt_cache_store = None;
    }

    pub(super) fn generate_with_store<F: FnMut(SemanticEvent)>(
        &mut self,
        mut request: GenerationRequest<'_, B, F>,
        started: Instant,
    ) -> Result<GenerationOutput, PreparedChatError> {
        let (checkpoint, store) = self
            .prompt_cache_store
            .take()
            .expect("configured prompt-cache store");
        let cancel = request.cancellation.clone();
        let result = (|| {
            self.reset().map_err(PreparedChatError::Backend)?;
            let chat = request.input.prepared_chat();
            request
                .input
                .validate_session(&self.session_identity)
                .map_err(PreparedChatError::Backend)?;
            let original =
                std::mem::replace(&mut request.input, PreparedChatInput::rendered_prompt(chat));
            let tokens = match original {
                PreparedChatInput::TokenIds { token_ids, .. } => Some(token_ids),
                PreparedChatInput::RenderedPrompt(_) => {
                    Some(self.encode(chat.rendered_prompt(), false)?)
                }
                other => {
                    request.input = other;
                    None
                }
            };
            let mut stats = PromptCacheStats::default();
            if let Some(tokens) = tokens {
                stats.input_tokens = tokens.len();
                let (config, _) = self.resolve_text_generation_settings(request.settings)?;
                let suffix = if request.control_handle.is_none() {
                    self.prefill_with_store(
                        &tokens,
                        config,
                        &cancel,
                        &*store,
                        &checkpoint,
                        &mut stats,
                    )?
                } else {
                    store.clear()?;
                    tokens
                };
                request.input = PreparedChatInput::token_ids(chat, suffix);
            } else {
                store.clear()?;
            }
            let mode = request.output_mode;
            let output = self.generate_prepared(request, None, started, mode, None)?;
            Ok(output.map_stats(|mut s| {
                s.prompt_cache = Some(stats);
                s
            }))
        })();
        let cancelled = cancel.is_cancelled()
            || result
                .as_ref()
                .is_ok_and(|o: &GenerationOutput| o.finish_reason() == FinishReason::Cancelled);
        let cleanup = self.reset().map_err(PreparedChatError::Backend);
        let eviction = if result.is_err() || cancelled || cleanup.is_err() {
            store.clear().map_err(PreparedChatError::from)
        } else {
            Ok(())
        };
        self.prompt_cache_store = Some((checkpoint, store));
        cleanup?;
        eviction?;
        result
    }
    fn prefill_with_store(
        &mut self,
        tokens: &[u32],
        config: TextGenerationConfig,
        cancel: &GenerationCancellationToken,
        store: &dyn PromptCacheStore,
        checkpoint: &str,
        stats: &mut PromptCacheStats,
    ) -> Result<Vec<u32>, PreparedChatError> {
        let now = Instant::now();
        if cancel.is_cancelled()
            || !matches!(
                config.prefill_chunk_policy(),
                PrefillChunkPolicy::Bounded(_)
            )
            || !matches!(
                config
                    .decoding_mode()
                    .unwrap_or_else(|| B::default_decoding_mode(&self.runtime)),
                eredu_core::decoding::DecodingMode::Autoregressive
            )
            || B::text_prefill_chunking_support(&self.runtime).is_err()
            || B::supports_block_refinement(&self.runtime)
        {
            store.clear()?;
            return Ok(tokens.to_vec());
        }
        let identity = match B::prompt_cache_model_identity(&self.runtime) {
            Ok(i) => i,
            Err(e) => return Err(PreparedChatError::Backend(e)),
        };
        let mut reused = 0;
        if let Some(read) = store.open()? {
            let manifest = read.manifest();
            let n = manifest.total_prefix_tokens;
            if n > 0
                && n < tokens.len()
                && manifest.prefix_sha256 == prompt_cache_token_fingerprint(&tokens[..n])
            {
                let descriptor = PromptCacheDescriptor::from_model_identity(
                    identity.clone(),
                    checkpoint,
                    B::prompt_cache_prefix_fingerprint(&self.runtime, &tokens[..n])
                        .map_err(PreparedChatError::Backend)?,
                    1,
                )
                .map_err(|e| backend(e))?;
                if manifest
                    .validate_compatibility(&descriptor, &tokens[..n])
                    .is_ok()
                {
                    B::load_prompt_cache(&mut self.runtime, store, &descriptor, &tokens[..n])
                        .map_err(PreparedChatError::Backend)?;
                    reused = n;
                }
            }
        }
        // Keep the original exact prefix on hits. A suffix prepared in isolation
        // cannot attest the provenance of an extended composite input.
        if reused > 0 {
            stats.reused_tokens = reused;
            stats.prefill_time = now.elapsed();
            stats.retained_bytes = store.retained_bytes();
            return Ok(tokens[reused..].to_vec());
        }
        store.clear()?;
        let PrefillChunkPolicy::Bounded(chunk) = config.prefill_chunk_policy() else {
            unreachable!()
        };
        let consumed = tokens.len().saturating_sub(1) / chunk.get() * chunk.get();
        if consumed == 0 {
            return Ok(tokens.to_vec());
        }
        let prompt = B::prepare_text_prompt(self.runtime.backend(), tokens[..consumed].to_vec())
            .map_err(backend)?;
        let mut driver = TextGenerationDriver::new(&mut self.runtime);
        let mut state = driver.start(prompt, config, PrefillOnly).map_err(backend)?;
        state.set_cancellation_token(cancel.clone());
        loop {
            if cancel.is_cancelled() {
                break;
            }
            let advanced = driver.advance_prefill(&mut state);
            let drained = driver.take_completed_step(&mut state);
            let advanced = advanced.map_err(backend)?;
            drained.map_err(backend)?;
            if !advanced {
                break;
            }
        }
        if cancel.is_cancelled() {
            drop(state);
            drop(driver);
            return Ok(tokens[consumed..].to_vec());
        }
        let completed = driver.complete_prefill(state).map_err(backend)?;
        drop(driver);
        if !completed {
            self.reset().map_err(PreparedChatError::Backend)?;
            return Ok(tokens.to_vec());
        }
        if consumed > 0 && !cancel.is_cancelled() {
            if let Some(bytes) = B::estimate_prompt_cache_state(&self.runtime)
                .map_err(PreparedChatError::Backend)?
                .and_then(eredu_runtime::prompt_cache_export_allowance)
            {
                stats.capture_estimate_bytes = Some(bytes);
                if bytes <= store.max_bytes().saturating_sub(store.retained_bytes()) {
                    let prefix = &tokens[..consumed];
                    let descriptor = PromptCacheDescriptor::from_model_identity(
                        identity,
                        checkpoint,
                        B::prompt_cache_prefix_fingerprint(&self.runtime, prefix)
                            .map_err(PreparedChatError::Backend)?,
                        1,
                    )
                    .map_err(backend)?;
                    let options = PromptCacheOptions::new(None, true).map_err(backend)?;
                    B::save_prompt_cache(&mut self.runtime, store, descriptor, prefix, &options)
                        .map_err(PreparedChatError::Backend)?;
                }
            }
        }
        stats.reused_tokens = reused;
        stats.prefill_time = now.elapsed();
        stats.retained_bytes = store.retained_bytes();
        Ok(tokens[consumed..].to_vec())
    }
}
