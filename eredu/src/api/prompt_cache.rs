//! Automatic reuse and publication for local plain-text generation.
use super::{LoadedModel, PreparedChatError};
use eredu_core::{
    cache::{
        prompt_cache_token_fingerprint, PromptCacheDescriptor, PromptCacheModelIdentity,
        PromptCacheOptions, PromptCacheRead, PromptCacheStore, PromptCacheStoreError,
    },
    AutomaticPromptCacheSession, BackendFailure, GenerationCancellationToken, ModelRuntime,
    PrefillChunkPolicy, TextExecutionSession, TextGenerationBackend, TextGenerationConfig,
};
use std::time::{Duration, Instant};

/// Metadata for generation using a configured prompt-cache store.
#[derive(Debug, Clone, Default)]
pub struct PromptCacheStats {
    /// Complete input extent before cache reuse.
    pub input_tokens: usize,
    /// Exact prefix restored without replay.
    pub reused_tokens: usize,
    /// Time spent looking up, restoring, and publishing cache state.
    pub prefill_time: Duration,
    /// Store bytes retained after publication.
    pub retained_bytes: u64,
    /// Estimated export cost before the store admits the exact staging reservation.
    pub capture_estimate_bytes: Option<u64>,
}
/// Retains the exact operations proved when the application attaches a store.
pub(super) struct AttachedPromptCache<B: TextGenerationBackend, S: TextExecutionSession<B>> {
    checkpoint: String,
    pub(super) store: Box<dyn PromptCacheStore>,
    identity: fn(&ModelRuntime<B, S>) -> Result<PromptCacheModelIdentity, BackendFailure>,
    fingerprint: fn(&ModelRuntime<B, S>, &[u32]) -> Result<String, BackendFailure>,
    restore: fn(
        &mut ModelRuntime<B, S>,
        &dyn PromptCacheRead,
        &PromptCacheDescriptor,
        &[u32],
    ) -> Result<eredu_core::cache::PromptCacheManifest, BackendFailure>,
    estimate: fn(&ModelRuntime<B, S>) -> Result<Option<u64>, BackendFailure>,
    save: fn(
        &mut ModelRuntime<B, S>,
        &dyn PromptCacheStore,
        PromptCacheDescriptor,
        &[u32],
        &PromptCacheOptions,
    ) -> Result<eredu_core::cache::PromptCacheManifest, BackendFailure>,
}

pub(super) struct PromptCacheRun<'a, B: TextGenerationBackend, S: TextExecutionSession<B>> {
    attached: &'a AttachedPromptCache<B, S>,
    pub(super) stats: PromptCacheStats,
    publication: Option<(u64, Vec<u32>)>,
}

impl<B: TextGenerationBackend, S: TextExecutionSession<B>> PromptCacheRun<'_, B, S> {
    pub(super) fn publication_due(&self, step: u64, cancel: &GenerationCancellationToken) -> bool {
        !cancel.is_cancelled()
            && self
                .publication
                .as_ref()
                .is_some_and(|(expected, _)| step == *expected)
    }

    pub(super) fn publish(
        &mut self,
        runtime: &mut ModelRuntime<B, S>,
        cancel: &GenerationCancellationToken,
    ) -> Result<(), PreparedChatError> {
        if cancel.is_cancelled() {
            return Ok(());
        }
        let Some((_, prefix)) = self.publication.take() else {
            return Ok(());
        };
        let started = Instant::now();
        self.stats.capture_estimate_bytes = (self.attached.estimate)(runtime)
            .map_err(PreparedChatError::Backend)?
            .and_then(eredu_runtime::prompt_cache_export_allowance);
        let identity = (self.attached.identity)(runtime).map_err(PreparedChatError::Backend)?;
        let descriptor = self.attached.descriptor(runtime, identity, &prefix)?;
        let options = PromptCacheOptions::new(None, true)
            .map_err(backend)?
            .with_cancellation(cancel.clone());
        if !cancel.is_cancelled() {
            match (self.attached.save)(
                runtime,
                &*self.attached.store,
                descriptor,
                &prefix,
                &options,
            ) {
                Ok(_) => {}
                Err(error) if PromptCacheStoreError::publication_declined(&error) => {}
                Err(error) => return Err(PreparedChatError::Backend(error)),
            }
        }
        self.stats.prefill_time += started.elapsed();
        self.stats.retained_bytes = self.attached.store.retained_bytes();
        Ok(())
    }
}

impl<B: TextGenerationBackend, S: TextExecutionSession<B>> AttachedPromptCache<B, S> {
    fn eligible(runtime: &ModelRuntime<B, S>, config: &TextGenerationConfig) -> bool {
        config.prefill_chunk_policy() != PrefillChunkPolicy::Unchunked
            && matches!(
                config
                    .decoding_mode()
                    .unwrap_or_else(|| runtime.session().default_decoding_mode()),
                eredu_core::decoding::DecodingMode::Autoregressive
            )
            && config.sampling().max_new_tokens != Some(0)
            && config.refinement_canvas().is_none()
            && runtime.session().active_parameter_overlay().is_none()
            && runtime.session().text_prefill_chunking_support().is_ok()
            && !runtime.session().supports_block_refinement()
    }

    fn descriptor(
        &self,
        runtime: &ModelRuntime<B, S>,
        identity: PromptCacheModelIdentity,
        prefix: &[u32],
    ) -> Result<PromptCacheDescriptor, PreparedChatError> {
        PromptCacheDescriptor::from_model_identity(
            identity,
            &self.checkpoint,
            (self.fingerprint)(runtime, prefix).map_err(PreparedChatError::Backend)?,
            1,
        )
        .map_err(backend)
    }

    fn restore_prefix(
        &self,
        runtime: &mut ModelRuntime<B, S>,
        identity: PromptCacheModelIdentity,
        tokens: &[u32],
    ) -> Result<usize, PreparedChatError> {
        let Some(read) = self.store.open()? else {
            return Ok(0);
        };
        let manifest = read.manifest();
        let n = manifest.total_prefix_tokens;
        if n == 0
            || n >= tokens.len()
            || manifest.prefix_sha256 != prompt_cache_token_fingerprint(&tokens[..n])
        {
            return Ok(0);
        }
        let descriptor = self.descriptor(runtime, identity, &tokens[..n])?;
        if manifest
            .validate_compatibility(&descriptor, &tokens[..n])
            .is_err()
        {
            return Ok(0);
        }
        (self.restore)(runtime, &*read, &descriptor, &tokens[..n])
            .map_err(PreparedChatError::Backend)?;
        Ok(n)
    }

    pub(super) fn prepare(
        &self,
        runtime: &mut ModelRuntime<B, S>,
        config: &mut TextGenerationConfig,
        tokens: &[u32],
        cancel: &GenerationCancellationToken,
    ) -> Result<Option<PromptCacheRun<'_, B, S>>, PreparedChatError> {
        if !Self::eligible(runtime, config) {
            return Ok(None);
        }
        if cancel.is_cancelled() {
            return Ok(None);
        }
        let identity = (self.identity)(runtime).map_err(PreparedChatError::Backend)?;
        let topology = identity.topology();
        if [topology.stage(), topology.shard(), topology.addressable()]
            .into_iter()
            .flatten()
            .any(|(size, _)| size > 1)
        {
            return Ok(None);
        }
        let started = Instant::now();
        runtime.reset().map_err(PreparedChatError::Backend)?;
        let reused = self.restore_prefix(runtime, identity, tokens)?;
        if let Some(reserve) = config.prefill_chunk_policy().automatic_reserve_bytes() {
            let limit = S::automatic_prefill_chunk(runtime, tokens.len() - reused, reused, reserve);
            *config = config
                .clone()
                .with_prefill_chunk_policy(PrefillChunkPolicy::Bounded(limit));
        }
        let chunk = config.prefill_chunk_policy().unresolved_limit().get();
        let chunks = tokens.len().saturating_sub(1) / chunk;
        Ok(Some(PromptCacheRun {
            attached: self,
            publication: (reused == 0 && chunks != 0)
                .then(|| (chunks as u64, tokens[..chunks * chunk].to_vec())),
            stats: PromptCacheStats {
                input_tokens: tokens.len(),
                reused_tokens: reused,
                prefill_time: started.elapsed(),
                retained_bytes: self.store.retained_bytes(),
                capture_estimate_bytes: None,
            },
        }))
    }
}

fn backend(error: impl std::error::Error + Send + Sync + 'static) -> PreparedChatError {
    PreparedChatError::Backend(eredu_core::BackendFailure::from_error(error))
}
impl<B: TextGenerationBackend, S: TextExecutionSession<B> + AutomaticPromptCacheSession<B>>
    LoadedModel<B, S>
{
    /// Uses the same prompt-cache format and import/export path with any store.
    /// The fingerprint must identify the exact admitted checkpoint. It is also
    /// used for directory stores across loads; a model path alone is insufficient
    /// if its contents can change. Supply the full prompt to generation.
    /// Automatic reuse applies to local plain text with
    /// bounded autoregressive prefill and no activation capture or edits.
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
        S::prompt_cache_model_identity(&self.runtime).map_err(PreparedChatError::Backend)?;
        self.prompt_cache_store = Some(AttachedPromptCache {
            checkpoint,
            store,
            identity: S::prompt_cache_model_identity,
            fingerprint: S::prompt_cache_prefix_fingerprint,
            restore: S::restore_prompt_cache,
            estimate: S::estimate_prompt_cache_state,
            save: S::save_prompt_cache,
        });
        Ok(())
    }
    /// Configured store, including its explicit capture allowance and retained bytes.
    pub fn prompt_cache_store(&self) -> Option<&dyn PromptCacheStore> {
        self.prompt_cache_store.as_ref().map(|cache| &*cache.store)
    }
    /// Disables automatic reuse and releases this model's store owner.
    pub fn remove_prompt_cache_store(&mut self) {
        self.prompt_cache_store = None;
    }
}
