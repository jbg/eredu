//! Architecture-erased MLX model-session execution.

use eredu_core::{
    BackendSession, Completion, ModelRuntime, Submission, TextGenerationBackend,
    TextGenerationConfig, TokenFilter, TokenOutput,
};
use eredu_runtime::CausalModel;
use safemlx::{
    error::Exception,
    ops::indexing::{NewAxis, TryIndexOp},
    random::RandomState,
    Array, Stream,
};
use std::path::Path;

#[cfg(feature = "mlx-media")]
use crate::backend::mlx::runtime::media::{ModelProcessor, PreparedModelInput};
use crate::core::generation::MtpConfig;
use crate::{
    backend::mlx::error::Error,
    backend::mlx::runtime::execution::inspection::ActivationObserver,
    backend::mlx::runtime::generation::sampler::{DefaultSampler, Sampler, SpeculativeSampler},
    backend::mlx::runtime::media::input,
    composition::mlx_architectures::distributed::{
        expert::ExpertParallelCache,
        pipeline::{PipelineCache, PipelineStageCompletion, PipelineStep},
    },
    PromptCacheDescriptor, PromptCacheManifest, PromptCacheOptions,
};
use eredu_core::MtpStats;
use eredu_runtime::{CacheResidencyPolicy, PagedCacheOptions};

use super::{
    MlxBackend, MlxCompletion, MlxDistributedSession, MlxModel, MlxModelKind, Model, ModelCache,
};

/// Backend-owned output of one MLX model-session submission.
///
/// Pipeline ranks that do not own the output projection complete with no local
/// logits. The final rank and every non-pipeline session complete with logits.
#[derive(Debug, Clone)]
pub struct MlxModelOutput {
    logits: Option<Array>,
}

impl MlxModelOutput {
    const fn new(logits: Option<Array>) -> Self {
        Self { logits }
    }

    /// Borrows local logits when this rank owns them.
    pub const fn logits(&self) -> Option<&Array> {
        self.logits.as_ref()
    }

    /// Consumes the output and returns local logits when present.
    pub fn into_logits(self) -> Option<Array> {
        self.logits
    }
}

/// Opaque exact completion for any MLX model-session submission.
pub struct MlxSessionCompletion {
    inner: MlxSessionCompletionKind,
}

/// MLX token handle yielded by backend-generic text generation.
#[derive(Clone)]
pub struct MlxTextToken {
    value: Array,
    stream: Stream,
}

impl TokenOutput for MlxTextToken {
    type Error = Error;

    fn token_id(&self) -> Result<u32, Self::Error> {
        self.value
            .clone()
            .try_item::<u32>(&self.stream)
            .map_err(Into::into)
    }
}

/// Exact completion retaining both model execution and sampled token output.
pub struct MlxTextCompletion {
    model: MlxSessionCompletion,
    token: MlxCompletion,
}

impl Completion for MlxTextCompletion {
    type Error = Error;

    fn is_complete(&self) -> Result<bool, Self::Error> {
        Ok(self.model.is_complete()? && self.token.is_complete()?)
    }

    fn wait(&self) -> Result<(), Self::Error> {
        self.model.wait()?;
        self.token.wait()
    }
}

/// MLX sampling and randomness state for backend-generic text generation.
pub struct MlxTextGenerationState {
    temperature: f32,
    prng: Option<RandomState>,
    sampler: crate::backend::mlx::runtime::generation::sampler::GenerationSampler,
}

enum MlxSessionCompletionKind {
    Model(MlxCompletion),
    Pipeline(PipelineStageCompletion),
}

impl Completion for MlxSessionCompletion {
    type Error = Error;

    fn is_complete(&self) -> Result<bool, Self::Error> {
        match &self.inner {
            MlxSessionCompletionKind::Model(completion) => completion.is_complete(),
            MlxSessionCompletionKind::Pipeline(completion) => completion.is_complete(),
        }
    }

    fn wait(&self) -> Result<(), Self::Error> {
        match &self.inner {
            MlxSessionCompletionKind::Model(completion) => completion.wait(),
            MlxSessionCompletionKind::Pipeline(completion) => completion.synchronize(),
        }
    }
}

/// MLX-owned prefill input.
///
/// Arrays are cloned handles, not copied tensor storage. Owning the handles
/// makes submission independent of the caller's temporary `ModelInput` view.
#[derive(Debug, Clone)]
pub struct MlxModelInput {
    parts: Vec<MlxInputPart>,
}

#[derive(Debug, Clone)]
struct MlxInputPart {
    modality: input::Modality,
    payload: MlxInputPayload,
    metadata: MlxInputMetadata,
}

#[derive(Debug, Clone)]
enum MlxInputPayload {
    TokenIds(Array),
    Tensor(Array),
    Embeddings(Array),
}

#[derive(Debug, Clone, Default)]
struct MlxInputMetadata {
    qwen_grid_thw: Option<Array>,
    vision_grid_thw: Option<Array>,
    patch_position_ids: Option<Array>,
    audio_mask: Option<Array>,
}

impl From<input::ModelInput<'_>> for MlxModelInput {
    fn from(input: input::ModelInput<'_>) -> Self {
        Self {
            parts: input
                .parts
                .iter()
                .map(|part| MlxInputPart {
                    modality: part.modality,
                    payload: match part.payload {
                        input::InputPayload::TokenIds(value) => {
                            MlxInputPayload::TokenIds(value.clone())
                        }
                        input::InputPayload::Tensor(value) => {
                            MlxInputPayload::Tensor(value.clone())
                        }
                        input::InputPayload::Embeddings(value) => {
                            MlxInputPayload::Embeddings(value.clone())
                        }
                    },
                    metadata: MlxInputMetadata {
                        qwen_grid_thw: part.metadata.qwen_grid_thw.cloned(),
                        vision_grid_thw: part.metadata.vision_grid_thw.cloned(),
                        patch_position_ids: part.metadata.patch_position_ids.cloned(),
                        audio_mask: part.metadata.audio_mask.cloned(),
                    },
                })
                .collect(),
        }
    }
}

impl MlxModelInput {
    /// Converts processor-owned MLX values into an opaque backend prompt.
    #[cfg(feature = "mlx-media")]
    pub(crate) fn from_prepared(input: &PreparedModelInput) -> Self {
        input.with_model_input(|input| Self::from(input))
    }

    pub(crate) fn with_borrowed<T>(&self, execute: impl FnOnce(input::ModelInput<'_>) -> T) -> T {
        let parts = self
            .parts
            .iter()
            .map(|part| input::InputPart {
                modality: part.modality,
                payload: match &part.payload {
                    MlxInputPayload::TokenIds(value) => input::InputPayload::TokenIds(value),
                    MlxInputPayload::Tensor(value) => input::InputPayload::Tensor(value),
                    MlxInputPayload::Embeddings(value) => input::InputPayload::Embeddings(value),
                },
                metadata: input::InputMetadata {
                    qwen_grid_thw: part.metadata.qwen_grid_thw.as_ref(),
                    vision_grid_thw: part.metadata.vision_grid_thw.as_ref(),
                    patch_position_ids: part.metadata.patch_position_ids.as_ref(),
                    audio_mask: part.metadata.audio_mask.as_ref(),
                },
            })
            .collect::<Vec<_>>();
        execute(input::ModelInput::new(&parts))
    }
}

/// The single MLX implementation of architecture-erased prefill and decode.
///
/// Cache state and optional communication belong to the same selected model
/// session so callers cannot accidentally execute a sharded model with an
/// unrelated communicator.
pub struct MlxModelSession<'a> {
    inner: MlxSessionKind,
    distributed: Option<MlxDistributedSession<'a>>,
    #[cfg(feature = "mlx-media")]
    processor: Option<ModelProcessor>,
}

enum MlxSessionKind {
    Complete(Model, ModelCache),
    Pipeline(
        crate::composition::mlx_architectures::distributed::pipeline::PipelineModel,
        PipelineCache,
    ),
    Expert(
        crate::composition::mlx_architectures::distributed::expert::ExpertParallelModel,
        ExpertParallelCache,
    ),
}

impl<'a> MlxModelSession<'a> {
    pub(crate) fn from_model(
        model: MlxModel,
        distributed: Option<MlxDistributedSession<'a>>,
    ) -> Result<Self, Error> {
        let topology = model.topology();
        match (topology, distributed.as_ref()) {
            (None, None) => {}
            (Some(expected), Some(session)) if session.topology() == expected => {}
            (Some(_), None) => {
                return Err(Error::Parallel(
                    "distributed MLX model has no session-owned communication".into(),
                ))
            }
            (None, Some(_)) => {
                return Err(Error::Parallel(
                    "replicated MLX model cannot own distributed communication".into(),
                ))
            }
            (Some(expected), Some(session)) => {
                return Err(Error::Parallel(format!(
                    "model topology {expected:?} does not match session topology {:?}",
                    session.topology()
                )))
            }
        }
        #[cfg(feature = "mlx-media")]
        let processor = model.processor;
        let inner = match model.inner {
            MlxModelKind::Complete(model) => {
                let cache = model.new_cache();
                MlxSessionKind::Complete(model, cache)
            }
            MlxModelKind::Pipeline(model) => {
                let cache = model.new_cache()?;
                MlxSessionKind::Pipeline(model, cache)
            }
            MlxModelKind::Expert(model) => {
                let cache = model.new_cache();
                MlxSessionKind::Expert(model, cache)
            }
        };
        Ok(Self {
            inner,
            distributed,
            #[cfg(feature = "mlx-media")]
            processor,
        })
    }

    #[cfg(feature = "mlx-media")]
    pub(crate) fn processor(&self) -> Option<&ModelProcessor> {
        self.processor.as_ref()
    }

    /// Returns the normalized architecture name of the session-owned model.
    pub fn model_type(&self) -> &str {
        match &self.inner {
            MlxSessionKind::Complete(model, _) => model.model_type(),
            MlxSessionKind::Pipeline(model, _) => model.stage_info().model_kind.model_type_name(),
            MlxSessionKind::Expert(model, _) => model.info().model_kind.model_type_name(),
        }
    }

    /// Returns bounded parameter-residency telemetry when available.
    pub fn residency_report(&self) -> Result<Option<eredu_runtime::ResidencyReport>, Error> {
        match &self.inner {
            MlxSessionKind::Complete(model, _) => model.residency_report(),
            MlxSessionKind::Pipeline(model, _) => model.parameter_residency_report(),
            MlxSessionKind::Expert(_, _) => Ok(None),
        }
    }

    /// Returns dense checkpoint-streaming telemetry when enabled.
    pub fn dense_stream_report(
        &self,
    ) -> Result<Option<eredu_runtime::DenseDiskStreamReport>, Error> {
        match &self.inner {
            MlxSessionKind::Complete(model, _) => model.dense_stream_report(),
            MlxSessionKind::Pipeline(model, _) => model.dense_stream_report(),
            MlxSessionKind::Expert(model, _) => model.dense_stream_report(),
        }
    }

    /// Returns sparse routed-expert cache telemetry when enabled.
    pub fn expert_cache_report(
        &self,
    ) -> Result<
        Option<crate::backend::mlx::runtime::residency::expert_cache::ExpertCacheReport>,
        Error,
    > {
        match &self.inner {
            MlxSessionKind::Complete(model, _) => model.expert_cache_report(),
            MlxSessionKind::Pipeline(model, _) => model.expert_cache_report(),
            MlxSessionKind::Expert(model, _) => model.expert_cache_report(),
        }
    }

    /// Returns checkpoint-native quantization storage statistics when the
    /// selected session has one complete local model.
    pub fn native_quantization_stats(
        &self,
    ) -> Option<&safemlx::native_quantization::NativeQuantizationStats> {
        match &self.inner {
            MlxSessionKind::Complete(model, _) => model.native_quantization_stats(),
            MlxSessionKind::Pipeline(_, _) | MlxSessionKind::Expert(_, _) => None,
        }
    }

    /// Returns the canonical cache-relevant architecture identity.
    pub fn prompt_cache_architecture_fingerprint(&self) -> Result<String, Error> {
        match &self.inner {
            MlxSessionKind::Complete(model, _) => model
                .prompt_cache_architecture_fingerprint()
                .map_err(Into::into),
            MlxSessionKind::Pipeline(model, _) => Ok(model
                .prompt_cache_model_identity()?
                .architecture_fingerprint),
            MlxSessionKind::Expert(model, _) => Ok(model
                .prompt_cache_model_identity()?
                .architecture_fingerprint),
        }
    }

    /// Returns the exact ordered rank-local prompt-cache layout.
    pub fn prompt_cache_layer_layout(
        &self,
    ) -> Result<crate::LayerSchedule<crate::LayerCachePolicy>, Error> {
        match &self.inner {
            MlxSessionKind::Complete(model, _) => {
                model.prompt_cache_layer_layout().map_err(Into::into)
            }
            MlxSessionKind::Pipeline(model, _) => {
                Ok(model.prompt_cache_model_identity()?.layer_layout)
            }
            MlxSessionKind::Expert(model, _) => {
                Ok(model.prompt_cache_model_identity()?.layer_layout)
            }
        }
    }

    /// Returns each owned layer's processed-token delta from the persisted prefix.
    pub fn prompt_cache_layer_prefix_offsets(&self) -> Result<Vec<i32>, Error> {
        match &self.inner {
            MlxSessionKind::Complete(model, _) => model
                .prompt_cache_layer_prefix_offsets()
                .map_err(Into::into),
            MlxSessionKind::Pipeline(model, _) => {
                Ok(model.prompt_cache_model_identity()?.layer_prefix_offsets)
            }
            MlxSessionKind::Expert(model, _) => {
                Ok(model.prompt_cache_model_identity()?.layer_prefix_offsets)
            }
        }
    }

    pub(crate) fn complete_model(&self) -> &Model {
        match &self.inner {
            MlxSessionKind::Complete(model, _) => model,
            MlxSessionKind::Pipeline(_, _) | MlxSessionKind::Expert(_, _) => {
                unreachable!("replicated facade contains a distributed MLX session")
            }
        }
    }

    pub(super) fn complete_model_for_capabilities(&self) -> Option<&Model> {
        match &self.inner {
            MlxSessionKind::Complete(model, _) => Some(model),
            MlxSessionKind::Pipeline(_, _) | MlxSessionKind::Expert(_, _) => None,
        }
    }

    pub(crate) fn complete_parts_mut(&mut self) -> (&mut Model, &mut ModelCache) {
        match &mut self.inner {
            MlxSessionKind::Complete(model, cache) => (model, cache),
            MlxSessionKind::Pipeline(_, _) | MlxSessionKind::Expert(_, _) => {
                unreachable!("replicated facade contains a distributed MLX session")
            }
        }
    }

    pub(crate) fn new_complete_cache(&self) -> ModelCache {
        self.complete_model().new_cache()
    }

    #[cfg(test)]
    pub(crate) fn test_complete_model(&self) -> &Model {
        match &self.inner {
            MlxSessionKind::Complete(model, _) => model,
            MlxSessionKind::Pipeline(_, _) | MlxSessionKind::Expert(_, _) => {
                panic!("test expected a replicated MLX model session")
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn test_complete_cache(&self) -> &ModelCache {
        match &self.inner {
            MlxSessionKind::Complete(_, cache) => cache,
            MlxSessionKind::Pipeline(_, _) | MlxSessionKind::Expert(_, _) => {
                panic!("test expected a replicated MLX model-session cache")
            }
        }
    }

    /// Clears all backend-owned cache state while preserving session topology.
    pub fn reset(&mut self) -> Result<(), Error> {
        match &mut self.inner {
            MlxSessionKind::Complete(model, cache) => {
                *cache = model.new_cache();
                Ok(())
            }
            MlxSessionKind::Pipeline(_, cache) => cache.reset(),
            MlxSessionKind::Expert(_, cache) => cache.reset(),
        }
    }

    /// Returns aggregate cache-residency telemetry for this session.
    pub fn cache_residency_report(
        &self,
    ) -> Result<Option<eredu_runtime::CacheResidencyReport>, Error> {
        match &self.inner {
            MlxSessionKind::Complete(_, cache) => cache
                .residency_report()
                .map_err(|error| Error::Parallel(error.to_string())),
            MlxSessionKind::Pipeline(model, cache) => model.cache_residency_report(cache),
            MlxSessionKind::Expert(model, cache) => model.cache_residency_report(cache),
        }
    }

    /// Replaces this session's cache with one allocated under `policy`.
    ///
    /// Cache representation remains an MLX backend detail for complete,
    /// pipeline, and expert models alike.
    pub fn configure_cache(&mut self, policy: CacheResidencyPolicy) -> Result<(), Error> {
        match &mut self.inner {
            MlxSessionKind::Complete(model, cache) => {
                *cache = model.new_cache_with_options(policy)?;
            }
            MlxSessionKind::Pipeline(model, cache) => {
                *cache = model.new_cache_with_options(policy)?;
            }
            MlxSessionKind::Expert(model, cache) => {
                *cache = model.new_cache_with_options(policy)?;
            }
        }
        Ok(())
    }

    /// Atomically persists the completed prefix owned by this session.
    pub fn save_prompt_cache(
        &mut self,
        backend: &MlxBackend<'a>,
        root: impl AsRef<Path>,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
    ) -> Result<PromptCacheManifest, Error> {
        match &mut self.inner {
            MlxSessionKind::Complete(model, cache) => model
                .save_prompt_cache(
                    cache,
                    root,
                    descriptor,
                    prefix_token_ids,
                    options,
                    backend.stream(),
                )
                .map_err(Into::into),
            MlxSessionKind::Pipeline(model, cache) => model.save_prompt_cache(
                cache,
                root,
                descriptor,
                prefix_token_ids,
                options,
                backend.stream(),
            ),
            MlxSessionKind::Expert(model, cache) => model.save_prompt_cache(
                cache,
                root,
                descriptor,
                prefix_token_ids,
                options,
                backend.stream(),
            ),
        }
    }

    /// Opens a compatible persisted prefix and replaces this session's cache.
    pub fn load_prompt_cache(
        &mut self,
        backend: &MlxBackend<'a>,
        root: impl AsRef<Path>,
        expected: &PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: PagedCacheOptions,
    ) -> Result<PromptCacheManifest, Error> {
        let manifest = match &mut self.inner {
            MlxSessionKind::Complete(model, cache) => {
                let (loaded_cache, manifest) = model.load_prompt_cache(
                    root,
                    expected,
                    prefix_token_ids,
                    options,
                    backend.stream(),
                )?;
                *cache = loaded_cache;
                manifest
            }
            MlxSessionKind::Pipeline(model, cache) => {
                let (loaded_cache, manifest) = model.load_prompt_cache(
                    root,
                    expected,
                    prefix_token_ids,
                    options,
                    backend.stream(),
                )?;
                *cache = loaded_cache;
                manifest
            }
            MlxSessionKind::Expert(model, cache) => {
                let (loaded_cache, manifest) = model.load_prompt_cache(
                    root,
                    expected,
                    prefix_token_ids,
                    options,
                    backend.stream(),
                )?;
                *cache = loaded_cache;
                manifest
            }
        };
        Ok(manifest)
    }

    /// Runs embedded multi-token prediction through this session's opaque
    /// model, cache, and optional distributed capability.
    pub fn generate_embedded_mtp<S: SpeculativeSampler + Clone>(
        &mut self,
        backend: &MlxBackend<'a>,
        input: MlxModelInput,
        config: &MtpConfig,
        prng_key: Option<Array>,
        sampler: &mut S,
    ) -> Result<(Vec<u32>, MtpStats), Error> {
        let distributed = self.distributed.as_ref();
        input.with_borrowed(|input| match &mut self.inner {
            MlxSessionKind::Complete(model, cache) => match distributed {
                Some(execution) => model.generate_embedded_mtp_distributed(
                    cache, input, config, prng_key, sampler, execution,
                ),
                None => model.generate_embedded_mtp_input_with_sampler(
                    cache,
                    input,
                    config,
                    prng_key,
                    sampler,
                    backend.stream(),
                ),
            }
            .map_err(Into::into),
            MlxSessionKind::Pipeline(model, cache) => model
                .generate_embedded_mtp_distributed(
                    cache,
                    input,
                    config,
                    prng_key,
                    sampler,
                    distributed.ok_or_else(|| {
                        safemlx::error::Exception::custom(
                            "pipeline embedded MTP requires session-owned communication",
                        )
                    })?,
                )
                .map_err(Into::into),
            MlxSessionKind::Expert(model, cache) => model
                .generate_embedded_mtp_distributed(
                    cache,
                    input,
                    config,
                    prng_key,
                    sampler,
                    distributed.ok_or_else(|| {
                        safemlx::error::Exception::custom(
                            "expert embedded MTP requires session-owned communication",
                        )
                    })?,
                )
                .map_err(Into::into),
        })
    }

    /// Returns communication when this is a distributed session.
    pub const fn distributed(&self) -> Option<&MlxDistributedSession<'a>> {
        self.distributed.as_ref()
    }

    /// Samples on the canonical rank and synchronizes the result for this
    /// distributed model session.
    #[allow(clippy::too_many_arguments)]
    pub fn sample_and_synchronize<S: Sampler>(
        &self,
        logits: Option<&Array>,
        batch_size: i32,
        sampler: &mut S,
        temperature: f32,
        prng_state: Option<&mut RandomState>,
        finished: bool,
    ) -> Result<crate::backend::mlx::runtime::distributed::parallel::SynchronizedToken, Error> {
        self.distributed
            .as_ref()
            .ok_or_else(|| {
                Error::Parallel(
                    "sampling synchronization requires a distributed model session".into(),
                )
            })?
            .sample_and_synchronize(
                logits,
                batch_size,
                sampler,
                temperature,
                prng_state,
                finished,
            )
    }

    /// Runs one MLX instrumented pass through the architecture-erased adapter.
    pub(crate) fn forward_with_observer(
        model: &mut Model,
        input_tokens: &Array,
        mask: Option<&Array>,
        cache: &mut ModelCache,
        stream: &Stream,
        observer: &mut impl ActivationObserver,
    ) -> Result<Array, Error> {
        let result = match (model, cache) {
            (Model::DeepSeekV3(model), ModelCache::DeepSeekV3(cache)) => {
                if mask.is_some() {
                    return Err(Error::UnsupportedArchitecture(
                        "an explicit DeepSeek observer mask is unsupported; the adapter constructs the causal mask from cache state".into(),
                    ));
                }
                model.forward_with_observer(input_tokens, cache, stream, observer)
            }
            (Model::KimiLinear(model), ModelCache::KimiLinear(cache)) => {
                if mask.is_some() {
                    return Err(Error::UnsupportedArchitecture(
                        "an explicit Kimi Linear observer mask is unsupported; the adapter constructs the causal mask from cache state".into(),
                    ));
                }
                model.forward_with_observer(input_tokens, cache, stream, observer)
            }
            (Model::Llama(model), ModelCache::Llama(cache)) => {
                model.forward_with_observer(input_tokens, mask, cache, stream, observer)
            }
            (Model::DenseQwen(model), ModelCache::KeyValue(cache)) => {
                model.forward_with_observer(input_tokens, mask, cache, stream, observer)
            }
            (Model::DenseQwen(model), ModelCache::PagedKeyValue(cache)) => {
                model.forward_paged_with_observer(input_tokens, mask, cache, stream, observer)
            }
            (Model::MuseGlimmer(model), ModelCache::KeyValue(cache)) => {
                model.forward_with_observer(input_tokens, mask, cache, stream, observer)
            }
            (Model::MuseGlimmer(model), ModelCache::PagedKeyValue(cache)) => {
                model.forward_paged_with_observer(input_tokens, mask, cache, stream, observer)
            }
            (Model::Qwen3Next(model), ModelCache::Qwen3Next(cache))
            | (Model::Qwen35(model), ModelCache::Qwen35(cache)) => {
                if mask.is_some() {
                    return Err(Error::UnsupportedArchitecture(
                        "an explicit Qwen hybrid observer mask is unsupported; the adapter constructs the causal mask from cache state".into(),
                    ));
                }
                model.forward_with_observer(input_tokens, cache, stream, observer)
            }
            (Model::Gemma4(model), ModelCache::Gemma4(cache)) => {
                if mask.is_some() {
                    return Err(Error::UnsupportedArchitecture(
                        "an explicit Gemma observer mask is unsupported; the adapter constructs its per-layer masks from cache state".into(),
                    ));
                }
                model.forward_with_observer(input_tokens, cache, stream, observer)
            }
            (model, _) => {
                return Err(Error::UnsupportedArchitecture(format!(
                    "activation observation is unavailable for model type {} or the supplied cache does not match",
                    model.model_type()
                )))
            }
        };
        result.map_err(|error| safemlx::error::Exception::custom(error.to_string()).into())
    }

    /// Submits instrumented prefill through this selected MLX session.
    pub fn submit_prefill_with_observer(
        &mut self,
        backend: &MlxBackend<'a>,
        input: MlxModelInput,
        observer: &mut impl ActivationObserver,
    ) -> Result<Submission<Array, MlxCompletion>, Error> {
        match &mut self.inner {
            MlxSessionKind::Complete(model, cache) => Self::submit_complete_prefill_with_observer(
                model,
                input,
                cache,
                backend.stream(),
                observer,
            ),
            MlxSessionKind::Pipeline(_, _) | MlxSessionKind::Expert(_, _) => {
                Err(Error::UnsupportedArchitecture(
                    "activation observation is unavailable for distributed MLX sessions".into(),
                ))
            }
        }
    }

    pub(crate) fn submit_complete_prefill_with_observer(
        model: &mut Model,
        input: MlxModelInput,
        cache: &mut ModelCache,
        stream: &Stream,
        observer: &mut impl ActivationObserver,
    ) -> Result<Submission<Array, MlxCompletion>, Error> {
        let output = input.with_borrowed(|input| {
            if let (Model::Gemma4(model), ModelCache::Gemma4(cache)) = (&mut *model, &mut *cache) {
                return model
                    .prefill_input_with_observer(input, cache, stream, observer)
                    .map_err(|error| {
                        Error::Exception(safemlx::error::Exception::custom(error.to_string()))
                    })?
                    .try_index_device((.., -1, ..), stream)
                    .map_err(Error::from);
            }
            match (&mut *model, &mut *cache) {
                (Model::Qwen3Next(model), ModelCache::Qwen3Next(cache))
                | (Model::Qwen35(model), ModelCache::Qwen35(cache)) => model
                    .prefill_input_with_observer(input, cache, stream, observer)
                    .map_err(|error| {
                        Error::Exception(safemlx::error::Exception::custom(error.to_string()))
                    })?
                    .try_index_device((.., -1, ..), stream)
                    .map_err(Error::from),
                _ => {
                    let tokens = input::text_token_ids(input, stream)?;
                    Self::forward_with_observer(model, &tokens, None, cache, stream, observer)?
                        .try_index_device((.., -1, ..), stream)
                        .map_err(Error::from)
                }
            }
        })?;
        MlxCompletion::submission(output)
    }
}

impl<'a> BackendSession<MlxBackend<'a>> for MlxModelSession<'a> {
    type PrefillInput = MlxModelInput;
    type DecodeInput = Array;
    type Output = MlxModelOutput;
    type Completion = MlxSessionCompletion;

    fn prefill(
        &mut self,
        backend: &MlxBackend<'a>,
        input: Self::PrefillInput,
    ) -> Result<Submission<Self::Output, Self::Completion>, Error> {
        match &mut self.inner {
            MlxSessionKind::Complete(model, cache) => {
                let output = match &self.distributed {
                    Some(distributed) => input.with_borrowed(|input| {
                        prefill_model_tensor_parallel(
                            model,
                            cache,
                            input,
                            distributed,
                            backend.stream(),
                        )
                    })?,
                    None => input.with_borrowed(|input| {
                        prefill_model(model, cache, input, backend.stream())
                    })?,
                };
                model_submission(output)
            }
            MlxSessionKind::Pipeline(model, cache) => {
                let distributed = self.distributed.as_ref().ok_or_else(|| {
                    Error::Parallel("pipeline model session has no communication".into())
                })?;
                let completion = input.with_borrowed(|borrowed| {
                    let tokens = input::text_token_ids(borrowed, backend.stream())?;
                    let step = PipelineStep::new(tokens.dim(0), tokens.dim(1))?;
                    let multimodal = borrowed
                        .parts
                        .iter()
                        .any(|part| part.modality != input::Modality::Text);
                    if multimodal {
                        model.prefill_distributed(
                            model.stage_info().is_first.then_some(borrowed),
                            step,
                            None,
                            cache,
                            distributed,
                        )
                    } else {
                        model.forward_distributed(
                            model.stage_info().is_first.then_some(&tokens),
                            step,
                            None,
                            cache,
                            distributed,
                        )
                    }
                })?;
                pipeline_submission(completion)
            }
            MlxSessionKind::Expert(model, cache) => {
                let distributed = self.distributed.as_ref().ok_or_else(|| {
                    Error::Parallel("expert model session has no communication".into())
                })?;
                let output = input.with_borrowed(|borrowed| {
                    let tokens = input::text_token_ids(borrowed, backend.stream())?;
                    model.forward(&tokens, None, cache, distributed)
                })?;
                model_submission(output)
            }
        }
    }

    fn decode(
        &mut self,
        backend: &MlxBackend<'a>,
        input: Self::DecodeInput,
    ) -> Result<Submission<Self::Output, Self::Completion>, Error> {
        match &mut self.inner {
            MlxSessionKind::Complete(model, cache) => {
                let output = match &self.distributed {
                    Some(distributed) => decode_model_tensor_parallel(
                        model,
                        cache,
                        &input,
                        distributed,
                        backend.stream(),
                    )?,
                    None => decode_model(model, cache, &input, backend.stream())?,
                };
                model_submission(output)
            }
            MlxSessionKind::Pipeline(model, cache) => {
                let distributed = self.distributed.as_ref().ok_or_else(|| {
                    Error::Parallel("pipeline model session has no communication".into())
                })?;
                let step = PipelineStep::new(input.dim(0), input.dim(1))?;
                let completion = model.forward_distributed(
                    model.stage_info().is_first.then_some(&input),
                    step,
                    None,
                    cache,
                    distributed,
                )?;
                pipeline_submission(completion)
            }
            MlxSessionKind::Expert(model, cache) => {
                let distributed = self.distributed.as_ref().ok_or_else(|| {
                    Error::Parallel("expert model session has no communication".into())
                })?;
                model_submission(model.forward(&input, None, cache, distributed)?)
            }
        }
    }
}

impl<'a> TextGenerationBackend for MlxBackend<'a> {
    type Prompt = MlxModelInput;
    type Token = MlxTextToken;
    type TextGenerationState = MlxTextGenerationState;
    type TextCompletion = MlxTextCompletion;

    fn start_text_generation(
        _: &Self,
        config: TextGenerationConfig,
    ) -> Result<Self::TextGenerationState, Error> {
        let sampling = config.sampling();
        let prng = if sampling.temperature == 0.0 {
            None
        } else {
            Some(RandomState::from_key(safemlx::random::key(config.seed())?))
        };
        Ok(MlxTextGenerationState {
            temperature: sampling.temperature,
            prng,
            sampler:
                crate::backend::mlx::runtime::generation::sampler::GenerationSampler::from_resolved(
                    sampling,
                ),
        })
    }

    fn prepare_text_prompt(
        backend: &Self,
        prompt_token_ids: Vec<u32>,
    ) -> Result<Self::Prompt, Error> {
        if prompt_token_ids.is_empty() {
            return Err(Error::UnsupportedArchitecture(
                "text generation requires at least one prompt token".into(),
            ));
        }
        let tokens =
            Array::from(prompt_token_ids.as_slice()).try_index_device(NewAxis, backend.stream())?;
        let parts = [input::InputPart::text_token_ids(&tokens)];
        Ok(MlxModelInput::from(input::ModelInput::new(&parts)))
    }

    fn submit_text_prefill(
        runtime: &mut ModelRuntime<Self>,
        prompt: Self::Prompt,
        filter: &TokenFilter,
        state: &mut Self::TextGenerationState,
    ) -> Result<Submission<Self::Token, Self::TextCompletion>, Error> {
        let stream = runtime.backend().stream().clone();
        let submission = runtime.prefill(prompt)?;
        sample_text_submission(submission, filter, state, stream)
    }

    fn submit_text_decode(
        runtime: &mut ModelRuntime<Self>,
        token: Self::Token,
        filter: &TokenFilter,
        state: &mut Self::TextGenerationState,
    ) -> Result<Submission<Self::Token, Self::TextCompletion>, Error> {
        let stream = runtime.backend().stream().clone();
        let input = token.value.try_index_device((.., NewAxis), &stream)?;
        let submission = runtime.decode(input)?;
        sample_text_submission(submission, filter, state, stream)
    }
}

fn sample_text_submission(
    submission: Submission<MlxModelOutput, MlxSessionCompletion>,
    filter: &TokenFilter,
    state: &mut MlxTextGenerationState,
    stream: Stream,
) -> Result<Submission<MlxTextToken, MlxTextCompletion>, Error> {
    let logits = submission.output.into_logits().ok_or_else(|| {
        Error::Parallel("text generation requires logits on the local session rank".into())
    })?;
    let logits = crate::backend::mlx::runtime::generation::sampler::apply_token_filter(
        &logits, filter, &stream,
    )?;
    let token = state
        .sampler
        .sample(&logits, state.temperature, state.prng.as_mut(), &stream)?;
    let sampled = MlxCompletion::submission(token)?;
    Ok(Submission {
        output: MlxTextToken {
            value: sampled.output,
            stream,
        },
        completion: MlxTextCompletion {
            model: submission.completion,
            token: sampled.completion,
        },
    })
}

fn model_submission(
    output: Array,
) -> Result<Submission<MlxModelOutput, MlxSessionCompletion>, Error> {
    let submission = MlxCompletion::submission(output)?;
    Ok(Submission {
        output: MlxModelOutput::new(Some(submission.output)),
        completion: MlxSessionCompletion {
            inner: MlxSessionCompletionKind::Model(submission.completion),
        },
    })
}

fn pipeline_submission(
    completion: PipelineStageCompletion,
) -> Result<Submission<MlxModelOutput, MlxSessionCompletion>, Error> {
    let output = MlxModelOutput::new(completion.logits().cloned());
    Ok(Submission {
        output,
        completion: MlxSessionCompletion {
            inner: MlxSessionCompletionKind::Pipeline(completion),
        },
    })
}

fn prefill_pair<M, C>(
    model: &mut M,
    cache: &mut C,
    input: input::ModelInput<'_>,
    stream: &Stream,
) -> Result<Array, Error>
where
    M: CausalModel<C, Tensor = Array, Error = Exception>,
    for<'input> M: CausalModel<C, Input<'input> = input::ModelInput<'input>>,
{
    let logits = model.prefill_input_logits(input, cache, stream)?;
    model
        .adjust_prefill_logits(logits, cache, stream)
        .map_err(Into::into)
}

fn decode_pair<M, C>(
    model: &mut M,
    cache: &mut C,
    input: &Array,
    stream: &Stream,
) -> Result<Array, Error>
where
    M: CausalModel<C, Tensor = Array, Error = Exception>,
    for<'input> M: CausalModel<C, Input<'input> = input::ModelInput<'input>>,
{
    model
        .decode_logits(input, cache, stream)
        .map_err(Into::into)
}

fn prefill_model(
    model: &mut Model,
    cache: &mut ModelCache,
    input: input::ModelInput<'_>,
    stream: &Stream,
) -> Result<Array, Error> {
    match (model, cache) {
        (Model::DeepSeekV3(model), ModelCache::DeepSeekV3(cache)) => {
            prefill_pair(model, cache, input, stream)
        }
        (Model::DeepSeekV4(model), ModelCache::DeepSeekV4(cache)) => {
            prefill_pair(model.as_mut(), cache, input, stream)
        }
        (Model::DeepSeekV4Layerwise(model), ModelCache::DeepSeekV4(cache)) => {
            prefill_pair(model.as_mut(), cache, input, stream)
        }
        (Model::Gemma4(model), ModelCache::Gemma4(cache)) => {
            prefill_pair(model.as_mut(), cache, input, stream)
        }
        (Model::GptOss(model), ModelCache::GptOss(cache)) => {
            prefill_pair(model, cache, input, stream)
        }
        (Model::Inkling(model), ModelCache::Inkling(cache)) => {
            prefill_pair(model, cache, input, stream)
        }
        (Model::KimiLinear(model), ModelCache::KimiLinear(cache)) => {
            prefill_pair(model, cache, input, stream)
        }
        (Model::Llama(model), ModelCache::Llama(cache)) => {
            prefill_pair(model, cache, input, stream)
        }
        (Model::MuseGlimmer(model), ModelCache::KeyValue(cache)) => {
            prefill_pair(model, cache, input, stream)
        }
        (Model::MuseGlimmer(model), ModelCache::PagedKeyValue(cache)) => {
            prefill_pair(model, cache, input, stream)
        }
        (Model::Lfm2(model), ModelCache::Lfm2(cache)) => prefill_pair(model, cache, input, stream),
        (Model::NemotronH(model), ModelCache::NemotronH(cache)) => {
            prefill_pair(model, cache, input, stream)
        }
        (Model::DenseQwen(model), ModelCache::KeyValue(cache)) => {
            prefill_pair(model, cache, input, stream)
        }
        (Model::DenseQwen(model), ModelCache::PagedKeyValue(cache)) => {
            prefill_pair(model, cache, input, stream)
        }
        (Model::Qwen3Next(model), ModelCache::Qwen3Next(cache))
        | (Model::Qwen35(model), ModelCache::Qwen35(cache)) => {
            prefill_pair(model, cache, input, stream)
        }
        (Model::Qwen3Vl(model), ModelCache::Qwen3Vl(cache)) => {
            prefill_pair(model, cache, input, stream)
        }
        (Model::Qwen3VlMoe(model), ModelCache::Qwen3VlMoe(cache)) => {
            prefill_pair(model, cache, input, stream)
        }
        (model, _) => Err(Error::UnsupportedArchitecture(format!(
            "MLX cache does not match model type {}",
            model.model_type()
        ))),
    }
}

fn decode_model(
    model: &mut Model,
    cache: &mut ModelCache,
    input: &Array,
    stream: &Stream,
) -> Result<Array, Error> {
    match (model, cache) {
        (Model::DeepSeekV3(model), ModelCache::DeepSeekV3(cache)) => {
            decode_pair(model, cache, input, stream)
        }
        (Model::DeepSeekV4(model), ModelCache::DeepSeekV4(cache)) => {
            decode_pair(model.as_mut(), cache, input, stream)
        }
        (Model::DeepSeekV4Layerwise(model), ModelCache::DeepSeekV4(cache)) => {
            decode_pair(model.as_mut(), cache, input, stream)
        }
        (Model::Gemma4(model), ModelCache::Gemma4(cache)) => {
            decode_pair(model.as_mut(), cache, input, stream)
        }
        (Model::GptOss(model), ModelCache::GptOss(cache)) => {
            decode_pair(model, cache, input, stream)
        }
        (Model::Inkling(model), ModelCache::Inkling(cache)) => {
            decode_pair(model, cache, input, stream)
        }
        (Model::KimiLinear(model), ModelCache::KimiLinear(cache)) => {
            decode_pair(model, cache, input, stream)
        }
        (Model::Llama(model), ModelCache::Llama(cache)) => decode_pair(model, cache, input, stream),
        (Model::MuseGlimmer(model), ModelCache::KeyValue(cache)) => {
            decode_pair(model, cache, input, stream)
        }
        (Model::MuseGlimmer(model), ModelCache::PagedKeyValue(cache)) => {
            decode_pair(model, cache, input, stream)
        }
        (Model::Lfm2(model), ModelCache::Lfm2(cache)) => decode_pair(model, cache, input, stream),
        (Model::NemotronH(model), ModelCache::NemotronH(cache)) => {
            decode_pair(model, cache, input, stream)
        }
        (Model::DenseQwen(model), ModelCache::KeyValue(cache)) => {
            decode_pair(model, cache, input, stream)
        }
        (Model::DenseQwen(model), ModelCache::PagedKeyValue(cache)) => {
            decode_pair(model, cache, input, stream)
        }
        (Model::Qwen3Next(model), ModelCache::Qwen3Next(cache))
        | (Model::Qwen35(model), ModelCache::Qwen35(cache)) => {
            decode_pair(model, cache, input, stream)
        }
        (Model::Qwen3Vl(model), ModelCache::Qwen3Vl(cache)) => {
            decode_pair(model, cache, input, stream)
        }
        (Model::Qwen3VlMoe(model), ModelCache::Qwen3VlMoe(cache)) => {
            decode_pair(model, cache, input, stream)
        }
        (model, _) => Err(Error::UnsupportedArchitecture(format!(
            "MLX cache does not match model type {}",
            model.model_type()
        ))),
    }
}

pub(crate) fn submit_prefill_with_cache(
    model: &mut Model,
    cache: &mut ModelCache,
    input: MlxModelInput,
    stream: &Stream,
) -> Result<Submission<Array, MlxCompletion>, Error> {
    let output = input.with_borrowed(|input| prefill_model(model, cache, input, stream))?;
    MlxCompletion::submission(output)
}

pub(crate) fn submit_decode_with_cache(
    model: &mut Model,
    cache: &mut ModelCache,
    input: Array,
    stream: &Stream,
) -> Result<Submission<Array, MlxCompletion>, Error> {
    let output = decode_model(model, cache, &input, stream)?;
    MlxCompletion::submission(output)
}

fn last_token_logits(logits: Array, stream: &Stream) -> Result<Array, Error> {
    logits
        .try_index_device((.., -1, ..), stream)
        .map_err(Into::into)
}

fn with_dense_qwen_cache<T>(
    cache: &mut ModelCache,
    layout: eredu_runtime::StateLayout,
    execute: impl FnOnce(
        &mut crate::composition::mlx_architectures::qwen::dense::layerwise::DenseQwenLayerwiseCache,
    ) -> Result<T, Error>,
) -> Result<T, Error> {
    use crate::composition::mlx_architectures::qwen::dense::layerwise::DenseQwenLayerwiseCache;
    let mut owned = match cache {
        ModelCache::KeyValue(values) => {
            DenseQwenLayerwiseCache::concat(layout.clone(), std::mem::take(values))
        }
        ModelCache::PagedKeyValue(values) => {
            DenseQwenLayerwiseCache::paged(layout, std::mem::take(values))
        }
        _ => {
            return Err(Error::UnsupportedArchitecture(
                "dense Qwen cache does not match model".into(),
            ))
        }
    };
    let result = execute(&mut owned);
    match (cache, owned) {
        (
            ModelCache::KeyValue(values),
            DenseQwenLayerwiseCache::Concat {
                caches: restored, ..
            },
        ) => *values = restored,
        (
            ModelCache::PagedKeyValue(values),
            DenseQwenLayerwiseCache::Paged {
                caches: restored, ..
            },
        ) => *values = restored,
        _ => unreachable!("dense Qwen tensor-parallel cache wrapper changed variants"),
    }
    result
}

fn with_muse_cache<T>(
    cache: &mut ModelCache,
    layout: eredu_runtime::StateLayout,
    execute: impl FnOnce(
        &mut crate::composition::mlx_architectures::muse_glimmer::layerwise::MuseGlimmerLayerwiseCache,
    ) -> Result<T, Error>,
) -> Result<T, Error> {
    use crate::composition::mlx_architectures::muse_glimmer::layerwise::MuseGlimmerLayerwiseCache;
    let mut owned = match cache {
        ModelCache::KeyValue(values) => {
            MuseGlimmerLayerwiseCache::concat(layout.clone(), std::mem::take(values))
        }
        ModelCache::PagedKeyValue(values) => {
            MuseGlimmerLayerwiseCache::paged(layout, std::mem::take(values))
        }
        _ => {
            return Err(Error::UnsupportedArchitecture(
                "Muse-Glimmer cache does not match model".into(),
            ))
        }
    };
    let result = execute(&mut owned);
    match (cache, owned) {
        (
            ModelCache::KeyValue(values),
            MuseGlimmerLayerwiseCache::Concat {
                caches: restored, ..
            },
        ) => *values = restored,
        (
            ModelCache::PagedKeyValue(values),
            MuseGlimmerLayerwiseCache::Paged {
                caches: restored, ..
            },
        ) => *values = restored,
        _ => unreachable!("Muse-Glimmer tensor-parallel cache wrapper changed variants"),
    }
    result
}

fn prefill_model_tensor_parallel(
    model: &mut Model,
    cache: &mut ModelCache,
    input: input::ModelInput<'_>,
    distributed: &MlxDistributedSession<'_>,
    stream: &Stream,
) -> Result<Array, Error> {
    let group = distributed.tensor_group().ok_or_else(|| {
        Error::Parallel("tensor-parallel model session has no tensor communicator".into())
    })?;
    let logits = match (model, cache) {
        (Model::Gemma4(model), ModelCache::Gemma4(cache)) => {
            model.prefill_tensor_parallel(input, cache, group, stream)?
        }
        (Model::Inkling(model), ModelCache::Inkling(cache)) => {
            model.prefill_tensor_parallel(input, cache, group, stream)?
        }
        (Model::Qwen3Vl(model), ModelCache::Qwen3Vl(cache)) => {
            model.prefill_tensor_parallel(input, cache, group, stream)?
        }
        (Model::Qwen3VlMoe(model), ModelCache::Qwen3VlMoe(cache)) => {
            model.prefill_tensor_parallel(input, cache, group, stream)?
        }
        (Model::Qwen3Next(model), ModelCache::Qwen3Next(cache))
        | (Model::Qwen35(model), ModelCache::Qwen35(cache)) => {
            model.prefill_tensor_parallel(input, cache, group, stream)?
        }
        (
            Model::MuseGlimmer(model),
            cache @ (ModelCache::KeyValue(_) | ModelCache::PagedKeyValue(_)),
        ) => with_muse_cache(cache, model.state_layout()?, |cache| {
            model.prefill_tensor_parallel(input, cache, group, stream)
        })?,
        (model, cache) => {
            let tokens = input::text_token_ids(input, stream)?;
            forward_model_tensor_parallel(model, cache, &tokens, group, stream)?
        }
    };
    last_token_logits(logits, stream)
}

fn decode_model_tensor_parallel(
    model: &mut Model,
    cache: &mut ModelCache,
    input: &Array,
    distributed: &MlxDistributedSession<'_>,
    stream: &Stream,
) -> Result<Array, Error> {
    let group = distributed.tensor_group().ok_or_else(|| {
        Error::Parallel("tensor-parallel model session has no tensor communicator".into())
    })?;
    let logits = forward_model_tensor_parallel(model, cache, input, group, stream)?;
    last_token_logits(logits, stream)
}

fn forward_model_tensor_parallel(
    model: &mut Model,
    cache: &mut ModelCache,
    input: &Array,
    group: &safemlx::distributed::Group,
    stream: &Stream,
) -> Result<Array, Error> {
    match (model, cache) {
        (Model::DeepSeekV3(model), ModelCache::DeepSeekV3(cache)) => {
            model.forward_tensor_parallel(input, cache, group, stream)
        }
        (Model::DeepSeekV4Layerwise(model), ModelCache::DeepSeekV4(cache)) => {
            model.forward_tensor_parallel(input, cache, group, stream)
        }
        (Model::GptOss(model), ModelCache::GptOss(cache)) => {
            model.forward_tensor_parallel(input, cache, group, stream)
        }
        (Model::Inkling(model), ModelCache::Inkling(cache)) => {
            model.decode_tensor_parallel(input, cache, group, stream)
        }
        (Model::KimiLinear(model), ModelCache::KimiLinear(cache)) => {
            model.forward_tensor_parallel(input, cache, group, stream)
        }
        (Model::Llama(model), ModelCache::Llama(cache)) => {
            model.forward_tensor_parallel(input, cache, group, stream)
        }
        (Model::Lfm2(model), ModelCache::Lfm2(cache)) => {
            model.forward_tensor_parallel(input, cache, group, stream)
        }
        (Model::NemotronH(model), ModelCache::NemotronH(cache)) => {
            model.forward_tensor_parallel(input, cache, group, stream)
        }
        (Model::Gemma4(model), ModelCache::Gemma4(cache)) => {
            model.decode_tensor_parallel(input, cache, group, stream)
        }
        (
            Model::DenseQwen(model),
            cache @ (ModelCache::KeyValue(_) | ModelCache::PagedKeyValue(_)),
        ) => with_dense_qwen_cache(cache, model.cache_layout()?, |cache| {
            model.forward_tensor_parallel(input, None, cache, group, stream)
        }),
        (
            Model::MuseGlimmer(model),
            cache @ (ModelCache::KeyValue(_) | ModelCache::PagedKeyValue(_)),
        ) => with_muse_cache(cache, model.state_layout()?, |cache| {
            model.forward_tensor_parallel(input, None, cache, group, stream)
        }),
        (Model::Qwen3Next(model), ModelCache::Qwen3Next(cache))
        | (Model::Qwen35(model), ModelCache::Qwen35(cache)) => {
            model.forward_tensor_parallel(input, cache, group, stream)
        }
        (Model::Qwen3Vl(model), ModelCache::Qwen3Vl(cache)) => {
            model.decode_tensor_parallel(input, cache, group, stream)
        }
        (Model::Qwen3VlMoe(model), ModelCache::Qwen3VlMoe(cache)) => {
            model.decode_tensor_parallel(input, cache, group, stream)
        }
        (model, _) => Err(Error::UnsupportedArchitecture(format!(
            "tensor-parallel MLX cache does not match model type {}",
            model.model_type()
        ))),
    }
}

enum GenerationState {
    Prefill(MlxModelInput),
    Decode(Array),
}

/// Architecture-erased token generation over one MLX model session.
pub struct MlxGeneration<'a, S = DefaultSampler>
where
    S: Sampler,
{
    runtime: &'a mut ModelRuntime<MlxBackend<'static>>,
    temperature: f32,
    prng_state: Option<RandomState>,
    sampler: S,
    state: GenerationState,
    completions: Vec<MlxSessionCompletion>,
}

impl<'a> MlxGeneration<'a, DefaultSampler> {
    /// Creates an architecture-erased generation session with the default sampler.
    pub fn new(
        runtime: &'a mut ModelRuntime<MlxBackend<'static>>,
        temperature: f32,
        input: input::ModelInput<'_>,
        prng_key: Option<Array>,
    ) -> Self {
        Self::with_sampler(runtime, temperature, input, prng_key, DefaultSampler)
    }
}

impl<'a, S> MlxGeneration<'a, S>
where
    S: Sampler,
{
    /// Creates an architecture-erased generation session with a caller sampler.
    pub fn with_sampler(
        runtime: &'a mut ModelRuntime<MlxBackend<'static>>,
        temperature: f32,
        input: input::ModelInput<'_>,
        prng_key: Option<Array>,
        sampler: S,
    ) -> Self {
        Self {
            runtime,
            temperature,
            prng_state: prng_key.map(RandomState::from_key),
            sampler,
            state: GenerationState::Prefill(input.into()),
            completions: Vec::new(),
        }
    }

    /// Returns the sampler at its committed generated prefix.
    pub fn sampler_mut(&mut self) -> &mut S {
        &mut self.sampler
    }

    fn retain_completion(&mut self, completion: MlxSessionCompletion) -> Result<(), Error> {
        let mut retained = Vec::with_capacity(self.completions.len() + 1);
        for pending in self.completions.drain(..) {
            if !pending.is_complete()? {
                retained.push(pending);
            }
        }
        retained.push(completion);
        self.completions = retained;
        Ok(())
    }
}

impl<S> Iterator for MlxGeneration<'_, S>
where
    S: Sampler,
{
    type Item = Result<Array, safemlx::error::Exception>;

    fn next(&mut self) -> Option<Self::Item> {
        let stream = self.runtime.backend().stream().clone();
        let submission = match &self.state {
            GenerationState::Prefill(input) => self.runtime.prefill(input.clone()),
            GenerationState::Decode(token) => {
                let input = match token.try_index_device((.., NewAxis), &stream) {
                    Ok(input) => input,
                    Err(error) => return Some(Err(error)),
                };
                self.runtime.decode(input)
            }
        };
        let submission = match submission {
            Ok(submission) => submission,
            Err(error) => return Some(Err(safemlx::error::Exception::custom(error.to_string()))),
        };
        let Some(logits) = submission.output.into_logits() else {
            return Some(Err(safemlx::error::Exception::custom(
                "token generation requires logits on the local session rank",
            )));
        };
        if let Err(error) = self.retain_completion(submission.completion) {
            return Some(Err(safemlx::error::Exception::custom(error.to_string())));
        }
        let token =
            match self
                .sampler
                .sample(&logits, self.temperature, self.prng_state.as_mut(), &stream)
            {
                Ok(token) => token,
                Err(error) => return Some(Err(error)),
            };
        self.state = GenerationState::Decode(token.clone());
        Some(Ok(token))
    }
}

#[cfg(test)]
mod tests {
    use safemlx::Array;

    use super::*;

    #[test]
    fn owned_input_preserves_multimodal_parts_and_metadata() {
        let tokens = Array::from_slice(&[1_u32, 2], &[1, 2]);
        let image = Array::from_slice(&[0.0_f32; 8], &[1, 2, 4]);
        let grid = Array::from_slice(&[1_i32, 2, 2], &[1, 3]);
        let parts = [
            input::InputPart::text_token_ids(&tokens),
            input::InputPart::image_tensor(&image, input::InputMetadata::qwen_grid_thw(&grid)),
        ];

        let owned = MlxModelInput::from(input::ModelInput::new(&parts));
        owned.with_borrowed(|borrowed| {
            assert_eq!(borrowed.parts.len(), 2);
            assert_eq!(borrowed.parts[0].modality, input::Modality::Text);
            assert_eq!(borrowed.parts[1].modality, input::Modality::Image);
            assert_eq!(
                borrowed.parts[1]
                    .metadata
                    .qwen_grid_thw
                    .expect("grid metadata")
                    .shape(),
                &[1, 3]
            );
        });
    }

    #[test]
    fn model_session_is_the_backend_session_implementation() {
        fn assert_session<T: BackendSession<MlxBackend<'static>>>() {}
        assert_session::<MlxModelSession>();
    }
}
