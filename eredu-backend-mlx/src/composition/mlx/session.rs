//! Architecture-erased MLX model-session execution.

use eredu_core::{
    BackendSession, Completion, ModelRuntime, Submission, TextGenerationBackend,
    TextGenerationConfig, TextSamplingStrategy, TokenFilter, TokenOutput,
};
use eredu_runtime::{
    ActivationObserver as RuntimeActivationObserver, CausalModel, GenerationSampler,
    MirostatV2Sampler,
};
use ref_cast::RefCast;
use safemlx::{
    error::Exception,
    ops::indexing::{NewAxis, TryIndexOp},
    random::RandomState,
    Array, Stream,
};
use std::path::Path;

use crate::{
    backend::runtime::generation::sampler::{Sampler, SpeculativeSampler},
    backend::runtime::media::input,
    backend::{error::Error, MlxModelKind},
    composition::mlx::distributed::pipeline::{
        PipelineCache, PipelineStageCompletion, PipelineStep,
    },
    MlxTensor,
};
#[cfg(feature = "media")]
use crate::{backend::runtime::media::PreparedModelInput, composition::mlx::ModelProcessor};
use eredu_core::cache::{PromptCacheDescriptor, PromptCacheManifest, PromptCacheOptions};
use eredu_core::generation::MtpConfig;
use eredu_core::{MtpCapability, MtpStats};
use eredu_runtime::{CacheResidencyPolicy, PagedCacheOptions};

use super::{MlxBackend, MlxCompletion, MlxDistributedSession, MlxModel, Model, ModelCache};

struct ArrayObserverAdapter<'a, O: ?Sized> {
    inner: &'a mut O,
}

impl<O> RuntimeActivationObserver<Array, Exception> for ArrayObserverAdapter<'_, O>
where
    O: RuntimeActivationObserver<MlxTensor, Exception> + ?Sized,
{
    fn observe(&mut self, path: &str, value: &Array) -> Result<(), Exception> {
        self.inner.observe(path, MlxTensor::ref_cast(value))
    }

    fn intervene(&mut self, path: &str, value: &Array) -> Result<Option<Array>, Exception> {
        self.inner
            .intervene(path, MlxTensor::ref_cast(value))
            .map(|replacement| replacement.map(MlxTensor::into_array))
    }

    fn observe_routing(
        &mut self,
        routing: eredu_runtime::RoutingObservation<'_, Array>,
    ) -> Result<(), Exception> {
        self.inner
            .observe_routing(eredu_runtime::RoutingObservation {
                path: routing.path,
                selected_experts: MlxTensor::ref_cast(routing.selected_experts),
                selected_scores: MlxTensor::ref_cast(routing.selected_scores),
                route_weights: MlxTensor::ref_cast(routing.route_weights),
                routed_output: MlxTensor::ref_cast(routing.routed_output),
                local_routed_output: routing.local_routed_output.map(MlxTensor::ref_cast),
                reduced_routed_output: routing.reduced_routed_output.map(MlxTensor::ref_cast),
                shared_output: routing.shared_output.map(MlxTensor::ref_cast),
                combined_output: routing.combined_output.map(MlxTensor::ref_cast),
                expert_count: routing.expert_count,
            })
    }
}

/// Backend-owned output of one MLX model-session submission.
///
/// Pipeline ranks that do not own the output projection complete with no local
/// logits. The final rank and every non-pipeline session complete with logits.
#[derive(Debug, Clone)]
pub struct MlxModelOutput {
    logits: Option<MlxTensor>,
}

impl MlxModelOutput {
    const fn new(logits: Option<MlxTensor>) -> Self {
        Self { logits }
    }

    /// Borrows local logits when this rank owns them.
    pub const fn logits(&self) -> Option<&MlxTensor> {
        self.logits.as_ref()
    }

    /// Consumes the output and returns local logits when present.
    pub fn into_logits(self) -> Option<MlxTensor> {
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
    sampler: MlxTextSampler,
}

enum MlxTextSampler {
    Standard(GenerationSampler),
    MirostatV2(MirostatV2Sampler),
}

impl MlxTextSampler {
    fn sample(
        &mut self,
        logits: &Array,
        temperature: f32,
        random: Option<&mut RandomState>,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        match self {
            Self::Standard(sampler) => sampler.sample(logits, temperature, random, stream),
            Self::MirostatV2(sampler) => sampler.sample(logits, temperature, random, stream),
        }
    }
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
    patch_grid: Option<Array>,
    patch_positions: Option<Array>,
    audio_mask: Option<Array>,
    patch_extent: Option<[i32; 3]>,
    audio_valid_frames: Option<i32>,
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
                        patch_grid: part.metadata.patch_grid.cloned(),
                        patch_positions: part.metadata.patch_positions.cloned(),
                        audio_mask: part.metadata.audio_mask.cloned(),
                        patch_extent: part.metadata.patch_extent,
                        audio_valid_frames: part.metadata.audio_valid_frames,
                    },
                })
                .collect(),
        }
    }
}

impl MlxModelInput {
    /// Converts processor-owned MLX values into an opaque backend prompt.
    #[cfg(feature = "media")]
    pub fn from_prepared(input: &PreparedModelInput) -> Self {
        input.with_model_input(|input| Self::from(input))
    }

    pub fn with_borrowed<T>(&self, execute: impl FnOnce(input::ModelInput<'_>) -> T) -> T {
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
                    patch_grid: part.metadata.patch_grid.as_ref(),
                    patch_positions: part.metadata.patch_positions.as_ref(),
                    audio_mask: part.metadata.audio_mask.as_ref(),
                    patch_extent: part.metadata.patch_extent,
                    audio_valid_frames: part.metadata.audio_valid_frames,
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
    runtime_state_dtype_bytes: std::num::NonZeroU8,
    distributed: Option<MlxDistributedSession<'a>>,
    #[cfg(feature = "media")]
    processor: Option<ModelProcessor>,
}

enum MlxSessionKind {
    Complete(Model, ModelCache),
    Pipeline(
        crate::composition::mlx::distributed::pipeline::PipelineModel,
        PipelineCache,
    ),
}

impl<'a> MlxModelSession<'a> {
    pub fn from_model(
        model: MlxModel,
        distributed: Option<MlxDistributedSession<'a>>,
    ) -> Result<Self, Error> {
        let runtime_state_dtype_bytes = model.runtime_state_dtype_bytes();
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
        #[cfg(feature = "media")]
        let mut model = model;
        #[cfg(feature = "media")]
        let processor = model.take_processor();
        let inner = match model.into_kind() {
            MlxModelKind::Complete(model) => {
                let cache = model.new_cache();
                MlxSessionKind::Complete(model, cache)
            }
            MlxModelKind::Pipeline(model) => {
                let cache = model.new_cache()?;
                MlxSessionKind::Pipeline(model, cache)
            }
        };
        Ok(Self {
            inner,
            runtime_state_dtype_bytes,
            distributed,
            #[cfg(feature = "media")]
            processor,
        })
    }

    pub(super) const fn runtime_state_dtype_bytes(&self) -> std::num::NonZeroU8 {
        self.runtime_state_dtype_bytes
    }

    #[cfg(feature = "media")]
    pub fn processor(&self) -> Option<&ModelProcessor> {
        self.processor.as_ref()
    }

    /// Returns the canonical architecture family of the session-owned model.
    pub fn model_family(&self) -> eredu_architectures::ModelKind {
        match &self.inner {
            MlxSessionKind::Complete(model, _) => model.model_family(),
            MlxSessionKind::Pipeline(model, _) => model.model_family(),
        }
    }

    /// Returns the effective model type preserved from the parsed configuration.
    pub fn effective_model_type(&self) -> &str {
        match &self.inner {
            MlxSessionKind::Complete(model, _) => model.effective_model_type(),
            MlxSessionKind::Pipeline(model, _) => model.effective_model_type(),
        }
    }

    /// Reports how the session-owned model exposes speculative weights.
    pub fn mtp_capability(&self) -> MtpCapability {
        match &self.inner {
            MlxSessionKind::Complete(model, _) => model.mtp_capability(),
            MlxSessionKind::Pipeline(model, _) => model.mtp_capability(),
        }
    }

    /// Returns bounded parameter-residency telemetry when available.
    pub fn residency_report(&self) -> Result<Option<eredu_runtime::ResidencyReport>, Error> {
        match &self.inner {
            MlxSessionKind::Complete(model, _) => model.residency_report(),
            MlxSessionKind::Pipeline(model, _) => model.parameter_residency_report(),
        }
    }

    /// Returns dense checkpoint-streaming telemetry when enabled.
    pub fn dense_stream_report(
        &self,
    ) -> Result<Option<eredu_runtime::DenseDiskStreamReport>, Error> {
        match &self.inner {
            MlxSessionKind::Complete(model, _) => model.dense_stream_report(),
            MlxSessionKind::Pipeline(model, _) => model.dense_stream_report(),
        }
    }

    /// Returns sparse routed-expert cache telemetry when enabled.
    pub fn expert_cache_report(
        &self,
    ) -> Result<Option<crate::backend::runtime::residency::expert_cache::ExpertCacheReport>, Error>
    {
        match &self.inner {
            MlxSessionKind::Complete(model, _) => model.expert_cache_report(),
            MlxSessionKind::Pipeline(model, _) => model.expert_cache_report(),
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
        }
    }

    /// Returns the exact ordered rank-local prompt-cache layout.
    pub fn prompt_cache_layer_layout(
        &self,
    ) -> Result<eredu_core::LayerSchedule<eredu_core::cache::LayerCachePolicy>, Error> {
        match &self.inner {
            MlxSessionKind::Complete(model, _) => {
                model.prompt_cache_layer_layout().map_err(Into::into)
            }
            MlxSessionKind::Pipeline(model, _) => {
                Ok(model.prompt_cache_model_identity()?.layer_layout)
            }
        }
    }

    /// Returns the global decoder-layer count used by prompt-cache identity.
    pub fn prompt_cache_layer_count(&self) -> Result<usize, Error> {
        match &self.inner {
            MlxSessionKind::Complete(model, _) => model
                .prompt_cache_layer_layout()
                .map(|layout| layout.len())
                .map_err(Into::into),
            MlxSessionKind::Pipeline(model, _) => {
                Ok(model.prompt_cache_model_identity()?.layer_count)
            }
        }
    }

    /// Returns the rank-local global decoder-layer range used by prompt-cache identity.
    pub fn prompt_cache_global_layer_range(&self) -> Result<std::ops::Range<usize>, Error> {
        match &self.inner {
            MlxSessionKind::Complete(model, _) => {
                let count = model
                    .prompt_cache_layer_layout()
                    .map(|layout| layout.len())
                    .map_err(Error::from)?;
                Ok(0..count)
            }
            MlxSessionKind::Pipeline(model, _) => {
                let identity = model.prompt_cache_model_identity()?;
                Ok(identity.global_layer_start..identity.global_layer_end)
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
        }
    }

    pub fn complete_model(&self) -> &Model {
        match &self.inner {
            MlxSessionKind::Complete(model, _) => model,
            MlxSessionKind::Pipeline(_, _) => {
                unreachable!("replicated facade contains a distributed MLX session")
            }
        }
    }

    pub(super) fn capability_estimate(
        &self,
    ) -> Result<eredu_architectures::capability::CapabilityEstimate, eredu_core::CapabilityError>
    {
        match &self.inner {
            MlxSessionKind::Complete(model, _) => model.architecture_capability_estimate(),
            MlxSessionKind::Pipeline(model, _) => model.capability_estimate(),
        }
    }

    pub(super) fn prepared_media_plan(
        &self,
        input: &eredu_architectures::media_plan::PreparedMediaInput,
    ) -> Result<eredu_architectures::media_plan::MediaShapePlan, eredu_core::CapabilityError> {
        match &self.inner {
            MlxSessionKind::Complete(model, _) => model.prepared_media_plan(input),
            MlxSessionKind::Pipeline(model, _) => model.prepared_media_plan(input),
        }
    }

    pub fn complete_parts_mut(&mut self) -> (&mut Model, &mut ModelCache) {
        match &mut self.inner {
            MlxSessionKind::Complete(model, cache) => (model, cache),
            MlxSessionKind::Pipeline(_, _) => {
                unreachable!("replicated facade contains a distributed MLX session")
            }
        }
    }

    pub fn new_complete_cache(&self) -> ModelCache {
        self.complete_model().new_cache()
    }

    #[cfg(test)]
    pub fn test_complete_model(&self) -> &Model {
        match &self.inner {
            MlxSessionKind::Complete(model, _) => model,
            MlxSessionKind::Pipeline(_, _) => {
                panic!("test expected a replicated MLX model session")
            }
        }
    }

    #[cfg(test)]
    pub fn test_complete_cache(&self) -> &ModelCache {
        match &self.inner {
            MlxSessionKind::Complete(_, cache) => cache,
            MlxSessionKind::Pipeline(_, _) => {
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
        }
    }

    /// Submits one cached decode position from a portable token id.
    pub fn submit_token_decode(
        &mut self,
        backend: &MlxBackend<'a>,
        token_id: u32,
    ) -> Result<Submission<MlxModelOutput, MlxSessionCompletion>, Error> {
        let token_ids = [token_id];
        let input =
            Array::from(token_ids.as_slice()).try_index_device(NewAxis, backend.stream())?;
        self.decode(backend, input)
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
        logits: Option<&MlxTensor>,
        batch_size: i32,
        sampler: &mut S,
        temperature: f32,
        prng_state: Option<&mut RandomState>,
        finished: bool,
    ) -> Result<crate::backend::runtime::distributed::parallel::SynchronizedToken, Error> {
        let distributed = self.distributed.as_ref().ok_or_else(|| {
            Error::Parallel("sampling synchronization requires a distributed model session".into())
        })?;
        let logits = logits.map(MlxTensor::as_array);
        match &self.inner {
            MlxSessionKind::Pipeline(model, _) => model.sample_and_synchronize_token(
                logits,
                batch_size,
                sampler,
                temperature,
                prng_state,
                finished,
                distributed,
            ),
            MlxSessionKind::Complete(_, _) => distributed.sample_and_synchronize(
                logits,
                batch_size,
                sampler,
                temperature,
                prng_state,
                finished,
            ),
        }
    }

    /// Runs one MLX instrumented pass through the architecture-erased adapter.
    pub fn forward_with_observer(
        model: &mut Model,
        input_tokens: &Array,
        mask: Option<&Array>,
        cache: &mut ModelCache,
        stream: &Stream,
        observer: &mut impl RuntimeActivationObserver<MlxTensor, Exception>,
    ) -> Result<Array, Error> {
        let mut observer = ArrayObserverAdapter { inner: observer };
        let result = match (model, cache) {
            (Model::DeepSeek(_, model), ModelCache::DeepSeek(cache)) => model
                .forward_with_observer(input_tokens, mask, cache, stream, &mut observer),
            (Model::KimiLinear(_, model), ModelCache::Hybrid(cache)) => {
                if mask.is_some() {
                    return Err(Error::ArchitectureModel(
                        "an explicit Kimi Linear observer mask is unsupported; the adapter constructs the causal mask from cache state".into(),
                    ));
                }
                model.forward_with_observer(input_tokens, mask, cache, stream, &mut observer)
            }
            (Model::Lfm2(_, model), ModelCache::Hybrid(cache)) => {
                model.forward_with_observer(input_tokens, mask, cache, stream, &mut observer)
            }
            (Model::Llama(_, model), ModelCache::Llama(cache)) => {
                model.forward_with_observer(input_tokens, mask, cache, stream, &mut observer)
            }
            (Model::Qwen(_, model), ModelCache::Qwen(cache)) => {
                model.forward_with_observer(input_tokens, mask, cache, stream, &mut observer)
            }
            (Model::MuseGlimmer(_, model), ModelCache::MuseGlimmer(cache)) => {
                if mask.is_some() {
                    return Err(Error::ArchitectureModel(
                        "explicit Muse-Glimmer observer masks are not bound yet".into(),
                    ));
                }
                let input_tokens = MlxTensor::from_array(input_tokens.clone());
                let output = model.forward_tokens(&input_tokens, cache, stream)?;
                observer.observe("model.logits", output.as_array())?;
                Ok(output.into_array())
            }
            (Model::Qwen3Next(_, model), ModelCache::Qwen3Next(cache))
            | (Model::Qwen35(_, model), ModelCache::Qwen35(cache)) => {
                if mask.is_some() {
                    return Err(Error::ArchitectureModel(
                        "an explicit Qwen hybrid observer mask is unsupported; the adapter constructs the causal mask from cache state".into(),
                    ));
                }
                model.forward_with_observer(input_tokens, cache, stream, &mut observer)
            }
            (Model::Gemma4(_, model), ModelCache::Hybrid(cache)) => {
                if mask.is_some() {
                    return Err(Error::ArchitectureModel(
                        "an explicit Gemma observer mask is unsupported; the adapter constructs its per-layer masks from cache state".into(),
                    ));
                }
                let input_tokens = MlxTensor::from_array(input_tokens.clone());
                let output = model.forward_tokens(&input_tokens, cache, stream)?;
                observer.observe("model.logits", output.as_array())?;
                Ok(output.into_array())
            }
            (model, _) => {
                return Err(Error::ArchitectureModel(format!(
                    "activation observation is unavailable for model type {} or the supplied cache does not match",
                    model.effective_model_type()
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
        observer: &mut impl RuntimeActivationObserver<MlxTensor, Exception>,
    ) -> Result<Submission<Array, MlxCompletion>, Error> {
        match &mut self.inner {
            MlxSessionKind::Complete(model, cache) => Self::submit_complete_prefill_with_observer(
                model,
                input,
                cache,
                backend.stream(),
                observer,
            ),
            MlxSessionKind::Pipeline(_, _) => Err(Error::ArchitectureModel(
                "activation observation is unavailable for distributed MLX sessions".into(),
            )),
        }
    }

    pub fn submit_complete_prefill_with_observer(
        model: &mut Model,
        input: MlxModelInput,
        cache: &mut ModelCache,
        stream: &Stream,
        observer: &mut impl RuntimeActivationObserver<MlxTensor, Exception>,
    ) -> Result<Submission<Array, MlxCompletion>, Error> {
        let output = input.with_borrowed(|input| match (&mut *model, &mut *cache) {
            (Model::Qwen3Next(_, model), ModelCache::Qwen3Next(cache))
            | (Model::Qwen35(_, model), ModelCache::Qwen35(cache)) => model
                .prefill_input_with_observer(
                    input,
                    cache,
                    stream,
                    &mut ArrayObserverAdapter { inner: observer },
                )
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
        let sampler = match config.strategy() {
            TextSamplingStrategy::Standard => {
                MlxTextSampler::Standard(GenerationSampler::from_resolved(sampling))
            }
            TextSamplingStrategy::MirostatV2 { tau, eta } => {
                let sampler = MirostatV2Sampler::new(tau, eta)
                    .map_err(|error| eredu_core::BackendError::Execution {
                        session: "text-generation".into(),
                        operation: "configure Mirostat V2".into(),
                        message: error.to_string(),
                    })?
                    .penalties(
                        sampling.repetition_penalty,
                        sampling.repeat_last_n,
                        sampling.frequency_penalty,
                        sampling.presence_penalty,
                    );
                MlxTextSampler::MirostatV2(sampler)
            }
        };
        Ok(MlxTextGenerationState {
            temperature: sampling.temperature,
            prng,
            sampler,
        })
    }

    fn prepare_text_prompt(
        backend: &Self,
        prompt_token_ids: Vec<u32>,
    ) -> Result<Self::Prompt, Error> {
        if prompt_token_ids.is_empty() {
            return Err(Error::ArchitectureModel(
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
    let logits = crate::backend::runtime::generation::sampler::apply_token_filter(
        logits.as_array(),
        filter,
        &stream,
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
        output: MlxModelOutput::new(Some(MlxTensor::from_array(submission.output))),
        completion: MlxSessionCompletion {
            inner: MlxSessionCompletionKind::Model(submission.completion),
        },
    })
}

fn pipeline_submission(
    completion: PipelineStageCompletion,
) -> Result<Submission<MlxModelOutput, MlxSessionCompletion>, Error> {
    let output = MlxModelOutput::new(completion.logits().cloned().map(MlxTensor::from_array));
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
    M: CausalModel<C, Tensor = MlxTensor, Error = Exception>,
    for<'input> M: CausalModel<C, Input<'input> = input::ModelInput<'input>>,
{
    let logits = model.prefill_input_logits(input, cache, stream)?;
    model
        .adjust_prefill_logits(logits, cache, stream)
        .map(MlxTensor::into_array)
        .map_err(Into::into)
}

fn decode_pair<M, C>(
    model: &mut M,
    cache: &mut C,
    input: &Array,
    stream: &Stream,
) -> Result<Array, Error>
where
    M: CausalModel<C, Tensor = MlxTensor, Error = Exception>,
    for<'input> M: CausalModel<C, Input<'input> = input::ModelInput<'input>>,
{
    let input = MlxTensor::from_array(input.clone());
    model
        .decode_logits(&input, cache, stream)
        .map(MlxTensor::into_array)
        .map_err(Into::into)
}

fn prefill_model(
    model: &mut Model,
    cache: &mut ModelCache,
    input: input::ModelInput<'_>,
    stream: &Stream,
) -> Result<Array, Error> {
    match (model, cache) {
        (Model::DeepSeek(_, model), ModelCache::DeepSeek(cache)) => {
            prefill_pair(model.as_mut(), cache, input, stream)
        }
        (Model::Gemma4(_, model), ModelCache::Hybrid(cache)) => {
            prefill_pair(model, cache, input, stream)
        }
        (Model::GptOss(_, model), ModelCache::GptOss(cache)) => {
            prefill_pair(model, cache, input, stream)
        }
        (Model::Inkling(_, model), ModelCache::Inkling(cache)) => {
            prefill_pair(model, cache, input, stream)
        }
        (Model::KimiLinear(_, model), ModelCache::Hybrid(cache)) => {
            prefill_pair(model, cache, input, stream)
        }
        (Model::Llama(_, model), ModelCache::Llama(cache)) => {
            prefill_pair(model, cache, input, stream)
        }
        (Model::MuseGlimmer(_, model), ModelCache::MuseGlimmer(cache)) => {
            prefill_pair(model, cache, input, stream)
        }
        (Model::Lfm2(_, model), ModelCache::Hybrid(cache)) => {
            prefill_pair(model, cache, input, stream)
        }
        (Model::NemotronH(_, model), ModelCache::Hybrid(cache)) => {
            prefill_pair(model, cache, input, stream)
        }
        (Model::Qwen(_, model), ModelCache::Qwen(cache)) => {
            prefill_pair(model, cache, input, stream)
        }
        (Model::Qwen3Next(_, model), ModelCache::Qwen3Next(cache))
        | (Model::Qwen35(_, model), ModelCache::Qwen35(cache)) => {
            prefill_pair(model, cache, input, stream)
        }
        (Model::Qwen3Vl(_, model), ModelCache::Qwen3Vl(cache)) => {
            prefill_pair(model, cache, input, stream)
        }
        (Model::Qwen3VlMoe(_, model), ModelCache::Qwen3VlMoe(cache)) => {
            prefill_pair(model, cache, input, stream)
        }
        (model, _) => Err(Error::ArchitectureModel(format!(
            "MLX cache does not match model type {}",
            model.effective_model_type()
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
        (Model::DeepSeek(_, model), ModelCache::DeepSeek(cache)) => {
            decode_pair(model.as_mut(), cache, input, stream)
        }
        (Model::Gemma4(_, model), ModelCache::Hybrid(cache)) => {
            decode_pair(model, cache, input, stream)
        }
        (Model::GptOss(_, model), ModelCache::GptOss(cache)) => {
            decode_pair(model, cache, input, stream)
        }
        (Model::Inkling(_, model), ModelCache::Inkling(cache)) => {
            decode_pair(model, cache, input, stream)
        }
        (Model::KimiLinear(_, model), ModelCache::Hybrid(cache)) => {
            decode_pair(model, cache, input, stream)
        }
        (Model::Llama(_, model), ModelCache::Llama(cache)) => {
            decode_pair(model, cache, input, stream)
        }
        (Model::MuseGlimmer(_, model), ModelCache::MuseGlimmer(cache)) => {
            decode_pair(model, cache, input, stream)
        }
        (Model::Lfm2(_, model), ModelCache::Hybrid(cache)) => {
            decode_pair(model, cache, input, stream)
        }
        (Model::NemotronH(_, model), ModelCache::Hybrid(cache)) => {
            decode_pair(model, cache, input, stream)
        }
        (Model::Qwen(_, model), ModelCache::Qwen(cache)) => {
            decode_pair(model, cache, input, stream)
        }
        (Model::Qwen3Next(_, model), ModelCache::Qwen3Next(cache))
        | (Model::Qwen35(_, model), ModelCache::Qwen35(cache)) => {
            decode_pair(model, cache, input, stream)
        }
        (Model::Qwen3Vl(_, model), ModelCache::Qwen3Vl(cache)) => {
            decode_pair(model, cache, input, stream)
        }
        (Model::Qwen3VlMoe(_, model), ModelCache::Qwen3VlMoe(cache)) => {
            decode_pair(model, cache, input, stream)
        }
        (model, _) => Err(Error::ArchitectureModel(format!(
            "MLX cache does not match model type {}",
            model.effective_model_type()
        ))),
    }
}

pub fn submit_prefill_with_cache(
    model: &mut Model,
    cache: &mut ModelCache,
    input: MlxModelInput,
    stream: &Stream,
) -> Result<Submission<Array, MlxCompletion>, Error> {
    let output = input.with_borrowed(|input| prefill_model(model, cache, input, stream))?;
    MlxCompletion::submission(output)
}

pub fn submit_decode_with_cache(
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
        (Model::Gemma4(_, model), ModelCache::Hybrid(cache)) => model
            .prefill_tensor_parallel(input, cache, group, stream)?
            .into_array(),
        (Model::Inkling(_, model), ModelCache::Inkling(cache)) => model
            .prefill_tensor_parallel(input, cache, group, stream)?
            .into_array(),
        (Model::Qwen3Vl(_, model), ModelCache::Qwen3Vl(cache)) => {
            model.prefill_tensor_parallel(input, cache, group, stream)?
        }
        (Model::Qwen3VlMoe(_, model), ModelCache::Qwen3VlMoe(cache)) => {
            model.prefill_tensor_parallel(input, cache, group, stream)?
        }
        (Model::Qwen3Next(_, model), ModelCache::Qwen3Next(cache))
        | (Model::Qwen35(_, model), ModelCache::Qwen35(cache)) => {
            model.prefill_tensor_parallel(input, cache, group, stream)?
        }
        (Model::MuseGlimmer(_, model), ModelCache::MuseGlimmer(cache)) => model
            .prefill_tensor_parallel(input, cache, group, stream)?
            .into_array(),
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
    let tensor_input = MlxTensor::from_array(input.clone());
    match (model, cache) {
        (Model::GptOss(_, model), ModelCache::GptOss(cache)) => {
            model.forward_tensor_parallel(input, cache, group, stream)
        }
        (Model::Inkling(_, model), ModelCache::Inkling(cache)) => model
            .forward_tensor_parallel(&tensor_input, cache, group, stream)
            .map(MlxTensor::into_array),
        (Model::KimiLinear(_, model), ModelCache::Hybrid(cache)) => {
            model.forward_tensor_parallel(input, cache, group, stream)
        }
        (Model::Llama(_, model), ModelCache::Llama(cache)) => {
            model.forward_tensor_parallel(input, cache, group, stream)
        }
        (Model::Lfm2(_, model), ModelCache::Hybrid(cache)) => {
            model.forward_tensor_parallel(input, cache, group, stream)
        }
        (Model::NemotronH(_, model), ModelCache::Hybrid(cache)) => {
            model.forward_tensor_parallel(input, cache, group, stream)
        }
        (Model::Gemma4(_, model), ModelCache::Hybrid(cache)) => model
            .forward_tensor_parallel(&tensor_input, cache, group, stream)
            .map(MlxTensor::into_array),
        (Model::Qwen(_, model), ModelCache::Qwen(cache)) => {
            model.forward_tensor_parallel(input, cache, group, stream)
        }
        (Model::MuseGlimmer(_, model), ModelCache::MuseGlimmer(cache)) => model
            .forward_tensor_parallel(&tensor_input, cache, group, stream)
            .map(MlxTensor::into_array),
        (Model::Qwen3Next(_, model), ModelCache::Qwen3Next(cache))
        | (Model::Qwen35(_, model), ModelCache::Qwen35(cache)) => {
            model.forward_tensor_parallel(input, cache, group, stream)
        }
        (Model::Qwen3Vl(_, model), ModelCache::Qwen3Vl(cache)) => {
            model.decode_tensor_parallel(input, cache, group, stream)
        }
        (Model::Qwen3VlMoe(_, model), ModelCache::Qwen3VlMoe(cache)) => {
            model.decode_tensor_parallel(input, cache, group, stream)
        }
        (model, _) => Err(Error::ArchitectureModel(format!(
            "tensor-parallel MLX cache does not match model type {}",
            model.effective_model_type()
        ))),
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
            input::InputPart::image_tensor(&image, input::InputMetadata::patch_grid(&grid)),
        ];

        let owned = MlxModelInput::from(input::ModelInput::new(&parts));
        owned.with_borrowed(|borrowed| {
            assert_eq!(borrowed.parts.len(), 2);
            assert_eq!(borrowed.parts[0].modality, input::Modality::Text);
            assert_eq!(borrowed.parts[1].modality, input::Modality::Image);
            assert_eq!(
                borrowed.parts[1]
                    .metadata
                    .patch_grid
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
