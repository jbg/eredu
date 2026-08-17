//! Architecture-erased MLX model-session execution.

use safemlx::{
    ops::indexing::{NewAxis, TryIndexOp},
    random::RandomState,
    Array, Stream,
};
use safemlx_lm_core::{BackendSession, Completion, Submission};

use crate::{
    api::{input, Model, ModelCache},
    error::Error,
    nn::generation::CausalLm,
    runtime::execution::inspection::ActivationObserver,
    runtime::generation::sampler::{DefaultSampler, Sampler},
};

use super::{MlxBackend, MlxCompletion, MlxDistributedSession};

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
    cache: ModelCache,
    distributed: Option<MlxDistributedSession<'a>>,
}

impl<'a> MlxModelSession<'a> {
    pub(crate) const fn new(cache: ModelCache) -> Self {
        Self {
            cache,
            distributed: None,
        }
    }

    pub(crate) fn new_distributed(
        cache: ModelCache,
        distributed: MlxDistributedSession<'a>,
    ) -> Result<Self, Error> {
        if distributed.topology().is_replicated() {
            return Err(Error::Parallel(
                "distributed model sessions require a non-replicated topology".into(),
            ));
        }
        Ok(Self {
            cache,
            distributed: Some(distributed),
        })
    }

    /// Returns this session's cache state.
    pub const fn cache(&self) -> &ModelCache {
        &self.cache
    }

    /// Returns this session's mutable cache state.
    pub fn cache_mut(&mut self) -> &mut ModelCache {
        &mut self.cache
    }

    /// Replaces cache state after an explicitly validated restore.
    pub fn replace_cache(&mut self, cache: ModelCache) -> ModelCache {
        std::mem::replace(&mut self.cache, cache)
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
    ) -> Result<crate::SynchronizedToken, Error> {
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

    /// Submits instrumented prefill through the MLX adapter.
    pub(crate) fn submit_prefill_with_observer(
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
    type Output = Array;
    type Completion = MlxCompletion;

    fn prefill(
        &mut self,
        backend: &MlxBackend<'a>,
        model: &mut Model,
        input: Self::PrefillInput,
    ) -> Result<Submission<Self::Output, Self::Completion>, Error> {
        let output = match &self.distributed {
            Some(distributed) => input.with_borrowed(|input| {
                prefill_model_tensor_parallel(
                    model,
                    &mut self.cache,
                    input,
                    distributed,
                    backend.stream(),
                )
            })?,
            None => input.with_borrowed(|input| {
                prefill_model(model, &mut self.cache, input, backend.stream())
            })?,
        };
        MlxCompletion::submission(output)
    }

    fn decode(
        &mut self,
        backend: &MlxBackend<'a>,
        model: &mut Model,
        input: Self::DecodeInput,
    ) -> Result<Submission<Self::Output, Self::Completion>, Error> {
        let output = match &self.distributed {
            Some(distributed) => decode_model_tensor_parallel(
                model,
                &mut self.cache,
                &input,
                distributed,
                backend.stream(),
            )?,
            None => decode_model(model, &mut self.cache, &input, backend.stream())?,
        };
        MlxCompletion::submission(output)
    }
}

fn prefill_pair<M, C>(
    model: &mut M,
    cache: &mut C,
    input: input::ModelInput<'_>,
    stream: &Stream,
) -> Result<Array, Error>
where
    M: CausalLm<C>,
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
    M: CausalLm<C>,
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
    execute: impl FnOnce(
        &mut crate::architectures::qwen::dense::layerwise::DenseQwenLayerwiseCache,
    ) -> Result<T, Error>,
) -> Result<T, Error> {
    use crate::architectures::qwen::dense::layerwise::DenseQwenLayerwiseCache;
    let mut owned = match cache {
        ModelCache::KeyValue(values) => DenseQwenLayerwiseCache::Concat(std::mem::take(values)),
        ModelCache::PagedKeyValue(values) => DenseQwenLayerwiseCache::Paged(std::mem::take(values)),
        _ => {
            return Err(Error::UnsupportedArchitecture(
                "dense Qwen cache does not match model".into(),
            ))
        }
    };
    let result = execute(&mut owned);
    match (cache, owned) {
        (ModelCache::KeyValue(values), DenseQwenLayerwiseCache::Concat(restored)) => {
            *values = restored
        }
        (ModelCache::PagedKeyValue(values), DenseQwenLayerwiseCache::Paged(restored)) => {
            *values = restored
        }
        _ => unreachable!("dense Qwen tensor-parallel cache wrapper changed variants"),
    }
    result
}

fn with_muse_cache<T>(
    cache: &mut ModelCache,
    execute: impl FnOnce(
        &mut crate::architectures::muse_glimmer::layerwise::MuseGlimmerLayerwiseCache,
    ) -> Result<T, Error>,
) -> Result<T, Error> {
    use crate::architectures::muse_glimmer::layerwise::MuseGlimmerLayerwiseCache;
    let mut owned = match cache {
        ModelCache::KeyValue(values) => MuseGlimmerLayerwiseCache::Concat(std::mem::take(values)),
        ModelCache::PagedKeyValue(values) => {
            MuseGlimmerLayerwiseCache::Paged(std::mem::take(values))
        }
        _ => {
            return Err(Error::UnsupportedArchitecture(
                "Muse-Glimmer cache does not match model".into(),
            ))
        }
    };
    let result = execute(&mut owned);
    match (cache, owned) {
        (ModelCache::KeyValue(values), MuseGlimmerLayerwiseCache::Concat(restored)) => {
            *values = restored
        }
        (ModelCache::PagedKeyValue(values), MuseGlimmerLayerwiseCache::Paged(restored)) => {
            *values = restored
        }
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
        ) => with_muse_cache(cache, |cache| {
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
        ) => with_dense_qwen_cache(cache, |cache| {
            model.forward_tensor_parallel(input, None, cache, group, stream)
        }),
        (
            Model::MuseGlimmer(model),
            cache @ (ModelCache::KeyValue(_) | ModelCache::PagedKeyValue(_)),
        ) => with_muse_cache(cache, |cache| {
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
    model: &'a mut Model,
    cache: &'a mut ModelCache,
    backend: MlxBackend<'a>,
    temperature: f32,
    prng_state: Option<RandomState>,
    sampler: S,
    state: GenerationState,
    completions: Vec<MlxCompletion>,
}

impl<'a> MlxGeneration<'a, DefaultSampler> {
    /// Creates an architecture-erased generation session with the default sampler.
    pub fn new(
        model: &'a mut Model,
        cache: &'a mut ModelCache,
        temperature: f32,
        input: input::ModelInput<'_>,
        prng_key: Option<Array>,
        stream: &'a Stream,
    ) -> Self {
        Self::with_sampler(
            model,
            cache,
            temperature,
            input,
            prng_key,
            stream,
            DefaultSampler,
        )
    }
}

impl<'a, S> MlxGeneration<'a, S>
where
    S: Sampler,
{
    /// Creates an architecture-erased generation session with a caller sampler.
    pub fn with_sampler(
        model: &'a mut Model,
        cache: &'a mut ModelCache,
        temperature: f32,
        input: input::ModelInput<'_>,
        prng_key: Option<Array>,
        stream: &'a Stream,
        sampler: S,
    ) -> Self {
        Self {
            model,
            cache,
            backend: MlxBackend::new(stream),
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

    fn retain_completion(&mut self, completion: MlxCompletion) -> Result<(), Error> {
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
        let submission = match &self.state {
            GenerationState::Prefill(input) => submit_prefill_with_cache(
                self.model,
                self.cache,
                input.clone(),
                self.backend.stream(),
            ),
            GenerationState::Decode(token) => {
                let input = match token.try_index_device((.., NewAxis), self.backend.stream()) {
                    Ok(input) => input,
                    Err(error) => return Some(Err(error)),
                };
                submit_decode_with_cache(self.model, self.cache, input, self.backend.stream())
            }
        };
        let submission = match submission {
            Ok(submission) => submission,
            Err(error) => return Some(Err(safemlx::error::Exception::custom(error.to_string()))),
        };
        let logits = submission.output;
        if let Err(error) = self.retain_completion(submission.completion) {
            return Some(Err(safemlx::error::Exception::custom(error.to_string())));
        }
        let token = match self.sampler.sample(
            &logits,
            self.temperature,
            self.prng_state.as_mut(),
            self.backend.stream(),
        ) {
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
