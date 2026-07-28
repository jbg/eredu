//! Language-model loading and generation utilities built on `safemlx`.
//!
//! `safemlx-lm` provides model implementations, tokenizer loading, checkpoint
//! loading, cache management, and simple token generation for MLX-compatible
//! language models. The highest-level entry point is [`models::LoadedModel`],
//! which loads a model directory containing a Hugging Face-style `config.json`,
//! `tokenizer.json`, and safetensors weights.
//!
//! [`offload`] contains architecture-independent residency planning and
//! observability contracts. [`residency`] executes those plans for logical
//! weight units without coupling them to a model family.
//! [`weight_store`] catalogs safetensors checkpoints and safely materializes
//! lazily acquired selections from bounded persistent mappings.
//! [`layerwise`] provides a model-family adapter contract and a reusable
//! host-backed decoder engine. [`llama`] exposes one Llama/Mistral model API
//! across fully resident and bounded layer-execution residency policies.
//! [`expert_cache`] adds opt-in expert-granular hot-device, warm-host, and
//! cold-checkpoint residency for every supported safetensors MoE family,
//! including rank-owned expert-parallel catalogs and separate prefill/decode
//! telemetry.

#![warn(missing_docs)]

/// Model-family implementations and architecture-specific adapters.
pub mod architectures;
/// Architecture-neutral neural-network building blocks.
pub mod nn;
/// Architecture-independent model execution infrastructure.
pub mod runtime;
/// Compatibility path for attention key/value cache implementations.
pub use runtime::cache::kv as cache;
/// Compatibility path for attention-cache residency and persistence.
pub use runtime::cache::residency as cache_residency;
/// Chat-template preparation and native tool-runtime contracts.
pub mod chat;
/// Compatibility path for DeepSeek-V3 bounded execution.
pub use architectures::deepseek_v3::layerwise as deepseek_v3;
/// Compatibility path for bounded dense-layer checkpoint streaming.
pub use runtime::residency::dense_stream;
/// Error types returned by the language-model runtime.
pub mod error;
/// Compatibility path for executable expert-parallel model adapters.
pub use architectures::distributed::expert as expert_parallel;
/// Compatibility path for sparse routed-expert caching and telemetry.
pub use runtime::residency::expert_cache;
mod format_dialect;
/// Compatibility path for Gemma 4 bounded execution.
pub use architectures::gemma4::layerwise as gemma4;
pub(crate) use architectures::gemma4::mtp as gemma4_mtp;
pub(crate) use architectures::gpt_oss::format as harmony_format;
/// Compatibility path for GPT-OSS bounded execution.
pub use architectures::gpt_oss::layerwise as gpt_oss;
/// Compatibility path for Inkling bounded execution.
pub use architectures::inkling::layerwise as inkling;
pub(crate) use architectures::lfm2::format as lfm2_format;
/// Compatibility path for LFM2 bounded execution.
pub use architectures::lfm2::layerwise as lfm2;
/// Compatibility path for Llama bounded execution.
pub use architectures::llama::layerwise as llama;
/// Compatibility path for Moshi bounded execution.
pub use architectures::moshi::layerwise as moshi;
/// Compatibility path for Nemotron-H bounded execution.
pub use architectures::nemotron_h::layerwise as nemotron_h;
pub(crate) use architectures::qwen::hybrid::mtp as qwen_mtp;
/// Compatibility path for checkpoint binding and resident assignment.
pub use runtime::checkpoint::binding as module_binding;
/// Compatibility path for activation inspection hooks.
pub use runtime::execution::inspection;
/// Compatibility path for host-backed layerwise execution.
pub use runtime::execution::layerwise;
/// Compatibility path for weight-residency policy and telemetry.
pub use runtime::residency::policy as offload;
// pub mod generate;
/// High-level model loading, dispatch, and request APIs.
pub mod api;
/// Compatibility path for the former model facade.
pub use api as models;
/// Architecture-independent multi-token prediction and speculative decoding.
pub mod mtp;
/// Compatibility path for executable pipeline-parallel model adapters.
pub use architectures::distributed::pipeline;
/// Compatibility path for runtime topology and placement planning.
pub use runtime::distributed::topology as parallel;
/// Model-agnostic media processing and prepared-input helpers.
#[cfg(feature = "media-processing")]
pub mod processor;
/// Compatibility path for hybrid Qwen bounded execution.
pub use architectures::qwen::hybrid::layerwise as qwen_hybrid;
/// Compatibility path for Qwen3 bounded execution.
pub use architectures::qwen::qwen3::layerwise as qwen3;
/// Compatibility path for Qwen3-VL bounded execution.
pub use architectures::qwen::vl::layerwise as qwen3_vl;
/// Compatibility path for checkpoint quantization and conversion.
pub use runtime::checkpoint::quantization;
/// Codec-free realtime speech-to-speech token APIs.
pub mod realtime;
/// Compatibility path for immutable-weight residency management.
pub use runtime::residency::manager as residency;
/// Token sampling strategies.
pub mod sampler;
/// Protocol-independent semantic streaming contracts and machinery.
pub mod streaming;
/// Compatibility path for executable tensor-parallel model adapters.
pub use architectures::distributed::tensor as tensor_parallel;
#[cfg(test)]
mod test_utils;
mod tool_constraints;
/// Shared tensor, RoPE, attention, and tokenizer utilities.
pub mod utils;
/// Compatibility path for strict checkpoint loading and validation.
pub use runtime::checkpoint::load as weights;
/// Compatibility path for checkpoint-derived weight recipes.
pub use runtime::checkpoint::recipe as weight_recipe;
/// Compatibility path for persistent lazy checkpoint storage.
pub use runtime::checkpoint::store as weight_store;

pub use cache::PagedKeyValueCache;
pub use cache_residency::{
    inspect_prompt_cache, CacheBlockId, CacheBlockLifecycle, CacheLayerResidencyReport,
    CacheLayerResidencyStats, CacheRankIdentity, CacheRepresentation, CacheResidencyError,
    CacheResidencyManager, CacheResidencyPolicy, CacheResidencyReport, CacheTier,
    LiveCacheDiskPolicy, PagedCacheOptions, PromptCacheBlock, PromptCacheDescriptor,
    PromptCacheManifest, PromptCacheOptions, PromptCacheTopology,
    CACHE_RESIDENCY_LAYER_REPORT_LIMIT,
};
pub use dense_stream::{BackgroundPrefetchReport, DenseDiskStreamLoadOptions, DenseStreamError};
pub use expert_cache::SparseExpertDenseStreamLoadOptions;
pub use layerwise::{
    load_general_layerwise_model, DenseCacheMetrics, DenseDiskStreamReport,
    DenseExecutionGroupReport, DensePassReport, DenseTierResidencyReport, GeneralLayerwiseModel,
    GeneralLayerwiseModelAdapter, LayerExecutionLoadOptions, LayerwiseForwardState,
    LayerwiseLoadOptions, LayerwiseModel, LayerwiseModelAdapter, LayerwiseModelMetadata,
    WeightResidency,
};
pub use llama::{load_llama_model, LlamaCache, LlamaLoadOptions, LlamaModel};
pub use models::{
    check_model_config, check_model_config_json, check_model_dir, ModelConfigSupport,
    ModelLoadOptions, SupportedModelConfig,
};
pub use parallel::{
    DeviceAssignment, ParallelTopology, PlacementPlan, RankPartition, TensorPlacement,
};
pub use realtime::{
    load_model as load_realtime_model, load_model_with_options as load_realtime_model_with_options,
    LoadedRealtimeModel, RealtimeModelKind, RealtimeState,
};

use safemlx::Array;

use crate::models::qwen3 as resident_qwen3;

/// Builder passed to [`ModelInput`] implementations during generic generation.
pub struct ModelInputBuilder<'a, C, T> {
    /// Token ids or prompt ids for the current model step.
    pub y: &'a Array,
    /// Mutable per-layer cache used by the model implementation.
    pub cache: &'a mut Vec<Option<C>>,
    /// Caller-owned generation state carried across steps.
    pub state: &'a mut T,
}

/// Converts generic generation state into a model-specific input value.
pub trait ModelInput<'a, C, T> {
    /// Builds the concrete model input expected by a [`safemlx::module::Module`].
    fn from_model_input_builder(builder: ModelInputBuilder<'a, C, T>) -> Self;
}

impl<'a, C> ModelInput<'a, C, Option<Array>> for resident_qwen3::ModelInput<'a, C> {
    fn from_model_input_builder(builder: ModelInputBuilder<'a, C, Option<Array>>) -> Self {
        let ModelInputBuilder { y, cache, state } = builder;

        Self {
            inputs: y,
            mask: state.as_ref(),
            cache,
        }
    }
}

/// Output type that exposes logits for token sampling.
pub trait ModelOutput {
    /// Returns the logits tensor for the current generation step.
    fn logits(&self) -> &Array;
}

impl ModelOutput for Array {
    fn logits(&self) -> &Array {
        self
    }
}
