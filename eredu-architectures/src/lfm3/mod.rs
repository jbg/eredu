//! Backend-neutral LFM3 tri-hybrid decoder architecture.

pub mod declarations;
pub use declarations::Declaration;

/// Exact heterogeneous block equations and residual order.
pub mod block;
/// Pure SafeTensors/GGUF plans and canonical name translation.
pub mod checkpoint;
/// Strict configuration normalization and heterogeneous state geometry.
pub mod config;
/// Dense SwiGLU feed-forward policy.
pub mod feed_forward;
/// Semantic placement and rank-local construction geometry.
pub mod parallel;
/// Shared complete and selected execution over exact TP/PP resource geometry.
pub mod partitioned;
/// Independently constructible hybrid vision encoder and spatial projector.
pub mod vision;

pub use block::{Block, BlockGeometry, TokenMixer};
pub use checkpoint::{
    gguf_plan, load_time_quantization, safetensors_plan, translate_gguf_weight_name, unit_recipes,
    with_checkpoint_formats,
};

pub use config::{
    model_args_from_config_reader, model_args_from_config_value, model_args_from_gguf_catalog,
    prompt_cache_architecture_fingerprint, ConfigError, FeedForwardPolicy, LayerPolicy, ModelArgs,
    OperatorPolicy, RopeConfig,
};
pub use feed_forward::{DenseSwiGlu, FeedForward};
pub use parallel::{
    local_block_geometry, local_geometry, partition_local_geometry, partitioned_state_layout,
    LocalGeometry, PartitionLocalGeometry,
};
pub use partitioned::{
    CompleteLfm3Geometry, CompleteLfm3Kernel, Lfm3Geometry, Lfm3Kernel, PartitionedLayeredModel,
};

use crate::decoder::StaticModuleSpec;
use eredu_core::cache::PromptCacheTopology;
use eredu_nn::Error;
use eredu_runtime::{ModelStateIdentity, StateLayout};

/// Shared cold and executable declaration for validated incremental text passes.
pub(crate) const CHUNKED_TEXT_PREFILL: bool = true;

/// Architecture-owned values retained for one LFM3 forward pass.
pub struct ForwardContext<T> {
    mask: Option<T>,
}

/// Complete LFM3 execution through the shared typed component adapter.
pub type LayeredModel<B> = eredu_runtime::ComponentAdapter<
    std::sync::Arc<eredu_runtime::ComponentDefinition>,
    CompleteLfm3Kernel<B>,
    eredu_runtime::KernelParameterModules,
>;

/// Declares cache identity independently of its backend realization.
pub fn state_identity(
    args: &ModelArgs,
    layout: &StateLayout,
    global_layer_start: usize,
    topology: PromptCacheTopology,
) -> Result<ModelStateIdentity, Error> {
    let layer_count = usize::try_from(args.num_hidden_layers).map_err(Error::backend)?;
    let global_layer_end = global_layer_start
        .checked_add(layout.len())
        .ok_or_else(|| Error::backend("LFM3 owned layer range overflowed"))?;
    if global_layer_end > layer_count {
        return Err(Error::backend(format!(
            "LFM3 owns layers {global_layer_start}..{global_layer_end}, outside {layer_count} layers"
        )));
    }
    eredu_runtime::ModelStateIdentity::new(
        "lfm3",
        args.model_type.clone(),
        prompt_cache_architecture_fingerprint(args),
        layer_count,
        global_layer_start,
        0,
        topology,
    )
    .map_err(Error::backend)
}

pub(crate) use block::execution_topology;

pub(crate) fn static_spec(args: &ModelArgs) -> StaticModuleSpec {
    let embedding_name = "model.embed_tokens.weight";
    StaticModuleSpec {
        normalization_groups: None,
        embedding_weight: embedding_name.into(),
        normalization_weight: "model.embedding_norm.weight".into(),
        head_weight: "lm_head.weight".into(),
        vocabulary: args.vocab_size,
        hidden_size: args.hidden_size,
        normalization_epsilon: args.norm_eps,
        normalization_offset: 0.0,
        embedding_format: args.weight_quantization_for(embedding_name).into(),
        head_format: args.weight_quantization_for("lm_head.weight").into(),
        tied_head: args.tie_word_embeddings,
    }
}
