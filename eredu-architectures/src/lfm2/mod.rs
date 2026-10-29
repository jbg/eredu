//! Backend-neutral LFM2 and LFM2-MoE architecture policy.

/// Exact heterogeneous block equations and residual order.
pub mod block;
/// Pure SafeTensors/GGUF plans and canonical name translation.
pub mod checkpoint;
/// Strict configuration normalization and heterogeneous state geometry.
pub mod config;
/// Dense and routed feed-forward policy.
pub mod moe;
/// Semantic placement and rank-local construction geometry.
pub mod parallel;
/// Shared complete and selected execution over exact TP/PP resource geometry.
pub mod partitioned;

pub use block::{Block, BlockGeometry, TokenMixer};
pub use checkpoint::{
    expert_recipes, expert_residency_catalog, gguf_plan, load_time_quantization,
    normalize_weight_formats, safetensors_plan, translate_gguf_weight_name, unit_recipes,
    with_checkpoint_formats,
};

pub use config::{
    model_args_from_config_reader, model_args_from_config_value, model_args_from_gguf_catalog,
    prompt_cache_architecture_fingerprint, state_layout, state_layout_with_geometry, ConfigError,
    FeedForwardPolicy, LayerCacheGeometry, LayerPolicy, ModelArgs, OperatorPolicy, RopeConfig,
};
pub use moe::{
    expert_realization_plan, replicated_expert_realization_plan, DenseSwiGlu, FeedForward,
    RoutedGatedProduct,
};
pub use parallel::{
    dense_parameter_description, layer_parallel_parameter_groups, local_block_geometry,
    local_geometry, local_state_geometry, parameter_description, partition_local_geometry,
    partition_local_routed_geometry, partitioned_state_layout, static_parallel_parameter_groups,
    LocalGeometry, PartitionExpertBankOwnership, PartitionLocalGeometry,
};
pub use partitioned::{
    project_parameters, CompleteLfm2Geometry, CompleteLfm2Kernel, Lfm2Geometry, Lfm2Kernel,
    PartitionedLayeredModel,
};

use crate::decoder::StaticModuleSpec;
use eredu_core::cache::PromptCacheTopology;
use eredu_nn::Error;
use eredu_runtime::{ModelStateIdentity, StateLayout};

/// Shared cold and executable declaration for validated incremental text passes.
pub(crate) const CHUNKED_TEXT_PREFILL: bool = true;

/// Architecture-owned values retained for one LFM2 forward pass.
pub struct ForwardContext<T> {
    mask: Option<T>,
}

/// Validates complete LFM2 parameter, invocation, state and transport declarations.
/// This operation requires no native backend resources.
pub fn component_definition(args: &ModelArgs) -> Result<eredu_runtime::ComponentDefinition, Error> {
    partitioned::component_definition(
        parameter_description(args).map_err(Error::backend)?,
        state_layout(args).map_err(Error::backend)?,
    )
}

/// Complete LFM2 execution through the shared typed component adapter.
pub type LayeredModel<B> = eredu_runtime::ComponentAdapter<
    std::sync::Arc<eredu_runtime::ComponentDefinition>,
    CompleteLfm2Kernel<B>,
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
        .ok_or_else(|| Error::backend("LFM2 owned layer range overflowed"))?;
    if global_layer_end > layer_count {
        return Err(Error::backend(format!(
            "LFM2 owns layers {global_layer_start}..{global_layer_end}, outside {layer_count} layers"
        )));
    }
    eredu_runtime::ModelStateIdentity::new(
        "lfm2",
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
