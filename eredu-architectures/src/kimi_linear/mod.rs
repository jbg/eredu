//! Backend-neutral Kimi Linear configuration and operators.

/// Kimi normalization, residual, and typed block assembly.
pub mod block;
/// Pure SafeTensors/GGUF catalog plans and canonical name translation.
pub mod checkpoint;
pub mod config;
/// Kimi Delta Attention over neutral recurrent operators.
pub mod kda;
/// Kimi no-positional compressed latent attention.
pub mod mla;
/// Kimi dense-prefix and grouped routed-expert feed-forward operators.
pub mod moe;
/// Semantic placement and rank-local Kimi geometry.
pub mod parallel;
mod partitioned;

pub use block::{Block, BlockGeometry, TokenMixer};
pub use checkpoint::{
    expert_recipes, expert_residency_catalog, gguf_plan, load_time_quantization,
    normalize_weight_formats, safetensors_plan, translate_gguf_weight_name, unit_recipes,
    with_checkpoint_formats,
};
pub use config::{
    model_args_from_config_reader, model_args_from_config_value, model_args_from_gguf_catalog,
    prompt_cache_architecture_fingerprint, state_layout, state_layout_with_geometry, AttentionKind,
    ConfigError, FeedForwardPolicy, KdaConfig, LayerCacheGeometry, LayerPolicy, ModelArgs,
};
pub use kda::KimiDeltaAttention;
pub use mla::KimiLatentAttention;
pub use moe::{
    expert_realization_plan, replicated_expert_realization_plan, DenseSwiGlu, FeedForward,
    SparseMoe,
};
pub use parallel::{
    layer_parallel_parameter_groups, local_block_geometry, local_geometry, local_state_geometry,
    parameter_description, partition_local_geometry, partition_local_routed_geometry,
    partitioned_state_layout, static_parallel_parameter_groups, LocalGeometry,
    PartitionExpertBankOwnership, PartitionLocalGeometry,
};
pub use partitioned::{
    project_parameters, CompleteKimiGeometry, CompleteKimiKernel, KimiGeometry, KimiKernel, PartitionedLayeredModel,
};

use crate::decoder::StaticModuleSpec;
use eredu_core::cache::PromptCacheTopology;
use eredu_nn::Error;
use eredu_runtime::{ModelStateIdentity, StateLayout};

/// Architecture-owned values retained for one Kimi forward pass.
pub struct ForwardContext<T> {
    mask: Option<T>,
}

impl<T> ForwardContext<T> {
    /// Borrows the architecture-prepared attention mask at a partition boundary.
    pub const fn mask(&self) -> Option<&T> {
        self.mask.as_ref()
    }
}

/// Validates complete Kimi parameter, invocation, state and transport declarations.
/// This operation requires no native resources.
pub fn component_definition(args: &ModelArgs) -> Result<eredu_runtime::ComponentDefinition, Error> {
    partitioned::component_definition(
        parameter_description(args).map_err(Error::backend)?,
        state_layout(args).map_err(Error::backend)?,
    )
}

/// Complete Kimi execution through the shared typed component adapter.
pub type LayeredModel<B> = eredu_runtime::ComponentAdapter<
    std::sync::Arc<eredu_runtime::ComponentDefinition>,
    CompleteKimiKernel<B>,
    eredu_runtime::KernelParameterModules,
>;

pub(crate) fn static_spec(args: &ModelArgs) -> StaticModuleSpec {
    let embedding_name = "model.embed_tokens.weight";
    StaticModuleSpec {
        normalization_groups: None,
        embedding_weight: embedding_name.into(),
        normalization_weight: "model.norm.weight".into(),
        head_weight: "lm_head.weight".into(),
        vocabulary: args.vocab_size,
        hidden_size: args.hidden_size,
        normalization_epsilon: args.rms_norm_eps,
        normalization_offset: 0.0,
        embedding_quantization: args.weight_quantization_for(embedding_name),
        head_format: args.weight_quantization_for("lm_head.weight").into(),
        tied_head: args.tie_word_embeddings,
    }
}

/// Declares prompt-cache identity independently of the backend realization.
pub fn state_identity(
    args: &ModelArgs,
    layout: &StateLayout,
    global_layer_start: usize,
    topology: PromptCacheTopology,
) -> Result<ModelStateIdentity, Error> {
    let layer_count = usize::try_from(args.num_hidden_layers).map_err(Error::backend)?;
    let global_layer_end = global_layer_start
        .checked_add(layout.len())
        .ok_or_else(|| Error::backend("Kimi owned layer range overflowed"))?;
    if global_layer_end > layer_count {
        return Err(Error::backend(format!(
            "Kimi owns layers {global_layer_start}..{global_layer_end}, outside {layer_count} layers"
        )));
    }
    eredu_runtime::ModelStateIdentity::new(
        crate::ModelKind::KimiLinear.canonical_name(),
        args.model_type.clone(),
        prompt_cache_architecture_fingerprint(args),
        layer_count,
        global_layer_start,
        0,
        topology,
    )
    .map_err(Error::backend)
}
