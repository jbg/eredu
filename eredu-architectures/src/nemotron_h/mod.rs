//! Backend-neutral Nemotron-H architecture.

pub mod attention;
pub mod block;
pub mod checkpoint;
pub mod config;
pub mod mamba;
pub mod mlp;
pub mod model;
pub mod mtp;
pub mod parallel;

pub use attention::{new_attention, new_attention_at};
pub use block::{Block, Operator};
pub use checkpoint::{
    expert_recipes, gguf_plan, normalized_checkpoint_keys, safetensors_plan, static_recipes,
    translate_gguf_weight_name, unit_recipes, unit_recipes_flat,
};
pub use config::{
    model_args_from_config_reader, model_args_from_config_value, model_args_from_gguf_catalog,
    prompt_cache_architecture_fingerprint, state_layout, state_layout_with_geometry, ConfigError,
    GgufTensorCatalog, LayerGeometry, LayerPolicy, ModelArgs, WeightDtype,
};
pub use mamba::Mamba2;
pub use mlp::{DenseMlp, SparseMoe};
pub use model::{
    ForwardContext, LayeredModel, TargetBoundary, TargetBoundarySchema, TargetPartitionInput, Unit,
};
pub use mtp::{EmbeddedInput, ForwardMode, PredictionUnit, RetainedValues};
pub use parallel::{
    layer_parallel_parameter_groups, local_block_geometry, local_geometry, local_state_geometry,
    static_parallel_parameter_groups, unit_parallel_parameter_groups, LocalGeometry,
};

/// Declares cache identity independently of its backend realization.
pub fn state_identity(
    args: &ModelArgs,
    layout: &eredu_runtime::StateLayout,
    global_layer_start: usize,
    topology: eredu_core::cache::PromptCacheTopology,
) -> Result<eredu_runtime::ModelStateIdentity, eredu_nn::Error> {
    let target = usize::try_from(args.num_hidden_layers).map_err(eredu_nn::Error::backend)?;
    let total = target
        .checked_add(args.mtp_policies().map_err(eredu_nn::Error::backend)?.len())
        .ok_or_else(|| eredu_nn::Error::backend("Nemotron-H state layer count overflowed"))?;
    let global_layer_end = global_layer_start
        .checked_add(layout.len())
        .ok_or_else(|| eredu_nn::Error::backend("Nemotron-H owned state range overflowed"))?;
    if global_layer_end > total {
        return Err(eredu_nn::Error::backend(format!(
            "Nemotron-H owns state layers {global_layer_start}..{global_layer_end}, outside {total} layers"
        )));
    }
    let mut layer_prefix_offsets = Vec::with_capacity(layout.len());
    for global_layer in global_layer_start..global_layer_end {
        // An embedded prediction group consumes the prompt's shifted
        // token/hidden pairs, so its recurrent and attention frontier is one
        // token behind the ordinary target group.
        layer_prefix_offsets.push(if global_layer < target { 0 } else { -1 });
    }
    Ok(eredu_runtime::ModelStateIdentity {
        model_family: "nemotron_h".into(),
        effective_model_type: args.model_type.clone(),
        architecture_fingerprint: prompt_cache_architecture_fingerprint(args),
        layer_count: total,
        global_layer_start,
        sink_tokens: 0,
        layer_prefix_offsets,
        topology,
    })
}
