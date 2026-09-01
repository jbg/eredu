//! Backend-neutral Llama/Mistral decoder implementation.

mod checkpoint;
mod config;

pub use checkpoint::{
    gguf_plan, load_time_quantization, safetensors_plan, translate_gguf_weight_name,
    with_checkpoint_formats, SafetensorsPlanError,
};
pub use config::{
    model_args_from_config_reader, model_args_from_config_value, model_args_from_gguf_catalog,
    prompt_cache_architecture_fingerprint, ConfigError, ModelArgs,
};

pub use crate::decoder::{
    cache_layout, cache_layout_with_key_value_heads, create_caches,
    layer_parallel_parameter_groups, state_layout, static_parallel_parameter_groups,
    validate_caches, Attention, AttentionInput, AttentionProjection, Config, ForwardContext,
    LayeredInput, Mlp, StaticModules, TransformerBlock,
};

use eredu_nn::Error;
use eredu_runtime::{ModelStateIdentity, ParallelPlanError, StateLayout};

/// Declares Llama's cache compatibility identity independently of its state storage backend.
pub fn state_identity(
    args: &ModelArgs,
    layout: &StateLayout,
    global_layer_start: usize,
    topology: eredu_core::cache::PromptCacheTopology,
) -> Result<ModelStateIdentity, Error> {
    let global_layer_end = global_layer_start
        .checked_add(layout.len())
        .ok_or_else(|| Error::backend("Llama owned layer range overflowed"))?;
    let layer_count = usize::try_from(args.num_hidden_layers).map_err(Error::backend)?;
    if global_layer_end > layer_count {
        return Err(Error::backend(format!(
            "Llama owns layers {global_layer_start}..{global_layer_end}, outside {layer_count} layers"
        )));
    }
    eredu_runtime::ModelStateIdentity::new(
        "llama",
        args.model_type.clone(),
        prompt_cache_architecture_fingerprint(args),
        layer_count,
        global_layer_start,
        0,
        topology,
    )
    .map_err(Error::backend)
}

/// Derives rank-local construction geometry for one tensor-parallel block.
pub fn local_block_args(
    args: &ModelArgs,
    layer: usize,
    layout: &eredu_runtime::LocalModelLayout,
) -> Result<ModelArgs, ParallelPlanError> {
    let prefix = format!("model.layers.{layer}");
    let tensor = |suffix: &str| {
        layout
            .tensor(&format!("{prefix}.{suffix}.weight"))
            .ok_or_else(|| {
                ParallelPlanError::InvalidTensor(format!(
                    "missing local layout for {prefix}.{suffix}.weight"
                ))
            })
    };
    let query = tensor("self_attn.q_proj")?;
    let key = tensor("self_attn.k_proj")?;
    let gate = tensor("mlp.gate_proj")?;
    let query_width = i32::try_from(*query.local_shape().first().ok_or_else(|| {
        ParallelPlanError::InvalidTensor("local Llama query projection is scalar".into())
    })?)
    .map_err(|_| ParallelPlanError::InvalidTensor("local Llama query width exceeds i32".into()))?;
    let key_width = i32::try_from(*key.local_shape().first().ok_or_else(|| {
        ParallelPlanError::InvalidTensor("local Llama key projection is scalar".into())
    })?)
    .map_err(|_| ParallelPlanError::InvalidTensor("local Llama key width exceeds i32".into()))?;
    if query_width <= 0
        || key_width <= 0
        || query_width % args.head_dim != 0
        || key_width % args.head_dim != 0
    {
        return Err(ParallelPlanError::InvalidTensor(format!(
            "local Llama attention widths q={query_width}, k={key_width} split head dimension {}",
            args.head_dim
        )));
    }
    let mut local = args.clone();
    local.num_attention_heads = query_width / args.head_dim;
    local.num_key_value_heads = key_width / args.head_dim;
    local.intermediate_size = i32::try_from(*gate.local_shape().first().ok_or_else(|| {
        ParallelPlanError::InvalidTensor("local Llama gate projection is scalar".into())
    })?)
    .map_err(|_| ParallelPlanError::InvalidTensor("local Llama MLP width exceeds i32".into()))?;
    if local.intermediate_size <= 0 {
        return Err(ParallelPlanError::InvalidTensor(
            "local Llama MLP width must be positive".into(),
        ));
    }
    Ok(local)
}

/// Returns the rank-local key/value head count for every decoder layer.
pub fn local_key_value_heads(
    args: &ModelArgs,
    layout: &eredu_runtime::LocalModelLayout,
) -> Result<Vec<i32>, ParallelPlanError> {
    (0..args.num_hidden_layers as usize)
        .map(|layer| Ok(local_block_args(args, layer, layout)?.num_key_value_heads))
        .collect()
}

/// Complete rank-local geometry for the canonical neutral Llama model.
pub type LocalGeometry = crate::decoder::LocalGeometry<ModelArgs>;

/// Derives Llama unit, vocabulary, and state geometry from one typed plan.
pub fn local_geometry(
    args: &ModelArgs,
    layout: &eredu_runtime::LocalModelLayout,
) -> Result<LocalGeometry, ParallelPlanError> {
    crate::decoder::local_geometry(args, layout, local_block_args)
}

/// Shared dense-decoder lifecycle specialized to Llama configuration policy.
pub type LayeredModel<B> = crate::decoder::LayeredModel<B, ModelArgs>;
