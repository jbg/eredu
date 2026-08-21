//! Backend-neutral Qwen2, Qwen3, and Qwen3-MoE text architecture policy.

mod checkpoint;
mod config;
pub mod hybrid;
mod moe;
mod parallel;
pub mod vision;
pub mod vl;

pub use checkpoint::{
    expert_recipes, gguf_plan, safetensors_plan, safetensors_plan_with_root,
    translate_gguf_weight_name,
};
pub use config::{
    model_args_from_config_reader, model_args_from_config_value, model_args_from_gguf_catalog,
    model_args_from_gguf_catalog_with_context, model_args_from_text_config_value,
    prompt_cache_architecture_fingerprint, ConfigError, GgufTensorCatalog, ModelArgs, QwenVariant,
    TextConfigContext,
};
pub use moe::{FeedForward, RoutedGatedProduct};
pub use parallel::{layer_parallel_parameter_groups, local_block_args, local_key_value_heads};

pub use crate::decoder::{
    cache_layout, cache_layout_with_key_value_heads, create_caches, state_layout,
    static_parallel_parameter_groups, validate_caches, Attention, AttentionInput, ForwardContext,
    LayeredInput, Mlp, StaticModules,
};

use eredu_core::cache::PromptCacheTopology;
use eredu_nn::{Error, NormalizationSpec, ParameterSpec, RoutedNeuralBackend, Tensor};
use eredu_runtime::{ModelStateIdentity, StateLayout};

/// The one Qwen decoder block type; dense versus MoE is its feed-forward policy.
pub type TransformerBlock<B> = crate::decoder::TransformerBlock<B, FeedForward<B>>;

/// Resident Qwen transformer body using the same dense-or-routed block policy.
pub type Decoder<B> = crate::decoder::Decoder<B, FeedForward<B>>;

/// Builds one unloaded resident Qwen decoder body.
pub fn new_decoder<B: RoutedNeuralBackend>(
    args: &ModelArgs,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<Decoder<B>, Error> {
    crate::decoder::Decoder::new_with_factory::<ModelArgs, QwenBlockFactory>(args, context)
}

/// Statically dispatched Qwen block construction policy.
pub struct QwenBlockFactory;

/// Builds one unloaded Qwen dense-or-routed decoder block.
pub fn new_block<B: RoutedNeuralBackend>(
    args: &ModelArgs,
    layer: usize,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<TransformerBlock<B>, Error> {
    <QwenBlockFactory as crate::decoder::BlockFactory<B, ModelArgs>>::build(args, layer, context)
}

impl<B> crate::decoder::BlockFactory<B, ModelArgs> for QwenBlockFactory
where
    B: RoutedNeuralBackend,
{
    type FeedForward = FeedForward<B>;

    fn build(
        args: &ModelArgs,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<TransformerBlock<B>, Error> {
        Ok(crate::decoder::TransformerBlock {
            self_attention: Attention::new(args, layer, context)?,
            mlp: FeedForward::new(args, layer, context)?,
            input_norm: B::rms_norm(
                NormalizationSpec {
                    dimensions: args.hidden_size,
                    epsilon: args.rms_norm_eps,
                    weight: ParameterSpec::trainable(format!(
                        "{}.layers.{layer}.input_layernorm.weight",
                        args.parameter_root
                    ))
                    .map_err(Error::backend)?,
                },
                context,
            )?,
            post_attention_norm: B::rms_norm(
                NormalizationSpec {
                    dimensions: args.hidden_size,
                    epsilon: args.rms_norm_eps,
                    weight: ParameterSpec::trainable(format!(
                        "{}.layers.{layer}.post_attention_layernorm.weight",
                        args.parameter_root
                    ))
                    .map_err(Error::backend)?,
                },
                context,
            )?,
        })
    }
}

/// Shared layered lifecycle specialized to Qwen dense-or-routed block policy.
pub type LayeredModel<B> = crate::decoder::LayeredModel<B, ModelArgs, QwenBlockFactory>;

/// Declares Qwen cache identity independently of concrete cache storage.
pub fn state_identity(
    args: &ModelArgs,
    layout: &StateLayout,
    global_layer_start: usize,
    topology: PromptCacheTopology,
) -> Result<ModelStateIdentity, Error> {
    let global_layer_end = global_layer_start
        .checked_add(layout.len())
        .ok_or_else(|| Error::backend("Qwen owned layer range overflowed"))?;
    let layer_count = usize::try_from(args.num_hidden_layers).map_err(Error::backend)?;
    if global_layer_end > layer_count {
        return Err(Error::backend(format!(
            "Qwen owns layers {global_layer_start}..{global_layer_end}, outside {layer_count} layers"
        )));
    }
    Ok(ModelStateIdentity {
        model_family: "qwen".into(),
        effective_model_type: args.model_type.clone(),
        architecture_fingerprint: prompt_cache_architecture_fingerprint(args),
        layer_count,
        global_layer_start,
        sink_tokens: 0,
        layer_prefix_offsets: vec![0; layout.len()],
        topology,
    })
}
