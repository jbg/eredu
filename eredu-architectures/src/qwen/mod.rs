//! Backend-neutral Qwen2, Qwen3, and Qwen3-MoE text architecture policy.

mod checkpoint;
mod config;
pub mod hybrid;
mod moe;
mod parallel;
pub mod vision;
pub mod vl;

pub use checkpoint::{
    expert_recipes, expert_residency_catalog, expert_unit_recipes, gguf_plan,
    load_time_quantization, normalize_weight_formats, rank_local_expert_recipes, safetensors_plan,
    safetensors_plan_with_root, translate_gguf_weight_name, with_checkpoint_formats,
};
pub use config::{
    model_args_from_config_reader, model_args_from_config_value, model_args_from_gguf_catalog,
    model_args_from_gguf_catalog_with_context, model_args_from_text_config_value,
    prompt_cache_architecture_fingerprint, ConfigError, GgufTensorCatalog, ModelArgs, QwenVariant,
    TextConfigContext,
};
pub use moe::{expert_bank_spec, localized_expert_bank_spec, FeedForward, RoutedGatedProduct};
pub use parallel::{
    layer_parallel_parameter_groups, local_block_args, local_geometry, local_key_value_heads,
    routed_layer_parallel_parameter_groups, LocalGeometry,
};

pub use crate::decoder::{
    cache_layout, cache_layout_with_key_value_heads, create_caches, state_layout,
    static_parallel_parameter_groups, validate_caches, Attention, AttentionInput, ForwardContext,
    LayeredInput, Mlp, StaticModules,
};

use eredu_core::cache::PromptCacheTopology;
use eredu_nn::{
    Error, NeuralBackend, NormalizationConstructionSpec, ParameterSpec, RoutedNeuralBackend, Tensor,
};
use eredu_runtime::{ModelStateIdentity, StateLayout};

/// Dense Qwen decoder block.
pub type TransformerBlock<B> = crate::decoder::TransformerBlock<B>;

/// Dense Qwen transformer body.
pub type Decoder<B> = crate::decoder::Decoder<B>;

/// Qwen decoder block selected dynamically between dense and routed feed-forward policy.
pub type RoutedTransformerBlock<B> = crate::decoder::TransformerBlock<B, FeedForward<B>>;

/// Qwen transformer body selected dynamically between dense and routed feed-forward policy.
pub type RoutedDecoder<B> = crate::decoder::Decoder<B, FeedForward<B>>;

fn assemble_block<B: NeuralBackend, F>(
    args: &ModelArgs,
    layer: usize,
    mlp: F,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<crate::decoder::TransformerBlock<B, F>, Error> {
    Ok(crate::decoder::TransformerBlock {
        self_attention: Attention::new(args, layer, context)?,
        mlp,
        input_norm: B::normalization(
            NormalizationConstructionSpec::learned(
                args.hidden_size,
                args.rms_norm_eps,
                ParameterSpec::trainable(format!(
                    "{}.layers.{layer}.input_layernorm.weight",
                    args.parameter_root
                ))
                .map_err(Error::backend)?,
            ),
            context,
        )?,
        post_attention_norm: B::normalization(
            NormalizationConstructionSpec::learned(
                args.hidden_size,
                args.rms_norm_eps,
                ParameterSpec::trainable(format!(
                    "{}.layers.{layer}.post_attention_layernorm.weight",
                    args.parameter_root
                ))
                .map_err(Error::backend)?,
            ),
            context,
        )?,
    })
}

/// Builds one unloaded resident Qwen decoder body.
pub fn new_decoder<B: NeuralBackend>(
    args: &ModelArgs,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<Decoder<B>, Error> {
    crate::decoder::Decoder::new_with_factory::<ModelArgs, QwenBlockFactory>(args, context)
}

/// Statically dispatched dense Qwen block construction policy.
pub struct QwenBlockFactory;

/// Builds one unloaded dense Qwen decoder block.
pub fn new_block<B: NeuralBackend>(
    args: &ModelArgs,
    layer: usize,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<TransformerBlock<B>, Error> {
    <QwenBlockFactory as crate::decoder::BlockFactory<B, ModelArgs>>::build(args, layer, context)
}

impl<B> crate::decoder::BlockFactory<B, ModelArgs> for QwenBlockFactory
where
    B: NeuralBackend,
{
    type FeedForward = Mlp<B>;

    fn validate(args: &ModelArgs) -> Result<(), Error> {
        if args.is_moe() {
            return Err(Error::backend(
                "dense Qwen construction does not accept a routed MoE configuration",
            ));
        }
        Ok(())
    }

    fn build(
        args: &ModelArgs,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<TransformerBlock<B>, Error> {
        <Self as crate::decoder::BlockFactory<B, ModelArgs>>::validate(args)?;
        assemble_block(args, layer, Mlp::new(args, layer, context)?, context)
    }

    fn parameter_groups(
        block: &crate::decoder::TransformerBlock<B, Self::FeedForward>,
        args: &ModelArgs,
        layer: usize,
    ) -> Result<Vec<eredu_runtime::ParameterGroupSpec>, eredu_runtime::ParallelPlanError> {
        parallel::layer_parallel_parameter_groups(block, args, layer)
    }
}

/// Shared layered lifecycle specialized to dense Qwen policy.
pub type LayeredModel<B> = crate::decoder::LayeredModel<B, ModelArgs, QwenBlockFactory>;

/// Statically dispatched Qwen policy for adapters that admit dense and MoE configurations.
pub struct RoutedQwenBlockFactory;

impl<B> crate::decoder::BlockFactory<B, ModelArgs> for RoutedQwenBlockFactory
where
    B: RoutedNeuralBackend,
{
    type FeedForward = FeedForward<B>;

    fn build(
        args: &ModelArgs,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<RoutedTransformerBlock<B>, Error> {
        assemble_block(
            args,
            layer,
            FeedForward::new(args, layer, context)?,
            context,
        )
    }

    fn parameter_groups(
        block: &RoutedTransformerBlock<B>,
        args: &ModelArgs,
        layer: usize,
    ) -> Result<Vec<eredu_runtime::ParameterGroupSpec>, eredu_runtime::ParallelPlanError> {
        parallel::routed_layer_parallel_parameter_groups(block, args, layer)
    }
}

/// Builds a Qwen body for a backend adapter that dynamically admits dense or MoE configuration.
pub fn new_routed_decoder<B: RoutedNeuralBackend>(
    args: &ModelArgs,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<RoutedDecoder<B>, Error> {
    crate::decoder::Decoder::new_with_factory::<ModelArgs, RoutedQwenBlockFactory>(args, context)
}

/// Builds a Qwen block for a backend adapter that dynamically admits dense or MoE configuration.
pub fn new_routed_block<B: RoutedNeuralBackend>(
    args: &ModelArgs,
    layer: usize,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<RoutedTransformerBlock<B>, Error> {
    <RoutedQwenBlockFactory as crate::decoder::BlockFactory<B, ModelArgs>>::build(
        args, layer, context,
    )
}

/// Layered Qwen lifecycle for adapters that dynamically admit dense and MoE configurations.
pub type RoutedLayeredModel<B> = crate::decoder::LayeredModel<B, ModelArgs, RoutedQwenBlockFactory>;

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
        topology,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // This generic body is a compile-time contract: none of these dense Qwen
    // entry points may acquire a RoutedNeuralBackend bound.
    #[allow(dead_code)]
    fn dense_qwen_accepts_any_neural_backend<B: NeuralBackend>(
        args: ModelArgs,
        context: &<B::Tensor as Tensor>::Context,
    ) {
        let _: Result<Decoder<B>, Error> = new_decoder::<B>(&args, context);
        let _: Result<TransformerBlock<B>, Error> = new_block::<B>(&args, 0, context);
        let _: Result<LayeredModel<B>, Error> = LayeredModel::<B>::new(args, context);
    }

    #[test]
    fn dense_qwen_api_has_no_routed_backend_bound() {
        // The generic helper above is type-checked even though no concrete
        // backend is needed to exercise this compile-time assertion.
    }
}
