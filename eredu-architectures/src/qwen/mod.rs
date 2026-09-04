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
    prompt_cache_architecture_fingerprint, ConfigError, ModelArgs, QwenVariant, TextConfigContext,
};
pub use moe::{
    expert_bank_spec, expert_realization_plan, partition_expert_realization_plan,
    replicated_expert_realization_plan, FeedForward, RoutedGatedProduct,
};
pub use parallel::{
    layer_parallel_parameter_groups, local_block_args, local_geometry, local_key_value_heads,
    partition_local_routed_geometry, routed_layer_parallel_parameter_groups, LocalGeometry,
};

pub use crate::decoder::{
    cache_layout, cache_layout_with_key_value_heads, create_caches, dense_parameter_description,
    partition_local_geometry, state_layout, static_parallel_parameter_groups, validate_caches,
    Attention, AttentionInput, ForwardContext, LayeredInput, Mlp, PartitionLocalGeometry,
    PartitionStaticModules, StaticModules,
};

use eredu_core::cache::PromptCacheTopology;
use eredu_nn::{
    Error, GroupedNeuralBackend, NeuralBackend, NormalizationConstructionSpec, ParameterSpec,
    Tensor,
};
use eredu_runtime::{ModelStateIdentity, StateLayout};

/// Dense Qwen decoder block.
pub type TransformerBlock<B> = crate::decoder::TransformerBlock<B>;

/// Qwen decoder block selected dynamically between dense and routed feed-forward policy.
pub type RoutedTransformerBlock<B> = crate::decoder::TransformerBlock<B, FeedForward<B>>;

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

/// Genuinely pipeline-local neutral dense Qwen2/Qwen3 model.
pub type PartitionedLayeredModel<B> =
    crate::decoder::PartitionedLayeredModel<B, ModelArgs, QwenBlockFactory>;

impl crate::decoder::PartitionedConfig for ModelArgs {
    fn set_local_geometry(
        &mut self,
        query_heads: i32,
        key_value_heads: i32,
        intermediate: i32,
    ) -> Result<(), eredu_nn::Error> {
        if self.is_moe() {
            return Err(eredu_nn::Error::backend(
                "local dense Qwen geometry does not accept routed MoE configuration",
            ));
        }
        if query_heads <= 0 || key_value_heads <= 0 || intermediate <= 0 {
            return Err(eredu_nn::Error::backend(
                "local dense Qwen geometry must be positive",
            ));
        }
        self.num_attention_heads = query_heads;
        self.num_key_value_heads = key_value_heads;
        self.intermediate_size = intermediate;
        Ok(())
    }

    fn local_block_config(
        &self,
        layer: usize,
        layout: &eredu_runtime::LocalModelLayout,
    ) -> Result<Self, eredu_nn::Error> {
        if self.is_moe() {
            return Err(eredu_nn::Error::backend(
                "local dense Qwen geometry does not accept routed MoE configuration",
            ));
        }
        parallel::local_block_args(self, layer, layout).map_err(eredu_nn::Error::backend)
    }

    fn validate_partition_parameters(
        &self,
        parameters: &eredu_runtime::ArchitectureParameterDescription,
    ) -> Result<(), eredu_nn::Error> {
        if !self.is_moe() {
            let expected = crate::decoder::dense_parameter_description(self)
                .map_err(eredu_nn::Error::backend)?;
            return (&expected == parameters)
                .then_some(())
                .ok_or_else(|| eredu_nn::Error::backend("dense Qwen parameter topology drifted"));
        }
        crate::decoder::validate_partitioned_decoder_description(self, parameters)?;
        validate_routed_partition_parameters(self, parameters)
    }
}

/// Statically dispatched Qwen policy for adapters that admit dense and MoE configurations.
pub struct RoutedQwenBlockFactory;

impl<B> crate::decoder::BlockFactory<B, ModelArgs> for RoutedQwenBlockFactory
where
    B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
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

    fn build_partitioned(
        global: &ModelArgs,
        local: &ModelArgs,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<RoutedTransformerBlock<B>, Error> {
        if !global.is_moe() || !local.is_moe() {
            return Err(Error::backend(
                "partitioned routed Qwen construction requires Qwen3-MoE",
            ));
        }
        assemble_block(
            local,
            layer,
            FeedForward::new_partitioned(global, local, layer, context)?,
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

/// Builds a Qwen block for a backend adapter that dynamically admits dense or MoE configuration.
pub fn new_routed_block<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend>(
    args: &ModelArgs,
    layer: usize,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<RoutedTransformerBlock<B>, Error> {
    <RoutedQwenBlockFactory as crate::decoder::BlockFactory<B, ModelArgs>>::build(
        args, layer, context,
    )
}

/// Builds one partition-local routed block with a global router and the exact
/// TP/EP-local expert bank selected by architecture geometry.
pub fn new_partitioned_routed_block<
    B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
>(
    global: &ModelArgs,
    local: &ModelArgs,
    layer: usize,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<RoutedTransformerBlock<B>, Error> {
    <RoutedQwenBlockFactory as crate::decoder::BlockFactory<B, ModelArgs>>::build_partitioned(
        global, local, layer, context,
    )
}

/// Layered Qwen lifecycle for adapters that dynamically admit dense and MoE configurations.
pub type RoutedLayeredModel<B> = crate::decoder::LayeredModel<B, ModelArgs, RoutedQwenBlockFactory>;

/// Pipeline-local Qwen3-MoE model with global routing and local expert banks.
pub type PartitionedRoutedLayeredModel<B> =
    crate::decoder::PartitionedLayeredModel<B, ModelArgs, RoutedQwenBlockFactory>;

fn validate_routed_partition_parameters(
    args: &ModelArgs,
    parameters: &eredu_runtime::ArchitectureParameterDescription,
) -> Result<(), Error> {
    let experts = usize::try_from(args.num_experts).map_err(Error::backend)?;
    let layers = usize::try_from(args.num_hidden_layers).map_err(Error::backend)?;
    for layer in 0..layers {
        let prefix = format!("{}.layers.{layer}", args.parameter_root);
        for required in [
            format!("{prefix}.mlp.gate"),
            format!("{prefix}.mlp.experts.intermediate"),
            format!("{prefix}.self_attn.q_norm"),
            format!("{prefix}.self_attn.k_norm"),
        ] {
            if !parameters
                .groups()
                .iter()
                .any(|owned| owned.group().logical_name() == required)
            {
                return Err(Error::backend(format!(
                    "routed Qwen parameter description omits {required}"
                )));
            }
        }
        let expert_group = parameters
            .groups()
            .iter()
            .find(|owned| {
                owned.group().logical_name() == format!("{prefix}.mlp.experts.intermediate")
            })
            .expect("required expert group was checked above");
        if expert_group
            .group()
            .members()
            .iter()
            .any(|member| member.global_shape().first().copied() != Some(experts))
        {
            return Err(Error::backend(format!(
                "routed Qwen unit {layer} expert axis differs from {experts}"
            )));
        }
    }
    Ok(())
}

/// Declares Qwen cache identity independently of concrete cache storage.
pub fn state_identity(
    args: &ModelArgs,
    layout: &StateLayout,
    global_layer_start: usize,
    topology: PromptCacheTopology,
) -> Result<ModelStateIdentity, Error> {
    crate::decoder::state_identity(args, layout, global_layer_start, topology)
}

#[cfg(test)]
mod tests {
    use super::*;

    // This generic body is a compile-time contract: none of these dense Qwen
    // entry points may acquire a GroupedNeuralBackend bound.
    #[allow(dead_code)]
    fn dense_qwen_accepts_any_neural_backend<B: NeuralBackend>(
        args: ModelArgs,
        context: &<B::Tensor as Tensor>::Context,
    ) {
        let _: Result<TransformerBlock<B>, Error> = new_block::<B>(&args, 0, context);
        let _: Result<LayeredModel<B>, Error> = LayeredModel::<B>::new(args, context);
    }

    #[test]
    fn dense_qwen_api_has_no_routed_backend_bound() {
        // The generic helper above is type-checked even though no concrete
        // backend is needed to exercise this compile-time assertion.
    }
}
