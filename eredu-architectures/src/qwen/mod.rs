//! Backend-neutral Qwen2, Qwen3, and Qwen3-MoE text architecture policy.

mod checkpoint;
mod config;
pub mod hybrid;
pub mod ingress;
mod moe;
mod parallel;
pub(crate) use parallel::parameter_description;
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
use eredu_nn::{Error, GroupedNeuralBackend, NeuralBackend, Tensor};
use eredu_runtime::{ModelStateIdentity, StateLayout};

/// Dense Qwen decoder block.
pub type TransformerBlock<B> = crate::decoder::TransformerBlock<B>;

/// Qwen decoder block selected dynamically between dense and routed feed-forward policy.
pub type RoutedTransformerBlock<B> = crate::decoder::TransformerBlock<B, FeedForward<B>>;

/// Statically dispatched dense Qwen block construction policy.
pub struct QwenBlockFactory;

/// Builds one unloaded dense Qwen decoder block.
pub fn new_block<B: NeuralBackend>(
    args: &ModelArgs,
    layer: usize,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<TransformerBlock<B>, Error> {
    TransformerBlock::from_factory::<_, QwenBlockFactory>(args, args, layer, context)
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

    fn build_feed_forward(
        args: &ModelArgs,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::FeedForward, Error> {
        Mlp::new(args, layer, context)
    }

    fn parameter_description(
        args: &ModelArgs,
    ) -> Result<eredu_runtime::ArchitectureParameterDescription, eredu_runtime::ParallelPlanError>
    {
        crate::decoder::dense_parameter_description(args)
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

    fn partition_parameter_description(
        &self,
    ) -> Result<eredu_runtime::ArchitectureParameterDescription, eredu_runtime::ParallelPlanError>
    {
        parallel::parameter_description(self)
    }
}

/// Statically dispatched Qwen policy for adapters that admit dense and MoE configurations.
pub struct RoutedQwenBlockFactory;

impl<B> crate::decoder::BlockFactory<B, ModelArgs> for RoutedQwenBlockFactory
where
    B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
{
    type FeedForward = FeedForward<B>;

    fn validate_partitioned(global: &ModelArgs, local: &ModelArgs) -> Result<(), Error> {
        if global.is_moe() != local.is_moe() {
            return Err(Error::backend("global and local Qwen equations differ"));
        }
        Ok(())
    }

    fn build_feed_forward(
        args: &ModelArgs,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::FeedForward, Error> {
        FeedForward::new(args, layer, context)
    }

    fn build_partitioned_feed_forward(
        global: &ModelArgs,
        local: &ModelArgs,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::FeedForward, Error> {
        if !global.is_moe() {
            return Self::build_feed_forward(local, layer, context);
        }
        FeedForward::new_partitioned(global, local, layer, context)
    }

    fn parameter_description(
        args: &ModelArgs,
    ) -> Result<eredu_runtime::ArchitectureParameterDescription, eredu_runtime::ParallelPlanError>
    {
        parallel::parameter_description(args)
    }
}

/// Builds a Qwen block for a backend adapter that dynamically admits dense or MoE configuration.
pub fn new_routed_block<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend>(
    args: &ModelArgs,
    layer: usize,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<RoutedTransformerBlock<B>, Error> {
    RoutedTransformerBlock::from_factory::<_, RoutedQwenBlockFactory>(args, args, layer, context)
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
    RoutedTransformerBlock::from_factory::<_, RoutedQwenBlockFactory>(global, local, layer, context)
}

/// Layered Qwen lifecycle for adapters that dynamically admit dense and MoE configurations.
pub type RoutedLayeredModel<B> = crate::decoder::LayeredModel<B, ModelArgs, RoutedQwenBlockFactory>;

/// Pipeline-local Qwen3-MoE model with global routing and local expert banks.
pub type PartitionedRoutedLayeredModel<B> =
    crate::decoder::PartitionedLayeredModel<B, ModelArgs, RoutedQwenBlockFactory>;

/// Declares Qwen cache identity independently of concrete cache storage.
pub fn state_identity(
    args: &ModelArgs,
    layout: &StateLayout,
    global_layer_start: usize,
    topology: PromptCacheTopology,
) -> Result<ModelStateIdentity, Error> {
    crate::decoder::state_identity(args, layout, global_layer_start, topology)
}

pub(crate) use moe::execution_topology;
