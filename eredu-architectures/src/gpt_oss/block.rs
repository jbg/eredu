//! Shared pre-normalized residual block specialized to GPT-OSS routed experts.

use eredu_nn::{
    AttentionCache, Error, GroupedNeuralBackend, NormalizationConstructionSpec, ParameterSpec,
    Tensor,
};
use eredu_runtime::{ExpertPass, RoutedExpertProvider};

use crate::decoder::{Attention, AttentionInput, BlockFactory};

use super::{config::ModelArgs, moe::RoutedMlp};

/// One GPT-OSS RMS-pre-norm attention-plus-MoE residual block.
pub type TransformerBlock<B> = crate::decoder::TransformerBlock<B, RoutedMlp<B>>;

/// Statically dispatched GPT-OSS block construction policy.
pub struct GptOssBlockFactory;

impl<B> BlockFactory<B, ModelArgs> for GptOssBlockFactory
where
    B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
{
    type FeedForward = RoutedMlp<B>;

    fn build(
        args: &ModelArgs,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<TransformerBlock<B>, Error> {
        let prefix = format!("{}.layers.{layer}", args.parameter_root);
        Ok(crate::decoder::TransformerBlock {
            self_attention: Attention::new(args, layer, context)?,
            mlp: RoutedMlp::new(args, layer, context)?,
            input_norm: B::normalization(
                NormalizationConstructionSpec::learned(
                    args.hidden_size,
                    args.rms_norm_eps,
                    ParameterSpec::trainable(format!("{prefix}.input_layernorm.weight"))
                        .map_err(Error::backend)?,
                ),
                context,
            )?,
            post_attention_norm: B::normalization(
                NormalizationConstructionSpec::learned(
                    args.hidden_size,
                    args.rms_norm_eps,
                    ParameterSpec::trainable(format!("{prefix}.post_attention_layernorm.weight"))
                        .map_err(Error::backend)?,
                ),
                context,
            )?,
        })
    }

    fn parameter_groups(
        block: &crate::decoder::TransformerBlock<B, Self::FeedForward>,
        args: &ModelArgs,
        layer: usize,
    ) -> Result<Vec<eredu_runtime::ParameterGroupSpec>, eredu_runtime::ParallelPlanError> {
        super::parallel::layer_parallel_parameter_groups(block, args, layer)
    }
}

/// Builds one unloaded global GPT-OSS decoder layer.
pub fn new_block<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend>(
    args: &ModelArgs,
    layer: usize,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<TransformerBlock<B>, Error> {
    <GptOssBlockFactory as BlockFactory<B, ModelArgs>>::build(args, layer, context)
}

/// Executes a block with runtime-owned expert residency.
pub fn forward_with_provider<B, C, P>(
    block: &mut TransformerBlock<B>,
    input: AttentionInput<'_, B::Tensor, C>,
    pass: ExpertPass,
    provider: &mut P,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<B::Tensor, Error>
where
    B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    C: AttentionCache<B::Tensor>,
    P: RoutedExpertProvider<B>,
    P::Error: std::fmt::Display,
{
    block.forward_with_feed_forward(input, context, |mlp, hidden, context| {
        mlp.forward_with_provider(hidden, pass, provider, context)
    })
}

/// Executes tensor-parallel attention and provider-backed routed experts.
pub fn forward_parallel_with_provider<B, C, P>(
    block: &mut TransformerBlock<B>,
    input: AttentionInput<'_, B::Tensor, C>,
    pass: ExpertPass,
    parallel: &B::ParallelContext,
    provider: &mut P,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<B::Tensor, Error>
where
    B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    C: AttentionCache<B::Tensor>,
    P: eredu_runtime::TensorParallelRoutedExpertProvider<B>,
    P::Error: std::fmt::Display,
{
    block.forward_tensor_parallel_with_feed_forward(
        input,
        parallel,
        context,
        |mlp, hidden, context| {
            mlp.forward_parallel_with_provider(hidden, pass, parallel, provider, context)
        },
    )
}
