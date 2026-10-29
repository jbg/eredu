//! Shared pre-normalized residual block specialized to GPT-OSS routed experts.

use eredu_nn::{AttentionCache, Error, GroupedNeuralBackend, Tensor};
use eredu_runtime::{ExpertPass, ParameterProvider};

use crate::decoder::{AttentionInput, BlockFactory};

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

    fn build_feed_forward(
        args: &ModelArgs,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::FeedForward, Error> {
        RoutedMlp::new(args, layer, context)
    }

    fn build_partitioned_feed_forward(
        global: &ModelArgs,
        local: &ModelArgs,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::FeedForward, Error> {
        RoutedMlp::new_partitioned(global, local, layer, context)
    }

    fn parameter_description(
        args: &ModelArgs,
    ) -> Result<eredu_runtime::ArchitectureParameterDescription, eredu_runtime::ParallelPlanError>
    {
        super::parallel::parameter_description(args)
    }
}

/// Builds one unloaded global GPT-OSS decoder layer.
pub fn new_block<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend>(
    args: &ModelArgs,
    layer: usize,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<TransformerBlock<B>, Error> {
    TransformerBlock::from_factory::<_, GptOssBlockFactory>(args, args, layer, context)
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
    P: ParameterProvider<B>,
    P::Error: std::fmt::Display,
{
    block.body.forward_with_feed_forward(
        block.output_norm.as_mut(),
        input,
        context,
        |mlp, hidden, context| mlp.forward_with_provider(hidden, pass, provider, context),
    )
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
    P: eredu_runtime::TensorParallelParameterProvider<B>,
    P::Error: std::fmt::Display,
{
    block.body.forward_tensor_parallel_with_feed_forward(
        block.output_norm.as_mut(),
        input,
        parallel,
        context,
        |mlp, hidden, context| {
            mlp.forward_parallel_with_provider(hidden, pass, parallel, provider, context)
        },
    )
}
