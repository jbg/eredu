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

    type Parameters = eredu_nn::RoutedGatedProductSpec;
    fn prepare_parameters(
        _global: &ModelArgs,
        local: &ModelArgs,
        _layer: usize,
        dense: Option<&eredu_nn::GatedProductSpec>,
        retained: Option<&Self::Parameters>,
    ) -> Result<Self::Parameters, Error> {
        if dense.is_some() {
            return Err(Error::backend(
                "GPT-OSS cannot consume dense feed-forward slots",
            ));
        }
        super::moe::local_spec(
            local,
            retained.ok_or_else(|| {
                Error::backend("GPT-OSS requires retained projection declarations")
            })?,
        )
    }
    fn parameter_description(
        config: &ModelArgs,
        common: &crate::decoder::DecoderDeclaration,
        projections: Option<&crate::decoder::ProjectionDeclarations<Self::Parameters>>,
    ) -> Result<eredu_runtime::ArchitectureParameterDescription, Error> {
        let projections = projections
            .ok_or_else(|| Error::backend("GPT-OSS requires retained projection declarations"))?;
        let declaration = super::Declaration::from_scopes(
            config.clone(),
            std::sync::Arc::new(common.clone()),
            projections.clone(),
        )?;
        super::parallel::parameter_description(
            &declaration,
            crate::decoder::ParameterEncoding::Stored,
        )
        .map_err(Error::backend_source)
    }
    fn build_feed_forward(
        _global: &ModelArgs,
        _local: &ModelArgs,
        layer: usize,
        parameters: &Self::Parameters,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::FeedForward, Error> {
        RoutedMlp::from_spec(layer, parameters.clone(), context)
    }
}

/// Builds one unloaded global GPT-OSS decoder layer.
pub fn new_block<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend>(
    args: &ModelArgs,
    layer: usize,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<TransformerBlock<B>, Error> {
    TransformerBlock::from_factory::<_, GptOssBlockFactory>(
        args,
        args,
        layer,
        &crate::decoder::topology::BlockParameters::with_feed_forward(args, layer, None)?,
        context,
    )
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
