//! Shared decoder-block construction and state-aware execution.

use crate::decoder::{AttentionInput, TransformerBlock};
use eredu_nn::{AttentionCache, Error, NeuralBackend, Tensor};
use eredu_runtime::LayerRuntimeState;

use super::MoshiTransformerConfig;

/// The sole transformer block implementation used by both model stacks.
pub type Block<B> = TransformerBlock<B>;

/// Builds one unloaded shared decoder block.
pub fn build<B: NeuralBackend + eredu_nn::DistributedNeuralBackend>(
    config: &MoshiTransformerConfig,
    layer: usize,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<Block<B>, Error> {
    TransformerBlock::new(config, layer, context)
}

/// Executes one shared block against its architecture-global state slot.
pub fn forward<B, S>(
    block: &mut Block<B>,
    state_ordinal: usize,
    hidden: &B::Tensor,
    mask: Option<&B::Tensor>,
    allow_sliding_prefill: bool,
    state: &mut S,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<B::Tensor, Error>
where
    B: NeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor>,
{
    let cache = state.layer(state_ordinal).map_err(Error::backend)?;
    block.forward(
        AttentionInput {
            hidden,
            mask,
            cache: Some(cache),
            allow_sliding_prefill,
            rotary_position: None,
        },
        context,
    )
}

/// Executes one rank-local shared block with backend collectives.
#[allow(clippy::too_many_arguments)]
pub fn forward_parallel<B, S>(
    block: &mut Block<B>,
    state_ordinal: usize,
    hidden: &B::Tensor,
    mask: Option<&B::Tensor>,
    allow_sliding_prefill: bool,
    state: &mut S,
    parallel: &B::ParallelContext,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<B::Tensor, Error>
where
    B: NeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor>,
{
    let cache = state.layer(state_ordinal).map_err(Error::backend)?;
    block.forward_tensor_parallel(
        AttentionInput {
            hidden,
            mask,
            cache: Some(cache),
            allow_sliding_prefill,
            rotary_position: None,
        },
        parallel,
        context,
    )
}
