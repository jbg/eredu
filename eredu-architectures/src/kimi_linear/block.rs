//! Kimi Linear normalization, residual, and heterogeneous mixer assembly.

use eredu_nn::{
    BlockwiseAttentionBackend, CompressedAttentionCache, Error, GroupedNeuralBackend,
    NormalizationConstructionSpec, NormalizationOperator, ParameterSpec, Parameterized, Tensor,
};
use eredu_runtime::{RoutedExpertProvider, RuntimeStateComponents};

use super::{AttentionKind, FeedForward, KimiDeltaAttention, KimiLatentAttention, ModelArgs};

/// Scheduled KDA or no-positional MLA token mixer.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub enum TokenMixer<B>
where
    B: GroupedNeuralBackend + BlockwiseAttentionBackend,
{
    /// Kimi Delta Attention.
    Kda(KimiDeltaAttention<B>),
    /// Compressed no-positional multi-head latent attention.
    Mla(KimiLatentAttention<B>),
}

/// One exact Kimi Linear decoder block.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct Block<B>
where
    B: GroupedNeuralBackend + BlockwiseAttentionBackend,
{
    /// Scheduled heterogeneous token mixer.
    pub mixer: TokenMixer<B>,
    /// Scheduled dense-prefix or sparse feed-forward operator.
    pub feed_forward: FeedForward<B>,
    /// Token-mixer pre-normalization.
    pub input_norm: B::Normalization,
    /// Feed-forward pre-normalization.
    pub post_attention_norm: B::Normalization,
}

/// Placement-resolved local widths for one Kimi block.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct BlockGeometry {
    /// KDA heads owned by this rank.
    pub kda_heads: i32,
    /// MLA query heads owned by this rank.
    pub mla_heads: i32,
    /// Dense SwiGLU intermediate channels owned by this rank.
    pub dense_intermediate: i32,
    /// Routed expert intermediate channels owned by this rank.
    pub routed_intermediate: i32,
    /// Shared expert intermediate channels owned by this rank.
    pub shared_intermediate: i32,
}

impl BlockGeometry {
    /// Returns global replicated geometry.
    pub const fn replicated(args: &ModelArgs) -> Self {
        Self {
            kda_heads: args.kda_config.num_heads,
            mla_heads: args.num_attention_heads,
            dense_intermediate: args.intermediate_size,
            routed_intermediate: args.moe_intermediate_size,
            shared_intermediate: args.moe_intermediate_size * args.num_shared_experts,
        }
    }
}

impl<B> Block<B>
where
    B: GroupedNeuralBackend + BlockwiseAttentionBackend,
{
    /// Builds one unloaded block from the validated physical schedule.
    pub fn new(
        args: &ModelArgs,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        Self::new_with_geometry(args, layer, BlockGeometry::replicated(args), context)
    }

    /// Builds one unloaded block from placement-resolved local geometry.
    pub fn new_with_geometry(
        args: &ModelArgs,
        layer: usize,
        geometry: BlockGeometry,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let policy = args
            .layer_policy(layer)
            .ok_or_else(|| Error::backend(format!("Kimi Linear has no layer {layer}")))?;
        let mixer = match policy.attention {
            AttentionKind::Kda => TokenMixer::Kda(KimiDeltaAttention::new_with_heads(
                args,
                layer,
                geometry.kda_heads,
                context,
            )?),
            AttentionKind::Mla => TokenMixer::Mla(KimiLatentAttention::new_with_heads(
                args,
                layer,
                geometry.mla_heads,
                context,
            )?),
        };
        let norm = |field: &str| {
            B::normalization(
                NormalizationConstructionSpec::learned(
                    args.hidden_size,
                    args.rms_norm_eps,
                    ParameterSpec::trainable(format!("model.layers.{layer}.{field}.weight"))
                        .map_err(Error::backend)?,
                ),
                context,
            )
        };
        Ok(Self {
            mixer,
            feed_forward: FeedForward::new_with_geometry(
                args,
                layer,
                geometry.dense_intermediate,
                geometry.routed_intermediate,
                geometry.shared_intermediate,
                context,
            )?,
            input_norm: norm("input_layernorm")?,
            post_attention_norm: norm("post_attention_layernorm")?,
        })
    }

    /// Executes one block with resident experts.
    pub fn forward<C>(
        &mut self,
        hidden: &B::Tensor,
        mask: Option<&B::Tensor>,
        state: &mut C,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        C: RuntimeStateComponents<B> + CompressedAttentionCache<B::Tensor>,
    {
        self.forward_with_feed_forward(hidden, mask, state, context, |policy, input, context| {
            crate::decoder::FeedForwardOperator::forward_feed_forward(policy, input, context)
        })
    }

    /// Executes one block while delegating routed experts to the runtime.
    pub fn forward_with_provider<C, P>(
        &mut self,
        hidden: &B::Tensor,
        mask: Option<&B::Tensor>,
        state: &mut C,
        context: &<B::Tensor as Tensor>::Context,
        provider: &mut P,
    ) -> Result<B::Tensor, Error>
    where
        C: RuntimeStateComponents<B> + CompressedAttentionCache<B::Tensor>,
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        let pass = if hidden.dim(1) > 1 {
            eredu_runtime::ExpertPass::Prefill
        } else {
            eredu_runtime::ExpertPass::Decode
        };
        self.forward_with_feed_forward(hidden, mask, state, context, |policy, input, context| {
            policy.forward_with_provider(input, pass, context, provider)
        })
    }

    /// Executes a block while delegating the scheduled feed-forward policy.
    pub fn forward_with_feed_forward<C, F>(
        &mut self,
        hidden: &B::Tensor,
        mask: Option<&B::Tensor>,
        state: &mut C,
        context: &<B::Tensor as Tensor>::Context,
        feed_forward: F,
    ) -> Result<B::Tensor, Error>
    where
        C: RuntimeStateComponents<B> + CompressedAttentionCache<B::Tensor>,
        F: FnOnce(
            &mut FeedForward<B>,
            &B::Tensor,
            &<B::Tensor as Tensor>::Context,
        ) -> Result<B::Tensor, Error>,
    {
        let normalized = self.input_norm.forward(hidden, context)?;
        let mixed = match &mut self.mixer {
            TokenMixer::Kda(mixer) => mixer.forward(&normalized, state, context)?,
            TokenMixer::Mla(mixer) => mixer.forward(&normalized, mask, Some(state), context)?,
        };
        let hidden = hidden.add(&mixed, context)?;
        let normalized = self.post_attention_norm.forward(&hidden, context)?;
        let feed_forward = feed_forward(&mut self.feed_forward, &normalized, context)?;
        hidden.add(&feed_forward, context)
    }

    /// Executes tensor-partitioned token mixing and feed-forward projections.
    pub fn forward_parallel<C>(
        &mut self,
        hidden: &B::Tensor,
        mask: Option<&B::Tensor>,
        state: &mut C,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        C: RuntimeStateComponents<B> + CompressedAttentionCache<B::Tensor>,
    {
        self.forward_parallel_with_feed_forward(
            hidden,
            mask,
            state,
            parallel,
            context,
            |policy, input, parallel, context| {
                crate::decoder::FeedForwardOperator::forward_feed_forward_parallel(
                    policy, input, parallel, context,
                )
            },
        )
    }

    /// Executes tensor-partitioned mixing while delegating feed-forward execution.
    pub fn forward_parallel_with_feed_forward<C, F>(
        &mut self,
        hidden: &B::Tensor,
        mask: Option<&B::Tensor>,
        state: &mut C,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        feed_forward: F,
    ) -> Result<B::Tensor, Error>
    where
        C: RuntimeStateComponents<B> + CompressedAttentionCache<B::Tensor>,
        F: FnOnce(
            &mut FeedForward<B>,
            &B::Tensor,
            &B::ParallelContext,
            &<B::Tensor as Tensor>::Context,
        ) -> Result<B::Tensor, Error>,
    {
        let normalized = self.input_norm.forward(hidden, context)?;
        let mixed = match &mut self.mixer {
            TokenMixer::Kda(mixer) => {
                mixer.forward_parallel(&normalized, state, parallel, context)?
            }
            TokenMixer::Mla(mixer) => {
                mixer.forward_parallel(&normalized, mask, Some(state), parallel, context)?
            }
        };
        let hidden = hidden.add(&mixed, context)?;
        let normalized = self.post_attention_norm.forward(&hidden, context)?;
        let output = feed_forward(&mut self.feed_forward, &normalized, parallel, context)?;
        hidden.add(&output, context)
    }
}
