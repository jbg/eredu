//! Kimi Linear normalization, residual, and heterogeneous mixer assembly.

use crate::decoder::ComponentInstrumentation;
use eredu_nn::{
    BlockwiseAttentionBackend, CompressedAttentionCache, Error, GroupedNeuralBackend,
    NeuralBackend, NormalizationConstructionSpec, NormalizationOperator, ParameterSpec,
    Parameterized, Tensor,
};
use eredu_runtime::{RoutedExpertProvider, RuntimeStateComponents};

use super::{AttentionKind, FeedForward, KimiDeltaAttention, KimiLatentAttention, ModelArgs};

/// Scheduled KDA or no-positional MLA token mixer.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub enum TokenMixer<B>
where
    B: NeuralBackend + BlockwiseAttentionBackend,
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
    B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend + BlockwiseAttentionBackend,
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

/// Dense-only Kimi Linear unit used by replicated text composition.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub(crate) struct ReplicatedBlock<B: NeuralBackend + BlockwiseAttentionBackend> {
    mixer: TokenMixer<B>,
    feed_forward: super::moe::DenseSwiGlu<B>,
    input_norm: B::Normalization,
    post_attention_norm: B::Normalization,
}

#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub(crate) struct KdaReplicatedBlock<B: NeuralBackend> {
    mixer: KimiDeltaAttention<B>,
    feed_forward: super::moe::DenseSwiGlu<B>,
    input_norm: B::Normalization,
    post_attention_norm: B::Normalization,
}

impl<B: NeuralBackend> KdaReplicatedBlock<B> {
    pub(crate) fn new(
        args: &ModelArgs,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let policy = args
            .layer_policy(layer)
            .ok_or_else(|| Error::backend(format!("Kimi Linear has no layer {layer}")))?;
        if policy.feed_forward != super::FeedForwardPolicy::Dense
            || policy.attention != AttentionKind::Kda
        {
            return Err(Error::backend(
                "fixed-state Kimi unit requires dense KDA policy",
            ));
        }
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
            mixer: KimiDeltaAttention::new(args, layer, context)?,
            feed_forward: super::moe::DenseSwiGlu::new(
                args,
                &format!("model.layers.{layer}.mlp"),
                args.intermediate_size,
                context,
            )?,
            input_norm: norm("input_layernorm")?,
            post_attention_norm: norm("post_attention_layernorm")?,
        })
    }

    pub(crate) fn forward<C: RuntimeStateComponents<B>>(
        &mut self,
        hidden: &B::Tensor,
        state: &mut C,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.forward_instrumented(
            hidden,
            state,
            context,
            &mut ComponentInstrumentation::disabled(),
        )
    }

    pub(crate) fn forward_instrumented<C: RuntimeStateComponents<B>>(
        &mut self,
        hidden: &B::Tensor,
        state: &mut C,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error> {
        let normalized = self.input_norm.forward(hidden, context)?;
        let normalized = instrumentation.apply("attention.input", normalized)?;
        let mixed =
            self.mixer
                .forward_instrumented(&normalized, state, None, context, instrumentation)?;
        let hidden = finish_mixer::<B>(hidden, mixed, context, instrumentation)?;
        let normalized = self.post_attention_norm.forward(&hidden, context)?;
        let normalized = instrumentation.apply("feed_forward.input", normalized)?;
        let output =
            self.feed_forward
                .forward_instrumented(&normalized, None, context, instrumentation)?;
        finish_feed_forward::<B>(&hidden, output, false, context, instrumentation)
    }
}

impl<B: NeuralBackend + BlockwiseAttentionBackend> ReplicatedBlock<B> {
    pub(crate) fn new(
        args: &ModelArgs,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let policy = args
            .layer_policy(layer)
            .ok_or_else(|| Error::backend(format!("Kimi Linear has no layer {layer}")))?;
        if policy.feed_forward != super::FeedForwardPolicy::Dense {
            return Err(Error::backend(
                "replicated Kimi Linear unit requires a dense feed-forward policy",
            ));
        }
        let mixer = match policy.attention {
            AttentionKind::Kda => TokenMixer::Kda(KimiDeltaAttention::new(args, layer, context)?),
            AttentionKind::Mla => TokenMixer::Mla(KimiLatentAttention::new(args, layer, context)?),
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
            feed_forward: super::moe::DenseSwiGlu::new(
                args,
                &format!("model.layers.{layer}.mlp"),
                args.intermediate_size,
                context,
            )?,
            input_norm: norm("input_layernorm")?,
            post_attention_norm: norm("post_attention_layernorm")?,
        })
    }

    pub(crate) fn forward<C>(
        &mut self,
        hidden: &B::Tensor,
        mask: Option<&B::Tensor>,
        state: &mut C,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        C: RuntimeStateComponents<B> + CompressedAttentionCache<B::Tensor>,
    {
        self.forward_instrumented(
            hidden,
            mask,
            state,
            context,
            &mut ComponentInstrumentation::disabled(),
        )
    }

    pub(crate) fn forward_instrumented<C>(
        &mut self,
        hidden: &B::Tensor,
        mask: Option<&B::Tensor>,
        state: &mut C,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error>
    where
        C: RuntimeStateComponents<B> + CompressedAttentionCache<B::Tensor>,
    {
        let normalized = self.input_norm.forward(hidden, context)?;
        let normalized = instrumentation.apply("attention.input", normalized)?;
        let mixed = forward_mixer::<B, C>(
            &mut self.mixer,
            &normalized,
            mask,
            state,
            None,
            context,
            instrumentation,
        )?;
        let hidden = finish_mixer::<B>(hidden, mixed, context, instrumentation)?;
        let normalized = self.post_attention_norm.forward(&hidden, context)?;
        let normalized = instrumentation.apply("feed_forward.input", normalized)?;
        let output =
            self.feed_forward
                .forward_instrumented(&normalized, None, context, instrumentation)?;
        finish_feed_forward::<B>(&hidden, output, false, context, instrumentation)
    }
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
    B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend + BlockwiseAttentionBackend,
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
        Self::new_with_geometry_and_routed_spec(args, layer, geometry, None, context)
    }

    pub(crate) fn new_with_geometry_and_routed_spec(
        args: &ModelArgs,
        layer: usize,
        geometry: BlockGeometry,
        routed_spec: Option<eredu_nn::GroupedGatedProductSpec>,
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
            feed_forward: FeedForward::new_with_geometry_and_routed_spec(
                args,
                layer,
                geometry.dense_intermediate,
                geometry.routed_intermediate,
                geometry.shared_intermediate,
                routed_spec,
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
            crate::decoder::DecoderProjectionOperator::forward_feed_forward(policy, input, context)
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
        self.forward_partition_instrumented_with_feed_forward(
            hidden,
            mask,
            state,
            None,
            context,
            &mut ComponentInstrumentation::disabled(),
            |policy, input, context, _| feed_forward(policy, input, context),
        )
    }

    /// Runs the same residual equations for ordinary and observed local partitions.
    pub fn forward_partition_instrumented_with_feed_forward<C, F>(
        &mut self,
        hidden: &B::Tensor,
        mask: Option<&B::Tensor>,
        state: &mut C,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
        feed_forward: F,
    ) -> Result<B::Tensor, Error>
    where
        C: RuntimeStateComponents<B> + CompressedAttentionCache<B::Tensor>,
        F: FnOnce(
            &mut FeedForward<B>,
            &B::Tensor,
            &<B::Tensor as Tensor>::Context,
            &mut ComponentInstrumentation<'_, B::Tensor>,
        ) -> Result<B::Tensor, Error>,
    {
        let normalized = self.input_norm.forward(hidden, context)?;
        let normalized = instrumentation.apply("attention.input", normalized)?;
        let mixed = forward_mixer::<B, C>(
            &mut self.mixer,
            &normalized,
            mask,
            state,
            parallel,
            context,
            instrumentation,
        )?;
        let hidden = finish_mixer::<B>(hidden, mixed, context, instrumentation)?;
        let normalized = self.post_attention_norm.forward(&hidden, context)?;
        let normalized = instrumentation.apply("feed_forward.input", normalized)?;
        let output = feed_forward(
            &mut self.feed_forward,
            &normalized,
            context,
            instrumentation,
        )?;
        finish_feed_forward::<B>(
            &hidden,
            output,
            matches!(self.feed_forward, FeedForward::Sparse(_)),
            context,
            instrumentation,
        )
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
        B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    {
        self.forward_parallel_with_feed_forward(
            hidden,
            mask,
            state,
            parallel,
            context,
            |policy, input, parallel, context| {
                crate::decoder::TensorParallelProjectionOperator::forward_feed_forward_parallel(
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
        self.forward_partition_instrumented_with_feed_forward(
            hidden,
            mask,
            state,
            Some(parallel),
            context,
            &mut ComponentInstrumentation::disabled(),
            |policy, input, context, _| feed_forward(policy, input, parallel, context),
        )
    }
}

fn forward_mixer<B, C>(
    mixer: &mut TokenMixer<B>,
    input: &B::Tensor,
    mask: Option<&B::Tensor>,
    state: &mut C,
    parallel: Option<&B::ParallelContext>,
    context: &<B::Tensor as Tensor>::Context,
    instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
) -> Result<B::Tensor, Error>
where
    B: BlockwiseAttentionBackend,
    C: RuntimeStateComponents<B> + CompressedAttentionCache<B::Tensor>,
{
    match mixer {
        TokenMixer::Kda(mixer) => {
            mixer.forward_instrumented(input, state, parallel, context, instrumentation)
        }
        TokenMixer::Mla(mixer) => {
            mixer.forward_instrumented(input, mask, Some(state), parallel, context, instrumentation)
        }
    }
}

fn finish_mixer<B: NeuralBackend>(
    hidden: &B::Tensor,
    output: B::Tensor,
    context: &<B::Tensor as Tensor>::Context,
    instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
) -> Result<B::Tensor, Error> {
    let write = instrumentation.apply("attention.write", output)?;
    let output = instrumentation.apply("attention.output", write)?;
    instrumentation.apply("attention.residual", hidden.add(&output, context)?)
}

fn finish_feed_forward<B: NeuralBackend>(
    hidden: &B::Tensor,
    output: B::Tensor,
    sparse: bool,
    context: &<B::Tensor as Tensor>::Context,
    instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
) -> Result<B::Tensor, Error> {
    let output = if sparse {
        instrumentation.apply("feed_forward.contribution", output)?
    } else {
        let write = instrumentation.apply("feed_forward.write", output)?;
        instrumentation.apply("feed_forward.output", write)?
    };
    instrumentation.apply("feed_forward.residual", hidden.add(&output, context)?)
}
