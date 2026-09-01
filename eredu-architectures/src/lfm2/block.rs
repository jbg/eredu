//! LFM2 normalization, residual, and token-mixing assembly.

use eredu_core::cache::StateTensorRole;
use eredu_nn::{
    AttentionCache, CausalDepthwiseConvolutionSpec, ConvolutionActivation, Error,
    GatedShortConvolution, GatedShortConvolutionSpec, GroupedNeuralBackend, LinearSpec,
    NormalizationConstructionSpec, NormalizationOperator, ParameterSpec, Parameterized, RotarySpec,
    Tensor,
};
use eredu_runtime::RuntimeStateComponents;

use crate::decoder::{
    Attention, AttentionInput, FeedForwardOperator, TensorParallelFeedForwardOperator,
};

use super::{FeedForward, ModelArgs, OperatorPolicy};

/// Scheduled LFM2 token mixer.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub enum TokenMixer<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> {
    /// Grouped-query self attention.
    Attention(Attention<B>),
    /// Gated causal short convolution.
    ShortConvolution(GatedShortConvolution<B>),
}

/// One exact LFM2 decoder block.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct Block<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> {
    /// Scheduled token mixer.
    pub mixer: TokenMixer<B>,
    /// Scheduled dense or routed feed-forward operator.
    pub feed_forward: FeedForward<B>,
    /// Token-mixer pre-normalization.
    pub operator_norm: B::Normalization,
    /// Feed-forward pre-normalization.
    pub feed_forward_norm: B::Normalization,
}

/// Rank-local operator widths resolved by semantic placement.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct BlockGeometry {
    /// Query heads owned by this rank.
    pub query_heads: i32,
    /// Key/value heads owned by this rank.
    pub key_value_heads: i32,
    /// Short-convolution channels owned by this rank.
    pub convolution_channels: i32,
    /// Dense SwiGLU intermediate channels owned by this rank.
    pub dense_intermediate: i32,
    /// Routed-expert intermediate channels owned by this rank.
    pub expert_intermediate: i32,
}

impl BlockGeometry {
    /// Returns global replicated geometry.
    pub const fn replicated(args: &ModelArgs) -> Self {
        Self {
            query_heads: args.num_attention_heads,
            key_value_heads: args.num_key_value_heads,
            convolution_channels: args.hidden_size,
            dense_intermediate: args.dense_intermediate_size,
            expert_intermediate: args.moe_intermediate_size,
        }
    }
}

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> Block<B> {
    /// Builds one unloaded block from the normalized physical schedule.
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
            .ok_or_else(|| Error::backend(format!("LFM2 has no layer {layer}")))?;
        let root = format!("model.layers.{layer}");
        let mixer = match policy.operator {
            OperatorPolicy::CausalConvolution => {
                TokenMixer::ShortConvolution(GatedShortConvolution::new(
                    short_convolution_spec(args, &root, geometry.convolution_channels)?,
                    context,
                )?)
            }
            OperatorPolicy::SelfAttention(attention) => {
                let head_dim = args.hidden_size / args.num_attention_heads;
                let prefix = format!("{root}.self_attn");
                let linear = |field: &str, input, output| {
                    let name = format!("{prefix}.{field}.weight");
                    B::linear(
                        LinearSpec {
                            input,
                            output,
                            weight: ParameterSpec::trainable(&name).map_err(Error::backend)?,
                            bias: None,
                            format: crate::linear_format::standard_linear_format(
                                &name,
                                args.weight_quantization_for(&name).into(),
                            )?,
                        },
                        context,
                    )
                };
                let norm = |field: &str| {
                    B::normalization(
                        NormalizationConstructionSpec::learned(
                            head_dim,
                            args.norm_eps,
                            ParameterSpec::trainable(format!("{prefix}.{field}.weight"))
                                .map_err(Error::backend)?,
                        ),
                        context,
                    )
                };
                TokenMixer::Attention(Attention::from_parts(
                    geometry.query_heads,
                    geometry.key_value_heads,
                    head_dim,
                    linear("q_proj", args.hidden_size, geometry.query_heads * head_dim)?,
                    linear(
                        "k_proj",
                        args.hidden_size,
                        geometry.key_value_heads * head_dim,
                    )?,
                    linear(
                        "v_proj",
                        args.hidden_size,
                        geometry.key_value_heads * head_dim,
                    )?,
                    linear(
                        "out_proj",
                        geometry.query_heads * head_dim,
                        args.hidden_size,
                    )?,
                    Some(norm("q_layernorm")?),
                    Some(norm("k_layernorm")?),
                    Some(B::rotary(
                        RotarySpec {
                            dimensions: head_dim,
                            base: args.rope.theta,
                            traditional: false,
                            algorithm: eredu_nn::RotaryAlgorithm::Default,
                        },
                        context,
                    )?),
                    attention.sliding_window_i32().map_err(Error::backend)?,
                )?)
            }
        };
        let normalization = |field: &str| {
            B::normalization(
                NormalizationConstructionSpec::learned(
                    args.hidden_size,
                    args.norm_eps,
                    ParameterSpec::trainable(format!("{root}.{field}.weight"))
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
                geometry.expert_intermediate,
                context,
            )?,
            operator_norm: normalization("operator_norm")?,
            feed_forward_norm: normalization("ffn_norm")?,
        })
    }

    /// Executes one replicated heterogeneous block.
    pub fn forward<C>(
        &mut self,
        hidden: &B::Tensor,
        mask: Option<&B::Tensor>,
        state: &mut C,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        C: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
    {
        self.forward_with_feed_forward(
            hidden,
            mask,
            state,
            context,
            |policy, normalized, context| policy.forward_feed_forward(normalized, context),
        )
    }

    /// Executes the block while delegating its feed-forward policy.
    pub fn forward_with_feed_forward<C, F>(
        &mut self,
        hidden: &B::Tensor,
        mask: Option<&B::Tensor>,
        state: &mut C,
        context: &<B::Tensor as Tensor>::Context,
        feed_forward: F,
    ) -> Result<B::Tensor, Error>
    where
        C: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
        F: FnOnce(
            &mut FeedForward<B>,
            &B::Tensor,
            &<B::Tensor as Tensor>::Context,
        ) -> Result<B::Tensor, Error>,
    {
        let normalized = self.operator_norm.forward(hidden, context)?;
        let mixed = match &mut self.mixer {
            TokenMixer::Attention(attention) => attention.forward(
                AttentionInput {
                    hidden: &normalized,
                    mask,
                    cache: Some(&mut *state),
                    allow_sliding_prefill: true,
                    rotary_position: None,
                },
                context,
            )?,
            TokenMixer::ShortConvolution(convolution) => {
                let role = StateTensorRole::Convolution { slot: 0 };
                let result = {
                    let history = state.fixed_component(role).map_err(Error::backend)?;
                    convolution.forward(&normalized, history.as_ref(), context)?
                };
                *state.fixed_component(role).map_err(Error::backend)? = result.history;
                state.advance_fixed(hidden.dim(1)).map_err(Error::backend)?;
                result.output
            }
        };
        let hidden = hidden.add(&mixed, context)?;
        let normalized = self.feed_forward_norm.forward(&hidden, context)?;
        let feed_forward = feed_forward(&mut self.feed_forward, &normalized, context)?;
        hidden.add(&feed_forward, context)
    }

    /// Executes the same block under tensor-parallel placement.
    pub fn forward_parallel<C>(
        &mut self,
        hidden: &B::Tensor,
        mask: Option<&B::Tensor>,
        state: &mut C,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        C: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
        B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    {
        let normalized = self.operator_norm.forward(hidden, context)?;
        let mixed = match &mut self.mixer {
            TokenMixer::Attention(attention) => attention.forward_parallel(
                AttentionInput {
                    hidden: &normalized,
                    mask,
                    cache: Some(&mut *state),
                    allow_sliding_prefill: true,
                    rotary_position: None,
                },
                parallel,
                context,
            )?,
            TokenMixer::ShortConvolution(convolution) => {
                let role = StateTensorRole::Convolution { slot: 0 };
                let result = {
                    let history = state.fixed_component(role).map_err(Error::backend)?;
                    convolution.forward_parallel(
                        &normalized,
                        history.as_ref(),
                        parallel,
                        context,
                    )?
                };
                *state.fixed_component(role).map_err(Error::backend)? = result.history;
                state.advance_fixed(hidden.dim(1)).map_err(Error::backend)?;
                result.output
            }
        };
        let hidden = hidden.add(&mixed, context)?;
        let normalized = self.feed_forward_norm.forward(&hidden, context)?;
        let feed_forward =
            self.feed_forward
                .forward_feed_forward_parallel(&normalized, parallel, context)?;
        hidden.add(&feed_forward, context)
    }

    /// Executes tensor-partitioned token mixing while delegating feed-forward
    /// execution to a placement-aware caller.
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
        C: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
        F: FnOnce(
            &mut FeedForward<B>,
            &B::Tensor,
            &B::ParallelContext,
            &<B::Tensor as Tensor>::Context,
        ) -> Result<B::Tensor, Error>,
    {
        let normalized = self.operator_norm.forward(hidden, context)?;
        let mixed = match &mut self.mixer {
            TokenMixer::Attention(attention) => attention.forward_parallel(
                AttentionInput {
                    hidden: &normalized,
                    mask,
                    cache: Some(&mut *state),
                    allow_sliding_prefill: true,
                    rotary_position: None,
                },
                parallel,
                context,
            )?,
            TokenMixer::ShortConvolution(convolution) => {
                let role = StateTensorRole::Convolution { slot: 0 };
                let result = {
                    let history = state.fixed_component(role).map_err(Error::backend)?;
                    convolution.forward_parallel(
                        &normalized,
                        history.as_ref(),
                        parallel,
                        context,
                    )?
                };
                *state.fixed_component(role).map_err(Error::backend)? = result.history;
                state.advance_fixed(hidden.dim(1)).map_err(Error::backend)?;
                result.output
            }
        };
        let hidden = hidden.add(&mixed, context)?;
        let normalized = self.feed_forward_norm.forward(&hidden, context)?;
        let feed_forward = feed_forward(&mut self.feed_forward, &normalized, parallel, context)?;
        hidden.add(&feed_forward, context)
    }
}

fn short_convolution_spec(
    args: &ModelArgs,
    root: &str,
    channels: i32,
) -> Result<GatedShortConvolutionSpec, Error> {
    let prefix = format!("{root}.conv");
    let parameter = |name: String| ParameterSpec::trainable(name).map_err(Error::backend);
    let linear = |field: &str, input, output| {
        let weight_name = format!("{prefix}.{field}.weight");
        Ok(LinearSpec {
            input,
            output,
            weight: parameter(weight_name.clone())?,
            bias: args
                .conv_bias
                .then(|| parameter(format!("{prefix}.{field}.bias")))
                .transpose()?,
            format: crate::linear_format::standard_linear_format(
                &weight_name,
                args.weight_quantization_for(&weight_name).into(),
            )?,
        })
    };
    Ok(GatedShortConvolutionSpec {
        input_dimensions: args.hidden_size,
        channels,
        output_dimensions: args.hidden_size,
        input_projection: linear("in_proj", args.hidden_size, 3 * channels)?,
        output_projection: linear("out_proj", channels, args.hidden_size)?,
        convolution: CausalDepthwiseConvolutionSpec {
            channels,
            kernel_size: args.conv_l_cache,
            weight: parameter(format!("{prefix}.conv.weight"))?,
            bias: args
                .conv_bias
                .then(|| parameter(format!("{prefix}.conv.bias")))
                .transpose()?,
            activation: ConvolutionActivation::Identity,
        },
    })
}
