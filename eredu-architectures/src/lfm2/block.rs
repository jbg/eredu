//! LFM2 normalization, residual, and token-mixing assembly.

use eredu_core::cache::StateTensorRole;
use eredu_nn::{
    AttentionCache, CausalDepthwiseConvolutionSpec, ConvolutionActivation, Error,
    GatedShortConvolution, GatedShortConvolutionSpec, GroupedNeuralBackend, LinearSpec,
    NeuralBackend, NormalizationConstructionSpec, NormalizationOperator, ParameterSpec,
    Parameterized, RotarySpec, Tensor,
};
use eredu_runtime::RuntimeStateComponents;

use crate::decoder::{
    Attention, AttentionInput, ComponentInstrumentation, DecoderProjectionOperator,
    TensorParallelProjectionOperator,
};

use super::{FeedForward, ModelArgs, OperatorPolicy};

/// Scheduled LFM2 token mixer.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub enum TokenMixer<B: NeuralBackend> {
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

/// Dense-only LFM2 unit used by replicated text composition.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub(crate) struct ReplicatedBlock<B: NeuralBackend> {
    mixer: TokenMixer<B>,
    feed_forward: super::moe::DenseSwiGlu<B>,
    operator_norm: B::Normalization,
    feed_forward_norm: B::Normalization,
}

impl<B: NeuralBackend> ReplicatedBlock<B> {
    pub(crate) fn new(
        args: &ModelArgs,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let policy = args
            .layer_policy(layer)
            .ok_or_else(|| Error::backend(format!("LFM2 has no layer {layer}")))?;
        if policy.feed_forward != super::FeedForwardPolicy::Dense {
            return Err(Error::backend(
                "replicated LFM2 unit requires a dense feed-forward policy",
            ));
        }
        let root = format!("model.layers.{layer}");
        let mixer = match policy.operator {
            OperatorPolicy::CausalConvolution => {
                TokenMixer::ShortConvolution(GatedShortConvolution::new(
                    short_convolution_spec(args, &root, args.hidden_size)?,
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
                let mut mixer = Attention::from_parts(
                    args.num_attention_heads,
                    args.num_key_value_heads,
                    head_dim,
                    linear("q_proj", args.hidden_size, args.hidden_size)?,
                    linear(
                        "k_proj",
                        args.hidden_size,
                        args.num_key_value_heads * head_dim,
                    )?,
                    linear(
                        "v_proj",
                        args.hidden_size,
                        args.num_key_value_heads * head_dim,
                    )?,
                    linear("out_proj", args.hidden_size, args.hidden_size)?,
                    Some(norm("q_layernorm")?),
                    Some(norm("k_layernorm")?),
                    Some(B::rotary(
                        RotarySpec {
                            arithmetic: eredu_nn::RotaryArithmetic::InputProducts,
                            dimensions: head_dim,
                            base: args.rope.theta,
                            traditional: false,
                            algorithm: eredu_nn::RotaryAlgorithm::Default,
                        },
                        context,
                    )?),
                    attention.sliding_window_i32().map_err(Error::backend)?,
                )?;
                // Match the released equation's input-dtype score and
                // probability boundaries, including low-precision inference.
                mixer.arithmetic = eredu_nn::AttentionArithmetic::InputScores;
                TokenMixer::Attention(mixer)
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
            feed_forward: super::moe::DenseSwiGlu::new(
                args,
                layer,
                args.dense_intermediate_size,
                context,
            )?,
            operator_norm: normalization("operator_norm")?,
            feed_forward_norm: normalization("ffn_norm")?,
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
        C: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
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
        C: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
    {
        let hidden = forward_mixer::<B, C>(
            &mut self.mixer,
            &mut self.operator_norm,
            hidden,
            mask,
            state,
            None,
            context,
            instrumentation,
        )?;
        let normalized = self.feed_forward_norm.forward(&hidden, context)?;
        let normalized = instrumentation.apply("feed_forward.input", normalized)?;
        let feed_forward =
            self.feed_forward
                .forward_instrumented(&normalized, None, context, instrumentation)?;
        finish_feed_forward::<B>(
            &hidden,
            feed_forward,
            "feed_forward.output",
            context,
            instrumentation,
        )
    }
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
                let mut mixer = Attention::from_parts(
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
                            arithmetic: eredu_nn::RotaryArithmetic::InputProducts,
                            dimensions: head_dim,
                            base: args.rope.theta,
                            traditional: false,
                            algorithm: eredu_nn::RotaryAlgorithm::Default,
                        },
                        context,
                    )?),
                    attention.sliding_window_i32().map_err(Error::backend)?,
                )?;
                // Match the released equation's input-dtype score and
                // probability boundaries, including low-precision inference.
                mixer.arithmetic = eredu_nn::AttentionArithmetic::InputScores;
                TokenMixer::Attention(mixer)
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
            feed_forward: FeedForward::new_with_geometry_and_routed_spec(
                args,
                layer,
                geometry.dense_intermediate,
                geometry.expert_intermediate,
                routed_spec,
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
        self.forward_instrumented_with_feed_forward(
            hidden,
            mask,
            state,
            context,
            &mut ComponentInstrumentation::disabled(),
            |policy, normalized, context, instrumentation| {
                policy.forward_feed_forward_observed(normalized, context, instrumentation)
            },
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
        self.forward_instrumented_with_feed_forward(
            hidden,
            mask,
            state,
            context,
            &mut ComponentInstrumentation::disabled(),
            |policy, normalized, context, _| feed_forward(policy, normalized, context),
        )
    }

    /// Executes shared residual equations with actual component boundaries.
    pub fn forward_instrumented_with_feed_forward<C, F>(
        &mut self,
        hidden: &B::Tensor,
        mask: Option<&B::Tensor>,
        state: &mut C,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
        feed_forward: F,
    ) -> Result<B::Tensor, Error>
    where
        C: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
        F: FnOnce(
            &mut FeedForward<B>,
            &B::Tensor,
            &<B::Tensor as Tensor>::Context,
            &mut ComponentInstrumentation<'_, B::Tensor>,
        ) -> Result<B::Tensor, Error>,
    {
        self.forward_partition_instrumented_with_feed_forward(
            hidden,
            mask,
            state,
            None,
            context,
            instrumentation,
            feed_forward,
        )
    }

    /// Executes shared residual equations with actual component boundaries.
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
        C: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
        F: FnOnce(
            &mut FeedForward<B>,
            &B::Tensor,
            &<B::Tensor as Tensor>::Context,
            &mut ComponentInstrumentation<'_, B::Tensor>,
        ) -> Result<B::Tensor, Error>,
    {
        let hidden = forward_mixer::<B, C>(
            &mut self.mixer,
            &mut self.operator_norm,
            hidden,
            mask,
            state,
            parallel,
            context,
            instrumentation,
        )?;
        let normalized = self.feed_forward_norm.forward(&hidden, context)?;
        let normalized = instrumentation.apply("feed_forward.input", normalized)?;
        let feed_forward = feed_forward(
            &mut self.feed_forward,
            &normalized,
            context,
            instrumentation,
        )?;
        let output_path = if matches!(self.feed_forward, FeedForward::Routed(_)) {
            // The provider already owns feed_forward.output. This later boundary
            // is the complete contribution entering residual addition.
            "feed_forward.contribution"
        } else {
            "feed_forward.output"
        };
        finish_feed_forward::<B>(&hidden, feed_forward, output_path, context, instrumentation)
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
        self.forward_parallel_observed(
            hidden,
            mask,
            state,
            parallel,
            context,
            &mut ComponentInstrumentation::disabled(),
        )
    }

    /// Emits component values consumed by tensor-parallel projections.
    pub fn forward_parallel_observed<C>(
        &mut self,
        hidden: &B::Tensor,
        mask: Option<&B::Tensor>,
        state: &mut C,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error>
    where
        C: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
        B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    {
        self.forward_partition_instrumented_with_feed_forward(
            hidden,
            mask,
            state,
            Some(parallel),
            context,
            instrumentation,
            |policy, normalized, context, instrumentation| {
                policy.forward_feed_forward_parallel_observed(
                    normalized,
                    parallel,
                    context,
                    instrumentation,
                )
            },
        )
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
        self.forward_partition_instrumented_with_feed_forward(
            hidden,
            mask,
            state,
            Some(parallel),
            context,
            &mut ComponentInstrumentation::disabled(),
            |policy, normalized, context, _| feed_forward(policy, normalized, parallel, context),
        )
    }
}

/// Attention and convolution share residual ordering and state ownership; only
/// attention has scalar aggregated-channel declarations.
#[allow(clippy::too_many_arguments)]
fn forward_mixer<B: NeuralBackend, C>(
    mixer: &mut TokenMixer<B>,
    normalization: &mut B::Normalization,
    hidden: &B::Tensor,
    mask: Option<&B::Tensor>,
    state: &mut C,
    parallel: Option<&B::ParallelContext>,
    context: &<B::Tensor as Tensor>::Context,
    instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
) -> Result<B::Tensor, Error>
where
    C: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
{
    let attention = matches!(mixer, TokenMixer::Attention(_));
    let normalized = normalization.forward(hidden, context)?;
    let normalized = instrumentation.apply(
        if attention {
            "attention.input"
        } else {
            "mixer.input"
        },
        normalized,
    )?;
    let mixed = match mixer {
        TokenMixer::Attention(attention) => attention.forward_instrumented(
            AttentionInput {
                hidden: &normalized,
                mask,
                cache: Some(&mut *state),
                allow_sliding_prefill: true,
                rotary_position: None,
            },
            parallel,
            context,
            instrumentation,
        )?,
        TokenMixer::ShortConvolution(convolution) => {
            let role = StateTensorRole::Convolution { slot: 0 };
            let result = {
                let history = state.fixed_component(role).map_err(Error::backend)?;
                match parallel {
                    Some(parallel) => convolution.forward_parallel(
                        &normalized,
                        history.as_ref(),
                        parallel,
                        context,
                    )?,
                    None => convolution.forward(&normalized, history.as_ref(), context)?,
                }
            };
            *state.fixed_component(role).map_err(Error::backend)? = result.history;
            state.advance_fixed(hidden.dim(1)).map_err(Error::backend)?;
            result.output
        }
    };
    let mixed = instrumentation.apply(
        if attention {
            "attention.write"
        } else {
            "mixer.write"
        },
        mixed,
    )?;
    let mixed = instrumentation.apply(
        if attention {
            "attention.output"
        } else {
            "mixer.output"
        },
        mixed,
    )?;
    let hidden = hidden.add(&mixed, context)?;
    instrumentation.apply(
        if attention {
            "attention.residual"
        } else {
            "mixer.residual"
        },
        hidden,
    )
}

fn finish_feed_forward<B: NeuralBackend>(
    hidden: &B::Tensor,
    feed_forward: B::Tensor,
    output_path: &str,
    context: &<B::Tensor as Tensor>::Context,
    instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
) -> Result<B::Tensor, Error> {
    let feed_forward = instrumentation.apply("feed_forward.write", feed_forward)?;
    let feed_forward = instrumentation.apply(output_path, feed_forward)?;
    instrumentation.apply("feed_forward.residual", hidden.add(&feed_forward, context)?)
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
