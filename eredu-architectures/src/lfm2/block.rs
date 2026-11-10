//! LFM2 normalization, residual, and token-mixing assembly.

use eredu_core::cache::StateTensorRole;
use eredu_nn::{
    AttentionCache, CausalDepthwiseConvolutionSpec, ConvolutionActivation,
    DeclareLogicalParameters, Error, GatedShortConvolution, GatedShortConvolutionSpec,
    GroupedNeuralBackend, LinearSpec, NeuralBackend, NormalizationConstructionSpec,
    NormalizationOperator, ParameterSpec, Parameterized, RotarySpec, Tensor,
};
use eredu_runtime::RuntimeStateComponents;

use crate::decoder::{
    Attention, AttentionInput, ComponentInstrumentation, DecoderProjectionOperator,
    TensorParallelProjectionOperator,
};

use super::{FeedForward, ModelArgs};

pub(crate) fn normalization_spec(
    args: &ModelArgs,
    parameter: &str,
    dimensions: i32,
) -> Result<NormalizationConstructionSpec, Error> {
    let spec = NormalizationConstructionSpec::learned(
        dimensions,
        args.norm_eps,
        ParameterSpec::trainable(parameter).map_err(Error::backend)?,
    );
    spec.logical_parameters()?;
    Ok(spec)
}

fn attention_head_dim(args: &ModelArgs) -> Result<i32, Error> {
    let head = args
        .hidden_size
        .checked_div(args.num_attention_heads)
        .filter(|&n| {
            args.hidden_size > 0
                && args.num_attention_heads > 0
                && n > 0
                && args.hidden_size % args.num_attention_heads == 0
        })
        .ok_or_else(|| Error::backend("invalid LFM2 attention head width"))?;
    Ok(head)
}

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
    pub(crate) fn from_parameters(
        parameters: super::declarations::BlockParameters,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        use super::declarations::{FeedForwardParameters as F, MixerParameters as M};
        let F::Dense(feed_forward) = parameters.feed_forward else {
            return Err(Error::backend(
                "replicated LFM2 requires a dense feed-forward scope",
            ));
        };
        let mixer = match parameters.mixer {
            M::Convolution(spec) => {
                TokenMixer::ShortConvolution(GatedShortConvolution::new(spec, context)?)
            }
            M::Attention(spec, policy) => {
                TokenMixer::Attention(spec.construct::<B>(policy, context)?)
            }
        };
        Ok(Self {
            mixer,
            feed_forward: super::moe::DenseSwiGlu::from_parameters(feed_forward, context)?,
            operator_norm: B::normalization(parameters.operator_norm, context)?,
            feed_forward_norm: B::normalization(parameters.feed_forward_norm, context)?,
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
    pub(crate) fn from_parameters(
        layer: usize,
        parameters: super::declarations::BlockParameters,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        eredu_nn::validate_logical_parameter_topology(&parameters.logical_parameters()?)
            .map_err(Error::backend_source)?;
        let mixer = match parameters.mixer {
            super::declarations::MixerParameters::Convolution(spec) => {
                TokenMixer::ShortConvolution(GatedShortConvolution::new(spec, context)?)
            }
            super::declarations::MixerParameters::Attention(spec, policy) => {
                TokenMixer::Attention(spec.construct::<B>(policy, context)?)
            }
        };
        Ok(Self {
            mixer,
            feed_forward: FeedForward::from_parameters(layer, parameters.feed_forward, context)?,
            operator_norm: B::normalization(parameters.operator_norm, context)?,
            feed_forward_norm: B::normalization(parameters.feed_forward_norm, context)?,
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
                selected_positions: None,
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

pub(crate) fn short_convolution_spec(
    args: &ModelArgs,
    root: &str,
    channels: i32,
) -> Result<GatedShortConvolutionSpec, Error> {
    let prefix = format!("{root}.conv");
    let parameter = |name: String| ParameterSpec::trainable(name).map_err(Error::backend);
    let linear = |field: &str, input, output| -> Result<LinearSpec, Error> {
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
    let fused = GatedShortConvolutionSpec::input_layout(channels)?.output_width();
    let spec = GatedShortConvolutionSpec {
        input_dimensions: args.hidden_size,
        channels,
        output_dimensions: args.hidden_size,
        input_projection: linear("in_proj", args.hidden_size, fused)?,
        output_projection: linear("out_proj", channels, args.hidden_size)?,
        convolution: CausalDepthwiseConvolutionSpec {
            arithmetic: eredu_nn::ConvolutionArithmetic::Native,
            dilation: 1,
            channels,
            kernel_size: args.conv_l_cache,
            weight: parameter(format!("{prefix}.conv.weight"))?,
            bias: args
                .conv_bias
                .then(|| parameter(format!("{prefix}.conv.bias")))
                .transpose()?,
            activation: ConvolutionActivation::Identity,
        },
    };
    spec.validate()?;
    Ok(spec)
}

/// Exact global or rank-local attention slots, head geometry and arithmetic.
#[derive(Debug, Clone)]
pub(crate) struct AttentionParameters {
    pub projection: eredu_nn::SelfAttentionProjectionSpec,
    pub query_norm: NormalizationConstructionSpec,
    pub key_norm: NormalizationConstructionSpec,
    pub rotary: RotarySpec,
    pub arithmetic: eredu_nn::AttentionArithmetic,
}

impl AttentionParameters {
    pub fn new(args: &ModelArgs, root: &str, geometry: BlockGeometry) -> Result<Self, Error> {
        let head = attention_head_dim(args)?;
        let heads = eredu_nn::AttentionHeadGeometry::new(
            geometry.query_heads,
            geometry.key_value_heads,
            head,
            head,
        )?;
        let prefix = format!("{root}.self_attn");
        let projection =
            |field, input, output| attention_projection_spec(args, &prefix, field, input, output);
        let norm = |field| normalization_spec(args, &format!("{prefix}.{field}.weight"), head);
        let spec = Self {
            projection: eredu_nn::SelfAttentionProjectionSpec::new(
                heads,
                eredu_nn::AttentionQueryLayout::QueryOnly,
                projection("q_proj", args.hidden_size, heads.query_width())?,
                projection("k_proj", args.hidden_size, heads.key_width())?,
                projection("v_proj", args.hidden_size, heads.value_width())?,
                projection("out_proj", heads.attended_width(), args.hidden_size)?,
            )?,
            query_norm: norm("q_layernorm")?,
            key_norm: norm("k_layernorm")?,
            rotary: RotarySpec {
                arithmetic: eredu_nn::RotaryArithmetic::InputProducts,
                dimensions: head,
                base: args.rope.theta,
                traditional: false,
                algorithm: eredu_nn::RotaryAlgorithm::Default,
            },
            arithmetic: eredu_nn::AttentionArithmetic::InputScores,
        };
        spec.logical_parameters()?;
        Ok(spec)
    }

    pub(crate) fn construct<B: NeuralBackend>(
        self,
        policy: eredu_core::AttentionPolicy,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Attention<B>, Error> {
        self.logical_parameters()?;
        let heads = self.projection.geometry;
        let window = policy.sliding_window_i32().map_err(Error::backend)?;
        let mut attention = Attention::from_parts(
            heads.query_heads(),
            heads.key_value_heads(),
            heads.key_dimensions(),
            B::linear(self.projection.query, context)?,
            B::linear(self.projection.key, context)?,
            B::linear(self.projection.value, context)?,
            B::linear(self.projection.output, context)?,
            Some(B::normalization(self.query_norm, context)?),
            Some(B::normalization(self.key_norm, context)?),
            Some(B::rotary(self.rotary, context)?),
            window,
        )?;
        attention.arithmetic = self.arithmetic;
        Ok(attention)
    }
}

impl DeclareLogicalParameters for AttentionParameters {
    fn logical_parameters(&self) -> Result<Vec<eredu_nn::LogicalParameterDeclaration<'_>>, Error> {
        let heads = self.projection.geometry;
        if self.projection.query_layout != eredu_nn::AttentionQueryLayout::QueryOnly
            || self.projection.query.input != self.projection.output.output
            || heads.key_dimensions() != heads.value_dimensions()
            || self.query_norm.dimensions != heads.key_dimensions()
            || self.key_norm.dimensions != heads.key_dimensions()
            || self.rotary.dimensions != heads.key_dimensions()
        {
            return Err(Error::backend(
                "LFM2 attention residual, normalization or rotary geometry mismatch",
            ));
        }
        let mut parameters = self.projection.logical_parameters()?;
        parameters.extend(self.query_norm.logical_parameters()?);
        parameters.extend(self.key_norm.logical_parameters()?);
        eredu_nn::validate_logical_parameter_topology(&parameters)
            .map_err(Error::backend_source)?;
        Ok(parameters)
    }
}

fn attention_projection_spec(
    args: &ModelArgs,
    prefix: &str,
    field: &str,
    input: i32,
    output: i32,
) -> Result<LinearSpec, Error> {
    let name = format!("{prefix}.{field}.weight");
    let spec = LinearSpec {
        input,
        output,
        weight: ParameterSpec::trainable(&name).map_err(Error::backend)?,
        bias: None,
        format: crate::linear_format::standard_linear_format(
            &name,
            args.weight_quantization_for(&name).into(),
        )?,
    };
    spec.logical_parameters()?;
    Ok(spec)
}

/// Ordinary block construction topology, independent of native memory policy.
pub(crate) fn execution_topology(
    declaration: &super::Declaration,
) -> Result<eredu_runtime::execution_topology::TextExecutionTopology, Error> {
    use eredu_runtime::execution_topology::*;
    let gain = |spec: &NormalizationConstructionSpec| -> Result<String, Error> {
        match &spec.scale {
            eredu_nn::NormalizationScale::Learned(weight) => Ok(weight.id.to_string()),
            _ => Err(Error::backend(
                "LFM2 topology requires learned normalization",
            )),
        }
    };
    let positive = |n: i32| {
        u64::try_from(n)
            .ok()
            .filter(|n| *n > 0)
            .ok_or_else(|| Error::backend("text topology dimensions must be positive"))
    };
    use super::declarations::{FeedForwardParameters as F, MixerParameters as M};
    let layers = declaration
        .blocks
        .iter()
        .map(|scope| {
            let mixer = match &scope.mixer {
                M::Convolution(spec) => TokenMixerTopology::GatedConvolution {
                    channels: positive(spec.channels)?,
                    kernel: positive(spec.convolution.kernel_size)?,
                    dilation: positive(spec.convolution.dilation)?,
                    projections: [&spec.input_projection, &spec.output_projection]
                        .into_iter()
                        .map(ProjectionTopology::from_spec)
                        .collect::<Result<_, _>>()?,
                },
                M::Attention(spec, _) => TokenMixerTopology::Attention {
                    query_heads: positive(spec.projection.geometry.query_heads())?,
                    kv_heads: positive(spec.projection.geometry.key_value_heads())?,
                    key_width: positive(spec.projection.geometry.key_dimensions())?,
                    value_width: positive(spec.projection.geometry.value_dimensions())?,
                    input_scores: spec.arithmetic == eredu_nn::AttentionArithmetic::InputScores,
                    softcap: false,
                    sinks: false,
                    projections: spec
                        .projection
                        .projections()
                        .into_iter()
                        .map(ProjectionTopology::from_spec)
                        .collect::<Result<_, _>>()?,
                    output_gate: false,
                    query_key_normalization: true,
                    rotary: true,
                },
            };
            let feed_forward = match &scope.feed_forward {
                F::Dense(spec) => FeedForwardTopology::Gated {
                    intermediate_size: positive(spec.down.input)?,
                    projections: spec
                        .projections()
                        .into_iter()
                        .map(ProjectionTopology::from_spec)
                        .collect::<Result<_, _>>()?,
                },
                F::Routed(spec) => {
                    FeedForwardTopology::from_grouped_specs(&spec.selector, &spec.experts)?
                }
            };
            Ok(TextLayerTopology {
                input_projections: Vec::new(),
                mixer,
                feed_forward,
                normalization_count: 2,
            })
        })
        .collect::<Result<Vec<_>, Error>>()?;
    // Learned, ungrouped RMS normalization promotes through its gain. LFM2's
    // attention/short-convolution and dense SwiGLU equations preserve its dtype
    // through their downstream projections (there is no narrowing cast).
    let mut projection_input_normalizations = std::collections::BTreeMap::new();
    for (index, layer) in layers.iter().enumerate() {
        let scope = &declaration.blocks[index];
        if let TokenMixerTopology::Attention { projections, .. }
        | TokenMixerTopology::GatedConvolution { projections, .. } = &layer.mixer
        {
            for projection in projections {
                projection_input_normalizations.insert(
                    projection.parameter.clone(),
                    vec![gain(&scope.operator_norm)?],
                );
            }
        }
        // Routed bank arithmetic has a separate contract; leave it conservative.
        if let FeedForwardTopology::Gated { projections, .. } = &layer.feed_forward {
            for projection in projections {
                projection_input_normalizations.insert(
                    projection.parameter.clone(),
                    vec![gain(&scope.feed_forward_norm)?],
                );
            }
        }
    }
    let statics = &declaration.statics;
    let output = ProjectionTopology::from_spec(&statics.output_projection_spec())?;
    projection_input_normalizations.insert(
        output.parameter.clone(),
        vec![gain(statics.normalization_spec())?],
    );
    Ok(TextExecutionTopology {
        iterative: None,
        hidden_size: positive(statics.embedding_spec().dimensions)?,
        vocabulary_size: positive(statics.embedding_spec().vocabulary)?,
        layers,
        output,
        output_invocations: 1,
        output_softcap: false,
        selected_parameter_promotion_bytes: None,
        selected_parameter_promotion_payloads: Default::default(),
        projection_storage: Default::default(),
        projection_input_normalizations,
        f32_rms_normalization_gains: Default::default(),
        missing: Vec::new(),
    })
}
