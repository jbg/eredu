//! Backend-neutral Muse-Glimmer decoder equations.

use eredu_nn::{
    AttentionCache, AttentionRequest, EmbeddingOperator, EmbeddingSpec, Error,
    GatedProductGroupLayout, GroupScoring, GroupSelectionOperator, GroupedGatedProductSpec,
    GroupedNeuralBackend, LinearOperator, LinearSpec, NormalizationConstructionSpec,
    NormalizationOperator, Parameter, ParameterSpec, Parameterized, RotaryOperator, RotaryPosition,
    RotarySpec, Tensor, TopKGroupSelectionSpec, TopKGroupSelectorSpec,
};
use eredu_runtime::{
    ExpertPass, RoutedExpertProvider, RoutedExpertRequest, TensorParallelRoutedExpertProvider,
};

use crate::decoder::ComponentInstrumentation;
use crate::linear_format::standard_expert_projection;

use super::{DecoderConfig, LocalGeometry, WeightConvention};

/// RMS normalization whose checkpoint scale may store `(scale - 1)`.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct CenteredRmsNorm<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> {
    /// Stored scale parameter.
    pub weight: Parameter<B::Tensor>,
    #[parameter(skip)]
    epsilon: f32,
    #[parameter(skip)]
    centered: bool,
}

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> CenteredRmsNorm<B> {
    fn new(
        dimensions: i32,
        epsilon: f32,
        centered: bool,
        name: impl Into<String>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        Ok(Self {
            weight: Parameter::unloaded(
                ParameterSpec::trainable(name.into()).map_err(Error::backend)?,
                &[dimensions],
                context,
            )?,
            epsilon,
            centered,
        })
    }

    fn forward(
        &mut self,
        input: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        // Derive the gain from the current authoritative slot. Keeping it on
        // the module would retain derived parameter storage after unloading and
        // reuse stale values after a parameter overlay or replacement.
        let effective_scale = if self.centered {
            Some(self.weight.as_ref().add(
                &B::Tensor::full_f32(1.0, self.weight.as_ref().shape(), context)?,
                context,
            )?)
        } else {
            None
        };
        let scale = effective_scale
            .as_ref()
            .unwrap_or_else(|| self.weight.as_ref());
        B::rms_norm_with_weight(input, scale, self.epsilon, context)
    }
}

/// Gated grouped-query attention with per-layer RoPE/NoPE policy.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct Attention<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> {
    #[parameter(skip)]
    query_heads: i32,
    #[parameter(skip)]
    key_value_heads: i32,
    #[parameter(skip)]
    head_dimensions: i32,
    #[parameter(skip)]
    scale: f32,
    #[parameter(skip)]
    window: Option<i32>,
    #[parameter(skip)]
    uses_rope: bool,
    #[parameter(skip)]
    qk_norm_epsilon: f32,
    #[parameter(skip)]
    query_scale: Option<B::Tensor>,
    /// Query projection.
    pub query: B::Linear,
    /// Key projection.
    pub key: B::Linear,
    /// Value projection.
    pub value: B::Linear,
    /// Per-head sigmoid gate projection.
    pub gate: B::Linear,
    /// Output projection.
    pub output: B::Linear,
    /// Optional synthesized GGUF query norm.
    pub query_norm: Option<B::Normalization>,
    /// Optional synthesized GGUF key norm.
    pub key_norm: Option<B::Normalization>,
    /// Full-head rotary operator, bypassed by NoPE layers.
    pub rotary: B::Rotary,
}

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> Attention<B> {
    /// Builds one unloaded scheduled attention unit.
    pub fn new(
        args: &DecoderConfig,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let policy = args
            .attention_schedule
            .get(layer)
            .copied()
            .ok_or_else(|| Error::backend(format!("missing Muse-Glimmer layer {layer}")))?;
        let prefix = format!("model.layers.{layer}.self_attn");
        let linear = |field: &str, input, output| {
            let weight = format!("{prefix}.{field}.weight");
            B::linear(
                LinearSpec {
                    input,
                    output,
                    weight: ParameterSpec::trainable(&weight).map_err(Error::backend)?,
                    bias: None,
                    format: crate::linear_format::standard_linear_format(
                        &weight,
                        args.linear_format_for(&weight),
                    )?,
                },
                context,
            )
        };
        let gguf = args.weight_convention == WeightConvention::Gguf;
        let norm = |field: &str| {
            B::normalization(
                NormalizationConstructionSpec::learned(
                    args.head_dim,
                    args.rms_norm_eps,
                    ParameterSpec::trainable(format!("{prefix}.{field}.weight"))
                        .map_err(Error::backend)?,
                ),
                context,
            )
        };
        Ok(Self {
            query_heads: args.num_attention_heads,
            key_value_heads: args.num_key_value_heads,
            head_dimensions: args.head_dim,
            scale: (args.head_dim as f32).sqrt().recip(),
            window: policy.window().map(|window| window.get() as i32),
            uses_rope: *args
                .layer_uses_rope
                .get(layer)
                .ok_or_else(|| Error::backend("missing Muse-Glimmer RoPE policy"))?,
            qk_norm_epsilon: args.rms_norm_eps,
            query_scale: (!gguf)
                .then(|| B::Tensor::full_f32(args.qk_scale_factor, &[args.head_dim], context))
                .transpose()?,
            query: linear(
                "q_proj",
                args.hidden_size,
                args.num_attention_heads * args.head_dim,
            )?,
            key: linear(
                "k_proj",
                args.hidden_size,
                args.num_key_value_heads * args.head_dim,
            )?,
            value: linear(
                "v_proj",
                args.hidden_size,
                args.num_key_value_heads * args.head_dim,
            )?,
            gate: linear(
                "gate_proj",
                args.hidden_size,
                args.num_attention_heads * args.head_dim,
            )?,
            output: linear(
                "o_proj",
                args.num_attention_heads * args.head_dim,
                args.hidden_size,
            )?,
            query_norm: gguf.then(|| norm("q_norm")).transpose()?,
            key_norm: gguf.then(|| norm("k_norm")).transpose()?,
            rotary: B::rotary(
                RotarySpec {
                    arithmetic: eredu_nn::RotaryArithmetic::Native,
                    dimensions: args.head_dim,
                    base: args.rope_theta,
                    traditional: args.weight_convention.uses_traditional_rope(),
                    algorithm: crate::rotary::normalize_algorithm(args.rope_scaling.as_ref())
                        .expect("validated Muse-Glimmer RoPE algorithm"),
                },
                context,
            )?,
        })
    }

    fn attend<C: AttentionCache<B::Tensor>>(
        &mut self,
        hidden: &B::Tensor,
        explicit_mask: Option<&B::Tensor>,
        cache: Option<&mut C>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let batch = hidden.dim(0);
        let sequence = hidden.dim(1);
        let offset = cache.as_ref().map_or(0, |cache| cache.offset());
        let reshape = |value: B::Tensor, heads| {
            value
                .reshape(&[batch, sequence, heads, self.head_dimensions], context)?
                .transpose_axes(&[0, 2, 1, 3], context)
        };
        let mut queries = reshape(self.query.forward(hidden, context)?, self.query_heads)?;
        let mut keys = reshape(self.key.forward(hidden, context)?, self.key_value_heads)?;
        let values = reshape(self.value.forward(hidden, context)?, self.key_value_heads)?;
        queries = match self.query_norm.as_mut() {
            Some(norm) => norm.forward(&queries, context)?,
            None => B::rms_norm_with_weight(
                &queries,
                self.query_scale
                    .as_ref()
                    .ok_or_else(|| Error::backend("Muse-Glimmer query scale is missing"))?,
                self.qk_norm_epsilon,
                context,
            )?,
        };
        keys = match self.key_norm.as_mut() {
            Some(norm) => norm.forward(&keys, context)?,
            None => B::rms_norm_without_weight(&keys, self.qk_norm_epsilon, context)?,
        };
        if self.uses_rope {
            queries = self
                .rotary
                .forward(&queries, RotaryPosition::Offset(offset), context)?;
            keys = self
                .rotary
                .forward(&keys, RotaryPosition::Offset(offset), context)?;
        }
        let generated = if explicit_mask.is_none() && sequence > 1 {
            Some(B::causal_mask(
                sequence,
                offset,
                self.window.map(|window| window - 1),
                context,
            )?)
        } else {
            None
        };
        let mask = explicit_mask.or(generated.as_ref());
        let attended = match cache {
            Some(cache) => {
                let (keys, values) = cache.update_for_attention(keys, values, context)?;
                cache.attention(
                    AttentionRequest {
                        arithmetic: eredu_nn::AttentionArithmetic::Fused,
                        softcap: None,
                        queries,
                        keys,
                        values,
                        scale: self.scale,
                        mask,
                        sinks: None,
                    },
                    context,
                )?
            }
            None => B::attention_with_sinks(
                AttentionRequest {
                    arithmetic: eredu_nn::AttentionArithmetic::Fused,
                    softcap: None,
                    queries,
                    keys,
                    values,
                    scale: self.scale,
                    mask,
                    sinks: None,
                },
                context,
            )?,
        };
        let attended = attended.transpose_axes(&[0, 2, 1, 3], context)?.reshape(
            &[batch, sequence, self.query_heads * self.head_dimensions],
            context,
        )?;
        let gate = B::sigmoid(self.gate.forward(hidden, context)?, context)?;
        attended.multiply(&gate, context)
    }

    /// Runs cache-aware attention and gates each attended head before output.
    pub fn forward<C: AttentionCache<B::Tensor>>(
        &mut self,
        hidden: &B::Tensor,
        explicit_mask: Option<&B::Tensor>,
        cache: Option<&mut C>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.forward_instrumented(
            hidden,
            explicit_mask,
            cache,
            None,
            context,
            &mut ComponentInstrumentation::disabled(),
        )
    }

    /// Runs rank-local gated attention and reduces the output projection.
    pub fn forward_parallel<C: AttentionCache<B::Tensor>>(
        &mut self,
        hidden: &B::Tensor,
        explicit_mask: Option<&B::Tensor>,
        cache: Option<&mut C>,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.forward_instrumented(
            hidden,
            explicit_mask,
            cache,
            Some(parallel),
            context,
            &mut ComponentInstrumentation::disabled(),
        )
    }

    fn forward_instrumented<C: AttentionCache<B::Tensor>>(
        &mut self,
        hidden: &B::Tensor,
        explicit_mask: Option<&B::Tensor>,
        cache: Option<&mut C>,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error> {
        let attended = self.attend(hidden, explicit_mask, cache, context)?;
        let attended = instrumentation.apply("attention.channels", attended)?;
        instrumentation.project::<B>(
            "attention.write_input",
            &mut self.output,
            &attended,
            parallel,
            context,
        )
    }
}

/// Dense SwiGLU branch.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct Mlp<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> {
    /// Gate projection.
    pub gate: B::Linear,
    /// Up projection.
    pub up: B::Linear,
    /// Down projection.
    pub down: B::Linear,
}

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> Mlp<B> {
    fn new(
        args: &DecoderConfig,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let prefix = format!("model.layers.{layer}.mlp");
        let linear = |field: &str, input, output| {
            let weight = format!("{prefix}.{field}.weight");
            B::linear(
                LinearSpec {
                    input,
                    output,
                    weight: ParameterSpec::trainable(&weight).map_err(Error::backend)?,
                    bias: None,
                    format: crate::linear_format::standard_linear_format(
                        &weight,
                        args.linear_format_for(&weight),
                    )?,
                },
                context,
            )
        };
        Ok(Self {
            gate: linear("gate_proj", args.hidden_size, args.intermediate_size)?,
            up: linear("up_proj", args.hidden_size, args.intermediate_size)?,
            down: linear("down_proj", args.intermediate_size, args.hidden_size)?,
        })
    }

    fn forward_instrumented(
        &mut self,
        hidden: &B::Tensor,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error> {
        let gate = self.gate.forward(hidden, context)?;
        let up = self.up.forward(hidden, context)?;
        let hidden = B::gated_product(gate, up, eredu_nn::GatedProductPolicy::default(), context)?;
        let hidden = instrumentation.apply("feed_forward.units", hidden)?;
        instrumentation.project::<B>(
            "feed_forward.write_input",
            &mut self.down,
            &hidden,
            parallel,
            context,
        )
    }
}

/// Softmax top-k routed gated-product branch used by Muse-Glimmer MoE checkpoints.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct SparseMoe<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> {
    /// Learned softmax router.
    pub router: B::Selector,
    /// Packed routed expert bank.
    pub experts: B::GatedProductGroups,
    #[parameter(skip)]
    hidden_size: i32,
}

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> SparseMoe<B> {
    fn new(
        args: &DecoderConfig,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        Self::new_with_spec(args, layer, expert_bank_spec(args, layer)?, context)
    }

    pub(crate) fn new_with_spec(
        args: &DecoderConfig,
        layer: usize,
        spec: GroupedGatedProductSpec,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let prefix = format!("model.layers.{layer}.mlp");
        Ok(Self {
            router: B::top_k_group_selector(
                TopKGroupSelectorSpec::new(
                    args.hidden_size,
                    ParameterSpec::trainable(format!("{prefix}.gate.weight"))
                        .map_err(Error::backend)?,
                    crate::linear_format::standard_linear_format(
                        &format!("{prefix}.gate.weight"),
                        args.linear_format_for(&format!("{prefix}.gate.weight")),
                    )?,
                    TopKGroupSelectionSpec::new(
                        args.num_experts,
                        args.num_experts_per_tok,
                        GroupScoring::Softmax,
                        args.norm_topk_prob,
                    )?,
                )?,
                context,
            )?,
            experts: B::grouped_gated_product(spec, context)?,
            hidden_size: args.hidden_size,
        })
    }

    fn forward_with_provider<P>(
        &mut self,
        hidden: &B::Tensor,
        layer: usize,
        pass: ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error>
    where
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        if hidden.shape().len() != 3 || hidden.dim(2) != self.hidden_size {
            return Err(Error::backend("invalid Muse-Glimmer MoE hidden geometry"));
        }
        let shape = hidden.shape().to_vec();
        let flat = hidden.reshape(&[-1, self.hidden_size], context)?;
        let routes = self.router.select(&flat, context)?;
        instrumentation
            .routed(
                "routing",
                RoutedExpertRequest {
                    unit_observer: None,
                    bank: eredu_runtime::RoutedBankId::new(0),
                    layer,
                    input: &flat,
                    routes: &routes,
                    pass,
                },
                |request| provider.forward_grouped(&mut self.experts, request, context),
            )?
            .reshape(&shape, context)
    }

    fn forward_parallel_with_provider<P>(
        &mut self,
        hidden: &B::Tensor,
        layer: usize,
        pass: ExpertPass,
        provider: &mut P,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error>
    where
        P: TensorParallelRoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        if hidden.shape().len() != 3 || hidden.dim(2) != self.hidden_size {
            return Err(Error::backend("invalid Muse-Glimmer MoE hidden geometry"));
        }
        let shape = hidden.shape().to_vec();
        let flat = hidden.reshape(&[-1, self.hidden_size], context)?;
        let routes = self.router.select(&flat, context)?;
        let output = instrumentation.routed(
            "routing",
            RoutedExpertRequest {
                unit_observer: None,
                bank: eredu_runtime::RoutedBankId::new(0),
                layer,
                input: &flat,
                routes: &routes,
                pass,
            },
            |request| {
                provider.forward_grouped_tensor_parallel(
                    &mut self.experts,
                    request,
                    B::parallel_size(parallel),
                    context,
                )
            },
        )?;
        let output =
            eredu_runtime::reduce_routed_expert_tensor_parallel::<B>(output, parallel, context)?;
        output.reshape(&shape, context)
    }
}

/// Returns the architecture-owned routed expert specification for one layer.
pub fn expert_bank_spec(
    args: &DecoderConfig,
    layer: usize,
) -> Result<GroupedGatedProductSpec, Error> {
    let prefix = format!("model.layers.{layer}.mlp.experts");
    let gate_up = format!("{prefix}.gate_up_proj");
    let down = format!("{prefix}.down_proj");
    GroupedGatedProductSpec::new(
        args.num_experts,
        args.hidden_size,
        args.moe_intermediate_size,
        args.hidden_size,
        eredu_nn::GatedProductPolicy::ordinary_silu(),
        GatedProductGroupLayout::Packed {
            gate_up: standard_expert_projection(&gate_up, None, args.linear_format_for(&gate_up))?,
            down: standard_expert_projection(&down, None, args.linear_format_for(&down))?,
        },
    )
}

/// Returns the same architecture-owned bank at placement-resolved geometry.
pub(crate) fn localized_expert_bank_spec(
    args: &DecoderConfig,
    layer: usize,
    expert_count: i32,
    intermediate_dimensions: i32,
) -> Result<GroupedGatedProductSpec, Error> {
    expert_bank_spec(args, layer)?.with_group_geometry(expert_count, intermediate_dimensions)
}

/// Dense or routed feed-forward branch selected by normalized checkpoint geometry.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub enum FeedForward<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> {
    /// Dense SwiGLU branch.
    Dense(Mlp<B>),
    /// Softmax top-k routed gated-product branch.
    Sparse(SparseMoe<B>),
}

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> FeedForward<B> {
    fn new_with_routed_spec(
        args: &DecoderConfig,
        layer: usize,
        routed_spec: Option<GroupedGatedProductSpec>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        if args.is_moe() {
            Ok(Self::Sparse(match routed_spec {
                Some(spec) => SparseMoe::new_with_spec(args, layer, spec, context)?,
                None => SparseMoe::new(args, layer, context)?,
            }))
        } else {
            Ok(Self::Dense(Mlp::new(args, layer, context)?))
        }
    }

    fn forward_with_provider<P>(
        &mut self,
        hidden: &B::Tensor,
        layer: usize,
        pass: ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error>
    where
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        match self {
            Self::Dense(dense) => {
                dense.forward_instrumented(hidden, None, context, instrumentation)
            }
            Self::Sparse(sparse) => sparse.forward_with_provider(
                hidden,
                layer,
                pass,
                provider,
                context,
                instrumentation,
            ),
        }
    }

    fn forward_parallel_with_provider<P>(
        &mut self,
        hidden: &B::Tensor,
        layer: usize,
        pass: ExpertPass,
        provider: &mut P,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error>
    where
        P: TensorParallelRoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        match self {
            Self::Dense(dense) => {
                dense.forward_instrumented(hidden, Some(parallel), context, instrumentation)
            }
            Self::Sparse(sparse) => sparse.forward_parallel_with_provider(
                hidden,
                layer,
                pass,
                provider,
                parallel,
                context,
                instrumentation,
            ),
        }
    }
}

/// One post-normalized decoder block.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct TransformerBlock<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> {
    #[parameter(skip)]
    layer: usize,
    /// Gated self-attention.
    pub attention: Attention<B>,
    /// Dense SwiGLU branch.
    pub feed_forward: FeedForward<B>,
    /// Pre-attention norm.
    pub input_norm: CenteredRmsNorm<B>,
    /// Attention-delta norm before residual addition.
    pub post_attention_norm: CenteredRmsNorm<B>,
    /// Pre-feed-forward norm.
    pub pre_feed_forward_norm: CenteredRmsNorm<B>,
    /// Feed-forward-delta norm before residual addition.
    pub post_feed_forward_norm: CenteredRmsNorm<B>,
}

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> TransformerBlock<B> {
    /// Builds one unloaded block.
    pub fn new(
        args: &DecoderConfig,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        Self::new_with_routed_spec(args, layer, None, context)
    }

    pub(crate) fn new_with_routed_spec(
        args: &DecoderConfig,
        layer: usize,
        routed_spec: Option<GroupedGatedProductSpec>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let root = format!("model.layers.{layer}");
        let centered = args.weight_convention == WeightConvention::HuggingFace;
        let norm = |field: &str, epsilon| {
            CenteredRmsNorm::new(
                args.hidden_size,
                epsilon,
                centered,
                format!("{root}.{field}.weight"),
                context,
            )
        };
        Ok(Self {
            layer,
            attention: Attention::new(args, layer, context)?,
            feed_forward: FeedForward::new_with_routed_spec(args, layer, routed_spec, context)?,
            input_norm: norm("input_layernorm", args.rms_norm_eps)?,
            post_attention_norm: norm("post_attention_layernorm", args.post_norm_eps)?,
            pre_feed_forward_norm: norm("pre_feedforward_layernorm", args.rms_norm_eps)?,
            post_feed_forward_norm: norm("post_feedforward_layernorm", args.post_norm_eps)?,
        })
    }

    /// Executes exact pre/post normalization and residual order.
    pub fn forward<C: AttentionCache<B::Tensor>>(
        &mut self,
        hidden: &B::Tensor,
        mask: Option<&B::Tensor>,
        cache: Option<&mut C>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.forward_observed(
            hidden,
            mask,
            cache,
            context,
            &mut ComponentInstrumentation::disabled(),
        )
    }

    /// Executes the same block with component observations and interventions.
    pub fn forward_observed<C: AttentionCache<B::Tensor>>(
        &mut self,
        hidden: &B::Tensor,
        mask: Option<&B::Tensor>,
        cache: Option<&mut C>,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error> {
        let pass = if hidden.dim(1) > 1 {
            ExpertPass::Prefill
        } else {
            ExpertPass::Decode
        };
        self.forward_with_provider_observed(
            hidden,
            mask,
            cache,
            pass,
            &mut eredu_runtime::ResidentExpertProvider,
            context,
            instrumentation,
        )
    }

    /// Executes the canonical block while runtime policy owns routed experts.
    pub fn forward_with_provider<C, P>(
        &mut self,
        hidden: &B::Tensor,
        mask: Option<&B::Tensor>,
        cache: Option<&mut C>,
        pass: ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        C: AttentionCache<B::Tensor>,
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        self.forward_with_provider_observed(
            hidden,
            mask,
            cache,
            pass,
            provider,
            context,
            &mut ComponentInstrumentation::disabled(),
        )
    }

    /// Observes the exact routed request and consumed sublayer values.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_with_provider_observed<C, P>(
        &mut self,
        hidden: &B::Tensor,
        mask: Option<&B::Tensor>,
        cache: Option<&mut C>,
        pass: ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error>
    where
        C: AttentionCache<B::Tensor>,
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        self.forward_inner(
            hidden,
            mask,
            cache,
            None,
            context,
            instrumentation,
            |feed_forward, hidden, layer, instrumentation| {
                feed_forward.forward_with_provider(
                    hidden,
                    layer,
                    pass,
                    provider,
                    context,
                    instrumentation,
                )
            },
        )
    }

    /// Executes the canonical block with rank-local projections and resident experts.
    pub fn forward_parallel<C: AttentionCache<B::Tensor>>(
        &mut self,
        hidden: &B::Tensor,
        mask: Option<&B::Tensor>,
        cache: Option<&mut C>,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        B: eredu_nn::TensorParallelGroupedNeuralBackend,
    {
        self.forward_parallel_observed(
            hidden,
            mask,
            cache,
            parallel,
            context,
            &mut ComponentInstrumentation::disabled(),
        )
    }

    /// Executes the same TP block with local component evidence.
    pub fn forward_parallel_observed<C: AttentionCache<B::Tensor>>(
        &mut self,
        hidden: &B::Tensor,
        mask: Option<&B::Tensor>,
        cache: Option<&mut C>,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error>
    where
        B: eredu_nn::TensorParallelGroupedNeuralBackend,
    {
        let pass = if hidden.dim(1) > 1 {
            ExpertPass::Prefill
        } else {
            ExpertPass::Decode
        };
        self.forward_parallel_with_provider_observed(
            hidden,
            mask,
            cache,
            pass,
            &mut eredu_runtime::ResidentExpertProvider,
            parallel,
            context,
            instrumentation,
        )
    }

    /// Executes the canonical TP block while runtime owns routed experts.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_parallel_with_provider<C, P>(
        &mut self,
        hidden: &B::Tensor,
        mask: Option<&B::Tensor>,
        cache: Option<&mut C>,
        pass: ExpertPass,
        provider: &mut P,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        C: AttentionCache<B::Tensor>,
        P: TensorParallelRoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        self.forward_parallel_with_provider_observed(
            hidden,
            mask,
            cache,
            pass,
            provider,
            parallel,
            context,
            &mut ComponentInstrumentation::disabled(),
        )
    }

    /// Observes and intervenes in the same TP/provider block driver.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_parallel_with_provider_observed<C, P>(
        &mut self,
        hidden: &B::Tensor,
        mask: Option<&B::Tensor>,
        cache: Option<&mut C>,
        pass: ExpertPass,
        provider: &mut P,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error>
    where
        C: AttentionCache<B::Tensor>,
        P: TensorParallelRoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        self.forward_inner(
            hidden,
            mask,
            cache,
            Some(parallel),
            context,
            instrumentation,
            |feed_forward, hidden, layer, instrumentation| {
                feed_forward.forward_parallel_with_provider(
                    hidden,
                    layer,
                    pass,
                    provider,
                    parallel,
                    context,
                    instrumentation,
                )
            },
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_inner<C, F>(
        &mut self,
        hidden: &B::Tensor,
        mask: Option<&B::Tensor>,
        cache: Option<&mut C>,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
        feed_forward: F,
    ) -> Result<B::Tensor, Error>
    where
        C: AttentionCache<B::Tensor>,
        F: FnOnce(
            &mut FeedForward<B>,
            &B::Tensor,
            usize,
            &mut ComponentInstrumentation<'_, B::Tensor>,
        ) -> Result<B::Tensor, Error>,
    {
        let observed_input = if instrumentation.enabled() {
            Some(instrumentation.apply("input", hidden.clone())?)
        } else {
            None
        };
        let hidden = observed_input.as_ref().unwrap_or(hidden);
        let normalized = self.input_norm.forward(hidden, context)?;
        let normalized = instrumentation.apply("attention.input", normalized)?;
        let attention = self.attention.forward_instrumented(
            &normalized,
            mask,
            cache,
            parallel,
            context,
            instrumentation,
        )?;
        let attention = instrumentation.apply("attention.write", attention)?;
        let attention = self.post_attention_norm.forward(&attention, context)?;
        let attention = instrumentation.apply("attention.output", attention)?;
        let hidden = hidden.add(&attention, context)?;
        let hidden = instrumentation.apply("attention.residual", hidden)?;
        let normalized = self.pre_feed_forward_norm.forward(&hidden, context)?;
        let normalized = instrumentation.apply("feed_forward.input", normalized)?;
        let feed_forward = feed_forward(
            &mut self.feed_forward,
            &normalized,
            self.layer,
            instrumentation,
        )?;
        let feed_forward = instrumentation.apply("feed_forward.write", feed_forward)?;
        let feed_forward = self
            .post_feed_forward_norm
            .forward(&feed_forward, context)?;
        let feed_forward = instrumentation.apply("feed_forward.output", feed_forward)?;
        let hidden = hidden.add(&feed_forward, context)?;
        let hidden = instrumentation.apply("feed_forward.residual", hidden)?;
        instrumentation.apply("output", hidden)
    }
}

/// Pinned token embedding, final norm, and tied/untied vocabulary head.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct StaticModules<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> {
    /// Token table.
    pub embeddings: B::Embedding,
    /// Final learned RMS norm.
    pub final_norm: B::Normalization,
    /// Optional untied head.
    pub head: Option<B::Linear>,
    #[parameter(skip)]
    embedding_epsilon: f32,
    #[parameter(skip)]
    output_multiplier: f32,
    #[parameter(skip)]
    logit_cap: f32,
}

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> StaticModules<B> {
    /// Builds pinned modules.
    pub fn new(
        args: &DecoderConfig,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let embedding = "model.embed_tokens.weight";
        let head = "lm_head.weight";
        Ok(Self {
            embeddings: B::embedding(
                EmbeddingSpec {
                    vocabulary: args.vocab_size,
                    dimensions: args.hidden_size,
                    weight: ParameterSpec::trainable(embedding).map_err(Error::backend)?,
                    format: crate::linear_format::standard_linear_format(
                        embedding,
                        args.linear_format_for(embedding),
                    )?,
                },
                context,
            )?,
            final_norm: B::normalization(
                NormalizationConstructionSpec::learned(
                    args.hidden_size,
                    args.rms_norm_eps,
                    ParameterSpec::trainable("model.norm.weight").map_err(Error::backend)?,
                ),
                context,
            )?,
            head: (!args.tie_word_embeddings)
                .then(|| {
                    B::linear(
                        LinearSpec {
                            input: args.hidden_size,
                            output: args.vocab_size,
                            weight: ParameterSpec::trainable(head).map_err(Error::backend)?,
                            bias: None,
                            format: crate::linear_format::standard_linear_format(
                                head,
                                args.linear_format_for(head),
                            )?,
                        },
                        context,
                    )
                })
                .transpose()?,
            embedding_epsilon: args.rms_norm_eps,
            output_multiplier: args.output_multiplier,
            logit_cap: args.final_logit_softcapping,
        })
    }

    /// Builds pinned vocabulary modules from planner-derived row ownership.
    pub fn new_parallel(
        args: &DecoderConfig,
        geometry: &LocalGeometry,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let embedding = "model.embed_tokens.weight";
        let head = "lm_head.weight";
        Ok(Self {
            embeddings: B::vocabulary_parallel_embedding(
                EmbeddingSpec {
                    vocabulary: args.vocab_size,
                    dimensions: args.hidden_size,
                    weight: ParameterSpec::trainable(embedding).map_err(Error::backend)?,
                    format: crate::linear_format::standard_linear_format(
                        embedding,
                        args.linear_format_for(embedding),
                    )?,
                },
                geometry.embedding_range().clone(),
                context,
            )?,
            final_norm: B::normalization(
                NormalizationConstructionSpec::learned(
                    args.hidden_size,
                    args.rms_norm_eps,
                    ParameterSpec::trainable("model.norm.weight").map_err(Error::backend)?,
                ),
                context,
            )?,
            head: (!args.tie_word_embeddings)
                .then(|| {
                    B::vocabulary_parallel_linear(
                        LinearSpec {
                            input: args.hidden_size,
                            output: args.vocab_size,
                            weight: ParameterSpec::trainable(head).map_err(Error::backend)?,
                            bias: None,
                            format: crate::linear_format::standard_linear_format(
                                head,
                                args.linear_format_for(head),
                            )?,
                        },
                        geometry
                            .output_range()
                            .cloned()
                            .ok_or_else(|| Error::backend("missing Muse-Glimmer output range"))?,
                        context,
                    )
                })
                .transpose()?,
            embedding_epsilon: args.rms_norm_eps,
            output_multiplier: args.output_multiplier,
            logit_cap: args.final_logit_softcapping,
        })
    }

    /// Embeds IDs and applies the released weightless normalization.
    pub fn embed(
        &mut self,
        tokens: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let hidden = self.embeddings.forward(tokens, context)?;
        B::rms_norm_without_weight(&hidden, self.embedding_epsilon, context)
    }

    /// Applies released weightless input normalization to gathered TP embeddings.
    pub fn normalize_embeddings(
        &self,
        hidden: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        B::rms_norm_without_weight(hidden, self.embedding_epsilon, context)
    }

    /// Applies the replicated final norm before the vocabulary projection.
    pub fn final_hidden(
        &mut self,
        hidden: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.final_norm.forward(hidden, context)
    }

    /// Applies output scaling and softcapping after vocabulary shards are gathered.
    pub fn finish_logits(
        &self,
        logits: B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        logits
            .multiply_scalar(self.output_multiplier / self.logit_cap, context)?
            .tanh(context)?
            .multiply_scalar(self.logit_cap, context)
    }

    /// Applies final norm, head, multiplier, then tanh softcap.
    pub fn logits(
        &mut self,
        hidden: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.logits_instrumented(
            hidden,
            None,
            context,
            &mut ComponentInstrumentation::disabled(),
        )
    }

    /// Applies the same readout with evidence from the actual projection input.
    pub(crate) fn logits_instrumented(
        &mut self,
        hidden: &B::Tensor,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error> {
        let hidden = instrumentation.normalize_readout(hidden, &mut self.final_norm, context)?;
        let logits = match (parallel, self.head.as_mut()) {
            (Some(parallel), Some(head)) => instrumentation.project_vocabulary::<B>(
                "projection_input",
                head,
                &hidden,
                parallel,
                context,
            )?,
            (Some(parallel), None) => instrumentation.project_vocabulary_embedding::<B>(
                "projection_input",
                &mut self.embeddings,
                &hidden,
                parallel,
                context,
            )?,
            (None, Some(head)) => {
                instrumentation.project::<B>("projection_input", head, &hidden, None, context)?
            }
            (None, None) => instrumentation.project_embedding(
                "projection_input",
                &mut self.embeddings,
                &hidden,
                context,
            )?,
        };
        let logits = instrumentation.apply("linear", logits)?;
        self.finish_logits(logits, context)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn scales_before_softcap() {
        let actual = (2.0_f32 * 3.0 / 4.0).tanh() * 4.0;
        assert!((actual - 3.620_594).abs() < 1e-5);
    }
}
