//! Shared Qwen3-Next/Qwen3.5 decoder block.

use eredu_nn::{
    AttentionCache, Error, ExpertProjectionSpec, GatedProductExpertBankSpec,
    GatedProductExpertLayout, LinearOperator, LinearSpec, NormalizationConstructionSpec,
    NormalizationOperator, NormalizationScale, ParameterSpec, Parameterized, RotarySpec,
    RoutedNeuralBackend, RoutingOperator, RoutingScoring, Tensor, TopKRouterSpec, TopKRoutingSpec,
};
use eredu_runtime::{
    ExpertPass, ResidentExpertProvider, RoutedExpertProvider, RoutedExpertRequest,
    RuntimeStateComponents,
};

use crate::decoder::{Attention, AttentionInput, FeedForwardOperator, Mlp};

use super::{HybridConfig, HybridLayerPolicy, LinearAttention};

/// Scheduled hybrid token mixer.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub enum TokenMixer<B: RoutedNeuralBackend> {
    /// Gated-delta recurrent attention.
    Linear(LinearAttention<B>),
    /// Gated grouped-query self attention.
    Attention(Attention<B>),
}

/// Routed experts plus the always-on Qwen shared expert.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct SharedRoutedGatedProduct<B: RoutedNeuralBackend> {
    #[parameter(skip)]
    layer: usize,
    /// Learned top-k router.
    pub router: B::Router,
    /// Packed routed expert bank.
    pub experts: B::GatedProductExpertBank,
    /// Always-on dense shared expert.
    pub shared_expert: Mlp<B>,
    /// Scalar gate applied to the shared expert output.
    pub shared_expert_gate: B::Linear,
}

impl<B: RoutedNeuralBackend> SharedRoutedGatedProduct<B> {
    fn new(
        config: &HybridConfig,
        layer: usize,
        prefix: &str,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let prefix = format!("{prefix}.mlp");
        let routing = TopKRoutingSpec::new(
            config.num_experts,
            config.num_experts_per_tok,
            RoutingScoring::Softmax,
            config.norm_topk_prob,
        )?;
        let router_name = format!("{prefix}.gate.weight");
        let router = B::top_k_router(
            TopKRouterSpec {
                input_dimensions: config.hidden_size,
                weight: parameter(&router_name)?,
                bias: None,
                correction_bias: None,
                input_transform: None,
                route_scale: None,
                quantization: config.quantization,
                routing,
            },
            context,
        )?;
        let experts = B::gated_product_expert_bank(
            expert_bank_spec_at(config, &format!("{prefix}.experts"))?,
            context,
        )?;
        Ok(Self {
            layer,
            router,
            experts,
            shared_expert: new_mlp(
                config,
                &format!("{prefix}.shared_expert"),
                config.shared_expert_intermediate_size,
                context,
            )?,
            shared_expert_gate: new_linear::<B>(
                config,
                &format!("{prefix}.shared_expert_gate"),
                config.hidden_size,
                1,
                false,
                context,
            )?,
        })
    }

    fn forward_with_provider<P>(
        &mut self,
        input: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
        provider: &mut P,
    ) -> Result<B::Tensor, Error>
    where
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        let routes = self.router.route(input, context)?;
        let routed = provider
            .forward_routed(
                &mut self.experts,
                RoutedExpertRequest {
                    layer: self.layer,
                    input,
                    routes: &routes,
                    pass: pass(input),
                },
                context,
            )
            .map_err(Error::backend)?;
        let shared = self.shared_expert.forward_feed_forward(input, context)?;
        let shared_gate = B::sigmoid(self.shared_expert_gate.forward(input, context)?, context)?;
        routed.add(&shared.multiply(&shared_gate, context)?, context)
    }

    fn forward_tensor_parallel_with_provider<P>(
        &mut self,
        input: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        provider: &mut P,
    ) -> Result<eredu_runtime::RoutedExpertTensorParallelOutput<B::Tensor>, Error>
    where
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        let routes = self.router.route(input, context)?;
        let routed = provider
            .forward_routed_tensor_parallel(
                &mut self.experts,
                RoutedExpertRequest {
                    layer: self.layer,
                    input,
                    routes: &routes,
                    pass: pass(input),
                },
                B::parallel_size(parallel),
                context,
            )
            .map_err(Error::backend)?;
        let shared = self.shared_expert.forward_feed_forward(input, context)?;
        let shared_gate = B::sigmoid(self.shared_expert_gate.forward(input, context)?, context)?;
        let shared = shared.multiply(&shared_gate, context)?;
        match routed {
            eredu_runtime::RoutedExpertTensorParallelOutput::Complete(routed) => {
                let shared = B::sum_parallel(shared, parallel, context)?;
                Ok(eredu_runtime::RoutedExpertTensorParallelOutput::Complete(
                    routed.add(&shared, context)?,
                ))
            }
            eredu_runtime::RoutedExpertTensorParallelOutput::Partial(mut routed) => {
                routed.reducible = routed.reducible.add(&shared, context)?;
                Ok(eredu_runtime::RoutedExpertTensorParallelOutput::Partial(
                    routed,
                ))
            }
        }
    }
}

/// Returns the architecture-owned routed expert specification for a target or MTP layer.
pub fn expert_bank_spec(
    config: &HybridConfig,
    layer: usize,
) -> Result<GatedProductExpertBankSpec, Error> {
    let target = config.num_hidden_layers as usize;
    let root = if layer < target {
        format!("model.layers.{layer}.mlp.experts")
    } else {
        format!("mtp.layers.{}.mlp.experts", layer - target)
    };
    expert_bank_spec_at(config, &root)
}

fn expert_bank_spec_at(
    config: &HybridConfig,
    expert_prefix: &str,
) -> Result<GatedProductExpertBankSpec, Error> {
    let gate_up_name = format!("{expert_prefix}.gate_up_proj");
    let down_name = format!("{expert_prefix}.down_proj");
    Ok(GatedProductExpertBankSpec {
        expert_count: config.num_experts,
        input_dimensions: config.hidden_size,
        intermediate_dimensions: config.moe_intermediate_size,
        output_dimensions: config.hidden_size,
        policy: eredu_nn::GatedProductPolicy::ordinary_silu(),
        layout: GatedProductExpertLayout::Packed {
            gate_up: ExpertProjectionSpec {
                weight: parameter(&gate_up_name)?,
                bias: None,
                format: config.linear_format(&gate_up_name),
            },
            down: ExpertProjectionSpec {
                weight: parameter(&down_name)?,
                bias: None,
                format: config.linear_format(&down_name),
            },
        },
    })
}

/// Dense or routed/shared-expert feed-forward policy selected by configuration.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub enum FeedForward<B: RoutedNeuralBackend> {
    /// Dense SwiGLU.
    Dense(Mlp<B>),
    /// Routed SwiGLU plus an always-on shared expert.
    Routed(SharedRoutedGatedProduct<B>),
}

impl<B: RoutedNeuralBackend> FeedForward<B> {
    fn new(
        config: &HybridConfig,
        layer: usize,
        prefix: &str,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        if config.is_moe() {
            SharedRoutedGatedProduct::new(config, layer, prefix, context).map(Self::Routed)
        } else {
            new_mlp(
                config,
                &format!("{prefix}.mlp"),
                config.intermediate_size,
                context,
            )
            .map(Self::Dense)
        }
    }

    /// Executes through a runtime-owned expert provider.
    pub fn forward_with_provider<P>(
        &mut self,
        input: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
        provider: &mut P,
    ) -> Result<B::Tensor, Error>
    where
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        match self {
            Self::Dense(mlp) => mlp.forward_feed_forward(input, context),
            Self::Routed(moe) => moe.forward_with_provider(input, context, provider),
        }
    }
}

/// One pre-normalized hybrid decoder block for every dense/MoE family form.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct Block<B: RoutedNeuralBackend> {
    /// Recurrent or full-attention policy.
    pub mixer: TokenMixer<B>,
    /// Dense or routed/shared-expert policy.
    pub feed_forward: FeedForward<B>,
    /// Learned-offset token-mixer normalization.
    pub input_norm: B::Normalization,
    /// Learned-offset feed-forward normalization.
    pub post_attention_norm: B::Normalization,
}

impl<B: RoutedNeuralBackend> Block<B> {
    /// Builds one global-geometry physical decoder layer.
    pub fn new(
        config: &HybridConfig,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let policy = config
            .layer_schedule
            .get(layer)
            .copied()
            .ok_or_else(|| Error::backend(format!("Qwen hybrid has no layer {layer}")))?;
        Self::new_at(
            config,
            layer,
            &format!("model.layers.{layer}"),
            policy,
            context,
        )
    }

    /// Builds one configured MTP prediction block at its checkpoint path.
    pub fn new_mtp(
        config: &HybridConfig,
        depth: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        if depth >= usize::try_from(config.mtp_num_hidden_layers).map_err(Error::backend)? {
            return Err(Error::backend(format!(
                "Qwen hybrid MTP depth {depth} is outside {} configured layers",
                config.mtp_num_hidden_layers
            )));
        }
        Self::new_at(
            config,
            config.num_hidden_layers as usize + depth,
            &format!("mtp.layers.{depth}"),
            HybridLayerPolicy::SelfAttention(eredu_core::attention::AttentionPolicy::Full),
            context,
        )
    }

    fn new_at(
        config: &HybridConfig,
        layer: usize,
        root: &str,
        policy: HybridLayerPolicy,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let mixer = match policy {
            HybridLayerPolicy::LinearAttention => {
                TokenMixer::Linear(LinearAttention::new(config, layer, context)?)
            }
            HybridLayerPolicy::SelfAttention(_) => {
                TokenMixer::Attention(new_attention(config, root, context)?)
            }
        };
        let norm = |field: &str| {
            B::normalization(
                NormalizationConstructionSpec {
                    dimensions: config.hidden_size,
                    epsilon: config.rms_norm_eps,
                    scale: NormalizationScale::LearnedOffset {
                        weight: parameter(format!("{root}.{field}.weight"))?,
                        offset: 1.0,
                    },
                },
                context,
            )
        };
        Ok(Self {
            mixer,
            feed_forward: FeedForward::new(config, layer, root, context)?,
            input_norm: norm("input_layernorm")?,
            post_attention_norm: norm("post_attention_layernorm")?,
        })
    }

    /// Executes one block with resident routed experts.
    pub fn forward<S>(
        &mut self,
        hidden: &B::Tensor,
        mask: Option<&B::Tensor>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        S: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
    {
        self.forward_with_provider(hidden, mask, state, context, &mut ResidentExpertProvider)
    }

    /// Executes one block through a runtime-owned routed-expert provider.
    pub fn forward_with_provider<S, P>(
        &mut self,
        hidden: &B::Tensor,
        mask: Option<&B::Tensor>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
        provider: &mut P,
    ) -> Result<B::Tensor, Error>
    where
        S: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        let normalized = self.input_norm.forward(hidden, context)?;
        let mixed = match &mut self.mixer {
            TokenMixer::Linear(linear) => linear.forward(&normalized, state, context)?,
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
        };
        let hidden = hidden.add(&mixed, context)?;
        let normalized = self.post_attention_norm.forward(&hidden, context)?;
        let feed_forward =
            self.feed_forward
                .forward_with_provider(&normalized, context, provider)?;
        hidden.add(&feed_forward, context)
    }

    /// Executes local projections and one row reduction per parallel output.
    pub fn forward_parallel<S, P>(
        &mut self,
        hidden: &B::Tensor,
        mask: Option<&B::Tensor>,
        state: &mut S,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        provider: &mut P,
    ) -> Result<B::Tensor, Error>
    where
        S: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        let normalized = self.input_norm.forward(hidden, context)?;
        let mixed = match &mut self.mixer {
            TokenMixer::Linear(linear) => {
                linear.forward_parallel(&normalized, state, parallel, context)?
            }
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
        };
        let hidden = hidden.add(&mixed, context)?;
        let normalized = self.post_attention_norm.forward(&hidden, context)?;
        let feed_forward = match &mut self.feed_forward {
            FeedForward::Dense(mlp) => {
                mlp.forward_feed_forward_parallel(&normalized, parallel, context)?
            }
            FeedForward::Routed(moe) => {
                let output = moe.forward_tensor_parallel_with_provider(
                    &normalized,
                    parallel,
                    context,
                    provider,
                )?;
                eredu_runtime::reduce_routed_expert_tensor_parallel::<B>(output, parallel, context)?
            }
        };
        hidden.add(&feed_forward, context)
    }
}

fn new_attention<B: RoutedNeuralBackend>(
    config: &HybridConfig,
    root: &str,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<Attention<B>, Error> {
    let prefix = format!("{root}.self_attn");
    let linear = |field: &str, input, output| {
        new_linear::<B>(
            config,
            &format!("{prefix}.{field}"),
            input,
            output,
            config.attention_bias,
            context,
        )
    };
    let norm = |field: &str| {
        B::normalization(
            NormalizationConstructionSpec {
                dimensions: config.head_dim,
                epsilon: config.rms_norm_eps,
                scale: NormalizationScale::LearnedOffset {
                    weight: parameter(format!("{prefix}.{field}.weight"))?,
                    offset: 1.0,
                },
            },
            context,
        )
    };
    let rope_config = config.rope_config();
    Attention::from_gated_parts(
        config.num_attention_heads,
        config.num_key_value_heads,
        config.head_dim,
        linear(
            "q_proj",
            config.hidden_size,
            2 * config.num_attention_heads * config.head_dim,
        )?,
        linear(
            "k_proj",
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
        )?,
        linear(
            "v_proj",
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
        )?,
        linear(
            "o_proj",
            config.num_attention_heads * config.head_dim,
            config.hidden_size,
        )?,
        Some(norm("q_norm")?),
        Some(norm("k_norm")?),
        Some(B::rotary(
            RotarySpec {
                dimensions: config.rope_dimensions(),
                base: config.rope_theta(),
                traditional: false,
                max_positions: config.max_position_embeddings,
                scaling: rope_config.as_ref(),
            },
            context,
        )?),
        None,
    )
}

fn new_mlp<B: RoutedNeuralBackend>(
    config: &HybridConfig,
    prefix: &str,
    intermediate: i32,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<Mlp<B>, Error> {
    Ok(Mlp::from_parts(
        new_linear::<B>(
            config,
            &format!("{prefix}.gate_proj"),
            config.hidden_size,
            intermediate,
            false,
            context,
        )?,
        new_linear::<B>(
            config,
            &format!("{prefix}.up_proj"),
            config.hidden_size,
            intermediate,
            false,
            context,
        )?,
        new_linear::<B>(
            config,
            &format!("{prefix}.down_proj"),
            intermediate,
            config.hidden_size,
            false,
            context,
        )?,
        None,
    ))
}

fn new_linear<B: RoutedNeuralBackend>(
    config: &HybridConfig,
    prefix: &str,
    input: i32,
    output: i32,
    bias: bool,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<B::Linear, Error> {
    let weight = format!("{prefix}.weight");
    B::linear(
        LinearSpec {
            input,
            output,
            weight: parameter(&weight)?,
            bias: bias
                .then(|| parameter(format!("{prefix}.bias")))
                .transpose()?,
            format: config.linear_format(&weight),
        },
        context,
    )
}

fn parameter(name: impl AsRef<str>) -> Result<ParameterSpec, Error> {
    ParameterSpec::trainable(name.as_ref()).map_err(Error::backend)
}

fn pass<T: Tensor>(input: &T) -> ExpertPass {
    if input
        .shape()
        .get(input.shape().len().saturating_sub(2))
        .copied()
        .unwrap_or(1)
        > 1
    {
        ExpertPass::Prefill
    } else {
        ExpertPass::Decode
    }
}
