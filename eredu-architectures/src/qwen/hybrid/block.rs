//! Shared Qwen3-Next/Qwen3.5 decoder block.

use eredu_nn::{
    AttentionCache, Error, GatedProductGroupLayout, GroupScoring, GroupSelectionOperator,
    GroupedGatedProductSpec, GroupedNeuralBackend, LinearOperator, LinearSpec, NeuralBackend,
    NormalizationConstructionSpec, NormalizationOperator, NormalizationScale, ParameterSpec,
    Parameterized, RotarySpec, Tensor, TopKGroupSelectionSpec, TopKGroupSelectorSpec,
};
use eredu_runtime::{
    ExpertPass, ResidentExpertProvider, RoutedExpertProvider, RoutedExpertRequest,
    RuntimeStateComponents, TensorParallelRoutedExpertProvider,
};

use crate::{
    decoder::{
        Attention, AttentionInput, FeedForwardOperator, Mlp, TensorParallelFeedForwardOperator,
    },
    linear_format::standard_expert_projection,
};

use super::{HybridConfig, HybridLayerPolicy, LinearAttention};

/// Scheduled hybrid token mixer.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub enum TokenMixer<B: NeuralBackend> {
    /// Gated-delta recurrent attention.
    Linear(LinearAttention<B>),
    /// Gated grouped-query self attention.
    Attention(Attention<B>),
}

/// Routed experts plus the always-on Qwen shared expert.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct SharedRoutedGatedProduct<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> {
    #[parameter(skip)]
    layer: usize,
    /// Learned top-k router.
    pub router: B::Selector,
    /// Packed routed expert bank.
    pub experts: B::GatedProductGroups,
    /// Always-on dense shared expert.
    pub shared_expert: Mlp<B>,
    /// Scalar gate applied to the shared expert output.
    pub shared_expert_gate: B::Linear,
}

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> SharedRoutedGatedProduct<B> {
    fn new(
        config: &HybridConfig,
        layer: usize,
        prefix: &str,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let prefix = format!("{prefix}.mlp");
        let routing = TopKGroupSelectionSpec::new(
            config.num_experts,
            config.num_experts_per_tok,
            GroupScoring::Softmax,
            config.norm_topk_prob,
        )?;
        let router_name = format!("{prefix}.gate.weight");
        let router = B::top_k_group_selector(
            TopKGroupSelectorSpec::new(
                config.hidden_size,
                parameter(&router_name)?,
                crate::linear_format::standard_linear_format(
                    &router_name,
                    config.quantization.into(),
                )?,
                routing,
            )?,
            context,
        )?;
        let experts = B::grouped_gated_product(
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
        let routes = self.router.select(input, context)?;
        let routed = provider
            .forward_grouped(
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

    fn forward_observed_with_provider<P, O>(
        &mut self,
        point: eredu_runtime::RoutedObservationPoint,
        input: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
        provider: &mut P,
        observer: &mut O,
    ) -> Result<B::Tensor, Error>
    where
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        let routes = self.router.select(input, context)?;
        let routed = provider
            .forward_grouped(
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
        let shared = shared.multiply(&shared_gate, context)?;
        let combined = routed.add(&shared, context)?;
        observer.observe_routing(eredu_runtime::RoutingObservation {
            path: point.path(),
            selected_experts: routes.group_indices(),
            selected_scores: routes.selected_scores(),
            coefficients: routes.coefficients(),
            routed_output: &routed,
            local_routed_output: None,
            reduced_routed_output: None,
            shared_output: Some(&shared),
            combined_output: Some(&combined),
            expert_count: point.expert_count(),
        })?;
        eredu_runtime::observe_and_intervene(
            observer,
            &format!("{}.output", point.path()),
            &combined,
        )
    }

    fn forward_tensor_parallel_with_provider<P>(
        &mut self,
        input: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        provider: &mut P,
    ) -> Result<eredu_runtime::RoutedExpertTensorParallelOutput<B::Tensor>, Error>
    where
        P: TensorParallelRoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        let routes = self.router.select(input, context)?;
        let routed = provider
            .forward_grouped_tensor_parallel(
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
            eredu_runtime::RoutedExpertTensorParallelOutput::Partial(routed) => {
                let (reducible, post_reduce) = routed.into_parts();
                Ok(eredu_runtime::RoutedExpertTensorParallelOutput::Partial(
                    eredu_nn::TensorParallelGroupedOutput::new(
                        reducible.add(&shared, context)?,
                        post_reduce,
                    ),
                ))
            }
        }
    }
}

/// Returns the architecture-owned routed expert specification for a target or MTP layer.
pub fn expert_bank_spec(
    config: &HybridConfig,
    layer: usize,
) -> Result<GroupedGatedProductSpec, Error> {
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
) -> Result<GroupedGatedProductSpec, Error> {
    let gate_up_name = format!("{expert_prefix}.gate_up_proj");
    let down_name = format!("{expert_prefix}.down_proj");
    GroupedGatedProductSpec::new(
        config.num_experts,
        config.hidden_size,
        config.moe_intermediate_size,
        config.hidden_size,
        eredu_nn::GatedProductPolicy::ordinary_silu(),
        GatedProductGroupLayout::Packed {
            gate_up: standard_expert_projection(
                &gate_up_name,
                None,
                config.linear_format(&gate_up_name),
            )?,
            down: standard_expert_projection(&down_name, None, config.linear_format(&down_name))?,
        },
    )
}

/// Returns the canonical expert bank at rank-local cardinality and width.
pub(crate) fn localized_expert_bank_spec(
    config: &HybridConfig,
    layer: usize,
    expert_count: i32,
    intermediate_dimensions: i32,
) -> Result<GroupedGatedProductSpec, Error> {
    expert_bank_spec(config, layer)?.with_group_geometry(expert_count, intermediate_dimensions)
}

/// Dense or routed/shared-expert feed-forward policy selected by configuration.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub enum FeedForward<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> {
    /// Dense SwiGLU.
    Dense(Mlp<B>),
    /// Routed SwiGLU plus an always-on shared expert.
    Routed(SharedRoutedGatedProduct<B>),
}

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> FeedForward<B> {
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

    fn forward_observed_with_provider<P, O>(
        &mut self,
        point: eredu_runtime::RoutedObservationPoint,
        input: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
        provider: &mut P,
        observer: &mut O,
    ) -> Result<B::Tensor, Error>
    where
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        match self {
            Self::Dense(mlp) => mlp.forward_feed_forward(input, context),
            Self::Routed(moe) => {
                moe.forward_observed_with_provider(point, input, context, provider, observer)
            }
        }
    }
}

/// One pre-normalized hybrid decoder block for every dense/MoE family form.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct Block<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> {
    /// Recurrent or full-attention policy.
    pub mixer: TokenMixer<B>,
    /// Dense or routed/shared-expert policy.
    pub feed_forward: FeedForward<B>,
    /// Learned-offset token-mixer normalization.
    pub input_norm: B::Normalization,
    /// Learned-offset feed-forward normalization.
    pub post_attention_norm: B::Normalization,
}

/// Dense-only Qwen hybrid unit used by replicated text composition.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub(crate) struct ReplicatedBlock<B: NeuralBackend> {
    mixer: TokenMixer<B>,
    feed_forward: Mlp<B>,
    input_norm: B::Normalization,
    post_attention_norm: B::Normalization,
}

impl<B: NeuralBackend> ReplicatedBlock<B> {
    pub(crate) fn new(
        config: &HybridConfig,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        if config.is_moe() {
            return Err(Error::backend(
                "replicated Qwen hybrid unit rejects routed computation",
            ));
        }
        let policy = config
            .layer_schedule
            .get(layer)
            .copied()
            .ok_or_else(|| Error::backend(format!("Qwen hybrid has no layer {layer}")))?;
        let root = format!("model.layers.{layer}");
        let mixer = match policy {
            HybridLayerPolicy::LinearAttention => {
                TokenMixer::Linear(LinearAttention::new(config, layer, context)?)
            }
            HybridLayerPolicy::SelfAttention(_) => {
                TokenMixer::Attention(new_attention(config, &root, context)?)
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
            feed_forward: new_mlp(
                config,
                &format!("{root}.mlp"),
                config.intermediate_size,
                context,
            )?,
            input_norm: norm("input_layernorm")?,
            post_attention_norm: norm("post_attention_layernorm")?,
        })
    }

    pub(crate) fn forward<S>(
        &mut self,
        hidden: &B::Tensor,
        mask: Option<&B::Tensor>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        S: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
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
        let feed_forward = self
            .feed_forward
            .forward_feed_forward(&normalized, context)?;
        hidden.add(&feed_forward, context)
    }
}

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> Block<B> {
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

    /// Executes one block while exposing the complete routed/shared contribution.
    pub fn forward_observed_with_provider<S, P, O>(
        &mut self,
        point: eredu_runtime::RoutedObservationPoint,
        hidden: &B::Tensor,
        mask: Option<&B::Tensor>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
        provider: &mut P,
        observer: &mut O,
    ) -> Result<B::Tensor, Error>
    where
        S: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
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
        let feed_forward = self.feed_forward.forward_observed_with_provider(
            point,
            &normalized,
            context,
            provider,
            observer,
        )?;
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
        P: eredu_runtime::TensorParallelRoutedExpertProvider<B>,
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

fn new_attention<B: NeuralBackend>(
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
                algorithm: crate::rotary::normalize_algorithm(rope_config.as_ref())
                    .expect("validated Qwen hybrid RoPE algorithm"),
            },
            context,
        )?),
        None,
    )
}

fn new_mlp<B: NeuralBackend>(
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

fn new_linear<B: NeuralBackend>(
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
            format: crate::linear_format::standard_linear_format(
                &weight,
                config.linear_format(&weight),
            )?,
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
