//! Shared Qwen3-Next/Qwen3.5 decoder block.

use eredu_nn::{
    AttentionCache, Error, GatedProductGroupLayout, GroupScoring, GroupedGatedProductSpec,
    GroupedNeuralBackend, LinearSpec, NeuralBackend, NormalizationConstructionSpec,
    NormalizationOperator, NormalizationScale, ParameterSpec, Parameterized, RotarySpec, Tensor,
    TopKGroupSelectionSpec, TopKGroupSelectorSpec,
};
use eredu_runtime::{
    ParameterProvider, ResidentExpertProvider, RuntimeStateComponents,
    TensorParallelParameterProvider,
};

use crate::{
    decoder::{
        Attention, AttentionInput, ComponentInstrumentation, DecoderProjectionOperator, Mlp,
        TensorParallelProjectionOperator,
    },
    linear_format::standard_expert_projection,
};

use super::{recurrent_spec, HybridConfig, HybridLayerPolicy};
use crate::gated_delta::GatedDeltaMixer;
use crate::shared_routed::{SharedRoutedGatedProduct, SharedRoutedGatedProductSpec};

/// Scheduled hybrid token mixer.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub enum TokenMixer<B: NeuralBackend> {
    /// Gated-delta recurrent attention.
    Linear(GatedDeltaMixer<B>),
    /// Gated grouped-query self attention.
    Attention(Attention<B>),
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
        routed_spec: Option<GroupedGatedProductSpec>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        if config.is_moe() {
            SharedRoutedGatedProduct::new(
                shared_routed_spec(config, layer, prefix, routed_spec)?,
                context,
            )
            .map(Self::Routed)
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
        P: ParameterProvider<B>,
        P::Error: std::fmt::Display,
    {
        match self {
            Self::Dense(mlp) => mlp.forward_feed_forward(input, context),
            Self::Routed(moe) => moe.forward_with_provider(input, context, provider),
        }
    }

    fn forward_observed_with_provider<P, O>(
        &mut self,
        point: eredu_runtime::RoutedObservationPoints,
        input: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
        provider: &mut P,
        observer: &mut O,
    ) -> Result<B::Tensor, Error>
    where
        P: ParameterProvider<B>,
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
            HybridLayerPolicy::LinearAttention => TokenMixer::Linear(GatedDeltaMixer::new(
                recurrent_spec(config, layer)?,
                context,
            )?),
            HybridLayerPolicy::SelfAttention(_) => {
                TokenMixer::Attention(new_attention(config, &root, context)?)
            }
        };
        let norm = |field: &str| {
            B::normalization(
                NormalizationConstructionSpec {
                    groups: None,
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
        self.forward_instrumented(
            hidden,
            mask,
            state,
            context,
            &mut ComponentInstrumentation::disabled(),
        )
    }

    pub(crate) fn forward_instrumented<S>(
        &mut self,
        hidden: &B::Tensor,
        mask: Option<&B::Tensor>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error>
    where
        S: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
    {
        let hidden = forward_mixer::<B, S>(
            &mut self.mixer,
            &mut self.input_norm,
            hidden,
            mask,
            state,
            context,
            instrumentation,
        )?;
        let normalized = instrumentation.apply(
            "feed_forward.input",
            self.post_attention_norm.forward(&hidden, context)?,
        )?;
        let feed_forward = self.feed_forward.forward_feed_forward_observed(
            &normalized,
            context,
            instrumentation,
        )?;
        finish_feed_forward::<B>(&hidden, feed_forward, context, instrumentation)
    }
}

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> Block<B> {
    /// Builds one global-geometry physical decoder layer.
    pub fn new(
        config: &HybridConfig,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        Self::new_with_routed_spec(config, layer, None, context)
    }

    pub(crate) fn new_with_routed_spec(
        config: &HybridConfig,
        layer: usize,
        routed_spec: Option<GroupedGatedProductSpec>,
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
            routed_spec,
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
            None,
            context,
        )
    }

    fn new_at(
        config: &HybridConfig,
        layer: usize,
        root: &str,
        policy: HybridLayerPolicy,
        routed_spec: Option<GroupedGatedProductSpec>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let mixer = match policy {
            HybridLayerPolicy::LinearAttention => TokenMixer::Linear(GatedDeltaMixer::new(
                recurrent_spec(config, layer)?,
                context,
            )?),
            HybridLayerPolicy::SelfAttention(_) => {
                TokenMixer::Attention(new_attention(config, root, context)?)
            }
        };
        let norm = |field: &str| {
            B::normalization(
                NormalizationConstructionSpec {
                    groups: None,
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
            feed_forward: FeedForward::new(config, layer, root, routed_spec, context)?,
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
        P: ParameterProvider<B>,
        P::Error: std::fmt::Display,
    {
        self.forward_instrumented_with_feed_forward(
            hidden,
            mask,
            state,
            context,
            &mut ComponentInstrumentation::disabled(),
            |ff, input, context, _| ff.forward_with_provider(input, context, provider),
        )
    }

    /// Executes one block while exposing the complete routed/shared contribution.
    pub fn forward_observed_with_provider<S, P, O>(
        &mut self,
        point: eredu_runtime::RoutedObservationPoints,
        hidden: &B::Tensor,
        mask: Option<&B::Tensor>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
        provider: &mut P,
        observer: &mut O,
    ) -> Result<B::Tensor, Error>
    where
        S: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
        P: ParameterProvider<B>,
        P::Error: std::fmt::Display,
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        self.forward_instrumented_with_feed_forward(
            hidden,
            mask,
            state,
            context,
            &mut ComponentInstrumentation::disabled(),
            |ff, input, context, _| {
                ff.forward_observed_with_provider(point, input, context, provider, observer)
            },
        )
    }

    /// Component and routed observations use the same architecture-owned boundaries.
    pub(crate) fn forward_components_with_provider<S, P, O>(
        &mut self,
        path: &str,
        point: eredu_runtime::RoutedObservationPoints,
        hidden: &B::Tensor,
        mask: Option<&B::Tensor>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
        provider: &mut P,
        observer: &mut O,
    ) -> Result<B::Tensor, Error>
    where
        S: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
        P: ParameterProvider<B>,
        P::Error: std::fmt::Display,
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        let mut borrowed = eredu_runtime::BorrowedActivationObserver(observer);
        let mut instrumentation = ComponentInstrumentation::new(path, &mut borrowed);
        self.forward_instrumented_with_feed_forward(
            hidden,
            mask,
            state,
            context,
            &mut instrumentation,
            |ff, input, context, instrumentation| match ff {
                FeedForward::Dense(mlp) => {
                    mlp.forward_feed_forward_observed(input, context, instrumentation)
                }
                FeedForward::Routed(moe) => match instrumentation.observer() {
                    Some(observer) => moe
                        .forward_observed_with_provider(point, input, context, provider, observer),
                    None => moe.forward_with_provider(input, context, provider),
                },
            },
        )
    }

    fn forward_instrumented_with_feed_forward<S, F>(
        &mut self,
        hidden: &B::Tensor,
        mask: Option<&B::Tensor>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
        feed_forward: F,
    ) -> Result<B::Tensor, Error>
    where
        S: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
        F: FnOnce(
            &mut FeedForward<B>,
            &B::Tensor,
            &<B::Tensor as Tensor>::Context,
            &mut ComponentInstrumentation<'_, B::Tensor>,
        ) -> Result<B::Tensor, Error>,
    {
        let hidden = forward_mixer::<B, S>(
            &mut self.mixer,
            &mut self.input_norm,
            hidden,
            mask,
            state,
            context,
            instrumentation,
        )?;
        let normalized = instrumentation.apply(
            "feed_forward.input",
            self.post_attention_norm.forward(&hidden, context)?,
        )?;
        let output = feed_forward(
            &mut self.feed_forward,
            &normalized,
            context,
            instrumentation,
        )?;
        finish_feed_forward::<B>(&hidden, output, context, instrumentation)
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
        P: eredu_runtime::TensorParallelParameterProvider<B>,
        P::Error: std::fmt::Display,
    {
        self.forward_parallel_instrumented(
            hidden,
            mask,
            state,
            parallel,
            context,
            provider,
            &mut ComponentInstrumentation::disabled(),
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_components_parallel_with_provider<S, P, O>(
        &mut self,
        path: &str,
        points: eredu_runtime::RoutedObservationPoints,
        hidden: &B::Tensor,
        mask: Option<&B::Tensor>,
        state: &mut S,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        provider: &mut P,
        observer: &mut O,
    ) -> Result<B::Tensor, Error>
    where
        S: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
        P: TensorParallelParameterProvider<B>,
        P::Error: std::fmt::Display,
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        let mut borrowed = eredu_runtime::BorrowedActivationObserver(observer);
        self.forward_parallel_instrumented(
            hidden,
            mask,
            state,
            parallel,
            context,
            provider,
            &mut ComponentInstrumentation::new(path, &mut borrowed),
            Some(points),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_parallel_instrumented<S, P>(
        &mut self,
        hidden: &B::Tensor,
        mask: Option<&B::Tensor>,
        state: &mut S,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        provider: &mut P,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
        points: Option<eredu_runtime::RoutedObservationPoints>,
    ) -> Result<B::Tensor, Error>
    where
        S: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
        P: TensorParallelParameterProvider<B>,
        P::Error: std::fmt::Display,
    {
        let normalized = self.input_norm.forward(hidden, context)?;
        let (mixed, boundary) = match &mut self.mixer {
            TokenMixer::Linear(linear) => {
                let normalized = instrumentation.apply("mixer.input", normalized)?;
                (
                    linear.forward_parallel_instrumented(
                        &normalized,
                        state,
                        parallel,
                        context,
                        instrumentation,
                    )?,
                    "mixer",
                )
            }
            TokenMixer::Attention(attention) => {
                let normalized = instrumentation.apply("attention.input", normalized)?;
                (
                    attention.forward_instrumented(
                        AttentionInput {
                            selected_positions: None,
                            hidden: &normalized,
                            mask,
                            cache: Some(&mut *state),
                            allow_sliding_prefill: true,
                            rotary_position: None,
                        },
                        Some(parallel),
                        context,
                        instrumentation,
                    )?,
                    "attention",
                )
            }
        };
        let mixed = instrumentation.apply(
            if boundary == "attention" {
                "attention.write"
            } else {
                "mixer.write"
            },
            mixed,
        )?;
        let mixed = instrumentation.apply(
            if boundary == "attention" {
                "attention.output"
            } else {
                "mixer.output"
            },
            mixed,
        )?;
        let hidden = instrumentation.apply(
            if boundary == "attention" {
                "attention.residual"
            } else {
                "mixer.residual"
            },
            hidden.add(&mixed, context)?,
        )?;
        let normalized = instrumentation.apply(
            "feed_forward.input",
            self.post_attention_norm.forward(&hidden, context)?,
        )?;
        let feed_forward = match &mut self.feed_forward {
            FeedForward::Dense(mlp) => mlp.forward_feed_forward_parallel_observed(
                &normalized,
                parallel,
                context,
                instrumentation,
            )?,
            FeedForward::Routed(moe) => {
                if let Some(observer) = instrumentation.observer() {
                    moe.forward_tensor_parallel_observed_with_provider(
                        points.expect("observed routed block declares its routing points"),
                        &normalized,
                        parallel,
                        context,
                        provider,
                        observer,
                    )?
                } else {
                    let output = moe.forward_tensor_parallel_with_provider(
                        &normalized,
                        parallel,
                        context,
                        provider,
                    )?;
                    eredu_runtime::reduce_routed_expert_tensor_parallel::<B>(
                        output, parallel, context,
                    )?
                }
            }
        };
        finish_feed_forward::<B>(&hidden, feed_forward, context, instrumentation)
    }
}

fn new_attention<B: NeuralBackend>(
    config: &HybridConfig,
    root: &str,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<Attention<B>, Error> {
    let prefix = format!("{root}.self_attn");
    let [query, key, value, output] = attention_projection_specs(config, root)?;
    let norm = |field: &str| {
        B::normalization(
            NormalizationConstructionSpec {
                groups: None,
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
        B::linear(query, context)?,
        B::linear(key, context)?,
        B::linear(value, context)?,
        B::linear(output, context)?,
        Some(norm("q_norm")?),
        Some(norm("k_norm")?),
        Some(B::rotary(
            RotarySpec {
                arithmetic: eredu_nn::RotaryArithmetic::Native,
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
    let [gate, up, down] = mlp_projection_specs(config, prefix, intermediate)?;
    Ok(Mlp::from_parts(
        B::linear(gate, context)?,
        B::linear(up, context)?,
        B::linear(down, context)?,
        None,
    ))
}

pub(super) fn attention_projection_specs(
    config: &HybridConfig,
    root: &str,
) -> Result<[LinearSpec; 4], Error> {
    let prefix = format!("{root}.self_attn");
    let query = config
        .num_attention_heads
        .checked_mul(config.head_dim)
        .ok_or_else(|| Error::backend("query width overflow"))?;
    let kv = config
        .num_key_value_heads
        .checked_mul(config.head_dim)
        .ok_or_else(|| Error::backend("key/value width overflow"))?;
    Ok([
        linear_spec(
            config,
            &format!("{prefix}.q_proj"),
            config.hidden_size,
            query
                .checked_mul(2)
                .ok_or_else(|| Error::backend("gated query width overflow"))?,
            config.attention_bias,
        )?,
        linear_spec(
            config,
            &format!("{prefix}.k_proj"),
            config.hidden_size,
            kv,
            config.attention_bias,
        )?,
        linear_spec(
            config,
            &format!("{prefix}.v_proj"),
            config.hidden_size,
            kv,
            config.attention_bias,
        )?,
        linear_spec(
            config,
            &format!("{prefix}.o_proj"),
            query,
            config.hidden_size,
            config.attention_bias,
        )?,
    ])
}

pub(super) fn mlp_projection_specs(
    config: &HybridConfig,
    prefix: &str,
    intermediate: i32,
) -> Result<[LinearSpec; 3], Error> {
    Ok([
        linear_spec(
            config,
            &format!("{prefix}.gate_proj"),
            config.hidden_size,
            intermediate,
            false,
        )?,
        linear_spec(
            config,
            &format!("{prefix}.up_proj"),
            config.hidden_size,
            intermediate,
            false,
        )?,
        linear_spec(
            config,
            &format!("{prefix}.down_proj"),
            intermediate,
            config.hidden_size,
            false,
        )?,
    ])
}

pub(super) fn linear_spec(
    config: &HybridConfig,
    prefix: &str,
    input: i32,
    output: i32,
    bias: bool,
) -> Result<LinearSpec, Error> {
    let weight = format!("{prefix}.weight");
    Ok(LinearSpec {
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
    })
}

fn parameter(name: impl AsRef<str>) -> Result<ParameterSpec, Error> {
    ParameterSpec::trainable(name.as_ref()).map_err(Error::backend)
}

impl HybridConfig {
    /// Selection policy shared by module construction and intervention discovery.
    pub(crate) fn routing_spec(&self) -> Result<TopKGroupSelectionSpec, Error> {
        TopKGroupSelectionSpec::new(
            self.num_experts,
            self.num_experts_per_tok,
            GroupScoring::Softmax,
            self.norm_topk_prob,
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn forward_mixer<B: NeuralBackend, S>(
    mixer: &mut TokenMixer<B>,
    norm: &mut B::Normalization,
    hidden: &B::Tensor,
    mask: Option<&B::Tensor>,
    state: &mut S,
    context: &<B::Tensor as Tensor>::Context,
    instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
) -> Result<B::Tensor, Error>
where
    S: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
{
    let normalized = norm.forward(hidden, context)?;
    let (mixed, boundary) = match mixer {
        TokenMixer::Linear(linear) => {
            let normalized = instrumentation.apply("mixer.input", normalized)?;
            (
                linear.forward_instrumented(&normalized, state, context, instrumentation)?,
                "mixer",
            )
        }
        TokenMixer::Attention(attention) => {
            let normalized = instrumentation.apply("attention.input", normalized)?;
            (
                attention.forward_instrumented(
                    AttentionInput {
                        selected_positions: None,
                        hidden: &normalized,
                        mask,
                        cache: Some(state),
                        allow_sliding_prefill: true,
                        rotary_position: None,
                    },
                    None,
                    context,
                    instrumentation,
                )?,
                "attention",
            )
        }
    };
    let (write, output, residual) = if boundary == "attention" {
        ("attention.write", "attention.output", "attention.residual")
    } else {
        ("mixer.write", "mixer.output", "mixer.residual")
    };
    let mixed = instrumentation.apply(write, mixed)?;
    let mixed = instrumentation.apply(output, mixed)?;
    instrumentation.apply(residual, hidden.add(&mixed, context)?)
}
fn finish_feed_forward<B: NeuralBackend>(
    hidden: &B::Tensor,
    output: B::Tensor,
    context: &<B::Tensor as Tensor>::Context,
    instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
) -> Result<B::Tensor, Error> {
    let output = instrumentation.apply("feed_forward.write", output)?;
    let output = instrumentation.apply("feed_forward.output", output)?;
    instrumentation.apply("feed_forward.residual", hidden.add(&output, context)?)
}

fn shared_routed_spec(
    config: &HybridConfig,
    layer: usize,
    prefix: &str,
    routed_spec: Option<GroupedGatedProductSpec>,
) -> Result<SharedRoutedGatedProductSpec, Error> {
    let prefix = format!("{prefix}.mlp");
    let router_name = format!("{prefix}.gate.weight");
    Ok(SharedRoutedGatedProductSpec {
        layer,
        bank: eredu_runtime::RoutedBankId::new(0),
        router: TopKGroupSelectorSpec::new(
            config.hidden_size,
            parameter(&router_name)?,
            crate::linear_format::standard_linear_format(&router_name, config.quantization.into())?,
            config.routing_spec()?,
        )?,
        experts: match routed_spec {
            Some(spec) => spec,
            None => expert_bank_spec_at(config, &format!("{prefix}.experts"))?,
        },
        shared: mlp_projection_specs(
            config,
            &format!("{prefix}.shared_expert"),
            config.shared_expert_intermediate_size,
        )?,
        shared_policy: eredu_nn::GatedProductPolicy::ordinary_silu(),
        shared_gate: linear_spec(
            config,
            &format!("{prefix}.shared_expert_gate"),
            config.hidden_size,
            1,
            false,
        )?,
        gate_activation: eredu_nn::OutputGateActivation::Sigmoid,
    })
}
