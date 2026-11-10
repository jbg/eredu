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

use super::HybridConfig;
use crate::gated_delta::GatedDeltaMixer;
use crate::shared_routed::{
    SharedRoutedGatedProduct, SharedRoutedGatedProductParameters, SharedRoutedGatedProductSpec,
};

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
    config.validate().map_err(Error::backend_source)?;
    let (schedule, _, layout) = super::parameters::execution_schedule(config)?;
    expert_bank_in_schedule(config, &schedule, &layout, layer)
}

fn expert_bank_in_schedule(
    config: &HybridConfig,
    schedule: &crate::decoder::SequentialPredictionGroups,
    layout: &eredu_runtime::ExecutionUnitLayout,
    layer: usize,
) -> Result<GroupedGatedProductSpec, Error> {
    let address = layout.address(layer).ok_or_else(|| {
        Error::backend(format!("Qwen hybrid has no declared expert layer {layer}"))
    })?;
    let root = schedule.unit_path(address.group(), address.index())?;
    let policy = if address.group() == 0 {
        *config
            .layer_schedule
            .get(address.index())
            .ok_or_else(|| Error::backend("expert unit has no declared policy"))?
    } else {
        super::HybridLayerPolicy::SelfAttention(eredu_core::AttentionPolicy::Full)
    };
    match super::BlockParameters::at(config, &root, policy)?.feed_forward {
        super::specs::FeedForwardParameters::Routed(spec) => Ok(spec.routed.experts),
        super::specs::FeedForwardParameters::Dense(_) => {
            Err(Error::backend("dense hybrid unit has no expert bank"))
        }
    }
}

fn expert_bank_spec_at(
    config: &super::HybridPolicy,
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
                eredu_nn::LinearRowLayout::equal_partitions(2)?,
            )?,
            down: standard_expert_projection(
                &down_name,
                None,
                config.linear_format(&down_name),
                eredu_nn::LinearRowLayout::Contiguous,
            )?,
        },
    )
}

/// Traverses validated target and prediction units into construction-owned banks.
pub(crate) fn expert_bank_specs(
    config: &HybridConfig,
) -> Result<std::collections::BTreeMap<usize, GroupedGatedProductSpec>, Error> {
    config.validate().map_err(Error::backend_source)?;
    if !config.is_moe() {
        return Ok(std::collections::BTreeMap::new());
    }
    let (schedule, _, layout) = super::parameters::execution_schedule(config)?;
    (0..layout.len())
        .map(|layer| {
            expert_bank_in_schedule(config, &schedule, &layout, layer).map(|bank| (layer, bank))
        })
        .collect()
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
    fn from_parameters(
        parameters: super::specs::FeedForwardParameters,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        match parameters {
            super::specs::FeedForwardParameters::Dense(spec) => {
                bind_mlp(spec, context).map(Self::Dense)
            }
            super::specs::FeedForwardParameters::Routed(parameters) => {
                SharedRoutedGatedProduct::new(
                    SharedRoutedGatedProductSpec {
                        layer,
                        bank: eredu_runtime::RoutedBankId::new(0),
                        parameters,
                    },
                    context,
                )
                .map(Self::Routed)
            }
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
    pub(crate) fn from_parameters(
        spec: super::BlockParameters,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let super::specs::FeedForwardParameters::Dense(feed_forward) = spec.feed_forward else {
            return Err(Error::backend(
                "replicated hybrid unit requires dense feed-forward",
            ));
        };
        let [input, post] = spec.norms;
        Ok(Self {
            mixer: bind_mixer(spec.mixer, context)?,
            feed_forward: bind_mlp(feed_forward, context)?,
            input_norm: B::normalization(input, context)?,
            post_attention_norm: B::normalization(post, context)?,
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
    /// Binds a complete prepared block without reconstructing parameter policy.
    pub fn from_parameters(
        spec: super::BlockParameters,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let [input, post] = spec.norms;
        Ok(Self {
            mixer: bind_mixer(spec.mixer, context)?,
            feed_forward: FeedForward::from_parameters(spec.feed_forward, layer, context)?,
            input_norm: B::normalization(input, context)?,
            post_attention_norm: B::normalization(post, context)?,
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

fn bind_mixer<B: NeuralBackend>(
    spec: super::specs::MixerParameters,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<TokenMixer<B>, Error> {
    match spec {
        super::specs::MixerParameters::Recurrent(s) => {
            GatedDeltaMixer::new(s, context).map(TokenMixer::Linear)
        }
        super::specs::MixerParameters::Attention(s) => {
            bind_attention(s, context).map(TokenMixer::Attention)
        }
    }
}

fn bind_attention<B: NeuralBackend>(
    spec: AttentionParameters,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<Attention<B>, Error> {
    let heads = spec.projection.geometry;
    Attention::from_gated_parts(
        heads.query_heads(),
        heads.key_value_heads(),
        heads.key_dimensions(),
        B::linear(spec.projection.query, context)?,
        B::linear(spec.projection.key, context)?,
        B::linear(spec.projection.value, context)?,
        B::linear(spec.projection.output, context)?,
        Some(B::normalization(spec.query_norm, context)?),
        Some(B::normalization(spec.key_norm, context)?),
        Some(B::rotary(spec.rotary, context)?),
        None,
    )
}

fn bind_mlp<B: NeuralBackend>(
    spec: eredu_nn::SplitGatedProductSpec,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<Mlp<B>, Error> {
    Ok(Mlp::from_parts(
        B::linear(spec.gate, context)?,
        B::linear(spec.up, context)?,
        B::linear(spec.down, context)?,
        spec.policy,
    ))
}

/// Interleaved query/gate attention slots and geometry shared by cold and native consumers.
#[derive(Debug, Clone)]
pub(crate) struct AttentionParameters {
    pub projection: eredu_nn::SelfAttentionProjectionSpec,
    pub query_norm: NormalizationConstructionSpec,
    pub key_norm: NormalizationConstructionSpec,
    pub rotary: RotarySpec,
}

impl AttentionParameters {
    pub const QUERY_LAYOUT: eredu_nn::AttentionQueryLayout =
        eredu_nn::AttentionQueryLayout::InterleavedGate;
    pub(super) fn new(config: &super::HybridPolicy, root: &str) -> Result<Self, Error> {
        let heads = config.attention_head_geometry()?;
        let prefix = format!("{root}.self_attn");
        let projection = |field: &str, input, output| {
            linear_spec(
                config,
                &format!("{prefix}.{field}"),
                input,
                output,
                config.attention_bias,
            )
        };
        let norm = |field: &str| {
            normalization_spec(config, format!("{prefix}.{field}.weight"), config.head_dim)
        };
        let rope_config = config.rope_config();
        let spec = Self {
            projection: eredu_nn::SelfAttentionProjectionSpec::new(
                heads,
                Self::QUERY_LAYOUT,
                projection(
                    "q_proj",
                    config.hidden_size,
                    Self::QUERY_LAYOUT.projected_width(heads)?,
                )?,
                projection("k_proj", config.hidden_size, heads.key_width())?,
                projection("v_proj", config.hidden_size, heads.value_width())?,
                projection("o_proj", heads.attended_width(), config.hidden_size)?,
            )?,
            query_norm: norm("q_norm")?,
            key_norm: norm("k_norm")?,
            rotary: RotarySpec {
                arithmetic: eredu_nn::RotaryArithmetic::Native,
                dimensions: config.rope_dimensions(),
                base: config.rope_theta(),
                traditional: false,
                algorithm: crate::rotary::normalize_algorithm(rope_config.as_ref())
                    .map_err(Error::backend)?,
            },
        };
        use eredu_nn::DeclareLogicalParameters;
        spec.logical_parameters()?;
        Ok(spec)
    }
    pub(crate) fn state_policy(
        &self,
        attention: eredu_core::AttentionPolicy,
    ) -> Result<eredu_core::cache::LayerCachePolicy, Error> {
        use eredu_nn::DeclareLogicalParameters;
        self.logical_parameters()?;
        let heads = self.projection.geometry;
        eredu_core::cache::LayerCachePolicy::key_value(
            attention,
            heads.key_value_heads(),
            heads.key_dimensions(),
        )
        .map_err(Error::backend_source)
    }
}

impl eredu_nn::DeclareLogicalParameters for AttentionParameters {
    fn logical_parameters(&self) -> Result<Vec<eredu_nn::LogicalParameterDeclaration<'_>>, Error> {
        let heads = self.projection.geometry;
        if self.projection.query_layout != Self::QUERY_LAYOUT
            || heads.key_dimensions() != heads.value_dimensions()
            || self.projection.output.output != self.projection.query.input
            || self.query_norm.dimensions != heads.key_dimensions()
            || self.key_norm.dimensions != heads.key_dimensions()
            || self.rotary.dimensions < 2
            || self.rotary.dimensions > heads.key_dimensions()
        {
            return Err(Error::backend(
                "Qwen hybrid attention residual, gate, normalization or rotary geometry mismatch",
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

pub(super) fn mlp_parameters(
    config: &super::HybridPolicy,
    prefix: &str,
    intermediate: i32,
) -> Result<eredu_nn::SplitGatedProductSpec, Error> {
    eredu_nn::SplitGatedProductSpec::new(
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
        eredu_nn::GatedProductPolicy::ordinary_silu(),
    )
}

pub(super) fn normalization_spec(
    config: &super::HybridPolicy,
    name: impl AsRef<str>,
    width: i32,
) -> Result<NormalizationConstructionSpec, Error> {
    let spec = NormalizationConstructionSpec {
        reduction: eredu_nn::NormalizationReduction::Native,
        groups: None,
        dimensions: width,
        epsilon: config.rms_norm_eps,
        scale: NormalizationScale::LearnedOffset {
            weight: parameter(name)?,
            offset: 1.0,
        },
    };
    spec.validate()?;
    Ok(spec)
}

pub(super) fn linear_spec(
    config: &super::HybridPolicy,
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

impl super::HybridPolicy {
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

pub(super) fn shared_routed_parameters(
    config: &super::HybridPolicy,
    prefix: &str,
    routed_spec: Option<GroupedGatedProductSpec>,
) -> Result<SharedRoutedGatedProductParameters, Error> {
    let prefix = format!("{prefix}.mlp");
    let router_name = format!("{prefix}.gate.weight");
    let shared = mlp_parameters(
        config,
        &format!("{prefix}.shared_expert"),
        config.shared_expert_intermediate_size,
    )?;
    let spec = SharedRoutedGatedProductParameters {
        routed: eredu_nn::RoutedGatedProductSpec {
            selector: TopKGroupSelectorSpec::new(
                config.hidden_size,
                parameter(&router_name)?,
                crate::linear_format::standard_linear_format(
                    &router_name,
                    config.linear_format(&router_name),
                )?,
                config.routing_spec()?,
            )?,
            experts: match routed_spec {
                Some(spec) => spec,
                None => expert_bank_spec_at(config, &format!("{prefix}.experts"))?,
            },
        },
        shared,
        shared_gate: linear_spec(
            config,
            &format!("{prefix}.shared_expert_gate"),
            config.hidden_size,
            1,
            false,
        )?,
        gate_activation: eredu_nn::OutputGateActivation::Sigmoid,
    };
    spec.validate()?;
    Ok(spec)
}
