//! Backend-neutral Gemma 4 decoder equations.

use std::collections::HashMap;

use eredu_core::AttentionPolicy;
use eredu_nn::{
    AttentionCache, AttentionRequest, AttentionStateSource, AttentionValueSource, Error,
    GatedProductGroupLayout, GroupScoring, GroupSelectionOperator, GroupedGatedProductSpec,
    GroupedNeuralBackend, LinearOperator, LinearSpec, NeuralBackend, NormalizationConstructionSpec,
    NormalizationOperator, Parameter, ParameterSpec, Parameterized, RotaryOperator, RotaryPosition,
    RotarySpec, RotarySubspace, SelectorInputTransformSpec, Tensor, TopKGroupSelectionSpec,
    TopKGroupSelectorSpec,
};
use eredu_runtime::{
    ExpertPass, ResidentExpertProvider, RoutedExpertProvider, RoutedExpertRequest,
    TensorParallelRoutedExpertProvider,
};

use crate::decoder::ComponentInstrumentation;
use crate::linear_format::standard_expert_projection;

use super::{FeedForwardPolicy, LayerPolicy, ModelArgs};

/// Shared normalized key/value states keyed by exact attention policy.
pub type SharedAttentionStates<T> = HashMap<AttentionPolicy, (T, T)>;

/// Stateful attention request for one Gemma 4 block.
pub struct AttentionInput<'a, T, C> {
    /// Pre-normalized hidden states.
    pub hidden: &'a T,
    /// Optional additive or boolean attention mask.
    pub mask: Option<&'a T>,
    /// Optional mutable history: the local cache for a publisher, or the
    /// publisher/receiver cache for a shared-state consumer.
    pub cache: Option<&'a mut C>,
    /// Shared publications from earlier compatible layers.
    pub shared: &'a mut SharedAttentionStates<T>,
    /// Optional caller-provided explicit rotary positions.
    pub rotary_position: Option<RotaryPosition<'a, T>>,
}

/// Gemma 4 grouped-query attention with local, publishing, or shared KV state.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct Attention<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> {
    #[parameter(skip)]
    query_heads: i32,
    #[parameter(skip)]
    key_value_heads: i32,
    #[parameter(skip)]
    rotary_dimensions: i32,
    #[parameter(skip)]
    policy: AttentionPolicy,
    #[parameter(skip)]
    state_source: AttentionStateSource,
    /// Query projection.
    pub query: B::Linear,
    /// Key projection, absent for shared-state consumers.
    pub key: Option<B::Linear>,
    /// Value projection, absent for key-as-value owners and shared consumers.
    pub value: Option<B::Linear>,
    /// Output projection.
    pub output: B::Linear,
    /// Learned per-head query normalization.
    pub query_norm: B::Normalization,
    /// Learned per-head key normalization for state owners.
    pub key_norm: Option<B::Normalization>,
    /// Partial or full rotary operator.
    pub rotary: B::Rotary,
}

impl<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> Attention<B> {
    /// Builds one unloaded attention unit from normalized layer policy.
    pub fn new(
        args: &ModelArgs,
        layer: usize,
        policy: LayerPolicy,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        Self::new_at(args, layer, policy, "model.language_model.layers", context)
    }

    /// Builds one attention unit under an explicit architecture-owned layer root.
    pub fn new_at(
        args: &ModelArgs,
        layer: usize,
        policy: LayerPolicy,
        layer_root: &str,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let prefix = format!("{layer_root}.{layer}.self_attn");
        let head_dim = policy.head_dim.get() as i32;
        let kv_heads = policy.num_key_value_heads.get() as i32;
        let linear = |field: &str, input: i32, output: i32| {
            let weight_name = format!("{prefix}.{field}.weight");
            B::linear(
                LinearSpec {
                    input,
                    output,
                    weight: ParameterSpec::trainable(&weight_name).map_err(Error::backend)?,
                    bias: args
                        .attention_bias
                        .then(|| ParameterSpec::trainable(format!("{prefix}.{field}.bias")))
                        .transpose()
                        .map_err(Error::backend)?,
                    format: crate::linear_format::standard_linear_format(
                        &weight_name,
                        args.linear_format_for(&weight_name),
                    )?,
                },
                context,
            )
        };
        let owns_state = policy.key_value.owns_state();
        let partial_dimensions =
            partial_rotary_dimensions(head_dim, args.rope_scaling_for(policy.attention));
        Ok(Self {
            query_heads: args.num_attention_heads,
            key_value_heads: kv_heads,
            rotary_dimensions: partial_dimensions,
            policy: policy.attention,
            state_source: policy.key_value,
            query: linear(
                "q_proj",
                args.hidden_size,
                args.num_attention_heads * head_dim,
            )?,
            key: owns_state
                .then(|| linear("k_proj", args.hidden_size, kv_heads * head_dim))
                .transpose()?,
            value: (policy.key_value.value() == Some(AttentionValueSource::Projected))
                .then(|| linear("v_proj", args.hidden_size, kv_heads * head_dim))
                .transpose()?,
            output: linear(
                "o_proj",
                args.num_attention_heads * head_dim,
                args.hidden_size,
            )?,
            query_norm: B::normalization(
                NormalizationConstructionSpec::learned(
                    head_dim,
                    args.rms_norm_eps,
                    ParameterSpec::trainable(format!("{prefix}.q_norm.weight"))
                        .map_err(Error::backend)?,
                ),
                context,
            )?,
            key_norm: owns_state
                .then(|| {
                    B::normalization(
                        NormalizationConstructionSpec::learned(
                            head_dim,
                            args.rms_norm_eps,
                            ParameterSpec::trainable(format!("{prefix}.k_norm.weight"))
                                .map_err(Error::backend)?,
                        ),
                        context,
                    )
                })
                .transpose()?,
            rotary: B::rotary(
                RotarySpec {
                    arithmetic: eredu_nn::RotaryArithmetic::Native,
                    dimensions: partial_dimensions,
                    base: args.rope_theta_for(policy.attention),
                    traditional: false,
                    algorithm: crate::rotary::normalize_algorithm(
                        args.rope_scaling_for(policy.attention),
                    )
                    .expect("validated Gemma 4 RoPE algorithm"),
                },
                context,
            )?,
        })
    }

    fn attend<C: AttentionCache<B::Tensor>>(
        &mut self,
        input: AttentionInput<'_, B::Tensor, C>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let batch = input.hidden.dim(0);
        let sequence = input.hidden.dim(1);
        let mut cache = input.cache;
        let reshape = |value: B::Tensor, heads: i32| {
            value
                .reshape(&[batch, sequence, heads, -1], context)?
                .transpose_axes(&[0, 2, 1, 3], context)
        };
        let queries = reshape(self.query.forward(input.hidden, context)?, self.query_heads)?;
        let queries = self.query_norm.forward(&queries, context)?;
        let position = match input.rotary_position {
            Some(position) => position,
            None => {
                let offset = cache.as_ref().map_or(0, |cache| cache.offset());
                // A shared history has already appended this submission in its
                // publisher. Queries still start at the submission's frontier.
                let offset = if self.state_source == AttentionStateSource::Shared && cache.is_some()
                {
                    offset
                        .checked_sub(sequence)
                        .filter(|offset| *offset >= 0)
                        .ok_or_else(|| {
                            Error::backend("shared Gemma history precedes this submission")
                        })?
                } else {
                    offset
                };
                RotaryPosition::Offset(offset)
            }
        };
        let queries = self.rotary.forward_subspace(
            &queries,
            RotarySubspace::Range {
                start: 0,
                dimensions: self.rotary_dimensions,
            },
            position,
            context,
        )?;
        let (keys, values) = if self.state_source == AttentionStateSource::Shared {
            input.shared.get(&self.policy).cloned().ok_or_else(|| {
                Error::backend(format!(
                    "missing shared attention state for {:?}",
                    self.policy
                ))
            })?
        } else {
            let key_projection = self
                .key
                .as_mut()
                .ok_or_else(|| Error::backend("state-owning attention has no key projection"))?
                .forward(input.hidden, context)?;
            let value_projection = match self.state_source.value() {
                Some(AttentionValueSource::ReuseKey) => key_projection.clone(),
                Some(AttentionValueSource::Projected) => self
                    .value
                    .as_mut()
                    .ok_or_else(|| {
                        Error::backend("projected-value attention has no value projection")
                    })?
                    .forward(input.hidden, context)?,
                None => return Err(Error::backend("invalid shared state owner")),
            };
            let keys = reshape(key_projection, self.key_value_heads)?;
            let keys = self
                .key_norm
                .as_mut()
                .ok_or_else(|| Error::backend("state-owning attention has no key norm"))?
                .forward(&keys, context)?;
            let keys = self.rotary.forward_subspace(
                &keys,
                RotarySubspace::Range {
                    start: 0,
                    dimensions: self.rotary_dimensions,
                },
                position,
                context,
            )?;
            let values =
                value_projection.reshape(&[batch, sequence, self.key_value_heads, -1], context)?;
            let values = B::rms_norm_without_weight(&values, 1e-6, context)?
                .transpose_axes(&[0, 2, 1, 3], context)?;
            let (keys, values) = match cache.as_deref_mut() {
                Some(cache) => cache.update_for_attention(keys, values, context)?,
                None => (keys, values),
            };
            if self.state_source.publishes_state() {
                input
                    .shared
                    .insert(self.policy, (keys.clone(), values.clone()));
            }
            (keys, values)
        };
        // A paged publication contains only this submission's tensors. Its
        // older history remains owned by the publisher or pipeline receiver.
        let attended = match cache {
            Some(cache) => cache.attention(
                AttentionRequest {
                    arithmetic: eredu_nn::AttentionArithmetic::Fused,
                    queries,
                    keys,
                    values,
                    scale: 1.0,
                    mask: input.mask,
                    softcap: None,
                    sinks: None,
                },
                context,
            )?,
            _ => B::attention(queries, keys, values, 1.0, input.mask, context)?,
        };
        let attended = attended
            .transpose_axes(&[0, 2, 1, 3], context)?
            .reshape(&[batch, sequence, -1], context)?;
        Ok(attended)
    }

    /// Executes attention and publishes normalized cached state when required.
    pub fn forward<C: AttentionCache<B::Tensor>>(
        &mut self,
        input: AttentionInput<'_, B::Tensor, C>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.forward_instrumented(
            input,
            None,
            context,
            &mut ComponentInstrumentation::disabled(),
        )
    }

    /// Executes rank-local attention followed by the collective output projection.
    pub fn forward_parallel<C: AttentionCache<B::Tensor>>(
        &mut self,
        input: AttentionInput<'_, B::Tensor, C>,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        B: GroupedNeuralBackend,
    {
        self.forward_instrumented(
            input,
            Some(parallel),
            context,
            &mut ComponentInstrumentation::disabled(),
        )
    }

    fn forward_instrumented<C: AttentionCache<B::Tensor>>(
        &mut self,
        input: AttentionInput<'_, B::Tensor, C>,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error> {
        let attended = self.attend(input, context)?;
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

fn partial_rotary_dimensions(
    head_dim: i32,
    scaling: Option<&std::collections::HashMap<String, crate::rotary::RopeValue>>,
) -> i32 {
    if matches!(
        scaling.and_then(|scaling| scaling.get("rope_type")),
        Some(crate::rotary::RopeValue::String(kind)) if kind == "proportional"
    ) {
        return head_dim;
    }
    let factor = scaling
        .and_then(|scaling| scaling.get("partial_rotary_factor"))
        .and_then(|value| match value {
            crate::rotary::RopeValue::Float(value) => Some(*value),
            crate::rotary::RopeValue::String(value) => value.parse().ok(),
            crate::rotary::RopeValue::Bool(_) => None,
        })
        .unwrap_or(1.0);
    ((head_dim as f32 * factor).round() as i32)
        .max(2)
        .min(head_dim)
        & !1
}

/// Dense GELU-gated feed-forward branch.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct DenseMlp<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> {
    /// GELU gate projection.
    pub gate: B::Linear,
    /// Multiplicative up projection.
    pub up: B::Linear,
    /// Output projection.
    pub down: B::Linear,
}

impl<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> DenseMlp<B> {
    fn new(
        args: &ModelArgs,
        layer: usize,
        intermediate: i32,
        layer_root: &str,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let prefix = format!("{layer_root}.{layer}.mlp");
        let linear = |field: &str, input, output| {
            let weight_name = format!("{prefix}.{field}.weight");
            B::linear(
                LinearSpec {
                    input,
                    output,
                    weight: ParameterSpec::trainable(&weight_name).map_err(Error::backend)?,
                    bias: None,
                    format: crate::linear_format::standard_linear_format(
                        &weight_name,
                        args.linear_format_for(&weight_name),
                    )?,
                },
                context,
            )
        };
        Ok(Self {
            gate: linear("gate_proj", args.hidden_size, intermediate)?,
            up: linear("up_proj", args.hidden_size, intermediate)?,
            down: linear("down_proj", intermediate, args.hidden_size)?,
        })
    }

    fn forward_instrumented(
        &mut self,
        input: &B::Tensor,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error> {
        let gate = self.gate.forward(input, context)?;
        let gate = B::Tensor::gelu(&gate, context)?;
        let up = self.up.forward(input, context)?;
        let hidden = gate.multiply(&up, context)?;
        let hidden = instrumentation.apply("dense_feed_forward.units", hidden)?;
        instrumentation.project::<B>(
            "dense_feed_forward.write_input",
            &mut self.down,
            &hidden,
            parallel,
            context,
        )
    }
}

/// Dense Gemma 4 transformer block. Sparse routing is constructed by the
/// routed block wrapper so the dense residual is shared exactly.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct DenseBlock<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> {
    #[parameter(skip)]
    layer: usize,
    /// Stateful self attention.
    pub attention: Attention<B>,
    /// Dense gated-GELU branch.
    pub mlp: DenseMlp<B>,
    /// Optional selected-softmax sparse router.
    pub router: Option<B::Selector>,
    /// Optional packed GELU-gated expert bank.
    pub experts: Option<B::GatedProductGroups>,
    /// Pre-attention normalization.
    pub input_norm: B::Normalization,
    /// Attention-delta normalization.
    pub post_attention_norm: B::Normalization,
    /// Pre-feed-forward normalization.
    pub pre_feed_forward_norm: B::Normalization,
    /// Feed-forward-delta normalization.
    pub post_feed_forward_norm: B::Normalization,
    /// Dense branch norm before combining a sparse residual.
    pub post_feed_forward_norm_1: Option<B::Normalization>,
    /// Sparse branch input norm.
    pub pre_feed_forward_norm_2: Option<B::Normalization>,
    /// Sparse branch output norm.
    pub post_feed_forward_norm_2: Option<B::Normalization>,
    /// Optional per-layer media gate.
    pub per_layer_gate: Option<B::Linear>,
    /// Optional per-layer media projection.
    pub per_layer_projection: Option<B::Linear>,
    /// Optional per-layer media delta normalization.
    pub per_layer_norm: Option<B::Normalization>,
    /// Learned output scalar.
    pub layer_scalar: Parameter<B::Tensor>,
}

/// One decoder-block request.
pub struct BlockInput<'a, T, C> {
    /// Residual hidden states.
    pub hidden: &'a T,
    /// Optional attention mask.
    pub mask: Option<&'a T>,
    /// Optional state-owner cache.
    pub cache: Option<&'a mut C>,
    /// Pass-local shared KV publications.
    pub shared: &'a mut SharedAttentionStates<T>,
    /// Optional per-layer prepared media embedding.
    pub per_layer_input: Option<&'a T>,
    /// Optional explicit rotary positions.
    pub rotary_position: Option<RotaryPosition<'a, T>>,
}

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> DenseBlock<B> {
    /// Builds one unloaded dense block.
    pub fn new(
        args: &ModelArgs,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        Self::new_at(args, layer, "model.language_model.layers", context)
    }

    /// Builds one ordinary block under an explicit released layer root.
    pub fn new_at(
        args: &ModelArgs,
        layer: usize,
        layer_root: &str,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        Self::new_at_with_routed_spec(args, layer, layer_root, None, context)
    }

    pub(crate) fn new_at_with_routed_spec(
        args: &ModelArgs,
        layer: usize,
        layer_root: &str,
        routed_spec: Option<GroupedGatedProductSpec>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let policy = args
            .layer_policy(layer)
            .ok_or_else(|| Error::backend(format!("missing Gemma 4 layer policy {layer}")))?;
        let prefix = format!("{layer_root}.{layer}");
        let norm = |field: &str, dimensions| {
            B::normalization(
                NormalizationConstructionSpec::learned(
                    dimensions,
                    args.rms_norm_eps,
                    ParameterSpec::trainable(format!("{prefix}.{field}.weight"))
                        .map_err(Error::backend)?,
                ),
                context,
            )
        };
        let media_linear = |field: &str, input, output| {
            let weight_name = format!("{prefix}.{field}.weight");
            B::linear(
                LinearSpec {
                    input,
                    output,
                    weight: ParameterSpec::trainable(&weight_name).map_err(Error::backend)?,
                    bias: None,
                    format: crate::linear_format::standard_linear_format(
                        &weight_name,
                        args.linear_format_for(&weight_name),
                    )?,
                },
                context,
            )
        };
        let media_width = args.hidden_size_per_layer_input;
        let sparse = policy.feed_forward == FeedForwardPolicy::DenseWithSparseMoe;
        let (router, experts) = if sparse {
            let expert_count = args
                .num_experts
                .ok_or_else(|| Error::backend("Gemma 4 sparse block has no expert count"))?;
            let top_k = args
                .top_k_experts
                .ok_or_else(|| Error::backend("Gemma 4 sparse block has no top-k count"))?;
            let router_prefix = format!("{prefix}.router");
            let router_weight = format!("{router_prefix}.proj.weight");
            let selector = TopKGroupSelectorSpec::new(
                args.hidden_size,
                ParameterSpec::trainable(&router_weight).map_err(Error::backend)?,
                crate::linear_format::standard_linear_format(
                    &router_weight,
                    args.linear_format_for(&router_weight),
                )?,
                TopKGroupSelectionSpec::new(
                    expert_count,
                    top_k,
                    GroupScoring::SelectedSoftmax,
                    false,
                )?,
            )?
            .with_input_transform(SelectorInputTransformSpec::new(
                args.rms_norm_eps,
                ParameterSpec::trainable(format!("{router_prefix}.scale"))
                    .map_err(Error::backend)?,
                true,
            )?)
            .with_coefficient_scale(
                ParameterSpec::trainable(format!("{router_prefix}.per_expert_scale"))
                    .map_err(Error::backend)?,
            );
            let router = B::top_k_group_selector(selector, context)?;
            let spec = match routed_spec {
                Some(spec) => spec,
                None => expert_bank_spec_at(args, &format!("{prefix}.experts.switch_glu"))?,
            };
            let experts = B::grouped_gated_product(spec, context)?;
            (Some(router), Some(experts))
        } else {
            (None, None)
        };
        Ok(Self {
            layer,
            attention: Attention::new_at(args, layer, policy, layer_root, context)?,
            mlp: DenseMlp::new(
                args,
                layer,
                policy.intermediate_size.get() as i32,
                layer_root,
                context,
            )?,
            router,
            experts,
            input_norm: norm("input_layernorm", args.hidden_size)?,
            post_attention_norm: norm("post_attention_layernorm", args.hidden_size)?,
            pre_feed_forward_norm: norm("pre_feedforward_layernorm", args.hidden_size)?,
            post_feed_forward_norm: norm("post_feedforward_layernorm", args.hidden_size)?,
            post_feed_forward_norm_1: sparse
                .then(|| norm("post_feedforward_layernorm_1", args.hidden_size))
                .transpose()?,
            pre_feed_forward_norm_2: sparse
                .then(|| norm("pre_feedforward_layernorm_2", args.hidden_size))
                .transpose()?,
            post_feed_forward_norm_2: sparse
                .then(|| norm("post_feedforward_layernorm_2", args.hidden_size))
                .transpose()?,
            per_layer_gate: (media_width > 0)
                .then(|| media_linear("per_layer_input_gate", args.hidden_size, media_width))
                .transpose()?,
            per_layer_projection: (media_width > 0)
                .then(|| media_linear("per_layer_projection", media_width, args.hidden_size))
                .transpose()?,
            per_layer_norm: (media_width > 0)
                .then(|| norm("post_per_layer_input_norm", args.hidden_size))
                .transpose()?,
            layer_scalar: Parameter::unloaded(
                ParameterSpec::trainable(format!("{prefix}.layer_scalar"))
                    .map_err(Error::backend)?,
                &[1],
                context,
            )?,
        })
    }

    /// Executes exact dense residual and optional per-layer media equations.
    pub fn forward<C: AttentionCache<B::Tensor>>(
        &mut self,
        input: BlockInput<'_, B::Tensor, C>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let pass = if input.hidden.dim(1) > 1 {
            ExpertPass::Prefill
        } else {
            ExpertPass::Decode
        };
        let mut provider = ResidentExpertProvider;
        self.forward_with_provider(input, pass, &mut provider, context)
    }

    /// Executes the same block with component observations and interventions.
    pub fn forward_observed<C: AttentionCache<B::Tensor>>(
        &mut self,
        input: BlockInput<'_, B::Tensor, C>,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error> {
        let pass = if input.hidden.dim(1) > 1 {
            ExpertPass::Prefill
        } else {
            ExpertPass::Decode
        };
        self.forward_with_provider_observed(
            input,
            pass,
            &mut ResidentExpertProvider,
            context,
            instrumentation,
        )
    }

    /// Executes through a runtime-owned resident, cached, or distributed provider.
    pub fn forward_with_provider<C, P>(
        &mut self,
        input: BlockInput<'_, B::Tensor, C>,
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
            input,
            pass,
            provider,
            context,
            &mut ComponentInstrumentation::disabled(),
        )
    }

    /// Observes the exact request consumed by the retained expert provider.
    pub fn forward_with_provider_observed<C, P>(
        &mut self,
        input: BlockInput<'_, B::Tensor, C>,
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
            input,
            pass,
            None,
            context,
            instrumentation,
            |experts, request| {
                provider
                    .forward_grouped(experts, request, context)
                    .map_err(Error::backend_source)
            },
        )
    }

    /// Executes the same equations with rank-local projections and routed experts.
    pub fn forward_parallel_with_provider<C, P>(
        &mut self,
        input: BlockInput<'_, B::Tensor, C>,
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
            input,
            pass,
            provider,
            parallel,
            context,
            &mut ComponentInstrumentation::disabled(),
        )
    }

    /// Observes local units and the already reduced complete expert write.
    pub fn forward_parallel_with_provider_observed<C, P>(
        &mut self,
        input: BlockInput<'_, B::Tensor, C>,
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
            input,
            pass,
            Some(parallel),
            context,
            instrumentation,
            |experts, request| {
                let value = provider
                    .forward_grouped_tensor_parallel(
                        experts,
                        request,
                        B::parallel_size(parallel),
                        context,
                    )
                    .map_err(Error::backend_source)?;
                eredu_runtime::reduce_routed_expert_tensor_parallel::<B>(value, parallel, context)
            },
        )
    }

    /// Executes the collective block with resident experts and shared hooks.
    pub fn forward_parallel<C: AttentionCache<B::Tensor>>(
        &mut self,
        input: BlockInput<'_, B::Tensor, C>,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        B: eredu_nn::TensorParallelGroupedNeuralBackend,
    {
        self.forward_parallel_observed(
            input,
            parallel,
            context,
            &mut ComponentInstrumentation::disabled(),
        )
    }

    /// Executes the same collective block with component instrumentation.
    pub fn forward_parallel_observed<C: AttentionCache<B::Tensor>>(
        &mut self,
        input: BlockInput<'_, B::Tensor, C>,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error>
    where
        B: eredu_nn::TensorParallelGroupedNeuralBackend,
    {
        let pass = if input.hidden.dim(1) > 1 {
            ExpertPass::Prefill
        } else {
            ExpertPass::Decode
        };
        self.forward_parallel_with_provider_observed(
            input,
            pass,
            &mut ResidentExpertProvider,
            parallel,
            context,
            instrumentation,
        )
    }

    fn forward_inner<C, F>(
        &mut self,
        input: BlockInput<'_, B::Tensor, C>,
        pass: ExpertPass,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
        execute_experts: F,
    ) -> Result<B::Tensor, Error>
    where
        C: AttentionCache<B::Tensor>,
        F: for<'data, 'unit> FnOnce(
            &mut B::GatedProductGroups,
            RoutedExpertRequest<'data, 'unit, B::Tensor>,
        ) -> Result<B::Tensor, Error>,
    {
        let normalized = self.input_norm.forward(input.hidden, context)?;
        let normalized = instrumentation.apply("attention.input", normalized)?;
        let attention = self.attention.forward_instrumented(
            AttentionInput {
                hidden: &normalized,
                mask: input.mask,
                cache: input.cache,
                shared: input.shared,
                rotary_position: input.rotary_position,
            },
            parallel,
            context,
            instrumentation,
        )?;
        let attention = instrumentation.apply("attention.write", attention)?;
        let attention = self.post_attention_norm.forward(&attention, context)?;
        let attention = instrumentation.apply("attention.output", attention)?;
        let hidden = input.hidden.add(&attention, context)?;
        let hidden = instrumentation.apply("attention.residual", hidden)?;
        let normalized = self.pre_feed_forward_norm.forward(&hidden, context)?;
        let normalized = instrumentation.apply("dense_feed_forward.input", normalized)?;
        let dense =
            self.mlp
                .forward_instrumented(&normalized, parallel, context, instrumentation)?;
        let sparse = self.router.is_some() && self.experts.is_some();
        let dense = instrumentation.apply(
            if sparse {
                "dense_feed_forward.write"
            } else {
                "feed_forward.write"
            },
            dense,
        )?;
        let mlp =
            if let (Some(router), Some(experts)) = (self.router.as_mut(), self.experts.as_mut()) {
                let dense = self
                    .post_feed_forward_norm_1
                    .as_mut()
                    .ok_or_else(|| Error::backend("sparse Gemma block has no dense branch norm"))?
                    .forward(&dense, context)?;
                let dense = instrumentation.apply("dense_feed_forward.output", dense)?;
                let shape = hidden.shape().to_vec();
                let flat = hidden.reshape(&[-1, hidden.dim(2)], context)?;
                let routed_input = self
                    .pre_feed_forward_norm_2
                    .as_mut()
                    .ok_or_else(|| Error::backend("sparse Gemma block has no routed input norm"))?
                    .forward(&flat, context)?;
                let routed_input = if instrumentation.enabled() {
                    instrumentation
                        .apply(
                            "routed_feed_forward.input",
                            routed_input.reshape(&shape, context)?,
                        )?
                        .reshape(&[-1, hidden.dim(2)], context)?
                } else {
                    routed_input
                };
                // Routing reads the raw residual independently of the expert input norm.
                let routing_input = if instrumentation.enabled() {
                    Some(instrumentation.apply("routing.input", hidden.clone())?)
                } else {
                    None
                };
                let routing_flat = routing_input
                    .as_ref()
                    .map(|value| value.reshape(&[-1, hidden.dim(2)], context))
                    .transpose()?;
                let routes = router.select(routing_flat.as_ref().unwrap_or(&flat), context)?;
                let routed = instrumentation
                    .routed(
                        "routing",
                        RoutedExpertRequest {
                            unit_observer: None,
                            bank: eredu_runtime::RoutedBankId::new(0),
                            layer: self.layer,
                            input: &routed_input,
                            routes: &routes,
                            pass,
                        },
                        |request| execute_experts(experts, request),
                    )?
                    .reshape(&shape, context)?;
                let routed = instrumentation.apply("routed_feed_forward.write", routed)?;
                let routed = self
                    .post_feed_forward_norm_2
                    .as_mut()
                    .ok_or_else(|| Error::backend("sparse Gemma block has no routed output norm"))?
                    .forward(&routed, context)?;
                let routed = instrumentation.apply("routed_feed_forward.output", routed)?;
                dense.add(&routed, context)?
            } else {
                dense
            };
        let mlp = if sparse {
            instrumentation.apply("feed_forward.write", mlp)?
        } else {
            mlp
        };
        let mlp = self.post_feed_forward_norm.forward(&mlp, context)?;
        let mlp = instrumentation.apply("feed_forward.output", mlp)?;
        let hidden = hidden.add(&mlp, context)?;
        let mut hidden = instrumentation.apply("feed_forward.residual", hidden)?;
        if let (Some(media), Some(gate), Some(projection), Some(norm)) = (
            input.per_layer_input,
            self.per_layer_gate.as_mut(),
            self.per_layer_projection.as_mut(),
            self.per_layer_norm.as_mut(),
        ) {
            let gate_input = if instrumentation.enabled() {
                Some(instrumentation.apply("per_layer.input", hidden.clone())?)
            } else {
                None
            };
            let gate = instrumentation.project::<B>(
                "per_layer.gate_input",
                gate,
                gate_input.as_ref().unwrap_or(&hidden),
                None,
                context,
            )?;
            let gate = B::Tensor::gelu(&gate, context)?;
            instrumentation.observe("per_layer.prepared", media)?;
            let media = gate.multiply(media, context)?;
            let media = instrumentation.apply("per_layer.units", media)?;
            let media = instrumentation.project::<B>(
                "per_layer.write_input",
                projection,
                &media,
                parallel,
                context,
            )?;
            let media = instrumentation.apply("per_layer.write", media)?;
            let media = norm.forward(&media, context)?;
            let media = instrumentation.apply("per_layer.output", media)?;
            hidden = hidden.add(&media, context)?;
        }
        let hidden = instrumentation.apply("residual.before_scale", hidden)?;
        let hidden = hidden.multiply(self.layer_scalar.as_ref(), context)?;
        instrumentation.apply("residual.scaled", hidden)
    }
}

/// Returns the architecture-owned routed expert specification for one sparse layer.
pub fn expert_bank_spec(args: &ModelArgs, layer: usize) -> Result<GroupedGatedProductSpec, Error> {
    expert_bank_spec_at(
        args,
        &format!("model.language_model.layers.{layer}.experts.switch_glu"),
    )
}

/// Returns the same architecture-owned bank at placement-resolved geometry.
pub(crate) fn localized_expert_bank_spec(
    args: &ModelArgs,
    layer: usize,
    expert_count: i32,
    intermediate_dimensions: i32,
) -> Result<GroupedGatedProductSpec, Error> {
    expert_bank_spec(args, layer)?.with_group_geometry(expert_count, intermediate_dimensions)
}

fn expert_bank_spec_at(
    args: &ModelArgs,
    experts_prefix: &str,
) -> Result<GroupedGatedProductSpec, Error> {
    let expert_count = args
        .num_experts
        .ok_or_else(|| Error::backend("Gemma 4 sparse layer has no expert count"))?;
    let expert_width = args
        .moe_intermediate_size
        .ok_or_else(|| Error::backend("Gemma 4 sparse layer has no expert width"))?;
    let gate_up_name = format!("{experts_prefix}.gate_up_proj");
    let down_name = format!("{experts_prefix}.down_proj");
    GroupedGatedProductSpec::new(
        expert_count,
        args.hidden_size,
        expert_width,
        args.hidden_size,
        eredu_nn::GatedProductPolicy::ordinary_gelu_approximate(),
        GatedProductGroupLayout::Packed {
            gate_up: standard_expert_projection(
                &gate_up_name,
                None,
                args.linear_format_for(&gate_up_name),
            )?,
            down: standard_expert_projection(&down_name, None, args.linear_format_for(&down_name))?,
        },
    )
}
