//! Backend-neutral Gemma 4 decoder equations.

use std::collections::HashMap;

use eredu_core::AttentionPolicy;
use eredu_nn::{
    AttentionCache, AttentionStateSource, AttentionValueSource, Error, GatedProductExpertBankSpec,
    GatedProductExpertLayout, LinearOperator, LinearSpec, NeuralBackend, NormalizationOperator,
    NormalizationSpec, Parameter, ParameterSpec, Parameterized, RotaryOperator, RotaryPosition,
    RotarySpec, RotarySubspace, RoutedNeuralBackend, RouterInputTransformSpec, RoutingOperator,
    RoutingScoring, Tensor, TopKRouterSpec, TopKRoutingSpec,
};
use eredu_runtime::{
    ExpertPass, ResidentExpertProvider, RoutedExpertProvider, RoutedExpertRequest,
};

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
    /// Optional mutable layer-local cache.
    pub cache: Option<&'a mut C>,
    /// Shared publications from earlier compatible layers.
    pub shared: &'a mut SharedAttentionStates<T>,
    /// Optional caller-provided explicit rotary positions.
    pub rotary_position: Option<RotaryPosition<'a, T>>,
}

/// Gemma 4 grouped-query attention with local, publishing, or shared KV state.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct Attention<B: NeuralBackend> {
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

impl<B: NeuralBackend> Attention<B> {
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
                    format: args.linear_format_for(&weight_name),
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
            query_norm: B::rms_norm(
                NormalizationSpec {
                    dimensions: head_dim,
                    epsilon: args.rms_norm_eps,
                    weight: ParameterSpec::trainable(format!("{prefix}.q_norm.weight"))
                        .map_err(Error::backend)?,
                },
                context,
            )?,
            key_norm: owns_state
                .then(|| {
                    B::rms_norm(
                        NormalizationSpec {
                            dimensions: head_dim,
                            epsilon: args.rms_norm_eps,
                            weight: ParameterSpec::trainable(format!("{prefix}.k_norm.weight"))
                                .map_err(Error::backend)?,
                        },
                        context,
                    )
                })
                .transpose()?,
            rotary: B::rotary(
                RotarySpec {
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
        let offset = input.cache.as_ref().map_or(0, |cache| cache.offset());
        let reshape = |value: B::Tensor, heads: i32| {
            value
                .reshape(&[batch, sequence, heads, -1], context)?
                .transpose_axes(&[0, 2, 1, 3], context)
        };
        let queries = reshape(self.query.forward(input.hidden, context)?, self.query_heads)?;
        let queries = self.query_norm.forward(&queries, context)?;
        let position = input
            .rotary_position
            .unwrap_or(RotaryPosition::Offset(offset));
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
            let (keys, values) = match input.cache {
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
        let attended = B::attention(queries, keys, values, 1.0, input.mask, context)?;
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
        let attended = self.attend(input, context)?;
        self.output.forward(&attended, context)
    }

    /// Executes rank-local attention followed by the collective output projection.
    pub fn forward_parallel<C: AttentionCache<B::Tensor>>(
        &mut self,
        input: AttentionInput<'_, B::Tensor, C>,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        B: RoutedNeuralBackend,
    {
        let attended = self.attend(input, context)?;
        B::row_parallel_linear(&mut self.output, &attended, parallel, context)
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
pub struct DenseMlp<B: NeuralBackend> {
    /// GELU gate projection.
    pub gate: B::Linear,
    /// Multiplicative up projection.
    pub up: B::Linear,
    /// Output projection.
    pub down: B::Linear,
}

impl<B: NeuralBackend> DenseMlp<B> {
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
                    format: args.linear_format_for(&weight_name),
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

    fn forward(
        &mut self,
        input: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let gate = self.gate.forward(input, context)?;
        let gate = B::Tensor::gelu(&gate, context)?;
        let up = self.up.forward(input, context)?;
        let hidden = gate.multiply(&up, context)?;
        self.down.forward(&hidden, context)
    }

    fn forward_parallel(
        &mut self,
        input: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        B: RoutedNeuralBackend,
    {
        let gate = self.gate.forward(input, context)?;
        let gate = B::Tensor::gelu(&gate, context)?;
        let up = self.up.forward(input, context)?;
        let hidden = gate.multiply(&up, context)?;
        B::row_parallel_linear(&mut self.down, &hidden, parallel, context)
    }
}

/// Dense Gemma 4 transformer block. Sparse routing is constructed by the
/// routed block wrapper so the dense residual is shared exactly.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct DenseBlock<B: RoutedNeuralBackend> {
    #[parameter(skip)]
    layer: usize,
    /// Stateful self attention.
    pub attention: Attention<B>,
    /// Dense gated-GELU branch.
    pub mlp: DenseMlp<B>,
    /// Optional selected-softmax sparse router.
    pub router: Option<B::Router>,
    /// Optional packed GELU-gated expert bank.
    pub experts: Option<B::GatedProductExpertBank>,
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

impl<B: RoutedNeuralBackend> DenseBlock<B> {
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
        let policy = args
            .layer_policy(layer)
            .ok_or_else(|| Error::backend(format!("missing Gemma 4 layer policy {layer}")))?;
        let prefix = format!("{layer_root}.{layer}");
        let norm = |field: &str, dimensions| {
            B::rms_norm(
                NormalizationSpec {
                    dimensions,
                    epsilon: args.rms_norm_eps,
                    weight: ParameterSpec::trainable(format!("{prefix}.{field}.weight"))
                        .map_err(Error::backend)?,
                },
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
                    format: args.linear_format_for(&weight_name),
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
            let router = B::top_k_router(
                TopKRouterSpec {
                    input_dimensions: args.hidden_size,
                    weight: ParameterSpec::trainable(&router_weight).map_err(Error::backend)?,
                    bias: None,
                    correction_bias: None,
                    input_transform: Some(RouterInputTransformSpec {
                        epsilon: args.rms_norm_eps,
                        scale: ParameterSpec::trainable(format!("{router_prefix}.scale"))
                            .map_err(Error::backend)?,
                        inverse_sqrt_dimensions: true,
                    }),
                    route_scale: Some(
                        ParameterSpec::trainable(format!("{router_prefix}.per_expert_scale"))
                            .map_err(Error::backend)?,
                    ),
                    quantization: args.linear_format_for(&router_weight).weight_quantization(),
                    routing: TopKRoutingSpec::new(
                        expert_count,
                        top_k,
                        RoutingScoring::SelectedSoftmax,
                        false,
                    )?,
                },
                context,
            )?;
            let experts = B::gated_product_expert_bank(
                expert_bank_spec_at(args, &format!("{prefix}.experts.switch_glu"))?,
                context,
            )?;
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

    /// Executes through a runtime-owned resident, cached, or distributed expert provider.
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
        let normalized = self.input_norm.forward(input.hidden, context)?;
        let attention = self.attention.forward(
            AttentionInput {
                hidden: &normalized,
                mask: input.mask,
                cache: input.cache,
                shared: input.shared,
                rotary_position: input.rotary_position,
            },
            context,
        )?;
        let attention = self.post_attention_norm.forward(&attention, context)?;
        let hidden = input.hidden.add(&attention, context)?;
        let normalized = self.pre_feed_forward_norm.forward(&hidden, context)?;
        let dense = self.mlp.forward(&normalized, context)?;
        let mlp =
            if let (Some(router), Some(experts)) = (self.router.as_mut(), self.experts.as_mut()) {
                let dense = self
                    .post_feed_forward_norm_1
                    .as_mut()
                    .ok_or_else(|| Error::backend("sparse Gemma block has no dense branch norm"))?
                    .forward(&dense, context)?;
                let shape = hidden.shape().to_vec();
                let flat = hidden.reshape(&[-1, hidden.dim(2)], context)?;
                let routed_input = self
                    .pre_feed_forward_norm_2
                    .as_mut()
                    .ok_or_else(|| Error::backend("sparse Gemma block has no routed input norm"))?
                    .forward(&flat, context)?;
                let routes = router.route(&flat, context)?;
                let routed = provider
                    .forward_routed(
                        experts,
                        RoutedExpertRequest {
                            layer: self.layer,
                            input: &routed_input,
                            routes: &routes,
                            pass,
                        },
                        context,
                    )
                    .map_err(Error::backend)?
                    .reshape(&shape, context)?;
                let routed = self
                    .post_feed_forward_norm_2
                    .as_mut()
                    .ok_or_else(|| Error::backend("sparse Gemma block has no routed output norm"))?
                    .forward(&routed, context)?;
                dense.add(&routed, context)?
            } else {
                dense
            };
        let mlp = self.post_feed_forward_norm.forward(&mlp, context)?;
        let mut hidden = hidden.add(&mlp, context)?;
        if let (Some(media), Some(gate), Some(projection), Some(norm)) = (
            input.per_layer_input,
            self.per_layer_gate.as_mut(),
            self.per_layer_projection.as_mut(),
            self.per_layer_norm.as_mut(),
        ) {
            let gate = gate.forward(&hidden, context)?;
            let gate = B::Tensor::gelu(&gate, context)?;
            let media = gate.multiply(media, context)?;
            let media = projection.forward(&media, context)?;
            let media = norm.forward(&media, context)?;
            hidden = hidden.add(&media, context)?;
        }
        hidden.multiply(self.layer_scalar.as_ref(), context)
    }

    /// Executes the ordinary block equations with rank-local projections and
    /// runtime-owned routed experts.
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
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        let normalized = self.input_norm.forward(input.hidden, context)?;
        let attention = self.attention.forward_parallel(
            AttentionInput {
                hidden: &normalized,
                mask: input.mask,
                cache: input.cache,
                shared: input.shared,
                rotary_position: input.rotary_position,
            },
            parallel,
            context,
        )?;
        let attention = self.post_attention_norm.forward(&attention, context)?;
        let hidden = input.hidden.add(&attention, context)?;
        let normalized = self.pre_feed_forward_norm.forward(&hidden, context)?;
        let dense = self.mlp.forward_parallel(&normalized, parallel, context)?;
        let mlp =
            if let (Some(router), Some(experts)) = (self.router.as_mut(), self.experts.as_mut()) {
                let dense = self
                    .post_feed_forward_norm_1
                    .as_mut()
                    .ok_or_else(|| Error::backend("sparse Gemma block has no dense branch norm"))?
                    .forward(&dense, context)?;
                let shape = hidden.shape().to_vec();
                let flat = hidden.reshape(&[-1, hidden.dim(2)], context)?;
                let routed_input = self
                    .pre_feed_forward_norm_2
                    .as_mut()
                    .ok_or_else(|| Error::backend("sparse Gemma block has no routed input norm"))?
                    .forward(&flat, context)?;
                let routes = router.route(&flat, context)?;
                let routed = provider
                    .forward_routed_tensor_parallel(
                        experts,
                        RoutedExpertRequest {
                            layer: self.layer,
                            input: &routed_input,
                            routes: &routes,
                            pass,
                        },
                        B::parallel_size(parallel),
                        context,
                    )
                    .map_err(Error::backend)?;
                let routed = eredu_runtime::reduce_routed_expert_tensor_parallel::<B>(
                    routed, parallel, context,
                )?
                .reshape(&shape, context)?;
                let routed = self
                    .post_feed_forward_norm_2
                    .as_mut()
                    .ok_or_else(|| Error::backend("sparse Gemma block has no routed output norm"))?
                    .forward(&routed, context)?;
                dense.add(&routed, context)?
            } else {
                dense
            };
        let mlp = self.post_feed_forward_norm.forward(&mlp, context)?;
        let mut hidden = hidden.add(&mlp, context)?;
        if let (Some(media), Some(gate), Some(projection), Some(norm)) = (
            input.per_layer_input,
            self.per_layer_gate.as_mut(),
            self.per_layer_projection.as_mut(),
            self.per_layer_norm.as_mut(),
        ) {
            let gate = gate.forward(&hidden, context)?;
            let gate = B::Tensor::gelu(&gate, context)?;
            let media = gate.multiply(media, context)?;
            let media = B::row_parallel_linear(projection, &media, parallel, context)?;
            let media = norm.forward(&media, context)?;
            hidden = hidden.add(&media, context)?;
        }
        hidden.multiply(self.layer_scalar.as_ref(), context)
    }

    /// Executes the collective block with resident experts.
    pub fn forward_parallel<C: AttentionCache<B::Tensor>>(
        &mut self,
        input: BlockInput<'_, B::Tensor, C>,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let pass = if input.hidden.dim(1) > 1 {
            ExpertPass::Prefill
        } else {
            ExpertPass::Decode
        };
        let mut provider = ResidentExpertProvider;
        self.forward_parallel_with_provider(input, pass, &mut provider, parallel, context)
    }
}

/// Returns the architecture-owned routed expert specification for one sparse layer.
pub fn expert_bank_spec(
    args: &ModelArgs,
    layer: usize,
) -> Result<GatedProductExpertBankSpec, Error> {
    expert_bank_spec_at(
        args,
        &format!("model.language_model.layers.{layer}.experts.switch_glu"),
    )
}

fn expert_bank_spec_at(
    args: &ModelArgs,
    experts_prefix: &str,
) -> Result<GatedProductExpertBankSpec, Error> {
    let expert_count = args
        .num_experts
        .ok_or_else(|| Error::backend("Gemma 4 sparse layer has no expert count"))?;
    let expert_width = args
        .moe_intermediate_size
        .ok_or_else(|| Error::backend("Gemma 4 sparse layer has no expert width"))?;
    let gate_up_name = format!("{experts_prefix}.gate_up_proj");
    let down_name = format!("{experts_prefix}.down_proj");
    Ok(GatedProductExpertBankSpec {
        expert_count,
        input_dimensions: args.hidden_size,
        intermediate_dimensions: expert_width,
        output_dimensions: args.hidden_size,
        policy: eredu_nn::GatedProductPolicy::ordinary_gelu_approximate(),
        layout: GatedProductExpertLayout::Packed {
            gate_up: standard_expert_projection(
                &gate_up_name,
                None,
                args.linear_format_for(&gate_up_name),
            )?,
            down: standard_expert_projection(&down_name, None, args.linear_format_for(&down_name))?,
        },
    })
}
