//! Backend-neutral Gemma 4 decoder equations.

use std::collections::HashMap;

use eredu_core::AttentionPolicy;
use eredu_nn::{
    AttentionCache, AttentionRequest, AttentionStateSource, AttentionValueSource, Error,
    GroupSelectionOperator, GroupedGatedProductSpec, GroupedNeuralBackend, LinearOperator,
    NeuralBackend, NormalizationOperator, Parameter, Parameterized, RotaryOperator, RotaryPosition,
    RotarySubspace, Tensor,
};
use eredu_runtime::{
    ExpertPass, ParameterProvider, ResidentExpertProvider, RoutedExpertRequest,
    TensorParallelParameterProvider,
};

use crate::decoder::ComponentInstrumentation;
/// Shared static vocabulary and per-layer input resource declarations.
pub mod pinned;
/// Shared sparse-branch declarations for construction and cold consumers.
pub mod routed;
/// Ordinary projection declarations shared by construction and cold consumers.
pub mod specs;

use super::ModelArgs;

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
    #[parameter(skip)]
    arithmetic: eredu_nn::AttentionArithmetic,
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
    /// Builds one unloaded attention unit from its immutable family declaration.
    pub fn new(
        config: &super::FamilyConfig,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        Self::from_spec(
            config.attention(layer)?.clone(),
            config.text.attention_arithmetic,
            context,
        )
    }

    fn from_spec(
        spec: specs::AttentionSpec,
        arithmetic: eredu_nn::AttentionArithmetic,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        Ok(Self {
            query_heads: spec.geometry.query_heads(),
            key_value_heads: spec.geometry.key_value_heads(),
            rotary_dimensions: spec.rotary.dimensions,
            policy: spec.policy.attention,
            state_source: spec.policy.key_value,
            arithmetic,
            query: B::linear(spec.query, context)?,
            key: spec.key.map(|s| B::linear(s, context)).transpose()?,
            value: spec.value.map(|s| B::linear(s, context)).transpose()?,
            output: B::linear(spec.output, context)?,
            query_norm: B::normalization(spec.query_norm, context)?,
            key_norm: spec
                .key_norm
                .map(|s| B::normalization(s, context))
                .transpose()?,
            rotary: B::rotary(spec.rotary, context)?,
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
                    arithmetic: self.arithmetic,
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
    // Proportional RoPE retains the full pair layout and zeroes frequencies
    // outside its rotated fraction; shrinking the head changes both pairing
    // and the frequency denominator.
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
    #[parameter(skip)]
    policy: eredu_nn::GatedProductPolicy,
    /// GELU gate projection.
    pub gate: B::Linear,
    /// Multiplicative up projection.
    pub up: B::Linear,
    /// Output projection.
    pub down: B::Linear,
}

impl<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> DenseMlp<B> {
    fn from_spec(
        spec: specs::DenseSpec,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        Ok(Self {
            policy: spec.product.policy,
            gate: B::linear(spec.product.gate, context)?,
            up: B::linear(spec.product.up, context)?,
            down: B::linear(spec.product.down, context)?,
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
        let gate = if self.policy.activation() == eredu_nn::GatedProductActivation::GeluApproximate
        {
            B::gelu_approximate(gate, context)?
        } else {
            B::Tensor::gelu(&gate, context)?
        };
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
    /// Builds one unloaded block from the family-owned gain and layer-input scope.
    pub fn new(
        config: &super::FamilyConfig,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        Self::from_scopes(
            &config.text,
            layer,
            config.block(layer)?,
            config.dense(layer)?,
            config.attention(layer)?,
            config.routed(layer)?,
            None,
            context,
        )
    }

    pub(crate) fn from_scopes(
        args: &ModelArgs,
        layer: usize,
        block: &specs::BlockSpec,
        dense: &specs::DenseSpec,
        attention: &specs::AttentionSpec,
        routed: Option<&super::text::routed::Declaration>,
        routed_spec: Option<GroupedGatedProductSpec>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let block = block
            .clone()
            .with_input_width(args.hidden_size_per_layer_input)?;
        let width = args
            .layer_policy(layer)
            .ok_or_else(|| Error::backend("dense layer outside local schedule"))?
            .intermediate_size
            .get() as i32;
        let dense = dense.clone().with_intermediate(width)?;
        let attention = attention.for_invocation(args, layer)?;
        let routed = match routed {
            Some(source) => Some(source.for_invocation(args, layer, routed_spec)?),
            None if routed_spec.is_none() => None,
            None => return Err(Error::backend("selected bank has no routed source owner")),
        };
        let spec =
            specs::DecoderBlockSpec::from_scopes(args, layer, block, dense, attention, routed)?;
        Self::from_spec(layer, spec, context)
    }

    pub(crate) fn from_spec(
        layer: usize,
        spec: specs::DecoderBlockSpec,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let specs::DecoderBlockSpec {
            attention,
            attention_arithmetic,
            dense,
            block,
            routed,
        } = spec;
        let (per_layer_gate, per_layer_projection, per_layer_norm) =
            if let Some(spec) = block.per_layer {
                (
                    Some(B::linear(spec.gate, context)?),
                    Some(B::linear(spec.projection, context)?),
                    Some(B::normalization(spec.norm, context)?),
                )
            } else {
                (None, None, None)
            };
        let (router, experts, sparse_norms) = if let Some(declaration) = routed {
            (
                Some(B::top_k_group_selector(
                    declaration.routed.selector,
                    context,
                )?),
                Some(B::grouped_gated_product(
                    declaration.routed.experts,
                    context,
                )?),
                declaration
                    .norms
                    .map(|spec| B::normalization(spec, context))
                    .into_iter()
                    .collect::<Result<Vec<_>, _>>()?,
            )
        } else {
            (None, None, Vec::new())
        };
        let mut sparse_norms = sparse_norms.into_iter();
        Ok(Self {
            layer,
            attention: Attention::from_spec(attention, attention_arithmetic, context)?,
            mlp: DenseMlp::from_spec(dense, context)?,
            router,
            experts,
            input_norm: B::normalization(block.input, context)?,
            post_attention_norm: B::normalization(block.post_attention, context)?,
            pre_feed_forward_norm: B::normalization(block.pre_feed_forward, context)?,
            post_feed_forward_norm: B::normalization(block.post_feed_forward, context)?,
            post_feed_forward_norm_1: sparse_norms.next(),
            pre_feed_forward_norm_2: sparse_norms.next(),
            post_feed_forward_norm_2: sparse_norms.next(),
            per_layer_gate,
            per_layer_projection,
            per_layer_norm,
            layer_scalar: block.residual_scale.unloaded(context)?,
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
        P: ParameterProvider<B>,
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
        P: ParameterProvider<B>,
        P::Error: std::fmt::Display,
    {
        self.forward_inner(
            input,
            pass,
            None,
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
        P: TensorParallelParameterProvider<B>,
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
        P: TensorParallelParameterProvider<B>,
        P::Error: std::fmt::Display,
    {
        self.forward_inner(
            input,
            pass,
            Some(parallel),
            None,
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

    /// Executes shared block parameters with an invocation-specific residual
    /// scale. Encoder/decoder sharing does not duplicate the learned matrices.
    pub fn forward_with_layer_scalar<C, P>(
        &mut self,
        input: BlockInput<'_, B::Tensor, C>,
        scalar: &B::Tensor,
        pass: ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error>
    where
        C: AttentionCache<B::Tensor>,
        P: ParameterProvider<B>,
        P::Error: std::fmt::Display,
    {
        self.forward_inner(
            input,
            pass,
            None,
            Some(scalar),
            context,
            instrumentation,
            |experts, request| {
                provider
                    .forward_grouped(experts, request, context)
                    .map_err(Error::backend_source)
            },
        )
    }

    /// Executes the same invocation-specific scale with rank-local projections
    /// and the retained distributed expert provider.
    pub fn forward_parallel_with_layer_scalar<C, P>(
        &mut self,
        input: BlockInput<'_, B::Tensor, C>,
        scalar: &B::Tensor,
        pass: ExpertPass,
        provider: &mut P,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error>
    where
        C: AttentionCache<B::Tensor>,
        P: TensorParallelParameterProvider<B>,
        P::Error: std::fmt::Display,
    {
        self.forward_inner(
            input,
            pass,
            Some(parallel),
            Some(scalar),
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

    fn forward_inner<C, F>(
        &mut self,
        input: BlockInput<'_, B::Tensor, C>,
        pass: ExpertPass,
        parallel: Option<&B::ParallelContext>,
        scalar: Option<&B::Tensor>,
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
        let format = hidden.element_type();
        let hidden = hidden.multiply(scalar.unwrap_or(self.layer_scalar.as_ref()), context)?;
        let hidden = match format {
            Some(format) => hidden.cast_float(format, context)?,
            None => hidden,
        };
        instrumentation.apply("residual.scaled", hidden)
    }
}
