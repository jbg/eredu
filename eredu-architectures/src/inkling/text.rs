//! Backend-neutral Inkling decoder equations and bounded operator state.

use eredu_core::AttentionPolicy;
use eredu_nn::{
    AttentionCache, AttentionRequest, AuxiliaryConvolutionState, CausalDepthwiseConvolution,
    CausalDepthwiseConvolutionSpec, ConvolutionActivation, EmbeddingOperator, EmbeddingSpec, Error,
    GatedProductGroupLayout, GroupSelection, GroupedGatedProductOperator, GroupedGatedProductSpec,
    GroupedNeuralBackend, JointGroupSelectionInput, JointGroupSelectionSpec, LinearOperator,
    LinearSpec, NeuralBackend, NormalizationConstructionSpec, NormalizationOperator, Parameter,
    ParameterSpec, Parameterized, RelativeAttentionInput, Tensor,
};
use eredu_runtime::{ExpertPass, RoutedExpertProvider, RoutedExpertRequest};

use crate::linear_format::standard_expert_projection;

use super::{FeedForwardPolicy, LayerPolicy, ModelArgs, TextArgs};

/// Four bounded causal histories owned by every Inkling decoder layer.
#[derive(Debug, Clone)]
pub struct ConvolutionState<T> {
    /// Key-projection history.
    pub key: Option<T>,
    /// Value-projection history.
    pub value: Option<T>,
    /// Attention-output history.
    pub attention: Option<T>,
    /// Feed-forward-output history.
    pub feed_forward: Option<T>,
}

impl<T> Default for ConvolutionState<T> {
    fn default() -> Self {
        Self {
            key: None,
            value: None,
            attention: None,
            feed_forward: None,
        }
    }
}

/// Complete mutable state for one decoder layer.
#[derive(Debug, Clone)]
pub struct LayerState<T, C> {
    /// Layer-local key/value cache.
    pub attention: C,
    /// Exact four short-convolution histories.
    pub convolutions: ConvolutionState<T>,
}

impl<T, C> AttentionCache<T> for LayerState<T, C>
where
    T: Tensor,
    C: AttentionCache<T>,
{
    fn offset(&self) -> i32 {
        self.attention.offset()
    }

    fn max_size(&self) -> Option<i32> {
        self.attention.max_size()
    }

    fn update_for_attention(
        &mut self,
        keys: T,
        values: T,
        context: &T::Context,
    ) -> Result<(T, T), Error> {
        self.attention.update_for_attention(keys, values, context)
    }

    fn attention(
        &mut self,
        request: AttentionRequest<'_, T>,
        context: &T::Context,
    ) -> Result<T, Error> {
        self.attention.attention(request, context)
    }
}

impl<T, C> AuxiliaryConvolutionState<T> for LayerState<T, C>
where
    T: Tensor,
    C: AttentionCache<T>,
{
    fn convolution_state(&mut self, slot: u32) -> Result<&mut Option<T>, Error> {
        match slot {
            0 => Ok(&mut self.convolutions.key),
            1 => Ok(&mut self.convolutions.value),
            2 => Ok(&mut self.convolutions.attention),
            3 => Ok(&mut self.convolutions.feed_forward),
            _ => Err(Error::backend(format!(
                "Inkling convolution state slot {slot} is out of range"
            ))),
        }
    }
}

/// Inkling learned-relative grouped-query attention.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct Attention<B: NeuralBackend> {
    #[parameter(skip)]
    query_heads: i32,
    #[parameter(skip)]
    key_value_heads: i32,
    #[parameter(skip)]
    head_dimensions: i32,
    #[parameter(skip)]
    relative_dimensions: i32,
    #[parameter(skip)]
    relative_extent: i32,
    #[parameter(skip)]
    policy: AttentionPolicy,
    #[parameter(skip)]
    log_scaling_floor: Option<i32>,
    #[parameter(skip)]
    log_scaling_alpha: f32,
    /// Query projection.
    pub query: B::Linear,
    /// Key projection.
    pub key: B::Linear,
    /// Value projection.
    pub value: B::Linear,
    /// Per-query relative-feature projection.
    pub relative: B::Linear,
    /// Output projection.
    pub output: B::Linear,
    /// Per-head query normalization.
    pub query_norm: B::Normalization,
    /// Per-head key normalization.
    pub key_norm: B::Normalization,
    /// Relative-feature-to-distance table `[d_rel, extent]`.
    pub relative_projection: Parameter<B::Tensor>,
    /// Residual causal short convolution over projected keys.
    pub key_convolution: CausalDepthwiseConvolution<B>,
    /// Residual causal short convolution over projected values.
    pub value_convolution: CausalDepthwiseConvolution<B>,
}

impl<B: NeuralBackend> Attention<B> {
    /// Builds one attention layer under the released parameter root.
    pub fn new(
        args: &TextArgs,
        layer: usize,
        policy: AttentionPolicy,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        Self::new_at(args, policy, &format!("model.layers.{layer}"), context)
    }

    /// Builds attention under an explicit architecture-owned block root.
    pub fn new_at(
        args: &TextArgs,
        policy: AttentionPolicy,
        block_root: &str,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let local = policy.window().is_some();
        let query_heads = args.query_heads(local);
        let key_value_heads = args.key_value_heads(local);
        let head_dimensions = args.attention_head_dim(local);
        let relative_extent = policy
            .window()
            .map(|window| window.get() as i32)
            .unwrap_or(args.rel_extent);
        let prefix = format!("{block_root}.self_attn");
        let linear = |field: &str, input: i32, output: i32, bias: bool| {
            let weight = format!("{prefix}.{field}.weight");
            B::linear(
                LinearSpec {
                    input,
                    output,
                    weight: ParameterSpec::trainable(&weight).map_err(Error::backend)?,
                    bias: bias
                        .then(|| ParameterSpec::trainable(format!("{prefix}.{field}.bias")))
                        .transpose()
                        .map_err(Error::backend)?,
                    format: crate::linear_format::standard_linear_format(
                        &weight,
                        args.linear_format_for(&weight),
                    )?,
                },
                context,
            )
        };
        let norm = |field: &str| {
            B::normalization(
                NormalizationConstructionSpec::learned(
                    head_dimensions,
                    args.rms_norm_eps,
                    ParameterSpec::trainable(format!("{prefix}.{field}.weight"))
                        .map_err(Error::backend)?,
                ),
                context,
            )
        };
        let convolution = |field: &str, channels| {
            CausalDepthwiseConvolution::new(
                CausalDepthwiseConvolutionSpec {
                    channels,
                    kernel_size: args.sconv_kernel_size,
                    weight: ParameterSpec::trainable(format!("{prefix}.{field}.weight"))
                        .map_err(Error::backend)?,
                    bias: None,
                    activation: ConvolutionActivation::Identity,
                },
                context,
            )
        };
        Ok(Self {
            query_heads,
            key_value_heads,
            head_dimensions,
            relative_dimensions: args.d_rel,
            relative_extent,
            policy,
            log_scaling_floor: args.log_scaling_n_floor,
            log_scaling_alpha: args.log_scaling_alpha,
            query: linear(
                "q_proj",
                args.hidden_size,
                query_heads * head_dimensions,
                args.q_bias,
            )?,
            key: linear(
                "k_proj",
                args.hidden_size,
                key_value_heads * head_dimensions,
                false,
            )?,
            value: linear(
                "v_proj",
                args.hidden_size,
                key_value_heads * head_dimensions,
                false,
            )?,
            relative: linear("r_proj", args.hidden_size, query_heads * args.d_rel, false)?,
            output: linear(
                "o_proj",
                query_heads * head_dimensions,
                args.hidden_size,
                args.o_bias,
            )?,
            query_norm: norm("q_norm")?,
            key_norm: norm("k_norm")?,
            relative_projection: Parameter::unloaded(
                ParameterSpec::trainable(format!("{prefix}.rel_proj")).map_err(Error::backend)?,
                &[args.d_rel, relative_extent],
                context,
            )?,
            key_convolution: convolution("k_sconv", key_value_heads * head_dimensions)?,
            value_convolution: convolution("v_sconv", key_value_heads * head_dimensions)?,
        })
    }

    /// Applies attention and replaces the two projection-convolution histories.
    pub fn forward<C: AuxiliaryConvolutionState<B::Tensor>>(
        &mut self,
        hidden: &B::Tensor,
        state: Option<&mut C>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let batch = hidden.dim(0);
        let sequence = hidden.dim(1);
        let query_offset = state.as_ref().map_or(0, |state| state.offset());
        let query = self.query.forward(hidden, context)?;
        let key = self.key.forward(hidden, context)?;
        let value = self.value.forward(hidden, context)?;
        let relative = self.relative.forward(hidden, context)?;
        let (key, value, keys, values, key_offset) = if let Some(state) = state {
            let key = residual_convolution(
                &self.key_convolution,
                &key,
                state.convolution_state(0)?,
                context,
            )?;
            let value = residual_convolution(
                &self.value_convolution,
                &value,
                state.convolution_state(1)?,
                context,
            )?;
            let normalized_key = self.key_norm.forward(
                &key.reshape(
                    &[batch, sequence, self.key_value_heads, self.head_dimensions],
                    context,
                )?,
                context,
            )?;
            let normalized_key = normalized_key.transpose_axes(&[0, 2, 1, 3], context)?;
            let value_heads = value
                .reshape(
                    &[batch, sequence, self.key_value_heads, self.head_dimensions],
                    context,
                )?
                .transpose_axes(&[0, 2, 1, 3], context)?;
            let (keys, values) =
                state.update_for_attention(normalized_key, value_heads, context)?;
            let key_offset = query_offset + sequence - keys.dim(2);
            (key, value, keys, values, key_offset)
        } else {
            let mut key_history = None;
            let mut value_history = None;
            let key = residual_convolution(&self.key_convolution, &key, &mut key_history, context)?;
            let value =
                residual_convolution(&self.value_convolution, &value, &mut value_history, context)?;
            let keys = self.key_norm.forward(
                &key.reshape(
                    &[batch, sequence, self.key_value_heads, self.head_dimensions],
                    context,
                )?,
                context,
            )?;
            let keys = keys.transpose_axes(&[0, 2, 1, 3], context)?;
            let values = value
                .reshape(
                    &[batch, sequence, self.key_value_heads, self.head_dimensions],
                    context,
                )?
                .transpose_axes(&[0, 2, 1, 3], context)?;
            (key, value, keys, values, 0)
        };
        let _ = (key, value);
        let queries = self.query_norm.forward(
            &query.reshape(
                &[batch, sequence, self.query_heads, self.head_dimensions],
                context,
            )?,
            context,
        )?;
        let queries = queries.transpose_axes(&[0, 2, 1, 3], context)?;
        let profiles = relative.reshape(
            &[batch, sequence, self.query_heads, self.relative_dimensions],
            context,
        )?;
        let profiles = B::Tensor::matmul(&profiles, self.relative_projection.as_ref(), context)?
            .transpose_axes(&[0, 2, 1, 3], context)?;
        debug_assert_eq!(profiles.dim(3), self.relative_extent);
        let attended = B::relative_attention(
            RelativeAttentionInput {
                queries: &queries,
                keys: &keys,
                values: &values,
                profiles: &profiles,
                query_offset,
                key_offset,
                window: self.policy.window().map(|window| window.get() as i32),
                log_scaling_floor: self.log_scaling_floor,
                log_scaling_alpha: self.log_scaling_alpha,
            },
            context,
        )?;
        let attended = attended.transpose_axes(&[0, 2, 1, 3], context)?.reshape(
            &[batch, sequence, self.query_heads * self.head_dimensions],
            context,
        )?;
        self.output.forward(&attended, context)
    }

    /// Applies attention with a row-parallel output projection.
    pub fn forward_parallel<C: AuxiliaryConvolutionState<B::Tensor>>(
        &mut self,
        hidden: &B::Tensor,
        state: Option<&mut C>,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        B: GroupedNeuralBackend,
    {
        let batch = hidden.dim(0);
        let sequence = hidden.dim(1);
        let query_offset = state.as_ref().map_or(0, |state| state.offset());
        let query = self.query.forward(hidden, context)?;
        let key = self.key.forward(hidden, context)?;
        let value = self.value.forward(hidden, context)?;
        let relative = self.relative.forward(hidden, context)?;
        let (keys, values, key_offset) = if let Some(state) = state {
            let key = residual_convolution(
                &self.key_convolution,
                &key,
                state.convolution_state(0)?,
                context,
            )?;
            let value = residual_convolution(
                &self.value_convolution,
                &value,
                state.convolution_state(1)?,
                context,
            )?;
            let keys = self.key_norm.forward(
                &key.reshape(
                    &[batch, sequence, self.key_value_heads, self.head_dimensions],
                    context,
                )?,
                context,
            )?;
            let keys = keys.transpose_axes(&[0, 2, 1, 3], context)?;
            let values = value
                .reshape(
                    &[batch, sequence, self.key_value_heads, self.head_dimensions],
                    context,
                )?
                .transpose_axes(&[0, 2, 1, 3], context)?;
            let (keys, values) = state.update_for_attention(keys, values, context)?;
            let key_offset = query_offset + sequence - keys.dim(2);
            (keys, values, key_offset)
        } else {
            let mut key_history = None;
            let mut value_history = None;
            let key = residual_convolution(&self.key_convolution, &key, &mut key_history, context)?;
            let value =
                residual_convolution(&self.value_convolution, &value, &mut value_history, context)?;
            let keys = self.key_norm.forward(
                &key.reshape(
                    &[batch, sequence, self.key_value_heads, self.head_dimensions],
                    context,
                )?,
                context,
            )?;
            let keys = keys.transpose_axes(&[0, 2, 1, 3], context)?;
            let values = value
                .reshape(
                    &[batch, sequence, self.key_value_heads, self.head_dimensions],
                    context,
                )?
                .transpose_axes(&[0, 2, 1, 3], context)?;
            (keys, values, 0)
        };
        let queries = self.query_norm.forward(
            &query.reshape(
                &[batch, sequence, self.query_heads, self.head_dimensions],
                context,
            )?,
            context,
        )?;
        let queries = queries.transpose_axes(&[0, 2, 1, 3], context)?;
        let profiles = relative.reshape(
            &[batch, sequence, self.query_heads, self.relative_dimensions],
            context,
        )?;
        let profiles = B::Tensor::matmul(&profiles, self.relative_projection.as_ref(), context)?
            .transpose_axes(&[0, 2, 1, 3], context)?;
        let attended = B::relative_attention(
            RelativeAttentionInput {
                queries: &queries,
                keys: &keys,
                values: &values,
                profiles: &profiles,
                query_offset,
                key_offset,
                window: self.policy.window().map(|window| window.get() as i32),
                log_scaling_floor: self.log_scaling_floor,
                log_scaling_alpha: self.log_scaling_alpha,
            },
            context,
        )?;
        let attended = attended.transpose_axes(&[0, 2, 1, 3], context)?.reshape(
            &[batch, sequence, self.query_heads * self.head_dimensions],
            context,
        )?;
        B::row_parallel_linear(&mut self.output, &attended, parallel, context)
    }
}

fn residual_convolution<B: NeuralBackend>(
    convolution: &CausalDepthwiseConvolution<B>,
    input: &B::Tensor,
    history: &mut Option<B::Tensor>,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<B::Tensor, Error> {
    let output = convolution.forward(input, history.as_ref(), context)?;
    *history = output.history;
    input.add(&output.output, context)
}

/// Dense SwiGLU branch with its learned global scalar.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct DenseMlp<B: NeuralBackend> {
    /// Gate projection.
    pub gate: B::Linear,
    /// Up projection.
    pub up: B::Linear,
    /// Down projection.
    pub down: B::Linear,
    /// Learned global branch scalar.
    pub global_scale: Parameter<B::Tensor>,
}

impl<B: NeuralBackend> DenseMlp<B> {
    fn new_at(
        args: &TextArgs,
        block_root: &str,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let prefix = format!("{block_root}.dense");
        let intermediate = args.dense_intermediate_size();
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
            gate: linear("gate_proj", args.hidden_size, intermediate)?,
            up: linear("up_proj", args.hidden_size, intermediate)?,
            down: linear("down_proj", intermediate, args.hidden_size)?,
            global_scale: Parameter::unloaded(
                ParameterSpec::trainable(format!("{block_root}.dense_global_scale"))
                    .map_err(Error::backend)?,
                &[1],
                context,
            )?,
        })
    }

    fn forward(
        &mut self,
        hidden: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let gate = self.gate.forward(hidden, context)?;
        let up = self.up.forward(hidden, context)?;
        let hidden = B::gated_product(gate, up, eredu_nn::GatedProductPolicy::default(), context)?;
        self.down
            .forward(&hidden, context)?
            .multiply(self.global_scale.as_ref(), context)
    }

    fn forward_parallel(
        &mut self,
        hidden: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        B: GroupedNeuralBackend,
    {
        let gate = self.gate.forward(hidden, context)?;
        let up = self.up.forward(hidden, context)?;
        let hidden = B::gated_product(gate, up, eredu_nn::GatedProductPolicy::default(), context)?;
        B::row_parallel_linear(&mut self.down, &hidden, parallel, context)?
            .multiply(self.global_scale.as_ref(), context)
    }
}

/// Sparse Inkling branch with jointly normalized routed and shared experts.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct SparseMlp<B: GroupedNeuralBackend> {
    #[parameter(skip)]
    routed_count: i32,
    #[parameter(skip)]
    shared_count: i32,
    #[parameter(skip)]
    top_k: i32,
    #[parameter(skip)]
    coefficient_scale: f32,
    /// Joint routed/shared router projection.
    pub router_weight: Parameter<B::Tensor>,
    /// Routed top-k correction bias.
    pub router_bias: Parameter<B::Tensor>,
    /// Learned global route multiplier.
    pub global_scale: Parameter<B::Tensor>,
    /// Selectable routed experts.
    pub routed_experts: B::GatedProductGroups,
    /// Always-on shared experts.
    pub shared_experts: B::GatedProductGroups,
}

impl<B: GroupedNeuralBackend> SparseMlp<B> {
    fn new_at(
        args: &TextArgs,
        block_root: &str,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let prefix = format!("{block_root}.moe");
        let bank = |field: &str, count| {
            B::grouped_gated_product(expert_bank_spec_at(args, &prefix, field, count)?, context)
        };
        Ok(Self {
            routed_count: args.n_routed_experts,
            shared_count: args.n_shared_experts,
            top_k: args.num_experts_per_tok,
            coefficient_scale: args.route_scale,
            router_weight: Parameter::unloaded(
                ParameterSpec::trainable(format!("{prefix}.router.weight"))
                    .map_err(Error::backend)?,
                &[
                    args.n_routed_experts + args.n_shared_experts,
                    args.hidden_size,
                ],
                context,
            )?,
            router_bias: Parameter::unloaded(
                ParameterSpec::trainable(format!("{prefix}.router.bias"))
                    .map_err(Error::backend)?,
                &[args.n_routed_experts],
                context,
            )?,
            global_scale: Parameter::unloaded(
                ParameterSpec::trainable(format!("{prefix}.router.global_scale"))
                    .map_err(Error::backend)?,
                &[1],
                context,
            )?,
            routed_experts: bank("experts", args.n_routed_experts)?,
            shared_experts: bank("shared_experts", args.n_shared_experts)?,
        })
    }

    fn forward(
        &mut self,
        hidden: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let routes = B::joint_group_selection(
            JointGroupSelectionInput::new(
                hidden,
                self.router_weight.as_ref(),
                self.router_bias.as_ref(),
                self.global_scale.as_ref(),
                JointGroupSelectionSpec::new(
                    self.routed_count,
                    self.shared_count,
                    self.top_k,
                    self.coefficient_scale,
                )?,
            )?,
            context,
        )?;
        let routed = self.routed_experts.forward_grouped(
            hidden,
            &GroupSelection::new(
                routes.primary_indices().clone(),
                routes.primary_coefficients().clone(),
                routes.primary_coefficients().clone(),
            ),
            context,
        )?;
        let tokens = hidden.shape()[..hidden.shape().len() - 1]
            .iter()
            .try_fold(1_i32, |tokens, dimension| tokens.checked_mul(*dimension))
            .ok_or_else(|| Error::backend("Inkling token count overflowed"))?;
        let shared_ids = B::Tensor::from_i32_slice(
            &(0..self.shared_count).collect::<Vec<_>>(),
            &[1, self.shared_count],
            context,
        )?
        .broadcast_to(&[tokens, self.shared_count], context)?;
        let shared = self.shared_experts.forward_grouped(
            hidden,
            &GroupSelection::new(
                shared_ids,
                routes.always_on_coefficients().clone(),
                routes.always_on_coefficients().clone(),
            ),
            context,
        )?;
        routed.add(&shared, context)
    }

    fn forward_parallel(
        &mut self,
        hidden: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let routes = B::joint_group_selection(
            JointGroupSelectionInput::new(
                hidden,
                self.router_weight.as_ref(),
                self.router_bias.as_ref(),
                self.global_scale.as_ref(),
                JointGroupSelectionSpec::new(
                    self.routed_count,
                    self.shared_count,
                    self.top_k,
                    self.coefficient_scale,
                )?,
            )?,
            context,
        )?;
        let routed = self.routed_experts.forward_grouped_tensor_parallel(
            hidden,
            &GroupSelection::new(
                routes.primary_indices().clone(),
                routes.primary_coefficients().clone(),
                routes.primary_coefficients().clone(),
            ),
            B::parallel_size(parallel),
            context,
        )?;
        let tokens = hidden.shape()[..hidden.shape().len() - 1]
            .iter()
            .try_fold(1_i32, |tokens, dimension| tokens.checked_mul(*dimension))
            .ok_or_else(|| Error::backend("Inkling token count overflowed"))?;
        let shared_ids = B::Tensor::from_i32_slice(
            &(0..self.shared_count).collect::<Vec<_>>(),
            &[1, self.shared_count],
            context,
        )?
        .broadcast_to(&[tokens, self.shared_count], context)?;
        let shared = self.shared_experts.forward_grouped_tensor_parallel(
            hidden,
            &GroupSelection::new(
                shared_ids,
                routes.always_on_coefficients().clone(),
                routes.always_on_coefficients().clone(),
            ),
            B::parallel_size(parallel),
            context,
        )?;
        let output =
            eredu_runtime::combine_tensor_parallel_expert_outputs::<B>(routed, shared, context)?;
        eredu_runtime::reduce_tensor_parallel_expert_output::<B>(output, parallel, context)
    }

    fn forward_with_provider<P>(
        &mut self,
        hidden: &B::Tensor,
        layer: usize,
        shared_layer: usize,
        pass: ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        let routes = B::joint_group_selection(
            JointGroupSelectionInput::new(
                hidden,
                self.router_weight.as_ref(),
                self.router_bias.as_ref(),
                self.global_scale.as_ref(),
                JointGroupSelectionSpec::new(
                    self.routed_count,
                    self.shared_count,
                    self.top_k,
                    self.coefficient_scale,
                )?,
            )?,
            context,
        )?;
        let routed_routes = GroupSelection::new(
            routes.primary_indices().clone(),
            routes.primary_coefficients().clone(),
            routes.primary_coefficients().clone(),
        );
        let routed = provider
            .forward_grouped(
                &mut self.routed_experts,
                RoutedExpertRequest {
                    layer,
                    input: hidden,
                    routes: &routed_routes,
                    pass,
                },
                context,
            )
            .map_err(Error::backend)?;
        let tokens = hidden.shape()[..hidden.shape().len() - 1]
            .iter()
            .try_fold(1_i32, |tokens, dimension| tokens.checked_mul(*dimension))
            .ok_or_else(|| Error::backend("Inkling token count overflowed"))?;
        let shared_ids = B::Tensor::from_i32_slice(
            &(0..self.shared_count).collect::<Vec<_>>(),
            &[1, self.shared_count],
            context,
        )?
        .broadcast_to(&[tokens, self.shared_count], context)?;
        let shared_routes = GroupSelection::new(
            shared_ids,
            routes.always_on_coefficients().clone(),
            routes.always_on_coefficients().clone(),
        );
        let shared = provider
            .forward_grouped(
                &mut self.shared_experts,
                RoutedExpertRequest {
                    layer: shared_layer,
                    input: hidden,
                    routes: &shared_routes,
                    pass,
                },
                context,
            )
            .map_err(Error::backend)?;
        routed.add(&shared, context)
    }

    fn forward_parallel_with_provider<P>(
        &mut self,
        hidden: &B::Tensor,
        layer: usize,
        shared_layer: usize,
        pass: ExpertPass,
        provider: &mut P,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        let routes = B::joint_group_selection(
            JointGroupSelectionInput::new(
                hidden,
                self.router_weight.as_ref(),
                self.router_bias.as_ref(),
                self.global_scale.as_ref(),
                JointGroupSelectionSpec::new(
                    self.routed_count,
                    self.shared_count,
                    self.top_k,
                    self.coefficient_scale,
                )?,
            )?,
            context,
        )?;
        let routed_routes = GroupSelection::new(
            routes.primary_indices().clone(),
            routes.primary_coefficients().clone(),
            routes.primary_coefficients().clone(),
        );
        let routed = provider
            .forward_grouped_tensor_parallel(
                &mut self.routed_experts,
                RoutedExpertRequest {
                    layer,
                    input: hidden,
                    routes: &routed_routes,
                    pass,
                },
                B::parallel_size(parallel),
                context,
            )
            .map_err(Error::backend)?;
        let tokens = hidden.shape()[..hidden.shape().len() - 1]
            .iter()
            .try_fold(1_i32, |tokens, dimension| tokens.checked_mul(*dimension))
            .ok_or_else(|| Error::backend("Inkling token count overflowed"))?;
        let shared_ids = B::Tensor::from_i32_slice(
            &(0..self.shared_count).collect::<Vec<_>>(),
            &[1, self.shared_count],
            context,
        )?
        .broadcast_to(&[tokens, self.shared_count], context)?;
        let shared_routes = GroupSelection::new(
            shared_ids,
            routes.always_on_coefficients().clone(),
            routes.always_on_coefficients().clone(),
        );
        let shared = provider
            .forward_grouped_tensor_parallel(
                &mut self.shared_experts,
                RoutedExpertRequest {
                    layer: shared_layer,
                    input: hidden,
                    routes: &shared_routes,
                    pass,
                },
                B::parallel_size(parallel),
                context,
            )
            .map_err(Error::backend)?;
        let output =
            eredu_runtime::combine_routed_expert_tensor_parallel::<B>(routed, shared, context)?;
        eredu_runtime::reduce_routed_expert_tensor_parallel::<B>(output, parallel, context)
    }
}

/// Returns the architecture-owned routed or shared expert specification for one cache layer.
pub fn expert_bank_spec(
    args: &ModelArgs,
    cache_layer: usize,
) -> Result<GroupedGatedProductSpec, Error> {
    let layers = args.text_config.num_hidden_layers as usize;
    let (layer, field, count) = if cache_layer < layers {
        (cache_layer, "experts", args.text_config.n_routed_experts)
    } else {
        (
            cache_layer - layers,
            "shared_experts",
            args.text_config.n_shared_experts,
        )
    };
    expert_bank_spec_at(
        &args.text_config,
        &format!("model.layers.{layer}.moe"),
        field,
        count,
    )
}

/// Exact routed and shared bank specifications for one localized sparse layer.
pub(crate) fn localized_expert_bank_specs(
    args: &ModelArgs,
    layer: usize,
    local: &TextArgs,
    routed_expert_count: i32,
) -> Result<(GroupedGatedProductSpec, GroupedGatedProductSpec), Error> {
    let routed = expert_bank_spec(args, layer)?
        .with_group_geometry(routed_expert_count, local.moe_intermediate_size())?;
    let cache_layer = usize::try_from(args.text_config.num_hidden_layers)
        .map_err(Error::backend)?
        .checked_add(layer)
        .ok_or_else(|| Error::backend("Inkling shared expert layer overflowed"))?;
    let shared = expert_bank_spec(args, cache_layer)?
        .with_group_geometry(1, local.moe_intermediate_size())?;
    Ok((routed, shared))
}

fn expert_bank_spec_at(
    args: &TextArgs,
    prefix: &str,
    field: &str,
    count: i32,
) -> Result<GroupedGatedProductSpec, Error> {
    let gate_up = format!("{prefix}.{field}.gate_up_proj");
    let down = format!("{prefix}.{field}.down_proj");
    GroupedGatedProductSpec::new(
        count,
        args.hidden_size,
        args.moe_intermediate_size(),
        args.hidden_size,
        eredu_nn::GatedProductPolicy::ordinary_silu(),
        GatedProductGroupLayout::Packed {
            gate_up: standard_expert_projection(&gate_up, None, args.linear_format_for(&gate_up))?,
            down: standard_expert_projection(&down, None, args.linear_format_for(&down))?,
        },
    )
}

/// Dense or sparse feed-forward branch selected by the normalized schedule.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub enum FeedForward<B: GroupedNeuralBackend> {
    /// Dense SwiGLU branch.
    Dense(DenseMlp<B>),
    /// Routed plus shared expert branch.
    Sparse(SparseMlp<B>),
}

impl<B: GroupedNeuralBackend> FeedForward<B> {
    fn forward(
        &mut self,
        hidden: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        match self {
            Self::Dense(dense) => dense.forward(hidden, context),
            Self::Sparse(sparse) => sparse.forward(hidden, context),
        }
    }

    fn forward_parallel(
        &mut self,
        hidden: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        match self {
            Self::Dense(dense) => dense.forward_parallel(hidden, parallel, context),
            Self::Sparse(sparse) => sparse.forward_parallel(hidden, parallel, context),
        }
    }

    fn forward_with_provider<P>(
        &mut self,
        hidden: &B::Tensor,
        layer: usize,
        shared_layer: usize,
        pass: ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        match self {
            Self::Dense(dense) => dense.forward(hidden, context),
            Self::Sparse(sparse) => {
                sparse.forward_with_provider(hidden, layer, shared_layer, pass, provider, context)
            }
        }
    }

    fn forward_parallel_with_provider<P>(
        &mut self,
        hidden: &B::Tensor,
        layer: usize,
        shared_layer: usize,
        pass: ExpertPass,
        provider: &mut P,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        match self {
            Self::Dense(dense) => dense.forward_parallel(hidden, parallel, context),
            Self::Sparse(sparse) => sparse.forward_parallel_with_provider(
                hidden,
                layer,
                shared_layer,
                pass,
                provider,
                parallel,
                context,
            ),
        }
    }
}

/// One ordinary Inkling decoder layer.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct DecoderLayer<B: GroupedNeuralBackend> {
    #[parameter(skip)]
    layer: usize,
    #[parameter(skip)]
    shared_expert_layer: usize,
    /// Pre-attention normalization.
    pub input_norm: B::Normalization,
    /// Learned-relative attention.
    pub attention: Attention<B>,
    /// Residual causal convolution over the attention delta.
    pub attention_convolution: CausalDepthwiseConvolution<B>,
    /// Pre-feed-forward normalization.
    pub post_attention_norm: B::Normalization,
    /// Scheduled dense or sparse branch.
    pub feed_forward: FeedForward<B>,
    /// Residual causal convolution over the feed-forward delta.
    pub feed_forward_convolution: CausalDepthwiseConvolution<B>,
}

impl<B: GroupedNeuralBackend> DecoderLayer<B> {
    /// Builds one scheduled decoder layer.
    pub fn new(
        args: &TextArgs,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let policy = args
            .layer_policy(layer)
            .ok_or_else(|| Error::backend(format!("missing Inkling layer policy {layer}")))?;
        Self::new_at_with_expert_layers(
            args,
            policy,
            &format!("model.layers.{layer}"),
            layer,
            args.num_hidden_layers as usize + layer,
            context,
        )
    }

    /// Builds one layer under an explicit architecture-owned block root.
    pub fn new_at(
        args: &TextArgs,
        policy: LayerPolicy,
        block_root: &str,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        Self::new_at_with_expert_layers(args, policy, block_root, 0, 0, context)
    }

    fn new_at_with_expert_layers(
        args: &TextArgs,
        policy: LayerPolicy,
        block_root: &str,
        layer: usize,
        shared_expert_layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let prefix = block_root;
        let norm = |field: &str| {
            B::normalization(
                NormalizationConstructionSpec::learned(
                    args.hidden_size,
                    args.rms_norm_eps,
                    ParameterSpec::trainable(format!("{prefix}.{field}.weight"))
                        .map_err(Error::backend)?,
                ),
                context,
            )
        };
        let convolution = |field: &str| {
            CausalDepthwiseConvolution::new(
                CausalDepthwiseConvolutionSpec {
                    channels: args.hidden_size,
                    kernel_size: args.sconv_kernel_size,
                    weight: ParameterSpec::trainable(format!("{prefix}.{field}.weight"))
                        .map_err(Error::backend)?,
                    bias: None,
                    activation: ConvolutionActivation::Identity,
                },
                context,
            )
        };
        Ok(Self {
            layer,
            shared_expert_layer,
            input_norm: norm("input_layernorm")?,
            attention: Attention::new_at(args, policy.attention, block_root, context)?,
            attention_convolution: convolution("attn_sconv")?,
            post_attention_norm: norm("post_attention_layernorm")?,
            feed_forward: match policy.feed_forward {
                FeedForwardPolicy::Dense => {
                    FeedForward::Dense(DenseMlp::new_at(args, block_root, context)?)
                }
                FeedForwardPolicy::SparseMoe => {
                    FeedForward::Sparse(SparseMlp::new_at(args, block_root, context)?)
                }
            },
            feed_forward_convolution: convolution("mlp_sconv")?,
        })
    }

    /// Runs one layer and replaces all four bounded convolution histories.
    pub fn forward<C: AuxiliaryConvolutionState<B::Tensor>>(
        &mut self,
        hidden: &B::Tensor,
        state: Option<&mut C>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        match state {
            Some(state) => {
                let normalized = self.input_norm.forward(hidden, context)?;
                let attention = self
                    .attention
                    .forward(&normalized, Some(&mut *state), context)?;
                let attention = residual_convolution(
                    &self.attention_convolution,
                    &attention,
                    state.convolution_state(2)?,
                    context,
                )?;
                let hidden = hidden.add(&attention, context)?;
                let normalized = self.post_attention_norm.forward(&hidden, context)?;
                let feed_forward = self.feed_forward.forward(&normalized, context)?;
                let feed_forward = residual_convolution(
                    &self.feed_forward_convolution,
                    &feed_forward,
                    state.convolution_state(3)?,
                    context,
                )?;
                hidden.add(&feed_forward, context)
            }
            None => {
                let normalized = self.input_norm.forward(hidden, context)?;
                let attention = self
                    .attention
                    .forward::<NoCache>(&normalized, None, context)?;
                let mut attention_history = None;
                let attention = residual_convolution(
                    &self.attention_convolution,
                    &attention,
                    &mut attention_history,
                    context,
                )?;
                let hidden = hidden.add(&attention, context)?;
                let normalized = self.post_attention_norm.forward(&hidden, context)?;
                let feed_forward = self.feed_forward.forward(&normalized, context)?;
                let mut feed_forward_history = None;
                let feed_forward = residual_convolution(
                    &self.feed_forward_convolution,
                    &feed_forward,
                    &mut feed_forward_history,
                    context,
                )?;
                hidden.add(&feed_forward, context)
            }
        }
    }

    /// Runs the canonical layer with rank-local projections and TP reductions.
    pub fn forward_parallel<C: AuxiliaryConvolutionState<B::Tensor>>(
        &mut self,
        hidden: &B::Tensor,
        state: Option<&mut C>,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        match state {
            Some(state) => {
                let normalized = self.input_norm.forward(hidden, context)?;
                let attention = self.attention.forward_parallel(
                    &normalized,
                    Some(&mut *state),
                    parallel,
                    context,
                )?;
                let attention = residual_convolution(
                    &self.attention_convolution,
                    &attention,
                    state.convolution_state(2)?,
                    context,
                )?;
                let hidden = hidden.add(&attention, context)?;
                let normalized = self.post_attention_norm.forward(&hidden, context)?;
                let feed_forward =
                    self.feed_forward
                        .forward_parallel(&normalized, parallel, context)?;
                let feed_forward = residual_convolution(
                    &self.feed_forward_convolution,
                    &feed_forward,
                    state.convolution_state(3)?,
                    context,
                )?;
                hidden.add(&feed_forward, context)
            }
            None => {
                let normalized = self.input_norm.forward(hidden, context)?;
                let attention = self.attention.forward_parallel::<NoCache>(
                    &normalized,
                    None,
                    parallel,
                    context,
                )?;
                let mut attention_history = None;
                let attention = residual_convolution(
                    &self.attention_convolution,
                    &attention,
                    &mut attention_history,
                    context,
                )?;
                let hidden = hidden.add(&attention, context)?;
                let normalized = self.post_attention_norm.forward(&hidden, context)?;
                let feed_forward =
                    self.feed_forward
                        .forward_parallel(&normalized, parallel, context)?;
                let mut feed_forward_history = None;
                let feed_forward = residual_convolution(
                    &self.feed_forward_convolution,
                    &feed_forward,
                    &mut feed_forward_history,
                    context,
                )?;
                hidden.add(&feed_forward, context)
            }
        }
    }

    /// Runs the canonical layer equations through a runtime-owned expert provider.
    pub fn forward_with_provider<C, P>(
        &mut self,
        hidden: &B::Tensor,
        state: Option<&mut C>,
        pass: ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        C: AuxiliaryConvolutionState<B::Tensor>,
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        match state {
            Some(state) => {
                let normalized = self.input_norm.forward(hidden, context)?;
                let attention = self
                    .attention
                    .forward(&normalized, Some(&mut *state), context)?;
                let attention = residual_convolution(
                    &self.attention_convolution,
                    &attention,
                    state.convolution_state(2)?,
                    context,
                )?;
                let hidden = hidden.add(&attention, context)?;
                let normalized = self.post_attention_norm.forward(&hidden, context)?;
                let feed_forward = self.feed_forward.forward_with_provider(
                    &normalized,
                    self.layer,
                    self.shared_expert_layer,
                    pass,
                    provider,
                    context,
                )?;
                let feed_forward = residual_convolution(
                    &self.feed_forward_convolution,
                    &feed_forward,
                    state.convolution_state(3)?,
                    context,
                )?;
                hidden.add(&feed_forward, context)
            }
            None => {
                let normalized = self.input_norm.forward(hidden, context)?;
                let attention = self
                    .attention
                    .forward::<NoCache>(&normalized, None, context)?;
                let mut attention_history = None;
                let attention = residual_convolution(
                    &self.attention_convolution,
                    &attention,
                    &mut attention_history,
                    context,
                )?;
                let hidden = hidden.add(&attention, context)?;
                let normalized = self.post_attention_norm.forward(&hidden, context)?;
                let feed_forward = self.feed_forward.forward_with_provider(
                    &normalized,
                    self.layer,
                    self.shared_expert_layer,
                    pass,
                    provider,
                    context,
                )?;
                let mut feed_forward_history = None;
                let feed_forward = residual_convolution(
                    &self.feed_forward_convolution,
                    &feed_forward,
                    &mut feed_forward_history,
                    context,
                )?;
                hidden.add(&feed_forward, context)
            }
        }
    }

    /// Runs the canonical TP layer while runtime owns routed expert residency.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_parallel_with_provider<C, P>(
        &mut self,
        hidden: &B::Tensor,
        state: Option<&mut C>,
        pass: ExpertPass,
        provider: &mut P,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        C: AuxiliaryConvolutionState<B::Tensor>,
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        match state {
            Some(state) => {
                let normalized = self.input_norm.forward(hidden, context)?;
                let attention = self.attention.forward_parallel(
                    &normalized,
                    Some(&mut *state),
                    parallel,
                    context,
                )?;
                let attention = residual_convolution(
                    &self.attention_convolution,
                    &attention,
                    state.convolution_state(2)?,
                    context,
                )?;
                let hidden = hidden.add(&attention, context)?;
                let normalized = self.post_attention_norm.forward(&hidden, context)?;
                let feed_forward = self.feed_forward.forward_parallel_with_provider(
                    &normalized,
                    self.layer,
                    self.shared_expert_layer,
                    pass,
                    provider,
                    parallel,
                    context,
                )?;
                let feed_forward = residual_convolution(
                    &self.feed_forward_convolution,
                    &feed_forward,
                    state.convolution_state(3)?,
                    context,
                )?;
                hidden.add(&feed_forward, context)
            }
            None => {
                let normalized = self.input_norm.forward(hidden, context)?;
                let attention = self.attention.forward_parallel::<NoCache>(
                    &normalized,
                    None,
                    parallel,
                    context,
                )?;
                let mut attention_history = None;
                let attention = residual_convolution(
                    &self.attention_convolution,
                    &attention,
                    &mut attention_history,
                    context,
                )?;
                let hidden = hidden.add(&attention, context)?;
                let normalized = self.post_attention_norm.forward(&hidden, context)?;
                let feed_forward = self.feed_forward.forward_parallel_with_provider(
                    &normalized,
                    self.layer,
                    self.shared_expert_layer,
                    pass,
                    provider,
                    parallel,
                    context,
                )?;
                let mut feed_forward_history = None;
                let feed_forward = residual_convolution(
                    &self.feed_forward_convolution,
                    &feed_forward,
                    &mut feed_forward_history,
                    context,
                )?;
                hidden.add(&feed_forward, context)
            }
        }
    }
}

/// Uninhabited cache adapter used by stateless prefill calls.
#[derive(Debug, Clone)]
struct NoCache;

impl<T: Tensor> AttentionCache<T> for NoCache {
    fn offset(&self) -> i32 {
        0
    }
    fn max_size(&self) -> Option<i32> {
        None
    }
    fn update_for_attention(&mut self, _: T, _: T, _: &T::Context) -> Result<(T, T), Error> {
        unreachable!("stateless Inkling attention never updates NoCache")
    }
    fn attention(&mut self, _: AttentionRequest<'_, T>, _: &T::Context) -> Result<T, Error> {
        unreachable!("stateless Inkling attention never calls NoCache")
    }
}

impl<T: Tensor> AuxiliaryConvolutionState<T> for NoCache {
    fn convolution_state(&mut self, _: u32) -> Result<&mut Option<T>, Error> {
        unreachable!("stateless Inkling attention never borrows NoCache histories")
    }
}

/// Inkling token embedding, ordinary decoder layers, and final norm.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct TextModel<B: GroupedNeuralBackend> {
    /// Token embedding table.
    pub embeddings: B::Embedding,
    /// Required post-embedding RMS normalization.
    pub embedding_norm: B::Normalization,
    /// Ordinary scheduled decoder layers.
    pub layers: Vec<DecoderLayer<B>>,
    /// Final decoder RMS normalization.
    pub final_norm: B::Normalization,
    /// Untied vocabulary projection.
    pub output: B::Linear,
    #[parameter(skip)]
    logits_scale: f32,
    #[parameter(skip)]
    output_vocabulary: i32,
}

impl<B: GroupedNeuralBackend> TextModel<B> {
    /// Builds the complete neutral text model.
    pub fn new(args: &TextArgs, context: &<B::Tensor as Tensor>::Context) -> Result<Self, Error> {
        let norm = |name: &str| {
            B::normalization(
                NormalizationConstructionSpec::learned(
                    args.hidden_size,
                    args.rms_norm_eps,
                    ParameterSpec::trainable(name).map_err(Error::backend)?,
                ),
                context,
            )
        };
        let output_weight = "lm_head.weight";
        Ok(Self {
            embeddings: B::embedding(
                EmbeddingSpec {
                    vocabulary: args.vocab_size,
                    dimensions: args.hidden_size,
                    weight: ParameterSpec::trainable("model.embed_tokens.weight")
                        .map_err(Error::backend)?,
                    format: crate::linear_format::standard_linear_format(
                        "model.embed_tokens.weight",
                        args.linear_format_for("model.embed_tokens.weight"),
                    )?,
                },
                context,
            )?,
            embedding_norm: norm("model.embed_norm.weight")?,
            layers: (0..args.num_hidden_layers as usize)
                .map(|layer| DecoderLayer::new(args, layer, context))
                .collect::<Result<_, _>>()?,
            final_norm: norm("model.norm.weight")?,
            output: B::linear(
                LinearSpec {
                    input: args.hidden_size,
                    output: args.vocab_size,
                    weight: ParameterSpec::trainable(output_weight).map_err(Error::backend)?,
                    bias: None,
                    format: crate::linear_format::standard_linear_format(
                        output_weight,
                        args.linear_format_for(output_weight),
                    )?,
                },
                context,
            )?,
            logits_scale: args.logits_mup_width_multiplier,
            output_vocabulary: args.unpadded_vocab_size.unwrap_or(args.vocab_size),
        })
    }

    /// Embeds and normalizes token IDs.
    pub fn embed(
        &mut self,
        tokens: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let hidden = self.embeddings.forward(tokens, context)?;
        self.embedding_norm.forward(&hidden, context)
    }

    /// Projects normalized hidden states using the exact muP divisor and
    /// protocol-visible vocabulary truncation.
    pub fn logits(
        &mut self,
        hidden: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let scaled = hidden.multiply_scalar(1.0 / self.logits_scale, context)?;
        let logits = self.output.forward(&scaled, context)?;
        if self.output_vocabulary == logits.shape()[logits.shape().len() - 1] {
            return Ok(logits);
        }
        let mut indexes = vec![eredu_nn::Index::Full; logits.shape().len()];
        *indexes.last_mut().expect("logits have vocabulary axis") =
            eredu_nn::Index::Range(0, self.output_vocabulary);
        logits.index(&indexes, context)
    }
}

/// Returns the exact history tensor shape for one convolution channel count.
pub fn convolution_history_shape(
    batch: i32,
    kernel_size: i32,
    channels: i32,
) -> Result<[i32; 3], Error> {
    if batch <= 0 || kernel_size <= 0 || channels <= 0 {
        return Err(Error::backend("invalid Inkling convolution state geometry"));
    }
    Ok([batch, kernel_size - 1, channels])
}

#[cfg(test)]
mod tests {
    use super::convolution_history_shape;

    #[test]
    fn declares_four_bounded_histories_with_exact_width() {
        assert_eq!(convolution_history_shape(2, 4, 16).unwrap(), [2, 3, 16]);
        assert!(convolution_history_shape(0, 4, 16).is_err());
    }
}
