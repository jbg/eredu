//! Backend-neutral Inkling decoder equations and bounded operator state.

use eredu_core::AttentionPolicy;
use eredu_nn::{
    AttentionCache, AttentionRequest, AuxiliaryConvolutionState, CausalDepthwiseConvolution,
    CausalDepthwiseConvolutionSpec, ConvolutionActivation, EmbeddingOperator, EmbeddingSpec, Error,
    GatedProductGroupLayout, GroupSelection, GroupedGatedProductSpec, GroupedNeuralBackend,
    JointGroupSelectionInput, JointGroupSelectionSpec, LinearOperator, LinearSpec, NeuralBackend,
    NormalizationConstructionSpec, NormalizationOperator, Parameter, ParameterSpec, Parameterized,
    RelativeAttentionInput, Tensor,
};
use eredu_runtime::{
    ExpertPass, RoutedExpertProvider, RoutedExpertRequest, TensorParallelRoutedExpertProvider,
};

use crate::decoder::ComponentInstrumentation;
use crate::linear_format::standard_expert_projection;

use super::{FeedForwardPolicy, LayerPolicy, ModelArgs, TextArgs};
pub(crate) mod construction;
pub(crate) use construction::{DenseLayerSpec, DecoderLayerSpec};

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
    fn uses_blockwise_attention(&self) -> bool {
        self.attention.uses_blockwise_attention()
    }
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
    fn relative_attention<B: NeuralBackend<Tensor = T>>(
        &mut self,
        request: RelativeAttentionInput<'_, T>,
        context: &T::Context,
    ) -> Result<T, Error> {
        self.attention.relative_attention::<B>(request, context)
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
pub struct Attention<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> {
    #[parameter(skip, metadata)]
    query_heads: i32,
    #[parameter(skip, metadata)]
    key_value_heads: i32,
    #[parameter(skip, metadata)]
    head_dimensions: i32,
    #[parameter(skip, metadata)]
    relative_dimensions: i32,
    #[parameter(skip, metadata)]
    relative_extent: i32,
    #[parameter(skip, metadata)]
    policy: AttentionPolicy,
    #[parameter(skip, metadata)]
    log_scaling_floor: Option<i32>,
    #[parameter(skip, metadata)]
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

impl<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> Attention<B> {
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
        crate::decoder::construction_specs::require_source_compiler::<B>(context)?;
        construction::AttentionSpec::new(args, policy, block_root)?.instantiate::<B>(context)
    }

    /// Applies attention and replaces the two projection-convolution histories.
    pub fn forward<C: AuxiliaryConvolutionState<B::Tensor>>(
        &mut self,
        hidden: &B::Tensor,
        state: Option<&mut C>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.forward_instrumented(
            hidden,
            state,
            None,
            context,
            &mut ComponentInstrumentation::disabled(),
        )
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
        B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    {
        self.forward_instrumented(
            hidden,
            state,
            Some(parallel),
            context,
            &mut ComponentInstrumentation::disabled(),
        )
    }

    /// Captures and changes aggregated channels before their actual projection.
    pub fn forward_instrumented<C: AuxiliaryConvolutionState<B::Tensor>>(
        &mut self,
        hidden: &B::Tensor,
        mut state: Option<&mut C>,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error> {
        let batch = hidden.dim(0);
        let sequence = hidden.dim(1);
        let query_offset = state.as_ref().map_or(0, |state| state.offset());
        let query = self.query.forward(hidden, context)?;
        let key = self.key.forward(hidden, context)?;
        let value = self.value.forward(hidden, context)?;
        let relative = self.relative.forward(hidden, context)?;
        instrumentation.observe("attention.query.projected", &query)?;
        instrumentation.observe("attention.key.projected", &key)?;
        instrumentation.observe("attention.value.projected", &value)?;
        instrumentation.observe("attention.relative.projected", &relative)?;
        let (key, value, keys, values, key_offset) = if let Some(state) = state.as_deref_mut() {
            let key = residual_convolution_with_state(
                &self.key_convolution,
                &key,
                Some(&mut *state),
                0,
                context,
            )?;
            let value = residual_convolution_with_state(
                &self.value_convolution,
                &value,
                Some(&mut *state),
                1,
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
        // These are the current positions after causal convolution, before key
        // head normalization and cache reuse. No historical cache is exported.
        instrumentation.observe("attention.key.convolved", &key)?;
        instrumentation.observe("attention.value.convolved", &value)?;
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
        let profiles = B::Tensor::matmul(&profiles, self.relative_projection.as_ref(), context)?;
        instrumentation.observe("attention.relative.profiles", &profiles)?;
        let profiles = profiles.transpose_axes(&[0, 2, 1, 3], context)?;
        debug_assert_eq!(profiles.dim(3), self.relative_extent);
        let request = RelativeAttentionInput {
            queries: &queries,
            keys: &keys,
            values: &values,
            profiles: &profiles,
            query_offset,
            key_offset,
            window: self.policy.window().map(|window| window.get() as i32),
            log_scaling_floor: self.log_scaling_floor,
            log_scaling_alpha: self.log_scaling_alpha,
        };
        let attended = match state {
            Some(state) => state.relative_attention::<B>(request, context)?,
            None => B::relative_attention(request, context)?,
        };
        let attended = attended.transpose_axes(&[0, 2, 1, 3], context)?.reshape(
            &[batch, sequence, self.query_heads * self.head_dimensions],
            context,
        )?;
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

fn residual_convolution<B: NeuralBackend + eredu_nn::DistributedNeuralBackend>(
    convolution: &CausalDepthwiseConvolution<B>,
    input: &B::Tensor,
    history: &mut Option<B::Tensor>,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<B::Tensor, Error> {
    let output = convolution.forward(input, history.as_ref(), context)?;
    *history = output.history;
    input.add(&output.output, context)
}

// The actual operator determines whether a history role exists. Wider kernels
// retain the exact slot lookup/error; width one never asks a KV-only owner for it.
fn residual_convolution_with_state<B, C>(
    convolution: &CausalDepthwiseConvolution<B>,
    input: &B::Tensor,
    state: Option<&mut C>,
    slot: u32,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<B::Tensor, Error>
where
    B: NeuralBackend + eredu_nn::DistributedNeuralBackend,
    C: AuxiliaryConvolutionState<B::Tensor>,
{
    let mut temporary = None;
    let history = if convolution.history_len() == 0 {
        &mut temporary
    } else {
        match state {
            Some(state) => state.convolution_state(slot)?,
            None => &mut temporary,
        }
    };
    residual_convolution(convolution, input, history, context)
}

/// Dense SwiGLU branch with its learned global scalar.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct DenseMlp<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> {
    /// Gate projection.
    pub gate: B::Linear,
    /// Up projection.
    pub up: B::Linear,
    /// Down projection.
    pub down: B::Linear,
    /// Learned global branch scalar.
    pub global_scale: Parameter<B::Tensor>,
}

impl<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> DenseMlp<B> {
    fn new_at(
        args: &TextArgs,
        block_root: &str,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        crate::decoder::construction_specs::require_source_compiler::<B>(context)?;
        construction::DenseSpec::new(args, block_root)?.instantiate::<B>(context)
    }

    /// Executes dense SwiGLU units and the learned scalar on the canonical path.
    pub fn forward_instrumented(
        &mut self,
        hidden: &B::Tensor,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error> {
        let gate = self.gate.forward(hidden, context)?;
        let up = self.up.forward(hidden, context)?;
        let units = B::gated_product(gate, up, eredu_nn::GatedProductPolicy::default(), context)?;
        let units = instrumentation.apply("feed_forward.units", units)?;
        let write = instrumentation.project::<B>(
            "feed_forward.write_input",
            &mut self.down,
            &units,
            parallel,
            context,
        )?;
        let write = instrumentation.apply("feed_forward.projection", write)?;
        instrumentation.observe("feed_forward.global_scale", self.global_scale.as_ref())?;
        write.multiply(self.global_scale.as_ref(), context)
    }
}

/// Sparse Inkling branch with jointly normalized routed and shared experts.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct SparseMlp<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> {
    #[parameter(skip, metadata)]
    routed_count: i32,
    #[parameter(skip, metadata)]
    shared_count: i32,
    #[parameter(skip, metadata)]
    top_k: i32,
    #[parameter(skip, metadata)]
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

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> SparseMlp<B> {
    fn shared_group_indices(
        tokens: i32,
        shared_count: i32,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let capacity = usize::try_from(tokens)
            .ok()
            .and_then(|tokens| {
                usize::try_from(shared_count)
                    .ok()
                    .and_then(|shared| tokens.checked_mul(shared))
            })
            .ok_or_else(|| Error::backend("Inkling shared route count overflowed"))?;
        let mut indices = Vec::with_capacity(capacity);
        for _ in 0..tokens {
            indices.extend(0..shared_count);
        }
        B::Tensor::from_i32_slice(&indices, &[tokens, shared_count], context)
    }

    fn new_at(
        args: &TextArgs,
        block_root: &str,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        crate::decoder::construction_specs::require_source_compiler::<B>(context)?;
        construction::SparseSpec::new(args, block_root, None)?.instantiate::<B>(context)
    }

    fn new_at_with_specs(args: &TextArgs, block_root: &str,
        routed: GroupedGatedProductSpec, shared: GroupedGatedProductSpec,
        context: &<B::Tensor as Tensor>::Context) -> Result<Self, Error> {
        crate::decoder::construction_specs::require_source_compiler::<B>(context)?;
        construction::SparseSpec::new(args, block_root, Some(crate::inkling::ExpertBankRealization {routed, shared}))?.instantiate::<B>(context)
    }

    fn forward_with_provider_instrumented<P>(
        &mut self,
        hidden: &B::Tensor,
        layer: usize,
        shared_layer: usize,
        pass: ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
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
        let routed = instrumentation.routed(
            "routing",
            RoutedExpertRequest {
                unit_observer: None,
                bank: eredu_runtime::RoutedBankId::new(0),
                layer: layer,
                input: hidden,
                routes: &routed_routes,
                pass,
            },
            |request| provider.forward_grouped(&mut self.routed_experts, request, context),
        )?;
        let tokens = hidden.shape()[..hidden.shape().len() - 1]
            .iter()
            .try_fold(1_i32, |tokens, dimension| tokens.checked_mul(*dimension))
            .ok_or_else(|| Error::backend("Inkling token count overflowed"))?;
        let shared_ids = Self::shared_group_indices(tokens, self.shared_count, context)?;
        let shared_routes = GroupSelection::new(
            shared_ids,
            routes.always_on_coefficients().clone(),
            routes.always_on_coefficients().clone(),
        );
        let shared = instrumentation.routed(
            "shared.routing",
            RoutedExpertRequest {
                unit_observer: None,
                bank: eredu_runtime::RoutedBankId::new(0),
                layer: shared_layer,
                input: hidden,
                routes: &shared_routes,
                pass,
            },
            |request| provider.forward_grouped(&mut self.shared_experts, request, context),
        )?;
        routed.add(&shared, context)
    }

    fn forward_parallel_with_provider_instrumented<P>(
        &mut self,
        hidden: &B::Tensor,
        layer: usize,
        shared_layer: usize,
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
        let routed = instrumentation.routed(
            "routing",
            RoutedExpertRequest {
                unit_observer: None,
                bank: eredu_runtime::RoutedBankId::new(0),
                layer: layer,
                input: hidden,
                routes: &routed_routes,
                pass,
            },
            |request| {
                provider.forward_grouped_tensor_parallel(
                    &mut self.routed_experts,
                    request,
                    B::parallel_size(parallel),
                    context,
                )
            },
        )?;
        let tokens = hidden.shape()[..hidden.shape().len() - 1]
            .iter()
            .try_fold(1_i32, |tokens, dimension| tokens.checked_mul(*dimension))
            .ok_or_else(|| Error::backend("Inkling token count overflowed"))?;
        let shared_ids = Self::shared_group_indices(tokens, self.shared_count, context)?;
        let shared_routes = GroupSelection::new(
            shared_ids,
            routes.always_on_coefficients().clone(),
            routes.always_on_coefficients().clone(),
        );
        let shared = instrumentation.routed(
            "shared.routing",
            RoutedExpertRequest {
                unit_observer: None,
                bank: eredu_runtime::RoutedBankId::new(0),
                layer: shared_layer,
                input: hidden,
                routes: &shared_routes,
                pass,
            },
            |request| {
                provider.forward_grouped_tensor_parallel(
                    &mut self.shared_experts,
                    request,
                    B::parallel_size(parallel),
                    context,
                )
            },
        )?;
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
    let shared = expert_bank_spec(args, cache_layer)?.with_group_geometry(
        args.text_config.n_shared_experts,
        local.moe_intermediate_size(),
    )?;
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
pub enum FeedForward<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> {
    /// Dense SwiGLU branch.
    Dense(DenseMlp<B>),
    /// Routed plus shared expert branch.
    Sparse(SparseMlp<B>),
}

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> FeedForward<B> {
    fn forward_with_provider_instrumented<P>(
        &mut self,
        hidden: &B::Tensor,
        layer: usize,
        shared_layer: usize,
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
            Self::Sparse(sparse) => sparse.forward_with_provider_instrumented(
                hidden,
                layer,
                shared_layer,
                pass,
                provider,
                context,
                instrumentation,
            ),
        }
    }

    fn forward_parallel_with_provider_instrumented<P>(
        &mut self,
        hidden: &B::Tensor,
        layer: usize,
        shared_layer: usize,
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
            Self::Sparse(sparse) => sparse.forward_parallel_with_provider_instrumented(
                hidden,
                layer,
                shared_layer,
                pass,
                provider,
                parallel,
                context,
                instrumentation,
            ),
        }
    }
}

/// One ordinary Inkling decoder layer.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct DecoderLayer<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> {
    #[parameter(skip, metadata)]
    layer: usize,
    #[parameter(skip, metadata)]
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

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> DecoderLayer<B> {
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
            None,
        )
    }

    /// Builds one layer under an explicit architecture-owned block root.
    pub fn new_at(
        args: &TextArgs,
        policy: LayerPolicy,
        block_root: &str,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        Self::new_at_with_expert_layers(args, policy, block_root, 0, 0, context, None)
    }

    pub(crate) fn new_with_expert_realization(
        args: &TextArgs,
        layer: usize,
        realization: crate::inkling::ExpertBankRealization,
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
            Some(realization),
        )
    }

    fn new_at_with_expert_layers(
        args: &TextArgs,
        policy: LayerPolicy,
        block_root: &str,
        layer: usize,
        shared_expert_layer: usize,
        context: &<B::Tensor as Tensor>::Context,
        realization: Option<crate::inkling::ExpertBankRealization>,
    ) -> Result<Self, Error> {
        crate::decoder::construction_specs::require_source_compiler::<B>(context)?;
        DecoderLayerSpec::new(args, policy, block_root, layer, shared_expert_layer, realization)?.instantiate::<B>(context)
    }

    /// Runs the canonical layer with all four bounded convolution histories.
    pub fn forward<C: AuxiliaryConvolutionState<B::Tensor>>(
        &mut self,
        hidden: &B::Tensor,
        state: Option<&mut C>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.forward_with_provider_instrumented(
            hidden,
            state,
            ExpertPass::Prefill,
            &mut eredu_runtime::ResidentExpertProvider,
            context,
            &mut ComponentInstrumentation::disabled(),
        )
    }

    /// Runs the canonical equations through the runtime-owned expert provider.
    #[allow(clippy::too_many_arguments)]
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
        self.forward_with_provider_instrumented(
            hidden,
            state,
            pass,
            provider,
            context,
            &mut ComponentInstrumentation::disabled(),
        )
    }

    /// Runs the canonical equations through the runtime-owned expert provider.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_with_provider_instrumented<C, P>(
        &mut self,
        hidden: &B::Tensor,
        state: Option<&mut C>,
        pass: ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error>
    where
        C: AuxiliaryConvolutionState<B::Tensor>,
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        let layer = self.layer;
        let shared = self.shared_expert_layer;
        self.forward_with_operators(
            hidden,
            state,
            None,
            context,
            instrumentation,
            |feed_forward, normalized, instrumentation| {
                feed_forward.forward_with_provider_instrumented(
                    normalized,
                    layer,
                    shared,
                    pass,
                    provider,
                    context,
                    instrumentation,
                )
            },
        )
    }

    /// Runs the canonical layer with all four bounded convolution histories.
    pub fn forward_parallel<C: AuxiliaryConvolutionState<B::Tensor>>(
        &mut self,
        hidden: &B::Tensor,
        state: Option<&mut C>,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        B: eredu_nn::TensorParallelGroupedNeuralBackend,
    {
        self.forward_parallel_with_provider_instrumented(
            hidden,
            state,
            ExpertPass::Prefill,
            &mut eredu_runtime::ResidentExpertProvider,
            parallel,
            context,
            &mut ComponentInstrumentation::disabled(),
        )
    }

    /// Runs the canonical equations through the runtime-owned expert provider.
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
        P: TensorParallelRoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        self.forward_parallel_with_provider_instrumented(
            hidden,
            state,
            pass,
            provider,
            parallel,
            context,
            &mut ComponentInstrumentation::disabled(),
        )
    }

    /// Runs the canonical equations through the runtime-owned expert provider.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_parallel_with_provider_instrumented<C, P>(
        &mut self,
        hidden: &B::Tensor,
        state: Option<&mut C>,
        pass: ExpertPass,
        provider: &mut P,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error>
    where
        C: AuxiliaryConvolutionState<B::Tensor>,
        P: TensorParallelRoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        let layer = self.layer;
        let shared = self.shared_expert_layer;
        self.forward_with_operators(
            hidden,
            state,
            Some(parallel),
            context,
            instrumentation,
            |feed_forward, normalized, instrumentation| {
                feed_forward.forward_parallel_with_provider_instrumented(
                    normalized,
                    layer,
                    shared,
                    pass,
                    provider,
                    parallel,
                    context,
                    instrumentation,
                )
            },
        )
    }

    fn forward_with_operators<C: AuxiliaryConvolutionState<B::Tensor>>(
        &mut self,
        hidden: &B::Tensor,
        mut state: Option<&mut C>,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
        feed_forward: impl FnOnce(
            &mut FeedForward<B>,
            &B::Tensor,
            &mut ComponentInstrumentation<'_, B::Tensor>,
        ) -> Result<B::Tensor, Error>,
    ) -> Result<B::Tensor, Error> {
        let normalized =
            instrumentation.apply("attention.input", self.input_norm.forward(hidden, context)?)?;
        let attention = self.attention.forward_instrumented(
            &normalized,
            state.as_deref_mut(),
            parallel,
            context,
            instrumentation,
        )?;
        let attention = instrumentation.apply("attention.write", attention)?;
        let attention = residual_convolution_with_state(
            &self.attention_convolution,
            &attention,
            state.as_deref_mut(),
            2,
            context,
        )?;
        let attention = instrumentation.apply("attention.contribution", attention)?;
        let hidden =
            instrumentation.apply("attention.residual", hidden.add(&attention, context)?)?;
        let normalized = instrumentation.apply(
            "feed_forward.input",
            self.post_attention_norm.forward(&hidden, context)?,
        )?;
        let feed_forward = feed_forward(&mut self.feed_forward, &normalized, instrumentation)?;
        let feed_forward = instrumentation.apply("feed_forward.write", feed_forward)?;
        let feed_forward = residual_convolution_with_state(
            &self.feed_forward_convolution,
            &feed_forward,
            state.as_deref_mut(),
            3,
            context,
        )?;
        let feed_forward = instrumentation.apply("feed_forward.contribution", feed_forward)?;
        instrumentation.apply("feed_forward.residual", hidden.add(&feed_forward, context)?)
    }
}

/// Inkling token embedding, ordinary decoder layers, and final norm.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct TextModel<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> {
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
    #[parameter(skip, metadata)]
    logits_scale: f32,
    #[parameter(skip, metadata)]
    output_vocabulary: i32,
}

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> TextModel<B> {
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
