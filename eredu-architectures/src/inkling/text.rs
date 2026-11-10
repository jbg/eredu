//! Backend-neutral Inkling decoder equations and bounded operator state.

use eredu_core::AttentionPolicy;
use eredu_nn::{
    AttentionCache, AttentionRequest, AuxiliaryConvolutionState, CausalDepthwiseConvolution,
    DeclareLogicalParameters, EmbeddingOperator, Error, GatedProductGroupLayout, GroupSelection,
    GroupedGatedProductSpec, GroupedNeuralBackend, JointGroupSelectionInput,
    JointGroupSelectionSpec, LinearOperator, LinearSpec, LogicalParameterDeclaration,
    LogicalParameterRole, NeuralBackend, NormalizationOperator, Parameter, ParameterSpec,
    Parameterized, RelativeAttentionInput, Tensor,
};
use eredu_runtime::{
    ExpertPass, ParameterProvider, RoutedExpertRequest, TensorParallelParameterProvider,
};

use crate::decoder::ComponentInstrumentation;
use crate::linear_format::standard_expert_projection;

use super::{FeedForwardPolicy, ModelArgs, TextArgs};

mod pinned;
mod specs;
pub(crate) use pinned::PinnedParameters;
pub(crate) use specs::{AttentionParameters, CommonParameters, SparseParameters};

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

impl<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> Attention<B> {
    /// Builds one retained target attention invocation.
    pub fn new(
        args: &ModelArgs,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        Self::from_parameters(args.common(layer)?.attention.clone(), context)
    }

    fn from_parameters(
        parameters: AttentionParameters,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        parameters.logical_parameters()?;
        let geometry = parameters.projection.geometry;
        Ok(Self {
            query_heads: geometry.query_heads(),
            key_value_heads: geometry.key_value_heads(),
            head_dimensions: geometry.key_dimensions(),
            relative_dimensions: parameters.relative_projection.shape[0],
            relative_extent: parameters.relative_projection.shape[1],
            policy: parameters.policy,
            log_scaling_floor: parameters.log_scaling_floor,
            log_scaling_alpha: parameters.log_scaling_alpha,
            query: B::linear(parameters.projection.query, context)?,
            key: B::linear(parameters.projection.key, context)?,
            value: B::linear(parameters.projection.value, context)?,
            relative: B::linear(parameters.relative, context)?,
            output: B::linear(parameters.projection.output, context)?,
            query_norm: B::normalization(parameters.query_norm, context)?,
            key_norm: B::normalization(parameters.key_norm, context)?,
            relative_projection: parameters.relative_projection.unloaded(context)?,
            key_convolution: CausalDepthwiseConvolution::new(parameters.key_convolution, context)?,
            value_convolution: CausalDepthwiseConvolution::new(
                parameters.value_convolution,
                context,
            )?,
        })
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

/// Dense branch parameters shared by target and embedded prediction construction.
#[derive(Debug, Clone)]
pub(crate) struct DenseParameters {
    pub product: eredu_nn::SplitGatedProductSpec,
    pub global_scale: ParameterSpec,
}
impl DenseParameters {
    pub(crate) const SCALE_SHAPE: [i32; 1] = [1];

    pub(in crate::inkling) fn new_at(args: &TextArgs, block_root: &str) -> Result<Self, Error> {
        let prefix = format!("{block_root}.dense");
        let intermediate = args.dense_intermediate_size();
        let linear = |field: &str, input, output| {
            let weight = format!("{prefix}.{field}.weight");
            Ok::<_, Error>(LinearSpec {
                input,
                output,
                weight: ParameterSpec::trainable(&weight).map_err(Error::backend)?,
                bias: None,
                format: crate::linear_format::standard_linear_format(
                    &weight,
                    args.linear_format_for(&weight),
                )?,
            })
        };
        let parameters = Self {
            product: eredu_nn::SplitGatedProductSpec::new(
                linear("gate_proj", args.hidden_size, intermediate)?,
                linear("up_proj", args.hidden_size, intermediate)?,
                linear("down_proj", intermediate, args.hidden_size)?,
                eredu_nn::GatedProductPolicy::ordinary_silu(),
            )?,
            global_scale: ParameterSpec::trainable(format!("{block_root}.dense_global_scale"))
                .map_err(Error::backend)?,
        };
        eredu_nn::validate_logical_parameter_topology(&parameters.logical_parameters()?)
            .map_err(Error::backend_source)?;
        Ok(parameters)
    }

    pub(in crate::inkling) fn project_formats(&mut self, args: &TextArgs) -> Result<(), Error> {
        for spec in [
            &mut self.product.gate,
            &mut self.product.up,
            &mut self.product.down,
        ] {
            let mut format = crate::linear_format::standard_linear_format(
                spec.weight.id.as_str(),
                args.linear_format_for(spec.weight.id.as_str()),
            )?
            .with_arithmetic(spec.format.arithmetic())?;
            if spec.format.exact_zero_inputs() {
                format = format.with_exact_zero_inputs();
            }
            spec.format = format;
        }
        eredu_nn::validate_logical_parameter_topology(&self.logical_parameters()?)
            .map_err(Error::backend_source)
    }
    fn localized(&self, width: i32) -> Result<Self, Error> {
        if width <= 0 || width > self.product.gate.output {
            return Err(Error::backend(
                "Inkling local dense width exceeds its source",
            ));
        }
        let mut parameters = self.clone();
        parameters.product.gate.output = width;
        parameters.product.up.output = width;
        parameters.product.down.input = width;
        eredu_nn::validate_logical_parameter_topology(&parameters.logical_parameters()?)
            .map_err(Error::backend_source)?;
        Ok(parameters)
    }
    pub(crate) fn scale_declaration(&self) -> Result<LogicalParameterDeclaration<'_>, Error> {
        LogicalParameterDeclaration::new(
            &self.global_scale,
            &Self::SCALE_SHAPE,
            LogicalParameterRole::ResidualScale,
        )
    }
}
impl DeclareLogicalParameters for DenseParameters {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        if self.product.gate.input != self.product.down.output {
            return Err(Error::backend(
                "Inkling dense branch changes residual width",
            ));
        }
        let mut parameters = self.product.logical_parameters()?;
        parameters.push(self.scale_declaration()?);
        Ok(parameters)
    }
}

/// Dense SwiGLU branch with its learned global scalar.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct DenseMlp<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> {
    #[parameter(skip)]
    policy: eredu_nn::GatedProductPolicy,
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
    pub(crate) fn from_parameters(
        parameters: DenseParameters,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        Ok(Self {
            policy: parameters.product.policy,
            gate: B::linear(parameters.product.gate, context)?,
            up: B::linear(parameters.product.up, context)?,
            down: B::linear(parameters.product.down, context)?,
            global_scale: Parameter::unloaded(
                parameters.global_scale,
                &DenseParameters::SCALE_SHAPE,
                context,
            )?,
        })
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
        let units = B::gated_product(gate, up, self.policy, context)?;
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
    #[parameter(skip)]
    selection: JointGroupSelectionSpec,
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

    fn from_parameters(
        parameters: SparseParameters,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        parameters.validate()?;
        let router = parameters.router;
        Ok(Self {
            selection: router.selection(),
            router_weight: Parameter::unloaded(
                router.weight().clone(),
                &router.weight_shape()?,
                context,
            )?,
            router_bias: Parameter::unloaded(
                router.correction_bias().clone(),
                &router.correction_bias_shape(),
                context,
            )?,
            global_scale: Parameter::unloaded(
                router.coefficient_scale().clone(),
                &router.coefficient_scale_shape(),
                context,
            )?,
            routed_experts: B::grouped_gated_product(parameters.routed, context)?,
            shared_experts: B::grouped_gated_product(parameters.shared, context)?,
        })
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
        P: ParameterProvider<B>,
        P::Error: std::fmt::Display,
    {
        let routes = B::joint_group_selection(
            JointGroupSelectionInput::new(
                hidden,
                self.router_weight.as_ref(),
                self.router_bias.as_ref(),
                self.global_scale.as_ref(),
                self.selection,
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
                layer,
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
        let shared_ids =
            Self::shared_group_indices(tokens, self.selection.always_on_groups(), context)?;
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
        P: TensorParallelParameterProvider<B>,
        P::Error: std::fmt::Display,
    {
        let routes = B::joint_group_selection(
            JointGroupSelectionInput::new(
                hidden,
                self.router_weight.as_ref(),
                self.router_bias.as_ref(),
                self.global_scale.as_ref(),
                self.selection,
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
                layer,
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
        let shared_ids =
            Self::shared_group_indices(tokens, self.selection.always_on_groups(), context)?;
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

/// Resolves a shared bank's cache identity independently of its decoder invocation.
pub(crate) fn shared_expert_cache_layer(args: &TextArgs, layer: usize) -> Result<usize, Error> {
    let layers = usize::try_from(args.num_hidden_layers).map_err(Error::backend)?;
    if layer >= layers {
        return Err(Error::backend(format!(
            "Inkling layer {layer} is outside {layers} declared layers"
        )));
    }
    layers
        .checked_add(layer)
        .ok_or_else(|| Error::backend("Inkling shared expert unit overflowed"))
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
            gate_up: standard_expert_projection(
                &gate_up,
                None,
                args.linear_format_for(&gate_up),
                eredu_nn::LinearRowLayout::equal_partitions(2)?,
            )?,
            down: standard_expert_projection(
                &down,
                None,
                args.linear_format_for(&down),
                eredu_nn::LinearRowLayout::Contiguous,
            )?,
        },
    )
}

/// One target invocation's exact feed-forward ownership, separate from execution state.
#[derive(Debug, Clone)]
pub(crate) enum FeedForwardParameters {
    Dense(DenseParameters),
    Sparse(SparseParameters),
}
impl FeedForwardParameters {
    pub(in crate::inkling) fn new(args: &TextArgs, layer: usize) -> Result<Self, Error> {
        let root = format!("model.layers.{layer}");
        match args
            .layer_policy(layer)
            .ok_or_else(|| Error::backend("Inkling feed-forward invocation is out of range"))?
            .feed_forward
        {
            FeedForwardPolicy::Dense => Ok(Self::Dense(DenseParameters::new_at(args, &root)?)),
            FeedForwardPolicy::SparseMoe => {
                Ok(Self::Sparse(SparseParameters::new_at(args, &root)?))
            }
        }
    }
    pub(in crate::inkling) fn project_formats(&mut self, args: &TextArgs) -> Result<(), Error> {
        match self {
            Self::Dense(p) => p.project_formats(args),
            Self::Sparse(p) => p.project_formats(args),
        }
    }
    pub(in crate::inkling) fn localized(&self, local: &TextArgs) -> Result<Self, Error> {
        Ok(match self {
            Self::Dense(p) => Self::Dense(p.localized(local.dense_intermediate_size())?),
            Self::Sparse(p) => Self::Sparse(p.localized(local.moe_intermediate_size())?),
        })
    }
}
impl DeclareLogicalParameters for FeedForwardParameters {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        match self {
            Self::Dense(p) => p.logical_parameters(),
            Self::Sparse(p) => p.logical_parameters(),
        }
    }
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
        P: ParameterProvider<B>,
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
        P: TensorParallelParameterProvider<B>,
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

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> DecoderLayer<B> {
    /// Builds one scheduled decoder layer from retained target scopes.
    pub fn new(
        args: &ModelArgs,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        Self::from_selection(args, &args.text_config, layer, None, context)
    }
    pub(crate) fn from_selection(
        args: &ModelArgs,
        local: &TextArgs,
        layer: usize,
        realization: Option<crate::inkling::ExpertBankRealization>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let parameters = args.local_common(layer, local)?;
        let mut feed_forward = args.local_feed_forward(layer, local)?;
        if let Some(realization) = realization {
            let FeedForwardParameters::Sparse(parameters) = &mut feed_forward else {
                return Err(Error::backend(
                    "Inkling dense invocation cannot bind expert banks",
                ));
            };
            *parameters = parameters
                .clone()
                .with_banks(realization.routed, realization.shared)?;
        }
        let shared_layer = shared_expert_cache_layer(&args.text_config, layer)?;
        Self::from_parameters(parameters, layer, shared_layer, context, |context| {
            Ok(match feed_forward {
                FeedForwardParameters::Dense(parameters) => {
                    FeedForward::Dense(DenseMlp::from_parameters(parameters, context)?)
                }
                FeedForwardParameters::Sparse(parameters) => {
                    FeedForward::Sparse(SparseMlp::from_parameters(parameters, context)?)
                }
            })
        })
    }

    pub(crate) fn from_parameters(
        parameters: CommonParameters,
        layer: usize,
        shared_expert_layer: usize,
        context: &<B::Tensor as Tensor>::Context,
        feed_forward: impl FnOnce(&<B::Tensor as Tensor>::Context) -> Result<FeedForward<B>, Error>,
    ) -> Result<Self, Error> {
        Ok(Self {
            layer,
            shared_expert_layer,
            input_norm: B::normalization(parameters.input_norm, context)?,
            attention: Attention::from_parameters(parameters.attention, context)?,
            attention_convolution: CausalDepthwiseConvolution::new(
                parameters.attention_convolution,
                context,
            )?,
            post_attention_norm: B::normalization(parameters.post_attention_norm, context)?,
            feed_forward: feed_forward(context)?,
            feed_forward_convolution: CausalDepthwiseConvolution::new(
                parameters.feed_forward_convolution,
                context,
            )?,
        })
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
        P: ParameterProvider<B>,
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
        P: ParameterProvider<B>,
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
        P: TensorParallelParameterProvider<B>,
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
        P: TensorParallelParameterProvider<B>,
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
        let mut temporary_history = None;
        let history = match state.as_deref_mut() {
            Some(state) => state.convolution_state(2)?,
            None => &mut temporary_history,
        };
        let attention =
            residual_convolution(&self.attention_convolution, &attention, history, context)?;
        let attention = instrumentation.apply("attention.contribution", attention)?;
        let hidden =
            instrumentation.apply("attention.residual", hidden.add(&attention, context)?)?;
        let normalized = instrumentation.apply(
            "feed_forward.input",
            self.post_attention_norm.forward(&hidden, context)?,
        )?;
        let feed_forward = feed_forward(&mut self.feed_forward, &normalized, instrumentation)?;
        let feed_forward = instrumentation.apply("feed_forward.write", feed_forward)?;
        let mut temporary_history = None;
        let history = match state {
            Some(state) => state.convolution_state(3)?,
            None => &mut temporary_history,
        };
        let feed_forward = residual_convolution(
            &self.feed_forward_convolution,
            &feed_forward,
            history,
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
    #[parameter(skip)]
    logits_scale: f32,
    #[parameter(skip)]
    output_vocabulary: i32,
}

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> TextModel<B> {
    /// Builds the complete neutral text model from retained source statics.
    pub fn new(args: &ModelArgs, context: &<B::Tensor as Tensor>::Context) -> Result<Self, Error> {
        let pinned = args.pinned();
        let text = &args.text_config;
        let common = crate::decoder::StaticModules::<B>::from_construction(
            pinned.ordinary.clone(),
            context,
        )?;
        Ok(Self {
            embeddings: common.embeddings,
            embedding_norm: B::normalization(pinned.embedding_norm.clone(), context)?,
            layers: (0..text.num_hidden_layers as usize)
                .map(|layer| DecoderLayer::new(args, layer, context))
                .collect::<Result<_, _>>()?,
            final_norm: common.boundary.norm,
            output: common.lm_head.expect("Inkling declares an untied readout"),
            logits_scale: text.logits_mup_width_multiplier,
            output_vocabulary: text.unpadded_vocab_size.unwrap_or(text.vocab_size),
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
