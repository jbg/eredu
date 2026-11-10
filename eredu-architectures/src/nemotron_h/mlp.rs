//! Nemotron-H dense and routed ReLU-squared feed-forward operators.

use eredu_nn::{
    DeclareLogicalParameters, Error, GroupScoring, GroupSelectionOperator, GroupedNeuralBackend,
    GroupedRelu2Spec, LogicalParameterDeclaration, NeuralBackend, ParameterSpec, Parameterized,
    Tensor, TopKGroupSelectionSpec, TopKGroupSelectorSpec,
};
use eredu_runtime::{
    ParameterProvider, ResidentExpertProvider, RoutedExpertRequest, TensorParallelParameterProvider,
};

use crate::linear_format::standard_expert_projection;

use super::ModelArgs;

/// Complete dense or shared ReLU-squared branch at global or local width.
#[derive(Debug, Clone)]
pub(crate) struct DenseParameters {
    pub network: eredu_nn::UnaryFeedForwardSpec,
}

impl DenseParameters {
    pub fn new(args: &ModelArgs, prefix: &str, intermediate: i32) -> Result<Self, Error> {
        let up = super::attention::linear_spec(
            args,
            &format!("{prefix}.up_proj"),
            args.hidden_size,
            intermediate,
            args.mlp_bias,
        )?;
        let mut down = super::attention::linear_spec(
            args,
            &format!("{prefix}.down_proj"),
            intermediate,
            args.hidden_size,
            args.mlp_bias,
        )?;
        down.format = down.format.with_exact_zero_inputs();
        let spec = Self {
            network: eredu_nn::UnaryFeedForwardSpec::new(
                up,
                down,
                eredu_nn::UnaryActivation::ReluSquared,
            )?,
        };
        eredu_nn::validate_logical_parameter_topology(&spec.logical_parameters()?)
            .map_err(Error::backend_source)?;
        Ok(spec)
    }

    pub(crate) fn construct<B: NeuralBackend>(
        self,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<DenseMlp<B>, Error> {
        self.logical_parameters()?;
        Ok(DenseMlp {
            up_proj: B::linear(self.network.up, context)?,
            down_proj: B::linear(self.network.down, context)?,
            activation: self.network.activation,
        })
    }
}

impl DeclareLogicalParameters for DenseParameters {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        if self.network.up.input != self.network.down.output
            || self.network.activation != eredu_nn::UnaryActivation::ReluSquared
            || !self.network.down.format.exact_zero_inputs()
        {
            return Err(Error::backend(
                "incompatible Nemotron dense residual geometry or ReLU-squared policy",
            ));
        }
        self.network.logical_parameters()
    }
}

/// Dense up/ReLU²/down projection pair.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct DenseMlp<B: NeuralBackend> {
    /// Up projection.
    pub up_proj: B::Linear,
    /// Down projection.
    pub down_proj: B::Linear,
    #[parameter(skip)]
    activation: eredu_nn::UnaryActivation,
}

impl<B: NeuralBackend> DenseMlp<B> {
    /// Executes replicated dense computation.
    pub fn forward(
        &mut self,
        input: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.forward_instrumented(
            input,
            None,
            context,
            &mut crate::decoder::ComponentInstrumentation::disabled(),
        )
    }

    /// Executes genuine non-gated unit hooks before the down projection.
    pub fn forward_instrumented(
        &mut self,
        input: &B::Tensor,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut crate::decoder::ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error> {
        crate::decoder::unary::forward::<B>(
            input,
            &mut self.up_proj,
            &mut self.down_proj,
            self.activation,
            parallel,
            context,
            instrumentation,
        )
    }
}

impl<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> DenseMlp<B> {
    /// Executes a row-parallel down projection.
    pub fn forward_parallel(
        &mut self,
        input: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.forward_instrumented(
            input,
            Some(parallel),
            context,
            &mut crate::decoder::ComponentInstrumentation::disabled(),
        )
    }
}

/// Complete typed sparse-unit construction scope, independent of provider residency.
#[derive(Debug, Clone)]
pub(crate) struct SparseMoeSpec {
    pub(crate) selector: TopKGroupSelectorSpec,
    pub(crate) experts: GroupedRelu2Spec,
    pub(crate) shared: DenseParameters,
}

impl DeclareLogicalParameters for SparseMoeSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        let hidden = self.experts.hidden_dimensions();
        if self.selector.input_dimensions() != hidden
            || self.shared.network.up.input != hidden
            || self.shared.network.down.output != hidden
            || self.experts.group_count() > self.selector.selection().group_count()
        {
            return Err(Error::backend(
                "incompatible Nemotron routed/shared geometry",
            ));
        }
        let mut parameters = self.selector.logical_parameters()?;
        parameters.extend(self.experts.logical_parameters()?);
        parameters.extend(self.shared.logical_parameters()?);
        eredu_nn::validate_logical_parameter_topology(&parameters)
            .map_err(Error::backend_source)?;
        Ok(parameters)
    }
}

impl SparseMoeSpec {
    /// Canonical target or appended prediction ownership, independent of invocation state.
    pub(crate) fn for_layer(args: &ModelArgs, identity: usize) -> Result<Self, Error> {
        args.validate().map_err(Error::backend_source)?;
        let target = usize::try_from(args.num_hidden_layers).map_err(Error::backend)?;
        let (policy, prefix) = if identity < target {
            let policy = args
                .layer_schedule
                .get(identity)
                .copied()
                .ok_or_else(|| Error::backend(format!("Nemotron-H has no layer {identity}")))?;
            (policy, format!("model.layers.{identity}.moe"))
        } else {
            let physical = identity - target;
            let policies = args.mtp_policies().map_err(Error::backend)?;
            let policy = policies.get(physical).copied().ok_or_else(|| {
                Error::backend(format!("Nemotron-H has no appended MTP layer {physical}"))
            })?;
            (policy, format!("model.mtp.layers.{physical}.mixer"))
        };
        if policy != super::LayerPolicy::SparseMoe {
            return Err(Error::backend(format!(
                "Nemotron-H layer {identity} is not a sparse expert unit"
            )));
        }
        Self::at(args, &prefix)
    }

    /// Complete slots beneath an explicit architecture-owned source or construction root.
    pub(crate) fn at(args: &ModelArgs, prefix: &str) -> Result<Self, Error> {
        let spec = Self {
            selector: selector_spec(args, prefix)?,
            experts: expert_bank_spec_at(args, &format!("{prefix}.experts"))?,
            shared: DenseParameters::new(
                args,
                &format!("{prefix}.shared_experts"),
                args.moe_shared_expert_intermediate_size,
            )?,
        };
        spec.logical_parameters()?;
        Ok(spec)
    }

    /// Local routed and shared channels retain the global selector and exact parameter slots.
    pub(crate) fn with_local_geometry(
        mut self,
        experts: GroupedRelu2Spec,
        shared_intermediate: i32,
    ) -> Result<Self, Error> {
        if experts.intermediate_dimensions() > self.experts.intermediate_dimensions()
            || shared_intermediate <= 0
            || shared_intermediate > self.shared.network.up.output
        {
            return Err(Error::backend(
                "local Nemotron expert widths exceed their declarations",
            ));
        }
        self.experts = experts;
        self.shared.network.up.output = shared_intermediate;
        self.shared.network.down.input = shared_intermediate;
        self.logical_parameters()?;
        Ok(self)
    }
}

/// Grouped sigmoid selection: routing, packed ReLU² experts, and one shared expert.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct SparseMoe<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> {
    #[parameter(skip)]
    layer: usize,
    #[parameter(skip)]
    resident_unit_coordinates: Option<(eredu_core::component::ComponentCoordinateMap, bool)>,
    /// Grouped correction-bias router.
    pub gate: B::Selector,
    /// Packed routed experts.
    pub experts: B::Relu2Groups,
    /// Always-executed shared expert.
    pub shared_experts: DenseMlp<B>,
}

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> SparseMoe<B> {
    pub(crate) fn from_spec(
        layer: usize,
        spec: SparseMoeSpec,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        spec.logical_parameters()?;
        let gate = B::top_k_group_selector(spec.selector, context)?;
        let experts = B::grouped_relu2(spec.experts, context)?;
        Ok(Self {
            layer,
            resident_unit_coordinates: None,
            gate,
            experts,
            shared_experts: spec.shared.construct::<B>(context)?,
        })
    }

    /// Executes resident routed and shared experts.
    pub fn forward(
        &mut self,
        input: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.forward_with_provider(
            input,
            eredu_runtime::ExpertPass::Prefill,
            context,
            &mut ResidentExpertProvider,
        )
    }

    /// Executes routed experts through one runtime-owned provider.
    pub fn forward_with_provider<P>(
        &mut self,
        input: &B::Tensor,
        pass: eredu_runtime::ExpertPass,
        context: &<B::Tensor as Tensor>::Context,
        provider: &mut P,
    ) -> Result<B::Tensor, Error>
    where
        P: ParameterProvider<B>,
        P::Error: std::fmt::Display,
    {
        let routes = self.gate.select(input, context)?;
        let routed = provider
            .forward_relu2_routed(
                &mut self.experts,
                RoutedExpertRequest {
                    unit_observer: None,
                    bank: eredu_runtime::RoutedBankId::new(0),
                    layer: self.layer,
                    input,
                    routes: &routes,
                    pass,
                },
                context,
            )
            .map_err(Error::backend_source)?;
        let routed = restore_routed_precision(routed, input, context)?;
        routed.add(&self.shared_experts.forward(input, context)?, context)
    }

    /// Executes routed and shared experts while emitting one normalized routing observation.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_observed_with_provider<O, P>(
        &mut self,
        path: &str,
        expert_count: i32,
        input: &B::Tensor,
        pass: eredu_runtime::ExpertPass,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
        provider: &mut P,
    ) -> Result<B::Tensor, Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
        P: ParameterProvider<B>,
        P::Error: std::fmt::Display,
    {
        let routes = eredu_runtime::select_routes_with_observer(
            &mut self.gate,
            input,
            context,
            path,
            observer,
        )?;
        let routed = eredu_runtime::with_routed_unit_observer(
            observer,
            path,
            RoutedExpertRequest {
                unit_observer: None,
                bank: eredu_runtime::RoutedBankId::new(0),
                layer: self.layer,
                input,
                routes: &routes,
                pass,
            },
            |request| {
                eredu_runtime::with_resident_unit_coordinates(
                    self.resident_unit_coordinates.as_ref(),
                    request,
                    |request| provider.forward_relu2_routed(&mut self.experts, request, context),
                )
            },
        )
        .map_err(eredu_runtime::ObservedExpertProviderError::into_neural_error)?;
        let routed = restore_routed_precision(routed, input, context)?;
        let shared = self.forward_shared_observed(path, input, None, context, observer)?;
        let combined = routed.add(&shared, context)?;
        observer.observe_routing(eredu_runtime::RoutingObservation {
            path,
            selected_experts: routes.group_indices(),
            selected_scores: routes.selected_scores(),
            coefficients: routes.coefficients(),
            routed_output: &routed,
            local_routed_output: None,
            reduced_routed_output: None,
            shared_output: Some(&shared),
            combined_output: Some(&combined),
            expert_count,
        })?;
        eredu_runtime::observe_and_intervene(observer, &format!("{path}.output"), &combined)
    }

    /// Observes actual TP expert units and the reduced routed/shared result.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_parallel_observed_with_provider<O, P>(
        &mut self,
        path: &str,
        expert_count: i32,
        input: &B::Tensor,
        pass: eredu_runtime::ExpertPass,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
        provider: &mut P,
    ) -> Result<B::Tensor, Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
        P: TensorParallelParameterProvider<B>,
        P::Error: std::fmt::Display,
    {
        let routes = eredu_runtime::select_routes_with_observer(
            &mut self.gate,
            input,
            context,
            path,
            observer,
        )?;
        let routed = eredu_runtime::with_routed_unit_observer(
            observer,
            path,
            RoutedExpertRequest {
                unit_observer: None,
                bank: eredu_runtime::RoutedBankId::new(0),
                layer: self.layer,
                input,
                routes: &routes,
                pass,
            },
            |request| {
                eredu_runtime::with_resident_unit_coordinates(
                    self.resident_unit_coordinates.as_ref(),
                    request,
                    |request| {
                        provider.forward_relu2_routed_tensor_parallel(
                            &mut self.experts,
                            request,
                            B::parallel_size(parallel),
                            context,
                        )
                    },
                )
            },
        )
        .map_err(eredu_runtime::ObservedExpertProviderError::into_neural_error)?;
        let routed =
            eredu_runtime::reduce_routed_expert_tensor_parallel::<B>(routed, parallel, context)?;
        let routed = restore_routed_precision(routed, input, context)?;
        let shared =
            self.forward_shared_observed(path, input, Some(parallel), context, observer)?;
        let combined = routed.add(&shared, context)?;
        observer.observe_routing(eredu_runtime::RoutingObservation {
            path,
            selected_experts: routes.group_indices(),
            selected_scores: routes.selected_scores(),
            coefficients: routes.coefficients(),
            routed_output: &routed,
            local_routed_output: None,
            reduced_routed_output: Some(&routed),
            shared_output: Some(&shared),
            combined_output: Some(&combined),
            expert_count,
        })?;
        eredu_runtime::observe_and_intervene(observer, &format!("{path}.output"), &combined)
    }

    pub(crate) fn bind_resident_unit_coordinates(
        &mut self,
        coordinates: eredu_core::component::ComponentCoordinateMap,
        partitioned: bool,
    ) {
        self.resident_unit_coordinates = Some((coordinates, partitioned));
    }

    // The block's routing seam is a sibling of its always-executed shared branch.
    // The shared unary driver uses the public routing path for the same logical
    // invocation, including appended MTP units.
    fn forward_shared_observed<O>(
        &mut self,
        routing_path: &str,
        input: &B::Tensor,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        let unit_path = routing_path
            .strip_suffix(".routing")
            .unwrap_or(routing_path);
        let path = format!("{unit_path}.shared");
        let mut borrowed = eredu_runtime::BorrowedActivationObserver(observer);
        let mut instrumentation =
            crate::decoder::ComponentInstrumentation::new(&path, &mut borrowed);
        let input = instrumentation.apply("feed_forward.input", input.clone())?;
        let output = self.shared_experts.forward_instrumented(
            &input,
            parallel,
            context,
            &mut instrumentation,
        )?;
        let output = instrumentation.apply("feed_forward.write", output)?;
        instrumentation.apply("feed_forward.output", output)
    }

    /// Executes TP shared projections while the provider owns routed experts.
    pub fn forward_parallel_with_provider<P>(
        &mut self,
        input: &B::Tensor,
        pass: eredu_runtime::ExpertPass,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        provider: &mut P,
    ) -> Result<B::Tensor, Error>
    where
        P: TensorParallelParameterProvider<B>,
        P::Error: std::fmt::Display,
    {
        let routes = self.gate.select(input, context)?;
        let routed = provider
            .forward_relu2_routed_tensor_parallel(
                &mut self.experts,
                RoutedExpertRequest {
                    unit_observer: None,
                    bank: eredu_runtime::RoutedBankId::new(0),
                    layer: self.layer,
                    input,
                    routes: &routes,
                    pass,
                },
                B::parallel_size(parallel),
                context,
            )
            .map_err(Error::backend_source)?;
        let routed =
            eredu_runtime::reduce_routed_expert_tensor_parallel::<B>(routed, parallel, context)?;
        let routed = restore_routed_precision(routed, input, context)?;
        routed.add(
            &self
                .shared_experts
                .forward_parallel(input, parallel, context)?,
            context,
        )
    }
}

// Router coefficients and their reduction may use F32. The complete routed
// contribution returns to the input precision before adding the shared branch.
fn restore_routed_precision<T: Tensor>(
    routed: T,
    input: &T,
    context: &T::Context,
) -> Result<T, Error> {
    let element = input
        .element_type()
        .ok_or_else(|| Error::backend("unknown Nemotron routed input precision"))?;
    if routed.element_type() == Some(element) {
        Ok(routed)
    } else {
        routed.cast_float(element, context)
    }
}

/// Declares the sigmoid router and correction-bias resources at one canonical path.
fn selector_spec(args: &ModelArgs, prefix: &str) -> Result<TopKGroupSelectorSpec, Error> {
    let gate_weight = format!("{prefix}.gate.weight");
    let routing = TopKGroupSelectionSpec::new(
        args.n_routed_experts,
        args.num_experts_per_tok,
        GroupScoring::Sigmoid,
        args.norm_topk_prob,
    )?
    .with_groups(args.n_group, args.topk_group)?
    .with_weight_policy(1e-20, args.routed_scaling_factor)?;
    TopKGroupSelectorSpec::new(
        args.hidden_size,
        ParameterSpec::trainable(&gate_weight).map_err(Error::backend)?,
        crate::linear_format::standard_linear_format(
            &gate_weight,
            args.weight_quantization_for(&gate_weight).into(),
        )?,
        routing,
    )?
    .with_correction_bias(
        ParameterSpec::trainable(format!("{prefix}.gate.e_score_correction_bias"))
            .map_err(Error::backend)?,
    )
}

fn expert_bank_spec_at(args: &ModelArgs, experts_prefix: &str) -> Result<GroupedRelu2Spec, Error> {
    let up_name = format!("{experts_prefix}.up_proj");
    let down_name = format!("{experts_prefix}.down_proj");
    GroupedRelu2Spec::new(
        args.n_routed_experts,
        args.hidden_size,
        args.moe_intermediate_size,
        standard_expert_projection(
            &up_name,
            None,
            args.weight_quantization_for(&up_name).into(),
            eredu_nn::LinearRowLayout::Contiguous,
        )?,
        standard_expert_projection(
            &down_name,
            None,
            args.weight_quantization_for(&down_name).into(),
            eredu_nn::LinearRowLayout::Contiguous,
        )?,
    )
}
