//! Shared/routed feed-forward execution with explicit construction policy.
/// Exact reusable packed/split expert recipe construction.
pub mod checkpoint;
use crate::decoder::{ComponentInstrumentation, DecoderProjectionOperator, Mlp};
use eredu_nn::{
    Error, GatedProductPolicy, GroupSelectionOperator, GroupedGatedProductSpec,
    GroupedNeuralBackend, LinearSpec, OutputGateActivation, Parameterized, Tensor,
    TopKGroupSelectorSpec,
};
use eredu_runtime::{
    ExpertPass, ParameterProvider, RoutedExpertRequest, TensorParallelParameterProvider,
};

/// Exact construction of a routed bank and its always-on shared contribution.
#[derive(Debug, Clone)]
pub struct SharedRoutedGatedProductSpec {
    /// Physical layer coordinate supplied to the residency provider.
    pub layer: usize,
    /// Architecture-declared bank identity within the layer.
    pub bank: eredu_runtime::RoutedBankId,
    /// Router parameters, formats and selection policy.
    pub router: TopKGroupSelectorSpec,
    /// Independently addressable expert bank.
    pub experts: GroupedGatedProductSpec,
    /// Shared gate, up and down projections, in that order.
    pub shared: [LinearSpec; 3],
    /// Shared feed-forward activation and clipping policy.
    pub shared_policy: GatedProductPolicy,
    /// Projection to one scalar gate per input row.
    pub shared_gate: LinearSpec,
    /// Shared output gate activation.
    pub gate_activation: OutputGateActivation,
}

impl SharedRoutedGatedProductSpec {
    /// Validates composition geometry before backend construction.
    pub fn validate(&self) -> Result<(), Error> {
        self.router.validate()?;
        self.experts.validate()?;
        self.shared_policy.validate()?;
        let input = self.experts.input_dimensions();
        let output = self.experts.output_dimensions();
        let [gate, up, down] = &self.shared;
        if self.router.input_dimensions() != input
            || gate.input != input
            || up.input != input
            || gate.output <= 0
            || gate.output != up.output
            || down.input != gate.output
            || down.output != output
            || self.shared_gate.input != input
            || self.shared_gate.output != 1
        {
            return Err(Error::backend(
                "shared/routed feed-forward geometry mismatch",
            ));
        }
        for projection in [gate, up, down, &self.shared_gate] {
            projection.format.validate_for_weight(&projection.weight)?;
        }
        Ok(())
    }
}

/// Routed experts plus an always-on, gated shared expert.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct SharedRoutedGatedProduct<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> {
    #[parameter(skip)]
    layer: usize,
    #[parameter(skip)]
    bank: eredu_runtime::RoutedBankId,
    #[parameter(skip)]
    gate_activation: OutputGateActivation,
    #[parameter(skip)]
    resident_unit_coordinates: Option<(eredu_core::component::ComponentCoordinateMap, bool)>,
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
    /// Builds shared and routed operators from exact architecture-owned specs.
    pub fn new(
        spec: SharedRoutedGatedProductSpec,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        spec.validate()?;
        let [gate, up, down] = spec.shared;
        Ok(Self {
            layer: spec.layer,
            bank: spec.bank,
            gate_activation: spec.gate_activation,
            resident_unit_coordinates: None,
            router: B::top_k_group_selector(spec.router, context)?,
            experts: B::grouped_gated_product(spec.experts, context)?,
            shared_expert: Mlp::from_parts(
                B::linear(gate, context)?,
                B::linear(up, context)?,
                B::linear(down, context)?,
                Some(spec.shared_policy),
            ),
            shared_expert_gate: B::linear(spec.shared_gate, context)?,
        })
    }

    /// Retains global scalar coordinates for a prepared resident prediction bank.
    pub(crate) fn bind_resident_unit_coordinates(
        &mut self,
        coordinates: eredu_core::component::ComponentCoordinateMap,
        partitioned: bool,
    ) {
        self.resident_unit_coordinates = Some((coordinates, partitioned));
    }

    /// Executes one invocation through the selected expert residency provider.
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
        let routes = self.router.select(input, context)?;
        let routed = provider
            .forward_grouped(
                &mut self.experts,
                RoutedExpertRequest {
                    unit_observer: None,
                    bank: self.bank,
                    layer: self.layer,
                    input,
                    routes: &routes,
                    pass: pass(input),
                },
                context,
            )
            .map_err(Error::backend_source)?;
        let shared =
            self.forward_shared(input, context, &mut ComponentInstrumentation::disabled())?;
        routed.add(&shared, context)
    }

    /// Executes shared/routed feed-forward work with pre-dispatch routing controls
    /// and separately attributable shared contributions.
    pub fn forward_observed_with_provider<P, O>(
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
        let point = point
            .bank(self.bank)
            .ok_or_else(|| Error::backend("missing feed-forward routing observation"))?;
        let routes = eredu_runtime::select_routes_with_observer(
            &mut self.router,
            input,
            context,
            point.path(),
            observer,
        )?;
        let routed = eredu_runtime::with_routed_unit_observer(
            observer,
            point.path(),
            RoutedExpertRequest {
                unit_observer: None,
                bank: self.bank,
                layer: self.layer,
                input,
                routes: &routes,
                pass: pass(input),
            },
            |request| {
                eredu_runtime::with_resident_unit_coordinates(
                    self.resident_unit_coordinates.as_ref(),
                    request,
                    |request| provider.forward_grouped(&mut self.experts, request, context),
                )
            },
        )
        .map_err(eredu_runtime::ObservedExpertProviderError::into_neural_error)?;
        let shared = {
            let path = format!("{}.shared_expert", point.path());
            let mut borrowed = eredu_runtime::BorrowedActivationObserver(observer);
            self.forward_shared(
                input,
                context,
                &mut ComponentInstrumentation::new(&path, &mut borrowed),
            )?
        };
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

    /// Executes local expert work while retaining the shared contribution for reduction.
    pub fn forward_tensor_parallel_with_provider<P>(
        &mut self,
        input: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        provider: &mut P,
    ) -> Result<eredu_runtime::RoutedExpertTensorParallelOutput<B::Tensor>, Error>
    where
        P: TensorParallelParameterProvider<B>,
        P::Error: std::fmt::Display,
    {
        let routes = self.router.select(input, context)?;
        let routed = provider
            .forward_grouped_tensor_parallel(
                &mut self.experts,
                RoutedExpertRequest {
                    unit_observer: None,
                    bank: self.bank,
                    layer: self.layer,
                    input,
                    routes: &routes,
                    pass: pass(input),
                },
                B::parallel_size(parallel),
                context,
            )
            .map_err(Error::backend_source)?;
        let shared =
            self.forward_shared(input, context, &mut ComponentInstrumentation::disabled())?;
        Self::combine_tensor_parallel(routed, shared, parallel, context)
    }

    fn combine_tensor_parallel(
        routed: eredu_runtime::RoutedExpertTensorParallelOutput<B::Tensor>,
        shared: B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<eredu_runtime::RoutedExpertTensorParallelOutput<B::Tensor>, Error> {
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

    /// Executes partitioned shared/routed work with ordinary observation hooks.
    pub fn forward_tensor_parallel_observed_with_provider<P, O>(
        &mut self,
        points: eredu_runtime::RoutedObservationPoints,
        input: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        provider: &mut P,
        observer: &mut O,
    ) -> Result<B::Tensor, Error>
    where
        P: TensorParallelParameterProvider<B>,
        P::Error: std::fmt::Display,
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        let point = points
            .bank(self.bank)
            .ok_or_else(|| Error::backend("missing feed-forward routing observation"))?;
        let routes = eredu_runtime::select_routes_with_observer(
            &mut self.router,
            input,
            context,
            point.path(),
            observer,
        )?;
        let routed = eredu_runtime::with_routed_unit_observer(
            observer,
            point.path(),
            RoutedExpertRequest {
                unit_observer: None,
                bank: self.bank,
                layer: self.layer,
                input,
                routes: &routes,
                pass: pass(input),
            },
            |request| {
                eredu_runtime::with_resident_unit_coordinates(
                    self.resident_unit_coordinates.as_ref(),
                    request,
                    |request| {
                        provider.forward_grouped_tensor_parallel(
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
        let shared = {
            let path = format!("{}.shared_expert", point.path());
            let mut borrowed = eredu_runtime::BorrowedActivationObserver(observer);
            self.forward_shared(
                input,
                context,
                &mut ComponentInstrumentation::new(&path, &mut borrowed),
            )?
        };
        let combined = eredu_runtime::reduce_routed_expert_tensor_parallel::<B>(
            Self::combine_tensor_parallel(routed, shared, parallel, context)?,
            parallel,
            context,
        )?;
        eredu_runtime::observe_and_intervene(
            observer,
            &format!("{}.output", point.path()),
            &combined,
        )
    }

    fn forward_shared(
        &mut self,
        input: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error> {
        let shared =
            self.shared_expert
                .forward_feed_forward_observed(input, context, instrumentation)?;
        let shared = instrumentation.apply("feed_forward.write", shared)?;
        instrumentation.observe("gate.input", input)?;
        let gate = instrumentation.project::<B>(
            "gate.projection_input",
            &mut self.shared_expert_gate,
            input,
            None,
            context,
        )?;
        let gate = match self.gate_activation {
            OutputGateActivation::Sigmoid => B::sigmoid(gate, context)?,
            OutputGateActivation::Silu => B::silu(gate, context)?,
        };
        let gate = instrumentation.apply("gate", gate)?;
        instrumentation.apply("feed_forward.output", shared.multiply(&gate, context)?)
    }
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
