//! Typed routed/shared declarations and execution for DeepSeek V3 and V4.

use eredu_nn::{
    DeclareLogicalParameters, Error, GatedProductPolicy, GroupSelectionOperator,
    GroupedNeuralBackend, LinearOperator, Parameterized, Tensor,
};
use eredu_runtime::{
    observe_and_intervene, ActivationObserver, ExpertPass, ParameterProvider,
    ResidentExpertProvider, RoutedExpertRequest, RoutingObservation,
    TensorParallelParameterProvider,
};

/// Typed parameters and execution policy shared by cold consumers and construction.
#[derive(Debug, Clone)]
pub(crate) struct RoutedSpec {
    /// Learned selector and provider-owned bank, possibly localized by retained placement.
    pub parameters: eredu_nn::RoutedGatedProductSpec,
    /// Always-on branch with independent geometry, encoding and gated policy.
    pub shared: eredu_nn::SplitGatedProductSpec,
}
impl DeclareLogicalParameters for RoutedSpec {
    fn logical_parameters(&self) -> Result<Vec<eredu_nn::LogicalParameterDeclaration<'_>>, Error> {
        if self.shared.gate.input != self.parameters.experts.input_dimensions()
            || self.shared.down.output != self.parameters.experts.output_dimensions()
        {
            return Err(Error::backend(
                "incompatible DeepSeek routed/shared branch geometry",
            ));
        }
        let mut parameters = self.parameters.logical_parameters()?;
        parameters.extend(self.shared.logical_parameters()?);
        eredu_nn::validate_logical_parameter_topology(&parameters)
            .map_err(Error::backend_source)?;
        Ok(parameters)
    }
}

/// Learned routes or caller-selected token/hash routes.
#[allow(missing_docs)]
pub enum RouteSource<'a, T> {
    Learned,
    Selected(&'a T),
}

/// One shared implementation of routed and always-on shared experts.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
#[allow(missing_docs)]
pub struct RoutedPlusShared<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> {
    #[parameter(skip)]
    layer: usize,
    #[parameter(skip)]
    expert_count: i32,
    pub router: B::Selector,
    pub experts: B::GatedProductGroups,
    shared_gate: B::Linear,
    shared_up: B::Linear,
    shared_down: B::Linear,
    #[parameter(skip)]
    shared_policy: GatedProductPolicy,
    #[parameter(skip)]
    resident_unit_coordinates: Option<(eredu_core::component::ComponentCoordinateMap, bool)>,
}

#[allow(missing_docs)]
impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> RoutedPlusShared<B> {
    pub(crate) fn from_spec(
        layer: usize,
        spec: RoutedSpec,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        spec.logical_parameters()?;
        let expert_count = spec.parameters.selector.selection().group_count();
        let router = B::top_k_group_selector(spec.parameters.selector, context)?;
        let experts = B::grouped_gated_product(spec.parameters.experts, context)?;
        Ok(Self {
            layer,
            expert_count,
            router,
            experts,
            shared_gate: B::linear(spec.shared.gate, context)?,
            shared_up: B::linear(spec.shared.up, context)?,
            shared_down: B::linear(spec.shared.down, context)?,
            shared_policy: spec.shared.policy,
            resident_unit_coordinates: None,
        })
    }

    /// Binds the scalar order compiled for an independently resident prediction bank.
    pub(crate) fn bind_resident_unit_coordinates(
        &mut self,
        coordinates: eredu_core::component::ComponentCoordinateMap,
        partitioned: bool,
    ) {
        self.resident_unit_coordinates = Some((coordinates, partitioned));
    }

    pub fn forward(
        &mut self,
        input: &B::Tensor,
        source: RouteSource<'_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let mut provider = ResidentExpertProvider;
        self.forward_with_provider(input, source, ExpertPass::Decode, &mut provider, context)
    }

    pub fn forward_with_provider<P: ParameterProvider<B>>(
        &mut self,
        input: &B::Tensor,
        source: RouteSource<'_, B::Tensor>,
        pass: ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        P::Error: std::fmt::Display,
    {
        let mut observer = eredu_runtime::NoopObserver;
        self.forward_with_provider_observed(
            "routed_feed_forward",
            input,
            source,
            pass,
            provider,
            context,
            &mut observer,
        )
    }

    /// Executes routed/shared TP work with one reduction and literal post-bias.
    pub fn forward_tensor_parallel_with_provider<P, F>(
        &mut self,
        input: &B::Tensor,
        source: RouteSource<'_, B::Tensor>,
        pass: ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
        mut reduce: F,
    ) -> Result<B::Tensor, Error>
    where
        P: TensorParallelParameterProvider<B>,
        P::Error: std::fmt::Display,
        F: FnMut(B::Tensor, &<B::Tensor as Tensor>::Context) -> Result<B::Tensor, Error>,
    {
        let routes = match source {
            RouteSource::Learned => self.router.select(input, context)?,
            RouteSource::Selected(ids) => self.router.select_indices(input, ids, context)?,
        };
        let routed = provider
            .forward_grouped_tensor_parallel(
                &mut self.experts,
                RoutedExpertRequest {
                    unit_observer: None,
                    bank: eredu_runtime::RoutedBankId::new(0),
                    layer: self.layer,
                    input,
                    routes: &routes,
                    pass,
                },
                1,
                context,
            )
            .map_err(Error::backend_source)?;
        let shared = self.forward_shared(
            input,
            context,
            &mut crate::decoder::ComponentInstrumentation::disabled(),
        )?;
        Self::combine_tensor_parallel(routed, shared, context, &mut reduce)
    }

    // Keep the ordinary fused reduction and literal post-reduction bias identical
    // for observed and unobserved execution.
    fn combine_tensor_parallel<F>(
        routed: eredu_runtime::RoutedExpertTensorParallelOutput<B::Tensor>,
        shared: B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
        reduce: &mut F,
    ) -> Result<B::Tensor, Error>
    where
        F: FnMut(B::Tensor, &<B::Tensor as Tensor>::Context) -> Result<B::Tensor, Error>,
    {
        match routed {
            eredu_runtime::RoutedExpertTensorParallelOutput::Complete(routed) => {
                routed.add(&reduce(shared, context)?, context)
            }
            eredu_runtime::RoutedExpertTensorParallelOutput::Partial(routed) => {
                let (reducible, post_reduce) = routed.into_parts();
                let reduced = reduce(reducible.add(&shared, context)?, context)?;
                match post_reduce {
                    Some(bias) => reduced.add(&bias, context),
                    None => Ok(reduced),
                }
            }
        }
    }

    /// Observes actual routed units and shared scalar/write terms before the
    /// ordinary fused TP reduction. Shared write/output callbacks are additive
    /// terms, as declared by the architecture placement. The combined output is
    /// complete. No separately reduced routed/shared diagnostic tensor is made.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_tensor_parallel_with_provider_observed<P, O, F>(
        &mut self,
        path: &str,
        input: &B::Tensor,
        source: RouteSource<'_, B::Tensor>,
        pass: ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
        reduce: F,
    ) -> Result<B::Tensor, Error>
    where
        P: TensorParallelParameterProvider<B>,
        P::Error: std::fmt::Display,
        O: ActivationObserver<B::Tensor, Error> + ?Sized,
        F: FnMut(B::Tensor, &<B::Tensor as Tensor>::Context) -> Result<B::Tensor, Error>,
    {
        self.forward_tensor_parallel_observed(
            path,
            input,
            source,
            pass,
            context,
            observer,
            reduce,
            |experts, request| {
                provider.forward_grouped_tensor_parallel(experts, request, 1, context)
            },
        )
    }

    /// Observes the ordinary resident-bank TP equation without requiring the
    /// separate provider TP mechanism. The ordinary local expert output joins
    /// the shared term before the same single reduction used by that path.
    pub(crate) fn forward_tensor_parallel_resident_observed<O, F>(
        &mut self,
        path: &str,
        input: &B::Tensor,
        source: RouteSource<'_, B::Tensor>,
        pass: ExpertPass,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
        reduce: F,
    ) -> Result<B::Tensor, Error>
    where
        O: ActivationObserver<B::Tensor, Error> + ?Sized,
        F: FnMut(B::Tensor, &<B::Tensor as Tensor>::Context) -> Result<B::Tensor, Error>,
    {
        self.forward_tensor_parallel_observed(
            path,
            input,
            source,
            pass,
            context,
            observer,
            reduce,
            |experts, request| {
                <ResidentExpertProvider as ParameterProvider<B>>::forward_grouped(
                    &mut ResidentExpertProvider,
                    experts,
                    request,
                    context,
                )
                .map(|value| {
                    eredu_runtime::RoutedExpertTensorParallelOutput::Partial(
                        eredu_nn::TensorParallelGroupedOutput::new(value, None),
                    )
                })
            },
        )
    }

    fn forward_tensor_parallel_observed<O, F, E>(
        &mut self,
        path: &str,
        input: &B::Tensor,
        source: RouteSource<'_, B::Tensor>,
        pass: ExpertPass,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
        mut reduce: F,
        execute: impl FnOnce(
            &mut B::GatedProductGroups,
            RoutedExpertRequest<'_, '_, B::Tensor>,
        )
            -> Result<eredu_runtime::RoutedExpertTensorParallelOutput<B::Tensor>, E>,
    ) -> Result<B::Tensor, Error>
    where
        O: ActivationObserver<B::Tensor, Error> + ?Sized,
        F: FnMut(B::Tensor, &<B::Tensor as Tensor>::Context) -> Result<B::Tensor, Error>,
        E: std::error::Error + Send + Sync + 'static,
    {
        let routes = match source {
            RouteSource::Learned => eredu_runtime::select_routes_with_observer(
                &mut self.router,
                input,
                context,
                path,
                observer,
            )?,
            RouteSource::Selected(ids) => self.router.select_indices(input, ids, context)?,
        };
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
                    |request| execute(&mut self.experts, request),
                )
            },
        )
        .map_err(eredu_runtime::ObservedExpertProviderError::into_neural_error)?;
        let shared = {
            let path = format!("{path}.shared");
            let mut borrowed = eredu_runtime::BorrowedActivationObserver(observer);
            let mut instrumentation =
                crate::decoder::ComponentInstrumentation::new(&path, &mut borrowed);
            let shared = self.forward_shared(input, context, &mut instrumentation)?;
            let shared = instrumentation.apply("write", shared)?;
            instrumentation.apply("output", shared)?
        };
        let combined = Self::combine_tensor_parallel(routed, shared, context, &mut reduce)?;
        observe_and_intervene(observer, &format!("{path}.output"), &combined)
    }

    /// Executes routed/shared experts with normalized route observation and a
    /// stable intervention point on their combined contribution.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_with_provider_observed<P, O>(
        &mut self,
        path: &str,
        input: &B::Tensor,
        source: RouteSource<'_, B::Tensor>,
        pass: ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Error>
    where
        P: ParameterProvider<B>,
        P::Error: std::fmt::Display,
        O: ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        let routes = match source {
            RouteSource::Learned => eredu_runtime::select_routes_with_observer(
                &mut self.router,
                input,
                context,
                path,
                observer,
            )?,
            RouteSource::Selected(ids) => self.router.select_indices(input, ids, context)?,
        };
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
                    |request| provider.forward_grouped(&mut self.experts, request, context),
                )
            },
        )
        .map_err(eredu_runtime::ObservedExpertProviderError::into_neural_error)?;
        let shared = {
            let path = format!("{path}.shared");
            let mut borrowed = eredu_runtime::BorrowedActivationObserver(observer);
            let mut instrumentation =
                crate::decoder::ComponentInstrumentation::new(&path, &mut borrowed);
            let shared = self.forward_shared(input, context, &mut instrumentation)?;
            let shared = instrumentation.apply("write", shared)?;
            instrumentation.apply("output", shared)?
        };
        let combined = routed.add(&shared, context)?;
        observer.observe_routing(RoutingObservation {
            path,
            selected_experts: routes.group_indices(),
            selected_scores: routes.selected_scores(),
            coefficients: routes.coefficients(),
            routed_output: &routed,
            local_routed_output: None,
            reduced_routed_output: None,
            shared_output: Some(&shared),
            combined_output: Some(&combined),
            expert_count: self.expert_count,
        })?;
        observe_and_intervene(observer, &format!("{path}.output"), &combined)
    }

    // A TP caller owns reduction of this write. The scalar and actual-input
    // seams are shared with local execution; no partial write is presented as
    // a complete residual contribution.
    fn forward_shared(
        &mut self,
        input: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut crate::decoder::ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error> {
        let input = instrumentation.apply("input", input.clone())?;
        let gate = self.shared_gate.forward(&input, context)?;
        let up = self.shared_up.forward(&input, context)?;
        let units = B::gated_product(gate, up, self.shared_policy, context)?;
        let units = instrumentation.apply("units", units)?;
        instrumentation.project::<B>("write_input", &mut self.shared_down, &units, None, context)
    }
}
