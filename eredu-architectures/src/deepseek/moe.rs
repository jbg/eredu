//! Shared routed-plus-shared SwiGLU policy for DeepSeek V3 and V4.

use eredu_checkpoint::LinearFormat;
use eredu_nn::{
    Error, GatedProductGroupLayout, GatedProductPolicy, GroupScoring, GroupSelectionOperator,
    GroupedGatedProductSpec, GroupedNeuralBackend, LinearOperator, LinearSpec, ParameterSpec,
    Parameterized, Tensor, TopKGroupSelectionSpec, TopKGroupSelectorSpec,
};
use eredu_runtime::{
    observe_and_intervene, ActivationObserver, ExpertPass, ResidentExpertProvider,
    RoutedExpertProvider, RoutedExpertRequest, RoutingObservation,
    TensorParallelRoutedExpertProvider,
};

use crate::linear_format::standard_expert_projection;

/// Complete family-neutral assembly policy for one DeepSeek MoE layer.
#[derive(Debug, Clone)]
#[allow(missing_docs)]
pub struct MoePolicy {
    pub layer: usize,
    pub hidden: i32,
    pub expert_count: i32,
    pub routes_per_token: i32,
    pub expert_width: i32,
    pub shared_width: i32,
    pub scoring: GroupScoring,
    pub normalize_routes: bool,
    pub normalization_epsilon: f32,
    pub routed_scaling: f32,
    pub expert_groups: i32,
    pub selected_groups: i32,
    pub router_weight: String,
    pub router_format: LinearFormat,
    pub correction_bias: Option<String>,
    pub expert_gate_up: String,
    pub expert_down: String,
    pub shared_gate: String,
    pub shared_up: String,
    pub shared_down: String,
    pub shared_gate_format: LinearFormat,
    pub shared_up_format: LinearFormat,
    pub shared_down_format: LinearFormat,
    pub expert_gate_up_format: LinearFormat,
    pub expert_down_format: LinearFormat,
    pub shared_limit: Option<GatedProductPolicy>,
    pub limit: Option<GatedProductPolicy>,
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
    shared_limit: Option<GatedProductPolicy>,
    #[parameter(skip)]
    resident_unit_coordinates: Option<(eredu_core::component::ComponentCoordinateMap, bool)>,
}

#[allow(missing_docs)]
impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> RoutedPlusShared<B> {
    pub fn new(
        policy: &MoePolicy,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let expert_spec = expert_bank_spec(policy)?;
        Self::new_with_expert_spec(policy, expert_spec, context)
    }

    /// Builds the routed block with an already-selected rank-local expert bank.
    pub(crate) fn new_with_expert_spec(
        policy: &MoePolicy,
        expert_spec: eredu_nn::GroupedGatedProductSpec,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let routing = TopKGroupSelectionSpec::new(
            policy.expert_count,
            policy.routes_per_token,
            policy.scoring,
            policy.normalize_routes,
        )?
        .with_groups(policy.expert_groups, policy.selected_groups)?
        .with_weight_policy(policy.normalization_epsilon, policy.routed_scaling)?;
        let mut selector = TopKGroupSelectorSpec::new(
            policy.hidden,
            parameter(&policy.router_weight)?,
            crate::linear_format::standard_linear_format(
                &policy.router_weight,
                policy.router_format,
            )?,
            routing,
        )?;
        if let Some(correction_bias) = policy
            .correction_bias
            .as_deref()
            .map(parameter)
            .transpose()?
        {
            selector = selector.with_correction_bias(correction_bias)?;
        }
        let router = B::top_k_group_selector(selector, context)?;
        let experts = B::grouped_gated_product(expert_spec, context)?;
        let shared = |weight: &str, input, output, format| {
            B::linear(
                LinearSpec {
                    input,
                    output,
                    weight: parameter(weight)?,
                    bias: None,
                    format: crate::linear_format::standard_linear_format(weight, format)?,
                },
                context,
            )
        };
        Ok(Self {
            layer: policy.layer,
            expert_count: policy.expert_count,
            router,
            experts,
            shared_gate: shared(
                &policy.shared_gate,
                policy.hidden,
                policy.shared_width,
                policy.shared_gate_format,
            )?,
            shared_up: shared(
                &policy.shared_up,
                policy.hidden,
                policy.shared_width,
                policy.shared_up_format,
            )?,
            shared_down: shared(
                &policy.shared_down,
                policy.shared_width,
                policy.hidden,
                policy.shared_down_format,
            )?,
            shared_limit: policy.shared_limit,
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

    pub fn forward_with_provider<P: RoutedExpertProvider<B>>(
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
        P: TensorParallelRoutedExpertProvider<B>,
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
        P: TensorParallelRoutedExpertProvider<B>,
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
                <ResidentExpertProvider as RoutedExpertProvider<B>>::forward_grouped(
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
        P: RoutedExpertProvider<B>,
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
        let units = B::gated_product(gate, up, self.shared_limit.unwrap_or_default(), context)?;
        let units = instrumentation.apply("units", units)?;
        instrumentation.project::<B>("write_input", &mut self.shared_down, &units, None, context)
    }
}

/// Returns the architecture-owned routed expert specification for one policy.
pub fn expert_bank_spec(policy: &MoePolicy) -> Result<GroupedGatedProductSpec, Error> {
    GroupedGatedProductSpec::new(
        policy.expert_count,
        policy.hidden,
        policy.expert_width,
        policy.hidden,
        policy.limit.unwrap_or_default(),
        GatedProductGroupLayout::Packed {
            gate_up: standard_expert_projection(
                &policy.expert_gate_up,
                None,
                policy.expert_gate_up_format,
            )?,
            down: standard_expert_projection(&policy.expert_down, None, policy.expert_down_format)?,
        },
    )
}

fn parameter(name: &str) -> Result<ParameterSpec, Error> {
    ParameterSpec::trainable(name).map_err(Error::backend)
}
