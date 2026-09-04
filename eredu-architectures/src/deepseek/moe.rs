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
                eredu_checkpoint::LinearFormat::Dense,
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
        })
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
                    layer: self.layer,
                    input,
                    routes: &routes,
                    pass,
                },
                1,
                context,
            )
            .map_err(Error::backend)?;
        let gate = self.shared_gate.forward(input, context)?;
        let up = self.shared_up.forward(input, context)?;
        let shared = B::gated_product(gate, up, self.shared_limit.unwrap_or_default(), context)?;
        let shared = self.shared_down.forward(&shared, context)?;
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
            RouteSource::Learned => self.router.select(input, context)?,
            RouteSource::Selected(ids) => self.router.select_indices(input, ids, context)?,
        };
        let routed = provider
            .forward_grouped(
                &mut self.experts,
                RoutedExpertRequest {
                    layer: self.layer,
                    input,
                    routes: &routes,
                    pass,
                },
                context,
            )
            .map_err(Error::backend)?;
        let gate = self.shared_gate.forward(input, context)?;
        let up = self.shared_up.forward(input, context)?;
        let shared = B::gated_product(gate, up, self.shared_limit.unwrap_or_default(), context)?;
        let shared = self.shared_down.forward(&shared, context)?;
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
