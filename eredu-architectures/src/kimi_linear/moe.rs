//! Kimi dense-prefix and grouped routed-expert feed-forward operators.

use eredu_nn::{
    DeclareLogicalParameters, Error, GatedProductGroupLayout, GroupScoring, GroupSelectionOperator,
    GroupedGatedProductSpec, GroupedNeuralBackend, LinearOperator, NeuralBackend, ParameterSpec,
    Parameterized, Tensor, TopKGroupSelectionSpec, TopKGroupSelectorSpec,
};
use eredu_runtime::{
    ParameterProvider, ResidentExpertProvider, RoutedExpertRequest, TensorParallelParameterProvider,
};

use crate::{
    decoder::{DecoderProjectionOperator, TensorParallelProjectionOperator},
    linear_format::standard_expert_projection,
};

use super::{specs::dense_parameters, FeedForwardPolicy, ModelArgs};

/// Checkpoint-compatible dense SwiGLU used by dense and shared paths.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct DenseSwiGlu<B: NeuralBackend> {
    /// Gate projection.
    pub gate: B::Linear,
    /// Down projection.
    pub down: B::Linear,
    /// Up projection.
    pub up: B::Linear,
    #[parameter(skip)]
    policy: eredu_nn::GatedProductPolicy,
}

impl<B: NeuralBackend> DenseSwiGlu<B> {
    pub(crate) fn from_parameters(
        parameters: eredu_nn::SplitGatedProductSpec,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        Ok(Self {
            gate: B::linear(parameters.gate, context)?,
            down: B::linear(parameters.down, context)?,
            up: B::linear(parameters.up, context)?,
            policy: parameters.policy,
        })
    }

    fn hidden(
        &mut self,
        input: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let gate = self.gate.forward(input, context)?;
        let up = self.up.forward(input, context)?;
        B::gated_product(gate, up, self.policy, context)
    }

    pub(crate) fn forward_instrumented(
        &mut self,
        input: &B::Tensor,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut crate::decoder::ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error> {
        let hidden = self.hidden(input, context)?;
        let units = instrumentation.apply("feed_forward.units", hidden)?;
        instrumentation.project::<B>(
            "feed_forward.write_input",
            &mut self.down,
            &units,
            parallel,
            context,
        )
    }
}

/// Grouped sigmoid router, routed bank, and one shared expert.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct SparseMoe<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> {
    #[parameter(skip)]
    layer: usize,
    #[parameter(skip)]
    observation: eredu_runtime::RoutedObservationPoints,
    /// Grouped sigmoid router with selection correction bias.
    pub router: B::Selector,
    /// Packed routed gated-product experts.
    pub experts: B::GatedProductGroups,
    /// Always-executed shared gated-product expert.
    pub shared: DenseSwiGlu<B>,
}

/// Complete sparse parameter scope, independent of layer invocation and mutable state.
#[derive(Debug, Clone)]
pub(crate) struct SparseParameters {
    pub routed: eredu_nn::RoutedGatedProductSpec,
    pub shared: eredu_nn::SplitGatedProductSpec,
}

impl DeclareLogicalParameters for SparseParameters {
    fn logical_parameters(&self) -> Result<Vec<eredu_nn::LogicalParameterDeclaration<'_>>, Error> {
        if self.shared.gate.input != self.routed.experts.input_dimensions()
            || self.shared.down.output != self.routed.experts.output_dimensions()
            || self.shared.down.output != self.shared.gate.input
            || self.shared.policy != eredu_nn::GatedProductPolicy::ordinary_silu()
            || self.routed.experts.policy() != eredu_nn::GatedProductPolicy::ordinary_silu()
        {
            return Err(Error::backend(
                "incompatible Kimi sparse feed-forward geometry or equation",
            ));
        }
        let mut parameters = self.routed.logical_parameters()?;
        parameters.extend(self.shared.logical_parameters()?);
        eredu_nn::validate_logical_parameter_topology(&parameters)
            .map_err(Error::backend_source)?;
        Ok(parameters)
    }
}

impl SparseParameters {
    fn validate(&self) -> Result<(), Error> {
        self.logical_parameters().map(|_| ())
    }

    /// Retains the global selector while restricting independent routed/shared channels.
    pub(crate) fn with_local_geometry(
        mut self,
        experts: GroupedGatedProductSpec,
        shared_intermediate: i32,
    ) -> Result<Self, Error> {
        if shared_intermediate <= 0
            || shared_intermediate > self.shared.gate.output
            || experts.intermediate_dimensions() > self.routed.experts.intermediate_dimensions()
        {
            return Err(Error::backend(
                "local Kimi expert widths exceed their declarations",
            ));
        }
        self.routed.experts = experts;
        self.shared.gate.output = shared_intermediate;
        self.shared.up.output = shared_intermediate;
        self.shared.down.input = shared_intermediate;
        self.validate()?;
        Ok(self)
    }
}

pub(crate) fn sparse_parameters(args: &ModelArgs, layer: usize) -> Result<SparseParameters, Error> {
    let parameters = SparseParameters {
        routed: eredu_nn::RoutedGatedProductSpec {
            selector: selector(args, layer)?,
            experts: grouped_experts(args, layer)?,
        },
        shared: dense_parameters(
            args,
            &format!("model.layers.{layer}.mlp.shared_experts"),
            args.moe_intermediate_size,
        )?,
    };
    parameters.validate()?;
    Ok(parameters)
}

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> SparseMoe<B> {
    fn from_parameters(
        layer: usize,
        parameters: SparseParameters,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        parameters.validate()?;
        let observation = eredu_runtime::RoutedObservationPoints::new(
            eredu_runtime::RoutedBankId::new(0),
            format!("model.layers.{layer}.mlp"),
            parameters.routed.selector.selection().group_count(),
        );
        Ok(Self {
            layer,
            observation,
            router: B::top_k_group_selector(parameters.routed.selector, context)?,
            experts: B::grouped_gated_product(parameters.routed.experts, context)?,
            shared: DenseSwiGlu::from_parameters(parameters.shared, context)?,
        })
    }
}

fn selector(args: &ModelArgs, layer: usize) -> Result<TopKGroupSelectorSpec, Error> {
    let prefix = format!("model.layers.{layer}.mlp");
    let gate_name = format!("{prefix}.gate.weight");
    let mut routing = TopKGroupSelectionSpec::new(
        args.num_experts,
        args.num_experts_per_token,
        GroupScoring::Sigmoid,
        args.moe_renormalize,
    )?;
    if args.use_grouped_topk {
        routing = routing.with_groups(args.num_expert_group, args.topk_group)?;
    }
    routing = routing.with_weight_policy(1e-20, args.routed_scaling_factor)?;
    TopKGroupSelectorSpec::new(
        args.hidden_size,
        ParameterSpec::trainable(&gate_name).map_err(Error::backend)?,
        crate::linear_format::standard_linear_format(
            &gate_name,
            args.weight_quantization_for(&gate_name).into(),
        )?,
        routing,
    )?
    .with_correction_bias(
        ParameterSpec::trainable(format!("{prefix}.gate.e_score_correction_bias"))
            .map_err(Error::backend)?,
    )
}

fn grouped_experts(args: &ModelArgs, layer: usize) -> Result<GroupedGatedProductSpec, Error> {
    args.validate().map_err(Error::backend_source)?;
    let policy = args
        .layer_policy(layer)
        .ok_or_else(|| Error::backend(format!("Kimi Linear has no layer {layer}")))?;
    if policy.feed_forward != FeedForwardPolicy::SparseMoe {
        return Err(Error::backend(format!(
            "Kimi Linear layer {layer} is not sparse MoE"
        )));
    }
    let experts_prefix = format!("model.layers.{layer}.mlp.experts");
    let gate_up_name = format!("{experts_prefix}.gate_up_proj");
    let down_name = format!("{experts_prefix}.down_proj");
    GroupedGatedProductSpec::new(
        args.num_experts,
        args.hidden_size,
        args.moe_intermediate_size,
        args.hidden_size,
        eredu_nn::GatedProductPolicy::ordinary_silu(),
        GatedProductGroupLayout::Packed {
            gate_up: standard_expert_projection(
                &gate_up_name,
                None,
                args.weight_quantization_for(&gate_up_name).into(),
                eredu_nn::LinearRowLayout::equal_partitions(2)?,
            )?,
            down: standard_expert_projection(
                &down_name,
                None,
                args.weight_quantization_for(&down_name).into(),
                eredu_nn::LinearRowLayout::Contiguous,
            )?,
        },
    )
}

/// Derives expert ownership from retained banks and optional admitted local geometry.
/// This operation requires no backend type, native modules or execution context.
pub fn expert_realization_plan(
    declaration: &super::Declaration,
    geometry: Option<&super::LocalGeometry>,
    topology: eredu_core::ParallelRankTopology,
) -> Result<Option<crate::ExpertRealizationPlan<GroupedGatedProductSpec>>, Error> {
    if let Some(geometry) = geometry {
        geometry.validate_for(declaration).map_err(Error::backend)?;
    }
    if declaration.expert_banks().is_empty() {
        return Ok(None);
    }
    let banks = declaration.expert_banks();
    let global_experts = banks
        .values()
        .next()
        .ok_or_else(|| Error::backend("sparse schedule has no declared expert bank"))?
        .group_count() as usize;
    let local_experts = i32::try_from(
        eredu_core::balanced_contiguous_range(
            global_experts,
            topology.expert_parallel_size(),
            topology.expert_parallel_rank(),
            false,
        )
        .map_err(Error::backend)?
        .len(),
    )
    .map_err(Error::backend)?;
    let owner_group = eredu_runtime::ExecutionGroupId::new(crate::decoder::TARGET_EXECUTION_GROUP)
        .map_err(Error::backend)?;
    let mut unit_specs = std::collections::BTreeMap::new();
    for (layer, bank) in banks {
        let width = geometry
            .and_then(|geometry| geometry.block(layer))
            .map_or(bank.intermediate_dimensions(), |geometry| {
                geometry.routed_intermediate
            });
        let spec = bank.with_group_geometry(local_experts, width)?;
        unit_specs.insert((owner_group.clone(), layer), spec);
    }
    crate::ExpertRealizationPlan::balanced(global_experts, topology, unit_specs)
        .map(Some)
        .map_err(Error::backend)
}

/// Per-layer dense-prefix or sparse Kimi feed-forward policy.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub enum FeedForward<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> {
    /// Dense prefix SwiGLU.
    Dense(DenseSwiGlu<B>),
    /// Routed plus shared experts.
    Sparse(SparseMoe<B>),
}

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> FeedForward<B> {
    pub(crate) fn from_parameters(
        layer: usize,
        parameters: super::declarations::FeedForwardParameters,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        match parameters {
            super::declarations::FeedForwardParameters::Dense(p) => {
                DenseSwiGlu::from_parameters(p, context).map(Self::Dense)
            }
            super::declarations::FeedForwardParameters::Sparse(p) => {
                SparseMoe::from_parameters(layer, p, context).map(Self::Sparse)
            }
        }
    }

    /// Executes dense or provider-backed sparse computation.
    pub fn forward_instrumented_with_provider<P>(
        &mut self,
        input: &B::Tensor,
        pass: eredu_runtime::ExpertPass,
        context: &<B::Tensor as Tensor>::Context,
        provider: &mut P,
        instrumentation: &mut crate::decoder::ComponentInstrumentation<'_, B::Tensor>,
        points: Option<eredu_runtime::RoutedObservationPoints>,
    ) -> Result<B::Tensor, Error>
    where
        P: ParameterProvider<B>,
        P::Error: std::fmt::Display,
    {
        self.forward_instrumented_with_executor(
            input,
            pass,
            None,
            context,
            instrumentation,
            points,
            |bank, request, context| {
                provider
                    .forward_grouped(bank, request, context)
                    .map_err(Error::backend_source)
            },
        )
    }

    /// Keeps local shared units and exchanged routed units in their declared scopes.
    pub fn forward_parallel_instrumented_with_provider<P>(
        &mut self,
        input: &B::Tensor,
        pass: eredu_runtime::ExpertPass,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        provider: &mut P,
        instrumentation: &mut crate::decoder::ComponentInstrumentation<'_, B::Tensor>,
        points: Option<eredu_runtime::RoutedObservationPoints>,
    ) -> Result<B::Tensor, Error>
    where
        P: TensorParallelParameterProvider<B>,
        P::Error: std::fmt::Display,
    {
        self.forward_instrumented_with_executor(
            input,
            pass,
            Some(parallel),
            context,
            instrumentation,
            points,
            |bank, request, context| {
                let output = provider
                    .forward_grouped_tensor_parallel(
                        bank,
                        request,
                        B::parallel_size(parallel),
                        context,
                    )
                    .map_err(Error::backend_source)?;
                eredu_runtime::reduce_routed_expert_tensor_parallel::<B>(output, parallel, context)
            },
        )
    }

    fn forward_instrumented_with_executor<F>(
        &mut self,
        input: &B::Tensor,
        pass: eredu_runtime::ExpertPass,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut crate::decoder::ComponentInstrumentation<'_, B::Tensor>,
        points: Option<eredu_runtime::RoutedObservationPoints>,
        execute: F,
    ) -> Result<B::Tensor, Error>
    where
        F: FnOnce(
            &mut B::GatedProductGroups,
            RoutedExpertRequest<'_, '_, B::Tensor>,
            &<B::Tensor as Tensor>::Context,
        ) -> Result<B::Tensor, Error>,
    {
        let Self::Sparse(sparse) = self else {
            let Self::Dense(dense) = self else {
                unreachable!()
            };
            return dense.forward_instrumented(input, parallel, context, instrumentation);
        };
        let point = points
            .as_ref()
            .or_else(|| instrumentation.enabled().then_some(&sparse.observation))
            .and_then(|points| points.bank(eredu_runtime::RoutedBankId::new(0)));
        let routes = match (point, instrumentation.observer()) {
            (Some(point), Some(observer)) => eredu_runtime::select_routes_with_observer(
                &mut sparse.router,
                input,
                context,
                point.path(),
                observer,
            )?,
            _ => sparse.router.select(input, context)?,
        };
        let request = RoutedExpertRequest {
            unit_observer: None,
            bank: eredu_runtime::RoutedBankId::new(0),
            layer: sparse.layer,
            input,
            routes: &routes,
            pass,
        };
        let routed = match (point, instrumentation.observer()) {
            (Some(point), Some(observer)) => eredu_runtime::with_routed_unit_observer(
                observer,
                point.path(),
                request,
                |request| execute(&mut sparse.experts, request, context),
            )
            .map_err(eredu_runtime::ObservedExpertProviderError::into_neural_error)?,
            _ => execute(&mut sparse.experts, request, context)?,
        };
        let shared = instrumentation.with_scope("mlp.shared_experts", |instrumentation| {
            let shared =
                sparse
                    .shared
                    .forward_instrumented(input, parallel, context, instrumentation)?;
            let shared = instrumentation.apply("feed_forward.write", shared)?;
            instrumentation.apply("feed_forward.output", shared)
        })?;
        let combined = routed.add(&shared, context)?;
        match (point, instrumentation.observer()) {
            (Some(point), Some(observer)) => {
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
            _ => Ok(combined),
        }
    }

    /// Executes dense or provider-backed sparse computation.
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
        self.forward_instrumented_with_provider(
            input,
            pass,
            context,
            provider,
            &mut crate::decoder::ComponentInstrumentation::disabled(),
            None,
        )
    }

    /// Executes sparse routed/shared work with one complete semantic observation.
    pub fn forward_observed_with_provider<P, O>(
        &mut self,
        point: eredu_runtime::RoutedObservationPoints,
        input: &B::Tensor,
        pass: eredu_runtime::ExpertPass,
        context: &<B::Tensor as Tensor>::Context,
        provider: &mut P,
        observer: &mut O,
    ) -> Result<B::Tensor, Error>
    where
        P: ParameterProvider<B>,
        P::Error: std::fmt::Display,
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        let path = point
            .bank(eredu_runtime::RoutedBankId::new(0))
            .and_then(|point| point.path().strip_suffix(".mlp"))
            .ok_or_else(|| Error::backend("Kimi routing point does not name its MLP invocation"))?
            .to_owned();
        let mut borrowed = eredu_runtime::BorrowedActivationObserver(observer);
        self.forward_instrumented_with_provider(
            input,
            pass,
            context,
            provider,
            &mut crate::decoder::ComponentInstrumentation::new(&path, &mut borrowed),
            Some(point),
        )
    }

    /// Executes tensor-partitioned dense/shared projections with provider-backed experts.
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
        self.forward_parallel_instrumented_with_provider(
            input,
            pass,
            parallel,
            context,
            provider,
            &mut crate::decoder::ComponentInstrumentation::disabled(),
            None,
        )
    }
}

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> DecoderProjectionOperator<B>
    for FeedForward<B>
{
    fn residual_observation(&self) -> &'static str {
        if matches!(self, Self::Sparse(_)) {
            "feed_forward.contribution"
        } else {
            "feed_forward.output"
        }
    }

    fn forward_feed_forward_observed(
        &mut self,
        input: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut crate::decoder::ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error> {
        self.forward_instrumented_with_provider(
            input,
            if input.dim(1) > 1 {
                eredu_runtime::ExpertPass::Prefill
            } else {
                eredu_runtime::ExpertPass::Decode
            },
            context,
            &mut ResidentExpertProvider,
            instrumentation,
            None,
        )
    }

    fn forward_feed_forward(
        &mut self,
        input: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let mut provider = ResidentExpertProvider;
        self.forward_with_provider(
            input,
            if input.dim(input.shape().len() - 2) > 1 {
                eredu_runtime::ExpertPass::Prefill
            } else {
                eredu_runtime::ExpertPass::Decode
            },
            context,
            &mut provider,
        )
    }
}

impl<B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend>
    TensorParallelProjectionOperator<B> for FeedForward<B>
{
    fn forward_feed_forward_parallel_observed(
        &mut self,
        input: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut crate::decoder::ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error> {
        self.forward_parallel_instrumented_with_provider(
            input,
            if input.dim(1) > 1 {
                eredu_runtime::ExpertPass::Prefill
            } else {
                eredu_runtime::ExpertPass::Decode
            },
            parallel,
            context,
            &mut ResidentExpertProvider,
            instrumentation,
            None,
        )
    }

    fn forward_feed_forward_parallel(
        &mut self,
        input: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.forward_feed_forward_parallel_observed(
            input,
            parallel,
            context,
            &mut crate::decoder::ComponentInstrumentation::disabled(),
        )
    }
}
