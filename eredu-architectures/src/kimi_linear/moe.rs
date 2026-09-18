//! Kimi dense-prefix and grouped routed-expert feed-forward operators.

use eredu_nn::{
    Error, GatedProductGroupLayout, GroupScoring, GroupSelectionOperator, GroupedGatedProductSpec,
    GroupedNeuralBackend, LinearOperator, LinearSpec, NeuralBackend, ParameterSpec, Parameterized,
    Tensor, TopKGroupSelectionSpec, TopKGroupSelectorSpec,
};
use eredu_runtime::{
    ResidentExpertProvider, RoutedExpertProvider, RoutedExpertRequest,
    TensorParallelRoutedExpertProvider,
};

use crate::{
    decoder::{DecoderProjectionOperator, TensorParallelProjectionOperator},
    linear_format::standard_expert_projection,
};

use super::{FeedForwardPolicy, ModelArgs};

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
}

impl<B: NeuralBackend> DenseSwiGlu<B> {
    pub(crate) fn new(
        args: &ModelArgs,
        prefix: &str,
        intermediate: i32,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let linear = |field: &str, input, output| {
            let name = format!("{prefix}.{field}.weight");
            B::linear(
                LinearSpec {
                    input,
                    output,
                    weight: ParameterSpec::trainable(&name).map_err(Error::backend)?,
                    bias: None,
                    format: crate::linear_format::standard_linear_format(
                        &name,
                        args.weight_quantization_for(&name).into(),
                    )?,
                },
                context,
            )
        };
        Ok(Self {
            gate: linear("gate_proj", args.hidden_size, intermediate)?,
            down: linear("down_proj", intermediate, args.hidden_size)?,
            up: linear("up_proj", args.hidden_size, intermediate)?,
        })
    }

    fn hidden(
        &mut self,
        input: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let gate = self.gate.forward(input, context)?;
        let up = self.up.forward(input, context)?;
        B::gated_product(gate, up, eredu_nn::GatedProductPolicy::default(), context)
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
    #[parameter(skip, metadata)]
    layer: usize,
    #[parameter(skip, metadata)]
    observation: eredu_runtime::RoutedObservationPoints,
    /// Grouped sigmoid router with selection correction bias.
    pub router: B::Selector,
    /// Packed routed gated-product experts.
    pub experts: B::GatedProductGroups,
    /// Always-executed shared gated-product expert.
    pub shared: DenseSwiGlu<B>,
}

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> SparseMoe<B> {
    fn new(
        args: &ModelArgs,
        layer: usize,
        routed_intermediate: i32,
        shared_intermediate: i32,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let spec = expert_bank_spec_with_width(args, layer, routed_intermediate)?;
        Self::new_with_spec(args, layer, spec, shared_intermediate, context)
    }

    fn new_with_spec(
        args: &ModelArgs,
        layer: usize,
        spec: GroupedGatedProductSpec,
        shared_intermediate: i32,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
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
        let selector = TopKGroupSelectorSpec::new(
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
        )?;
        let router = B::top_k_group_selector(selector, context)?;
        let experts = B::grouped_gated_product(spec, context)?;
        Ok(Self {
            layer,
            observation: args
                .routed_observation_points(&format!("model.layers.{layer}"), layer, None)?
                .ok_or_else(|| Error::backend("Kimi sparse unit has no routing declaration"))?,
            router,
            experts,
            shared: DenseSwiGlu::new(
                args,
                &format!("{prefix}.shared_experts"),
                shared_intermediate,
                context,
            )?,
        })
    }
}

/// Returns the architecture-owned routed expert specification for one sparse layer.
pub fn expert_bank_spec(args: &ModelArgs, layer: usize) -> Result<GroupedGatedProductSpec, Error> {
    expert_bank_spec_with_width(args, layer, args.moe_intermediate_size)
}

fn expert_bank_spec_with_width(
    args: &ModelArgs,
    layer: usize,
    intermediate: i32,
) -> Result<GroupedGatedProductSpec, Error> {
    let experts_prefix = format!("model.layers.{layer}.mlp.experts");
    let gate_up_name = format!("{experts_prefix}.gate_up_proj");
    let down_name = format!("{experts_prefix}.down_proj");
    GroupedGatedProductSpec::new(
        args.num_experts,
        args.hidden_size,
        intermediate,
        args.hidden_size,
        eredu_nn::GatedProductPolicy::ordinary_silu(),
        GatedProductGroupLayout::Packed {
            gate_up: standard_expert_projection(
                &gate_up_name,
                None,
                args.weight_quantization_for(&gate_up_name).into(),
            )?,
            down: standard_expert_projection(
                &down_name,
                None,
                args.weight_quantization_for(&down_name).into(),
            )?,
        },
    )
}

/// Derives complete expert ownership and rank-local bank geometry from Kimi Linear.
pub fn expert_realization_plan<B>(
    architecture: &super::LayeredModel<B>,
    topology: eredu_core::ParallelRankTopology,
) -> Result<Option<crate::ExpertRealizationPlan<GroupedGatedProductSpec>>, Error>
where
    B: GroupedNeuralBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::BlockwiseAttentionBackend,
{
    let args = architecture.args();
    if !args.has_sparse_moe_layers() {
        return Ok(None);
    }
    let global_experts = usize::try_from(args.num_experts).map_err(Error::backend)?;
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
    for layer in 0..usize::try_from(args.num_hidden_layers).map_err(Error::backend)? {
        if args.layer_policy(layer).map(|policy| policy.feed_forward)
            != Some(FeedForwardPolicy::SparseMoe)
        {
            continue;
        }
        let width = architecture
            .parallel_geometry()
            .and_then(|geometry| geometry.block(layer))
            .map_or(args.moe_intermediate_size, |geometry| {
                geometry.routed_intermediate
            });
        let spec = expert_bank_spec_with_width(args, layer, width)?
            .with_group_geometry(local_experts, width)?;
        unit_specs.insert((owner_group.clone(), layer), spec);
    }
    crate::ExpertRealizationPlan::balanced(global_experts, topology, unit_specs)
        .map(Some)
        .map_err(Error::backend)
}

/// Derives the complete replicated target expert plan from normalized geometry.
pub fn replicated_expert_realization_plan(
    args: &ModelArgs,
) -> Result<crate::ExpertRealizationPlan<GroupedGatedProductSpec>, Error> {
    if !args.has_sparse_moe_layers() {
        return Err(Error::backend(
            "Kimi Linear routed text requires sparse units",
        ));
    }
    let global_experts = usize::try_from(args.num_experts).map_err(Error::backend)?;
    let owner_group = eredu_runtime::ExecutionGroupId::new(crate::decoder::TARGET_EXECUTION_GROUP)
        .map_err(Error::backend)?;
    let layers = usize::try_from(args.num_hidden_layers).map_err(Error::backend)?;
    let unit_specs = (0..layers)
        .filter(|layer| {
            args.layer_policy(*layer).map(|policy| policy.feed_forward)
                == Some(FeedForwardPolicy::SparseMoe)
        })
        .map(|layer| {
            expert_bank_spec_with_width(args, layer, args.moe_intermediate_size)
                .map(|spec| ((owner_group.clone(), layer), spec))
        })
        .collect::<Result<std::collections::BTreeMap<_, _>, _>>()?;
    crate::ExpertRealizationPlan::balanced(
        global_experts,
        eredu_core::ParallelRankTopology::new(
            eredu_core::ParallelTopology::new(1, 1, 1, 1).map_err(Error::backend)?,
            0,
        )
        .map_err(Error::backend)?,
        unit_specs,
    )
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
    /// Builds one scheduled feed-forward operator at replicated geometry.
    pub fn new(
        args: &ModelArgs,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        Self::new_with_geometry(
            args,
            layer,
            args.intermediate_size,
            args.moe_intermediate_size,
            args.moe_intermediate_size * args.num_shared_experts,
            context,
        )
    }

    /// Builds one scheduled operator from placement-resolved widths.
    pub fn new_with_geometry(
        args: &ModelArgs,
        layer: usize,
        dense_intermediate: i32,
        routed_intermediate: i32,
        shared_intermediate: i32,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        Self::new_with_geometry_and_routed_spec(
            args,
            layer,
            dense_intermediate,
            routed_intermediate,
            shared_intermediate,
            None,
            context,
        )
    }

    pub(crate) fn new_with_geometry_and_routed_spec(
        args: &ModelArgs,
        layer: usize,
        dense_intermediate: i32,
        routed_intermediate: i32,
        shared_intermediate: i32,
        routed_spec: Option<GroupedGatedProductSpec>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let policy = args
            .layer_policy(layer)
            .ok_or_else(|| Error::backend(format!("Kimi Linear has no layer {layer}")))?;
        let prefix = format!("model.layers.{layer}.mlp");
        match policy.feed_forward {
            FeedForwardPolicy::Dense => {
                DenseSwiGlu::new(args, &prefix, dense_intermediate, context).map(Self::Dense)
            }
            FeedForwardPolicy::SparseMoe => match routed_spec {
                Some(spec) => {
                    SparseMoe::new_with_spec(args, layer, spec, shared_intermediate, context)
                }
                None => SparseMoe::new(
                    args,
                    layer,
                    routed_intermediate,
                    shared_intermediate,
                    context,
                ),
            }
            .map(Self::Sparse),
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
        P: RoutedExpertProvider<B>,
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
                    .map_err(Error::backend_retained_source)
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
        P: TensorParallelRoutedExpertProvider<B>,
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
                    .map_err(Error::backend_retained_source)?;
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
        P: RoutedExpertProvider<B>,
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
        P: RoutedExpertProvider<B>,
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
        P: TensorParallelRoutedExpertProvider<B>,
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
