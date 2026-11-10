//! Qwen dense and routed gated-product feed-forward policy.

use std::collections::BTreeMap;

use eredu_nn::{
    Error, GatedProductGroupLayout, GroupScoring, GroupSelectionOperator, GroupedGatedProductSpec,
    GroupedNeuralBackend, ParameterSpec, Parameterized, Tensor, TopKGroupSelectionSpec,
    TopKGroupSelectorSpec,
};
use eredu_runtime::{
    ExpertPass, ParameterProvider, ResidentExpertProvider, RoutedExpertRequest,
    TensorParallelParameterProvider,
};

use crate::{
    decoder::{
        DecoderProjectionOperator, Mlp, TensorParallelProjectionOperator,
        TensorParallelRoutedProjectionOperator,
    },
    linear_format::standard_expert_projection,
};

use super::ModelArgs;

fn inferred_expert_pass<T: Tensor>(input: &T) -> ExpertPass {
    let sequence = input
        .shape()
        .get(input.shape().len().saturating_sub(2))
        .copied()
        .unwrap_or(1);
    if sequence > 1 {
        ExpertPass::Prefill
    } else {
        ExpertPass::Decode
    }
}

/// Dense or routed neural slots prepared before native materialization.
#[derive(Clone, Debug)]
pub enum FeedForwardParameters {
    /// Ordinary dense gated projections.
    Dense(eredu_nn::GatedProductSpec),
    /// Global routing and selected local expert geometry.
    Routed(eredu_nn::RoutedGatedProductSpec),
}
impl eredu_nn::DeclareLogicalParameters for FeedForwardParameters {
    fn logical_parameters(&self) -> Result<Vec<eredu_nn::LogicalParameterDeclaration<'_>>, Error> {
        match self {
            Self::Dense(spec) => spec.logical_parameters(),
            Self::Routed(spec) => spec.logical_parameters(),
        }
    }
}

/// Qwen3 top-k router and packed expert bank.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct RoutedGatedProduct<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> {
    /// Global decoder layer used for runtime expert identity.
    #[parameter(skip)]
    pub layer: usize,
    /// Learned top-k router.
    pub router: B::Selector,
    /// Packed or provider-materialized expert bank.
    pub experts: B::GatedProductGroups,
}

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> RoutedGatedProduct<B> {
    /// Builds one unloaded Qwen3-MoE feed-forward block.
    pub fn new(
        args: &ModelArgs,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        Self::from_spec(layer, routed_spec(args, layer)?, context)
    }

    /// Builds a partition-local bank while preserving the global router axis.
    pub fn new_partitioned(
        global: &ModelArgs,
        local: &ModelArgs,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        if !local.is_moe() {
            return Err(Error::backend(
                "partitioned routed Qwen construction requires Qwen3-MoE",
            ));
        }
        Self::from_spec(
            layer,
            routed_spec(global, layer)?
                .with_group_geometry(local.num_experts, local.moe_intermediate_size)?,
            context,
        )
    }

    pub(crate) fn from_spec(
        layer: usize,
        spec: eredu_nn::RoutedGatedProductSpec,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        spec.validate()?;
        Ok(Self {
            layer,
            router: B::top_k_group_selector(spec.selector, context)?,
            experts: B::grouped_gated_product(spec.experts, context)?,
        })
    }
}

pub(crate) fn execution_topology(
    declaration: &super::Declaration,
) -> Result<eredu_runtime::execution_topology::TextExecutionTopology, Error> {
    let args = declaration.args();
    if args.is_moe() {
        crate::decoder::topology::text_with_scopes(args, declaration.common(), |layer| {
            let spec = declaration.routed(layer)?;
            eredu_runtime::execution_topology::FeedForwardTopology::from_grouped_specs(
                &spec.selector,
                &spec.experts,
            )
        })
    } else {
        declaration.common().execution_topology(args)
    }
}

/// Declares the complete learned selector and packed bank for one routed invocation.
pub(crate) fn routed_spec(
    args: &ModelArgs,
    layer: usize,
) -> Result<eredu_nn::RoutedGatedProductSpec, Error> {
    if !args.is_moe() {
        return Err(Error::backend(
            "routed Qwen declaration requires Qwen3-MoE args",
        ));
    }
    args.validate().map_err(Error::backend_source)?;
    let layers = usize::try_from(args.num_hidden_layers).map_err(Error::backend)?;
    if layer >= layers {
        return Err(Error::backend(format!(
            "Qwen expert layer {layer} is outside {layers} declared layers"
        )));
    }
    let prefix = format!("{}.layers.{layer}.mlp", args.parameter_root);
    let router_name = format!("{prefix}.gate.weight");
    let selector = TopKGroupSelectorSpec::new(
        args.hidden_size,
        ParameterSpec::trainable(&router_name).map_err(Error::backend)?,
        crate::linear_format::standard_linear_format(
            &router_name,
            args.weight_quantization_for(&router_name).into(),
        )?,
        args.routing_spec()?,
    )?;
    let experts_prefix = format!("{}.layers.{layer}.mlp.experts", args.parameter_root);
    let gate_up_name = format!("{experts_prefix}.gate_up_proj");
    let down_name = format!("{experts_prefix}.down_proj");
    let experts = GroupedGatedProductSpec::new(
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
    )?;
    eredu_nn::RoutedGatedProductSpec::new(selector, experts)
}

/// Derives expert ownership and rank-local banks from validated neutral geometry.
/// This operation requires no constructed modules or backend context.
pub fn expert_realization_plan(
    declaration: &super::Declaration,
    geometry: Option<&super::LocalGeometry>,
    topology: eredu_core::ParallelRankTopology,
) -> Result<Option<crate::ExpertRealizationPlan<GroupedGatedProductSpec>>, Error> {
    let args = declaration.args();
    args.validate().map_err(Error::backend_source)?;
    if let Some(geometry) = geometry {
        geometry
            .validate_with_declaration(args, declaration.common())
            .map_err(Error::backend_source)?;
    }
    if !args.is_moe() {
        return Ok(None);
    }
    let banks = declaration.expert_banks()?;
    let global_experts = banks
        .values()
        .next()
        .ok_or_else(|| Error::backend("routed Qwen has no declared expert banks"))?
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
    let owner_group =
        eredu_runtime::ExecutionGroupId::new(crate::decoder::TEXT_DECODER_EXECUTION_GROUP)
            .map_err(Error::backend)?;
    let unit_specs = banks
        .into_iter()
        .map(|(layer, bank)| {
            let width = geometry
                .and_then(|geometry| geometry.block(layer))
                .map_or(bank.intermediate_dimensions(), |local| {
                    local.moe_intermediate_size
                });
            bank.with_group_geometry(local_experts, width)
                .map(|spec| ((owner_group.clone(), layer), spec))
        })
        .collect::<Result<BTreeMap<_, _>, _>>()?;
    crate::ExpertRealizationPlan::balanced(global_experts, topology, unit_specs)
        .map(Some)
        .map_err(Error::backend)
}

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> DecoderProjectionOperator<B>
    for RoutedGatedProduct<B>
{
    fn residual_observation(&self) -> &'static str {
        "feed_forward.contribution"
    }

    fn forward_feed_forward(
        &mut self,
        input: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let routes = self.router.select(input, context)?;
        let mut provider = ResidentExpertProvider;
        ParameterProvider::<B>::forward_grouped(
            &mut provider,
            &mut self.experts,
            RoutedExpertRequest {
                unit_observer: None,
                bank: eredu_runtime::RoutedBankId::new(0),
                layer: self.layer,
                input,
                routes: &routes,
                pass: inferred_expert_pass(input),
            },
            context,
        )
    }
}

impl<B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend>
    TensorParallelProjectionOperator<B> for RoutedGatedProduct<B>
{
    fn forward_feed_forward_parallel(
        &mut self,
        input: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let routes = self.router.select(input, context)?;
        let mut provider = ResidentExpertProvider;
        let output = TensorParallelParameterProvider::<B>::forward_grouped_tensor_parallel(
            &mut provider,
            &mut self.experts,
            RoutedExpertRequest {
                unit_observer: None,
                bank: eredu_runtime::RoutedBankId::new(0),
                layer: self.layer,
                input,
                routes: &routes,
                pass: inferred_expert_pass(input),
            },
            B::parallel_size(parallel),
            context,
        )?;
        eredu_runtime::reduce_routed_expert_tensor_parallel::<B>(output, parallel, context)
    }
}

/// One Qwen feed-forward policy used by the single decoder block type.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub enum FeedForward<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> {
    /// Dense SwiGLU used by Qwen2 and dense Qwen3.
    Dense(Mlp<B>),
    /// Top-k routed gated-product used by Qwen3-MoE.
    Routed(RoutedGatedProduct<B>),
}

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> FeedForward<B> {
    /// Builds the validated dense or routed policy for one layer.
    pub fn new(
        args: &ModelArgs,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        if args.is_moe() {
            RoutedGatedProduct::new(args, layer, context).map(Self::Routed)
        } else {
            Mlp::new(args, layer, context).map(Self::Dense)
        }
    }

    /// Builds a routed partition with a global selector and owner-local bank.
    pub fn new_partitioned(
        global: &ModelArgs,
        local: &ModelArgs,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        RoutedGatedProduct::new_partitioned(global, local, layer, context).map(Self::Routed)
    }

    /// Executes through a runtime-owned routed-expert provider.
    ///
    /// Dense blocks retain their ordinary static path. Routed blocks submit
    /// one typed route batch without exposing backend storage or residency to
    /// architecture code.
    pub fn forward_with_provider<P>(
        &mut self,
        layer: usize,
        pass: ExpertPass,
        input: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
        provider: &mut P,
    ) -> Result<B::Tensor, Error>
    where
        P: ParameterProvider<B>,
        P::Error: std::fmt::Display,
    {
        match self {
            Self::Dense(mlp) => mlp.forward_feed_forward(input, context),
            Self::Routed(moe) => {
                let routes = eredu_runtime::select_routes_with_provider::<B, P>(
                    &mut moe.router,
                    input,
                    context,
                    provider,
                    eredu_runtime::RoutedBankId::new(0),
                )?;
                provider
                    .forward_grouped(
                        &mut moe.experts,
                        RoutedExpertRequest {
                            unit_observer: None,
                            bank: eredu_runtime::RoutedBankId::new(0),
                            layer,
                            input,
                            routes: &routes,
                            pass,
                        },
                        context,
                    )
                    .map_err(Error::backend_source)
            }
        }
    }

    /// Executes the selected feed-forward policy with rank-local parameters
    /// and exactly one row reduction for tensor-parallel output.
    pub fn forward_with_provider_parallel<P>(
        &mut self,
        layer: usize,
        pass: ExpertPass,
        input: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        provider: &mut P,
    ) -> Result<B::Tensor, Error>
    where
        P: TensorParallelParameterProvider<B>,
        P::Error: std::fmt::Display,
    {
        match self {
            Self::Dense(mlp) => mlp.forward_feed_forward_parallel(input, parallel, context),
            Self::Routed(moe) => {
                let routes = moe.router.select(input, context)?;
                let output = provider
                    .forward_grouped_tensor_parallel(
                        &mut moe.experts,
                        RoutedExpertRequest {
                            unit_observer: None,
                            bank: eredu_runtime::RoutedBankId::new(0),
                            layer,
                            input,
                            routes: &routes,
                            pass,
                        },
                        B::parallel_size(parallel),
                        context,
                    )
                    .map_err(Error::backend_source)?;
                eredu_runtime::reduce_routed_expert_tensor_parallel::<B>(output, parallel, context)
            }
        }
    }
}

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> DecoderProjectionOperator<B>
    for FeedForward<B>
{
    fn residual_observation(&self) -> &'static str {
        match self {
            Self::Dense(_) => "feed_forward.output",
            Self::Routed(_) => "feed_forward.contribution",
        }
    }

    fn forward_feed_forward(
        &mut self,
        input: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        match self {
            Self::Dense(mlp) => mlp.forward_feed_forward(input, context),
            Self::Routed(moe) => moe.forward_feed_forward(input, context),
        }
    }
    fn forward_feed_forward_observed(
        &mut self,
        input: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut crate::decoder::ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error> {
        match self {
            Self::Dense(mlp) => mlp.forward_feed_forward_observed(input, context, instrumentation),
            Self::Routed(moe) => moe.forward_feed_forward(input, context),
        }
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
        match self {
            Self::Dense(mlp) => mlp.forward_feed_forward_parallel_observed(
                input,
                parallel,
                context,
                instrumentation,
            ),
            Self::Routed(moe) => moe.forward_feed_forward_parallel(input, parallel, context),
        }
    }

    fn forward_feed_forward_parallel(
        &mut self,
        input: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        match self {
            Self::Dense(mlp) => mlp.forward_feed_forward_parallel(input, parallel, context),
            Self::Routed(moe) => moe.forward_feed_forward_parallel(input, parallel, context),
        }
    }
}

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend>
    crate::decoder::RoutedProjectionOperator<B> for FeedForward<B>
{
    fn forward_observed_with_provider<P>(
        &mut self,
        layer: usize,
        input: &B::Tensor,
        pass: ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut crate::decoder::ComponentInstrumentation<'_, B::Tensor>,
        points: Option<eredu_runtime::RoutedObservationPoints>,
    ) -> Result<B::Tensor, Error>
    where
        P: eredu_runtime::ParameterProvider<B>,
        P::Error: std::fmt::Display,
    {
        if let Self::Dense(mlp) = self {
            return mlp.forward_feed_forward_observed(input, context, instrumentation);
        }
        match (points, instrumentation.observer()) {
            (Some(points), Some(observer)) => eredu_runtime::ObservedExpertProvider::new(
                provider, observer, points,
            )
            .execute_neural(|provider| {
                <Self as crate::decoder::RoutedProjectionOperator<B>>::forward_with_provider(
                    self, layer, input, pass, provider, context,
                )
            }),
            _ => <Self as crate::decoder::RoutedProjectionOperator<B>>::forward_with_provider(
                self, layer, input, pass, provider, context,
            ),
        }
    }

    const COMPONENT_OBSERVATIONS: bool = true;

    fn forward_with_provider<P>(
        &mut self,
        layer: usize,
        input: &B::Tensor,
        pass: ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        P: ParameterProvider<B>,
        P::Error: std::fmt::Display,
    {
        FeedForward::forward_with_provider(self, layer, pass, input, context, provider)
    }
}

impl<B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend>
    TensorParallelRoutedProjectionOperator<B> for FeedForward<B>
{
    fn forward_parallel_observed_with_provider<P>(
        &mut self,
        layer: usize,
        input: &B::Tensor,
        pass: ExpertPass,
        provider: &mut P,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut crate::decoder::ComponentInstrumentation<'_, B::Tensor>,
        points: Option<eredu_runtime::RoutedObservationPoints>,
    ) -> Result<B::Tensor, Error>
    where
        P: eredu_runtime::TensorParallelParameterProvider<B>,
        P::Error: std::fmt::Display,
    {
        if let Self::Dense(mlp) = self {
            return mlp.forward_feed_forward_parallel_observed(
                input,
                parallel,
                context,
                instrumentation,
            );
        }
        match (points, instrumentation.observer()) {
            (Some(points), Some(observer)) => {
                eredu_runtime::ObservedExpertProvider::new(provider, observer, points)
                    .execute_neural(|provider| {
                        <Self as crate::decoder::TensorParallelRoutedProjectionOperator<B>>::forward_parallel_with_provider(
                            self, layer, input, pass, provider, parallel, context,
                        )
                    })
            }
            _ => {
                <Self as crate::decoder::TensorParallelRoutedProjectionOperator<B>>::forward_parallel_with_provider(
                    self, layer, input, pass, provider, parallel, context,
                )
            }
        }
    }

    fn forward_parallel_with_provider<P>(
        &mut self,
        layer: usize,
        input: &B::Tensor,
        pass: ExpertPass,
        provider: &mut P,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        P: TensorParallelParameterProvider<B>,
        P::Error: std::fmt::Display,
    {
        FeedForward::forward_with_provider_parallel(
            self, layer, pass, input, parallel, context, provider,
        )
    }
}

impl ModelArgs {
    /// The same neutral selection policy drives construction and intervention discovery.
    pub(crate) fn routing_spec(&self) -> Result<TopKGroupSelectionSpec, Error> {
        TopKGroupSelectionSpec::new(
            self.num_experts,
            self.num_experts_per_tok,
            GroupScoring::Softmax,
            self.norm_topk_prob,
        )
    }
}

impl FeedForwardParameters {
    pub(crate) fn prepare(
        global: &ModelArgs,
        local: &ModelArgs,
        _layer: usize,
        dense: Option<&eredu_nn::GatedProductSpec>,
        retained: Option<&Self>,
    ) -> Result<Self, Error> {
        if let Some(parameters) = retained {
            return match (global.is_moe(), parameters) {
                (true, Self::Routed(spec)) => Ok(Self::Routed(
                    spec.clone()
                        .with_group_geometry(local.num_experts, local.moe_intermediate_size)?,
                )),
                (false, Self::Dense(spec)) => Ok(Self::Dense(
                    spec.with_intermediate_dimensions(local.intermediate_size)?,
                )),
                _ => Err(Error::backend(
                    "Qwen projection declaration changes its equation",
                )),
            };
        }
        if global.is_moe() {
            return Err(Error::backend(
                "routed Qwen requires retained projection declarations",
            ));
        }
        Ok(Self::Dense(dense.cloned().ok_or_else(|| {
            Error::backend("dense Qwen requires retained feed-forward slots")
        })?))
    }
}
