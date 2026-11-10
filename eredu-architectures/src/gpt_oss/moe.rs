//! Backend-neutral GPT-OSS biased routing and bounded gated-product experts.

use eredu_checkpoint::WeightQuantization;
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
        DecoderProjectionOperator, TensorParallelProjectionOperator,
        TensorParallelRoutedProjectionOperator,
    },
    linear_format::standard_expert_projection,
};

use super::config::ModelArgs;

fn inferred_pass<T: Tensor>(input: &T) -> ExpertPass {
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

/// GPT-OSS SelectedSoftmax router and canonical biased MXFP4 expert bank.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct RoutedMlp<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> {
    /// Global layer identity used by runtime expert providers.
    #[parameter(skip)]
    pub layer: usize,
    /// Learned biased router.
    pub router: B::Selector,
    /// Canonical component-major expert bank.
    pub experts: B::GatedProductGroups,
}

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend>
    crate::decoder::RoutedProjectionOperator<B> for RoutedMlp<B>
{
    const COMPONENT_OBSERVATIONS: bool = true;

    fn forward_with_provider<P>(
        &mut self,
        _layer: usize,
        input: &B::Tensor,
        pass: ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        P: ParameterProvider<B>,
        P::Error: std::fmt::Display,
    {
        RoutedMlp::forward_with_provider(self, input, pass, provider, context)
    }
}

impl<B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend>
    TensorParallelRoutedProjectionOperator<B> for RoutedMlp<B>
{
    fn forward_parallel_with_provider<P>(
        &mut self,
        _layer: usize,
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
        RoutedMlp::forward_parallel_with_provider(self, input, pass, parallel, provider, context)
    }
}

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> RoutedMlp<B> {
    /// Builds one unloaded GPT-OSS routed feed-forward operator.
    pub fn new(
        args: &ModelArgs,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        Self::from_spec(layer, routed_spec(args, layer)?, context)
    }

    /// Builds a partition-local bank while retaining global SelectedSoftmax routing.
    pub fn new_partitioned(
        global: &ModelArgs,
        local: &ModelArgs,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        Self::from_spec(
            layer,
            local_spec(local, &routed_spec(global, layer)?)?,
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

    /// Executes routed experts through a runtime-owned provider.
    pub fn forward_with_provider<P>(
        &mut self,
        input: &B::Tensor,
        pass: ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        P: ParameterProvider<B>,
        P::Error: std::fmt::Display,
    {
        let routes = self.router.select(input, context)?;
        provider
            .forward_grouped(
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
            .map_err(Error::backend_source)
    }

    /// Executes provider-backed rank-local TP work and performs one reduction.
    pub fn forward_parallel_with_provider<P>(
        &mut self,
        input: &B::Tensor,
        pass: ExpertPass,
        parallel: &B::ParallelContext,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        P: TensorParallelParameterProvider<B>,
        P::Error: std::fmt::Display,
    {
        let routes = self.router.select(input, context)?;
        let output = provider
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
                B::parallel_size(parallel),
                context,
            )
            .map_err(Error::backend_source)?;
        eredu_runtime::reduce_routed_expert_tensor_parallel::<B>(output, parallel, context)
    }
}

/// Declares global biased routing and native MXFP4 experts as one validated scope.
pub(crate) fn routed_spec(
    args: &ModelArgs,
    layer: usize,
) -> Result<eredu_nn::RoutedGatedProductSpec, Error> {
    args.validate().map_err(Error::backend_source)?;
    let layers = usize::try_from(args.num_hidden_layers).map_err(Error::backend)?;
    if layer >= layers {
        return Err(Error::backend(format!(
            "GPT-OSS expert layer {layer} is outside {layers} declared layers"
        )));
    }
    let prefix = format!("{}.layers.{layer}.mlp", args.parameter_root);
    let router_weight = format!("{prefix}.router.weight");
    let router_bias = format!("{prefix}.router.bias");
    let routing = TopKGroupSelectionSpec::new(
        args.num_local_experts,
        args.num_experts_per_tok,
        GroupScoring::SelectedSoftmax,
        false,
    )?;
    let selector = TopKGroupSelectorSpec::new(
        args.hidden_size,
        ParameterSpec::trainable(&router_weight).map_err(Error::backend)?,
        crate::linear_format::standard_linear_format(
            &router_weight,
            args.checkpoint_weight_quantization_for(&router_weight)
                .into(),
        )?,
        routing,
    )?
    .with_bias(ParameterSpec::trainable(&router_bias).map_err(Error::backend)?)?;
    let expert_prefix = format!("{}.layers.{layer}.mlp.experts", args.parameter_root);
    let gate_up_weight = format!("{expert_prefix}.gate_up_proj");
    let gate_up_bias = format!("{expert_prefix}.gate_up_proj_bias");
    let down_weight = format!("{expert_prefix}.down_proj");
    let down_bias = format!("{expert_prefix}.down_proj_bias");
    let experts = GroupedGatedProductSpec::new(
        args.num_local_experts,
        args.hidden_size,
        args.intermediate_size,
        args.hidden_size,
        args.gated_product_policy,
        GatedProductGroupLayout::Packed {
            gate_up: standard_expert_projection(
                &gate_up_weight,
                Some(&gate_up_bias),
                WeightQuantization::MxFp4.into(),
                eredu_nn::LinearRowLayout::equal_partitions(2)?,
            )?,
            down: standard_expert_projection(
                &down_weight,
                Some(&down_bias),
                WeightQuantization::MxFp4.into(),
                eredu_nn::LinearRowLayout::Contiguous,
            )?,
        },
    )?;
    eredu_nn::RoutedGatedProductSpec::new(selector, experts)
}

/// Derives complete expert authority from normalized configuration and retained placement.
/// No native model, tensor or context participates in this projection.
pub fn expert_realization_plan(
    declaration: &super::Declaration,
    geometry: Option<&super::LocalGeometry>,
    topology: eredu_core::ParallelRankTopology,
) -> Result<crate::ExpertRealizationPlan<GroupedGatedProductSpec>, Error> {
    let args = declaration.args();
    args.validate().map_err(Error::backend_source)?;
    if let Some(geometry) = geometry {
        geometry
            .validate_with_declaration(args, declaration.common())
            .map_err(Error::backend_source)?;
    }
    let banks = declaration.expert_banks()?;
    let global_experts = banks
        .values()
        .next()
        .ok_or_else(|| Error::backend("GPT-OSS has no declared expert banks"))?
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
                    local.intermediate_size
                });
            bank.with_group_geometry(local_experts, width)
                .map(|spec| ((owner_group.clone(), layer), spec))
        })
        .collect::<Result<_, Error>>()?;
    crate::ExpertRealizationPlan::balanced(global_experts, topology, unit_specs)
        .map_err(Error::backend)
}

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> DecoderProjectionOperator<B>
    for RoutedMlp<B>
{
    fn residual_observation(&self) -> &'static str {
        "feed_forward.contribution"
    }

    fn forward_feed_forward(
        &mut self,
        input: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.forward_with_provider(
            input,
            inferred_pass(input),
            &mut ResidentExpertProvider,
            context,
        )
    }
}

impl<B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend>
    TensorParallelProjectionOperator<B> for RoutedMlp<B>
{
    fn forward_feed_forward_parallel(
        &mut self,
        input: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.forward_parallel_with_provider(
            input,
            inferred_pass(input),
            parallel,
            &mut ResidentExpertProvider,
            context,
        )
    }
}

/// Declares global routing with the selected local bank geometry.
pub(crate) fn local_spec(
    local: &ModelArgs,
    parameters: &eredu_nn::RoutedGatedProductSpec,
) -> Result<eredu_nn::RoutedGatedProductSpec, Error> {
    parameters
        .clone()
        .with_group_geometry(local.num_local_experts, local.intermediate_size)
}
