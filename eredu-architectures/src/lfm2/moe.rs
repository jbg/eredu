//! Dense and routed LFM2 feed-forward policies.

use eredu_nn::{
    Error, GatedProductGroupLayout, GroupScoring, GroupSelectionOperator, GroupedGatedProductSpec,
    GroupedNeuralBackend, LinearOperator, LinearSpec, NeuralBackend, ParameterSpec, Parameterized,
    Tensor, TopKGroupSelectionSpec, TopKGroupSelectorSpec,
};
use eredu_runtime::{
    ParameterProvider, ResidentExpertProvider, RoutedExpertRequest, TensorParallelParameterProvider,
};

use crate::{
    decoder::{DecoderProjectionOperator, TensorParallelProjectionOperator},
    linear_format::standard_expert_projection,
};

use super::{FeedForwardPolicy, ModelArgs};

/// Dense LFM2 SwiGLU with checkpoint-compatible projection identities.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct DenseSwiGlu<B: NeuralBackend> {
    /// Gate projection (`w1`).
    pub gate: B::Linear,
    /// Down projection (`w2`).
    pub down: B::Linear,
    /// Up projection (`w3`).
    pub up: B::Linear,
    #[parameter(skip)]
    policy: eredu_nn::GatedProductPolicy,
}

/// Declares one dense SwiGLU branch at global or rank-local intermediate width.
pub(crate) fn dense_parameters_at(
    args: &ModelArgs,
    root: &str,
    intermediate: i32,
) -> Result<eredu_nn::SplitGatedProductSpec, Error> {
    let prefix = format!("{root}.feed_forward");
    let linear = |field: &str, input, output| -> Result<LinearSpec, Error> {
        let name = format!("{prefix}.{field}.weight");
        Ok(LinearSpec {
            input,
            output,
            weight: ParameterSpec::trainable(&name).map_err(Error::backend)?,
            bias: None,
            format: crate::linear_format::standard_linear_format(
                &name,
                args.weight_quantization_for(&name).into(),
            )?,
        })
    };
    eredu_nn::SplitGatedProductSpec::new(
        linear("w1", args.hidden_size, intermediate)?,
        linear("w3", args.hidden_size, intermediate)?,
        linear("w2", intermediate, args.hidden_size)?,
        eredu_nn::GatedProductPolicy::default(),
    )
}

impl<B: NeuralBackend> DenseSwiGlu<B> {
    pub(crate) fn from_parameters(
        parameters: eredu_nn::SplitGatedProductSpec,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        eredu_nn::DeclareLogicalParameters::logical_parameters(&parameters)?;
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
        let hidden = instrumentation.apply("feed_forward.units", hidden)?;
        instrumentation.project::<B>(
            "feed_forward.write_input",
            &mut self.down,
            &hidden,
            parallel,
            context,
        )
    }
}

/// Sigmoid router and routed gated-product expert bank used by LFM2-MoE.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct RoutedGatedProduct<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> {
    /// Global physical layer identity.
    #[parameter(skip)]
    pub layer: usize,
    /// Learned sigmoid top-k router.
    pub router: B::Selector,
    /// Packed routed experts.
    pub experts: B::GatedProductGroups,
}

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> RoutedGatedProduct<B> {
    pub(crate) fn from_spec(
        layer: usize,
        spec: eredu_nn::RoutedGatedProductSpec,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        validate_routed_spec(&spec)?;
        Ok(Self {
            layer,
            router: B::top_k_group_selector(spec.selector, context)?,
            experts: B::grouped_gated_product(spec.experts, context)?,
        })
    }
}

fn validate_routed_spec(spec: &eredu_nn::RoutedGatedProductSpec) -> Result<(), Error> {
    if spec.experts.input_dimensions() != spec.experts.output_dimensions()
        || spec.experts.policy() != eredu_nn::GatedProductPolicy::ordinary_silu()
        || spec.experts.reduction() != eredu_nn::GroupReduction::SequentialGroupOrder
    {
        return Err(Error::backend(
            "incompatible LFM2 routed residual geometry or equation",
        ));
    }
    spec.validate()
}

/// Complete selector/bank parameter scope, independent of invocation and provider ownership.
pub(crate) fn routed_spec(
    args: &ModelArgs,
    layer: usize,
) -> Result<eredu_nn::RoutedGatedProductSpec, Error> {
    let spec = eredu_nn::RoutedGatedProductSpec {
        selector: selector(args, layer)?,
        experts: grouped_experts(args, layer)?,
    };
    validate_routed_spec(&spec)?;
    Ok(spec)
}

fn selector(args: &ModelArgs, layer: usize) -> Result<TopKGroupSelectorSpec, Error> {
    let prefix = format!("model.layers.{layer}.feed_forward");
    let gate_name = format!("{prefix}.gate.weight");
    let routing = TopKGroupSelectionSpec::new(
        args.num_experts,
        args.num_experts_per_tok,
        GroupScoring::Sigmoid,
        args.norm_topk_prob,
    )?
    .with_weight_policy(1e-6, args.routed_scaling_factor)?;
    let mut selector = TopKGroupSelectorSpec::new(
        args.hidden_size,
        ParameterSpec::trainable(&gate_name).map_err(Error::backend)?,
        crate::linear_format::standard_linear_format(
            &gate_name,
            args.weight_quantization_for(&gate_name).into(),
        )?,
        routing,
    )?
    // The released equation rounds projection, sigmoid, normalization and
    // mixture coefficients in the input dtype. The F32 correction bias
    // affects expert choice only; it must not promote the expert mixture.
    .with_arithmetic(eredu_nn::RoutingArithmetic::uniform(
        eredu_nn::RoutingPrecision::Input,
    ));
    if let Some(correction_bias) = args
        .use_expert_bias
        .then(|| ParameterSpec::trainable(format!("{prefix}.expert_bias")))
        .transpose()
        .map_err(Error::backend)?
    {
        selector = selector.with_correction_bias(correction_bias)?;
    }
    Ok(selector)
}

fn grouped_experts(args: &ModelArgs, layer: usize) -> Result<GroupedGatedProductSpec, Error> {
    args.validate().map_err(Error::backend_source)?;
    let policy = args
        .layer_policy(layer)
        .ok_or_else(|| Error::backend(format!("LFM2 has no layer {layer}")))?;
    if policy.feed_forward != FeedForwardPolicy::SparseMoe {
        return Err(Error::backend(format!(
            "LFM2 layer {layer} is not sparse MoE"
        )));
    }
    let experts_prefix = format!("model.layers.{layer}.feed_forward.experts");
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
    .map(|spec| spec.with_reduction(eredu_nn::GroupReduction::SequentialGroupOrder))
}

/// Derives expert ownership from retained banks and optional admitted local geometry.
/// This operation allocates no native modules and requires no backend context.
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
                geometry.expert_intermediate
            });
        let spec = bank.with_group_geometry(local_experts, width)?;
        unit_specs.insert((owner_group.clone(), layer), spec);
    }
    crate::ExpertRealizationPlan::balanced(global_experts, topology, unit_specs)
        .map(Some)
        .map_err(Error::backend)
}

/// Per-layer dense or routed feed-forward policy.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub enum FeedForward<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> {
    /// Dense SwiGLU.
    Dense(DenseSwiGlu<B>),
    /// Routed LFM2-MoE SwiGLU.
    Routed(RoutedGatedProduct<B>),
}

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> FeedForward<B> {
    pub(crate) fn from_parameters(
        layer: usize,
        parameters: super::declarations::FeedForwardParameters,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        match parameters {
            super::declarations::FeedForwardParameters::Dense(spec) => {
                DenseSwiGlu::from_parameters(spec, context).map(Self::Dense)
            }
            super::declarations::FeedForwardParameters::Routed(spec) => {
                RoutedGatedProduct::from_spec(layer, spec, context).map(Self::Routed)
            }
        }
    }

    /// Executes dense locally or delegates routed experts to one runtime provider.
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
        match self {
            Self::Dense(dense) => {
                let hidden = dense.hidden(input, context)?;
                dense.down.forward(&hidden, context)
            }
            Self::Routed(routed) => {
                let routes = routed.router.select(input, context)?;
                ParameterProvider::<B>::forward_grouped(
                    provider,
                    &mut routed.experts,
                    RoutedExpertRequest {
                        unit_observer: None,
                        bank: eredu_runtime::RoutedBankId::new(0),
                        layer: routed.layer,
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

    /// Preserves routed-provider ownership while exposing dense unit boundaries.
    pub fn forward_observed_with_provider<P>(
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
        match self {
            Self::Dense(dense) => dense.forward_instrumented(input, None, context, instrumentation),
            Self::Routed(_) => match (points, instrumentation.observer()) {
                (Some(points), Some(observer)) => {
                    eredu_runtime::ObservedExpertProvider::new(provider, observer, points)
                        .execute_neural(|provider| {
                            self.forward_with_provider(input, pass, context, provider)
                        })
                }
                _ => self.forward_with_provider(input, pass, context, provider),
            },
        }
    }

    /// Emits local dense components and preserves the routed provider's complete-output reduction.
    pub fn forward_parallel_observed_with_provider<P>(
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
        match self {
            Self::Dense(dense) => {
                dense.forward_instrumented(input, Some(parallel), context, instrumentation)
            }
            Self::Routed(_) => match (points, instrumentation.observer()) {
                (Some(points), Some(observer)) => {
                    eredu_runtime::ObservedExpertProvider::new(provider, observer, points)
                        .execute_neural(|provider| {
                            self.forward_parallel_with_provider(
                                input, pass, parallel, context, provider,
                            )
                        })
                }
                _ => self.forward_parallel_with_provider(input, pass, parallel, context, provider),
            },
        }
    }

    /// Executes a tensor-partitioned dense path or delegates routed experts.
    ///
    /// Resident providers return rank-local contributions while external
    /// providers may return complete hidden-width contributions.
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
        match self {
            Self::Dense(dense) => {
                let hidden = dense.hidden(input, context)?;
                B::row_parallel_linear(&mut dense.down, &hidden, parallel, context)
            }
            Self::Routed(routed) => {
                let routes = routed.router.select(input, context)?;
                let output = TensorParallelParameterProvider::<B>::forward_grouped_tensor_parallel(
                    provider,
                    &mut routed.experts,
                    RoutedExpertRequest {
                        unit_observer: None,
                        bank: eredu_runtime::RoutedBankId::new(0),
                        layer: routed.layer,
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
    fn forward_feed_forward_observed(
        &mut self,
        input: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut crate::decoder::ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error> {
        match self {
            Self::Dense(dense) => dense.forward_instrumented(input, None, context, instrumentation),
            Self::Routed(_) => self.forward_feed_forward(input, context),
        }
    }

    fn forward_feed_forward(
        &mut self,
        input: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        match self {
            Self::Dense(dense) => {
                let hidden = dense.hidden(input, context)?;
                dense.down.forward(&hidden, context)
            }
            Self::Routed(routed) => {
                let routes = routed.router.select(input, context)?;
                let mut provider = ResidentExpertProvider;
                ParameterProvider::<B>::forward_grouped(
                    &mut provider,
                    &mut routed.experts,
                    RoutedExpertRequest {
                        unit_observer: None,
                        bank: eredu_runtime::RoutedBankId::new(0),
                        layer: routed.layer,
                        input,
                        routes: &routes,
                        pass: if input.shape()[input.shape().len() - 2] > 1 {
                            eredu_runtime::ExpertPass::Prefill
                        } else {
                            eredu_runtime::ExpertPass::Decode
                        },
                    },
                    context,
                )
            }
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
            Self::Dense(dense) => {
                dense.forward_instrumented(input, Some(parallel), context, instrumentation)
            }
            Self::Routed(_) => self.forward_feed_forward_parallel(input, parallel, context),
        }
    }

    fn forward_feed_forward_parallel(
        &mut self,
        input: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        match self {
            Self::Dense(dense) => {
                let hidden = dense.hidden(input, context)?;
                B::row_parallel_linear(&mut dense.down, &hidden, parallel, context)
            }
            Self::Routed(routed) => {
                let routes = routed.router.select(input, context)?;
                let output = B::gated_product_groups_tensor_parallel(
                    &mut routed.experts,
                    input,
                    &routes,
                    B::parallel_size(parallel),
                    context,
                )?;
                eredu_runtime::reduce_tensor_parallel_expert_output::<B>(output, parallel, context)
            }
        }
    }
}
