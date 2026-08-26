//! Dense and routed LFM2 feed-forward policies.

use eredu_nn::{
    Error, GatedProductExpertBankOperator, GatedProductExpertBankSpec, GatedProductExpertLayout,
    LinearOperator, LinearSpec, ParameterSpec, Parameterized, RoutedNeuralBackend, RoutingOperator,
    RoutingScoring, Tensor, TopKRouterSpec, TopKRoutingSpec,
};
use eredu_runtime::{ResidentExpertProvider, RoutedExpertProvider, RoutedExpertRequest};

use crate::{decoder::FeedForwardOperator, linear_format::standard_expert_projection};

use super::{FeedForwardPolicy, ModelArgs};

/// Dense LFM2 SwiGLU with checkpoint-compatible projection identities.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct DenseSwiGlu<B: RoutedNeuralBackend> {
    /// Gate projection (`w1`).
    pub gate: B::Linear,
    /// Down projection (`w2`).
    pub down: B::Linear,
    /// Up projection (`w3`).
    pub up: B::Linear,
}

impl<B: RoutedNeuralBackend> DenseSwiGlu<B> {
    fn new(
        args: &ModelArgs,
        layer: usize,
        intermediate: i32,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let prefix = format!("model.layers.{layer}.feed_forward");
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
            gate: linear("w1", args.hidden_size, intermediate)?,
            down: linear("w2", intermediate, args.hidden_size)?,
            up: linear("w3", args.hidden_size, intermediate)?,
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
}

/// Sigmoid router and routed gated-product expert bank used by LFM2-MoE.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct RoutedGatedProduct<B: RoutedNeuralBackend> {
    /// Global physical layer identity.
    #[parameter(skip)]
    pub layer: usize,
    /// Learned sigmoid top-k router.
    pub router: B::Router,
    /// Packed routed experts.
    pub experts: B::GatedProductExpertBank,
}

impl<B: RoutedNeuralBackend> RoutedGatedProduct<B> {
    fn new(
        args: &ModelArgs,
        layer: usize,
        intermediate: i32,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let prefix = format!("model.layers.{layer}.feed_forward");
        let gate_name = format!("{prefix}.gate.weight");
        let routing = TopKRoutingSpec::new(
            args.num_experts,
            args.num_experts_per_tok,
            RoutingScoring::Sigmoid,
            args.norm_topk_prob,
        )?
        .with_weight_policy(1e-6, args.routed_scaling_factor)?;
        let router = B::top_k_router(
            TopKRouterSpec {
                input_dimensions: args.hidden_size,
                weight: ParameterSpec::trainable(&gate_name).map_err(Error::backend)?,
                bias: None,
                correction_bias: args
                    .use_expert_bias
                    .then(|| ParameterSpec::trainable(format!("{prefix}.expert_bias")))
                    .transpose()
                    .map_err(Error::backend)?,
                input_transform: None,
                route_scale: None,
                format: crate::linear_format::standard_linear_format(
                    &gate_name,
                    args.weight_quantization_for(&gate_name).into(),
                )?,
                routing,
            },
            context,
        )?;
        let experts = B::gated_product_expert_bank(
            expert_bank_spec_with_width(args, layer, intermediate)?,
            context,
        )?;
        Ok(Self {
            layer,
            router,
            experts,
        })
    }
}

/// Returns the architecture-owned routed expert specification for one layer.
pub fn expert_bank_spec(
    args: &ModelArgs,
    layer: usize,
) -> Result<GatedProductExpertBankSpec, Error> {
    expert_bank_spec_with_width(args, layer, args.moe_intermediate_size)
}

fn expert_bank_spec_with_width(
    args: &ModelArgs,
    layer: usize,
    intermediate: i32,
) -> Result<GatedProductExpertBankSpec, Error> {
    let experts_prefix = format!("model.layers.{layer}.feed_forward.experts");
    let gate_up_name = format!("{experts_prefix}.gate_up_proj");
    let down_name = format!("{experts_prefix}.down_proj");
    Ok(GatedProductExpertBankSpec {
        expert_count: args.num_experts,
        input_dimensions: args.hidden_size,
        intermediate_dimensions: intermediate,
        output_dimensions: args.hidden_size,
        policy: eredu_nn::GatedProductPolicy::ordinary_silu(),
        layout: GatedProductExpertLayout::Packed {
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
    })
}

/// Derives complete expert ownership and rank-local bank geometry from LFM2.
pub fn expert_realization_plan<B: RoutedNeuralBackend>(
    architecture: &super::LayeredModel<B>,
    topology: eredu_core::ParallelRankTopology,
) -> Result<Option<crate::ExpertRealizationPlan<GatedProductExpertBankSpec>>, Error> {
    let args = architecture.args();
    if !args.has_sparse_moe_layers() {
        return Ok(None);
    }
    let global_experts = usize::try_from(args.num_experts).map_err(Error::backend)?;
    let local_experts = i32::try_from(
        eredu_core::balanced_contiguous_range(
            global_experts,
            topology.expert_parallel_size,
            topology.expert_parallel_rank,
            false,
        )
        .map_err(Error::backend)?
        .len(),
    )
    .map_err(Error::backend)?;
    let owner_group =
        eredu_runtime::ExecutionGroupId::new("text_decoder").map_err(Error::backend)?;
    let mut unit_specs = std::collections::BTreeMap::new();
    for (layer, policy) in args.layer_schedule.iter().enumerate() {
        if policy.feed_forward != FeedForwardPolicy::SparseMoe {
            continue;
        }
        let width = architecture
            .parallel_geometry()
            .and_then(|geometry| geometry.block(layer))
            .map_or(args.moe_intermediate_size, |geometry| {
                geometry.expert_intermediate
            });
        let mut spec = expert_bank_spec_with_width(args, layer, width)?;
        spec.expert_count = local_experts;
        spec.validate()?;
        unit_specs.insert((owner_group.clone(), layer), spec);
    }
    crate::ExpertRealizationPlan::balanced(global_experts, topology, unit_specs)
        .map(Some)
        .map_err(Error::backend)
}

/// Per-layer dense or routed feed-forward policy.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub enum FeedForward<B: RoutedNeuralBackend> {
    /// Dense SwiGLU.
    Dense(DenseSwiGlu<B>),
    /// Routed LFM2-MoE SwiGLU.
    Routed(RoutedGatedProduct<B>),
}

impl<B: RoutedNeuralBackend> FeedForward<B> {
    /// Builds the exact scheduled feed-forward operator.
    pub fn new(
        args: &ModelArgs,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        Self::new_with_geometry(
            args,
            layer,
            args.dense_intermediate_size,
            args.moe_intermediate_size,
            context,
        )
    }

    /// Builds the scheduled operator from placement-resolved local widths.
    pub fn new_with_geometry(
        args: &ModelArgs,
        layer: usize,
        dense_intermediate: i32,
        expert_intermediate: i32,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        match args
            .layer_policy(layer)
            .ok_or_else(|| Error::backend(format!("LFM2 has no layer {layer}")))?
            .feed_forward
        {
            FeedForwardPolicy::Dense => {
                DenseSwiGlu::new(args, layer, dense_intermediate, context).map(Self::Dense)
            }
            FeedForwardPolicy::SparseMoe => {
                RoutedGatedProduct::new(args, layer, expert_intermediate, context).map(Self::Routed)
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
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        match self {
            Self::Dense(dense) => {
                let hidden = dense.hidden(input, context)?;
                dense.down.forward(&hidden, context)
            }
            Self::Routed(routed) => {
                let routes = routed.router.route(input, context)?;
                RoutedExpertProvider::<B>::forward_routed(
                    provider,
                    &mut routed.experts,
                    RoutedExpertRequest {
                        layer: routed.layer,
                        input,
                        routes: &routes,
                        pass,
                    },
                    context,
                )
                .map_err(|error| Error::backend(error.to_string()))
            }
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
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        match self {
            Self::Dense(dense) => {
                let hidden = dense.hidden(input, context)?;
                B::row_parallel_linear(&mut dense.down, &hidden, parallel, context)
            }
            Self::Routed(routed) => {
                let routes = routed.router.route(input, context)?;
                let output = RoutedExpertProvider::<B>::forward_routed_tensor_parallel(
                    provider,
                    &mut routed.experts,
                    RoutedExpertRequest {
                        layer: routed.layer,
                        input,
                        routes: &routes,
                        pass,
                    },
                    B::parallel_size(parallel),
                    context,
                )
                .map_err(|error| Error::backend(error.to_string()))?;
                eredu_runtime::reduce_routed_expert_tensor_parallel::<B>(output, parallel, context)
            }
        }
    }
}

impl<B: RoutedNeuralBackend> FeedForwardOperator<B> for FeedForward<B> {
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
                let routes = routed.router.route(input, context)?;
                let mut provider = ResidentExpertProvider;
                RoutedExpertProvider::<B>::forward_routed(
                    &mut provider,
                    &mut routed.experts,
                    RoutedExpertRequest {
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
                let routes = routed.router.route(input, context)?;
                let output = routed.experts.forward_routed_tensor_parallel(
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
