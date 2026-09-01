//! Kimi dense-prefix and grouped routed-expert feed-forward operators.

use eredu_nn::{
    Error, GatedProductGroupLayout, GroupScoring, GroupSelectionOperator,
    GroupedGatedProductOperator, GroupedGatedProductSpec, GroupedNeuralBackend, LinearOperator,
    LinearSpec, ParameterSpec, Parameterized, Tensor, TopKGroupSelectionSpec,
    TopKGroupSelectorSpec,
};
use eredu_runtime::{ResidentExpertProvider, RoutedExpertProvider, RoutedExpertRequest};

use crate::{decoder::FeedForwardOperator, linear_format::standard_expert_projection};

use super::{FeedForwardPolicy, ModelArgs};

/// Checkpoint-compatible dense SwiGLU used by dense and shared paths.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct DenseSwiGlu<B: GroupedNeuralBackend> {
    /// Gate projection.
    pub gate: B::Linear,
    /// Down projection.
    pub down: B::Linear,
    /// Up projection.
    pub up: B::Linear,
}

impl<B: GroupedNeuralBackend> DenseSwiGlu<B> {
    fn new(
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

    fn forward(
        &mut self,
        input: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let hidden = self.hidden(input, context)?;
        self.down.forward(&hidden, context)
    }
}

/// Grouped sigmoid router, routed bank, and one shared expert.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct SparseMoe<B: GroupedNeuralBackend> {
    #[parameter(skip)]
    layer: usize,
    /// Grouped sigmoid router with selection correction bias.
    pub router: B::Selector,
    /// Packed routed gated-product experts.
    pub experts: B::GatedProductGroups,
    /// Always-executed shared gated-product expert.
    pub shared: DenseSwiGlu<B>,
}

impl<B: GroupedNeuralBackend> SparseMoe<B> {
    fn new(
        args: &ModelArgs,
        layer: usize,
        routed_intermediate: i32,
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
        let experts = B::grouped_gated_product(
            expert_bank_spec_with_width(args, layer, routed_intermediate)?,
            context,
        )?;
        Ok(Self {
            layer,
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
    B: GroupedNeuralBackend + eredu_nn::BlockwiseAttentionBackend,
{
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

/// Per-layer dense-prefix or sparse Kimi feed-forward policy.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub enum FeedForward<B: GroupedNeuralBackend> {
    /// Dense prefix SwiGLU.
    Dense(DenseSwiGlu<B>),
    /// Routed plus shared experts.
    Sparse(SparseMoe<B>),
}

impl<B: GroupedNeuralBackend> FeedForward<B> {
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
        let policy = args
            .layer_policy(layer)
            .ok_or_else(|| Error::backend(format!("Kimi Linear has no layer {layer}")))?;
        let prefix = format!("model.layers.{layer}.mlp");
        match policy.feed_forward {
            FeedForwardPolicy::Dense => {
                DenseSwiGlu::new(args, &prefix, dense_intermediate, context).map(Self::Dense)
            }
            FeedForwardPolicy::SparseMoe => SparseMoe::new(
                args,
                layer,
                routed_intermediate,
                shared_intermediate,
                context,
            )
            .map(Self::Sparse),
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
        match self {
            Self::Dense(dense) => dense.forward(input, context),
            Self::Sparse(sparse) => {
                let routes = sparse.router.select(input, context)?;
                let routed = provider
                    .forward_grouped(
                        &mut sparse.experts,
                        RoutedExpertRequest {
                            layer: sparse.layer,
                            input,
                            routes: &routes,
                            pass,
                        },
                        context,
                    )
                    .map_err(|error| Error::backend(error.to_string()))?;
                routed.add(&sparse.shared.forward(input, context)?, context)
            }
        }
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
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        match self {
            Self::Dense(dense) => {
                let hidden = dense.hidden(input, context)?;
                B::row_parallel_linear(&mut dense.down, &hidden, parallel, context)
            }
            Self::Sparse(sparse) => {
                let routes = sparse.router.select(input, context)?;
                let routed = provider
                    .forward_grouped_tensor_parallel(
                        &mut sparse.experts,
                        RoutedExpertRequest {
                            layer: sparse.layer,
                            input,
                            routes: &routes,
                            pass,
                        },
                        B::parallel_size(parallel),
                        context,
                    )
                    .map_err(|error| Error::backend(error.to_string()))?;
                let routed = eredu_runtime::reduce_routed_expert_tensor_parallel::<B>(
                    routed, parallel, context,
                )?;
                let shared_hidden = sparse.shared.hidden(input, context)?;
                let shared = B::row_parallel_linear(
                    &mut sparse.shared.down,
                    &shared_hidden,
                    parallel,
                    context,
                )?;
                routed.add(&shared, context)
            }
        }
    }
}

impl<B: GroupedNeuralBackend> FeedForwardOperator<B> for FeedForward<B> {
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
            Self::Sparse(sparse) => {
                let routes = sparse.router.select(input, context)?;
                let routed = sparse.experts.forward_grouped_tensor_parallel(
                    input,
                    &routes,
                    B::parallel_size(parallel),
                    context,
                )?;
                let routed = eredu_runtime::reduce_tensor_parallel_expert_output::<B>(
                    routed, parallel, context,
                )?;
                let shared_hidden = sparse.shared.hidden(input, context)?;
                let shared = B::row_parallel_linear(
                    &mut sparse.shared.down,
                    &shared_hidden,
                    parallel,
                    context,
                )?;
                routed.add(&shared, context)
            }
        }
    }
}
