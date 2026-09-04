//! Qwen dense and routed gated-product feed-forward policy.

use std::collections::BTreeMap;

use eredu_nn::{
    Error, GatedProductGroupLayout, GroupScoring, GroupSelectionOperator, GroupedGatedProductSpec,
    GroupedNeuralBackend, ParameterSpec, Parameterized, Tensor, TopKGroupSelectionSpec,
    TopKGroupSelectorSpec,
};
use eredu_runtime::{
    ExpertPass, ResidentExpertProvider, RoutedExpertProvider, RoutedExpertRequest,
    TensorParallelRoutedExpertProvider,
};

use crate::{
    decoder::{
        FeedForwardOperator, Mlp, TensorParallelFeedForwardOperator,
        TensorParallelRoutedFeedForwardOperator,
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
        if !args.is_moe() {
            return Err(Error::backend("routed Qwen block requires Qwen3-MoE args"));
        }
        let prefix = format!("{}.layers.{layer}.mlp", args.parameter_root);
        let routing = TopKGroupSelectionSpec::new(
            args.num_experts,
            args.num_experts_per_tok,
            GroupScoring::Softmax,
            args.norm_topk_prob,
        )?;
        let router_name = format!("{prefix}.gate.weight");
        let router = B::top_k_group_selector(
            TopKGroupSelectorSpec::new(
                args.hidden_size,
                ParameterSpec::trainable(&router_name).map_err(Error::backend)?,
                crate::linear_format::standard_linear_format(
                    &router_name,
                    args.weight_quantization_for(&router_name).into(),
                )?,
                routing,
            )?,
            context,
        )?;
        let experts = B::grouped_gated_product(expert_bank_spec(args, layer)?, context)?;
        Ok(Self {
            layer,
            router,
            experts,
        })
    }

    /// Builds a partition-local bank while preserving the global router axis.
    pub fn new_partitioned(
        global: &ModelArgs,
        local: &ModelArgs,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        if !global.is_moe() || !local.is_moe() {
            return Err(Error::backend(
                "partitioned routed Qwen construction requires Qwen3-MoE",
            ));
        }
        let prefix = format!("{}.layers.{layer}.mlp", global.parameter_root);
        let routing = TopKGroupSelectionSpec::new(
            global.num_experts,
            global.num_experts_per_tok,
            GroupScoring::Softmax,
            global.norm_topk_prob,
        )?;
        let router_name = format!("{prefix}.gate.weight");
        let router = B::top_k_group_selector(
            TopKGroupSelectorSpec::new(
                global.hidden_size,
                ParameterSpec::trainable(&router_name).map_err(Error::backend)?,
                crate::linear_format::standard_linear_format(
                    &router_name,
                    global.weight_quantization_for(&router_name).into(),
                )?,
                routing,
            )?,
            context,
        )?;
        let experts = B::grouped_gated_product(
            localized_expert_bank_spec(
                global,
                layer,
                local.num_experts,
                local.moe_intermediate_size,
            )?,
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
pub fn expert_bank_spec(args: &ModelArgs, layer: usize) -> Result<GroupedGatedProductSpec, Error> {
    let experts_prefix = format!("{}.layers.{layer}.mlp.experts", args.parameter_root);
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
            )?,
            down: standard_expert_projection(
                &down_name,
                None,
                args.weight_quantization_for(&down_name).into(),
            )?,
        },
    )
}

/// Returns the architecture-owned routed expert specification at rank-local geometry.
///
/// Parameter identities, physical formats, and execution policy remain identical
/// to the global bank; only placement-resolved cardinality and width are localized.
pub(crate) fn localized_expert_bank_spec(
    args: &ModelArgs,
    layer: usize,
    expert_count: i32,
    intermediate_dimensions: i32,
) -> Result<GroupedGatedProductSpec, Error> {
    expert_bank_spec(args, layer)?.with_group_geometry(expert_count, intermediate_dimensions)
}

/// Derives complete expert ownership and rank-local bank geometry from Qwen.
///
/// The normalized architecture and its optional planner-derived geometry are
/// the only inputs that may determine expert cardinality or localized width.
pub fn expert_realization_plan<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend>(
    architecture: &super::RoutedLayeredModel<B>,
    topology: eredu_core::ParallelRankTopology,
) -> Result<Option<crate::ExpertRealizationPlan<GroupedGatedProductSpec>>, Error> {
    let args = architecture.args();
    if !args.is_moe() {
        return Ok(None);
    }
    let global_experts = usize::try_from(args.num_experts).map_err(Error::backend)?;
    let local_range = eredu_core::balanced_contiguous_range(
        global_experts,
        topology.expert_parallel_size(),
        topology.expert_parallel_rank(),
        false,
    )
    .map_err(Error::backend)?;
    let local_experts = i32::try_from(local_range.len()).map_err(Error::backend)?;
    let layers = usize::try_from(args.num_hidden_layers).map_err(Error::backend)?;
    let owner_group =
        eredu_runtime::ExecutionGroupId::new("text_decoder").map_err(Error::backend)?;
    let mut unit_specs = BTreeMap::new();
    for layer in 0..layers {
        let local_args = architecture
            .parallel_geometry()
            .and_then(|geometry| geometry.block(layer))
            .unwrap_or(args);
        unit_specs.insert(
            (owner_group.clone(), layer),
            localized_expert_bank_spec(
                args,
                layer,
                local_experts,
                local_args.moe_intermediate_size,
            )?,
        );
    }
    crate::ExpertRealizationPlan::balanced(global_experts, topology, unit_specs)
        .map(Some)
        .map_err(Error::backend)
}

/// Derives exact TP×EP-local expert geometry before native construction.
pub fn partition_expert_realization_plan(
    args: &ModelArgs,
    layout: &eredu_runtime::LocalModelLayout,
    topology: eredu_core::ParallelRankTopology,
) -> Result<crate::ExpertRealizationPlan<GroupedGatedProductSpec>, Error> {
    if !args.is_moe() {
        return Err(Error::backend(
            "partitioned expert planning requires routed Qwen geometry",
        ));
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
    let layers = usize::try_from(args.num_hidden_layers).map_err(Error::backend)?;
    let owner_group =
        eredu_runtime::ExecutionGroupId::new("text_decoder").map_err(Error::backend)?;
    let unit_specs = (0..layers)
        .map(|layer| {
            let local =
                super::parallel::local_block_args(args, layer, layout).map_err(Error::backend)?;
            localized_expert_bank_spec(args, layer, local_experts, local.moe_intermediate_size)
                .map(|spec| ((owner_group.clone(), layer), spec))
        })
        .collect::<Result<BTreeMap<_, _>, _>>()?;
    crate::ExpertRealizationPlan::balanced(global_experts, topology, unit_specs)
        .map_err(Error::backend)
}

/// Derives the complete replicated expert plan before architecture construction.
pub fn replicated_expert_realization_plan(
    args: &ModelArgs,
) -> Result<crate::ExpertRealizationPlan<GroupedGatedProductSpec>, Error> {
    if !args.is_moe() {
        return Err(Error::backend(
            "replicated expert planning requires a routed Qwen configuration",
        ));
    }
    let global_experts = usize::try_from(args.num_experts).map_err(Error::backend)?;
    let layers = usize::try_from(args.num_hidden_layers).map_err(Error::backend)?;
    let owner_group =
        eredu_runtime::ExecutionGroupId::new("text_decoder").map_err(Error::backend)?;
    let unit_specs = (0..layers)
        .map(|layer| expert_bank_spec(args, layer).map(|spec| ((owner_group.clone(), layer), spec)))
        .collect::<Result<BTreeMap<_, _>, _>>()?;
    let topology = eredu_core::ParallelRankTopology::new(
        eredu_core::ParallelTopology::new(1, 1, 1, 1).map_err(Error::backend)?,
        0,
    )
    .map_err(Error::backend)?;
    crate::ExpertRealizationPlan::balanced(global_experts, topology, unit_specs)
        .map_err(Error::backend)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn localized_spec_preserves_architecture_parameter_policy() {
        let args = crate::qwen::model_args_from_config_value(&serde_json::json!({
            "model_type": "qwen3_moe",
            "hidden_size": 16,
            "num_hidden_layers": 2,
            "intermediate_size": 0,
            "moe_intermediate_size": 8,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "norm_topk_prob": true,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 4,
            "rms_norm_eps": 0.000001,
            "vocab_size": 64,
            "max_position_embeddings": 128,
            "tie_word_embeddings": false
        }))
        .unwrap();
        let global = expert_bank_spec(&args, 1).unwrap();
        let local = localized_expert_bank_spec(&args, 1, 2, 4).unwrap();
        assert_eq!(local.group_count(), 2);
        assert_eq!(local.intermediate_dimensions(), 4);
        assert_eq!(local.policy(), global.policy());
        let GatedProductGroupLayout::Packed {
            gate_up: global_gate_up,
            down: global_down,
        } = global.layout()
        else {
            panic!("Qwen experts must be packed");
        };
        let GatedProductGroupLayout::Packed {
            gate_up: local_gate_up,
            down: local_down,
        } = local.layout()
        else {
            panic!("Qwen experts must be packed");
        };
        assert_eq!(local_gate_up.weight(), global_gate_up.weight());
        assert_eq!(local_gate_up.format(), global_gate_up.format());
        assert_eq!(local_down.weight(), global_down.weight());
        assert_eq!(local_down.format(), global_down.format());
    }
}

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> FeedForwardOperator<B>
    for RoutedGatedProduct<B>
{
    fn forward_feed_forward(
        &mut self,
        input: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let routes = self.router.select(input, context)?;
        let mut provider = ResidentExpertProvider;
        RoutedExpertProvider::<B>::forward_grouped(
            &mut provider,
            &mut self.experts,
            RoutedExpertRequest {
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
    TensorParallelFeedForwardOperator<B> for RoutedGatedProduct<B>
{
    fn forward_feed_forward_parallel(
        &mut self,
        input: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let routes = self.router.select(input, context)?;
        let mut provider = ResidentExpertProvider;
        let output = TensorParallelRoutedExpertProvider::<B>::forward_grouped_tensor_parallel(
            &mut provider,
            &mut self.experts,
            RoutedExpertRequest {
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
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        match self {
            Self::Dense(mlp) => mlp.forward_feed_forward(input, context),
            Self::Routed(moe) => {
                let routes = moe.router.select(input, context)?;
                provider
                    .forward_grouped(
                        &mut moe.experts,
                        RoutedExpertRequest {
                            layer,
                            input,
                            routes: &routes,
                            pass,
                        },
                        context,
                    )
                    .map_err(Error::backend)
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
        P: TensorParallelRoutedExpertProvider<B>,
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
                            layer,
                            input,
                            routes: &routes,
                            pass,
                        },
                        B::parallel_size(parallel),
                        context,
                    )
                    .map_err(Error::backend)?;
                eredu_runtime::reduce_routed_expert_tensor_parallel::<B>(output, parallel, context)
            }
        }
    }
}

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> FeedForwardOperator<B>
    for FeedForward<B>
{
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
}

impl<B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend>
    TensorParallelFeedForwardOperator<B> for FeedForward<B>
{
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
    crate::decoder::RoutedFeedForwardOperator<B> for FeedForward<B>
{
    fn forward_with_provider<P>(
        &mut self,
        layer: usize,
        input: &B::Tensor,
        pass: ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        FeedForward::forward_with_provider(self, layer, pass, input, context, provider)
    }
}

impl<B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend>
    TensorParallelRoutedFeedForwardOperator<B> for FeedForward<B>
{
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
        P: TensorParallelRoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        FeedForward::forward_with_provider_parallel(
            self, layer, pass, input, parallel, context, provider,
        )
    }
}
