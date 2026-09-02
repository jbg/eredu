//! Backend-neutral GPT-OSS biased routing and bounded gated-product experts.

use eredu_checkpoint::WeightQuantization;
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
        FeedForwardOperator, TensorParallelFeedForwardOperator,
        TensorParallelRoutedFeedForwardOperator,
    },
    linear_format::standard_expert_projection,
};

use super::config::ModelArgs;

#[cfg(test)]
fn expert_policy(limit: f32) -> Result<eredu_nn::GatedProductPolicy, Error> {
    eredu_nn::GatedProductPolicy::new(
        eredu_nn::GatedProductActivation::Silu,
        Some(limit),
        Some(limit),
        1.702,
        1.0,
    )
}

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
    crate::decoder::RoutedFeedForwardOperator<B> for RoutedMlp<B>
{
    fn forward_with_provider<P>(
        &mut self,
        _layer: usize,
        input: &B::Tensor,
        pass: ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        RoutedMlp::forward_with_provider(self, input, pass, provider, context)
    }
}

impl<B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend>
    TensorParallelRoutedFeedForwardOperator<B> for RoutedMlp<B>
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
        P: TensorParallelRoutedExpertProvider<B>,
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
        let router = B::top_k_group_selector(selector, context)?;

        let experts = B::grouped_gated_product(expert_bank_spec(args, layer)?, context)?;
        Ok(Self {
            layer,
            router,
            experts,
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
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        let routes = self.router.select(input, context)?;
        provider
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
            .map_err(Error::backend)
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
        P: TensorParallelRoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        let routes = self.router.select(input, context)?;
        let output = provider
            .forward_grouped_tensor_parallel(
                &mut self.experts,
                RoutedExpertRequest {
                    layer: self.layer,
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

/// Returns the architecture-owned routed expert specification for one layer.
pub fn expert_bank_spec(args: &ModelArgs, layer: usize) -> Result<GroupedGatedProductSpec, Error> {
    let expert_prefix = format!("{}.layers.{layer}.mlp.experts", args.parameter_root);
    let gate_up_weight = format!("{expert_prefix}.gate_up_proj");
    let gate_up_bias = format!("{expert_prefix}.gate_up_proj_bias");
    let down_weight = format!("{expert_prefix}.down_proj");
    let down_bias = format!("{expert_prefix}.down_proj_bias");
    GroupedGatedProductSpec::new(
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
            )?,
            down: standard_expert_projection(
                &down_weight,
                Some(&down_bias),
                WeightQuantization::MxFp4.into(),
            )?,
        },
    )
}

/// Returns the architecture-owned routed expert specification at rank-local geometry.
///
/// Parameter identities, native MXFP4 formats, biases, and gating policy remain
/// identical to the global bank; only placement-resolved geometry is localized.
pub(crate) fn localized_expert_bank_spec(
    args: &ModelArgs,
    layer: usize,
    expert_count: i32,
    intermediate_dimensions: i32,
) -> Result<GroupedGatedProductSpec, Error> {
    expert_bank_spec(args, layer)?.with_group_geometry(expert_count, intermediate_dimensions)
}

/// Derives the complete replicated expert plan directly from normalized geometry.
pub fn replicated_expert_realization_plan(
    args: &ModelArgs,
) -> Result<crate::ExpertRealizationPlan<GroupedGatedProductSpec>, Error> {
    let global_experts = usize::try_from(args.num_local_experts).map_err(Error::backend)?;
    let layers = usize::try_from(args.num_hidden_layers).map_err(Error::backend)?;
    let owner_group =
        eredu_runtime::ExecutionGroupId::new("text_decoder").map_err(Error::backend)?;
    let unit_specs = (0..layers)
        .map(|layer| expert_bank_spec(args, layer).map(|spec| ((owner_group.clone(), layer), spec)))
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

/// Derives complete expert ownership and rank-local bank geometry from GPT-OSS.
pub fn expert_realization_plan<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend>(
    architecture: &super::LayeredModel<B>,
    topology: eredu_core::ParallelRankTopology,
) -> Result<Option<crate::ExpertRealizationPlan<GroupedGatedProductSpec>>, Error> {
    let args = architecture.args();
    let global_experts = usize::try_from(args.num_local_experts).map_err(Error::backend)?;
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
    let mut unit_specs = std::collections::BTreeMap::new();
    for layer in 0..layers {
        let width = architecture
            .parallel_geometry()
            .and_then(|geometry| geometry.block(layer))
            .map_or(args.intermediate_size, |local| local.intermediate_size);
        unit_specs.insert(
            (owner_group.clone(), layer),
            localized_expert_bank_spec(args, layer, local_experts, width)?,
        );
    }
    crate::ExpertRealizationPlan::balanced(global_experts, topology, unit_specs)
        .map(Some)
        .map_err(Error::backend)
}

#[cfg(test)]
#[allow(
    clippy::items_after_test_module,
    reason = "the focused realization tests stay adjacent to the realization planner"
)]
mod tests {
    use super::*;

    fn args() -> ModelArgs {
        crate::gpt_oss::model_args_from_config_value(&serde_json::json!({
            "model_type": "gpt_oss",
            "hidden_size": 64,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "vocab_size": 128,
            "num_local_experts": 4,
            "num_experts_per_tok": 2,
            "rms_norm_eps": 1e-5,
            "sliding_window": 128,
            "max_position_embeddings": 4096,
            "rope_theta": 150000.0,
            "quantization_config": { "quant_method": "mxfp4" },
            "swiglu_limit": 7.0
        }))
        .unwrap()
    }

    #[test]
    fn gpt_oss_policy_preserves_published_gating_constants() {
        let policy = expert_policy(7.0).unwrap();
        assert_eq!(policy.activation(), eredu_nn::GatedProductActivation::Silu);
        assert_eq!(policy.gate_upper_bound(), Some(7.0));
        assert_eq!(policy.up_absolute_bound(), Some(7.0));
        assert_eq!(policy.sigmoid_multiplier(), 1.702);
        assert_eq!(policy.up_offset(), 1.0);
    }

    #[test]
    fn localized_spec_preserves_native_format_biases_and_policy() {
        let args = args();
        let global = expert_bank_spec(&args, 0).unwrap();
        let local = localized_expert_bank_spec(&args, 0, 2, 32).unwrap();
        assert_eq!(local.group_count(), 2);
        assert_eq!(local.intermediate_dimensions(), 32);
        assert_eq!(local.policy(), global.policy());
        let GatedProductGroupLayout::Packed {
            gate_up: global_gate_up,
            down: global_down,
        } = global.layout()
        else {
            panic!("GPT-OSS experts must be packed");
        };
        let GatedProductGroupLayout::Packed {
            gate_up: local_gate_up,
            down: local_down,
        } = local.layout()
        else {
            panic!("GPT-OSS experts must be packed");
        };
        assert_eq!(local_gate_up.weight(), global_gate_up.weight());
        assert_eq!(local_gate_up.bias(), global_gate_up.bias());
        assert_eq!(local_gate_up.format(), global_gate_up.format());
        assert_eq!(local_down.weight(), global_down.weight());
        assert_eq!(local_down.bias(), global_down.bias());
        assert_eq!(local_down.format(), global_down.format());
    }
}

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> FeedForwardOperator<B>
    for RoutedMlp<B>
{
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
    TensorParallelFeedForwardOperator<B> for RoutedMlp<B>
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
