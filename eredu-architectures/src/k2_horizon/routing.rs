//! Two independently specified K2 routing equations.
use super::{ExpertBank, ModelArgs};
use eredu_nn::{
    Error, GatedProductGroupLayout, GatedProductPolicy, GroupedGatedProductSpec,
    GroupedLinearActivation, GroupedLinearSpec, GroupedNeuralBackend, GroupedProjectionSpec,
    ParameterSpec, Tensor, TopKGroupSelectorSpec,
};

fn root(layer: usize, bank: ExpertBank) -> String {
    match bank {
        ExpertBank::AttentionValue => format!("model.layers.{layer}.self_attn.v_router"),
        ExpertBank::FeedForward => format!("model.layers.{layer}.mlp.gate"),
    }
}

/// Constructs a global router with a selection-only correction bias. Bias is
/// never added to the linear logits or gathered mixture coefficients.
pub fn new_router<B: GroupedNeuralBackend>(
    args: &ModelArgs,
    layer: usize,
    bank: ExpertBank,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<B::Selector, Error> {
    let root = root(layer, bank);
    let name = format!("{root}.weight");
    let mut spec = TopKGroupSelectorSpec::new(
        args.hidden_size,
        ParameterSpec::trainable(&name).map_err(Error::backend)?,
        crate::linear_format::standard_linear_format(&name, args.linear_format_for(&name))?,
        args.routing_spec(bank)?,
    )?;
    spec = spec.with_arithmetic(eredu_nn::RoutingArithmetic {
        projection: eredu_nn::RoutingPrecision::Input,
        scores: eredu_nn::RoutingPrecision::Float32,
        coefficients: eredu_nn::RoutingPrecision::Input,
    });
    if args.moe_gate_bias {
        spec = spec.with_correction_bias(
            ParameterSpec::trainable(format!("{root}.bias")).map_err(Error::backend)?,
        )?;
    }
    B::top_k_group_selector(spec, context)
}

/// Value-bank specification with SiLU before mixture weighting. Output rows
/// may be partitioned by KV head while every expert retains full hidden input.
pub fn value_expert_spec(args: &ModelArgs, layer: usize) -> Result<GroupedLinearSpec, Error> {
    if !args.is_mova_layer(layer) {
        return Err(Error::backend("layer does not declare a value expert bank"));
    }
    let name = format!("model.layers.{layer}.self_attn.v_experts.weight");
    GroupedLinearSpec::new(
        args.mova_num_experts,
        args.hidden_size,
        args.num_key_value_heads * args.head_dim,
        GroupedLinearActivation::Silu,
        GroupedProjectionSpec::new(
            ParameterSpec::trainable(&name).map_err(Error::backend)?,
            None,
            crate::linear_format::standard_linear_format(&name, args.linear_format_for(&name))?,
        )?,
    )
    .map(|spec| spec.with_reduction(eredu_nn::GroupReduction::SequentialGroupOrder))
}

/// Canonical packed SwiGLU bank consumed by ordinary expert providers.
pub fn feed_forward_expert_spec(
    args: &ModelArgs,
    layer: usize,
) -> Result<GroupedGatedProductSpec, Error> {
    if !args.is_sparse_layer(layer) {
        return Err(Error::backend(
            "layer does not declare feed-forward experts",
        ));
    }
    let root = format!("model.layers.{layer}.mlp.experts");
    let projection = |field: &str| {
        let name = format!("{root}.{field}");
        crate::linear_format::standard_expert_projection(&name, None, args.linear_format_for(&name))
    };
    GroupedGatedProductSpec::new(
        args.num_experts,
        args.hidden_size,
        args.moe_intermediate_size,
        args.hidden_size,
        GatedProductPolicy::ordinary_silu(),
        GatedProductGroupLayout::Packed {
            gate_up: projection("gate_up_proj")?,
            down: projection("down_proj")?,
        },
    )
    .map(|spec| spec.with_reduction(eredu_nn::GroupReduction::SequentialGroupOrder))
}
