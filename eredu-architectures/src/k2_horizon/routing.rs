//! Two independently specified K2 routing equations.
use super::{ExpertBank, ModelArgs};
use eredu_nn::{
    DeclareLogicalParameters, Error, GatedProductGroupLayout, GatedProductPolicy,
    GroupedGatedProductSpec, GroupedLinearActivation, GroupedLinearSpec, GroupedProjectionSpec,
    ParameterSpec, TopKGroupSelectorSpec,
};

fn root(layer: usize, bank: ExpertBank) -> String {
    match bank {
        ExpertBank::AttentionValue => format!("model.layers.{layer}.self_attn.v_router"),
        ExpertBank::FeedForward => format!("model.layers.{layer}.mlp.gate"),
    }
}

/// Declares global router parameters and selection-only correction bias.
/// Bias never enters the projection logits or gathered mixture coefficients.
pub(super) fn router_spec(
    args: &ModelArgs,
    layer: usize,
    bank: ExpertBank,
) -> Result<TopKGroupSelectorSpec, Error> {
    args.validate().map_err(Error::backend_source)?;
    if !(match bank {
        ExpertBank::AttentionValue => args.is_mova_layer(layer),
        ExpertBank::FeedForward => args.is_sparse_layer(layer),
    }) {
        return Err(Error::backend(
            "router bank is absent from this K2 invocation",
        ));
    }
    let root = root(layer, bank);
    let name = format!("{root}.weight");
    let parameter = |name: &str| {
        let mut parameter = ParameterSpec::trainable(name).map_err(Error::backend)?;
        parameter.group = Some(root.clone());
        Ok::<_, Error>(parameter)
    };
    let mut spec = TopKGroupSelectorSpec::new(
        args.hidden_size,
        parameter(&name)?,
        crate::linear_format::standard_linear_format(&name, args.linear_format_for(&name))?,
        args.routing_spec(bank)?,
    )?;
    spec = spec.with_arithmetic(eredu_nn::RoutingArithmetic {
        projection: eredu_nn::RoutingPrecision::Input,
        scores: eredu_nn::RoutingPrecision::Float32,
        coefficients: eredu_nn::RoutingPrecision::Input,
    });
    if args.moe_gate_bias {
        spec = spec.with_correction_bias(parameter(&format!("{root}.bias"))?)?;
    }
    eredu_nn::validate_logical_parameter_topology(&spec.logical_parameters()?)
        .map_err(Error::backend_source)?;
    Ok(spec)
}

/// Value-bank specification with SiLU before mixture weighting. Output rows
/// may be partitioned by KV head while every expert retains full hidden input.
pub(super) fn value_expert_spec(
    args: &ModelArgs,
    layer: usize,
) -> Result<GroupedLinearSpec, Error> {
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
pub(super) fn feed_forward_expert_spec(
    args: &ModelArgs,
    layer: usize,
) -> Result<GroupedGatedProductSpec, Error> {
    if !args.is_sparse_layer(layer) {
        return Err(Error::backend(
            "layer does not declare feed-forward experts",
        ));
    }
    let root = format!("model.layers.{layer}.mlp.experts");
    let projection = |field: &str, row_layout| {
        let name = format!("{root}.{field}");
        crate::linear_format::standard_expert_projection(
            &name,
            None,
            args.linear_format_for(&name),
            row_layout,
        )
    };
    GroupedGatedProductSpec::new(
        args.num_experts,
        args.hidden_size,
        args.moe_intermediate_size,
        args.hidden_size,
        GatedProductPolicy::ordinary_silu(),
        GatedProductGroupLayout::Packed {
            gate_up: projection(
                "gate_up_proj",
                eredu_nn::LinearRowLayout::equal_partitions(2)?,
            )?,
            down: projection("down_proj", eredu_nn::LinearRowLayout::Contiguous)?,
        },
    )
    .map(|spec| spec.with_reduction(eredu_nn::GroupReduction::SequentialGroupOrder))
}
