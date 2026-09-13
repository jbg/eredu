//! Always-executed ReLU² units nested within a Nemotron sparse write.
use super::*;

pub(in crate::discovery) mod prediction;

pub(in crate::discovery) fn shared_units(
    g: &mut Builder,
    c: &crate::nemotron_h::ModelArgs,
    layer: usize,
    path: &str,
    operator: &str,
) {
    shared_units_at(g, c, layer, path, operator, &format!("{path}.moe"));
}

fn shared_units_at(
    g: &mut Builder,
    c: &crate::nemotron_h::ModelArgs,
    layer: usize,
    path: &str,
    operator: &str,
    parameter_root: &str,
) {
    let node = format!("{operator}.shared");
    let root = format!("{parameter_root}.shared_experts");
    let parameter_group = format!("parameters:{parameter_root}");
    let scope = format!("{path}.shared.feed_forward");
    let width = c.moe_shared_expert_intermediate_size as usize;
    let read_weight = format!("{root}.up_proj.weight");
    let write_weight = format!("{root}.down_proj.weight");
    g.descriptor.components.push(ComponentGroup {
        id: format!("{node}.units"),
        node_id: node.clone(),
        layer_index: layer,
        count: width,
        activation: format!("{scope}.units"),
        effective_activation: format!("{scope}.units.effective"),
        write_input: Some(format!("{scope}.write_input")),
        write_output: Some(format!("{scope}.write")),
        output: Some(format!("{scope}.output")),
        write_partition: ComponentWritePartition::Complete,
        input: format!("{scope}.input"),
        reads: vec![ComponentRead {
            source: None,
            projection_output: None,
            role: ComponentReadRole::Input,
            weight: read_weight.clone(),
            shared_weight: read_weight,
            parameter_group: parameter_group.clone(),
            bias: c.mlp_bias.then(|| format!("{root}.up_proj.bias")),
            rows: ComponentRowMapping::Direct { offset: 0 },
            head_normalization: None,
            input_projections: Vec::new(),
        }],
        routed_reads: vec![],
        write_weight: write_weight.clone(),
        write_input_projection: None,
        shared_write_weight: write_weight,
        write_parameter_group: parameter_group,
        write_bias: c.mlp_bias.then(|| format!("{root}.down_proj.bias")),
        activation_equation: ComponentActivation::Unary {
            activation: crate::decoder::unary::Activation::ReluSquared.component(),
        },
        input_normalization: ComponentNormalization {
            kind: ComponentNormalizationKind::Rms,
            epsilon: ComponentScalar::new(c.layer_norm_epsilon),
            gain: Some(format!("{path}.norm.weight")),
            gain_offset: ComponentScalar::new(0.0),
            bias: None,
            groups: 1,
        },
        output_gate: None,
        output_normalization: None,
        residual_scale: ComponentScalar::new(1.0),
    });
    for (suffix, extent, axis, meaning) in [
        (
            "input",
            c.hidden_size as usize,
            "hidden",
            "Normalized shared-expert input",
        ),
        (
            "units",
            width,
            "component",
            "Shared ReLU-squared units before down projection",
        ),
        (
            "write",
            c.hidden_size as usize,
            "hidden",
            "Shared-expert affine write",
        ),
        (
            "output",
            c.hidden_size as usize,
            "hidden",
            "Shared-expert contribution to the sparse sum",
        ),
    ] {
        let mut shape = axes(extent);
        shape[2].name = axis.into();
        g.component_observation(&node, format!("{scope}.{suffix}"), meaning, shape);
    }
}
