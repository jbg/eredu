//! Always-executed ReLU² units nested within a Nemotron sparse write.
use super::*;

pub(in crate::discovery) mod prediction;

pub(in crate::discovery) fn shared_units(
    g: &mut Builder,
    spec: &crate::nemotron_h::mlp::SparseMoeSpec,
    norm: &eredu_nn::NormalizationConstructionSpec,
    layer: usize,
    path: &str,
    operator: &str,
    parameter_root: &str,
) {
    let node = format!("{operator}.shared");
    let parameter_group = format!("parameters:{parameter_root}");
    let scope = format!("{path}.shared.feed_forward");
    let up = &spec.shared.network.up;
    let down = &spec.shared.network.down;
    let width = up.output as usize;
    let write_weight = down.weight.id.as_str().to_owned();
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
        reads: vec![declared_component_read(
            up,
            parameter_group.clone(),
            ComponentReadRole::Input,
            ComponentRowMapping::Direct { offset: 0 },
        )],
        routed_reads: vec![],
        write_weight: write_weight.clone(),
        write_input_projection: None,
        shared_write_weight: declared_shared_weight(&down.weight).into(),
        write_parameter_group: parameter_group,
        write_bias: down.bias.as_ref().map(|p| p.id.as_str().into()),
        activation_equation: ComponentActivation::Unary {
            activation: crate::decoder::unary::component(spec.shared.network.activation),
        },
        input_normalization: declared_normalization(norm),
        output_gate: None,
        output_normalization: None,
        residual_scale: ComponentScalar::new(1.0),
    });
    for (suffix, extent, axis, meaning) in [
        (
            "input",
            up.input as usize,
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
            up.input as usize,
            "hidden",
            "Shared-expert affine write",
        ),
        (
            "output",
            up.input as usize,
            "hidden",
            "Shared-expert contribution to the sparse sum",
        ),
    ] {
        let mut shape = axes(extent);
        shape[2].name = axis.into();
        g.component_observation(&node, format!("{scope}.{suffix}"), meaning, shape);
    }
}
