//! Shared declarations for ordinary two-read SwiGLU units.
use super::*;

#[allow(clippy::too_many_arguments)]
pub(super) fn gated_units(
    g: &mut Builder,
    node: &str,
    layer: usize,
    parameter_group: &str,
    prefix: &str,
    scope: &str,
    input: &str,
    count: usize,
    width: usize,
    normalization: ComponentNormalization,
    write_partition: ComponentWritePartition,
) {
    let write = format!("{prefix}.down_proj.weight");
    g.descriptor.components.push(ComponentGroup {
        id: format!("{node}.units"),
        node_id: node.into(),
        layer_index: layer,
        count,
        activation: format!("{scope}.units"),
        effective_activation: format!("{scope}.units.effective"),
        write_input: Some(format!("{scope}.write_input")),
        write_output: Some(format!("{scope}.write")),
        output: Some(format!("{scope}.output")),
        write_partition,
        input: input.into(),
        reads: [
            ("gate_proj", ComponentReadRole::Gate),
            ("up_proj", ComponentReadRole::Value),
        ]
        .into_iter()
        .map(|(field, role)| {
            let weight = format!("{prefix}.{field}.weight");
            ComponentRead {
                source: None,
                projection_output: None,
                role,
                shared_weight: weight.clone(),
                weight,
                parameter_group: parameter_group.into(),
                bias: None,
                rows: ComponentRowMapping::Direct { offset: 0 },
                head_normalization: None,
                input_projections: vec![],
            }
        })
        .collect(),
        routed_reads: vec![],
        shared_write_weight: write.clone(),
        write_weight: write,
        write_input_projection: None,
        write_parameter_group: parameter_group.into(),
        write_bias: None,
        activation_equation: ComponentActivation::Gated {
            activation: ComponentNonlinearity::Silu {
                multiplier: ComponentScalar::new(1.0),
            },
            gate_upper_bound: None,
            value_absolute_bound: None,
            value_offset: ComponentScalar::new(0.0),
        },
        input_normalization: normalization,
        output_gate: None,
        output_normalization: None,
        residual_scale: ComponentScalar::new(1.0),
    });
    let mut units = axes(count);
    units[2].name = "component".into();
    g.component_observation(
        node,
        format!("{scope}.units"),
        "SwiGLU product before down projection",
        units,
    );
    for suffix in ["write", "output"] {
        g.component_observation(
            node,
            format!("{scope}.{suffix}"),
            "Gated FFN contribution before its enclosing sum",
            axes(width),
        );
    }
}
