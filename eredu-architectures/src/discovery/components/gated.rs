//! Shared declarations for ordinary two-read SwiGLU units.
use super::*;

#[allow(clippy::too_many_arguments)]
pub(super) fn gated_units(
    g: &mut Builder,
    node: &str,
    layer: usize,
    parameter_group: &str,
    projections: [&eredu_nn::LinearSpec; 3],
    policy: eredu_nn::GatedProductPolicy,
    scope: &str,
    input: &str,
    normalization: ComponentNormalization,
    write_partition: ComponentWritePartition,
) {
    let [gate, up, down] = projections;
    let count = down.input as usize;
    let width = gate.input as usize;
    let write = down.weight.id.as_str().to_owned();
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
            (gate, ComponentReadRole::Gate),
            (up, ComponentReadRole::Value),
        ]
        .into_iter()
        .map(|(projection, role)| {
            declared_component_read(
                projection,
                parameter_group.into(),
                role,
                ComponentRowMapping::Direct { offset: 0 },
            )
        })
        .collect(),
        routed_reads: vec![],
        shared_write_weight: declared_shared_weight(&down.weight).into(),
        write_weight: write,
        write_input_projection: None,
        write_parameter_group: parameter_group.into(),
        write_bias: down.bias.as_ref().map(|p| p.id.as_str().into()),
        activation_equation: declared_gated_product(policy)
            .expect("declared gated component policy"),
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
