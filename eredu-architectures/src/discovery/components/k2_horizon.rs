//! Shared SwiGLU and routed attention-value relationships in K2 Horizon.
use super::*;

pub(in crate::discovery) fn complete(g: &mut Builder, source: &crate::k2_horizon::Declaration) {
    let c = source.args();
    for layer in 0..c.num_hidden_layers as usize {
        let path = format!("model.layers.{layer}");
        let declaration = source
            .block(layer)
            .expect("validated K2 component declaration");
        if let Some(values) = &declaration.values {
            let selector = &values.router;
            let node = format!("decoder.layers.{layer}.attention.values");
            let parameter = values.experts.projection().weight().id.to_string();
            let parameter_group = format!("parameters:{path}.self_attn.v_experts");
            let group = g
                .descriptor
                .components
                .iter_mut()
                .find(|group| group.id == format!("decoder.layers.{layer}.attention.channels"))
                .expect("decoder attention group");
            group.routed_reads.push(ComponentRoutedRead {
                node_id: node.clone(),
                routing: format!("{path}.self_attn.values"),
                expert_count: selector.selection().group_count() as usize,
                input_width: selector.input_dimensions() as usize,
                read: RoutedComponentRead {
                    role: ComponentReadRole::Value,
                    weight: RoutedComponentParameter::Packed {
                        name: RoutedComponentParameterName {
                            parameter: parameter.clone(),
                            shared_parameter: parameter,
                            parameter_group,
                        },
                    },
                    bias: None,
                    projection_rows: values.experts.output_dimensions() as usize,
                    rows: ComponentRowMapping::GroupedQuery {
                        offset: 0,
                        head_width: c.head_dim as usize,
                        queries_per_kv: (c.num_attention_heads / c.num_key_value_heads) as usize,
                    },
                },
                activation: ComponentNonlinearity::Silu {
                    multiplier: ComponentScalar::new(1.0),
                },
                selector_weight: selector.weight().id.to_string(),
                selector_correction_bias: selector.correction_bias().map(|p| p.id.to_string()),
            });
            g.edge(
                &format!("{node}.sum"),
                &format!("decoder.layers.{layer}.attention"),
                ArchitectureEdgeKind::Data,
            );
        }
        let Some(shared) = declaration.routed.as_ref().and_then(|r| r.shared.as_ref()) else {
            continue;
        };
        let node = format!("decoder.layers.{layer}.feed_forward.shared");
        let group = format!("parameters:{path}.mlp");
        let scope = format!("{path}.shared.feed_forward");
        let width = shared.gate.output as usize;
        let read = |projection: &eredu_nn::LinearSpec, role| {
            declared_component_read(
                projection,
                group.clone(),
                role,
                ComponentRowMapping::Direct { offset: 0 },
            )
        };
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
            reads: vec![
                read(&shared.gate, ComponentReadRole::Gate),
                read(&shared.up, ComponentReadRole::Value),
            ],
            routed_reads: vec![],
            write_weight: shared.down.weight.id.to_string(),
            write_input_projection: None,
            shared_write_weight: declared_shared_weight(&shared.down.weight).into(),
            write_parameter_group: group,
            write_bias: shared.down.bias.as_ref().map(|bias| bias.id.to_string()),
            activation_equation: declared_gated_product(shared.policy)
                .expect("validated K2 shared gated-product policy"),
            input_normalization: declared_normalization(
                &declaration.common.normalizations.post_attention,
            ),
            output_gate: None,
            output_normalization: None,
            residual_scale: ComponentScalar::new(1.0),
        });
        for (suffix, extent, axis, meaning) in [
            (
                "input",
                shared.gate.input as usize,
                "hidden",
                "Normalized shared-expert input",
            ),
            (
                "units",
                width,
                "component",
                "Shared SwiGLU units before down projection",
            ),
            (
                "write",
                shared.down.output as usize,
                "hidden",
                "Shared-expert affine write",
            ),
            (
                "output",
                shared.down.output as usize,
                "hidden",
                "Shared-expert contribution to the feed-forward sum",
            ),
        ] {
            let mut shape = axes(extent);
            shape[2].name = axis.into();
            g.component_observation(&node, format!("{scope}.{suffix}"), meaning, shape);
        }
    }
}
