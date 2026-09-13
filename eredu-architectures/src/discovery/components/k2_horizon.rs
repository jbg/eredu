//! Shared SwiGLU and routed attention-value relationships in K2 Horizon.
use super::*;

pub(in crate::discovery) fn complete(g: &mut Builder, c: &crate::k2_horizon::ModelArgs) {
    for layer in 0..c.num_hidden_layers as usize {
        let path = format!("model.layers.{layer}");
        if c.is_mova_layer(layer) {
            let node = format!("decoder.layers.{layer}.attention.values");
            let parameter = format!("{path}.self_attn.v_experts.weight");
            let parameter_group = format!("parameters:{path}.self_attn.v_experts");
            let group = g
                .descriptor
                .components
                .iter_mut()
                .find(|group| group.id == format!("decoder.layers.{layer}.attention.channels"))
                .expect("decoder attention group");
            group
                .reads
                .retain(|read| read.role != ComponentReadRole::Value);
            group.routed_reads.push(ComponentRoutedRead {
                node_id: node.clone(),
                routing: format!("{path}.self_attn.values"),
                expert_count: c.mova_num_experts as usize,
                input_width: c.hidden_size as usize,
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
                    projection_rows: (c.num_key_value_heads * c.head_dim) as usize,
                    rows: ComponentRowMapping::GroupedQuery {
                        offset: 0,
                        head_width: c.head_dim as usize,
                        queries_per_kv: (c.num_attention_heads / c.num_key_value_heads) as usize,
                    },
                },
                activation: ComponentNonlinearity::Silu {
                    multiplier: ComponentScalar::new(1.0),
                },
                selector_weight: format!("{path}.self_attn.v_router.weight"),
                selector_correction_bias: c
                    .moe_gate_bias
                    .then(|| format!("{path}.self_attn.v_router.bias")),
            });
            g.edge(
                &format!("{node}.sum"),
                &format!("decoder.layers.{layer}.attention"),
                ArchitectureEdgeKind::Data,
            );
        }
        if !c.is_sparse_layer(layer) || c.num_shared_experts == 0 {
            continue;
        }
        let node = format!("decoder.layers.{layer}.feed_forward.shared");
        let root = format!("{path}.mlp.shared_experts");
        let group = format!("parameters:{path}.mlp");
        let scope = format!("{path}.shared.feed_forward");
        let width = (c.moe_intermediate_size * c.num_shared_experts) as usize;
        let read = |field, role| {
            let weight = format!("{root}.{field}.weight");
            ComponentRead {
                source: None,
                projection_output: None,
                role,
                weight: weight.clone(),
                shared_weight: weight,
                parameter_group: group.clone(),
                bias: None,
                rows: ComponentRowMapping::Direct { offset: 0 },
                head_normalization: None,
                input_projections: Vec::new(),
            }
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
                read("gate_proj", ComponentReadRole::Gate),
                read("up_proj", ComponentReadRole::Value),
            ],
            routed_reads: vec![],
            write_weight: format!("{root}.down_proj.weight"),
            write_input_projection: None,
            shared_write_weight: format!("{root}.down_proj.weight"),
            write_parameter_group: group,
            write_bias: None,
            activation_equation: ComponentActivation::Gated {
                activation: ComponentNonlinearity::Silu {
                    multiplier: ComponentScalar::new(1.0),
                },
                gate_upper_bound: None,
                value_absolute_bound: None,
                value_offset: ComponentScalar::new(0.0),
            },
            input_normalization: ComponentNormalization {
                kind: ComponentNormalizationKind::Rms,
                epsilon: ComponentScalar::new(c.rms_norm_eps),
                gain: Some(format!("{path}.post_attention_layernorm.weight")),
                gain_offset: ComponentScalar::new(0.0),
                bias: None,
                groups: c.layernorm_num_groups as usize,
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
                "Shared SwiGLU units before down projection",
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
                "Shared-expert contribution to the feed-forward sum",
            ),
        ] {
            let mut shape = axes(extent);
            shape[2].name = axis.into();
            g.component_observation(&node, format!("{scope}.{suffix}"), meaning, shape);
        }
    }
}
