//! Qwen hybrid component equations and interleaved query/gate read geometry.
use super::*;
pub(in crate::discovery) mod prediction;
mod recurrent;
use crate::qwen::hybrid::{HybridConfig, HybridLayerPolicy};

pub(in crate::discovery) fn unit(
    g: &mut Builder,
    c: &HybridConfig,
    layer: usize,
    path: &str,
    mixer: &str,
    ffn: &str,
) {
    let attention = matches!(
        c.layer_schedule.get(layer),
        Some(HybridLayerPolicy::SelfAttention(_))
    );
    unit_with_attention(g, c, layer, path, mixer, ffn, attention);
}

fn unit_with_attention(
    g: &mut Builder,
    c: &HybridConfig,
    layer: usize,
    path: &str,
    mixer: &str,
    ffn: &str,
    attention: bool,
) {
    let policy = if attention {
        HybridLayerPolicy::SelfAttention(eredu_core::AttentionPolicy::Full)
    } else {
        HybridLayerPolicy::LinearAttention
    };
    let scope = crate::qwen::hybrid::BlockParameters::at(c, path, policy)
        .expect("admitted hybrid component scope");
    let width = c.hidden_size as usize;
    let read =
        |spec: &eredu_nn::LinearSpec, parameter_group: &str, role, rows, head_normalization| {
            ComponentRead {
                head_normalization,
                ..declared_component_read(spec, parameter_group.into(), role, rows)
            }
        };
    if attention {
        let prefix = format!("{path}.self_attn");
        let crate::qwen::hybrid::MixerParameters::Attention(spec) = &scope.mixer else {
            unreachable!()
        };
        let heads = spec.projection.geometry.query_heads() as usize;
        let kv = spec.projection.geometry.key_value_heads() as usize;
        let dim = spec.projection.geometry.key_dimensions() as usize;
        let query_stride = spec
            .projection
            .query_layout
            .head_width(spec.projection.geometry)
            .expect("admitted hybrid query/gate row layout") as usize;
        let parameter_group = format!("parameters:{prefix}");
        let head_norm = |count, norm: &eredu_nn::NormalizationConstructionSpec| {
            Some(ComponentHeadNormalization {
                output_scale: ComponentScalar::new(1.0),
                heads: count,
                head_width: norm.dimensions as usize,
                independent_gains: false,
                normalization: declared_normalization(norm),
            })
        };
        g.descriptor.components.push(ComponentGroup {
            routed_reads: Vec::new(),
            id: format!("{mixer}.channels"),
            node_id: mixer.into(),
            layer_index: layer,
            count: heads * dim,
            activation: format!("{path}.attention.channels"),
            effective_activation: format!("{path}.attention.channels.effective"),
            write_input: Some(format!("{path}.attention.write_input")),
            write_output: Some(format!("{path}.attention.write")),
            output: Some(format!("{path}.attention.output")),
            write_partition: ComponentWritePartition::Complete,
            input: format!("{path}.attention.input"),
            reads: vec![
                read(
                    &spec.projection.query,
                    &parameter_group,
                    ComponentReadRole::Query,
                    ComponentRowMapping::HeadRows {
                        offset: 0,
                        component_head_width: dim,
                        read_head_width: dim,
                        component_heads_per_read_head: 1,
                        read_head_stride: Some(query_stride),
                    },
                    head_norm(heads, &spec.query_norm),
                ),
                read(
                    &spec.projection.key,
                    &parameter_group,
                    ComponentReadRole::Key,
                    ComponentRowMapping::HeadRows {
                        offset: 0,
                        component_head_width: dim,
                        read_head_width: dim,
                        component_heads_per_read_head: heads / kv,
                        read_head_stride: None,
                    },
                    head_norm(kv, &spec.key_norm),
                ),
                read(
                    &spec.projection.value,
                    &parameter_group,
                    ComponentReadRole::Value,
                    ComponentRowMapping::GroupedQuery {
                        offset: 0,
                        head_width: dim,
                        queries_per_kv: heads / kv,
                    },
                    None,
                ),
                read(
                    &spec.projection.query,
                    &parameter_group,
                    ComponentReadRole::OutputGate,
                    ComponentRowMapping::Blocked {
                        offset: dim,
                        block_width: dim,
                        block_stride: query_stride,
                    },
                    None,
                ),
            ],
            write_weight: spec.projection.output.weight.id.to_string(),
            write_input_projection: None,
            shared_write_weight: declared_shared_weight(&spec.projection.output.weight).into(),
            write_parameter_group: parameter_group,
            write_bias: spec
                .projection
                .output
                .bias
                .as_ref()
                .map(|bias| bias.id.to_string()),
            activation_equation: ComponentActivation::Attention {
                attended_head_normalization: None,
                query_heads: heads,
                key_value_heads: kv,
                head_width: dim,
                output_gate: Some(ComponentNonlinearity::Sigmoid),
            },
            input_normalization: declared_normalization(&scope.norms[0]),
            output_gate: None,
            output_normalization: None,
            residual_scale: ComponentScalar::new(1.0),
        });
        let mut shape = axes(heads * dim);
        shape[2].name = "component".into();
        g.component_observation(
            mixer,
            format!("{path}.attention.channels"),
            "Aggregated attention channels after sigmoid gating, before output projection",
            shape,
        );
    } else {
        let crate::qwen::hybrid::MixerParameters::Recurrent(spec) = &scope.mixer else {
            unreachable!()
        };
        recurrent::declare(g, spec, &scope.norms[0], layer, path, mixer);
    }
    {
        let shared = c.is_moe();
        let prefix = if shared {
            format!("{path}.mlp.shared_expert")
        } else {
            format!("{path}.mlp")
        };
        let node = if shared {
            format!("{ffn}.shared")
        } else {
            ffn.to_owned()
        };
        let point = if shared {
            format!("{prefix}.feed_forward")
        } else {
            format!("{path}.feed_forward")
        };
        let (projections, shared_gate) = match &scope.feed_forward {
            crate::qwen::hybrid::FeedForwardParameters::Routed(parameters) => {
                (&parameters.shared, Some(&parameters.shared_gate))
            }
            crate::qwen::hybrid::FeedForwardParameters::Dense(spec) => (spec, None),
        };
        let count = projections.gate.output as usize;
        let parameter_group = format!("parameters:{path}.mlp");
        g.descriptor.components.push(ComponentGroup {
            routed_reads: Vec::new(),
            id: format!("{node}.units"),
            node_id: node.clone(),
            layer_index: layer,
            count,
            activation: format!("{point}.units"),
            effective_activation: format!("{point}.units.effective"),
            write_input: Some(format!("{point}.write_input")),
            write_output: Some(format!("{point}.write")),
            output: Some(format!("{point}.output")),
            write_partition: if shared {
                ComponentWritePartition::TensorParallelSum
            } else {
                ComponentWritePartition::Complete
            },
            input: format!("{path}.feed_forward.input"),
            reads: vec![
                read(
                    &projections.gate,
                    &parameter_group,
                    ComponentReadRole::Gate,
                    ComponentRowMapping::Direct { offset: 0 },
                    None,
                ),
                read(
                    &projections.up,
                    &parameter_group,
                    ComponentReadRole::Value,
                    ComponentRowMapping::Direct { offset: 0 },
                    None,
                ),
            ],
            write_weight: projections.down.weight.id.to_string(),
            write_input_projection: None,
            shared_write_weight: declared_shared_weight(&projections.down.weight).into(),
            write_parameter_group: parameter_group.clone(),
            write_bias: projections
                .down
                .bias
                .as_ref()
                .map(|bias| bias.id.to_string()),
            activation_equation: declared_gated_product(projections.policy)
                .expect("admitted hybrid gated-product policy"),
            input_normalization: declared_normalization(&scope.norms[1]),
            output_gate: shared_gate.map(|gate| ComponentOutputGate {
                input: format!("{prefix}.gate.input"),
                projection_input: format!("{prefix}.gate.projection_input"),
                read: read(
                    &gate,
                    &parameter_group,
                    ComponentReadRole::OutputGate,
                    ComponentRowMapping::HeadRows {
                        offset: 0,
                        component_head_width: count,
                        read_head_width: gate.output as usize,
                        component_heads_per_read_head: 1,
                        read_head_stride: None,
                    },
                    None,
                ),
                activation: ComponentNonlinearity::Sigmoid,
                output: format!("{prefix}.gate"),
                effective_output: format!("{prefix}.gate.effective"),
            }),
            output_normalization: None,
            residual_scale: ComponentScalar::new(1.0),
        });
        let mut shape = axes(count);
        shape[2].name = "component".into();
        g.component_observation(
            &node,
            format!("{point}.units"),
            "Gated feed-forward units consumed by the down projection",
            shape,
        );
        if shared {
            for suffix in ["write", "output"] {
                g.component_observation(
                    &node,
                    format!("{point}.{suffix}"),
                    "Shared-expert contribution",
                    axes(width),
                );
            }
            g.read_only_observation(
                &node,
                format!("{prefix}.gate.input"),
                "Actual normalized scalar-gate input",
                axes(width),
            );
            g.projection_input_observation(
                &node,
                format!("{prefix}.gate.projection_input"),
                axes(width),
            );
            let mut gate_axes = axes(1);
            gate_axes[2].name = "gate".into();
            g.component_observation(
                &node,
                format!("{prefix}.gate"),
                "Sigmoid scalar multiplying the projected shared write",
                gate_axes,
            );
        }
    }
    for (node, boundary) in [
        (mixer, if attention { "attention" } else { "mixer" }),
        (ffn, "feed_forward"),
    ] {
        for (suffix, meaning) in [
            ("input", "Normalized operator input"),
            ("write", "Operator write before residual addition"),
            ("output", "Operator contribution before residual addition"),
            ("residual", "Residual after operator addition"),
        ] {
            g.component_observation(
                node,
                format!("{path}.{boundary}.{suffix}"),
                meaning,
                axes(width),
            );
        }
    }
}

pub(in crate::discovery) fn readout(g: &mut Builder, c: &HybridConfig) {
    let construction = c.statics();
    let embedding = construction.embedding_spec();
    let output = construction.output_projection_spec();
    readout_observations(
        g,
        embedding.dimensions as usize,
        embedding.vocabulary as usize,
    );
    let mut other_writes = vec![];
    for layer in 0..c.layer_schedule.len() {
        for (present, boundary) in [(c.is_moe(), "feed_forward")] {
            if present {
                other_writes.push(ComponentResidualWrite {
                    input: Some(format!("model.layers.{layer}.{boundary}.input")),
                    layer_index: layer,
                    node_id: format!("decoder.layers.{layer}.{boundary}"),
                    output: format!("model.layers.{layer}.{boundary}.output"),
                    effective_output: format!("model.layers.{layer}.{boundary}.output.effective"),
                    residual_scale: ComponentScalar::new(1.0),
                });
            }
        }
    }
    g.descriptor.component_readout = Some(ComponentReadout {
        token_embedding_normalization: None,
        embedding_normalization: None,
        embedding: "readout.embedding".into(),
        embedding_weight: embedding.weight.id.as_str().into(),
        embedding_scale: ComponentScalar::new(1.0),
        tied_embeddings: construction.head_spec().is_none(),
        equation: ComponentReadoutEquation {
            residual_cycles: vec![],
            block_transforms: vec![],
            score_writes: vec![],
            stream_residual: None,
            residual: "readout.residual".into(),
            normalized: "readout.normalized".into(),
            projection_input: Some("readout.projection_input".into()),
            linear_scores: "readout.linear".into(),
            logits: MODEL_LOGITS_OBSERVATION_PATH.into(),
            normalization: declared_normalization(construction.normalization_spec()),
            weight: output.weight.id.as_str().into(),
            bias: output.bias.as_ref().map(|bias| bias.id.as_str().into()),
            output_transform: ComponentOutputTransform::Identity,
            block_normalizations: vec![],
            other_writes,
        },
    });
}
