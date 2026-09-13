//! Qwen hybrid component equations and interleaved query/gate read geometry.
use super::*;
pub(in crate::discovery) mod prediction;
mod recurrent;
use crate::qwen::hybrid::{HybridConfig, HybridLayerPolicy};

fn normalization(c: &HybridConfig, gain: String) -> ComponentNormalization {
    ComponentNormalization {
        kind: ComponentNormalizationKind::Rms,
        epsilon: ComponentScalar::new(c.rms_norm_eps),
        gain: Some(gain),
        gain_offset: ComponentScalar::new(1.0),
        bias: None,
        groups: 1,
    }
}

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
    let width = c.hidden_size as usize;
    let heads = c.num_attention_heads as usize;
    let kv = c.num_key_value_heads as usize;
    let dim = c.head_dim as usize;
    let read = |prefix: &str, field: &str, role, rows, bias, head_normalization| {
        let weight = format!("{prefix}.{field}.weight");
        ComponentRead {
            source: None,
            projection_output: None,
            role,
            weight: weight.clone(),
            shared_weight: weight,
            parameter_group: format!("parameters:{prefix}"),
            bias: if bias {
                Some(format!("{prefix}.{field}.bias"))
            } else {
                None
            },
            rows,
            head_normalization,
            input_projections: Vec::new(),
        }
    };
    if attention {
        let prefix = format!("{path}.self_attn");
        let head_norm = |count, name| {
            Some(ComponentHeadNormalization {
                output_scale: ComponentScalar::new(1.0),
                heads: count,
                head_width: dim,
                independent_gains: false,
                normalization: normalization(c, format!("{prefix}.{name}.weight")),
            })
        };
        let weight = format!("{prefix}.o_proj.weight");
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
                    &prefix,
                    "q_proj",
                    ComponentReadRole::Query,
                    ComponentRowMapping::HeadRows {
                        offset: 0,
                        component_head_width: dim,
                        read_head_width: dim,
                        component_heads_per_read_head: 1,
                        read_head_stride: Some(2 * dim),
                    },
                    c.attention_bias,
                    head_norm(heads, "q_norm"),
                ),
                read(
                    &prefix,
                    "k_proj",
                    ComponentReadRole::Key,
                    ComponentRowMapping::HeadRows {
                        offset: 0,
                        component_head_width: dim,
                        read_head_width: dim,
                        component_heads_per_read_head: heads / kv,
                        read_head_stride: None,
                    },
                    c.attention_bias,
                    head_norm(kv, "k_norm"),
                ),
                read(
                    &prefix,
                    "v_proj",
                    ComponentReadRole::Value,
                    ComponentRowMapping::GroupedQuery {
                        offset: 0,
                        head_width: dim,
                        queries_per_kv: heads / kv,
                    },
                    c.attention_bias,
                    None,
                ),
                read(
                    &prefix,
                    "q_proj",
                    ComponentReadRole::OutputGate,
                    ComponentRowMapping::Blocked {
                        offset: dim,
                        block_width: dim,
                        block_stride: 2 * dim,
                    },
                    c.attention_bias,
                    None,
                ),
            ],
            write_weight: weight.clone(),
            write_input_projection: None,
            shared_write_weight: weight,
            write_parameter_group: format!("parameters:{prefix}"),
            write_bias: c.attention_bias.then(|| format!("{prefix}.o_proj.bias")),
            activation_equation: ComponentActivation::Attention {
                query_heads: heads,
                key_value_heads: kv,
                head_width: dim,
                output_gate: Some(ComponentNonlinearity::Sigmoid),
            },
            input_normalization: normalization(c, format!("{path}.input_layernorm.weight")),
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
        recurrent::declare(g, c, layer, path, mixer);
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
        let count = if shared {
            c.shared_expert_intermediate_size
        } else {
            c.intermediate_size
        } as usize;
        let parameter_group = format!("parameters:{path}.mlp");
        let weight = format!("{prefix}.down_proj.weight");
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
                    &prefix,
                    "gate_proj",
                    ComponentReadRole::Gate,
                    ComponentRowMapping::Direct { offset: 0 },
                    false,
                    None,
                ),
                read(
                    &prefix,
                    "up_proj",
                    ComponentReadRole::Value,
                    ComponentRowMapping::Direct { offset: 0 },
                    false,
                    None,
                ),
            ]
            .into_iter()
            .map(|mut read| {
                read.parameter_group = parameter_group.clone();
                read
            })
            .collect(),
            write_weight: weight.clone(),
            write_input_projection: None,
            shared_write_weight: weight,
            write_parameter_group: parameter_group.clone(),
            write_bias: None,
            activation_equation: ComponentActivation::Gated {
                activation: ComponentNonlinearity::Silu {
                    multiplier: ComponentScalar::new(1.0),
                },
                gate_upper_bound: None,
                value_absolute_bound: None,
                value_offset: ComponentScalar::new(0.0),
            },
            input_normalization: normalization(
                c,
                format!("{path}.post_attention_layernorm.weight"),
            ),
            output_gate: shared.then(|| ComponentOutputGate {
                input: format!("{prefix}.gate.input"),
                projection_input: format!("{prefix}.gate.projection_input"),
                read: ComponentRead {
                    source: None,
                    projection_output: None,
                    role: ComponentReadRole::OutputGate,
                    weight: format!("{path}.mlp.shared_expert_gate.weight"),
                    shared_weight: format!("{path}.mlp.shared_expert_gate.weight"),
                    parameter_group,
                    bias: None,
                    rows: ComponentRowMapping::HeadRows {
                        offset: 0,
                        component_head_width: count,
                        read_head_width: 1,
                        component_heads_per_read_head: 1,
                        read_head_stride: None,
                    },
                    head_normalization: None,
                    input_projections: vec![],
                },
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
    readout_observations(g, c.hidden_size as usize, c.vocab_size as usize);
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
        embedding_weight: "model.embed_tokens.weight".into(),
        embedding_scale: ComponentScalar::new(1.0),
        tied_embeddings: c.tie_word_embeddings,
        equation: ComponentReadoutEquation {
            block_transforms: vec![],
            score_writes: vec![],
            stream_residual: None,
            residual: "readout.residual".into(),
            normalized: "readout.normalized".into(),
            projection_input: Some("readout.projection_input".into()),
            linear_scores: "readout.linear".into(),
            logits: MODEL_LOGITS_OBSERVATION_PATH.into(),
            normalization: normalization(c, "model.norm.weight".into()),
            weight: if c.tie_word_embeddings {
                "model.embed_tokens.weight"
            } else {
                "lm_head.weight"
            }
            .into(),
            bias: None,
            output_transform: ComponentOutputTransform::Identity,
            block_normalizations: vec![],
            other_writes,
        },
    });
}
