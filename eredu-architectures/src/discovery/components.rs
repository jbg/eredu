//! Scalar component relationships from the same normalized decoder config used to build modules.
use super::*;
pub(super) mod deepseek_v3;
pub(super) mod deepseek_v4;
mod gated;
pub(super) mod gemma4;
pub(super) mod inkling;
pub(super) mod kimi_linear;
pub(super) mod muse;
use gated::gated_units;
pub(super) mod k2_horizon;
pub(super) mod nemotron;
pub(super) mod qwen_hybrid;
use crate::decoder::{
    AttentionProjection, AttentionProjectionLayout, Config, GatedProjectionLayout,
    OutputGateActivation,
};
use eredu_core::component::*;

pub(super) fn readout<C: Config>(g: &mut Builder, c: &C) {
    let root = c.parameter_root();
    let width = c.hidden_size() as usize;
    let normalization = |gain| ComponentNormalization {
        kind: ComponentNormalizationKind::Rms,
        epsilon: ComponentScalar::new(c.rms_norm_epsilon()),
        gain: Some(gain),
        gain_offset: ComponentScalar::new(c.normalization_offset()),
        bias: None,
        groups: c.normalization_groups().unwrap_or(1) as usize,
    };
    readout_observations(g, width, c.vocabulary_size() as usize);
    g.descriptor.component_readout = Some(ComponentReadout {
        token_embedding_normalization: None,
        embedding_normalization: None,
        embedding: "readout.embedding".into(),
        embedding_weight: format!("{root}.embed_tokens.weight"),
        embedding_scale: ComponentScalar::new(c.embedding_scale()),
        tied_embeddings: c.tie_word_embeddings(),
        equation: ComponentReadoutEquation {
            block_transforms: vec![],
            score_writes: vec![],
            stream_residual: None,
            residual: "readout.residual".into(),
            normalized: "readout.normalized".into(),
            projection_input: Some("readout.projection_input".into()),
            linear_scores: "readout.linear".into(),
            logits: MODEL_LOGITS_OBSERVATION_PATH.into(),
            normalization: normalization(format!("{root}.norm.weight")),
            weight: if c.tie_word_embeddings() {
                format!("{root}.embed_tokens.weight")
            } else {
                "lm_head.weight".into()
            },
            bias: None,
            output_transform: c.output_softcap().map_or(
                ComponentOutputTransform::Identity,
                |cap| ComponentOutputTransform::Softcap {
                    cap: ComponentScalar::new(cap),
                },
            ),
            block_normalizations: (0..c.num_hidden_layers() as usize)
                .filter_map(|layer_index| {
                    c.block_output_normalization(layer_index).map(|gain| {
                        ComponentResidualNormalization {
                            layer_index,
                            input: format!("{root}.layers.{layer_index}.feed_forward.residual"),
                            output: format!("{root}.layers.{layer_index}.output"),
                            normalization: normalization(gain),
                        }
                    })
                })
                .collect(),
            other_writes: vec![],
        },
    });
}

pub(super) fn decoder_unit<C: Config>(
    g: &mut Builder,
    c: &C,
    layer: usize,
    path: &str,
    attention: &str,
    ffn: &str,
    dense_ffn: bool,
) {
    let fields = c.block_parameter_fields();
    let attention_prefix = format!("{path}.{}", fields.attention);
    let ffn_prefix = format!("{path}.{}", fields.feed_forward);
    let heads = c.num_attention_heads() as usize;
    let kv = c.num_key_value_heads() as usize;
    let dim = c.head_dim() as usize;
    let width = c.intermediate_size() as usize;
    let normalization = |gain: String| ComponentNormalization {
        kind: ComponentNormalizationKind::Rms,
        epsilon: ComponentScalar::new(c.rms_norm_epsilon()),
        gain: Some(gain),
        gain_offset: ComponentScalar::new(c.normalization_offset()),
        bias: None,
        groups: c.normalization_groups().unwrap_or(1) as usize,
    };
    let read = |prefix: &str, field: &str, role, rows, bias: bool| {
        let weight = format!("{prefix}.{field}.weight");
        let head_normalization = match role {
            ComponentReadRole::Query => Some((heads, fields.attention_query_norm)),
            ComponentReadRole::Key => Some((kv, fields.attention_key_norm)),
            _ => None,
        }
        .and_then(|(heads, field)| {
            c.query_key_norm_epsilon()
                .map(|epsilon| ComponentHeadNormalization {
                    output_scale: ComponentScalar::new(1.0),
                    heads,
                    head_width: dim,
                    independent_gains: c.query_key_norm_per_head_weights(),
                    normalization: ComponentNormalization {
                        kind: ComponentNormalizationKind::Rms,
                        epsilon: ComponentScalar::new(epsilon),
                        gain: Some(format!("{prefix}.{field}.weight")),
                        // Q/K construction uses learned full gains independently of
                        // the sublayer normalization's checkpoint offset convention.
                        gain_offset: ComponentScalar::new(0.0),
                        bias: None,
                        groups: if c.query_key_norm_per_head_weights() {
                            heads
                        } else {
                            1
                        },
                    },
                })
        });
        ComponentRead {
            source: None,
            projection_output: None,
            role,
            shared_weight: weight.clone(),
            weight,
            parameter_group: format!("parameters:{prefix}"),
            bias: bias.then(|| format!("{prefix}.{field}.bias")),
            rows,
            head_normalization,
            input_projections: Vec::new(),
        }
    };
    let direct = |offset| ComponentRowMapping::Direct { offset };
    let grouped = |offset| ComponentRowMapping::GroupedQuery {
        offset,
        head_width: dim,
        queries_per_kv: heads / kv,
    };
    let head_rows = |offset, sharing| ComponentRowMapping::HeadRows {
        offset,
        component_head_width: dim,
        read_head_width: dim,
        component_heads_per_read_head: sharing,
        read_head_stride: None,
    };
    let mut reads = match c.attention_projection_layout() {
        AttentionProjectionLayout::Split => vec![
            read(
                &attention_prefix,
                fields.attention_query,
                ComponentReadRole::Query,
                head_rows(0, 1),
                c.attention_bias(AttentionProjection::Query),
            ),
            read(
                &attention_prefix,
                fields.attention_key,
                ComponentReadRole::Key,
                head_rows(0, heads / kv),
                c.attention_bias(AttentionProjection::Key),
            ),
            read(
                &attention_prefix,
                fields.attention_value,
                ComponentReadRole::Value,
                grouped(0),
                c.attention_bias(AttentionProjection::Value),
            ),
        ],
        AttentionProjectionLayout::Fused { field } => vec![
            read(
                &attention_prefix,
                field,
                ComponentReadRole::Query,
                head_rows(0, 1),
                c.attention_bias(AttentionProjection::Query),
            ),
            read(
                &attention_prefix,
                field,
                ComponentReadRole::Key,
                head_rows(heads * dim, heads / kv),
                c.attention_bias(AttentionProjection::Key),
            ),
            read(
                &attention_prefix,
                field,
                ComponentReadRole::Value,
                grouped((heads + kv) * dim),
                c.attention_bias(AttentionProjection::Value),
            ),
        ],
    };
    if let Some((field, _)) = c.attention_output_gate() {
        reads.push(read(
            &attention_prefix,
            field,
            ComponentReadRole::OutputGate,
            direct(0),
            false,
        ));
    }
    let mut group = ComponentGroup {
        routed_reads: Vec::new(),
        id: format!("{attention}.channels"),
        node_id: attention.into(),
        layer_index: layer,
        count: heads * dim,
        activation: format!("{path}.attention.channels"),
        effective_activation: format!("{path}.attention.channels.effective"),
        write_input: Some(format!("{path}.attention.write_input")),
        write_output: Some(format!("{path}.attention.write")),
        output: Some(format!("{path}.attention.output")),
        write_partition: ComponentWritePartition::Complete,
        input: format!("{path}.attention.input"),
        reads,
        write_weight: format!("{attention_prefix}.{}.weight", fields.attention_output),
        write_input_projection: None,
        write_parameter_group: format!("parameters:{attention_prefix}"),
        shared_write_weight: format!("{attention_prefix}.{}.weight", fields.attention_output),
        write_bias: c
            .attention_bias(AttentionProjection::Output)
            .then(|| format!("{attention_prefix}.{}.bias", fields.attention_output)),
        activation_equation: ComponentActivation::Attention {
            query_heads: heads,
            key_value_heads: kv,
            head_width: dim,
            output_gate: c
                .attention_output_gate()
                .map(|(_, activation)| match activation {
                    OutputGateActivation::Softplus(beta) => ComponentNonlinearity::Softplus {
                        beta: ComponentScalar::new(beta),
                    },
                    OutputGateActivation::Sigmoid => ComponentNonlinearity::Sigmoid,
                    OutputGateActivation::Silu => ComponentNonlinearity::Silu {
                        multiplier: ComponentScalar::new(1.0),
                    },
                }),
        },
        input_normalization: normalization(format!("{path}.{}.weight", fields.input_norm)),
        output_gate: None,
        output_normalization: c.attention_output_normalization(layer).map(&normalization),
        residual_scale: ComponentScalar::new(1.0),
    };
    g.descriptor.components.push(group.clone());
    if !dense_ffn {
        return;
    }
    group.id = format!("{ffn}.units");
    group.node_id = ffn.into();
    group.count = width;
    group.activation = format!("{path}.feed_forward.units");
    group.effective_activation = format!("{path}.feed_forward.units.effective");
    group.write_input = Some(format!("{path}.feed_forward.write_input"));
    group.write_output = Some(format!("{path}.feed_forward.write"));
    group.output = Some(format!("{path}.feed_forward.output"));
    group.input = format!("{path}.feed_forward.input");
    group.reads = match c.gated_projection_layout() {
        GatedProjectionLayout::Split => vec![
            read(
                &ffn_prefix,
                fields.feed_forward_gate,
                ComponentReadRole::Gate,
                direct(0),
                c.mlp_bias(),
            ),
            read(
                &ffn_prefix,
                fields.feed_forward_up,
                ComponentReadRole::Value,
                direct(0),
                c.mlp_bias(),
            ),
        ],
        GatedProjectionLayout::Fused { field } => vec![
            read(
                &ffn_prefix,
                field,
                ComponentReadRole::Gate,
                direct(0),
                c.mlp_bias(),
            ),
            read(
                &ffn_prefix,
                field,
                ComponentReadRole::Value,
                direct(width),
                c.mlp_bias(),
            ),
        ],
    };
    group.write_weight = format!("{ffn_prefix}.{}.weight", fields.feed_forward_output);
    group.shared_write_weight = group.write_weight.clone();
    group.write_parameter_group = format!("parameters:{ffn_prefix}");
    group.write_bias = c
        .mlp_bias()
        .then(|| format!("{ffn_prefix}.{}.bias", fields.feed_forward_output));
    let policy = c.gated_product_policy().unwrap_or_default();
    group.activation_equation = ComponentActivation::Gated {
        activation: match policy.activation() {
            eredu_nn::GatedProductActivation::Silu => ComponentNonlinearity::Silu {
                multiplier: ComponentScalar::new(policy.sigmoid_multiplier()),
            },
            eredu_nn::GatedProductActivation::GeluApproximate => {
                ComponentNonlinearity::GeluApproximate
            }
            _ => return,
        },
        gate_upper_bound: policy.gate_upper_bound().map(ComponentScalar::new),
        value_absolute_bound: policy.up_absolute_bound().map(ComponentScalar::new),
        value_offset: ComponentScalar::new(policy.up_offset()),
    };
    group.input_normalization =
        normalization(format!("{path}.{}.weight", fields.post_attention_norm));
    group.output_normalization = c
        .feed_forward_output_normalization(layer)
        .map(normalization);
    g.descriptor.components.push(group);
}

/// Nemotron physical units each contain a single pre-normalized residual operator.
pub(super) fn nemotron_unit(
    g: &mut Builder,
    c: &crate::nemotron_h::ModelArgs,
    layer: usize,
    path: &str,
    node: &str,
    attention: bool,
) {
    let field = if attention { "attention" } else { "mlp" };
    nemotron_unit_at(
        g,
        c,
        layer,
        path,
        node,
        attention,
        &format!("{path}.{field}"),
    );
}

fn nemotron_unit_at(
    g: &mut Builder,
    c: &crate::nemotron_h::ModelArgs,
    layer: usize,
    path: &str,
    node: &str,
    attention: bool,
    prefix: &str,
) {
    let boundary = if attention {
        "attention"
    } else {
        "feed_forward"
    };
    let scalar = if attention { "channels" } else { "units" };
    let heads = c.num_attention_heads as usize;
    let kv = c.num_key_value_heads as usize;
    let dim = c.head_dim as usize;
    let count = if attention {
        heads * dim
    } else {
        c.intermediate_size as usize
    };
    let bias = if attention {
        c.attention_bias
    } else {
        c.mlp_bias
    };
    let read = |field: &str, role, rows| {
        let weight = format!("{prefix}.{field}.weight");
        ComponentRead {
            source: None,
            projection_output: None,
            shared_weight: weight.clone(),
            weight,
            parameter_group: format!("parameters:{prefix}"),
            role,
            rows,
            bias: bias.then(|| format!("{prefix}.{field}.bias")),
            head_normalization: None,
            input_projections: Vec::new(),
        }
    };
    let direct = ComponentRowMapping::Direct { offset: 0 };
    let grouped = ComponentRowMapping::GroupedQuery {
        offset: 0,
        head_width: dim,
        queries_per_kv: heads / kv,
    };
    let head_rows = |sharing| ComponentRowMapping::HeadRows {
        offset: 0,
        component_head_width: dim,
        read_head_width: dim,
        component_heads_per_read_head: sharing,
        read_head_stride: None,
    };
    let reads = if attention {
        vec![
            read("q_proj", ComponentReadRole::Query, head_rows(1)),
            read("k_proj", ComponentReadRole::Key, head_rows(heads / kv)),
            read("v_proj", ComponentReadRole::Value, grouped),
        ]
    } else {
        vec![read("up_proj", ComponentReadRole::Input, direct)]
    };
    let write = if attention { "o_proj" } else { "down_proj" };
    let write_weight = format!("{prefix}.{write}.weight");
    g.descriptor.components.push(ComponentGroup {
        routed_reads: Vec::new(),
        id: format!("{node}.{scalar}"),
        node_id: node.into(),
        layer_index: layer,
        count,
        activation: format!("{path}.{boundary}.{scalar}"),
        effective_activation: format!("{path}.{boundary}.{scalar}.effective"),
        write_input: Some(format!("{path}.{boundary}.write_input")),
        write_output: Some(format!("{path}.{boundary}.write")),
        output: Some(format!("{path}.{boundary}.output")),
        write_partition: ComponentWritePartition::Complete,
        input: format!("{path}.{boundary}.input"),
        reads,
        shared_write_weight: write_weight.clone(),
        write_weight,
        write_input_projection: None,
        write_parameter_group: format!("parameters:{prefix}"),
        write_bias: bias.then(|| format!("{prefix}.{write}.bias")),
        activation_equation: if attention {
            ComponentActivation::Attention {
                query_heads: heads,
                key_value_heads: kv,
                head_width: dim,
                output_gate: None,
            }
        } else {
            ComponentActivation::Unary {
                activation: crate::decoder::unary::Activation::ReluSquared.component(),
            }
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
    for (suffix, meaning, extent) in [
        ("input", "Normalized operator input", c.hidden_size as usize),
        (
            scalar,
            "Activated units or aggregated channels consumed by the output projection",
            count,
        ),
        ("write", "Affine operator write", c.hidden_size as usize),
        (
            "output",
            "Operator contribution before residual addition",
            c.hidden_size as usize,
        ),
        (
            "residual",
            "Residual after operator addition",
            c.hidden_size as usize,
        ),
    ] {
        let mut shape = axes(extent);
        if suffix == scalar {
            shape[2].name = "component".into();
        }
        g.component_observation(node, format!("{path}.{boundary}.{suffix}"), meaning, shape);
    }
}

fn readout_observations(g: &mut Builder, width: usize, vocabulary: usize) {
    for (node, path, meaning) in [
        (
            "embedding",
            "readout.embedding",
            "Scaled token embeddings entering the decoder",
        ),
        (
            "output.norm",
            "readout.residual",
            "Residual before final normalization",
        ),
        (
            "output.norm",
            "readout.normalized",
            "Final normalized residual consumed by the output head",
        ),
    ] {
        g.component_observation(node, path.into(), meaning, axes(width));
    }
    let mut scores = axes(vocabulary);
    scores[2].name = "vocabulary".into();
    g.component_observation(
        "output",
        "readout.linear".into(),
        "Affine vocabulary scores before output softcap and sampling",
        scores,
    );
}

/// Complete non-component writes retain their own original/effective timing.
pub(super) fn nemotron_other_unit(
    g: &mut Builder,
    width: usize,
    path: &str,
    node: &str,
    boundary: &str,
) {
    for (suffix, meaning) in [
        ("input", "Normalized operator input"),
        ("write", "Complete operator write"),
        (
            "output",
            "Complete operator contribution before residual addition",
        ),
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

pub(super) fn nemotron_readout(g: &mut Builder, c: &crate::nemotron_h::ModelArgs) {
    readout_observations(g, c.hidden_size as usize, c.vocab_size as usize);
    g.descriptor.component_readout = Some(ComponentReadout {
        token_embedding_normalization: None,
        embedding_normalization: None,
        embedding: "readout.embedding".into(),
        embedding_weight: "model.embeddings.weight".into(),
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
            normalization: ComponentNormalization {
                kind: ComponentNormalizationKind::Rms,
                epsilon: ComponentScalar::new(c.layer_norm_epsilon),
                gain: Some("model.norm_f.weight".into()),
                gain_offset: ComponentScalar::new(0.0),
                bias: None,
                groups: 1,
            },
            weight: if c.tie_word_embeddings {
                "model.embeddings.weight"
            } else {
                "lm_head.weight"
            }
            .into(),
            bias: None,
            output_transform: ComponentOutputTransform::Identity,
            block_normalizations: vec![],
            other_writes: c
                .layer_schedule
                .iter()
                .enumerate()
                .filter_map(|(layer, policy)| {
                    let boundary = match policy {
                        crate::nemotron_h::LayerPolicy::Mamba => "mixer",
                        crate::nemotron_h::LayerPolicy::SparseMoe => "feed_forward",
                        _ => return None,
                    };
                    Some(ComponentResidualWrite {
                        input: Some(format!("model.layers.{layer}.{boundary}.input")),
                        layer_index: layer,
                        node_id: format!("decoder.layers.{layer}.operator"),
                        output: format!("model.layers.{layer}.{boundary}.output"),
                        effective_output: format!(
                            "model.layers.{layer}.{boundary}.output.effective"
                        ),
                        residual_scale: ComponentScalar::new(1.0),
                    })
                })
                .collect(),
        },
    });
}

/// LFM2's mixed schedule retains the same dense feed-forward component equation
/// after either attention or convolution. Convolution is an undecomposed write.
pub(super) fn lfm2_unit(
    g: &mut Builder,
    c: &crate::lfm2::ModelArgs,
    layer: usize,
    path: &str,
    mixer_node: &str,
    ffn_node: &str,
) {
    use crate::lfm2::{FeedForwardPolicy, OperatorPolicy};
    let policy = c.layer_policy(layer).expect("declared LFM2 layer");
    let width = c.hidden_size as usize;
    let heads = c.num_attention_heads as usize;
    let kv = c.num_key_value_heads as usize;
    let dim = width / heads;
    let norm = |gain: String| ComponentNormalization {
        kind: ComponentNormalizationKind::Rms,
        epsilon: ComponentScalar::new(c.norm_eps),
        gain: Some(gain),
        gain_offset: ComponentScalar::new(0.0),
        bias: None,
        groups: 1,
    };
    let read = |prefix: &str, field: &str, role, rows, head_normalization| {
        let weight = format!("{prefix}.{field}.weight");
        ComponentRead {
            source: None,
            projection_output: None,
            role,
            weight: weight.clone(),
            shared_weight: weight,
            parameter_group: format!("parameters:{prefix}"),
            bias: None,
            rows,
            head_normalization,
            input_projections: Vec::new(),
        }
    };
    let attention = matches!(policy.operator, OperatorPolicy::SelfAttention(_));
    if attention {
        let prefix = format!("{path}.self_attn");
        let head_rows = |sharing| ComponentRowMapping::HeadRows {
            offset: 0,
            component_head_width: dim,
            read_head_width: dim,
            component_heads_per_read_head: sharing,
            read_head_stride: None,
        };
        let head_norm = |heads, field| {
            Some(ComponentHeadNormalization {
                output_scale: ComponentScalar::new(1.0),
                heads,
                head_width: dim,
                independent_gains: false,
                normalization: norm(format!("{prefix}.{field}.weight")),
            })
        };
        let write_weight = format!("{prefix}.out_proj.weight");
        g.descriptor.components.push(ComponentGroup {
            routed_reads: Vec::new(),
            id: format!("{mixer_node}.channels"),
            node_id: mixer_node.into(),
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
                    head_rows(1),
                    head_norm(heads, "q_layernorm"),
                ),
                read(
                    &prefix,
                    "k_proj",
                    ComponentReadRole::Key,
                    head_rows(heads / kv),
                    head_norm(kv, "k_layernorm"),
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
                    None,
                ),
            ],
            write_weight: write_weight.clone(),
            write_input_projection: None,
            shared_write_weight: write_weight,
            write_parameter_group: format!("parameters:{prefix}"),
            write_bias: None,
            activation_equation: ComponentActivation::Attention {
                query_heads: heads,
                key_value_heads: kv,
                head_width: dim,
                output_gate: None,
            },
            input_normalization: norm(format!("{path}.operator_norm.weight")),
            output_gate: None,
            output_normalization: None,
            residual_scale: ComponentScalar::new(1.0),
        });
        let mut shape = axes(heads * dim);
        shape[2].name = "component".into();
        g.component_observation(
            mixer_node,
            format!("{path}.attention.channels"),
            "Aggregated attention channels consumed by the output projection",
            shape,
        );
    }
    if policy.feed_forward == FeedForwardPolicy::Dense {
        let prefix = format!("{path}.feed_forward");
        let write_weight = format!("{prefix}.w2.weight");
        let count = c.dense_intermediate_size as usize;
        g.descriptor.components.push(ComponentGroup {
            routed_reads: Vec::new(),
            id: format!("{ffn_node}.units"),
            node_id: ffn_node.into(),
            layer_index: layer,
            count,
            activation: format!("{path}.feed_forward.units"),
            effective_activation: format!("{path}.feed_forward.units.effective"),
            write_input: Some(format!("{path}.feed_forward.write_input")),
            write_output: Some(format!("{path}.feed_forward.write")),
            output: Some(format!("{path}.feed_forward.output")),
            write_partition: ComponentWritePartition::Complete,
            input: format!("{path}.feed_forward.input"),
            reads: vec![
                read(
                    &prefix,
                    "w1",
                    ComponentReadRole::Gate,
                    ComponentRowMapping::Direct { offset: 0 },
                    None,
                ),
                read(
                    &prefix,
                    "w3",
                    ComponentReadRole::Value,
                    ComponentRowMapping::Direct { offset: 0 },
                    None,
                ),
            ],
            write_weight: write_weight.clone(),
            write_input_projection: None,
            shared_write_weight: write_weight,
            write_parameter_group: format!("parameters:{prefix}"),
            write_bias: None,
            activation_equation: ComponentActivation::Gated {
                activation: ComponentNonlinearity::Silu {
                    multiplier: ComponentScalar::new(1.0),
                },
                gate_upper_bound: None,
                value_absolute_bound: None,
                value_offset: ComponentScalar::new(0.0),
            },
            input_normalization: norm(format!("{path}.ffn_norm.weight")),
            output_gate: None,
            output_normalization: None,
            residual_scale: ComponentScalar::new(1.0),
        });
        let mut shape = axes(count);
        shape[2].name = "component".into();
        g.component_observation(
            ffn_node,
            format!("{path}.feed_forward.units"),
            "Gated feed-forward units consumed by the down projection",
            shape,
        );
    }
    for (node, boundary) in [
        (mixer_node, if attention { "attention" } else { "mixer" }),
        (ffn_node, "feed_forward"),
    ] {
        for (suffix, meaning) in [
            ("input", "Normalized operator input"),
            ("write", "Operator write before residual addition"),
            ("output", "Operator contribution before residual addition"),
            ("residual", "Residual after operator addition"),
        ] {
            let suffix = if boundary == "feed_forward"
                && suffix == "output"
                && policy.feed_forward == FeedForwardPolicy::SparseMoe
            {
                "contribution"
            } else {
                suffix
            };
            g.component_observation(
                node,
                format!("{path}.{boundary}.{suffix}"),
                meaning,
                axes(width),
            );
        }
    }
}

pub(super) fn lfm2_readout(g: &mut Builder, c: &crate::lfm2::ModelArgs) {
    readout_observations(g, c.hidden_size as usize, c.vocab_size as usize);
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
            normalization: ComponentNormalization {
                kind: ComponentNormalizationKind::Rms,
                epsilon: ComponentScalar::new(c.norm_eps),
                gain: Some("model.embedding_norm.weight".into()),
                gain_offset: ComponentScalar::new(0.0),
                bias: None,
                groups: 1,
            },
            weight: if c.tie_word_embeddings {
                "model.embed_tokens.weight"
            } else {
                "lm_head.weight"
            }
            .into(),
            bias: None,
            output_transform: ComponentOutputTransform::Identity,
            block_normalizations: vec![],
            other_writes: vec![],
        },
    });
}

pub(super) fn lfm2_other_writes(g: &mut Builder, c: &crate::lfm2::ModelArgs) {
    use crate::lfm2::{FeedForwardPolicy, OperatorPolicy};
    let readout = g
        .descriptor
        .component_readout
        .as_mut()
        .expect("LFM2 readout");
    for (layer, policy) in c.layer_schedule.iter().enumerate() {
        for (present, boundary) in [
            (
                matches!(policy.operator, OperatorPolicy::CausalConvolution),
                "mixer",
            ),
            (
                policy.feed_forward == FeedForwardPolicy::SparseMoe,
                "feed_forward",
            ),
        ] {
            if present {
                let output_suffix = if boundary == "feed_forward" {
                    "contribution"
                } else {
                    "output"
                };
                readout.other_writes.push(ComponentResidualWrite {
                    input: Some(format!("model.layers.{layer}.{boundary}.input")),
                    layer_index: layer,
                    node_id: format!("decoder.layers.{layer}.{boundary}"),
                    output: format!("model.layers.{layer}.{boundary}.{output_suffix}"),
                    effective_output: format!(
                        "model.layers.{layer}.{boundary}.{output_suffix}.effective"
                    ),
                    residual_scale: ComponentScalar::new(1.0),
                });
            }
        }
    }
}
