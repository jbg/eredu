//! KDA and MLA declarations retain recurrent and latent read equations.
use super::*;
use crate::kimi_linear::{AttentionKind, FeedForwardPolicy, ModelArgs};

pub(in crate::discovery) fn unit(
    g: &mut Builder,
    c: &ModelArgs,
    layer: usize,
    path: &str,
    attention: &str,
    ffn: &str,
) {
    let width = c.hidden_size as usize;
    let policy = c.layer_schedule.get(layer).expect("declared Kimi layer");
    if policy.attention == AttentionKind::Kda {
        kda(g, c, layer, path, attention);
    } else {
        mla(g, c, layer, path, attention);
    }
    for (node, suffix) in [
        (attention, "attention.input"),
        (attention, "attention.write"),
        (attention, "attention.output"),
        (attention, "attention.residual"),
        (ffn, "feed_forward.input"),
        (ffn, "feed_forward.residual"),
    ] {
        g.component_observation(
            node,
            format!("{path}.{suffix}"),
            "Consumed Kimi sublayer boundary",
            axes(width),
        );
    }
    let normalization = norm(c, format!("{path}.post_attention_layernorm.weight"));
    if policy.feed_forward == FeedForwardPolicy::Dense {
        gated_units(
            g,
            ffn,
            layer,
            &format!("parameters:{path}.mlp"),
            &format!("{path}.mlp"),
            &format!("{path}.feed_forward"),
            &format!("{path}.feed_forward.input"),
            c.intermediate_size as usize,
            width,
            normalization,
            ComponentWritePartition::Complete,
        );
    } else {
        g.component_observation(
            ffn,
            format!("{path}.feed_forward.contribution"),
            "Complete routed and shared contribution after expert-output interventions",
            axes(width),
        );
        if c.num_shared_experts > 0 {
            gated_units(
                g,
                &format!("{ffn}.shared"),
                layer,
                &format!("parameters:{path}.mlp"),
                &format!("{path}.mlp.shared_experts"),
                &format!("{path}.mlp.shared_experts.feed_forward"),
                &format!("{path}.feed_forward.input"),
                (c.num_shared_experts * c.moe_intermediate_size) as usize,
                width,
                normalization,
                ComponentWritePartition::Complete,
            );
        }
    }
}

fn kda(g: &mut Builder, c: &ModelArgs, layer: usize, path: &str, node: &str) {
    let heads = c.kda_config.num_heads as usize;
    let dim = c.kda_config.head_dim as usize;
    let count = heads * dim;
    let prefix = format!("{path}.self_attn");
    let parameter = |name: &str| {
        let parameter = format!("{prefix}.{name}");
        ComponentTransformParameter {
            shared_parameter: parameter.clone(),
            parameter,
            parameter_group: format!("parameters:{prefix}"),
        }
    };
    let head_rows = |read_head_width| ComponentRowMapping::HeadRows {
        offset: 0,
        component_head_width: dim,
        read_head_width,
        component_heads_per_read_head: 1,
        read_head_stride: None,
    };
    let head_norm = || ComponentHeadNormalization {
        output_scale: ComponentScalar::new(1.0),
        heads,
        head_width: dim,
        independent_gains: false,
        normalization: ComponentNormalization {
            kind: ComponentNormalizationKind::Rms,
            epsilon: ComponentScalar::new(1e-6),
            gain: None,
            gain_offset: ComponentScalar::new(0.0),
            bias: None,
            groups: 1,
        },
    };
    let read = |field: &str, role, rows, projections, head_normalization, projection_output| {
        let weight = format!("{prefix}.{field}.weight");
        ComponentRead {
            source: None,
            shared_weight: weight.clone(),
            weight,
            parameter_group: format!("parameters:{prefix}"),
            role,
            rows,
            bias: None,
            input_projections: projections,
            head_normalization,
            projection_output,
        }
    };
    let mut reads = Vec::new();
    for (field, suffix, role, rows, normalization) in [
        (
            "q",
            "query",
            ComponentReadRole::Query,
            head_rows(dim),
            Some(head_norm()),
        ),
        (
            "k",
            "key",
            ComponentReadRole::Key,
            head_rows(dim),
            Some(head_norm()),
        ),
        (
            "v",
            "value",
            ComponentReadRole::Value,
            ComponentRowMapping::Direct { offset: 0 },
            None,
        ),
    ] {
        let input = format!("{path}.attention.{suffix}.projected");
        let output = format!("{path}.attention.{suffix}.convolved");
        reads.push(read(
            &format!("{field}_proj"),
            role,
            rows,
            vec![],
            normalization,
            Some(input.clone()),
        ));
        for boundary in [&input, &output] {
            g.read_only_observation(
                node,
                boundary.clone(),
                "Current-position KDA read before recurrence",
                axes(count),
            );
        }
        g.descriptor
            .component_transforms
            .push(ComponentTensorTransform {
                id: format!("{node}.{suffix}.convolution"),
                node_id: node.into(),
                input,
                output,
                effective_output: None,
                equation: ComponentTensorTransformEquation::CausalDepthwiseConvolution {
                    kernel: parameter(&format!("{field}_conv1d.weight")),
                    channels: count,
                    taps: c.kda_config.short_conv_kernel_size as usize,
                    residual: false,
                    activation: Some(ComponentNonlinearity::Silu {
                        multiplier: ComponentScalar::new(1.0),
                    }),
                },
            });
    }
    for (prefix_field, suffix, role, rows) in [
        ("f", "decay", ComponentReadRole::Decay, head_rows(dim)),
        (
            "g",
            "gate",
            ComponentReadRole::OutputGate,
            ComponentRowMapping::Direct { offset: 0 },
        ),
    ] {
        let weight = format!("{prefix}.{prefix_field}_a_proj.weight");
        let output = format!("{path}.attention.{suffix}.latent");
        g.read_only_observation(
            node,
            output.clone(),
            "Current-position low-rank recurrent control input",
            axes(dim),
        );
        reads.push(read(
            &format!("{prefix_field}_b_proj"),
            role,
            rows,
            vec![ComponentInputProjection {
                shared_weight: weight.clone(),
                weight,
                parameter_group: format!("parameters:{prefix}"),
                bias: None,
                rows: 0..dim,
                normalization: None,
                output,
            }],
            None,
            None,
        ));
    }
    reads.push(read(
        "b_proj",
        ComponentReadRole::Update,
        head_rows(1),
        vec![],
        None,
        None,
    ));
    let write = format!("{prefix}.o_proj.weight");
    g.descriptor.components.push(ComponentGroup {
        id: format!("{node}.channels"),
        node_id: node.into(),
        layer_index: layer,
        count,
        activation: format!("{path}.attention.channels"),
        effective_activation: format!("{path}.attention.channels.effective"),
        write_input: Some(format!("{path}.attention.write_input")),
        write_output: Some(format!("{path}.attention.write")),
        output: Some(format!("{path}.attention.output")),
        write_partition: ComponentWritePartition::Complete,
        input: format!("{path}.attention.input"),
        reads,
        routed_reads: vec![],
        shared_write_weight: write.clone(),
        write_weight: write,
        write_input_projection: None,
        write_parameter_group: format!("parameters:{prefix}"),
        write_bias: None,
        activation_equation: ComponentActivation::GatedDeltaAttention {
            key_heads: heads,
            value_heads: heads,
            key_head_width: dim,
            value_head_width: dim,
            decay_layout: ComponentDeltaDecay::KeyChannel,
            query_scale: ComponentScalar::new(1.0 / dim as f32),
            key_scale: ComponentScalar::new((dim as f32).sqrt().recip()),
            decay_rate: parameter("A_log"),
            decay_bias: parameter("dt_bias"),
            channel_normalization: ComponentHeadNormalization {
                output_scale: ComponentScalar::new(1.0),
                heads,
                head_width: dim,
                independent_gains: false,
                normalization: norm(c, format!("{prefix}.o_norm.weight")),
            },
            output_gate: ComponentNonlinearity::Sigmoid,
        },
        input_normalization: norm(c, format!("{path}.input_layernorm.weight")),
        output_gate: None,
        output_normalization: None,
        residual_scale: ComponentScalar::new(1.0),
    });
    let mut channels = axes(count);
    channels[2].name = "component".into();
    g.component_observation(
        node,
        format!("{path}.attention.channels"),
        "Recurrent channels after head normalization and sigmoid gating",
        channels,
    );
}

fn norm(c: &ModelArgs, gain: String) -> ComponentNormalization {
    ComponentNormalization {
        kind: ComponentNormalizationKind::Rms,
        epsilon: ComponentScalar::new(c.rms_norm_eps),
        gain: Some(gain),
        gain_offset: ComponentScalar::new(0.0),
        bias: None,
        groups: 1,
    }
}

fn mla(g: &mut Builder, c: &ModelArgs, layer: usize, path: &str, attention: &str) {
    let heads = c.num_attention_heads as usize;
    let value = c.v_head_dim as usize;
    let nope = c.qk_nope_head_dim as usize;
    let rope = c.qk_rope_head_dim as usize;
    let latent = c.kv_lora_rank as usize;
    let prefix = format!("{path}.self_attn");
    let read = |field: &str, role, rows, input_projections| {
        let weight = format!("{prefix}.{field}.weight");
        ComponentRead {
            source: None,
            projection_output: None,
            role,
            shared_weight: weight.clone(),
            weight,
            parameter_group: format!("parameters:{prefix}"),
            bias: None,
            rows,
            head_normalization: None,
            input_projections,
        }
    };
    let stage = |field: &str, gain: &str, count: usize, output: &str| {
        let weight = format!("{prefix}.{field}.weight");
        ComponentInputProjection {
            shared_weight: weight.clone(),
            weight,
            parameter_group: format!("parameters:{prefix}"),
            bias: None,
            rows: 0..count,
            normalization: Some(norm(c, format!("{prefix}.{gain}.weight"))),
            output: format!("{path}.attention.{output}.effective"),
        }
    };
    let head_rows = |offset, count, sharing, stride| ComponentRowMapping::HeadRows {
        offset,
        component_head_width: value,
        read_head_width: count,
        component_heads_per_read_head: sharing,
        read_head_stride: stride,
    };
    let (query_field, query_input) = match c.q_lora_rank {
        Some(rank) => (
            "q_b_proj",
            vec![stage(
                "q_a_proj",
                "q_a_layernorm",
                rank as usize,
                "query.latent",
            )],
        ),
        None => ("q_proj", vec![]),
    };
    let kv_stage = stage(
        "kv_a_proj_with_mqa",
        "kv_a_layernorm",
        latent,
        "key_value.latent",
    );
    let mut reads = vec![
        read(
            query_field,
            ComponentReadRole::Query,
            head_rows(0, nope + rope, 1, None),
            query_input,
        ),
        read(
            "kv_b_proj",
            ComponentReadRole::Key,
            head_rows(0, nope, 1, Some(nope + value)),
            vec![kv_stage.clone()],
        ),
        // The rotary key is shared by all heads and bypasses latent normalization.
        read(
            "kv_a_proj_with_mqa",
            ComponentReadRole::Key,
            head_rows(latent, rope, heads, None),
            vec![],
        ),
        read(
            "kv_b_proj",
            ComponentReadRole::Value,
            ComponentRowMapping::Blocked {
                offset: nope,
                block_width: value,
                block_stride: nope + value,
            },
            vec![kv_stage],
        ),
    ];
    if c.split_kv_b {
        reads[1].weight = format!("{prefix}.k_b_proj.weight");
        reads[1].shared_weight = reads[1].weight.clone();
        reads[1].rows = head_rows(0, nope, 1, None);
        reads[3].weight = format!("{prefix}.v_b_proj.weight");
        reads[3].shared_weight = reads[3].weight.clone();
        reads[3].rows = ComponentRowMapping::Direct { offset: 0 };
    }
    let write = format!("{prefix}.o_proj.weight");
    g.descriptor.components.push(ComponentGroup {
        id: format!("{attention}.channels"),
        node_id: attention.into(),
        layer_index: layer,
        count: heads * value,
        activation: format!("{path}.attention.channels"),
        effective_activation: format!("{path}.attention.channels.effective"),
        write_input: Some(format!("{path}.attention.write_input")),
        write_output: Some(format!("{path}.attention.write")),
        output: Some(format!("{path}.attention.output")),
        write_partition: ComponentWritePartition::Complete,
        input: format!("{path}.attention.input"),
        reads,
        routed_reads: vec![],
        shared_write_weight: write.clone(),
        write_weight: write,
        write_input_projection: None,
        write_parameter_group: format!("parameters:{prefix}"),
        write_bias: None,
        activation_equation: ComponentActivation::Attention {
            query_heads: heads,
            key_value_heads: heads,
            head_width: value,
            output_gate: None,
        },
        input_normalization: norm(c, format!("{path}.input_layernorm.weight")),
        output_gate: None,
        output_normalization: None,
        residual_scale: ComponentScalar::new(1.0),
    });
    let mut channels = axes(heads * value);
    channels[2].name = "component".into();
    g.component_observation(
        attention,
        format!("{path}.attention.channels"),
        "Aggregated MLA value channels before output projection",
        channels,
    );
    for (suffix, count) in [
        ("query.latent", c.q_lora_rank.map(|n| n as usize)),
        ("key_value.latent", Some(latent)),
    ] {
        if let Some(count) = count {
            g.component_observation(
                attention,
                format!("{path}.attention.{suffix}"),
                "Normalized latent projection for current input positions",
                axes(count),
            );
        }
    }
}

pub(in crate::discovery) fn readout(g: &mut Builder, c: &ModelArgs) {
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
            normalization: norm(c, "model.norm.weight".into()),
            weight: if c.tie_word_embeddings {
                "model.embed_tokens.weight"
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
                .filter(|(_, p)| p.feed_forward == FeedForwardPolicy::SparseMoe)
                .map(|(layer, _)| ComponentResidualWrite {
                    input: Some(format!("model.layers.{layer}.feed_forward.input")),
                    layer_index: layer,
                    node_id: format!("decoder.layers.{layer}.feed_forward"),
                    output: format!("model.layers.{layer}.feed_forward.contribution"),
                    effective_output: format!(
                        "model.layers.{layer}.feed_forward.contribution.effective"
                    ),
                    residual_scale: ComponentScalar::new(1.0),
                })
                .collect(),
        },
    });
}
