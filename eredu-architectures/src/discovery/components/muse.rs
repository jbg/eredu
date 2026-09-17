//! Muse's centered pre/post norms, channel gates, and scaled vocabulary readout.
use super::*;
use crate::muse_glimmer::{DecoderConfig, WeightConvention};

fn norm(
    c: &DecoderConfig,
    gain: Option<String>,
    post: bool,
    centered: bool,
) -> ComponentNormalization {
    ComponentNormalization {
        kind: ComponentNormalizationKind::Rms,
        epsilon: ComponentScalar::new(if post {
            c.post_norm_eps
        } else {
            c.rms_norm_eps
        }),
        gain,
        gain_offset: ComponentScalar::new(
            if centered && c.weight_convention == WeightConvention::HuggingFace {
                1.0
            } else {
                0.0
            },
        ),
        bias: None,
        groups: 1,
    }
}

pub(in crate::discovery) fn unit(g: &mut Builder, c: &DecoderConfig, layer: usize) {
    let block = format!("decoder.{layer}");
    let path = format!("model.layers.{layer}");
    let attention = format!("{block}.attention");
    let ffn = format!("{block}.feed_forward");
    let ap = format!("{path}.self_attn");
    let fp = format!("{path}.mlp");
    let width = c.hidden_size as usize;
    let heads = c.num_attention_heads as usize;
    let kv = c.num_key_value_heads as usize;
    let dim = c.head_dim as usize;
    g.attach_parameters(&attention, &ap);
    g.attach_parameters(&ffn, &fp);
    // Generic composite topology declares both original and effective boundaries.
    // Muse emits that same pair inside its block, so no outer copy is declared.
    for (boundary, node, field, post) in [
        ("attention.input", &attention, "input_layernorm", false),
        (
            "attention.output",
            &attention,
            "post_attention_layernorm",
            true,
        ),
        (
            "feed_forward.input",
            &ffn,
            "pre_feedforward_layernorm",
            false,
        ),
        (
            "feed_forward.output",
            &ffn,
            "post_feedforward_layernorm",
            true,
        ),
    ] {
        let prefix = format!("{path}.{field}");
        let id = format!("{block}.{field}");
        g.node(
            &id,
            ArchitectureNodeKind::Normalization,
            Some(&block),
            Some(&prefix),
            Some(width),
        );
        if boundary != "feed_forward.output" || c.num_experts > 0 {
            g.component_observation(
                node,
                format!("{path}.{boundary}"),
                "Muse normalized sublayer value",
                axes(width),
            );
        }
        if post {
            let branch = boundary.split('.').next().unwrap();
            g.descriptor
                .component_transforms
                .push(ComponentTensorTransform {
                    id,
                    node_id: node.clone(),
                    input: format!("{path}.{branch}.write.effective"),
                    output: format!("{path}.{boundary}"),
                    effective_output: Some(format!("{path}.{boundary}.effective")),
                    equation: ComponentTensorTransformEquation::Normalization {
                        normalization: norm(c, Some(format!("{prefix}.weight")), true, true),
                    },
                });
        }
    }
    for (node, branch) in [(&attention, "attention"), (&ffn, "feed_forward")] {
        for suffix in ["write", "residual"] {
            if branch != "feed_forward" || suffix != "write" || c.num_experts > 0 {
                g.component_observation(
                    node,
                    format!("{path}.{branch}.{suffix}"),
                    "Muse projection write or updated residual",
                    axes(width),
                );
            }
        }
    }
    let read = |field: &str, role, rows, head_count: Option<usize>| {
        let weight = format!("{ap}.{field}.weight");
        let head_normalization = head_count.map(|head_count| {
            let query = role == ComponentReadRole::Query;
            let gguf = c.weight_convention == WeightConvention::Gguf;
            ComponentHeadNormalization {
                output_scale: ComponentScalar::new(if query && !gguf {
                    c.qk_scale_factor
                } else {
                    1.0
                }),
                heads: head_count,
                head_width: dim,
                independent_gains: false,
                normalization: norm(
                    c,
                    gguf.then(|| format!("{ap}.{}_norm.weight", if query { "q" } else { "k" })),
                    false,
                    false,
                ),
            }
        });
        ComponentRead {
            source: None,
            projection_output: None,
            role,
            shared_weight: weight.clone(),
            weight,
            parameter_group: format!("parameters:{ap}"),
            bias: None,
            rows,
            head_normalization,
            input_projections: vec![],
        }
    };
    let head_rows = |sharing| ComponentRowMapping::HeadRows {
        offset: 0,
        component_head_width: dim,
        read_head_width: dim,
        component_heads_per_read_head: sharing,
        read_head_stride: None,
    };
    g.descriptor.components.push(ComponentGroup {
        id: format!("{attention}.channels"),
        node_id: attention.clone(),
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
                "q_proj",
                ComponentReadRole::Query,
                head_rows(1),
                Some(heads),
            ),
            read(
                "k_proj",
                ComponentReadRole::Key,
                head_rows(heads / kv),
                Some(kv),
            ),
            read(
                "v_proj",
                ComponentReadRole::Value,
                ComponentRowMapping::GroupedQuery {
                    offset: 0,
                    head_width: dim,
                    queries_per_kv: heads / kv,
                },
                None,
            ),
            read(
                "gate_proj",
                ComponentReadRole::OutputGate,
                ComponentRowMapping::Direct { offset: 0 },
                None,
            ),
        ],
        routed_reads: vec![],
        write_weight: format!("{ap}.o_proj.weight"),
        shared_write_weight: format!("{ap}.o_proj.weight"),
        write_parameter_group: format!("parameters:{ap}"),
        write_input_projection: None,
        write_bias: None,
        activation_equation: ComponentActivation::Attention {
            query_heads: heads,
            key_value_heads: kv,
            head_width: dim,
            output_gate: Some(ComponentNonlinearity::Sigmoid),
        },
        input_normalization: norm(
            c,
            Some(format!("{path}.input_layernorm.weight")),
            false,
            true,
        ),
        output_gate: None,
        output_normalization: Some(norm(
            c,
            Some(format!("{path}.post_attention_layernorm.weight")),
            true,
            true,
        )),
        residual_scale: ComponentScalar::new(1.0),
    });
    let mut channels = axes(heads * dim);
    channels[2].name = "component".into();
    g.component_observation(
        &attention,
        format!("{path}.attention.channels"),
        "Value aggregation after per-channel sigmoid gating",
        channels,
    );
    let input_norm = norm(
        c,
        Some(format!("{path}.pre_feedforward_layernorm.weight")),
        false,
        true,
    );
    if c.num_experts == 0 {
        gated_units(
            g,
            &ffn,
            layer,
            &format!("parameters:{fp}"),
            &fp,
            &format!("{path}.feed_forward"),
            &format!("{path}.feed_forward.input"),
            c.intermediate_size as usize,
            width,
            input_norm,
            ComponentWritePartition::Complete,
        );
        g.descriptor
            .components
            .last_mut()
            .unwrap()
            .output_normalization = Some(norm(
            c,
            Some(format!("{path}.post_feedforward_layernorm.weight")),
            true,
            true,
        ));
    } else {
        super::super::routed_components::gated(
            g,
            super::super::routed_components::Site {
                node: &ffn,
                layer,
                routing: &format!("{path}.routing"),
                input: Some(format!("{path}.feed_forward.input")),
                normalization: Some(input_norm),
                // Postnorm acts on the complete route-weighted sum, not each expert.
                residual_scale: None,
            },
            crate::muse_glimmer::text::expert_bank_spec(c, layer),
        );
    }
}

pub(in crate::discovery) fn readout(g: &mut Builder, c: &DecoderConfig) {
    let width = c.hidden_size as usize;
    g.attach_parameters("embedding", "model.embed_tokens");
    g.attach_parameters(
        "output",
        if c.tie_word_embeddings {
            "model.embed_tokens"
        } else {
            "lm_head"
        },
    );
    g.node(
        "output.norm",
        ArchitectureNodeKind::Normalization,
        Some("output"),
        Some("model.norm"),
        Some(width),
    );
    for (node, path, meaning) in [
        (
            "assembly",
            "readout.embedding",
            "Assembled normalized text rows and projected media entering the decoder",
        ),
        (
            "output.norm",
            "readout.residual",
            "Residual before final normalization",
        ),
        (
            "output.norm",
            "readout.normalized",
            "Normalized residual before the vocabulary projection",
        ),
    ] {
        g.component_observation(node, path.into(), meaning, axes(width));
    }
    let mut scores = axes(c.vocab_size as usize);
    scores[2].name = "vocabulary".into();
    g.component_observation(
        "output",
        "readout.linear".into(),
        "Actual affine scores before output multiplier and softcap",
        scores,
    );
    g.descriptor.component_readout = Some(ComponentReadout {
        embedding: "readout.embedding".into(),
        embedding_weight: "model.embed_tokens.weight".into(),
        embedding_scale: ComponentScalar::new(1.0),
        embedding_normalization: None,
        token_embedding_normalization: Some(norm(c, None, false, false)),
        tied_embeddings: c.tie_word_embeddings,
        equation: ComponentReadoutEquation {
            block_transforms: vec![],
            stream_residual: None,
            residual: "readout.residual".into(),
            normalized: "readout.normalized".into(),
            projection_input: Some("readout.projection_input".into()),
            linear_scores: "readout.linear".into(),
            logits: MODEL_LOGITS_OBSERVATION_PATH.into(),
            normalization: norm(c, Some("model.norm.weight".into()), false, false),
            weight: if c.tie_word_embeddings {
                "model.embed_tokens.weight"
            } else {
                "lm_head.weight"
            }
            .into(),
            bias: None,
            score_writes: vec![],
            output_transform: ComponentOutputTransform::ScaledSoftcap {
                scale: ComponentScalar::new(c.output_multiplier),
                cap: ComponentScalar::new(c.final_logit_softcapping),
            },
            block_normalizations: vec![],
            other_writes: if c.num_experts > 0 {
                (0..c.num_hidden_layers as usize)
                    .map(|layer| ComponentResidualWrite {
                        input: Some(format!("model.layers.{layer}.feed_forward.input")),
                        layer_index: layer,
                        node_id: format!("decoder.{layer}.feed_forward"),
                        output: format!("model.layers.{layer}.feed_forward.output"),
                        effective_output: format!(
                            "model.layers.{layer}.feed_forward.output.effective"
                        ),
                        residual_scale: ComponentScalar::new(1.0),
                    })
                    .collect()
            } else {
                vec![]
            },
        },
    });
}
