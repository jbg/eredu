//! MLA reads retain their normalized latent stages instead of implying a single affine read.
use super::*;
use crate::deepseek::{LayerPolicy, V3Args};

mod prediction;
pub(in crate::discovery) use prediction::declare as prediction_scopes;

fn norm(c: &V3Args, gain: String) -> ComponentNormalization {
    ComponentNormalization {
        kind: ComponentNormalizationKind::Rms,
        epsilon: ComponentScalar::new(c.rms_norm_eps),
        gain: Some(gain),
        gain_offset: ComponentScalar::new(0.0),
        bias: None,
        groups: 1,
    }
}

pub(in crate::discovery) fn unit(
    g: &mut Builder,
    c: &V3Args,
    layer: usize,
    path: &str,
    attention: &str,
    ffn: &str,
) {
    let width = c.hidden_size as usize;
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
    let reads = vec![
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
        output: Some(format!("{path}.compressed_attention.output")),
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
    for (node, suffix) in [
        (attention, "attention.input"),
        (attention, "attention.write"),
        (attention, "compressed_attention.output"),
        (attention, "attention.residual"),
        (ffn, "feed_forward.input"),
        (ffn, "feed_forward.residual"),
    ] {
        g.component_observation(
            node,
            format!("{path}.{suffix}"),
            "Actual V3 sublayer boundary",
            axes(width),
        );
    }
    if c.layer_schedule.get(layer) == Some(&LayerPolicy::DenseMlp) {
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
            norm(c, format!("{path}.post_attention_layernorm.weight")),
            ComponentWritePartition::Complete,
        );
    } else {
        g.component_observation(
            ffn,
            format!("{path}.feed_forward.contribution"),
            "Complete routed and shared contribution after expert-output interventions",
            axes(width),
        );
    }
}

pub(in crate::discovery) fn shared_units(
    g: &mut Builder,
    c: &V3Args,
    layer: usize,
    path: &str,
    ffn: &str,
) {
    g.component_observation(
        &format!("{ffn}.shared"),
        format!("{path}.feed_forward.shared.input"),
        "Normalized shared-expert input",
        axes(c.hidden_size as usize),
    );
    gated_units(
        g,
        &format!("{ffn}.shared"),
        layer,
        &format!("parameters:{path}.mlp"),
        &format!("{path}.mlp.shared_experts"),
        &format!("{path}.feed_forward.shared"),
        &format!("{path}.feed_forward.shared.input"),
        (c.moe_intermediate_size * c.n_shared_experts) as usize,
        c.hidden_size as usize,
        norm(c, format!("{path}.post_attention_layernorm.weight")),
        ComponentWritePartition::TensorParallelSum,
    );
}

pub(in crate::discovery) fn readout(g: &mut Builder, c: &V3Args) {
    readout_observations(g, c.hidden_size as usize, c.vocab_size as usize);
    g.descriptor.component_readout = Some(ComponentReadout {
        token_embedding_normalization: None,
        embedding_normalization: None,
        embedding: "readout.embedding".into(),
        embedding_weight: "model.embed_tokens.weight".into(),
        embedding_scale: ComponentScalar::new(1.0),
        tied_embeddings: false,
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
            weight: "lm_head.weight".into(),
            bias: None,
            output_transform: ComponentOutputTransform::Identity,
            block_normalizations: vec![],
            other_writes: c
                .layer_schedule
                .iter()
                .enumerate()
                .filter(|(_, p)| **p == LayerPolicy::SparseMoe)
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
