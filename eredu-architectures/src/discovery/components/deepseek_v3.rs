//! MLA reads retain their normalized latent stages instead of implying a single affine read.
use super::*;
use crate::deepseek::{LayerPolicy, V3Args};

mod prediction;
pub(in crate::discovery) use prediction::declare as prediction_scopes;

fn norm(c: &V3Args, gain: String) -> ComponentNormalization {
    let spec = crate::deepseek::v3_specs::normalization_spec(c, &gain, c.hidden_size)
        .expect("admitted V3 normalization declaration");
    declared_normalization(&spec)
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
    let prefix = format!("{path}.self_attn");
    let spec =
        crate::deepseek::v3_specs::attention_spec(c, &prefix).expect("admitted V3 MLA declaration");
    let heads = spec.geometry.heads() as usize;
    let value = spec.geometry.value_dimensions() as usize;
    let nope = spec.geometry.non_positional_dimensions() as usize;
    let rope = spec.geometry.positional_dimensions() as usize;
    let latent = spec.geometry.latent_dimensions() as usize;
    let read = |projection: &eredu_nn::LinearSpec, role, rows, input_projections| ComponentRead {
        input_projections,
        ..declared_component_read(projection, format!("parameters:{prefix}"), role, rows)
    };
    let stage = |projection: &eredu_nn::LinearSpec,
                 normalization: &eredu_nn::NormalizationConstructionSpec,
                 output: &str| {
        declared_input_projection(
            projection,
            format!("parameters:{prefix}"),
            0..normalization.dimensions as usize,
            Some(declared_normalization(normalization)),
            format!("{path}.attention.{output}.effective"),
        )
    };
    let head_rows = |offset, count, sharing, stride| ComponentRowMapping::HeadRows {
        offset,
        component_head_width: value,
        read_head_width: count,
        component_heads_per_read_head: sharing,
        read_head_stride: stride,
    };
    let (query, query_input, query_rank) = match &spec.query {
        crate::deepseek::v3_specs::QuerySpec::LowRank(query) => (
            &query.second,
            vec![stage(
                query
                    .first
                    .as_ref()
                    .expect("declared query input projection"),
                &query.normalization,
                "query.latent",
            )],
            Some(query.normalization.dimensions as usize),
        ),
        crate::deepseek::v3_specs::QuerySpec::Direct(query) => (query, vec![], None),
    };
    let kv_stage = stage(&spec.kv_a, &spec.kv_norm, "key_value.latent");
    let reads = vec![
        read(
            query,
            ComponentReadRole::Query,
            head_rows(0, spec.geometry.query_head_dimensions() as usize, 1, None),
            query_input,
        ),
        read(
            &spec.kv_b,
            ComponentReadRole::Key,
            head_rows(
                0,
                nope,
                1,
                Some(
                    spec.geometry
                        .reconstruction_head_dimensions()
                        .expect("admitted fused MLA head geometry") as usize,
                ),
            ),
            vec![kv_stage.clone()],
        ),
        // The rotary key is shared by all heads and bypasses latent normalization.
        read(
            &spec.kv_a,
            ComponentReadRole::Key,
            head_rows(latent, rope, heads, None),
            vec![],
        ),
        read(
            &spec.kv_b,
            ComponentReadRole::Value,
            ComponentRowMapping::Blocked {
                offset: nope,
                block_width: value,
                block_stride: spec
                    .geometry
                    .reconstruction_head_dimensions()
                    .expect("admitted fused MLA head geometry")
                    as usize,
            },
            vec![kv_stage],
        ),
    ];
    let write = spec.output.weight.id.as_str().to_owned();
    g.descriptor.components.push(ComponentGroup {
        id: format!("{attention}.channels"),
        node_id: attention.into(),
        layer_index: layer,
        count: spec.output.input as usize,
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
            attended_head_normalization: None,
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
    let mut channels = axes(spec.output.input as usize);
    channels[2].name = "component".into();
    g.component_observation(
        attention,
        format!("{path}.attention.channels"),
        "Aggregated MLA value channels before output projection",
        channels,
    );
    for (suffix, count) in [
        ("query.latent", query_rank),
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
        let projections = crate::deepseek::v3_specs::dense_parameters(c, &format!("{path}.mlp"))
            .expect("admitted V3 dense declaration");
        gated_units(
            g,
            ffn,
            layer,
            &format!("parameters:{path}.mlp"),
            projections.projections(),
            projections.policy,
            &format!("{path}.feed_forward"),
            &format!("{path}.feed_forward.input"),
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
    let declaration = crate::deepseek::v3::routed_spec(c, layer, None)
        .expect("admitted V3 routed/shared declaration");
    let projections = &declaration.shared;
    gated_units(
        g,
        &format!("{ffn}.shared"),
        layer,
        &format!("parameters:{path}.mlp"),
        projections.projections(),
        projections.policy,
        &format!("{path}.feed_forward.shared"),
        &format!("{path}.feed_forward.shared.input"),
        norm(c, format!("{path}.post_attention_layernorm.weight")),
        ComponentWritePartition::TensorParallelSum,
    );
}

pub(in crate::discovery) fn readout(g: &mut Builder, c: &V3Args) {
    let statics = &c.parameters().statics;
    let embedding = statics.embedding_spec();
    let output = statics.head_spec().expect("untied V3 output declaration");
    readout_observations(
        g,
        embedding.dimensions as usize,
        embedding.vocabulary as usize,
    );
    g.descriptor.component_readout = Some(ComponentReadout {
        token_embedding_normalization: None,
        embedding_normalization: None,
        embedding: "readout.embedding".into(),
        embedding_weight: embedding.weight.id.as_str().into(),
        embedding_scale: ComponentScalar::new(1.0),
        tied_embeddings: statics.head_spec().is_none(),
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
            normalization: declared_normalization(statics.normalization_spec()),
            weight: output.weight.id.as_str().into(),
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
