//! KDA and MLA declarations retain recurrent and latent read equations.
use super::*;
use crate::kimi_linear::{
    declarations::{FeedForwardParameters as F, MixerParameters as M},
    Declaration,
};

pub(in crate::discovery) fn unit(
    g: &mut Builder,
    declaration: &Declaration,
    layer: usize,
    path: &str,
    attention: &str,
    ffn: &str,
) {
    let width = declaration.statics.embedding_spec().dimensions as usize;
    let scope = &declaration.blocks[layer];
    match &scope.mixer {
        M::Kda(spec) => kda(g, spec, &scope.input_norm, layer, path, attention),
        M::Mla(spec) => mla(g, spec, &scope.input_norm, layer, path, attention),
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
    let normalization = declared_normalization(&scope.post_attention_norm);
    if let F::Dense(parameters) = &scope.feed_forward {
        gated_units(
            g,
            ffn,
            layer,
            &format!("parameters:{path}.mlp"),
            parameters.projections(),
            parameters.policy,
            &format!("{path}.feed_forward"),
            &format!("{path}.feed_forward.input"),
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
        if let F::Sparse(parameters) = &scope.feed_forward {
            let parameters = &parameters.shared;
            gated_units(
                g,
                &format!("{ffn}.shared"),
                layer,
                &format!("parameters:{path}.mlp"),
                parameters.projections(),
                parameters.policy,
                &format!("{path}.mlp.shared_experts.feed_forward"),
                &format!("{path}.feed_forward.input"),
                normalization,
                ComponentWritePartition::Complete,
            );
        }
    }
}

fn kda(
    g: &mut Builder,
    spec: &crate::kimi_linear::specs::KdaSpec,
    input_norm: &eredu_nn::NormalizationConstructionSpec,
    layer: usize,
    path: &str,
    node: &str,
) {
    let prefix = format!("{path}.self_attn");
    let heads = spec.geometry.value_heads() as usize;
    let dim = spec.geometry.value_dimensions() as usize;
    let count = spec.output.input as usize;
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
    let read = |projection: &eredu_nn::LinearSpec,
                role,
                rows,
                projections,
                head_normalization,
                projection_output| {
        ComponentRead {
            projection_output,
            head_normalization,
            input_projections: projections,
            ..declared_component_read(projection, format!("parameters:{prefix}"), role, rows)
        }
    };
    let mut reads = Vec::new();
    for (projection, convolution, suffix, role, rows, normalization) in [
        (
            &spec.q,
            &spec.convolutions[0],
            "query",
            ComponentReadRole::Query,
            head_rows(dim),
            Some(head_norm()),
        ),
        (
            &spec.k,
            &spec.convolutions[1],
            "key",
            ComponentReadRole::Key,
            head_rows(dim),
            Some(head_norm()),
        ),
        (
            &spec.v,
            &spec.convolutions[2],
            "value",
            ComponentReadRole::Value,
            ComponentRowMapping::Direct { offset: 0 },
            None,
        ),
    ] {
        let input = format!("{path}.attention.{suffix}.projected");
        let output = format!("{path}.attention.{suffix}.convolved");
        reads.push(read(
            projection,
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
                    kernel: declared_transform_parameter(
                        &convolution.weight,
                        format!("parameters:{prefix}"),
                    ),
                    channels: convolution.channels as usize,
                    taps: convolution.kernel_size as usize,
                    residual: false,
                    activation: match convolution.activation {
                        eredu_nn::ConvolutionActivation::Identity => None,
                        eredu_nn::ConvolutionActivation::Silu => {
                            Some(ComponentNonlinearity::Silu {
                                multiplier: ComponentScalar::new(1.0),
                            })
                        }
                    },
                },
            });
    }
    for (first, second, suffix, role, rows) in [
        (
            &spec.f_a,
            &spec.f_b,
            "decay",
            ComponentReadRole::Decay,
            head_rows(dim),
        ),
        (
            &spec.g_a,
            &spec.g_b,
            "gate",
            ComponentReadRole::OutputGate,
            ComponentRowMapping::Direct { offset: 0 },
        ),
    ] {
        let output = format!("{path}.attention.{suffix}.latent");
        g.read_only_observation(
            node,
            output.clone(),
            "Current-position low-rank recurrent control input",
            axes(first.output as usize),
        );
        reads.push(read(
            second,
            role,
            rows,
            vec![declared_input_projection(
                first,
                format!("parameters:{prefix}"),
                0..first.output as usize,
                None,
                output,
            )],
            None,
            None,
        ));
    }
    reads.push(read(
        &spec.beta,
        ComponentReadRole::Update,
        head_rows(1),
        vec![],
        None,
        None,
    ));
    let write = spec.output.weight.id.as_str().to_owned();
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
        shared_write_weight: declared_shared_weight(&spec.output.weight).into(),
        write_weight: write,
        write_input_projection: None,
        write_parameter_group: format!("parameters:{prefix}"),
        write_bias: spec.output.bias.as_ref().map(|bias| bias.id.to_string()),
        activation_equation: ComponentActivation::GatedDeltaAttention {
            key_heads: heads,
            value_heads: heads,
            key_head_width: dim,
            value_head_width: dim,
            decay_layout: ComponentDeltaDecay::KeyChannel,
            query_scale: ComponentScalar::new(1.0 / dim as f32),
            key_scale: ComponentScalar::new((dim as f32).sqrt().recip()),
            decay_rate: declared_transform_parameter(&spec.a_log, format!("parameters:{prefix}")),
            decay_bias: declared_transform_parameter(&spec.dt_bias, format!("parameters:{prefix}")),
            channel_normalization: ComponentHeadNormalization {
                output_scale: ComponentScalar::new(1.0),
                heads,
                head_width: dim,
                independent_gains: false,
                normalization: declared_normalization(&spec.norm),
            },
            output_gate: ComponentNonlinearity::Sigmoid,
        },
        input_normalization: declared_normalization(input_norm),
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

fn mla(
    g: &mut Builder,
    spec: &crate::kimi_linear::specs::MlaSpec,
    input_norm: &eredu_nn::NormalizationConstructionSpec,
    layer: usize,
    path: &str,
    attention: &str,
) {
    use crate::kimi_linear::specs::{LatentSpec, QuerySpec};
    let prefix = format!("{path}.self_attn");
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
        QuerySpec::LowRank(query) => (
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
        QuerySpec::Direct(query) => (query, vec![], None),
    };
    let kv_stage = stage(&spec.kv_a, &spec.kv_norm, "key_value.latent");
    let (key, key_rows, read_value, value_rows) = match &spec.kv_b {
        LatentSpec::Fused(projection) => (
            projection,
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
            projection,
            ComponentRowMapping::Blocked {
                offset: nope,
                block_width: value,
                block_stride: spec
                    .geometry
                    .reconstruction_head_dimensions()
                    .expect("admitted fused MLA head geometry")
                    as usize,
            },
        ),
        LatentSpec::Split {
            key,
            value: projection,
        } => (
            key,
            head_rows(0, nope, 1, None),
            projection,
            ComponentRowMapping::Direct { offset: 0 },
        ),
    };
    let reads = vec![
        read(
            query,
            ComponentReadRole::Query,
            head_rows(0, spec.geometry.query_head_dimensions() as usize, 1, None),
            query_input,
        ),
        read(
            key,
            ComponentReadRole::Key,
            key_rows,
            vec![kv_stage.clone()],
        ),
        // The nominal positional key is shared by all heads and bypasses latent normalization.
        read(
            &spec.kv_a,
            ComponentReadRole::Key,
            head_rows(latent, rope, heads, None),
            vec![],
        ),
        read(
            read_value,
            ComponentReadRole::Value,
            value_rows,
            vec![kv_stage],
        ),
    ];
    let count = spec.output.input as usize;
    let write = &spec.output.weight;
    g.descriptor.components.push(ComponentGroup {
        id: format!("{attention}.channels"),
        node_id: attention.into(),
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
        shared_write_weight: write.alias_of.as_ref().unwrap_or(&write.id).as_str().into(),
        write_weight: write.id.as_str().into(),
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
        input_normalization: declared_normalization(input_norm),
        output_gate: None,
        output_normalization: None,
        residual_scale: ComponentScalar::new(1.0),
    });
    let mut channels = axes(count);
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
}

pub(in crate::discovery) fn readout(g: &mut Builder, declaration: &Declaration) {
    let statics = &declaration.statics;
    let embedding = statics.embedding_spec();
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
            weight: statics
                .head_spec()
                .map(|p| p.weight.id.as_str())
                .unwrap_or(embedding.weight.id.as_str())
                .into(),
            bias: None,
            output_transform: ComponentOutputTransform::Identity,
            block_normalizations: vec![],
            other_writes: declaration
                .blocks
                .iter()
                .enumerate()
                .filter(|(_, p)| matches!(p.feed_forward, F::Sparse(_)))
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
