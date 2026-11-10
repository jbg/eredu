//! V4 scalar boundaries retain grouped writes and dynamic residual evidence.
use super::*;
use crate::deepseek::{v4_specs, V4Args};

mod prediction;
pub(in crate::discovery) use prediction::declare as declare_prediction;
mod dspark;
pub(in crate::discovery) use dspark::declare as declare_dspark;

fn norm(c: &V4Args, gain: Option<String>) -> ComponentNormalization {
    ComponentNormalization {
        kind: ComponentNormalizationKind::Rms,
        epsilon: ComponentScalar::new(c.rms_norm_eps),
        gain,
        gain_offset: ComponentScalar::new(0.0),
        bias: None,
        groups: 1,
    }
}

pub(in crate::discovery) fn readout(g: &mut Builder, c: &V4Args) {
    let text = &c.parameters().statics;
    let embedding_spec = text.embedding_spec();
    let output = text.head_spec().expect("V4 declares an untied output head");
    let head_spec =
        v4_specs::hyper_head_spec(c, "hc_head").expect("validated V4 hyper-head declaration");
    let width = head_spec.hidden_size as usize;
    let streams = head_spec.streams as usize;
    readout_observations(g, width, output.output as usize);
    let collapse = "output.stream_collapse";
    g.node(
        collapse,
        ArchitectureNodeKind::Sum,
        None,
        Some("hc_head"),
        Some(width),
    );
    for edge in &mut g.descriptor.edges {
        if edge.to == "output.norm" {
            edge.to = collapse.into();
        }
    }
    g.edge(collapse, "output.norm", ArchitectureEdgeKind::Data);
    let mut input = axes(width);
    input.insert(
        2,
        TensorAxis {
            name: "stream".into(),
            dimension: SymbolicDimension::Known(streams),
        },
    );
    g.component_observation(
        collapse,
        "readout.streams".into(),
        "Residual streams before the learned final collapse",
        input,
    );
    let mut coefficient_axes = axes(streams);
    coefficient_axes[2].name = "stream".into();
    g.read_only_observation(
        collapse,
        "readout.stream_coefficients".into(),
        "Actual input-dependent coefficients consumed by the final stream sum",
        coefficient_axes,
    );
    let cycles = (0..c.num_hidden_layers as usize)
        .flat_map(|layer| {
            let path = format!("layers.{layer}");
            stream_cycles(
                c,
                layer,
                &path,
                &path,
                &format!("decoder.layers.{layer}.attention"),
                &format!("decoder.layers.{layer}.feed_forward"),
            )
        })
        .collect();
    g.descriptor.component_readout = Some(ComponentReadout {
        token_embedding_normalization: None,
        embedding_normalization: None,
        embedding: "readout.embedding".into(),
        embedding_weight: embedding_spec.weight.id.as_str().into(),
        embedding_scale: ComponentScalar::new(1.0),
        tied_embeddings: text.head_spec().is_none(),
        equation: ComponentReadoutEquation {
            residual_cycles: vec![],
            block_transforms: vec![],
            score_writes: vec![],
            stream_residual: Some(ComponentStreamResidual {
                streams,
                base: ComponentStreamBase::Broadcast {
                    input: "readout.embedding.effective".into(),
                },
                cycles,
                head: ComponentStreamHead {
                    input: "readout.streams.effective".into(),
                    coefficients: "readout.stream_coefficients".into(),
                    parameters: head_coefficients("parameters:hc_head", &head_spec),
                },
            }),
            residual: "readout.residual".into(),
            normalized: "readout.normalized".into(),
            projection_input: Some("readout.projection_input".into()),
            linear_scores: "readout.linear".into(),
            logits: MODEL_LOGITS_OBSERVATION_PATH.into(),
            normalization: declared_normalization(text.normalization_spec()),
            weight: output.weight.id.as_str().into(),
            bias: output.bias.as_ref().map(|p| p.id.as_str().into()),
            output_transform: ComponentOutputTransform::Identity,
            block_normalizations: vec![],
            other_writes: vec![],
        },
    });
}

fn coefficients(
    group: &str,
    function: &eredu_nn::ParameterSpec,
    base: &eredu_nn::ParameterSpec,
    scale: &eredu_nn::ParameterSpec,
    norm_epsilon: f32,
    epsilon: f32,
) -> ComponentStreamCoefficients {
    ComponentStreamCoefficients {
        parameter_group: group.into(),
        function: function.id.as_str().into(),
        base: base.id.as_str().into(),
        scale: scale.id.as_str().into(),
        normalization: ComponentNormalization {
            kind: ComponentNormalizationKind::Rms,
            epsilon: ComponentScalar::new(norm_epsilon),
            gain: None,
            gain_offset: ComponentScalar::new(0.0),
            bias: None,
            groups: 1,
        },
        epsilon: ComponentScalar::new(epsilon),
    }
}

fn head_coefficients(group: &str, spec: &eredu_nn::HyperHeadSpec) -> ComponentStreamCoefficients {
    coefficients(
        group,
        &spec.function,
        &spec.base,
        &spec.scale,
        spec.norm_epsilon,
        spec.epsilon,
    )
}

fn stream_axes(head: &eredu_nn::HyperHeadSpec) -> Vec<TensorAxis> {
    let mut shape = axes(head.hidden_size as usize);
    shape.insert(
        2,
        TensorAxis {
            name: "stream".into(),
            dimension: SymbolicDimension::Known(head.streams as usize),
        },
    );
    shape
}

fn stream_cycles(
    c: &V4Args,
    layer: usize,
    path: &str,
    parameters: &str,
    attention_node: &str,
    ffn_node: &str,
) -> [ComponentStreamCycle; 2] {
    let hyper_spec = v4_specs::hyper_block_spec(c, parameters)
        .expect("validated V4 hyper-connection declaration");
    [
        (true, &hyper_spec.attention),
        (false, &hyper_spec.feed_forward),
    ]
    .map(|(attention, spec)| {
        let kind = if attention {
            "attention"
        } else {
            "feed_forward"
        };
        let hyper = format!("{path}.hyper.{kind}");
        ComponentStreamCycle {
            node_id: if attention { attention_node } else { ffn_node }.into(),
            layer_index: layer,
            input: if attention {
                format!("{path}.input.effective")
            } else {
                format!("{path}.hyper.attention.streams.effective")
            },
            collapsed: format!("{hyper}.collapsed"),
            write: if attention {
                format!("{path}.compressed_attention.output.effective")
            } else {
                format!("{path}.feed_forward.contribution.effective")
            },
            output: if attention {
                format!("{path}.hyper.attention.streams")
            } else {
                format!("{path}.output")
            },
            pre: format!("{hyper}.pre"),
            post: format!("{hyper}.post"),
            combination: format!("{hyper}.combination"),
            coefficients: coefficients(
                &format!("parameters:{parameters}"),
                &spec.function,
                &spec.base,
                &spec.scale,
                c.rms_norm_eps,
                spec.epsilon,
            ),
            sinkhorn_iterations: spec.sinkhorn_iterations,
        }
    })
}

pub(in crate::discovery) fn unit(
    g: &mut Builder,
    c: &V4Args,
    layer: usize,
    path: &str,
    parameters: &str,
    attention: &str,
    ffn: &str,
) {
    let prefix = format!("{parameters}.attn");
    let spec =
        v4_specs::attention_spec(c, layer, &prefix).expect("validated V4 attention declaration");
    let [attention_norm, _] = v4_specs::block_normalizations(c, parameters)
        .expect("validated V4 block normalization declarations");
    let width = spec.kv.input as usize;
    let heads = spec.geometry.query_heads() as usize;
    let dim = spec.geometry.key_dimensions() as usize;
    let query_first = spec
        .query
        .first
        .as_ref()
        .expect("V4 query has a bottleneck projection");
    let scope = format!("{path}.compressed_attention");
    let parameter_group = format!("parameters:{prefix}");
    let head_rows = |sharing| ComponentRowMapping::HeadRows {
        offset: 0,
        component_head_width: dim,
        read_head_width: dim,
        component_heads_per_read_head: sharing,
        read_head_stride: None,
    };
    let read =
        |projection: &eredu_nn::LinearSpec, role, sharing, normalization, input_projections| {
            ComponentRead {
                head_normalization: Some(ComponentHeadNormalization {
                    output_scale: ComponentScalar::new(1.0),
                    heads: heads / sharing,
                    head_width: dim,
                    independent_gains: false,
                    normalization,
                }),
                input_projections,
                ..declared_component_read(
                    projection,
                    parameter_group.clone(),
                    role,
                    head_rows(sharing),
                )
            }
        };
    let write = spec.output_b.weight.id.as_str().to_owned();
    let first = spec.output_a.weight.id.as_str().to_owned();
    g.descriptor.components.push(ComponentGroup {
        id: format!("{attention}.channels"),
        node_id: attention.into(),
        layer_index: layer,
        count: heads * dim,
        activation: format!("{scope}.channels"),
        effective_activation: format!("{scope}.channels.effective"),
        write_input: None,
        write_output: Some(format!("{scope}.write")),
        output: None,
        write_partition: ComponentWritePartition::TensorParallelSum,
        input: format!("{scope}.input"),
        reads: vec![
            read(
                &spec.query.second,
                ComponentReadRole::Query,
                1,
                ComponentNormalization {
                    epsilon: ComponentScalar::new(spec.normalization_epsilon),
                    ..norm(c, None)
                },
                vec![declared_input_projection(
                    query_first,
                    parameter_group.clone(),
                    0..query_first.output as usize,
                    Some(declared_normalization(&spec.query.normalization)),
                    format!("{scope}.query.latent.effective"),
                )],
            ),
            read(
                &spec.kv,
                ComponentReadRole::Key,
                heads,
                declared_normalization(&spec.kv_norm),
                vec![],
            ),
            read(
                &spec.kv,
                ComponentReadRole::Value,
                heads,
                declared_normalization(&spec.kv_norm),
                vec![],
            ),
        ],
        routed_reads: vec![],
        shared_write_weight: declared_shared_weight(&spec.output_b.weight).into(),
        write_weight: write,
        write_input_projection: Some(ComponentGroupedWriteProjection {
            shared_weight: declared_shared_weight(&spec.output_a.weight).into(),
            weight: first,
            parameter_group: parameter_group.clone(),
            bias: spec
                .output_a
                .bias
                .as_ref()
                .map(|p| p.id.as_str().to_owned()),
            groups: spec.groups as usize,
            rank: spec.output_rank() as usize,
            input: format!("{scope}.grouped_input"),
            output: format!("{scope}.projection"),
            final_input: format!("{scope}.projection_input"),
        }),
        write_parameter_group: parameter_group,
        write_bias: spec
            .output_b
            .bias
            .as_ref()
            .map(|p| p.id.as_str().to_owned()),
        activation_equation: ComponentActivation::Attention {
            attended_head_normalization: None,
            query_heads: heads,
            key_value_heads: spec.geometry.key_value_heads() as usize,
            head_width: dim,
            output_gate: None,
        },
        input_normalization: declared_normalization(&attention_norm),
        output_gate: None,
        output_normalization: None,
        residual_scale: ComponentScalar::new(1.0),
    });
    for (suffix, extent, axis, meaning) in [
        (
            "input",
            width,
            "hidden",
            "Normalized collapsed attention input",
        ),
        (
            "channels",
            heads * dim,
            "component",
            "Value aggregation after inverse rotary, before grouped output projection",
        ),
        (
            "query.latent",
            query_first.output as usize,
            "hidden",
            "Learned-normalized query latent before query-head projection",
        ),
        (
            "key_value.latent",
            dim,
            "hidden",
            "Shared normalized key/value before rotary and cache insertion",
        ),
        (
            "write",
            width,
            "hidden",
            "Attention write before tensor reduction and hyper expansion",
        ),
        (
            "output",
            width,
            "hidden",
            "Complete attention contribution before hyper expansion",
        ),
    ] {
        let mut shape = axes(extent);
        shape[2].name = axis.into();
        g.component_observation(attention, format!("{scope}.{suffix}"), meaning, shape);
    }
    for suffix in ["input", "contribution"] {
        g.component_observation(
            ffn,
            format!("{path}.feed_forward.{suffix}"),
            "Collapsed-normalized FFN input or complete expert contribution before hyper expansion",
            axes(width),
        );
    }
    hyper_observations(g, c, path, attention, ffn);
    shared(g, c, layer, path, parameters, ffn);
}

fn hyper_observations(g: &mut Builder, c: &V4Args, path: &str, attention: &str, ffn: &str) {
    let streams = c.hc_mult as usize;
    for (kind, node) in [("attention", attention), ("feed_forward", ffn)] {
        let scope = format!("{path}.hyper.{kind}");
        let mut vector = axes(streams);
        vector[2].name = "stream".into();
        for suffix in ["pre", "post"] {
            g.read_only_observation(
                node,
                format!("{scope}.{suffix}"),
                "Actual input-dependent hyper-connection coefficient",
                vector.clone(),
            );
        }
        vector[2].name = "input_stream".into();
        vector.push(TensorAxis {
            name: "output_stream".into(),
            dimension: SymbolicDimension::Known(streams),
        });
        g.read_only_observation(
            node,
            format!("{scope}.combination"),
            "Actual residual mixing matrix, indexed by input then output stream",
            vector,
        );
        g.component_observation(
            node,
            format!("{scope}.collapsed"),
            "Stream collapse before learned sublayer normalization",
            axes(c.hidden_size as usize),
        );
    }
    let mut shape = axes(c.hidden_size as usize);
    shape.insert(
        2,
        TensorAxis {
            name: "stream".into(),
            dimension: SymbolicDimension::Known(streams),
        },
    );
    g.component_observation(
        attention,
        format!("{path}.hyper.attention.streams"),
        "Expanded attention write plus mixed incoming residual streams",
        shape,
    );
}

fn shared(g: &mut Builder, c: &V4Args, layer: usize, path: &str, parameters: &str, ffn: &str) {
    let node = format!("{ffn}.shared");
    let scope = format!("{path}.feed_forward.shared");
    let spec = v4_specs::moe_spec(c, layer, &format!("{parameters}.ffn"), None)
        .expect("validated V4 routed declaration");
    let [_, normalization] = v4_specs::block_normalizations(c, parameters)
        .expect("validated V4 normalization declaration");
    g.component_observation(
        &node,
        format!("{scope}.input"),
        "Actual shared expert boundary before the routed/shared contribution sum",
        axes(spec.routed.shared.gate.input as usize),
    );
    super::gated::gated_units(
        g,
        &node,
        layer,
        &format!("parameters:{parameters}.ffn"),
        spec.routed.shared.projections(),
        spec.routed.shared.policy,
        &scope,
        &format!("{scope}.input"),
        declared_normalization(&normalization),
        ComponentWritePartition::TensorParallelSum,
    );
}
