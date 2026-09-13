//! V4 scalar boundaries retain grouped writes and dynamic residual evidence.
use super::*;
use crate::deepseek::V4Args;

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
    let width = c.hidden_size as usize;
    let streams = c.hc_mult as usize;
    readout_observations(g, width, c.vocab_size as usize);
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
        embedding_weight: "embed.weight".into(),
        embedding_scale: ComponentScalar::new(1.0),
        tied_embeddings: false,
        equation: ComponentReadoutEquation {
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
                    parameters: coefficients(c, "parameters:hc_head", "hc_head"),
                },
            }),
            residual: "readout.residual".into(),
            normalized: "readout.normalized".into(),
            projection_input: Some("readout.projection_input".into()),
            linear_scores: "readout.linear".into(),
            logits: MODEL_LOGITS_OBSERVATION_PATH.into(),
            normalization: norm(c, Some("norm.weight".into())),
            weight: "head.weight".into(),
            bias: None,
            output_transform: ComponentOutputTransform::Identity,
            block_normalizations: vec![],
            other_writes: vec![],
        },
    });
}

fn coefficients(c: &V4Args, group: &str, prefix: &str) -> ComponentStreamCoefficients {
    ComponentStreamCoefficients {
        parameter_group: group.into(),
        function: format!("{prefix}_fn"),
        base: format!("{prefix}_base"),
        scale: format!("{prefix}_scale"),
        normalization: norm(c, None),
        epsilon: ComponentScalar::new(c.hc_eps),
    }
}

fn stream_axes(c: &V4Args) -> Vec<TensorAxis> {
    let mut shape = axes(c.hidden_size as usize);
    shape.insert(
        2,
        TensorAxis {
            name: "stream".into(),
            dimension: SymbolicDimension::Known(c.hc_mult as usize),
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
    [true, false].map(|attention| {
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
                c,
                &format!("parameters:{parameters}"),
                &format!("{parameters}.hc_{}", if attention { "attn" } else { "ffn" }),
            ),
            sinkhorn_iterations: c.hc_sinkhorn_iters as usize,
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
    let width = c.hidden_size as usize;
    let heads = c.num_attention_heads as usize;
    let dim = c.head_dim as usize;
    let prefix = format!("{parameters}.attn");
    let scope = format!("{path}.compressed_attention");
    let parameter_group = format!("parameters:{prefix}");
    let head_rows = |sharing| ComponentRowMapping::HeadRows {
        offset: 0,
        component_head_width: dim,
        read_head_width: dim,
        component_heads_per_read_head: sharing,
        read_head_stride: None,
    };
    let read = |field: &str, role, sharing, normalization, input_projections| {
        let weight = format!("{prefix}.{field}.weight");
        ComponentRead {
            source: None,
            projection_output: None,
            role,
            shared_weight: weight.clone(),
            weight,
            parameter_group: parameter_group.clone(),
            bias: None,
            rows: head_rows(sharing),
            head_normalization: Some(ComponentHeadNormalization {
                output_scale: ComponentScalar::new(1.0),
                heads: heads / sharing,
                head_width: dim,
                independent_gains: false,
                normalization,
            }),
            input_projections,
        }
    };
    let write = format!("{prefix}.wo_b.weight");
    let first = format!("{prefix}.wo_a.weight");
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
                "wq_b",
                ComponentReadRole::Query,
                1,
                norm(c, None),
                vec![ComponentInputProjection {
                    weight: format!("{prefix}.wq_a.weight"),
                    shared_weight: format!("{prefix}.wq_a.weight"),
                    parameter_group: parameter_group.clone(),
                    bias: None,
                    rows: 0..c.q_lora_rank as usize,
                    normalization: Some(norm(c, Some(format!("{prefix}.q_norm.weight")))),
                    output: format!("{scope}.query.latent.effective"),
                }],
            ),
            read(
                "wkv",
                ComponentReadRole::Key,
                heads,
                norm(c, Some(format!("{prefix}.kv_norm.weight"))),
                vec![],
            ),
            read(
                "wkv",
                ComponentReadRole::Value,
                heads,
                norm(c, Some(format!("{prefix}.kv_norm.weight"))),
                vec![],
            ),
        ],
        routed_reads: vec![],
        shared_write_weight: write.clone(),
        write_weight: write,
        write_input_projection: Some(ComponentGroupedWriteProjection {
            shared_weight: first.clone(),
            weight: first,
            parameter_group: parameter_group.clone(),
            bias: None,
            groups: c.o_groups as usize,
            rank: c.o_lora_rank as usize,
            input: format!("{scope}.grouped_input"),
            output: format!("{scope}.projection"),
            final_input: format!("{scope}.projection_input"),
        }),
        write_parameter_group: parameter_group,
        write_bias: None,
        activation_equation: ComponentActivation::Attention {
            query_heads: heads,
            key_value_heads: 1,
            head_width: dim,
            output_gate: None,
        },
        input_normalization: norm(c, Some(format!("{parameters}.attn_norm.weight"))),
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
            c.q_lora_rank as usize,
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
    let prefix = format!("{parameters}.ffn.shared_experts");
    let scope = format!("{path}.feed_forward.shared");
    let parameter_group = format!("parameters:{parameters}.ffn");
    let count = (c.moe_intermediate_size * c.n_shared_experts) as usize;
    let write = format!("{prefix}.w2.weight");
    g.descriptor.components.push(ComponentGroup {
        id: format!("{node}.units"),
        node_id: node.clone(),
        layer_index: layer,
        count,
        activation: format!("{scope}.units"),
        effective_activation: format!("{scope}.units.effective"),
        write_input: Some(format!("{scope}.write_input")),
        write_output: Some(format!("{scope}.write")),
        output: Some(format!("{scope}.output")),
        write_partition: ComponentWritePartition::TensorParallelSum,
        input: format!("{scope}.input"),
        reads: [
            ("w1", ComponentReadRole::Gate),
            ("w3", ComponentReadRole::Value),
        ]
        .into_iter()
        .map(|(field, role)| {
            let weight = format!("{prefix}.{field}.weight");
            ComponentRead {
                source: None,
                projection_output: None,
                role,
                shared_weight: weight.clone(),
                weight,
                parameter_group: parameter_group.clone(),
                bias: None,
                rows: ComponentRowMapping::Direct { offset: 0 },
                head_normalization: None,
                input_projections: vec![],
            }
        })
        .collect(),
        routed_reads: vec![],
        shared_write_weight: write.clone(),
        write_weight: write,
        write_input_projection: None,
        write_parameter_group: parameter_group,
        write_bias: None,
        // V4 clamps routed experts; its shared experts use ordinary SwiGLU.
        activation_equation: ComponentActivation::Gated {
            activation: ComponentNonlinearity::Silu {
                multiplier: ComponentScalar::new(1.0),
            },
            gate_upper_bound: None,
            value_absolute_bound: None,
            value_offset: ComponentScalar::new(0.0),
        },
        input_normalization: norm(c, Some(format!("{parameters}.ffn_norm.weight"))),
        output_gate: None,
        output_normalization: None,
        residual_scale: ComponentScalar::new(1.0),
    });
    for suffix in ["input", "units", "write", "output"] {
        let mut shape = axes(if suffix == "units" {
            count
        } else {
            c.hidden_size as usize
        });
        if suffix == "units" {
            shape[2].name = "component".into();
        }
        g.component_observation(
            &node,
            format!("{scope}.{suffix}"),
            "Actual shared expert boundary before the routed/shared contribution sum",
            shape,
        );
    }
}
