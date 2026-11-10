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
use crate::decoder::{topology::AttentionInputParameters, Config, OutputGateActivation};
use eredu_core::component::*;

/// Prefix of an explicitly declared ordinary weight slot.
/// Execution-node identities and hook paths remain family-authored.
pub(super) fn declared_weight_prefix(weight: &eredu_nn::ParameterSpec) -> &str {
    weight
        .id
        .as_str()
        .strip_suffix(".weight")
        .expect("declared ordinary weight identity")
}

fn declared_shared_weight(weight: &eredu_nn::ParameterSpec) -> &str {
    weight.alias_of.as_ref().unwrap_or(&weight.id).as_str()
}

pub(super) fn declared_normalization_prefix(
    spec: &eredu_nn::NormalizationConstructionSpec,
) -> Option<&str> {
    use eredu_nn::{DeclareLogicalParameters, LogicalParameterRole};
    spec.logical_parameters()
        .expect("admitted normalization declaration")
        .into_iter()
        .find(|declaration| declaration.role == LogicalParameterRole::Normalization)
        .map(|declaration| declared_weight_prefix(declaration.parameter))
}

/// Projects affine parameter metadata into caller-authored component roles and row mappings.
fn declared_component_read(
    projection: &eredu_nn::LinearSpec,
    parameter_group: String,
    role: ComponentReadRole,
    rows: ComponentRowMapping,
) -> ComponentRead {
    ComponentRead {
        source: None,
        projection_output: None,
        role,
        weight: projection.weight.id.to_string(),
        shared_weight: declared_shared_weight(&projection.weight).into(),
        parameter_group,
        bias: projection.bias.as_ref().map(|bias| bias.id.to_string()),
        rows,
        head_normalization: None,
        input_projections: Vec::new(),
    }
}

/// Projects a declared affine stage while retaining its explicit slice and execution boundary.
fn declared_input_projection(
    projection: &eredu_nn::LinearSpec,
    parameter_group: String,
    rows: std::ops::Range<usize>,
    normalization: Option<ComponentNormalization>,
    output: String,
) -> ComponentInputProjection {
    ComponentInputProjection {
        weight: projection.weight.id.to_string(),
        shared_weight: declared_shared_weight(&projection.weight).into(),
        parameter_group,
        bias: projection.bias.as_ref().map(|bias| bias.id.to_string()),
        rows,
        normalization,
        output,
    }
}

fn declared_gated_product(policy: eredu_nn::GatedProductPolicy) -> Option<ComponentActivation> {
    Some(ComponentActivation::Gated {
        activation: match policy.activation() {
            eredu_nn::GatedProductActivation::Silu => ComponentNonlinearity::Silu {
                multiplier: ComponentScalar::new(policy.sigmoid_multiplier()),
            },
            eredu_nn::GatedProductActivation::Gelu => ComponentNonlinearity::Gelu,
            eredu_nn::GatedProductActivation::GeluApproximate => {
                ComponentNonlinearity::GeluApproximate
            }
            _ => return None,
        },
        gate_upper_bound: policy.gate_upper_bound().map(ComponentScalar::new),
        value_absolute_bound: policy.up_absolute_bound().map(ComponentScalar::new),
        value_offset: ComponentScalar::new(policy.up_offset()),
    })
}

fn declared_transform_parameter(
    parameter: &eredu_nn::ParameterSpec,
    parameter_group: String,
) -> ComponentTransformParameter {
    ComponentTransformParameter {
        parameter: parameter.id.to_string(),
        shared_parameter: declared_shared_weight(parameter).into(),
        parameter_group,
    }
}

pub(super) fn declared_normalization(
    spec: &eredu_nn::NormalizationConstructionSpec,
) -> ComponentNormalization {
    let (gain, offset) = match &spec.scale {
        eredu_nn::NormalizationScale::Learned(weight) => (Some(weight.id.as_str().to_owned()), 0.0),
        eredu_nn::NormalizationScale::LearnedOffset { weight, offset } => {
            (Some(weight.id.as_str().to_owned()), *offset)
        }
        eredu_nn::NormalizationScale::Unit => (None, 0.0),
    };
    ComponentNormalization {
        kind: ComponentNormalizationKind::Rms,
        epsilon: ComponentScalar::new(spec.epsilon),
        gain,
        gain_offset: ComponentScalar::new(offset),
        bias: None,
        groups: spec.groups.unwrap_or(1) as usize,
    }
}

pub(super) fn readout<C: Config>(
    g: &mut Builder,
    c: &C,
    declaration: &crate::decoder::DecoderDeclaration,
) {
    let root = c.parameter_root();
    let declared = declaration.static_construction();
    let embedding = declared.embedding_spec();
    let width = embedding.dimensions as usize;
    let terminals = (0..c.num_hidden_layers() as usize)
        .filter_map(|layer_index| {
            declaration
                .block_parameters(layer_index)
                .terminal
                .as_ref()
                .map(|spec| ComponentResidualNormalization {
                    layer_index,
                    input: format!("{root}.layers.{layer_index}.feed_forward.residual"),
                    output: format!("{root}.layers.{layer_index}.output"),
                    normalization: declared_normalization(spec),
                })
        })
        .collect();
    readout_observations(g, width, embedding.vocabulary as usize);
    g.descriptor.component_readout = Some(ComponentReadout {
        token_embedding_normalization: None,
        embedding_normalization: None,
        embedding: "readout.embedding".into(),
        embedding_weight: embedding.weight.id.as_str().to_owned(),
        embedding_scale: ComponentScalar::new(c.embedding_scale()),
        tied_embeddings: declared.head_spec().is_none(),
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
            normalization: declared_normalization(declared.normalization_spec()),
            weight: declared
                .head_spec()
                .map_or(&embedding.weight, |head| &head.weight)
                .id
                .as_str()
                .to_owned(),
            bias: None,
            output_transform: c.output_softcap().map_or(
                ComponentOutputTransform::Identity,
                |cap| ComponentOutputTransform::Softcap {
                    cap: ComponentScalar::new(cap),
                },
            ),
            block_normalizations: terminals,
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
    scope: &crate::decoder::topology::BlockParameters,
) {
    let fields = c.block_parameter_fields();
    let attention_prefix = format!("{path}.{}", fields.attention);
    let ffn_prefix = format!("{path}.{}", fields.feed_forward);
    let attention_parameters = &scope.attention;
    let feed_forward_projections = dense_ffn.then_some(scope.feed_forward.as_ref()).flatten();
    let normalizations = &scope.normalizations;
    let head_norms = &scope.query_key;
    let geometry = attention_parameters.geometry;
    let heads = geometry.query_heads() as usize;
    let kv = geometry.key_value_heads() as usize;
    let dim = geometry.key_dimensions() as usize;
    let attention_write = &attention_parameters.output;
    let read = |spec: &eredu_nn::LinearSpec, prefix: &str, role, rows| {
        let head_normalization = match role {
            ComponentReadRole::Query => head_norms[0].as_ref().map(|spec| (heads, spec)),
            ComponentReadRole::Key => head_norms[1].as_ref().map(|spec| (kv, spec)),
            _ => None,
        }
        .map(|(heads, spec)| ComponentHeadNormalization {
            output_scale: ComponentScalar::new(1.0),
            heads,
            head_width: dim,
            independent_gains: c.query_key_norm_per_head_weights(),
            normalization: declared_normalization(spec),
        });
        ComponentRead {
            head_normalization,
            ..declared_component_read(spec, format!("parameters:{prefix}"), role, rows)
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
    let mut reads = match &attention_parameters.input {
        AttentionInputParameters::Split { query, key, value } => {
            let mut reads = vec![
                read(
                    query,
                    &attention_prefix,
                    ComponentReadRole::Query,
                    head_rows(0, 1),
                ),
                read(
                    key,
                    &attention_prefix,
                    ComponentReadRole::Key,
                    head_rows(0, heads / kv),
                ),
            ];
            if let Some(value) = value {
                reads.push(read(
                    value,
                    &attention_prefix,
                    ComponentReadRole::Value,
                    grouped(0),
                ));
            }
            reads
        }
        AttentionInputParameters::Fused { projection, .. } => {
            let segments = attention_parameters
                .fused_segments()
                .expect("fused attention layout");
            vec![
                read(
                    projection,
                    &attention_prefix,
                    ComponentReadRole::Query,
                    head_rows(segments[0].start, 1),
                ),
                read(
                    projection,
                    &attention_prefix,
                    ComponentReadRole::Key,
                    head_rows(segments[1].start, heads / kv),
                ),
                read(
                    projection,
                    &attention_prefix,
                    ComponentReadRole::Value,
                    grouped(segments[2].start),
                ),
            ]
        }
    };
    if let Some((gate, _)) = &attention_parameters.gate {
        reads.push(read(
            gate,
            &attention_prefix,
            ComponentReadRole::OutputGate,
            direct(0),
        ));
    }
    let mut group = ComponentGroup {
        routed_reads: Vec::new(),
        id: format!("{attention}.channels"),
        node_id: attention.into(),
        layer_index: layer,
        count: attention_write.input as usize,
        activation: format!("{path}.attention.channels"),
        effective_activation: format!("{path}.attention.channels.effective"),
        write_input: Some(format!("{path}.attention.write_input")),
        write_output: Some(format!("{path}.attention.write")),
        output: Some(format!("{path}.attention.output")),
        write_partition: ComponentWritePartition::Complete,
        input: format!("{path}.attention.input"),
        reads,
        write_weight: attention_write.weight.id.to_string(),
        write_input_projection: None,
        write_parameter_group: format!("parameters:{attention_prefix}"),
        shared_write_weight: declared_shared_weight(&attention_write.weight).into(),
        write_bias: attention_write
            .bias
            .as_ref()
            .map(|bias| bias.id.to_string()),
        activation_equation: ComponentActivation::Attention {
            attended_head_normalization: None,
            query_heads: heads,
            key_value_heads: kv,
            head_width: dim,
            output_gate: attention_parameters.gate.as_ref().map(
                |(_, activation)| match *activation {
                    OutputGateActivation::Softplus(beta) => ComponentNonlinearity::Softplus {
                        beta: ComponentScalar::new(beta),
                    },
                    OutputGateActivation::Sigmoid => ComponentNonlinearity::Sigmoid,
                    OutputGateActivation::Silu => ComponentNonlinearity::Silu {
                        multiplier: ComponentScalar::new(1.0),
                    },
                },
            ),
        },
        input_normalization: declared_normalization(&normalizations.input),
        output_gate: None,
        output_normalization: normalizations
            .attention_output
            .as_ref()
            .map(declared_normalization),
        residual_scale: ComponentScalar::new(1.0),
    };
    g.descriptor.components.push(group.clone());
    if !dense_ffn {
        return;
    }
    let feed_forward_projections = feed_forward_projections
        .as_ref()
        .expect("declared dense feed-forward projections");
    let feed_forward_write = feed_forward_projections.down();
    let width = feed_forward_write.input as usize;
    group.id = format!("{ffn}.units");
    group.node_id = ffn.into();
    group.count = width;
    group.activation = format!("{path}.feed_forward.units");
    group.effective_activation = format!("{path}.feed_forward.units.effective");
    group.write_input = Some(format!("{path}.feed_forward.write_input"));
    group.write_output = Some(format!("{path}.feed_forward.write"));
    group.output = Some(format!("{path}.feed_forward.output"));
    group.input = format!("{path}.feed_forward.input");
    group.reads = match feed_forward_projections {
        eredu_nn::GatedProductSpec::Split(spec) => vec![
            read(&spec.gate, &ffn_prefix, ComponentReadRole::Gate, direct(0)),
            read(&spec.up, &ffn_prefix, ComponentReadRole::Value, direct(0)),
        ],
        eredu_nn::GatedProductSpec::Fused(spec) => vec![
            read(
                &spec.gate_up,
                &ffn_prefix,
                ComponentReadRole::Gate,
                direct(0),
            ),
            read(
                &spec.gate_up,
                &ffn_prefix,
                ComponentReadRole::Value,
                direct(width),
            ),
        ],
    };
    group.write_weight = feed_forward_write.weight.id.to_string();
    group.shared_write_weight = declared_shared_weight(&feed_forward_write.weight).into();
    group.write_parameter_group = format!("parameters:{ffn_prefix}");
    group.write_bias = feed_forward_write
        .bias
        .as_ref()
        .map(|bias| bias.id.to_string());
    let policy = feed_forward_projections.policy();
    let Some(activation) = declared_gated_product(policy) else {
        return;
    };
    group.activation_equation = activation;
    group.input_normalization = declared_normalization(&normalizations.post_attention);
    group.output_normalization = normalizations
        .feed_forward_output
        .as_ref()
        .map(declared_normalization);
    g.descriptor.components.push(group);
}

/// Nemotron physical units each contain a single pre-normalized residual operator.
pub(super) fn nemotron_unit(
    g: &mut Builder,
    scope: &crate::nemotron_h::declarations::TargetBlockParameters,
    layer: usize,
    path: &str,
    node: &str,
) {
    use crate::nemotron_h::declarations::TargetOperatorParameters as P;
    let (projections, heads, kv, dim, activation, attention) = match &scope.operator {
        P::Attention(spec) => (
            spec.projection
                .projections()
                .into_iter()
                .cloned()
                .collect::<Vec<_>>(),
            spec.projection.geometry.query_heads() as usize,
            spec.projection.geometry.key_value_heads() as usize,
            spec.projection.geometry.key_dimensions() as usize,
            None,
            true,
        ),
        P::Dense(spec) => (
            spec.network
                .projections()
                .into_iter()
                .cloned()
                .collect::<Vec<_>>(),
            0,
            1,
            0,
            Some(spec.network.activation),
            false,
        ),
        _ => unreachable!("component units are declared only for attention and dense scopes"),
    };
    let prefix = projections[0]
        .weight
        .id
        .as_str()
        .rsplit_once('.')
        .expect("affine weight identity")
        .0
        .rsplit_once('.')
        .expect("projection owner")
        .0;
    let width = projections[0].input as usize;
    let boundary = if attention {
        "attention"
    } else {
        "feed_forward"
    };
    let scalar = if attention { "channels" } else { "units" };
    let count = projections.last().expect("output projection").input as usize;
    let read = |projection: &eredu_nn::LinearSpec, role, rows| {
        declared_component_read(projection, format!("parameters:{prefix}"), role, rows)
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
            read(&projections[0], ComponentReadRole::Query, head_rows(1)),
            read(
                &projections[1],
                ComponentReadRole::Key,
                head_rows(heads / kv),
            ),
            read(&projections[2], ComponentReadRole::Value, grouped),
        ]
    } else {
        vec![read(&projections[0], ComponentReadRole::Input, direct)]
    };
    let write = projections.last().expect("output projection");
    let write_weight = write.weight.id.as_str().to_owned();
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
        shared_write_weight: declared_shared_weight(&write.weight).into(),
        write_weight,
        write_input_projection: None,
        write_parameter_group: format!("parameters:{prefix}"),
        write_bias: write.bias.as_ref().map(|p| p.id.as_str().into()),
        activation_equation: if attention {
            ComponentActivation::Attention {
                attended_head_normalization: None,
                query_heads: heads,
                key_value_heads: kv,
                head_width: dim,
                output_gate: None,
            }
        } else {
            ComponentActivation::Unary {
                activation: crate::decoder::unary::component(
                    activation.expect("declared dense activation"),
                ),
            }
        },
        input_normalization: declared_normalization(&scope.norm),
        output_gate: None,
        output_normalization: None,
        residual_scale: ComponentScalar::new(1.0),
    });
    for (suffix, meaning, extent) in [
        ("input", "Normalized operator input", width),
        (
            scalar,
            "Activated units or aggregated channels consumed by the output projection",
            count,
        ),
        ("write", "Affine operator write", width),
        (
            "output",
            "Operator contribution before residual addition",
            width,
        ),
        ("residual", "Residual after operator addition", width),
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

pub(super) fn nemotron_readout(g: &mut Builder, declaration: &crate::nemotron_h::Declaration) {
    let c = declaration.args();
    let statics = &declaration.statics;
    let embedding = statics.embedding_spec();
    let output = statics.head_spec();
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
            weight: output
                .map(|p| p.weight.id.as_str())
                .unwrap_or(embedding.weight.id.as_str())
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
    declaration: &crate::lfm2::Declaration,
    layer: usize,
    path: &str,
    mixer_node: &str,
    ffn_node: &str,
) {
    use crate::lfm2::declarations::{FeedForwardParameters as F, MixerParameters as M};
    let scope = &declaration.blocks[layer];
    let width = declaration.statics.embedding_spec().dimensions as usize;
    let read = |projection: &eredu_nn::LinearSpec, prefix: &str, role, rows, head_normalization| {
        ComponentRead {
            head_normalization,
            ..declared_component_read(projection, format!("parameters:{prefix}"), role, rows)
        }
    };
    let attention = matches!(scope.mixer, M::Attention(..));
    if let M::Attention(spec, _) = &scope.mixer {
        let prefix = format!("{path}.self_attn");
        let [query, key, value, write] = spec.projection.projections();
        let normalizations = [&spec.query_norm, &spec.key_norm];
        let dim = spec.projection.geometry.key_dimensions() as usize;
        let heads = spec.projection.geometry.query_heads() as usize;
        let kv = spec.projection.geometry.key_value_heads() as usize;
        let head_rows = |sharing| ComponentRowMapping::HeadRows {
            offset: 0,
            component_head_width: dim,
            read_head_width: dim,
            component_heads_per_read_head: sharing,
            read_head_stride: None,
        };
        let head_norm = |heads, index: usize| {
            Some(ComponentHeadNormalization {
                output_scale: ComponentScalar::new(1.0),
                heads,
                head_width: dim,
                independent_gains: false,
                normalization: declared_normalization(normalizations[index]),
            })
        };
        let write_weight = write.weight.id.as_str().to_owned();
        g.descriptor.components.push(ComponentGroup {
            routed_reads: Vec::new(),
            id: format!("{mixer_node}.channels"),
            node_id: mixer_node.into(),
            layer_index: layer,
            count: write.input as usize,
            activation: format!("{path}.attention.channels"),
            effective_activation: format!("{path}.attention.channels.effective"),
            write_input: Some(format!("{path}.attention.write_input")),
            write_output: Some(format!("{path}.attention.write")),
            output: Some(format!("{path}.attention.output")),
            write_partition: ComponentWritePartition::Complete,
            input: format!("{path}.attention.input"),
            reads: vec![
                read(
                    query,
                    &prefix,
                    ComponentReadRole::Query,
                    head_rows(1),
                    head_norm(heads, 0),
                ),
                read(
                    key,
                    &prefix,
                    ComponentReadRole::Key,
                    head_rows(heads / kv),
                    head_norm(kv, 1),
                ),
                read(
                    value,
                    &prefix,
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
                attended_head_normalization: None,
                query_heads: heads,
                key_value_heads: kv,
                head_width: dim,
                output_gate: None,
            },
            input_normalization: declared_normalization(&scope.operator_norm),
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
    if let F::Dense(spec) = &scope.feed_forward {
        let prefix = format!("{path}.feed_forward");
        let gate = &spec.gate;
        let up = &spec.up;
        let write = &spec.down;
        let write_weight = write.weight.id.as_str().to_owned();
        let count = write.input as usize;
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
                    gate,
                    &prefix,
                    ComponentReadRole::Gate,
                    ComponentRowMapping::Direct { offset: 0 },
                    None,
                ),
                read(
                    up,
                    &prefix,
                    ComponentReadRole::Value,
                    ComponentRowMapping::Direct { offset: 0 },
                    None,
                ),
            ],
            write_weight: write_weight.clone(),
            write_input_projection: None,
            shared_write_weight: declared_shared_weight(&write.weight).into(),
            write_parameter_group: format!("parameters:{prefix}"),
            write_bias: write.bias.as_ref().map(|bias| bias.id.to_string()),
            activation_equation: declared_gated_product(spec.policy)
                .expect("admitted LFM2 gated product policy"),
            input_normalization: declared_normalization(&scope.feed_forward_norm),
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
                && matches!(scope.feed_forward, F::Routed(_))
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

pub(super) fn lfm2_readout(g: &mut Builder, declaration: &crate::lfm2::Declaration) {
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
            other_writes: vec![],
        },
    });
}

pub(super) fn lfm2_other_writes(g: &mut Builder, declaration: &crate::lfm2::Declaration) {
    use crate::lfm2::declarations::{FeedForwardParameters as F, MixerParameters as M};
    let readout = g
        .descriptor
        .component_readout
        .as_mut()
        .expect("LFM2 readout");
    for (layer, scope) in declaration.blocks.iter().enumerate() {
        for (present, boundary) in [
            (matches!(scope.mixer, M::Convolution(_)), "mixer"),
            (matches!(scope.feed_forward, F::Routed(_)), "feed_forward"),
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
