//! Muse's centered pre/post norms, channel gates, and scaled vocabulary readout.
use super::*;
use crate::muse::{BlockNormalization, DecoderConfig, WeightConvention};

mod compressed;
mod residual;

#[derive(Clone, Copy)]
enum ReadNormStage {
    Head,
    Embedding,
}
fn read_norm(
    c: &DecoderConfig,
    gain: Option<String>,
    stage: ReadNormStage,
) -> ComponentNormalization {
    let mut kind = match c.block_normalization {
        BlockNormalization::OutputScaled => ComponentNormalizationKind::Rms,
        BlockNormalization::InputScaledFeedForward => ComponentNormalizationKind::RmsFloat32,
    };
    if matches!(stage, ReadNormStage::Embedding)
        && c.vocabulary_arithmetic() == crate::muse::VocabularyArithmetic::Float32
    {
        kind = ComponentNormalizationKind::RmsFloat32Output;
    }
    ComponentNormalization {
        kind,
        epsilon: ComponentScalar::new(c.rms_norm_eps),
        gain,
        gain_offset: ComponentScalar::new(0.0),
        bias: None,
        groups: 1,
    }
}

fn declared_target_norm(
    spec: &crate::muse::equations::specs::LearnedRmsNormSpec,
) -> ComponentNormalization {
    let mut normalization = super::declared_normalization(spec.normalization());
    normalization.kind = match spec.equation() {
        crate::muse::RmsNormEquation::OutputScaled(_) => ComponentNormalizationKind::Rms,
        crate::muse::RmsNormEquation::OutputScaledFp32(_) => ComponentNormalizationKind::RmsFloat32,
        crate::muse::RmsNormEquation::InputScaled(_) => {
            ComponentNormalizationKind::InputScaledRmsFloat32
        }
        _ => unreachable!(
            "assistant compute-precision equations are outside ordinary target component discovery"
        ),
    };
    normalization
}
fn declared_branch_norm(
    spec: &crate::muse::equations::specs::BranchNormSpec,
) -> ComponentNormalization {
    match spec {
        crate::muse::equations::specs::BranchNormSpec::Learned(spec) => declared_target_norm(spec),
        crate::muse::equations::specs::BranchNormSpec::Weightless { epsilon } => {
            ComponentNormalization {
                kind: ComponentNormalizationKind::RmsFloat32,
                epsilon: ComponentScalar::new(*epsilon),
                gain: None,
                gain_offset: ComponentScalar::new(0.0),
                bias: None,
                groups: 1,
            }
        }
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
    let spec = c
        .attention(layer)
        .expect("validated Muse attention declaration");
    let slots = &spec.parameters;
    let heads = slots.projection.geometry.query_heads() as usize;
    let kv = slots.projection.geometry.key_value_heads() as usize;
    let dim = slots.projection.geometry.key_dimensions() as usize;
    let scalar = c.scalar(layer).expect("validated Muse scalar declarations");
    g.attach_parameters(&attention, &ap);
    g.attach_parameters(&ffn, &fp);
    residual::attach(g, c, layer);
    // Generic composite topology already declares the mutable block boundaries.
    // These now execute inside the block and also supply effective evidence.
    for suffix in ["input", "output"] {
        let mut point = g
            .descriptor
            .observations
            .get(&format!("{path}.{suffix}"))
            .unwrap()
            .clone();
        point.path.push_str(".effective");
        point.position = ObservationPosition::AfterIntervention;
        point.meaning = format!("Effective Muse block {suffix}");
        g.get_mut(&block).observation_paths.push(point.path.clone());
        g.descriptor.observations.points.push(point);
    }
    for (boundary, node, field, post, normalization) in [
        (
            "attention.input",
            &attention,
            "input_layernorm",
            false,
            declared_target_norm(&scalar.input),
        ),
        (
            "attention.output",
            &attention,
            "post_attention_layernorm",
            true,
            declared_branch_norm(&scalar.post_attention),
        ),
        (
            "feed_forward.input",
            &ffn,
            "pre_feedforward_layernorm",
            false,
            declared_target_norm(&scalar.pre_feed_forward),
        ),
        (
            "feed_forward.output",
            &ffn,
            "post_feedforward_layernorm",
            true,
            declared_target_norm(&scalar.post_feed_forward),
        ),
    ] {
        let prefix = format!("{path}.{field}");
        let id = format!("{block}.{field}");
        g.node(
            &id,
            ArchitectureNodeKind::Normalization,
            Some(&block),
            normalization.gain.as_ref().map(|_| prefix.as_str()),
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
                    equation: ComponentTensorTransformEquation::Normalization { normalization },
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
    let read = |projection: &eredu_nn::LinearSpec, role, rows, head_count: Option<usize>| {
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
                normalization: read_norm(
                    c,
                    slots.norms.get(if query { 0 } else { 1 }).and_then(|norm| {
                        use eredu_nn::DeclareLogicalParameters;
                        norm.logical_parameters()
                            .expect("validated Muse head norm")
                            .first()
                            .map(|parameter| parameter.parameter.id.as_str().to_owned())
                    }),
                    ReadNormStage::Head,
                ),
            }
        });
        ComponentRead {
            head_normalization,
            ..declared_component_read(projection, format!("parameters:{ap}"), role, rows)
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
                &slots.projection.query,
                ComponentReadRole::Query,
                head_rows(1),
                Some(heads),
            ),
            read(
                &slots.projection.key,
                ComponentReadRole::Key,
                head_rows(heads / kv),
                Some(kv),
            ),
            read(
                &slots.projection.value,
                ComponentReadRole::Value,
                ComponentRowMapping::GroupedQuery {
                    offset: 0,
                    head_width: dim,
                    queries_per_kv: heads / kv,
                },
                None,
            ),
            read(
                &slots.gate,
                ComponentReadRole::OutputGate,
                ComponentRowMapping::Direct { offset: 0 },
                None,
            ),
        ],
        routed_reads: vec![],
        write_weight: slots.projection.output.weight.id.as_str().to_owned(),
        shared_write_weight: slots.projection.output.weight.id.as_str().to_owned(),
        write_parameter_group: format!("parameters:{ap}"),
        write_input_projection: None,
        write_bias: None,
        activation_equation: ComponentActivation::Attention {
            attended_head_normalization: matches!(
                c.attention_equation,
                crate::muse::AttentionEquation::NormalizedHeads
                    | crate::muse::AttentionEquation::NormalizedHeadsScoreScaled
            )
            .then(|| ComponentAttendedHeadNormalization {
                input: format!("{path}.attention.attended_heads"),
                normalization: read_norm(c, None, ReadNormStage::Head),
            }),
            query_heads: heads,
            key_value_heads: kv,
            head_width: dim,
            output_gate: Some(ComponentNonlinearity::Sigmoid),
        },
        input_normalization: declared_target_norm(&scalar.input),
        output_gate: None,
        output_normalization: Some(declared_branch_norm(&scalar.post_attention)),
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
    if matches!(
        c.attention_equation,
        crate::muse::AttentionEquation::NormalizedHeads
            | crate::muse::AttentionEquation::NormalizedHeadsScoreScaled
    ) {
        g.read_only_observation(
            &attention,
            format!("{path}.attention.attended_heads"),
            "Complete attended heads before per-head normalization and channel gating",
            vec![
                TensorAxis {
                    name: "batch".into(),
                    dimension: SymbolicDimension::Batch,
                },
                TensorAxis {
                    name: "head".into(),
                    dimension: SymbolicDimension::Known(heads),
                },
                TensorAxis {
                    name: "sequence".into(),
                    dimension: SymbolicDimension::Sequence,
                },
                TensorAxis {
                    name: "channel".into(),
                    dimension: SymbolicDimension::Known(dim),
                },
            ],
        );
    }
    let input_norm = declared_target_norm(&scalar.pre_feed_forward);
    if c.num_experts == 0 {
        let projections = c.dense(layer).expect("admitted Muse dense declaration");
        gated_units(
            g,
            &ffn,
            layer,
            &format!("parameters:{fp}"),
            projections.projections(),
            projections.policy,
            &format!("{path}.feed_forward"),
            &format!("{path}.feed_forward.input"),
            input_norm,
            ComponentWritePartition::Complete,
        );
        g.descriptor
            .components
            .last_mut()
            .unwrap()
            .output_normalization = Some(declared_target_norm(&scalar.post_feed_forward));
    } else {
        let compressed = compressed::declare(g, c, layer, &ffn, &path, &fp);
        let count = g.descriptor.routed_components.len();
        super::super::routed_components::gated(
            g,
            super::super::routed_components::Site {
                node: &ffn,
                layer,
                routing: &format!("{path}.routing"),
                input: Some(compressed.as_ref().map_or_else(
                    || format!("{path}.feed_forward.input"),
                    |site| site.input.clone(),
                )),
                normalization: Some(
                    compressed
                        .as_ref()
                        .map_or(input_norm, |site| site.normalization.clone()),
                ),
                residual_scale: None,
            },
            c.routed(layer)
                .map(|declaration| declaration.bank().clone()),
        );
        if let Some(compressed) = compressed {
            if g.descriptor.routed_components.len() == count + 1 {
                let group = g.descriptor.routed_components.last_mut().unwrap();
                group.routes_per_token = Some(
                    c.routed(layer)
                        .expect("validated Muse routed declaration")
                        .selector()
                        .selection()
                        .top_k() as usize,
                );
                group.expert_output_transform = Some(compressed.expert_transform);
                group.write_output = Some(format!("{path}.feed_forward.expert_mixture"));
                group.output = Some(format!("{path}.feed_forward.write"));
            }
        }
    }
}

pub(in crate::discovery) fn readout(
    g: &mut Builder,
    c: &DecoderConfig,
    construction: &crate::decoder::StaticModuleConstruction,
) {
    let embedding = construction.embedding_spec();
    let width = embedding.dimensions as usize;
    let output = construction
        .head_spec()
        .map(|head| head.weight.id.as_str())
        .unwrap_or(embedding.weight.id.as_str());
    let mut final_normalization = super::declared_normalization(construction.normalization_spec());
    final_normalization.kind =
        if c.vocabulary_arithmetic() == crate::muse::VocabularyArithmetic::Backend {
            ComponentNormalizationKind::Rms
        } else {
            ComponentNormalizationKind::RmsFloat32
        };
    let final_prefix = final_normalization
        .gain
        .as_deref()
        .expect("declared Muse final gain")
        .trim_end_matches(".weight")
        .to_owned();
    g.attach_parameters(
        "embedding",
        embedding.weight.id.as_str().trim_end_matches(".weight"),
    );
    g.attach_parameters("output", output.trim_end_matches(".weight"));
    g.node(
        "output.norm",
        ArchitectureNodeKind::Normalization,
        Some("output"),
        Some(&final_prefix),
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
    let mut scores = axes(embedding.vocabulary as usize);
    scores[2].name = "vocabulary".into();
    g.component_observation(
        "output",
        "readout.linear".into(),
        "Actual affine scores before output multiplier and softcap",
        scores,
    );
    g.descriptor.component_readout = Some(ComponentReadout {
        embedding: "readout.embedding".into(),
        embedding_weight: embedding.weight.id.as_str().into(),
        embedding_scale: ComponentScalar::new(1.0),
        embedding_normalization: None,
        token_embedding_normalization: Some(read_norm(c, None, ReadNormStage::Embedding)),
        tied_embeddings: construction.head_spec().is_none(),
        equation: ComponentReadoutEquation {
            residual_cycles: residual::cycles(c),
            block_transforms: vec![],
            stream_residual: None,
            residual: "readout.residual".into(),
            normalized: "readout.normalized".into(),
            projection_input: Some("readout.projection_input".into()),
            linear_scores: "readout.linear".into(),
            logits: MODEL_LOGITS_OBSERVATION_PATH.into(),
            normalization: final_normalization,
            weight: output.into(),
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
