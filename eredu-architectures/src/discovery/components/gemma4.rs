//! Gemma4 component inputs, shared KV publishers and whole-residual scaling.
use super::*;
use crate::gemma4::{FeedForwardPolicy, ModelArgs};

fn norm(c: &ModelArgs, gain: Option<String>) -> ComponentNormalization {
    ComponentNormalization {
        kind: ComponentNormalizationKind::Rms,
        epsilon: ComponentScalar::new(c.rms_norm_eps),
        gain,
        gain_offset: ComponentScalar::new(0.0),
        bias: None,
        groups: 1,
    }
}

fn declared_norm(spec: &eredu_nn::NormalizationConstructionSpec) -> ComponentNormalization {
    let (gain, offset) = match &spec.scale {
        eredu_nn::NormalizationScale::Unit => (None, 0.0),
        eredu_nn::NormalizationScale::Learned(weight) => (Some(weight.id.as_str().to_owned()), 0.0),
        eredu_nn::NormalizationScale::LearnedOffset { weight, offset } => {
            (Some(weight.id.as_str().to_owned()), *offset)
        }
    };
    ComponentNormalization {
        kind: ComponentNormalizationKind::Rms,
        epsilon: ComponentScalar::new(spec.epsilon),
        gain,
        gain_offset: ComponentScalar::new(offset),
        bias: None,
        groups: 1,
    }
}

fn post_norm(
    g: &mut Builder,
    normalization: (usize, ComponentNormalization),
    node: &str,
    id: String,
    input: String,
    output: String,
) {
    let (width, normalization) = normalization;
    let field = normalization
        .gain
        .as_deref()
        .and_then(|gain| gain.strip_suffix(".weight"))
        .expect("declared learned Gemma normalization")
        .to_owned();
    g.node(
        &id,
        ArchitectureNodeKind::Normalization,
        Some(node),
        Some(&field),
        Some(width),
    );
    g.descriptor
        .component_transforms
        .push(ComponentTensorTransform {
            id,
            node_id: node.into(),
            input,
            effective_output: Some(format!("{output}.effective")),
            output,
            equation: ComponentTensorTransformEquation::Normalization { normalization },
        });
}

pub(in crate::discovery) fn modality_projector(
    g: &mut Builder,
    node: &str,
    spec: &crate::gemma4::media::ProjectionSpec,
) {
    let weight = spec.projection.weight.id.as_str();
    g.attach_parameters(
        node,
        weight
            .strip_suffix(".weight")
            .expect("declared modality projection weight"),
    );
}

pub(in crate::discovery) fn unit(
    g: &mut Builder,
    c: &ModelArgs,
    layer: usize,
    block: &crate::gemma4::text::specs::BlockSpec,
    dense: &crate::gemma4::text::specs::DenseSpec,
    attention_scopes: &[crate::gemma4::text::specs::AttentionSpec],
    routed_scope: Option<&crate::gemma4::text::routed::Declaration>,
) {
    unit_at(
        g,
        c,
        layer,
        "model.language_model",
        block,
        dense,
        attention_scopes,
        routed_scope,
    );
}

pub(in crate::discovery) fn unit_at(
    g: &mut Builder,
    c: &ModelArgs,
    layer: usize,
    root: &str,
    block_spec: &crate::gemma4::text::specs::BlockSpec,
    dense_spec: &crate::gemma4::text::specs::DenseSpec,
    attention_scopes: &[crate::gemma4::text::specs::AttentionSpec],
    routed_scope: Option<&crate::gemma4::text::routed::Declaration>,
) {
    let policy = c.layer_policy(layer).expect("admitted Gemma layer");
    let attention_spec = attention_scopes
        .get(layer)
        .expect("admitted Gemma attention declaration");
    let sparse = policy.feed_forward == FeedForwardPolicy::DenseWithSparseMoe;
    let sparse_declaration = routed_scope;
    let block = format!("decoder.{layer}");
    let path = format!("{root}.layers.{layer}");
    let attention = format!("{block}.attention");
    let ffn = format!("{block}.feed_forward");
    let dense = if sparse {
        format!("{block}.dense_feed_forward")
    } else {
        ffn.clone()
    };
    let routed = format!("{ffn}.routed");
    let ap = format!("{path}.self_attn");
    let fp = format!("{path}.mlp");
    let width = c.hidden_size as usize;
    let heads = attention_spec.geometry.query_heads() as usize;
    let kv = attention_spec.geometry.key_value_heads() as usize;
    let dim = attention_spec.geometry.key_dimensions() as usize;
    g.attach_parameters(&block, &path);
    g.attach_parameters(&attention, &ap);
    g.attach_parameters(&dense, &fp);
    if sparse {
        // Both branches belong to the common post-normalized FFN write.
        g.get_mut(&dense).parent = Some(ffn.clone());
        let experts = format!("{path}.experts.switch_glu");
        g.attach_parameters(&ffn, &experts);
        g.node(
            &routed,
            ArchitectureNodeKind::RoutedExperts,
            Some(&ffn),
            Some(&experts),
            Some(width),
        );
        g.edge(&dense, &ffn, ArchitectureEdgeKind::Data);
        g.edge(&routed, &ffn, ArchitectureEdgeKind::Data);
    }
    for (node, boundary) in [
        (&attention, "attention.input"),
        (&attention, "attention.write"),
        (&attention, "attention.output"),
        (&attention, "attention.residual"),
        (&dense, "dense_feed_forward.input"),
        (&ffn, "feed_forward.write"),
        (&ffn, "feed_forward.output"),
        (&ffn, "feed_forward.residual"),
        (&block, "residual.before_scale"),
        (&block, "residual.scaled"),
    ] {
        g.component_observation(
            node,
            format!("{path}.{boundary}"),
            "Gemma4 consumed sublayer or residual boundary",
            axes(width),
        );
    }
    if sparse {
        for (node, boundary) in [
            (&dense, "dense_feed_forward.write"),
            (&dense, "dense_feed_forward.output"),
            (&routed, "routed_feed_forward.input"),
            (&routed, "routed_feed_forward.write"),
            (&routed, "routed_feed_forward.output"),
            (&ffn, "routing.input"),
        ] {
            g.component_observation(
                node,
                format!("{path}.{boundary}"),
                "Gemma4 dense/routed branch boundary",
                axes(width),
            );
        }
    }
    post_norm(
        g,
        (
            (&block_spec.post_attention).dimensions as usize,
            declared_norm(&block_spec.post_attention),
        ),
        &attention,
        format!("{attention}.post_norm"),
        format!("{path}.attention.write.effective"),
        format!("{path}.attention.output"),
    );
    post_norm(
        g,
        (
            (&block_spec.post_feed_forward).dimensions as usize,
            declared_norm(&block_spec.post_feed_forward),
        ),
        &ffn,
        format!("{ffn}.post_norm"),
        format!("{path}.feed_forward.write.effective"),
        format!("{path}.feed_forward.output"),
    );
    if sparse {
        post_norm(
            g,
            (
                sparse_declaration.as_ref().unwrap().norms[0].dimensions as usize,
                declared_norm(&sparse_declaration.as_ref().unwrap().norms[0]),
            ),
            &dense,
            format!("{dense}.post_norm"),
            format!("{path}.dense_feed_forward.write.effective"),
            format!("{path}.dense_feed_forward.output"),
        );
        post_norm(
            g,
            (
                sparse_declaration.as_ref().unwrap().norms[2].dimensions as usize,
                declared_norm(&sparse_declaration.as_ref().unwrap().norms[2]),
            ),
            &routed,
            format!("{routed}.post_norm"),
            format!("{path}.routed_feed_forward.write.effective"),
            format!("{path}.routed_feed_forward.output"),
        );
    }
    let publisher = c
        .attention_state_owner(layer)
        .expect("admitted Gemma state publisher");
    let source_spec = attention_scopes
        .get(publisher)
        .expect("admitted Gemma source declaration");
    let source_ap = format!("{root}.layers.{publisher}.self_attn");
    let head_rows = |sharing| ComponentRowMapping::HeadRows {
        offset: 0,
        component_head_width: dim,
        read_head_width: dim,
        component_heads_per_read_head: sharing,
        read_head_stride: None,
    };
    let read = |projection: &eredu_nn::LinearSpec,
                gain: Option<&eredu_nn::NormalizationConstructionSpec>,
                role,
                source: bool| {
        let prefix = if source { &source_ap } else { &ap };
        let query = role == ComponentReadRole::Query;
        let value = role == ComponentReadRole::Value;
        let mut normalization = gain.map_or_else(|| norm(c, None), declared_norm);
        if value {
            normalization.epsilon = ComponentScalar::new(1e-6);
        }
        ComponentRead {
            source: (source && publisher != layer).then(|| {
                ComponentReadSource::PublishedAttentionState {
                    component_group: format!("decoder.{publisher}.attention.channels"),
                }
            }),
            head_normalization: Some(ComponentHeadNormalization {
                output_scale: ComponentScalar::new(1.0),
                heads: if query { heads } else { kv },
                head_width: dim,
                independent_gains: false,
                normalization,
            }),
            ..declared_component_read(
                projection,
                format!("parameters:{prefix}"),
                role,
                head_rows(if query { 1 } else { heads / kv }),
            )
        }
    };
    let source_key = source_spec
        .key
        .as_ref()
        .expect("state publisher owns a key projection");
    let source_value = source_spec.value.as_ref().unwrap_or(source_key);
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
                &attention_spec.query,
                Some(&attention_spec.query_norm),
                ComponentReadRole::Query,
                false,
            ),
            read(
                source_key,
                source_spec.key_norm.as_ref(),
                ComponentReadRole::Key,
                true,
            ),
            read(source_value, None, ComponentReadRole::Value, true),
        ],
        routed_reads: vec![],
        write_weight: attention_spec.output.weight.id.as_str().to_owned(),
        write_input_projection: None,
        write_parameter_group: format!("parameters:{ap}"),
        shared_write_weight: attention_spec.output.weight.id.as_str().to_owned(),
        write_bias: attention_spec
            .output
            .bias
            .as_ref()
            .map(|b| b.id.as_str().to_owned()),
        activation_equation: ComponentActivation::Attention {
            attended_head_normalization: None,
            query_heads: heads,
            key_value_heads: kv,
            head_width: dim,
            output_gate: None,
        },
        input_normalization: declared_norm(&block_spec.input),
        output_normalization: Some(declared_norm(&block_spec.post_attention)),
        output_gate: None,
        residual_scale: ComponentScalar::new(1.0),
    });
    let mut channels = axes(heads * dim);
    channels[2].name = "component".into();
    g.component_observation(
        &attention,
        format!("{path}.attention.channels"),
        "Aggregated normalized values before output projection",
        channels,
    );
    let count = dense_spec.product.gate.output as usize;
    let branch = if sparse {
        "dense_feed_forward"
    } else {
        "feed_forward"
    };
    g.descriptor.components.push(ComponentGroup {
        id: format!("{dense}.units"),
        node_id: dense.clone(),
        layer_index: layer,
        count,
        activation: format!("{path}.dense_feed_forward.units"),
        effective_activation: format!("{path}.dense_feed_forward.units.effective"),
        write_input: Some(format!("{path}.dense_feed_forward.write_input")),
        write_output: Some(format!("{path}.{branch}.write")),
        output: Some(format!("{path}.{branch}.output")),
        write_partition: ComponentWritePartition::Complete,
        input: format!("{path}.dense_feed_forward.input"),
        reads: [
            (&dense_spec.product.gate, ComponentReadRole::Gate),
            (&dense_spec.product.up, ComponentReadRole::Value),
        ]
        .into_iter()
        .map(|(projection, role)| {
            declared_component_read(
                projection,
                format!("parameters:{fp}"),
                role,
                ComponentRowMapping::Direct { offset: 0 },
            )
        })
        .collect(),
        routed_reads: vec![],
        write_weight: dense_spec.product.down.weight.id.as_str().to_owned(),
        write_input_projection: None,
        write_parameter_group: format!("parameters:{fp}"),
        shared_write_weight: dense_spec.product.down.weight.id.as_str().to_owned(),
        write_bias: None,
        activation_equation: declared_gated_product(dense_spec.product.policy)
            .expect("declared Gemma GELU policy"),
        input_normalization: declared_norm(&block_spec.pre_feed_forward),
        output_normalization: Some(sparse_declaration.as_ref().map_or_else(
            || declared_norm(&block_spec.post_feed_forward),
            |d| declared_norm(&d.norms[0]),
        )),
        output_gate: None,
        residual_scale: ComponentScalar::new(1.0),
    });
    let mut units = axes(count);
    units[2].name = "component".into();
    g.component_observation(
        &dense,
        format!("{path}.dense_feed_forward.units"),
        "Exact GELU gate times value before down projection",
        units,
    );
    if sparse {
        super::super::routed_components::gated(
            g,
            super::super::routed_components::Site {
                node: &ffn,
                layer,
                routing: &format!("{path}.routing"),
                input: Some(format!("{path}.routed_feed_forward.input")),
                normalization: Some(declared_norm(
                    &sparse_declaration.as_ref().unwrap().norms[1],
                )),
                residual_scale: None,
            },
            Ok(sparse_declaration.as_ref().unwrap().routed.experts.clone()),
        );
        if let Some(group) =
            g.descriptor.routed_components.iter_mut().find(|group| {
                group.node_id == routed && group.layer_index == layer && group.bank == 0
            })
        {
            group.write_output = Some(format!("{path}.routed_feed_forward.write"));
            group.output = Some(format!("{path}.routed_feed_forward.output"));
        }
    }
    if let Some(spec) = &block_spec.per_layer {
        let node = format!("{block}.per_layer_write");
        g.node(
            &node,
            ArchitectureNodeKind::Projector,
            Some(&block),
            Some(
                spec.projection
                    .weight
                    .id
                    .as_str()
                    .strip_suffix(".weight")
                    .expect("declared per-layer projection"),
            ),
            Some(width),
        );
        for boundary in ["input", "write", "output"] {
            g.component_observation(
                &node,
                format!("{path}.per_layer.{boundary}"),
                "Input-gated prepared per-layer residual contribution",
                axes(width),
            );
        }
        post_norm(
            g,
            (
                (&block_spec.per_layer.as_ref().unwrap().norm).dimensions as usize,
                declared_norm(&block_spec.per_layer.as_ref().unwrap().norm),
            ),
            &node,
            format!("{node}.post_norm"),
            format!("{path}.per_layer.write.effective"),
            format!("{path}.per_layer.output"),
        );
    }
    let scale_id = format!("{block}.residual_scale");
    let parameter = block_spec.residual_scale.weight.id.as_str().to_owned();
    g.node(
        &scale_id,
        ArchitectureNodeKind::Projector,
        Some(&block),
        Some(&parameter),
        Some(width),
    );
    g.descriptor
        .component_transforms
        .push(ComponentTensorTransform {
            id: scale_id.clone(),
            node_id: scale_id,
            input: format!("{path}.residual.before_scale.effective"),
            output: format!("{path}.residual.scaled"),
            effective_output: Some(format!("{path}.residual.scaled.effective")),
            equation: ComponentTensorTransformEquation::LearnedScale {
                scale: declared_transform_parameter(
                    &block_spec.residual_scale.weight,
                    format!("parameters:{parameter}"),
                ),
            },
        });
}

pub(in crate::discovery) fn readout(
    g: &mut Builder,
    c: &ModelArgs,
    declaration: &crate::gemma4::text::pinned::Declaration,
) {
    let root = declaration.root();
    let width = declaration.embedding().dimensions as usize;
    let embedding_weight = declaration.embedding().weight.id.as_str().to_owned();
    let normalization = declared_norm(declaration.normalization());
    let weight = declaration.head().map_or_else(
        || embedding_weight.clone(),
        |head| head.weight.id.as_str().to_owned(),
    );
    g.attach_parameters(
        "text.embedding",
        embedding_weight
            .strip_suffix(".weight")
            .expect("declared token weight"),
    );
    g.attach_parameters("output", weight.strip_suffix(".weight").unwrap());
    g.node(
        "output.norm",
        ArchitectureNodeKind::Normalization,
        Some("output"),
        normalization
            .gain
            .as_deref()
            .and_then(|gain| gain.strip_suffix(".weight")),
        Some(width),
    );
    for (node, path) in [
        ("assembly", "readout.embedding"),
        ("output.norm", "readout.residual"),
        ("output.norm", "readout.normalized"),
    ] {
        g.component_observation(
            node,
            path.into(),
            "Gemma4 assembled residual and final normalization",
            axes(width),
        );
    }
    let mut scores = axes(declaration.embedding().vocabulary as usize);
    scores[2].name = "vocabulary".into();
    g.component_observation(
        "output",
        "readout.linear".into(),
        "Actual affine vocabulary scores before optional softcap",
        scores,
    );
    let mut other_writes = Vec::new();
    for (layer, policy) in c.layer_schedule.iter().enumerate() {
        let path = format!("{root}.layers.{layer}");
        if policy.feed_forward == FeedForwardPolicy::DenseWithSparseMoe {
            other_writes.push(ComponentResidualWrite {
                input: Some(format!("{path}.attention.residual.effective")),
                layer_index: layer,
                node_id: format!("decoder.{layer}.feed_forward"),
                output: format!("{path}.feed_forward.output"),
                effective_output: format!("{path}.feed_forward.output.effective"),
                residual_scale: ComponentScalar::new(1.0),
            });
        }
        if c.hidden_size_per_layer_input > 0 {
            other_writes.push(ComponentResidualWrite {
                input: Some(format!("{path}.per_layer.input")),
                layer_index: layer,
                node_id: format!("decoder.{layer}.per_layer_write"),
                output: format!("{path}.per_layer.output"),
                effective_output: format!("{path}.per_layer.output.effective"),
                residual_scale: ComponentScalar::new(1.0),
            });
        }
    }
    g.descriptor.component_readout = Some(ComponentReadout {
        embedding: "readout.embedding".into(),
        embedding_weight,
        embedding_scale: ComponentScalar::new((declaration.embedding().dimensions as f32).sqrt()),
        embedding_normalization: None,
        token_embedding_normalization: None,
        tied_embeddings: declaration.head().is_none(),
        equation: ComponentReadoutEquation {
            residual_cycles: vec![],
            stream_residual: None,
            residual: "readout.residual".into(),
            normalized: "readout.normalized".into(),
            projection_input: Some("readout.projection_input".into()),
            linear_scores: "readout.linear".into(),
            logits: MODEL_LOGITS_OBSERVATION_PATH.into(),
            normalization,
            weight,
            bias: None,
            score_writes: vec![],
            output_transform: c.final_logit_softcapping.map_or(
                ComponentOutputTransform::Identity,
                |cap| ComponentOutputTransform::Softcap {
                    cap: ComponentScalar::new(cap),
                },
            ),
            block_normalizations: vec![],
            block_transforms: (0..c.num_hidden_layers())
                .map(|layer| ComponentResidualTransform {
                    layer_index: layer,
                    transform_id: format!("decoder.{layer}.residual_scale"),
                })
                .collect(),
            other_writes,
        },
    });
}
