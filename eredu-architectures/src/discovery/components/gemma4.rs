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

fn post_norm(
    g: &mut Builder,
    c: &ModelArgs,
    node: &str,
    id: String,
    field: String,
    input: String,
    output: String,
) {
    g.node(
        &id,
        ArchitectureNodeKind::Normalization,
        Some(node),
        Some(&field),
        Some(c.hidden_size as usize),
    );
    g.descriptor
        .component_transforms
        .push(ComponentTensorTransform {
            id,
            node_id: node.into(),
            input,
            effective_output: Some(format!("{output}.effective")),
            output,
            equation: ComponentTensorTransformEquation::Normalization {
                normalization: norm(c, Some(format!("{field}.weight"))),
            },
        });
}

pub(in crate::discovery) fn unit(g: &mut Builder, c: &ModelArgs, layer: usize) {
    let policy = c.layer_policy(layer).expect("admitted Gemma layer");
    let sparse = policy.feed_forward == FeedForwardPolicy::DenseWithSparseMoe;
    let block = format!("decoder.{layer}");
    let path = format!("model.language_model.layers.{layer}");
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
    let heads = c.num_attention_heads as usize;
    let kv = policy.num_key_value_heads.get() as usize;
    let dim = policy.head_dim.get() as usize;
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
        c,
        &attention,
        format!("{attention}.post_norm"),
        format!("{path}.post_attention_layernorm"),
        format!("{path}.attention.write.effective"),
        format!("{path}.attention.output"),
    );
    post_norm(
        g,
        c,
        &ffn,
        format!("{ffn}.post_norm"),
        format!("{path}.post_feedforward_layernorm"),
        format!("{path}.feed_forward.write.effective"),
        format!("{path}.feed_forward.output"),
    );
    if sparse {
        post_norm(
            g,
            c,
            &dense,
            format!("{dense}.post_norm"),
            format!("{path}.post_feedforward_layernorm_1"),
            format!("{path}.dense_feed_forward.write.effective"),
            format!("{path}.dense_feed_forward.output"),
        );
        post_norm(
            g,
            c,
            &routed,
            format!("{routed}.post_norm"),
            format!("{path}.post_feedforward_layernorm_2"),
            format!("{path}.routed_feed_forward.write.effective"),
            format!("{path}.routed_feed_forward.output"),
        );
    }
    let publisher = if policy.key_value == eredu_nn::AttentionStateSource::Shared {
        (0..layer)
            .rev()
            .find(|index| {
                let earlier = c.layer_policy(*index).expect("preceding Gemma layer");
                earlier.attention == policy.attention && earlier.key_value.publishes_state()
            })
            .expect("admitted Gemma shared-state publisher")
    } else {
        layer
    };
    let published = c.layer_policy(publisher).expect("Gemma publisher");
    let source_ap = format!("model.language_model.layers.{publisher}.self_attn");
    let head_rows = |sharing| ComponentRowMapping::HeadRows {
        offset: 0,
        component_head_width: dim,
        read_head_width: dim,
        component_heads_per_read_head: sharing,
        read_head_stride: None,
    };
    let read = |field: &str, role, source: bool| {
        let prefix = if source { &source_ap } else { &ap };
        let weight = format!("{prefix}.{field}.weight");
        let query = role == ComponentReadRole::Query;
        let value = role == ComponentReadRole::Value;
        let mut normalization = norm(
            c,
            (!value).then(|| format!("{prefix}.{}_norm.weight", if query { "q" } else { "k" })),
        );
        if value {
            normalization.epsilon = ComponentScalar::new(1e-6);
        }
        ComponentRead {
            source: (source && publisher != layer).then(|| {
                ComponentReadSource::PublishedAttentionState {
                    component_group: format!("decoder.{publisher}.attention.channels"),
                }
            }),
            role,
            shared_weight: weight.clone(),
            weight,
            parameter_group: format!("parameters:{prefix}"),
            bias: c.attention_bias.then(|| format!("{prefix}.{field}.bias")),
            rows: head_rows(if query { 1 } else { heads / kv }),
            projection_output: None,
            head_normalization: Some(ComponentHeadNormalization {
                output_scale: ComponentScalar::new(1.0),
                heads: if query { heads } else { kv },
                head_width: dim,
                independent_gains: false,
                normalization,
            }),
            input_projections: vec![],
        }
    };
    let value_field =
        if published.key_value.value() == Some(eredu_nn::AttentionValueSource::ReuseKey) {
            "k_proj"
        } else {
            "v_proj"
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
            read("q_proj", ComponentReadRole::Query, false),
            read("k_proj", ComponentReadRole::Key, true),
            read(value_field, ComponentReadRole::Value, true),
        ],
        routed_reads: vec![],
        write_weight: format!("{ap}.o_proj.weight"),
        write_input_projection: None,
        write_parameter_group: format!("parameters:{ap}"),
        shared_write_weight: format!("{ap}.o_proj.weight"),
        write_bias: c.attention_bias.then(|| format!("{ap}.o_proj.bias")),
        activation_equation: ComponentActivation::Attention {
            query_heads: heads,
            key_value_heads: kv,
            head_width: dim,
            output_gate: None,
        },
        input_normalization: norm(c, Some(format!("{path}.input_layernorm.weight"))),
        output_normalization: Some(norm(
            c,
            Some(format!("{path}.post_attention_layernorm.weight")),
        )),
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
    let count = policy.intermediate_size.get() as usize;
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
            ("gate_proj", ComponentReadRole::Gate),
            ("up_proj", ComponentReadRole::Value),
        ]
        .into_iter()
        .map(|(field, role)| {
            let weight = format!("{fp}.{field}.weight");
            ComponentRead {
                source: None,
                role,
                shared_weight: weight.clone(),
                weight,
                parameter_group: format!("parameters:{fp}"),
                bias: None,
                rows: ComponentRowMapping::Direct { offset: 0 },
                projection_output: None,
                head_normalization: None,
                input_projections: vec![],
            }
        })
        .collect(),
        routed_reads: vec![],
        write_weight: format!("{fp}.down_proj.weight"),
        write_input_projection: None,
        write_parameter_group: format!("parameters:{fp}"),
        shared_write_weight: format!("{fp}.down_proj.weight"),
        write_bias: None,
        activation_equation: ComponentActivation::Gated {
            activation: ComponentNonlinearity::Gelu,
            gate_upper_bound: None,
            value_absolute_bound: None,
            value_offset: ComponentScalar::new(0.0),
        },
        input_normalization: norm(c, Some(format!("{path}.pre_feedforward_layernorm.weight"))),
        output_normalization: Some(norm(
            c,
            Some(format!(
                "{path}.{}.weight",
                if sparse {
                    "post_feedforward_layernorm_1"
                } else {
                    "post_feedforward_layernorm"
                }
            )),
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
                normalization: Some(norm(
                    c,
                    Some(format!("{path}.pre_feedforward_layernorm_2.weight")),
                )),
                residual_scale: None,
            },
            crate::gemma4::text::expert_bank_spec(c, layer),
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
    if c.hidden_size_per_layer_input > 0 {
        let node = format!("{block}.per_layer_write");
        g.node(
            &node,
            ArchitectureNodeKind::Projector,
            Some(&block),
            Some(&format!("{path}.per_layer_projection")),
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
            c,
            &node,
            format!("{node}.post_norm"),
            format!("{path}.post_per_layer_input_norm"),
            format!("{path}.per_layer.write.effective"),
            format!("{path}.per_layer.output"),
        );
    }
    let scale_id = format!("{block}.residual_scale");
    let parameter = format!("{path}.layer_scalar");
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
                scale: ComponentTransformParameter {
                    parameter: parameter.clone(),
                    shared_parameter: parameter.clone(),
                    parameter_group: format!("parameters:{parameter}"),
                },
            },
        });
}

pub(in crate::discovery) fn readout(g: &mut Builder, c: &ModelArgs) {
    let width = c.hidden_size as usize;
    let root = "model.language_model";
    g.attach_parameters("text.embedding", &format!("{root}.embed_tokens"));
    let weight = if c.tie_word_embeddings {
        format!("{root}.embed_tokens.weight")
    } else {
        "lm_head.weight".into()
    };
    g.attach_parameters("output", weight.strip_suffix(".weight").unwrap());
    g.node(
        "output.norm",
        ArchitectureNodeKind::Normalization,
        Some("output"),
        Some(&format!("{root}.norm")),
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
    let mut scores = axes(c.vocab_size as usize);
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
        embedding_weight: format!("{root}.embed_tokens.weight"),
        embedding_scale: ComponentScalar::new((c.hidden_size as f32).sqrt()),
        embedding_normalization: None,
        token_embedding_normalization: None,
        tied_embeddings: c.tie_word_embeddings,
        equation: ComponentReadoutEquation {
            stream_residual: None,
            residual: "readout.residual".into(),
            normalized: "readout.normalized".into(),
            projection_input: Some("readout.projection_input".into()),
            linear_scores: "readout.linear".into(),
            logits: MODEL_LOGITS_OBSERVATION_PATH.into(),
            normalization: norm(c, Some(format!("{root}.norm.weight"))),
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
