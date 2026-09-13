//! Inkling component equations share the ordinary decoder's normalized policy.
use super::*;
mod prediction;
use crate::inkling::{FeedForwardPolicy, LayerPolicy, TextArgs};
pub(in crate::discovery) use prediction::declare as prediction;

fn parameter(name: String, prefix: &str) -> ComponentTransformParameter {
    ComponentTransformParameter {
        shared_parameter: name.clone(),
        parameter: name,
        parameter_group: format!("parameters:{prefix}"),
    }
}

pub(in crate::discovery) fn decoder_transforms(
    g: &mut Builder,
    text: &TextArgs,
    policy: LayerPolicy,
    block: &str,
    path: &str,
) {
    let width = text.hidden_size as usize;
    let local = policy.attention.window().is_some();
    let kv_width = (text.key_value_heads(local) * text.attention_head_dim(local)) as usize;
    for (suffix, parent, field, channels, input, output, mutable) in [
        (
            "attention.key.convolution",
            "attention",
            "self_attn.k_sconv",
            kv_width,
            "attention.key.projected",
            "attention.key.convolved",
            false,
        ),
        (
            "attention.value.convolution",
            "attention",
            "self_attn.v_sconv",
            kv_width,
            "attention.value.projected",
            "attention.value.convolved",
            false,
        ),
        (
            "attention.convolution",
            "attention",
            "attn_sconv",
            width,
            "attention.write.effective",
            "attention.contribution",
            true,
        ),
        (
            "feed_forward.convolution",
            "feed_forward",
            "mlp_sconv",
            width,
            "feed_forward.write.effective",
            "feed_forward.contribution",
            true,
        ),
    ] {
        let node = format!("{block}.{suffix}");
        let parent = format!("{block}.{parent}");
        let prefix = format!("{path}.{field}");
        let input = format!("{path}.{input}");
        let output = format!("{path}.{output}");
        g.node(
            &node,
            ArchitectureNodeKind::Mixer,
            Some(&parent),
            Some(&prefix),
            Some(channels),
        );
        if mutable {
            g.component_observation(
                &parent,
                input.trim_end_matches(".effective").into(),
                "Operator write consumed by causal convolution",
                axes(channels),
            );
            g.component_observation(
                &node,
                output.clone(),
                "Complete current-position residual contribution after causal convolution",
                axes(channels),
            );
        } else {
            g.read_only_observation(
                &parent,
                input.clone(),
                "Current-position affine projection before causal convolution",
                axes(channels),
            );
            g.read_only_observation(
                &node,
                output.clone(),
                "Current-position causal projection before head normalization or cache reuse",
                axes(channels),
            );
        }
        g.descriptor
            .component_transforms
            .push(ComponentTensorTransform {
                id: node.clone(),
                node_id: node,
                input,
                effective_output: mutable.then(|| format!("{output}.effective")),
                output,
                equation: ComponentTensorTransformEquation::CausalDepthwiseConvolution {
                    kernel: parameter(format!("{prefix}.weight"), &prefix),
                    channels,
                    taps: text.sconv_kernel_size as usize,
                    residual: true,
                    activation: None,
                },
            });
    }
    let groups = text.query_heads(local) as usize;
    let input_width = text.d_rel as usize;
    let output_width = policy
        .attention
        .window()
        .map_or(text.rel_extent as usize, |window| window.get() as usize);
    let node = format!("{block}.attention.relative.projection");
    let parent = format!("{block}.attention");
    let name = format!("{path}.self_attn.rel_proj");
    let input = format!("{path}.attention.relative.projected");
    let output = format!("{path}.attention.relative.profiles");
    g.node(
        &node,
        ArchitectureNodeKind::Projector,
        Some(&parent),
        Some(&name),
        None,
    );
    g.read_only_observation(
        &parent,
        input.clone(),
        "Current-position relative-query affine read",
        axes(groups * input_width),
    );
    let mut shape = axes(output_width);
    shape[2].name = "relative_position".into();
    shape.insert(
        2,
        TensorAxis {
            name: "head".into(),
            dimension: SymbolicDimension::Known(groups),
        },
    );
    g.read_only_observation(
        &node,
        output.clone(),
        "Current-position learned relative profiles before attention scoring",
        shape,
    );
    g.descriptor
        .component_transforms
        .push(ComponentTensorTransform {
            id: node.clone(),
            node_id: node,
            input,
            output,
            effective_output: None,
            equation: ComponentTensorTransformEquation::SharedGroupedProjection {
                weight: parameter(name.clone(), &name),
                groups,
                input_width,
                output_width,
            },
        });
    if policy.feed_forward == FeedForwardPolicy::Dense {
        let node = format!("{block}.feed_forward.scale");
        let parent = format!("{block}.feed_forward");
        let name = format!("{path}.dense_global_scale");
        g.node(
            &node,
            ArchitectureNodeKind::Projector,
            Some(&parent),
            Some(&name),
            Some(width),
        );
        g.component_observation(
            &parent,
            format!("{path}.feed_forward.projection"),
            "Dense affine write before the learned scalar",
            axes(width),
        );
        g.read_only_observation(
            &node,
            format!("{path}.feed_forward.global_scale"),
            "Effective learned scalar multiplied into the dense write",
            vec![TensorAxis {
                name: "scalar".into(),
                dimension: SymbolicDimension::Known(1),
            }],
        );
        g.descriptor
            .component_transforms
            .push(ComponentTensorTransform {
                id: node.clone(),
                node_id: node,
                input: format!("{path}.feed_forward.projection.effective"),
                output: format!("{path}.feed_forward.write"),
                effective_output: Some(format!("{path}.feed_forward.write.effective")),
                equation: ComponentTensorTransformEquation::LearnedScale {
                    scale: parameter(name.clone(), &name),
                },
            });
    }
}

pub(in crate::discovery) fn decoder_scalars(
    g: &mut Builder,
    text: &TextArgs,
    policy: LayerPolicy,
    layer: usize,
    block: &str,
    path: &str,
) {
    let local = policy.attention.window().is_some();
    let heads = text.query_heads(local) as usize;
    let kv = text.key_value_heads(local) as usize;
    let dim = text.attention_head_dim(local) as usize;
    let width = text.hidden_size as usize;
    let attention = format!("{block}.attention");
    let feed_forward = format!("{block}.feed_forward");
    let prefix = format!("{path}.self_attn");
    let normalization = |gain: String| ComponentNormalization {
        kind: ComponentNormalizationKind::Rms,
        epsilon: ComponentScalar::new(text.rms_norm_eps),
        gain: Some(gain),
        gain_offset: ComponentScalar::new(0.0),
        bias: None,
        groups: 1,
    };
    g.attach_parameters(&attention, &prefix);
    g.attach_parameters(&format!("{block}.norm"), &format!("{path}.input_layernorm"));
    g.get_mut(&format!("{block}.norm")).completeness = DescriptionCompleteness::Complete;
    g.node(
        &format!("{feed_forward}.norm"),
        ArchitectureNodeKind::Normalization,
        Some(block),
        Some(&format!("{path}.post_attention_layernorm")),
        Some(width),
    );
    let head_rows =
        |read_head_width, component_heads_per_read_head| ComponentRowMapping::HeadRows {
            offset: 0,
            component_head_width: dim,
            read_head_width,
            component_heads_per_read_head,
            read_head_stride: None,
        };
    let read = |field: &str, role, rows, output: &str, norm: Option<(&str, usize)>, bias: bool| {
        let weight = format!("{prefix}.{field}.weight");
        ComponentRead {
            source: None,
            role,
            shared_weight: weight.clone(),
            weight,
            parameter_group: format!("parameters:{prefix}"),
            bias: bias.then(|| format!("{prefix}.{field}.bias")),
            rows,
            projection_output: Some(format!("{path}.attention.{output}")),
            head_normalization: norm.map(|(field, heads)| ComponentHeadNormalization {
                output_scale: ComponentScalar::new(1.0),
                heads,
                head_width: dim,
                independent_gains: false,
                normalization: normalization(format!("{prefix}.{field}.weight")),
            }),
            input_projections: Vec::new(),
        }
    };
    let reads = vec![
        read(
            "q_proj",
            ComponentReadRole::Query,
            head_rows(dim, 1),
            "query.projected",
            Some(("q_norm", heads)),
            text.q_bias,
        ),
        read(
            "k_proj",
            ComponentReadRole::Key,
            head_rows(dim, heads / kv),
            "key.projected",
            Some(("k_norm", kv)),
            false,
        ),
        read(
            "v_proj",
            ComponentReadRole::Value,
            ComponentRowMapping::GroupedQuery {
                offset: 0,
                head_width: dim,
                queries_per_kv: heads / kv,
            },
            "value.projected",
            None,
            false,
        ),
        // Relative-query features are an additional query dependency. Their
        // projection_output joins the shared learned-position table transform.
        read(
            "r_proj",
            ComponentReadRole::Query,
            head_rows(text.d_rel as usize, 1),
            "relative.projected",
            None,
            false,
        ),
    ];
    let mut group = ComponentGroup {
        id: format!("{attention}.channels"),
        node_id: attention.clone(),
        layer_index: layer,
        count: heads * dim,
        activation: format!("{path}.attention.channels"),
        effective_activation: format!("{path}.attention.channels.effective"),
        write_input: Some(format!("{path}.attention.write_input")),
        write_output: Some(format!("{path}.attention.write")),
        output: Some(format!("{path}.attention.write")),
        write_partition: ComponentWritePartition::Complete,
        input: format!("{path}.attention.input"),
        reads,
        routed_reads: Vec::new(),
        write_weight: format!("{prefix}.o_proj.weight"),
        write_input_projection: None,
        write_parameter_group: format!("parameters:{prefix}"),
        shared_write_weight: format!("{prefix}.o_proj.weight"),
        write_bias: text.o_bias.then(|| format!("{prefix}.o_proj.bias")),
        activation_equation: ComponentActivation::Attention {
            query_heads: heads,
            key_value_heads: kv,
            head_width: dim,
            output_gate: None,
        },
        input_normalization: normalization(format!("{path}.input_layernorm.weight")),
        output_normalization: None,
        output_gate: None,
        residual_scale: ComponentScalar::new(1.0),
    };
    g.component_observation(
        &attention,
        group.input.clone(),
        "Normalized attention input consumed by all content and relative reads",
        axes(width),
    );
    let mut component_axes = axes(heads * dim);
    component_axes[2].name = "component".into();
    g.component_observation(
        &attention,
        group.activation.clone(),
        "Value-aggregated attention channels before output projection",
        component_axes,
    );
    g.read_only_observation(
        &attention,
        format!("{path}.attention.query.projected"),
        "Content-query projection before head normalization",
        axes(heads * dim),
    );
    g.descriptor.components.push(group.clone());
    g.component_observation(
        &feed_forward,
        format!("{path}.feed_forward.input"),
        "Normalized feed-forward input",
        axes(width),
    );
    if policy.feed_forward != FeedForwardPolicy::Dense {
        return;
    }
    let prefix = format!("{path}.dense");
    g.attach_parameters(&feed_forward, &prefix);
    group.id = format!("{feed_forward}.units");
    group.node_id = feed_forward.clone();
    group.count = text.dense_intermediate_size() as usize;
    group.activation = format!("{path}.feed_forward.units");
    group.effective_activation = format!("{path}.feed_forward.units.effective");
    group.write_input = Some(format!("{path}.feed_forward.write_input"));
    group.write_output = Some(format!("{path}.feed_forward.projection"));
    group.output = Some(format!("{path}.feed_forward.projection"));
    group.input = format!("{path}.feed_forward.input");
    group.reads = [
        ("gate_proj", ComponentReadRole::Gate),
        ("up_proj", ComponentReadRole::Value),
    ]
    .into_iter()
    .map(|(field, role)| {
        let weight = format!("{prefix}.{field}.weight");
        ComponentRead {
            source: None,
            role,
            shared_weight: weight.clone(),
            weight,
            parameter_group: format!("parameters:{prefix}"),
            bias: None,
            rows: ComponentRowMapping::Direct { offset: 0 },
            projection_output: None,
            head_normalization: None,
            input_projections: Vec::new(),
        }
    })
    .collect();
    group.write_weight = format!("{prefix}.down_proj.weight");
    group.shared_write_weight = group.write_weight.clone();
    group.write_parameter_group = format!("parameters:{prefix}");
    group.write_bias = None;
    group.activation_equation = ComponentActivation::Gated {
        activation: ComponentNonlinearity::Silu {
            multiplier: ComponentScalar::new(1.0),
        },
        gate_upper_bound: None,
        value_absolute_bound: None,
        value_offset: ComponentScalar::new(0.0),
    };
    group.input_normalization = normalization(format!("{path}.post_attention_layernorm.weight"));
    let mut component_axes = axes(group.count);
    component_axes[2].name = "component".into();
    g.component_observation(
        &feed_forward,
        group.activation.clone(),
        "SwiGLU units before the down projection and learned branch scale",
        component_axes,
    );
    g.descriptor.components.push(group);
}

pub(in crate::discovery) fn target_readout(g: &mut Builder, text: &TextArgs) {
    let width = text.hidden_size as usize;
    let vocabulary = text.unpadded_vocab_size.unwrap_or(text.vocab_size) as usize;
    let normalization = |gain: &str| ComponentNormalization {
        kind: ComponentNormalizationKind::Rms,
        epsilon: ComponentScalar::new(text.rms_norm_eps),
        gain: Some(gain.into()),
        gain_offset: ComponentScalar::new(0.0),
        bias: None,
        groups: 1,
    };
    g.attach_parameters("embedding", "model.embed_tokens");
    g.attach_parameters("assembly", "model.embed_norm");
    g.attach_parameters("output", "lm_head");
    g.node(
        "output.norm",
        ArchitectureNodeKind::Normalization,
        None,
        Some("model.norm"),
        Some(width),
    );
    let last = format!("decoder.{}", text.num_hidden_layers - 1);
    g.descriptor
        .edges
        .retain(|edge| !(edge.from == last && edge.to == "output"));
    g.edge(&last, "output.norm", ArchitectureEdgeKind::Data);
    g.edge("output.norm", "output", ArchitectureEdgeKind::Data);
    g.component_observation(
        "assembly",
        "readout.embedding".into(),
        "Assembled text/media embedding after embedding normalization",
        axes(width),
    );
    g.component_observation(
        "output.norm",
        "readout.residual".into(),
        "Final residual before output normalization",
        axes(width),
    );
    g.component_observation(
        "output.norm",
        "readout.normalized".into(),
        "Effective normalized residual before muP scaling",
        axes(width),
    );
    g.read_only_observation(
        "output",
        "readout.scaled".into(),
        "Exact muP-scaled residual before projection input arithmetic",
        axes(width),
    );
    let mut scores = axes(vocabulary);
    scores[2].name = "vocabulary".into();
    g.get_mut("output").output_axes = Some(scores.clone());
    if let Some(point) = g
        .descriptor
        .observations
        .points
        .iter_mut()
        .find(|point| point.path == MODEL_LOGITS_OBSERVATION_PATH)
    {
        point.axes = Some(scores.clone());
    }
    g.component_observation(
        "output",
        "readout.linear".into(),
        "Unpadded vocabulary scores before sampling",
        scores,
    );
    let other_writes = (0..text.num_hidden_layers as usize)
        .flat_map(|layer_index| {
            ["attention", "feed_forward"].map(|branch| ComponentResidualWrite {
                input: Some(format!("model.layers.{layer_index}.{branch}.input")),
                layer_index,
                // Child unit/channel groups describe the pre-convolution projection.
                // Include this whole contribution once, or expand its transforms.
                node_id: format!("decoder.{layer_index}.{branch}"),
                output: format!("model.layers.{layer_index}.{branch}.contribution"),
                effective_output: format!(
                    "model.layers.{layer_index}.{branch}.contribution.effective"
                ),
                residual_scale: ComponentScalar::new(1.0),
            })
        })
        .collect();
    g.descriptor.component_readout = Some(ComponentReadout {
        token_embedding_normalization: None,
        embedding: "readout.embedding".into(),
        embedding_weight: "model.embed_tokens.weight".into(),
        embedding_scale: ComponentScalar::new(1.0),
        embedding_normalization: Some(normalization("model.embed_norm.weight")),
        tied_embeddings: false,
        equation: ComponentReadoutEquation {
            block_transforms: vec![],
            residual: "readout.residual".into(),
            normalized: "readout.normalized".into(),
            projection_input: Some("readout.projection_input".into()),
            linear_scores: "readout.linear".into(),
            logits: MODEL_LOGITS_OBSERVATION_PATH.into(),
            normalization: normalization("model.norm.weight"),
            weight: "lm_head.weight".into(),
            bias: None,
            score_writes: vec![],
            output_transform: ComponentOutputTransform::Identity,
            block_normalizations: vec![],
            other_writes,
            stream_residual: None,
        },
    });
    g.descriptor
        .component_transforms
        .push(ComponentTensorTransform {
            id: "output.scale".into(),
            node_id: "output".into(),
            input: "readout.normalized.effective".into(),
            output: "readout.scaled".into(),
            effective_output: None,
            equation: ComponentTensorTransformEquation::ConstantScale {
                scale: ComponentScalar::new(text.logits_mup_width_multiplier.recip()),
            },
        });
}

pub(in crate::discovery) fn decoder_routed(
    g: &mut Builder,
    args: &crate::inkling::ModelArgs,
    layer: usize,
    block: &str,
    path: &str,
) {
    use crate::discovery::routed_components::{gated, Site};
    let text = &args.text_config;
    if text
        .layer_schedule
        .get(layer)
        .expect("declared decoder layer")
        .feed_forward
        != FeedForwardPolicy::SparseMoe
    {
        return;
    }
    let node = format!("{block}.feed_forward");
    let prefix = format!("{path}.moe");
    g.attach_parameters(&node, &prefix);
    let attributes = g
        .get_mut(&node)
        .moe
        .clone()
        .expect("declared sparse branch");
    // The joint selector does not emit the ordinary standalone router event
    // protocol. Unit evidence carries the actual jointly normalized coefficients.
    g.moe(&node, &prefix, attributes, None, text.hidden_size as usize);
    g.attach_parameters(&format!("{node}.router"), &format!("{prefix}.router"));
    g.edge(
        &format!("{node}.router"),
        &format!("{node}.shared"),
        ArchitectureEdgeKind::Routing,
    );
    let normalization = crate::discovery::routed_components::rms(
        format!("{path}.post_attention_layernorm.weight"),
        text.rms_norm_eps,
        0.0,
    );
    for (owner, routing, cache_layer, routes, parameter_prefix) in [
        (
            node.clone(),
            format!("{path}.routing"),
            layer,
            text.num_experts_per_tok as usize,
            prefix.clone(),
        ),
        (
            format!("{node}.shared"),
            format!("{path}.shared.routing"),
            text.num_hidden_layers as usize + layer,
            text.n_shared_experts as usize,
            format!("{prefix}.shared_experts"),
        ),
    ] {
        g.attach_parameters(&owner, &parameter_prefix);
        gated(
            g,
            Site {
                node: &owner,
                layer,
                routing: &routing,
                input: Some(format!("{path}.feed_forward.input")),
                normalization: Some(normalization.clone()),
                // The outer causal convolution is declared as a tensor transform,
                // not a scalar residual multiplier on this grouped contribution.
                residual_scale: None,
            },
            crate::inkling::text::expert_bank_spec(args, cache_layer),
        );
        g.descriptor
            .routed_components
            .last_mut()
            .expect("valid expert specification")
            .routes_per_token = Some(routes);
    }
}
