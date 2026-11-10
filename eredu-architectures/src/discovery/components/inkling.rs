//! Inkling component equations share the ordinary decoder's normalized policy.
use super::*;
mod prediction;
use crate::inkling::{FeedForwardPolicy, LayerPolicy};
pub(in crate::discovery) use prediction::declare as prediction;

pub(in crate::discovery) fn media(g: &mut Builder, args: &crate::inkling::ModelArgs) {
    fn attach(g: &mut Builder, node: &str, spec: &dyn eredu_nn::DeclareLogicalParameters) {
        for parameter in spec
            .logical_parameters()
            .expect("admitted Inkling media declarations")
        {
            let id = parameter.parameter.id.as_str();
            g.attach_parameters(node, id.strip_suffix(".weight").unwrap_or(id));
        }
    }
    if let Some(config) = &args.audio_config {
        let spec = config.parameters();
        attach(g, "audio", spec);
        g.get_mut("audio").output_axes = Some(axes(spec.embedding.dimensions as usize));
    }
    if let Some(config) = &args.vision_config {
        for layer in 0..config.num_hidden_layers as usize {
            let spec = config
                .parameters()
                .layer(layer)
                .expect("validated Inkling vision layer");
            attach(g, "vision", spec);
        }
        let norm = &config.parameters().final_norm;
        attach(g, "vision", norm);
        g.get_mut("vision").output_axes = Some(axes(norm.dimensions as usize));
    }
}

pub(in crate::discovery) fn decoder_transforms(
    g: &mut Builder,
    policy: LayerPolicy,
    block: &str,
    path: &str,
    parameters: &crate::inkling::text::CommonParameters,
    dense: Option<&crate::inkling::text::DenseParameters>,
) {
    let heads = &parameters.attention;
    let width = heads.projection.query.input as usize;
    for (suffix, parent, convolution, input, output, mutable) in [
        (
            "attention.key.convolution",
            "attention",
            &heads.key_convolution,
            "attention.key.projected",
            "attention.key.convolved",
            false,
        ),
        (
            "attention.value.convolution",
            "attention",
            &heads.value_convolution,
            "attention.value.projected",
            "attention.value.convolved",
            false,
        ),
        (
            "attention.convolution",
            "attention",
            &parameters.attention_convolution,
            "attention.write.effective",
            "attention.contribution",
            true,
        ),
        (
            "feed_forward.convolution",
            "feed_forward",
            &parameters.feed_forward_convolution,
            "feed_forward.write.effective",
            "feed_forward.contribution",
            true,
        ),
    ] {
        let channels = convolution.channels as usize;
        let node = format!("{block}.{suffix}");
        let parent = format!("{block}.{parent}");
        let prefix = convolution
            .weight
            .id
            .as_str()
            .strip_suffix(".weight")
            .expect("declared convolution weight")
            .to_owned();
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
                    kernel: declared_transform_parameter(
                        &convolution.weight,
                        format!("parameters:{prefix}"),
                    ),
                    channels,
                    taps: convolution.kernel_size as usize,
                    residual: true,
                    activation: None,
                },
            });
    }
    let groups = heads.projection.geometry.query_heads() as usize;
    let input_width = heads.relative_projection.shape[0] as usize;
    let output_width = heads.relative_projection.shape[1] as usize;
    let node = format!("{block}.attention.relative.projection");
    let parent = format!("{block}.attention");
    let name = heads.relative_projection.weight.id.to_string();
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
                weight: declared_transform_parameter(
                    &heads.relative_projection.weight,
                    format!("parameters:{name}"),
                ),
                groups,
                input_width,
                output_width,
            },
        });
    if policy.feed_forward == FeedForwardPolicy::Dense {
        let parameters = dense.expect("declared Inkling dense branch");
        let scalar = parameters
            .scale_declaration()
            .expect("declared branch scale");
        let node = format!("{block}.feed_forward.scale");
        let parent = format!("{block}.feed_forward");
        let name = scalar.parameter.id.to_string();
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
                dimension: SymbolicDimension::Known(scalar.shape[0]),
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
                    scale: declared_transform_parameter(
                        scalar.parameter,
                        format!("parameters:{name}"),
                    ),
                },
            });
    }
}

pub(in crate::discovery) fn decoder_scalars(
    g: &mut Builder,
    policy: LayerPolicy,
    layer: usize,
    block: &str,
    path: &str,
    parameters: &crate::inkling::text::CommonParameters,
    dense: Option<&crate::inkling::text::DenseParameters>,
) {
    let declared = &parameters.attention;
    let heads = declared.projection.geometry.query_heads() as usize;
    let kv = declared.projection.geometry.key_value_heads() as usize;
    let dim = declared.projection.geometry.key_dimensions() as usize;
    let width = declared.projection.query.input as usize;
    let attention = format!("{block}.attention");
    let feed_forward = format!("{block}.feed_forward");
    let prefix = format!("{path}.self_attn");
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
    let read = |spec: &eredu_nn::LinearSpec,
                role,
                rows,
                output: &str,
                norm: Option<(&eredu_nn::NormalizationConstructionSpec, usize)>| {
        ComponentRead {
            projection_output: Some(format!("{path}.attention.{output}")),
            head_normalization: norm.map(|(spec, heads)| ComponentHeadNormalization {
                output_scale: ComponentScalar::new(1.0),
                heads,
                head_width: dim,
                independent_gains: false,
                normalization: super::declared_normalization(spec),
            }),
            ..declared_component_read(spec, format!("parameters:{prefix}"), role, rows)
        }
    };
    let reads = vec![
        read(
            &declared.projection.query,
            ComponentReadRole::Query,
            head_rows(dim, 1),
            "query.projected",
            Some((&declared.query_norm, heads)),
        ),
        read(
            &declared.projection.key,
            ComponentReadRole::Key,
            head_rows(dim, heads / kv),
            "key.projected",
            Some((&declared.key_norm, kv)),
        ),
        read(
            &declared.projection.value,
            ComponentReadRole::Value,
            ComponentRowMapping::GroupedQuery {
                offset: 0,
                head_width: dim,
                queries_per_kv: heads / kv,
            },
            "value.projected",
            None,
        ),
        read(
            &declared.relative,
            ComponentReadRole::Query,
            head_rows(declared.relative_projection.shape[0] as usize, 1),
            "relative.projected",
            None,
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
        write_weight: declared.projection.output.weight.id.to_string(),
        write_input_projection: None,
        write_parameter_group: format!("parameters:{prefix}"),
        shared_write_weight: declared.projection.output.weight.id.to_string(),
        write_bias: declared
            .projection
            .output
            .bias
            .as_ref()
            .map(|bias| bias.id.to_string()),
        activation_equation: ComponentActivation::Attention {
            attended_head_normalization: None,
            query_heads: heads,
            key_value_heads: kv,
            head_width: dim,
            output_gate: None,
        },
        input_normalization: super::declared_normalization(&parameters.input_norm),
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
    let dense = dense.expect("declared Inkling dense component geometry");
    let prefix = format!("{path}.dense");
    g.attach_parameters(&feed_forward, &prefix);
    group.id = format!("{feed_forward}.units");
    group.node_id = feed_forward.clone();
    group.count = dense.product.gate.output as usize;
    group.activation = format!("{path}.feed_forward.units");
    group.effective_activation = format!("{path}.feed_forward.units.effective");
    group.write_input = Some(format!("{path}.feed_forward.write_input"));
    group.write_output = Some(format!("{path}.feed_forward.projection"));
    group.output = Some(format!("{path}.feed_forward.projection"));
    group.input = format!("{path}.feed_forward.input");
    group.reads = [
        (&dense.product.gate, ComponentReadRole::Gate),
        (&dense.product.up, ComponentReadRole::Value),
    ]
    .into_iter()
    .map(|(spec, role)| {
        declared_component_read(
            spec,
            format!("parameters:{prefix}"),
            role,
            ComponentRowMapping::Direct { offset: 0 },
        )
    })
    .collect();
    group.write_weight = dense.product.down.weight.id.to_string();
    group.shared_write_weight = declared_shared_weight(&dense.product.down.weight).into();
    group.write_parameter_group = format!("parameters:{prefix}");
    group.write_bias = dense
        .product
        .down
        .bias
        .as_ref()
        .map(|bias| bias.id.to_string());
    group.activation_equation = declared_gated_product(dense.product.policy)
        .expect("admitted Inkling dense gated-product policy");
    group.input_normalization = super::declared_normalization(&parameters.post_attention_norm);
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

pub(in crate::discovery) fn target_readout(g: &mut Builder, args: &crate::inkling::ModelArgs) {
    let declared = &args.pinned().ordinary;
    let text = &args.text_config;
    let embedding_norm = &args.pinned().embedding_norm;
    let embedding_weight = declared.embedding_spec().weight.id.to_string();
    let output = declared.output_projection_spec();
    let output_weight = output.weight.id.to_string();
    let norm = super::declared_normalization(declared.normalization_spec());
    let embedding_norm = super::declared_normalization(&embedding_norm);
    let width = declared.embedding_spec().dimensions as usize;
    let vocabulary = text
        .unpadded_vocab_size
        .unwrap_or(declared.embedding_spec().vocabulary) as usize;
    g.attach_parameters("embedding", embedding_weight.trim_end_matches(".weight"));
    g.attach_parameters(
        "assembly",
        embedding_norm
            .gain
            .as_ref()
            .expect("declared gain")
            .trim_end_matches(".weight"),
    );
    g.attach_parameters("output", output_weight.trim_end_matches(".weight"));
    g.node(
        "output.norm",
        ArchitectureNodeKind::Normalization,
        None,
        Some(
            norm.gain
                .as_ref()
                .expect("declared gain")
                .trim_end_matches(".weight"),
        ),
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
        embedding_weight,
        embedding_scale: ComponentScalar::new(1.0),
        embedding_normalization: Some(embedding_norm),
        tied_embeddings: declared.head_spec().is_none(),
        equation: ComponentReadoutEquation {
            residual_cycles: vec![],
            block_transforms: vec![],
            residual: "readout.residual".into(),
            normalized: "readout.normalized".into(),
            projection_input: Some("readout.projection_input".into()),
            linear_scores: "readout.linear".into(),
            logits: MODEL_LOGITS_OBSERVATION_PATH.into(),
            normalization: norm,
            weight: output_weight,
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
    let parameters = args
        .sparse(layer)
        .expect("admitted Inkling joint routing declaration");
    let selection = parameters.router.selection();
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
    g.moe(
        &node,
        &prefix,
        attributes,
        None,
        parameters.routed.input_dimensions() as usize,
    );
    g.attach_parameters(&format!("{node}.router"), &format!("{prefix}.router"));
    g.edge(
        &format!("{node}.router"),
        &format!("{node}.shared"),
        ArchitectureEdgeKind::Routing,
    );
    let common = args
        .common(layer)
        .expect("admitted Inkling normalization declaration");
    let normalization = super::declared_normalization(&common.post_attention_norm);
    for (owner, routing, bank, routes, parameter_prefix) in [
        (
            node.clone(),
            format!("{path}.routing"),
            &parameters.routed,
            selection.top_k() as usize,
            prefix.clone(),
        ),
        (
            format!("{node}.shared"),
            format!("{path}.shared.routing"),
            &parameters.shared,
            selection.always_on_groups() as usize,
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
            Ok(bank.clone()),
        );
        g.descriptor
            .routed_components
            .last_mut()
            .expect("valid expert specification")
            .routes_per_token = Some(routes);
    }
}
