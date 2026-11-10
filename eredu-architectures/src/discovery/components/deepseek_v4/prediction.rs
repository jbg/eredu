//! A sequential V4 predictor has two projected inputs and a streamed residual.
use super::*;
use crate::discovery::routed_components::{self as routed_units, Site as RoutedUnitSite};

pub(in crate::discovery) fn declare(g: &mut Builder, c: &V4Args, target_output: &str) {
    let text = &c.parameters().statics;
    let embedding_spec = text.embedding_spec();
    let embedding_prefix = declared_weight_prefix(&embedding_spec.weight);
    let output = text.head_spec().expect("V4 declares an untied output head");
    g.node(
        "prediction",
        ArchitectureNodeKind::Prediction,
        None,
        None,
        None,
    );
    g.edge(target_output, "prediction", ArchitectureEdgeKind::Data);
    for depth in 0..c.num_nextn_predict_layers as usize {
        let layer = c.num_hidden_layers as usize + depth;
        let parameters = format!("mtp.{depth}");
        let attention_spec = v4_specs::attention_spec(c, layer, &format!("{parameters}.attn"))
            .expect("validated V4 attention declaration");
        let spec =
            v4_specs::prediction_spec(c, &parameters).expect("validated V4 prediction declaration");
        let width = spec.hyper_head.hidden_size as usize;
        let root = format!("prediction.layers.{depth}");
        let decoder = format!("{root}.decoder");
        let path = format!("{parameters}.decoder");
        let fusion = format!("{root}.fusion");
        let readout = format!("{parameters}.prediction.readout");
        g.node(
            &root,
            ArchitectureNodeKind::Prediction,
            Some("prediction"),
            Some(&parameters),
            None,
        );
        g.node(&fusion, ArchitectureNodeKind::Sum, Some(&root), None, None);
        g.node(
            &decoder,
            ArchitectureNodeKind::DecoderBlock,
            Some(&root),
            Some(&parameters),
            None,
        );
        g.get_mut(&decoder).layer_index = Some(layer);
        g.get_mut(&decoder).output_axes = Some(stream_axes(&spec.hyper_head));
        g.get_mut(&fusion).output_axes = Some(stream_axes(&spec.hyper_head));
        g.descriptor.layer_groups.push(ArchitectureLayerGroup {
            id: parameters.clone(),
            label: format!("Prediction depth {depth}"),
            physical_layer_count: 1,
            passes: vec![ArchitectureExecutionPass {
                index: 0,
                executions: vec![ArchitectureLayerExecution {
                    node_id: decoder.clone(),
                    physical_layer_index: 0,
                }],
            }],
            weight_sharing: LayerWeightSharing::None,
        });
        // The prediction driver owns the outer invocation boundaries; the
        // decoder's fused input/output are a distinct inner invocation.
        for suffix in ["input", "output"] {
            g.component_observation(
                &root,
                format!("{parameters}.{suffix}"),
                "Sequential prediction invocation boundary",
                stream_axes(&spec.hyper_head),
            );
            g.component_observation(
                &decoder,
                format!("{path}.{suffix}"),
                "Sequential prediction decoder residual streams",
                stream_axes(&spec.hyper_head),
            );
        }
        g.read_only_observation(
            &root,
            format!("{parameters}.capture"),
            "Hidden state supplied to this prediction invocation",
            stream_axes(&spec.hyper_head),
        );
        let mut inputs = Vec::new();
        for (term, norm_spec, projection_spec, broadcast) in [
            (
                "embedding",
                &spec.embedding_norm,
                &spec.embedding_projection,
                true,
            ),
            ("hidden", &spec.hidden_norm, &spec.hidden_projection, false),
        ] {
            let normalization_declaration = declared_normalization(norm_spec);
            let norm_group = declared_normalization_prefix(norm_spec);
            let projection_group = declared_weight_prefix(&projection_spec.weight);
            let source = format!("{root}.{term}");
            let normalization = format!("{source}.norm");
            let projection = format!("{source}.projection");
            let observation = format!("{parameters}.prediction.{term}");
            let shape = if broadcast {
                axes(width)
            } else {
                stream_axes(&spec.hyper_head)
            };
            g.node(
                &source,
                if broadcast {
                    ArchitectureNodeKind::Embedding
                } else {
                    ArchitectureNodeKind::Processor
                },
                Some(&root),
                broadcast.then_some(embedding_prefix),
                None,
            );
            g.node(
                &normalization,
                ArchitectureNodeKind::Normalization,
                Some(&root),
                norm_group,
                None,
            );
            g.node(
                &projection,
                ArchitectureNodeKind::Projector,
                Some(&root),
                Some(projection_group),
                None,
            );
            g.edge(&source, &normalization, ArchitectureEdgeKind::Data);
            g.edge(&normalization, &projection, ArchitectureEdgeKind::Data);
            g.edge(&projection, &fusion, ArchitectureEdgeKind::Data);
            g.read_only_observation(
                &source,
                observation.clone(),
                "Actual prediction source before normalization",
                shape.clone(),
            );
            g.component_observation(
                &normalization,
                format!("{observation}.normalized"),
                "Normalized prediction source consumed by its separate projection",
                shape.clone(),
            );
            g.component_observation(
                &projection,
                format!("{observation}.projected"),
                "Projected prediction source before stream expansion and summation",
                shape,
            );
            let weight = projection_spec.weight.id.as_str().to_owned();
            inputs.push(ComponentProjectedFusionInput {
                input: observation.clone(),
                source: if broadcast {
                    ComponentFusionSource::TokenEmbedding {
                        normalization: None,
                        weight: embedding_spec.weight.id.as_str().into(),
                        shared_weight: declared_shared_weight(&embedding_spec.weight).into(),
                        parameter_group: format!("parameters:{embedding_prefix}"),
                        scale: ComponentScalar::new(1.0),
                    }
                } else {
                    ComponentFusionSource::Observation {
                        path: observation.clone(),
                    }
                },
                normalization: normalization_declaration,
                normalized: format!("{observation}.normalized.effective"),
                weight: weight.clone(),
                shared_weight: declared_shared_weight(&projection_spec.weight).into(),
                parameter_group: format!("parameters:{projection_group}"),
                bias: projection_spec
                    .bias
                    .as_ref()
                    .map(|bias| bias.id.to_string()),
                projection_input: format!("{observation}.projection_input"),
                output: format!("{observation}.projected"),
                effective_output: format!("{observation}.projected.effective"),
                expansion: if broadcast {
                    ComponentFusionExpansion::BroadcastAxis {
                        axis: 2,
                        name: "stream".into(),
                        extent: spec.hyper_head.streams as usize,
                    }
                } else {
                    ComponentFusionExpansion::Identity
                },
            });
        }
        let fusion_output = format!("{parameters}.prediction.fusion.output");
        g.component_observation(
            &fusion,
            fusion_output.clone(),
            "Sum of separate embedding and hidden projections after declared stream broadcast",
            stream_axes(&spec.hyper_head),
        );
        g.edge(&fusion, &decoder, ArchitectureEdgeKind::Data);
        let attention = format!("{decoder}.attention");
        let ffn = format!("{decoder}.feed_forward");
        g.node(
            &attention,
            ArchitectureNodeKind::Attention,
            Some(&decoder),
            Some(&format!("{parameters}.attn")),
            Some(width),
        );
        g.get_mut(&attention).attention = Some(AttentionAttributes {
            mechanism: Some(AttentionMechanism::CompressedSparse),
            causal: Some(true),
            recurrent: Some(false),
            query_heads: Some(attention_spec.geometry.query_heads() as usize),
            key_value_heads: Some(attention_spec.geometry.key_value_heads() as usize),
            key_head_dimension: Some(attention_spec.geometry.key_dimensions() as usize),
            value_head_dimension: Some(attention_spec.geometry.value_dimensions() as usize),
            receptive_field: Some(ReceptiveField::Sliding {
                window: c.sliding_window as usize,
            }),
            positional_encoding: Some(PositionalEncoding::Rotary),
            ..Default::default()
        });
        g.node(
            &ffn,
            ArchitectureNodeKind::MixtureOfExperts,
            Some(&decoder),
            Some(&format!("{parameters}.ffn")),
            Some(width),
        );
        let component_start = g.descriptor.components.len();
        let routed_start = g.descriptor.routed_components.len();
        let routed_spec = v4_specs::moe_spec(c, layer, &format!("{parameters}.ffn"), None)
            .expect("validated V4 routed declaration");
        let selector = &routed_spec.routed.parameters.selector;
        g.moe(
            &ffn,
            &format!("{parameters}.ffn"),
            moe(
                selector.selection().group_count(),
                selector.selection().top_k(),
                routed_spec.routed.shared.gate.output
                    / routed_spec
                        .routed
                        .parameters
                        .experts
                        .intermediate_dimensions(),
                selector.selection().normalize_selected(),
                RoutingScoreTransform::SqrtSoftplus,
            ),
            Some(&format!("{path}.feed_forward")),
            width,
        );
        routed_units::gated(
            g,
            RoutedUnitSite {
                node: &ffn,
                layer,
                routing: &format!("{path}.feed_forward"),
                input: Some(format!("{path}.feed_forward.input")),
                normalization: Some(declared_normalization(
                    &v4_specs::block_normalizations(c, &parameters)
                        .expect("validated V4 block normalization")[1],
                )),
                residual_scale: None,
            },
            Ok(routed_spec.routed.parameters.experts),
        );
        unit(g, c, layer, &path, &parameters, &attention, &ffn);
        let components = g.descriptor.components.split_off(component_start);
        let routed_components = g.descriptor.routed_components.split_off(routed_start);
        let collapse = format!("{root}.head.collapse");
        let head_norm = format!("{root}.head.norm");
        let head = format!("{root}.head");
        g.node(
            &collapse,
            ArchitectureNodeKind::Sum,
            Some(&root),
            Some(&parameters),
            Some(width),
        );
        g.node(
            &head_norm,
            ArchitectureNodeKind::Normalization,
            Some(&root),
            declared_normalization_prefix(&spec.output_norm),
            Some(width),
        );
        g.node(
            &head,
            ArchitectureNodeKind::OutputHead,
            Some(&root),
            Some(declared_weight_prefix(&output.weight)),
            None,
        );
        g.edge(&decoder, &collapse, ArchitectureEdgeKind::Data);
        g.edge(&collapse, &head_norm, ArchitectureEdgeKind::Data);
        g.edge(&head_norm, &head, ArchitectureEdgeKind::Data);
        g.component_observation(
            &collapse,
            format!("{readout}.streams"),
            "Prediction streams consumed by final learned collapse",
            stream_axes(&spec.hyper_head),
        );
        let mut coefficients_shape = axes(spec.hyper_head.streams as usize);
        coefficients_shape[2].name = "stream".into();
        g.read_only_observation(
            &collapse,
            format!("{readout}.stream_coefficients"),
            "Actual prediction head coefficients consumed by stream collapse",
            coefficients_shape,
        );
        g.component_observation(
            &collapse,
            format!("{readout}.residual"),
            "Collapsed prediction residual before final normalization",
            axes(width),
        );
        g.component_observation(
            &head_norm,
            format!("{readout}.normalized"),
            "Normalized prediction residual consumed by shared vocabulary head",
            axes(width),
        );
        let mut vocabulary = axes(output.output as usize);
        vocabulary[2].name = "vocabulary".into();
        g.get_mut(&head).output_axes = Some(vocabulary.clone());
        g.component_observation(
            &head,
            format!("{readout}.linear"),
            "Prediction vocabulary scores before proposal sampling",
            vocabulary,
        );
        g.descriptor.component_scopes.push(ComponentExecutionScope {
            id: format!("prediction.{depth}"),
            node_id: root,
            execution_groups: vec![parameters.clone()],
            static_parameter_roles: vec!["embedding".into(), "output".into()],
            kind: ComponentExecutionScopeKind::Prediction { depth },
            components,
            routed_components,
            residual_base: ComponentResidualBase::ProjectedSum {
                inputs,
                output: fusion_output.clone(),
                effective_output: format!("{fusion_output}.effective"),
            },
            readout: ComponentReadoutEquation {
                residual_cycles: vec![],
                block_transforms: vec![],
                score_writes: vec![],
                stream_residual: Some(ComponentStreamResidual {
                    streams: spec.hyper_head.streams as usize,
                    base: ComponentStreamBase::Streams {
                        input: format!("{fusion_output}.effective"),
                    },
                    cycles: stream_cycles(c, layer, &path, &parameters, &attention, &ffn).into(),
                    head: ComponentStreamHead {
                        input: format!("{readout}.streams.effective"),
                        coefficients: format!("{readout}.stream_coefficients"),
                        parameters: head_coefficients(
                            &format!("parameters:{parameters}"),
                            &spec.hyper_head,
                        ),
                    },
                }),
                residual: format!("{readout}.residual"),
                normalized: format!("{readout}.normalized"),
                projection_input: Some(format!("{readout}.projection_input")),
                linear_scores: format!("{readout}.linear"),
                logits: format!("{readout}.linear.effective"),
                normalization: declared_normalization(&spec.output_norm),
                weight: output.weight.id.as_str().into(),
                bias: output.bias.as_ref().map(|bias| bias.id.to_string()),
                output_transform: ComponentOutputTransform::Identity,
                block_normalizations: vec![],
                other_writes: vec![],
            },
        });
    }
}
