//! A sequential V4 predictor has two projected inputs and a streamed residual.
use super::*;
use crate::discovery::routed_components::{self as routed_units, Site as RoutedUnitSite};

pub(in crate::discovery) fn declare(g: &mut Builder, c: &V4Args, target_output: &str) {
    let width = c.hidden_size as usize;
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
        g.get_mut(&decoder).output_axes = Some(stream_axes(c));
        g.get_mut(&fusion).output_axes = Some(stream_axes(c));
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
                stream_axes(c),
            );
            g.component_observation(
                &decoder,
                format!("{path}.{suffix}"),
                "Sequential prediction decoder residual streams",
                stream_axes(c),
            );
        }
        g.read_only_observation(
            &root,
            format!("{parameters}.capture"),
            "Hidden state supplied to this prediction invocation",
            stream_axes(c),
        );
        let mut inputs = Vec::new();
        for (term, norm_field, projection_field, broadcast) in [
            ("embedding", "enorm", "e_proj", true),
            ("hidden", "hnorm", "h_proj", false),
        ] {
            let source = format!("{root}.{term}");
            let normalization = format!("{source}.norm");
            let projection = format!("{source}.projection");
            let observation = format!("{parameters}.prediction.{term}");
            let shape = if broadcast {
                axes(width)
            } else {
                stream_axes(c)
            };
            g.node(
                &source,
                if broadcast {
                    ArchitectureNodeKind::Embedding
                } else {
                    ArchitectureNodeKind::Processor
                },
                Some(&root),
                broadcast.then_some("embed"),
                None,
            );
            g.node(
                &normalization,
                ArchitectureNodeKind::Normalization,
                Some(&root),
                Some(&format!("{parameters}.{norm_field}")),
                None,
            );
            g.node(
                &projection,
                ArchitectureNodeKind::Projector,
                Some(&root),
                Some(&format!("{parameters}.{projection_field}")),
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
            let weight = format!("{parameters}.{projection_field}.weight");
            inputs.push(ComponentProjectedFusionInput {
                input: observation.clone(),
                source: if broadcast {
                    ComponentFusionSource::TokenEmbedding {
                        normalization: None,
                        weight: "embed.weight".into(),
                        shared_weight: "embed.weight".into(),
                        parameter_group: "parameters:embed".into(),
                        scale: ComponentScalar::new(1.0),
                    }
                } else {
                    ComponentFusionSource::Observation {
                        path: observation.clone(),
                    }
                },
                normalization: norm(c, Some(format!("{parameters}.{norm_field}.weight"))),
                normalized: format!("{observation}.normalized.effective"),
                weight: weight.clone(),
                shared_weight: weight,
                parameter_group: format!("parameters:{parameters}.{projection_field}"),
                bias: None,
                projection_input: format!("{observation}.projection_input"),
                output: format!("{observation}.projected"),
                effective_output: format!("{observation}.projected.effective"),
                expansion: if broadcast {
                    ComponentFusionExpansion::BroadcastAxis {
                        axis: 2,
                        name: "stream".into(),
                        extent: c.hc_mult as usize,
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
            stream_axes(c),
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
            query_heads: Some(c.num_attention_heads as usize),
            key_value_heads: Some(1),
            key_head_dimension: Some(c.head_dim as usize),
            value_head_dimension: Some(c.head_dim as usize),
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
        g.moe(
            &ffn,
            &format!("{parameters}.ffn"),
            moe(
                c.n_routed_experts,
                c.num_experts_per_tok,
                c.n_shared_experts,
                c.norm_topk_prob,
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
                normalization: Some(routed_units::rms(
                    format!("{parameters}.ffn_norm.weight"),
                    c.rms_norm_eps,
                    0.0,
                )),
                residual_scale: None,
            },
            crate::deepseek::v4::expert_bank_spec(c, layer),
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
            Some(&format!("{parameters}.norm")),
            Some(width),
        );
        g.node(
            &head,
            ArchitectureNodeKind::OutputHead,
            Some(&root),
            Some("head"),
            None,
        );
        g.edge(&decoder, &collapse, ArchitectureEdgeKind::Data);
        g.edge(&collapse, &head_norm, ArchitectureEdgeKind::Data);
        g.edge(&head_norm, &head, ArchitectureEdgeKind::Data);
        g.component_observation(
            &collapse,
            format!("{readout}.streams"),
            "Prediction streams consumed by final learned collapse",
            stream_axes(c),
        );
        let mut coefficients_shape = axes(c.hc_mult as usize);
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
        let mut vocabulary = axes(c.vocab_size as usize);
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
                block_transforms: vec![],
                score_writes: vec![],
                stream_residual: Some(ComponentStreamResidual {
                    streams: c.hc_mult as usize,
                    base: ComponentStreamBase::Streams {
                        input: format!("{fusion_output}.effective"),
                    },
                    cycles: stream_cycles(c, layer, &path, &parameters, &attention, &ffn).into(),
                    head: ComponentStreamHead {
                        input: format!("{readout}.streams.effective"),
                        coefficients: format!("{readout}.stream_coefficients"),
                        parameters: coefficients(
                            c,
                            &format!("parameters:{parameters}"),
                            &format!("{parameters}.hc_head"),
                        ),
                    },
                }),
                residual: format!("{readout}.residual"),
                normalized: format!("{readout}.normalized"),
                projection_input: Some(format!("{readout}.projection_input")),
                linear_scores: format!("{readout}.linear"),
                logits: format!("{readout}.linear.effective"),
                normalization: norm(c, Some(format!("{parameters}.norm.weight"))),
                weight: "head.weight".into(),
                bias: None,
                output_transform: ComponentOutputTransform::Identity,
                block_normalizations: vec![],
                other_writes: vec![],
            },
        });
    }
}
