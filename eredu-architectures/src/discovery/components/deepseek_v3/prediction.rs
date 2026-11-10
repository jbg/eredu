//! Prediction invocations have a fused residual base and a separate score head.
use super::*;

pub(in crate::discovery) fn declare(g: &mut Builder, c: &V3Args, target_output: &str) {
    let statics = &c.parameters().statics;
    let embedding_slot = &statics.embedding_spec().weight;
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
        let path = format!("model.layers.{layer}");
        let spec = crate::deepseek::v3_specs::prediction_spec(c, &path)
            .expect("admitted V3 prediction declaration");
        let width = spec.fusion.output as usize;
        let embedding_width = spec.embedding_norm.dimensions as usize;
        let embedding_prefix = declared_weight_prefix(embedding_slot);
        let fusion_prefix = declared_weight_prefix(&spec.fusion.weight);
        let root = format!("prediction.layers.{depth}");
        let block = format!("{root}.decoder");
        let embedding = format!("{root}.embedding");
        let embedding_norm = format!("{root}.embedding.norm");
        let hidden_norm = format!("{root}.hidden.norm");
        let fusion = format!("{root}.fusion");
        let head_norm = format!("{root}.head.norm");
        let head = format!("{root}.head");
        g.node(
            &root,
            ArchitectureNodeKind::Prediction,
            Some("prediction"),
            Some(&path),
            Some(width),
        );
        g.node(
            &embedding,
            ArchitectureNodeKind::Embedding,
            Some(&root),
            Some(embedding_prefix),
            Some(width),
        );
        g.node(
            &embedding_norm,
            ArchitectureNodeKind::Normalization,
            Some(&root),
            declared_normalization_prefix(&spec.embedding_norm),
            Some(width),
        );
        g.node(
            &hidden_norm,
            ArchitectureNodeKind::Normalization,
            Some(&root),
            declared_normalization_prefix(&spec.hidden_norm),
            Some(width),
        );
        g.node(
            &fusion,
            ArchitectureNodeKind::Projector,
            Some(&root),
            Some(fusion_prefix),
            Some(width),
        );
        g.node(
            &block,
            ArchitectureNodeKind::DecoderBlock,
            Some(&root),
            Some(&path),
            Some(width),
        );
        g.get_mut(&block).layer_index = Some(layer);
        g.descriptor.layer_groups.push(ArchitectureLayerGroup {
            id: format!("mtp.{depth}"),
            label: format!("Prediction depth {depth}"),
            physical_layer_count: 1,
            passes: vec![ArchitectureExecutionPass {
                index: 0,
                executions: vec![ArchitectureLayerExecution {
                    node_id: block.clone(),
                    physical_layer_index: 0,
                }],
            }],
            weight_sharing: LayerWeightSharing::None,
        });
        g.edge(&embedding, &embedding_norm, ArchitectureEdgeKind::Data);
        // Prefill/replay can supply target hidden state to every depth; proposal
        // execution can supply the previous depth. The root's actual input
        // observation is authoritative for this invocation.
        g.edge(&root, &hidden_norm, ArchitectureEdgeKind::Data);
        g.edge(&embedding_norm, &fusion, ArchitectureEdgeKind::Data);
        g.edge(&hidden_norm, &fusion, ArchitectureEdgeKind::Data);
        g.edge(&fusion, &block, ArchitectureEdgeKind::Data);

        // Runtime traversal owns the outer unit boundaries; these are not the
        // decoder's fused input or its pre-head residual.
        for (point, meaning) in [
            (
                format!("{path}.input"),
                "Hidden input supplied to this prediction invocation",
            ),
            (
                format!("{path}.output"),
                "Prediction hidden output after head computation",
            ),
            (
                format!("mtp.{depth}.output"),
                "Prediction hidden output intervention after head computation",
            ),
        ] {
            g.observation(
                &root,
                point,
                meaning,
                ObservationDtype::Floating,
                Some(axes(width)),
                false,
            );
        }
        g.read_only_observation(
            &root,
            format!("mtp.{depth}.capture"),
            "Actual hidden input before prediction normalization and fusion",
            axes(width),
        );
        for (node, suffix, meaning) in [
            (
                &embedding_norm,
                "embedding.normalized",
                "Normalized current-token embedding consumed by fusion",
            ),
            (
                &hidden_norm,
                "hidden.normalized",
                "Normalized supplied hidden state consumed by fusion",
            ),
            (
                &fusion,
                "fusion.output",
                "Linear fusion output forming this prediction's residual base",
            ),
        ] {
            g.component_observation(
                node,
                format!("{path}.prediction.{suffix}"),
                meaning,
                axes(width),
            );
        }

        let component_start = g.descriptor.components.len();
        let routed_start = g.descriptor.routed_components.len();
        let output = crate::discovery::families::deepseek_v3_unit(g, c, layer, &path, &block);
        let components = g.descriptor.components.split_off(component_start);
        let routed_components = g.descriptor.routed_components.split_off(routed_start);
        g.component_observation(
            &output,
            format!("{path}.prediction.readout.residual"),
            "Prediction residual before head normalization",
            axes(width),
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
            Some(declared_weight_prefix(&spec.output.weight)),
            None,
        );
        g.edge(&output, &head_norm, ArchitectureEdgeKind::Data);
        g.edge(&head_norm, &head, ArchitectureEdgeKind::Data);
        g.component_observation(
            &head_norm,
            format!("{path}.prediction.readout.normalized"),
            "Normalized prediction residual consumed by its own head",
            axes(width),
        );
        let mut vocabulary = axes(spec.output.output as usize);
        vocabulary[2].name = "vocabulary".into();
        g.get_mut(&head).output_axes = Some(vocabulary.clone());
        g.component_observation(
            &head,
            format!("{path}.prediction.readout.linear"),
            "Prediction vocabulary scores before proposal sampling",
            vocabulary,
        );

        let embedding_weight = embedding_slot.id.as_str().to_owned();
        let fusion_weight = spec.fusion.weight.id.as_str().to_owned();
        g.descriptor.component_scopes.push(ComponentExecutionScope {
            id: format!("prediction.{depth}"),
            node_id: root,
            execution_groups: vec![format!("mtp.{depth}")],
            static_parameter_roles: vec!["embedding".into()],
            kind: ComponentExecutionScopeKind::Prediction { depth },
            components,
            routed_components,
            residual_base: ComponentResidualBase::LinearFusion {
                inputs: vec![
                    ComponentFusionInput {
                        source: ComponentFusionSource::TokenEmbedding {
                            normalization: None,
                            weight: embedding_weight.clone(),
                            shared_weight: declared_shared_weight(embedding_slot).into(),
                            parameter_group: format!("parameters:{embedding_prefix}"),
                            scale: ComponentScalar::new(1.0),
                        },
                        normalization: declared_normalization(&spec.embedding_norm),
                        output: format!("{path}.prediction.embedding.normalized.effective"),
                        columns: 0..embedding_width,
                    },
                    ComponentFusionInput {
                        source: ComponentFusionSource::Observation {
                            path: format!("mtp.{depth}.capture"),
                        },
                        normalization: declared_normalization(&spec.hidden_norm),
                        output: format!("{path}.prediction.hidden.normalized.effective"),
                        columns: embedding_width..spec.fusion.input as usize,
                    },
                ],
                weight: fusion_weight.clone(),
                shared_weight: declared_shared_weight(&spec.fusion.weight).into(),
                parameter_group: format!("parameters:{fusion_prefix}"),
                bias: spec.fusion.bias.as_ref().map(|bias| bias.id.to_string()),
                projection_input: format!("{path}.prediction.fusion.input"),
                output: format!("{path}.prediction.fusion.output"),
                effective_output: format!("{path}.prediction.fusion.output.effective"),
            },
            readout: ComponentReadoutEquation {
                residual_cycles: vec![],
                block_transforms: vec![],
                score_writes: vec![],
                stream_residual: None,
                residual: format!("{path}.prediction.readout.residual"),
                normalized: format!("{path}.prediction.readout.normalized"),
                projection_input: Some(format!("{path}.prediction.readout.projection_input")),
                linear_scores: format!("{path}.prediction.readout.linear"),
                logits: format!("{path}.prediction.readout.linear.effective"),
                normalization: declared_normalization(&spec.output_norm),
                weight: spec.output.weight.id.as_str().into(),
                bias: spec.output.bias.as_ref().map(|bias| bias.id.to_string()),
                output_transform: ComponentOutputTransform::Identity,
                block_normalizations: vec![],
                other_writes: vec![ComponentResidualWrite {
                    input: Some(format!("{path}.feed_forward.input")),
                    layer_index: layer,
                    node_id: format!("{block}.feed_forward"),
                    output: format!("{path}.feed_forward.contribution"),
                    effective_output: format!("{path}.feed_forward.contribution.effective"),
                    residual_scale: ComponentScalar::new(1.0),
                }],
            },
        });
    }
}
