//! Inkling's pinned prediction depths reuse the ordinary causal decoder equations.
use super::*;

pub(in crate::discovery) fn declare(g: &mut Builder, args: &crate::inkling::ModelArgs) {
    if args.prediction().depths.is_empty() {
        return;
    }
    let backbone = &args.text_config;
    let vocabulary = backbone.unpadded_vocab_size.unwrap_or(backbone.vocab_size) as usize;
    let prediction = args.prediction();
    let source_norm_spec = &args.pinned().embedding_norm;
    let statics = &args.pinned().ordinary;
    let embedding_weight = statics.embedding_spec().weight.id.to_string();
    let readout_weight = statics.output_projection_spec().weight.id.to_string();
    let source_gain = super::super::declared_normalization(&source_norm_spec)
        .gain
        .expect("declared embedding normalization");
    let chain_prefix = prediction.chain_norm.as_ref().map(|spec| {
        super::super::declared_normalization(spec)
            .gain
            .expect("declared chain normalization")
            .trim_end_matches(".weight")
            .to_owned()
    });
    for (depth, parameters) in prediction.depths.iter().enumerate() {
        let width = parameters.fusion.input_projection.output as usize;
        let hidden_width = parameters.fusion.hidden_norm.dimensions as usize;
        let hidden_gain = super::super::declared_normalization(&parameters.fusion.hidden_norm)
            .gain
            .expect("declared hidden normalization");
        let embedding_gain =
            super::super::declared_normalization(&parameters.fusion.embedding_norm)
                .gain
                .expect("declared prediction embedding normalization");
        let fusion_prefix = parameters
            .fusion
            .input_projection
            .weight
            .id
            .as_str()
            .trim_end_matches(".weight");
        let attention = parameters.common.attention.policy;
        let policy = crate::inkling::LayerPolicy {
            attention,
            feed_forward: crate::inkling::FeedForwardPolicy::Dense,
        };
        let layer = backbone.num_hidden_layers as usize + depth;
        let root = format!("prediction.layers.{depth}");
        let path = format!("model.mtp.layers.{depth}");
        let hooks = format!("{path}.prediction");
        let block = format!("{root}.decoder");
        let block_path = format!("{path}.transformer_block");
        let embedding = format!("{root}.embedding");
        let source_norm = format!("{embedding}.source_norm");
        let embedding_norm = format!("{embedding}.norm");
        let hidden_first = format!("{root}.hidden.first_norm");
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
        for (node, kind, prefix) in [
            (
                &embedding,
                ArchitectureNodeKind::Embedding,
                embedding_weight.trim_end_matches(".weight").to_owned(),
            ),
            (
                &source_norm,
                ArchitectureNodeKind::Normalization,
                source_gain.trim_end_matches(".weight").to_owned(),
            ),
            (
                &embedding_norm,
                ArchitectureNodeKind::Normalization,
                embedding_gain.trim_end_matches(".weight").to_owned(),
            ),
            (
                &hidden_first,
                ArchitectureNodeKind::Normalization,
                hidden_gain.trim_end_matches(".weight").to_owned(),
            ),
            (
                &hidden_norm,
                ArchitectureNodeKind::Normalization,
                hidden_gain.trim_end_matches(".weight").to_owned(),
            ),
            (
                &fusion,
                ArchitectureNodeKind::Projector,
                fusion_prefix.to_owned(),
            ),
            (
                &block,
                ArchitectureNodeKind::DecoderBlock,
                block_path.clone(),
            ),
            (
                &head,
                ArchitectureNodeKind::OutputHead,
                readout_weight.trim_end_matches(".weight").to_owned(),
            ),
        ] {
            g.node(node, kind, Some(&root), Some(&prefix), Some(width));
        }
        g.node(
            &head_norm,
            ArchitectureNodeKind::Normalization,
            Some(&root),
            chain_prefix.as_deref(),
            Some(width),
        );
        g.get_mut(&block).layer_index = Some(layer);
        g.descriptor.layer_groups.push(ArchitectureLayerGroup {
            id: format!("prediction.{depth}"),
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
        // The prepared parameter graph currently retains these as the static
        // "mtp" role. Do not invent a pageable physical decoder group here.
        for (from, to) in [
            (&embedding, &source_norm),
            (&source_norm, &embedding_norm),
            (&root, &hidden_first),
            (&hidden_first, &hidden_norm),
            (&hidden_norm, &fusion),
            (&embedding_norm, &fusion),
            (&fusion, &block),
            (&block, &head_norm),
            (&head_norm, &head),
        ] {
            g.edge(from, to, ArchitectureEdgeKind::Data);
        }
        let attention_node = format!("{block}.attention");
        let block_norm = format!("{block}.norm");
        let ffn = format!("{block}.feed_forward");
        g.node(
            &block_norm,
            ArchitectureNodeKind::Normalization,
            Some(&block),
            None,
            Some(width),
        );
        g.node(
            &attention_node,
            ArchitectureNodeKind::Attention,
            Some(&block),
            None,
            Some(width),
        );
        g.node(
            &ffn,
            ArchitectureNodeKind::FeedForward,
            Some(&block),
            None,
            Some(width),
        );
        let declared = &parameters.common.attention;
        let mut attrs = super::super::super::attention(
            declared.projection.geometry.query_heads(),
            declared.projection.geometry.key_value_heads(),
            declared.projection.geometry.key_dimensions(),
            attention,
        );
        attrs.positional_encoding = Some(PositionalEncoding::Relative);
        g.get_mut(&attention_node).attention = Some(attrs);
        let start = g.descriptor.components.len();
        decoder_scalars(
            g,
            policy,
            layer,
            &block,
            &block_path,
            &parameters.common,
            Some(&parameters.dense),
        );
        decoder_transforms(
            g,
            policy,
            &block,
            &block_path,
            &parameters.common,
            Some(&parameters.dense),
        );
        let components = g.descriptor.components.split_off(start);
        for (node, suffix, meaning) in [
            (
                &root,
                "hidden",
                "Actual supplied hidden state before its repeated normalization",
            ),
            (
                &source_norm,
                "embedding",
                "Token embedding after target embedding normalization",
            ),
        ] {
            g.read_only_observation(node, format!("{hooks}.{suffix}"), meaning, axes(width));
        }
        for (node, suffix, meaning) in [
            (
                &hidden_first,
                "hidden.first_normalized",
                "First application of the shared hidden norm",
            ),
            (
                &hidden_norm,
                "hidden.normalized",
                "Second application of the same hidden norm",
            ),
            (
                &embedding_norm,
                "embedding.normalized",
                "Prediction-normalized token embedding",
            ),
            (
                &fusion,
                "fusion.output",
                "Hidden-first fusion forming this depth's residual base",
            ),
            (
                &head_norm,
                "readout.residual",
                "Prediction residual before optional chain normalization",
            ),
            (
                &head_norm,
                "readout.normalized",
                "Prediction continuation consumed by the muP-scaled target head",
            ),
        ] {
            g.component_observation(node, format!("{hooks}.{suffix}"), meaning, axes(width));
        }
        g.read_only_observation(
            &head,
            format!("{hooks}.readout.scaled"),
            "Exact muP scaling before projection input arithmetic",
            axes(width),
        );
        let mut scores = axes(vocabulary);
        scores[2].name = "vocabulary".into();
        g.get_mut(&head).output_axes = Some(scores.clone());
        g.component_observation(
            &head,
            format!("{hooks}.readout.linear"),
            "Unpadded tentative prediction scores",
            scores,
        );
        g.descriptor
            .component_transforms
            .push(ComponentTensorTransform {
                id: hidden_first.clone(),
                node_id: hidden_first,
                input: format!("{hooks}.hidden"),
                output: format!("{hooks}.hidden.first_normalized"),
                effective_output: Some(format!("{hooks}.hidden.first_normalized.effective")),
                equation: ComponentTensorTransformEquation::Normalization {
                    normalization: super::super::declared_normalization(
                        &parameters.fusion.hidden_norm,
                    ),
                },
            });
        g.descriptor
            .component_transforms
            .push(ComponentTensorTransform {
                id: format!("{head}.scale"),
                node_id: head,
                input: format!("{hooks}.readout.normalized.effective"),
                output: format!("{hooks}.readout.scaled"),
                effective_output: None,
                equation: ComponentTensorTransformEquation::ConstantScale {
                    scale: ComponentScalar::new(1.0 / backbone.logits_mup_width_multiplier),
                },
            });
        let fusion_weight = parameters.fusion.input_projection.weight.id.to_string();
        g.descriptor.component_scopes.push(ComponentExecutionScope {
            id: format!("prediction.{depth}"),
            node_id: root,
            execution_groups: Vec::new(),
            static_parameter_roles: vec![
                "embedding".into(),
                "embedding_norm".into(),
                "mtp".into(),
                "output".into(),
            ],
            kind: ComponentExecutionScopeKind::Prediction { depth },
            components,
            routed_components: Vec::new(),
            residual_base: ComponentResidualBase::LinearFusion {
                inputs: vec![
                    ComponentFusionInput {
                        source: ComponentFusionSource::Observation {
                            path: format!("{hooks}.hidden.first_normalized.effective"),
                        },
                        normalization: super::super::declared_normalization(
                            &parameters.fusion.hidden_norm,
                        ),
                        output: format!("{hooks}.hidden.normalized.effective"),
                        columns: 0..hidden_width,
                    },
                    ComponentFusionInput {
                        source: ComponentFusionSource::TokenEmbedding {
                            weight: embedding_weight.clone(),
                            shared_weight: embedding_weight.clone(),
                            parameter_group: format!(
                                "parameters:{}",
                                embedding_weight.trim_end_matches(".weight")
                            ),
                            scale: ComponentScalar::new(1.0),
                            normalization: Some(super::super::declared_normalization(
                                &source_norm_spec,
                            )),
                        },
                        normalization: super::super::declared_normalization(
                            &parameters.fusion.embedding_norm,
                        ),
                        output: format!("{hooks}.embedding.normalized.effective"),
                        columns: hidden_width..parameters.fusion.input_projection.input as usize,
                    },
                ],
                weight: fusion_weight.clone(),
                shared_weight: fusion_weight,
                parameter_group: format!("parameters:{fusion_prefix}"),
                bias: parameters
                    .fusion
                    .input_projection
                    .bias
                    .as_ref()
                    .map(|bias| bias.id.to_string()),
                projection_input: format!("{hooks}.fusion.input"),
                output: format!("{hooks}.fusion.output"),
                effective_output: format!("{hooks}.fusion.output.effective"),
            },
            readout: ComponentReadoutEquation {
                residual_cycles: vec![],
                block_transforms: vec![],
                stream_residual: None,
                score_writes: Vec::new(),
                residual: format!("{hooks}.readout.residual"),
                normalized: format!("{hooks}.readout.normalized"),
                projection_input: Some(format!("{hooks}.readout.projection_input")),
                linear_scores: format!("{hooks}.readout.linear"),
                logits: format!("{hooks}.readout.linear.effective"),
                normalization: if let Some(norm) = &prediction.chain_norm {
                    super::super::declared_normalization(norm)
                } else {
                    ComponentNormalization {
                        kind: ComponentNormalizationKind::Identity,
                        epsilon: ComponentScalar::new(0.0),
                        gain: None,
                        gain_offset: ComponentScalar::new(0.0),
                        bias: None,
                        groups: 1,
                    }
                },
                weight: readout_weight.clone(),
                bias: None,
                output_transform: ComponentOutputTransform::Identity,
                block_normalizations: Vec::new(),
                other_writes: ["attention", "feed_forward"]
                    .into_iter()
                    .map(|branch| ComponentResidualWrite {
                        input: Some(format!("{block_path}.{branch}.input")),
                        layer_index: layer,
                        node_id: format!("{block}.{branch}"),
                        output: format!("{block_path}.{branch}.contribution"),
                        effective_output: format!("{block_path}.{branch}.contribution.effective"),
                        residual_scale: ComponentScalar::new(1.0),
                    })
                    .collect(),
            },
        });
    }
}
