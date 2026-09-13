//! Inkling's pinned prediction depths reuse the ordinary causal decoder equations.
use super::*;

pub(in crate::discovery) fn declare(g: &mut Builder, args: &crate::inkling::ModelArgs) {
    let Some(config) = args
        .mtp_config
        .as_ref()
        .filter(|config| config.num_nextn_predict_layers > 0)
    else {
        return;
    };
    let backbone = &args.text_config;
    let width = backbone.hidden_size as usize;
    let vocabulary = backbone.unpadded_vocab_size.unwrap_or(backbone.vocab_size) as usize;
    let norm = |gain: &str| ComponentNormalization {
        kind: ComponentNormalizationKind::Rms,
        epsilon: ComponentScalar::new(backbone.rms_norm_eps),
        gain: Some(gain.into()),
        gain_offset: ComponentScalar::new(0.0),
        bias: None,
        groups: 1,
    };
    for depth in 0..config.num_nextn_predict_layers as usize {
        let attention = if config.local_layer_ids.contains(&depth) {
            backbone
                .layer_schedule
                .iter()
                .find_map(|policy| policy.attention.window())
                .map(|window| AttentionPolicy::Sliding { window })
                .expect("admitted local MTP policy")
        } else {
            AttentionPolicy::Full
        };
        let text = crate::inkling::mtp_text_args(backbone, config, attention)
            .expect("admitted MTP geometry");
        let policy = *text.layer_schedule.get(0).expect("one MTP decoder");
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
                "model.embed_tokens".to_owned(),
            ),
            (
                &source_norm,
                ArchitectureNodeKind::Normalization,
                "model.embed_norm".to_owned(),
            ),
            (
                &embedding_norm,
                ArchitectureNodeKind::Normalization,
                format!("{path}.embed_norm"),
            ),
            (
                &hidden_first,
                ArchitectureNodeKind::Normalization,
                format!("{path}.hidden_norm"),
            ),
            (
                &hidden_norm,
                ArchitectureNodeKind::Normalization,
                format!("{path}.hidden_norm"),
            ),
            (
                &fusion,
                ArchitectureNodeKind::Projector,
                format!("{path}.input_proj"),
            ),
            (
                &block,
                ArchitectureNodeKind::DecoderBlock,
                block_path.clone(),
            ),
            (
                &head,
                ArchitectureNodeKind::OutputHead,
                "lm_head".to_owned(),
            ),
        ] {
            g.node(node, kind, Some(&root), Some(&prefix), Some(width));
        }
        g.node(
            &head_norm,
            ArchitectureNodeKind::Normalization,
            Some(&root),
            config
                .chain_hidden_post_norm
                .then_some("model.mtp.chain_norm"),
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
        let local = attention.window().is_some();
        let mut attrs = super::super::super::attention(
            text.query_heads(local),
            text.key_value_heads(local),
            text.attention_head_dim(local),
            attention,
        );
        attrs.positional_encoding = Some(PositionalEncoding::Relative);
        g.get_mut(&attention_node).attention = Some(attrs);
        let start = g.descriptor.components.len();
        decoder_scalars(g, &text, policy, layer, &block, &block_path);
        decoder_transforms(g, &text, policy, &block, &block_path);
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
                    normalization: norm(&format!("{path}.hidden_norm.weight")),
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
        let fusion_weight = format!("{path}.input_proj.weight");
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
                        normalization: norm(&format!("{path}.hidden_norm.weight")),
                        output: format!("{hooks}.hidden.normalized.effective"),
                        columns: 0..width,
                    },
                    ComponentFusionInput {
                        source: ComponentFusionSource::TokenEmbedding {
                            weight: "model.embed_tokens.weight".into(),
                            shared_weight: "model.embed_tokens.weight".into(),
                            parameter_group: "parameters:model.embed_tokens".into(),
                            scale: ComponentScalar::new(1.0),
                            normalization: Some(norm("model.embed_norm.weight")),
                        },
                        normalization: norm(&format!("{path}.embed_norm.weight")),
                        output: format!("{hooks}.embedding.normalized.effective"),
                        columns: width..width * 2,
                    },
                ],
                weight: fusion_weight.clone(),
                shared_weight: fusion_weight,
                parameter_group: format!("parameters:{path}.input_proj"),
                bias: None,
                projection_input: format!("{hooks}.fusion.input"),
                output: format!("{hooks}.fusion.output"),
                effective_output: format!("{hooks}.fusion.output.effective"),
            },
            readout: ComponentReadoutEquation {
                block_transforms: vec![],
                stream_residual: None,
                score_writes: Vec::new(),
                residual: format!("{hooks}.readout.residual"),
                normalized: format!("{hooks}.readout.normalized"),
                projection_input: Some(format!("{hooks}.readout.projection_input")),
                linear_scores: format!("{hooks}.readout.linear"),
                logits: format!("{hooks}.readout.linear.effective"),
                normalization: if config.chain_hidden_post_norm {
                    norm("model.mtp.chain_norm.weight")
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
                weight: "lm_head.weight".into(),
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
