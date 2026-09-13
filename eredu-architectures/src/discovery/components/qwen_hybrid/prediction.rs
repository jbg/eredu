//! Prediction invocations have a fused residual base and a separate score head.
use super::*;

pub(in crate::discovery) fn declare(g: &mut Builder, c: &HybridConfig, target_output: &str) {
    let width = c.hidden_size as usize;
    g.node(
        "prediction",
        ArchitectureNodeKind::Prediction,
        None,
        None,
        None,
    );
    g.edge(target_output, "prediction", ArchitectureEdgeKind::Data);
    for depth in 0..c.mtp_num_hidden_layers as usize {
        let layer = c.num_hidden_layers as usize + depth;
        let path = format!("mtp.layers.{depth}");
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
            Some("model.embed_tokens"),
            Some(width),
        );
        g.node(
            &embedding_norm,
            ArchitectureNodeKind::Normalization,
            Some(&root),
            Some("mtp.pre_fc_norm_embedding"),
            Some(width),
        );
        g.node(
            &hidden_norm,
            ArchitectureNodeKind::Normalization,
            Some(&root),
            Some("mtp.pre_fc_norm_hidden"),
            Some(width),
        );
        g.node(
            &fusion,
            ArchitectureNodeKind::Projector,
            Some(&root),
            Some("mtp.fc"),
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

        for (node, suffix) in [(&root, "hidden"), (&embedding, "embedding")] {
            g.read_only_observation(
                node,
                format!("{path}.prediction.{suffix}"),
                "Actual prediction input before normalization",
                axes(width),
            );
        }
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
        let (mixer, join) = g.sublayer(
            &block,
            &block,
            "attention",
            &format!("{path}.input_layernorm"),
            &format!("{path}.self_attn"),
            width,
            ArchitectureNodeKind::Attention,
        );
        g.get_mut(&mixer).attention = Some(attention(
            c.num_attention_heads,
            c.num_key_value_heads,
            c.head_dim,
            eredu_core::AttentionPolicy::Full,
        ));
        let (ffn, output) =
            crate::discovery::families::qwen_hybrid_ffn(g, c, layer, &block, &path, &join);
        unit_with_attention(g, c, layer, &path, &mixer, &ffn, true);
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
            Some("mtp.norm"),
            Some(width),
        );
        g.node(
            &head,
            ArchitectureNodeKind::OutputHead,
            Some(&root),
            Some(if c.tie_word_embeddings {
                "model.embed_tokens"
            } else {
                "lm_head"
            }),
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
        let mut vocabulary = axes(c.vocab_size as usize);
        vocabulary[2].name = "vocabulary".into();
        g.get_mut(&head).output_axes = Some(vocabulary.clone());
        g.component_observation(
            &head,
            format!("{path}.prediction.readout.linear"),
            "Prediction vocabulary scores before proposal sampling",
            vocabulary,
        );

        let embedding_weight = "model.embed_tokens.weight".to_owned();
        let fusion_weight = "mtp.fc.weight".to_owned();
        g.descriptor.component_scopes.push(ComponentExecutionScope {
            id: format!("prediction.{depth}"),
            node_id: root,
            execution_groups: vec![format!("mtp.{depth}")],
            static_parameter_roles: if c.tie_word_embeddings {
                vec!["embedding".into(), "mtp".into()]
            } else {
                vec!["embedding".into(), "mtp".into(), "output".into()]
            },
            kind: ComponentExecutionScopeKind::Prediction { depth },
            components,
            routed_components,
            residual_base: ComponentResidualBase::LinearFusion {
                inputs: vec![
                    ComponentFusionInput {
                        source: ComponentFusionSource::TokenEmbedding {
                            normalization: None,
                            weight: embedding_weight.clone(),
                            shared_weight: embedding_weight,
                            parameter_group: "parameters:model.embed_tokens".into(),
                            scale: ComponentScalar::new(1.0),
                        },
                        normalization: normalization(c, "mtp.pre_fc_norm_embedding.weight".into()),
                        output: format!("{path}.prediction.embedding.normalized.effective"),
                        columns: 0..width,
                    },
                    ComponentFusionInput {
                        source: ComponentFusionSource::Observation {
                            path: format!("{path}.prediction.hidden"),
                        },
                        normalization: normalization(c, "mtp.pre_fc_norm_hidden.weight".into()),
                        output: format!("{path}.prediction.hidden.normalized.effective"),
                        columns: width..width * 2,
                    },
                ],
                weight: fusion_weight.clone(),
                shared_weight: fusion_weight,
                parameter_group: "parameters:mtp.fc".into(),
                bias: None,
                projection_input: format!("{path}.prediction.fusion.input"),
                output: format!("{path}.prediction.fusion.output"),
                effective_output: format!("{path}.prediction.fusion.output.effective"),
            },
            readout: ComponentReadoutEquation {
                block_transforms: vec![],
                score_writes: vec![],
                stream_residual: None,
                residual: format!("{path}.prediction.readout.residual"),
                normalized: format!("{path}.prediction.readout.normalized"),
                projection_input: Some(format!("{path}.prediction.readout.projection_input")),
                linear_scores: format!("{path}.prediction.readout.linear"),
                logits: format!("{path}.prediction.readout.linear.effective"),
                normalization: normalization(c, "mtp.norm.weight".into()),
                weight: if c.tie_word_embeddings {
                    "model.embed_tokens.weight"
                } else {
                    "lm_head.weight"
                }
                .into(),
                bias: None,
                output_transform: ComponentOutputTransform::Identity,
                block_normalizations: vec![],
                other_writes: if c.is_moe() {
                    vec![ComponentResidualWrite {
                        input: Some(format!("{path}.feed_forward.input")),
                        layer_index: layer,
                        node_id: format!("{block}.feed_forward"),
                        output: format!("{path}.feed_forward.output"),
                        effective_output: format!("{path}.feed_forward.output.effective"),
                        residual_scale: ComponentScalar::new(1.0),
                    }]
                } else {
                    vec![]
                },
            },
        });
    }
}
