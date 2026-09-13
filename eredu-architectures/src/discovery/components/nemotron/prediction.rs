//! Each prediction depth owns an ordered physical schedule and fused residual base.
use super::*;
use crate::discovery::routed_components::{self as routed_units, Site as RoutedUnitSite};
use crate::nemotron_h::{LayerPolicy, ModelArgs};

fn normalization(c: &ModelArgs, gain: String) -> ComponentNormalization {
    ComponentNormalization {
        kind: ComponentNormalizationKind::Rms,
        epsilon: ComponentScalar::new(c.layer_norm_epsilon),
        gain: Some(gain),
        gain_offset: ComponentScalar::new(0.0),
        bias: None,
        groups: 1,
    }
}

pub(in crate::discovery) fn declare(g: &mut Builder, c: &ModelArgs, target_output: &str) {
    let policies = c
        .mtp_policies()
        .expect("admitted Nemotron prediction schedule");
    let depth_count = c.num_nextn_predict_layers as usize;
    let pattern = policies.len() / depth_count;
    let width = c.hidden_size as usize;
    g.node(
        "prediction",
        ArchitectureNodeKind::Prediction,
        None,
        None,
        None,
    );
    g.edge(target_output, "prediction", ArchitectureEdgeKind::Data);
    for depth in 0..depth_count {
        let first = format!("model.mtp.layers.{}", depth * pattern);
        let last = format!("model.mtp.layers.{}", (depth + 1) * pattern - 1);
        let root = format!("prediction.layers.{depth}");
        let embedding = format!("{root}.embedding");
        let embedding_norm = format!("{embedding}.norm");
        let hidden_norm = format!("{root}.hidden.norm");
        let fusion = format!("{root}.fusion");
        g.node(
            &root,
            ArchitectureNodeKind::Prediction,
            Some("prediction"),
            Some(&first),
            Some(width),
        );
        for (node, kind, parameter) in [
            (
                &embedding,
                ArchitectureNodeKind::Embedding,
                "model.embeddings".to_owned(),
            ),
            (
                &embedding_norm,
                ArchitectureNodeKind::Normalization,
                format!("{first}.enorm"),
            ),
            (
                &hidden_norm,
                ArchitectureNodeKind::Normalization,
                format!("{first}.hnorm"),
            ),
            (
                &fusion,
                ArchitectureNodeKind::Projector,
                format!("{first}.eh_proj"),
            ),
        ] {
            g.node(node, kind, Some(&root), Some(&parameter), Some(width));
        }
        g.edge(&embedding, &embedding_norm, ArchitectureEdgeKind::Data);
        g.edge(&root, &hidden_norm, ArchitectureEdgeKind::Data);
        g.edge(&embedding_norm, &fusion, ArchitectureEdgeKind::Data);
        g.edge(&hidden_norm, &fusion, ArchitectureEdgeKind::Data);
        for (node, suffix) in [(&root, "hidden"), (&embedding, "embedding")] {
            g.read_only_observation(
                node,
                format!("{first}.prediction.{suffix}"),
                "Actual prediction input before normalization",
                axes(width),
            );
        }
        for (node, suffix, meaning) in [
            (
                &embedding_norm,
                "embedding.normalized",
                "Normalized embedding consumed by fusion",
            ),
            (
                &hidden_norm,
                "hidden.normalized",
                "Normalized supplied hidden state consumed by fusion",
            ),
            (&fusion, "fusion.output", "Prediction fused residual base"),
        ] {
            g.component_observation(
                node,
                format!("{first}.prediction.{suffix}"),
                meaning,
                axes(width),
            );
        }
        let component_start = g.descriptor.components.len();
        let routed_start = g.descriptor.routed_components.len();
        let mut previous = fusion;
        let mut executions = Vec::with_capacity(pattern);
        let mut other_writes = Vec::new();
        for relative in 0..pattern {
            let physical = depth * pattern + relative;
            let layer = c.num_hidden_layers as usize + physical;
            let path = format!("model.mtp.layers.{physical}");
            let block = format!("{root}.units.{relative}");
            let parameters = format!("{path}.mixer");
            g.node(
                &block,
                ArchitectureNodeKind::DecoderBlock,
                Some(&root),
                Some(&path),
                Some(width),
            );
            g.get_mut(&block).layer_index = Some(layer);
            g.edge(&previous, &block, ArchitectureEdgeKind::Data);
            executions.push(ArchitectureLayerExecution {
                node_id: block.clone(),
                physical_layer_index: relative,
            });
            let policy = policies[physical];
            let kind = match policy {
                LayerPolicy::SelfAttention(_) => ArchitectureNodeKind::Attention,
                LayerPolicy::SparseMoe => ArchitectureNodeKind::MixtureOfExperts,
                _ => unreachable!("admitted prediction policies are attention or sparse MoE"),
            };
            let (operator, output) = g.sublayer(
                &block,
                &block,
                "operator",
                &format!("{path}.norm"),
                &parameters,
                width,
                kind,
            );
            match policy {
                LayerPolicy::SelfAttention(policy) => {
                    let mut attributes = attention(
                        c.num_attention_heads,
                        c.num_key_value_heads,
                        c.head_dim,
                        policy,
                    );
                    attributes.positional_encoding = Some(PositionalEncoding::None);
                    g.get_mut(&operator).attention = Some(attributes);
                    super::super::nemotron_unit_at(
                        g,
                        c,
                        layer,
                        &path,
                        &operator,
                        true,
                        &parameters,
                    );
                }
                LayerPolicy::SparseMoe => {
                    super::super::nemotron_other_unit(g, width, &path, &operator, "feed_forward");
                    let mut attributes = moe(
                        c.n_routed_experts,
                        c.num_experts_per_tok,
                        c.n_shared_experts,
                        c.norm_topk_prob,
                        RoutingScoreTransform::Sigmoid,
                    );
                    attributes.shared_expert_width =
                        Some(c.moe_shared_expert_intermediate_size as usize);
                    g.moe(
                        &operator,
                        &parameters,
                        attributes,
                        Some(&format!("{path}.routing")),
                        width,
                    );
                    super::shared_units_at(g, c, layer, &path, &operator, &parameters);
                    routed_units::relu2(
                        g,
                        RoutedUnitSite {
                            node: &operator,
                            layer,
                            routing: &format!("{path}.routing"),
                            input: Some(format!("{path}.feed_forward.input")),
                            normalization: Some(routed_units::rms(
                                format!("{path}.norm.weight"),
                                c.layer_norm_epsilon,
                                0.0,
                            )),
                            residual_scale: Some(ComponentScalar::new(1.0)),
                        },
                        crate::nemotron_h::expert_bank_spec(c, layer),
                    );
                    other_writes.push(ComponentResidualWrite {
                        input: Some(format!("{path}.feed_forward.input")),
                        layer_index: layer,
                        node_id: operator,
                        output: format!("{path}.feed_forward.output"),
                        effective_output: format!("{path}.feed_forward.output.effective"),
                        residual_scale: ComponentScalar::new(1.0),
                    });
                }
                _ => unreachable!(),
            }
            previous = output;
        }
        g.descriptor.layer_groups.push(ArchitectureLayerGroup {
            id: format!("mtp.{depth}"),
            label: format!("Prediction depth {depth}"),
            physical_layer_count: pattern,
            passes: vec![ArchitectureExecutionPass {
                index: 0,
                executions,
            }],
            weight_sharing: LayerWeightSharing::None,
        });
        let components = g.descriptor.components.split_off(component_start);
        let routed_components = g.descriptor.routed_components.split_off(routed_start);
        let head_norm = format!("{root}.head.norm");
        let head = format!("{root}.head");
        let output_parameter = if c.tie_word_embeddings {
            "model.embeddings"
        } else {
            "lm_head"
        };
        g.node(
            &head_norm,
            ArchitectureNodeKind::Normalization,
            Some(&root),
            Some(&format!("{last}.final_layernorm")),
            Some(width),
        );
        g.node(
            &head,
            ArchitectureNodeKind::OutputHead,
            Some(&root),
            Some(output_parameter),
            None,
        );
        g.edge(&previous, &head_norm, ArchitectureEdgeKind::Data);
        g.edge(&head_norm, &head, ArchitectureEdgeKind::Data);
        g.component_observation(
            &previous,
            format!("{last}.prediction.readout.residual"),
            "Prediction residual before head normalization",
            axes(width),
        );
        g.component_observation(
            &head_norm,
            format!("{last}.prediction.readout.normalized"),
            "Normalized prediction residual consumed by the shared head",
            axes(width),
        );
        let mut vocabulary = axes(c.vocab_size as usize);
        vocabulary[2].name = "vocabulary".into();
        g.get_mut(&head).output_axes = Some(vocabulary.clone());
        g.component_observation(
            &head,
            format!("{last}.prediction.readout.linear"),
            "Prediction vocabulary scores before sampling",
            vocabulary,
        );
        g.descriptor.component_scopes.push(ComponentExecutionScope {
            id: format!("prediction.{depth}"),
            node_id: root,
            execution_groups: vec![format!("mtp.{depth}")],
            static_parameter_roles: if c.tie_word_embeddings {
                vec!["embedding".into()]
            } else {
                vec!["embedding".into(), "output".into()]
            },
            kind: ComponentExecutionScopeKind::Prediction { depth },
            components,
            routed_components,
            residual_base: ComponentResidualBase::LinearFusion {
                inputs: vec![
                    ComponentFusionInput {
                        source: ComponentFusionSource::TokenEmbedding {
                            normalization: None,
                            weight: "model.embeddings.weight".into(),
                            shared_weight: "model.embeddings.weight".into(),
                            parameter_group: "parameters:model.embeddings".into(),
                            scale: ComponentScalar::new(1.0),
                        },
                        normalization: normalization(c, format!("{first}.enorm.weight")),
                        output: format!("{first}.prediction.embedding.normalized.effective"),
                        columns: 0..width,
                    },
                    ComponentFusionInput {
                        source: ComponentFusionSource::Observation {
                            path: format!("{first}.prediction.hidden"),
                        },
                        normalization: normalization(c, format!("{first}.hnorm.weight")),
                        output: format!("{first}.prediction.hidden.normalized.effective"),
                        columns: width..width * 2,
                    },
                ],
                weight: format!("{first}.eh_proj.weight"),
                shared_weight: format!("{first}.eh_proj.weight"),
                parameter_group: format!("parameters:{first}.eh_proj"),
                bias: None,
                projection_input: format!("{first}.prediction.fusion.input"),
                output: format!("{first}.prediction.fusion.output"),
                effective_output: format!("{first}.prediction.fusion.output.effective"),
            },
            readout: ComponentReadoutEquation {
                block_transforms: vec![],
                score_writes: vec![],
                stream_residual: None,
                residual: format!("{last}.prediction.readout.residual"),
                normalized: format!("{last}.prediction.readout.normalized"),
                projection_input: Some(format!("{last}.prediction.readout.projection_input")),
                linear_scores: format!("{last}.prediction.readout.linear"),
                logits: format!("{last}.prediction.readout.linear.effective"),
                normalization: normalization(c, format!("{last}.final_layernorm.weight")),
                weight: format!("{output_parameter}.weight"),
                bias: None,
                output_transform: ComponentOutputTransform::Identity,
                block_normalizations: vec![],
                other_writes,
            },
        });
    }
}
