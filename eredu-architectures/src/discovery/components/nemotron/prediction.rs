//! Each prediction depth owns an ordered physical schedule and fused residual base.
use super::*;
use crate::discovery::routed_components::{self as routed_units, Site as RoutedUnitSite};
use crate::nemotron_h::{declarations::TargetOperatorParameters as P, LayerPolicy};

pub(in crate::discovery) fn declare(
    g: &mut Builder,
    declaration: &crate::nemotron_h::Declaration,
    target_output: &str,
) {
    let c = declaration.args();
    let statics = &declaration.statics;
    let embedding_weight = &statics.embedding_spec().weight;
    let output_weight = statics
        .head_spec()
        .map(|p| &p.weight)
        .unwrap_or(embedding_weight);
    let depth_count = c.num_nextn_predict_layers as usize;
    let pattern = declaration.predictions.len() / depth_count;
    g.node(
        "prediction",
        ArchitectureNodeKind::Prediction,
        None,
        None,
        None,
    );
    g.edge(target_output, "prediction", ArchitectureEdgeKind::Data);
    for depth in 0..depth_count {
        let first_parameters = &declaration.predictions[depth * pattern];
        let last_parameters = &declaration.predictions[(depth + 1) * pattern - 1];
        let first = format!("model.mtp.layers.{}", first_parameters.physical);
        let last = format!("model.mtp.layers.{}", last_parameters.physical);
        let fusion_spec = first_parameters
            .fusion
            .as_ref()
            .expect("first prediction unit fusion");
        let embedding_norm_spec = first_parameters
            .embedding_norm
            .as_ref()
            .expect("first prediction embedding norm");
        let hidden_norm_spec = first_parameters
            .hidden_norm
            .as_ref()
            .expect("first prediction hidden norm");
        let final_norm_spec = last_parameters
            .final_norm
            .as_ref()
            .expect("last prediction final norm");
        let embedding_normalization = super::super::declared_normalization(embedding_norm_spec);
        let hidden_normalization = super::super::declared_normalization(hidden_norm_spec);
        let final_normalization = super::super::declared_normalization(final_norm_spec);
        let prefix = |normalization: &ComponentNormalization| {
            normalization
                .gain
                .as_ref()
                .expect("declared prediction normalization")
                .trim_end_matches(".weight")
                .to_owned()
        };
        let embedding_prefix = embedding_weight.id.as_str().trim_end_matches(".weight");
        let fusion_prefix = fusion_spec.weight.id.as_str().trim_end_matches(".weight");
        let width = fusion_spec.output as usize;
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
                embedding_prefix.to_owned(),
            ),
            (
                &embedding_norm,
                ArchitectureNodeKind::Normalization,
                prefix(&embedding_normalization),
            ),
            (
                &hidden_norm,
                ArchitectureNodeKind::Normalization,
                prefix(&hidden_normalization),
            ),
            (
                &fusion,
                ArchitectureNodeKind::Projector,
                fusion_prefix.to_owned(),
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
            let unit = &declaration.predictions[physical];
            let policy = unit.policy;
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
            match (&unit.block.operator, policy) {
                (P::Attention(spec), LayerPolicy::SelfAttention(policy)) => {
                    let geometry = spec.projection.geometry;
                    let mut attributes = attention(
                        geometry.query_heads(),
                        geometry.key_value_heads(),
                        geometry.key_dimensions(),
                        policy,
                    );
                    attributes.positional_encoding = Some(PositionalEncoding::None);
                    g.get_mut(&operator).attention = Some(attributes);
                    super::super::nemotron_unit(g, &unit.block, layer, &path, &operator);
                }
                (P::Sparse(spec), LayerPolicy::SparseMoe) => {
                    super::super::nemotron_other_unit(g, width, &path, &operator, "feed_forward");
                    let mut attributes =
                        declared_routing(spec.selector.selection(), c.n_shared_experts)
                            .expect("admitted prediction routing equation");
                    attributes.shared_expert_width = Some(spec.shared.network.up.output as usize);
                    g.moe(
                        &operator,
                        &parameters,
                        attributes,
                        Some(&format!("{path}.routing")),
                        width,
                    );
                    super::shared_units(
                        g,
                        spec,
                        &unit.block.norm,
                        layer,
                        &path,
                        &operator,
                        &parameters,
                    );
                    routed_units::relu2(
                        g,
                        RoutedUnitSite {
                            node: &operator,
                            layer,
                            routing: &format!("{path}.routing"),
                            input: Some(format!("{path}.feed_forward.input")),
                            normalization: Some(super::super::declared_normalization(
                                &unit.block.norm,
                            )),
                            residual_scale: Some(ComponentScalar::new(1.0)),
                        },
                        Ok(spec.experts.clone()),
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
        let output_parameter = output_weight
            .id
            .as_str()
            .strip_suffix(".weight")
            .expect("declared vocabulary weight");
        g.node(
            &head_norm,
            ArchitectureNodeKind::Normalization,
            Some(&root),
            Some(&prefix(&final_normalization)),
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
        let mut vocabulary = axes(statics.embedding_spec().vocabulary as usize);
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
            static_parameter_roles: if statics.head_spec().is_none() {
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
                            weight: embedding_weight.id.as_str().into(),
                            shared_weight: embedding_weight.id.as_str().into(),
                            parameter_group: format!("parameters:{embedding_prefix}"),
                            scale: ComponentScalar::new(1.0),
                        },
                        normalization: embedding_normalization,
                        output: format!("{first}.prediction.embedding.normalized.effective"),
                        columns: 0..embedding_norm_spec.dimensions as usize,
                    },
                    ComponentFusionInput {
                        source: ComponentFusionSource::Observation {
                            path: format!("{first}.prediction.hidden"),
                        },
                        normalization: hidden_normalization,
                        output: format!("{first}.prediction.hidden.normalized.effective"),
                        columns: embedding_norm_spec.dimensions as usize
                            ..fusion_spec.input as usize,
                    },
                ],
                weight: fusion_spec.weight.id.as_str().into(),
                shared_weight: fusion_spec
                    .weight
                    .alias_of
                    .as_ref()
                    .unwrap_or(&fusion_spec.weight.id)
                    .as_str()
                    .into(),
                parameter_group: format!("parameters:{fusion_prefix}"),
                bias: fusion_spec.bias.as_ref().map(|bias| bias.id.to_string()),
                projection_input: format!("{first}.prediction.fusion.input"),
                output: format!("{first}.prediction.fusion.output"),
                effective_output: format!("{first}.prediction.fusion.output.effective"),
            },
            readout: ComponentReadoutEquation {
                residual_cycles: vec![],
                block_transforms: vec![],
                score_writes: vec![],
                stream_residual: None,
                residual: format!("{last}.prediction.readout.residual"),
                normalized: format!("{last}.prediction.readout.normalized"),
                projection_input: Some(format!("{last}.prediction.readout.projection_input")),
                linear_scores: format!("{last}.prediction.readout.linear"),
                logits: format!("{last}.prediction.readout.linear.effective"),
                normalization: final_normalization,
                weight: output_weight.id.as_str().into(),
                bias: None,
                output_transform: ComponentOutputTransform::Identity,
                block_normalizations: vec![],
                other_writes,
            },
        });
    }
}
