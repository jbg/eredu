//! Logical execution, parameter reuse and state ownership from retained family specs.
use super::*;
use crate::qwen4_exp::{mtp::PredictionSpec, target::MixerSpec};
use eredu_core::{speculative::*, *};
use std::collections::BTreeMap;

pub(super) struct Graph {
    pub descriptor: ArchitectureDescriptor,
    canonical: BTreeMap<String, String>,
}
impl Graph {
    pub fn new(target: &TargetSpec, prediction: &PredictionSpec) -> Self {
        use ArchitectureEdgeKind as E;
        use ArchitectureNodeKind as K;
        let omissions = DescriptionCompleteness::Partial(vec![
            "Logical target/prediction execution and observation ownership are described; component score/readout equations and media processing are not yet projected".into(),
        ]);
        let mut g = Self {
            descriptor: ArchitectureDescriptor {
                schema_version: ARCHITECTURE_DESCRIPTOR_SCHEMA_VERSION,
                nodes: vec![],
                edges: vec![],
                parameter_groups: vec![],
                layer_groups: vec![],
                observations: ObservationCatalog {
                    schema_version: DISCOVERY_SCHEMA_VERSION,
                    points: vec![],
                    completeness: omissions.clone(),
                },
                components: vec![],
                component_transforms: vec![],
                routed_components: vec![],
                component_readout: None,
                component_scopes: vec![],
                speculative_invocations: vec![],
                completeness: omissions,
            },
            canonical: BTreeMap::new(),
        };
        g.node("target", K::Opaque, None, "Target invocation", None);
        g.invocation("target", SpeculativeCaptureScope::Target);
        g.node(
            "target.tokens",
            K::Processor,
            Some("target"),
            "Original token IDs retained across media replacement",
            None,
        );
        g.node(
            "target.embedding",
            K::Embedding,
            Some("target"),
            "Token embedding and residual expansion",
            None,
        );
        g.parameters("target.embedding", "embedding", "model.embed_tokens");
        g.edge("target.tokens", "target.embedding", E::Data);
        let mut previous = "target.embedding".to_owned();
        let mut executions = vec![];
        for (ordinal, unit) in target.units.iter().enumerate() {
            let id = format!("target.units.{ordinal}");
            let prefix = format!("model.layers.{}", unit.layer());
            match unit {
                UnitSpec::Lexical { spec, .. } => {
                    g.node(
                        &id,
                        K::Mixer,
                        Some("target"),
                        "N-gram gated lexical injection",
                        Some(ordinal),
                    );
                    g.parameters(&id, "injection", &format!("{prefix}.ple"));
                    let table = format!("{id}.table");
                    g.node(
                        &table,
                        K::Embedding,
                        Some(&id),
                        "Logical sharded n-gram row lookup",
                        None,
                    );
                    // One logical namespace, independent of physical shards and row count.
                    g.parameters(
                        &table,
                        "table",
                        &format!("{prefix}.ple.ple_embedding.ngram_embedding"),
                    );
                    g.edge("target.tokens", &table, E::Data);
                    g.edge(&table, &id, E::Data);
                    g.get(&id).mixer = Some(MixerAttributes {
                        mechanism: MixerMechanism::ShortConvolution,
                        recurrent: true,
                        convolution_width: Some(spec.convolution.kernel_size as usize),
                    });
                    g.state(&id, "Token IDs, convolution history and sequence position");
                }
                UnitSpec::Decoder {
                    mixer,
                    feed_forward,
                    ..
                } => {
                    g.node(
                        &id,
                        K::DecoderBlock,
                        Some("target"),
                        "Decoder layer",
                        Some(ordinal),
                    );
                    g.decoder(&id, &prefix, mixer, &feed_forward.feed_forward);
                    executions.push(ArchitectureLayerExecution {
                        node_id: id.clone(),
                        physical_layer_index: unit.layer(),
                    });
                }
            }
            g.edge(&previous, &id, E::Data);
            previous = id;
        }
        g.descriptor.layer_groups.push(ArchitectureLayerGroup {
            id: "target.decoder".into(),
            label:
                "Physical target decoder layers; lexical injections are separate execution units"
                    .into(),
            physical_layer_count: target.config.layers.len(),
            passes: vec![ArchitectureExecutionPass {
                index: 0,
                executions,
            }],
            weight_sharing: LayerWeightSharing::None,
        });
        g.readout(
            "target",
            "model.hyper_connection_mixer",
            &previous,
            target.config.tied_embeddings,
        );
        let mut prediction_executions = vec![];
        for unit in &prediction.units {
            let root = format!("prediction.{}", unit.depth);
            g.node(
                &root,
                K::Prediction,
                None,
                "Embedded prediction invocation",
                Some(unit.depth),
            );
            g.invocation(
                &root,
                SpeculativeCaptureScope::Prediction { depth: unit.depth },
            );
            let tokens = format!("{root}.tokens");
            g.node(
                &tokens,
                K::Processor,
                Some(&root),
                "Teacher-forced or proposed token IDs supplied by the driver",
                None,
            );
            let embedding = format!("{root}.embedding");
            g.node(
                &embedding,
                K::Embedding,
                Some(&root),
                "Shared target token embedding",
                None,
            );
            g.parameters(&embedding, "embedding", "model.embed_tokens");
            g.edge(&tokens, &embedding, E::Data);
            let fusion = format!("{root}.fusion");
            g.node(
                &fusion,
                K::Projector,
                Some(&root),
                "Residual and token embedding fusion",
                None,
            );
            for prefix in [
                "mtp.pre_fc_norm_embedding",
                "mtp.pre_fc_norm_hidden",
                "mtp.fc_embedding",
                "mtp.fc_hidden",
            ] {
                g.parameters(&fusion, prefix, prefix);
            }
            g.edge(&embedding, &fusion, E::Data);
            // Sequential proposal depths consume the preceding uncollapsed capture.
            let capture = if unit.depth == 0 {
                previous.clone()
            } else {
                format!("prediction.{}.feed_forward.residual", unit.depth - 1)
            };
            g.edge(&capture, &fusion, E::Data);
            g.decoder(
                &root,
                &format!("mtp.layers.{}", unit.depth),
                &unit.mixer,
                &unit.feed_forward.feed_forward,
            );
            g.readout(
                &root,
                "mtp.hyper_connection_mixer",
                &format!("{root}.feed_forward.residual"),
                target.config.tied_embeddings,
            );
            prediction_executions.push(ArchitectureLayerExecution {
                node_id: root,
                physical_layer_index: unit.depth,
            });
        }
        g.descriptor.layer_groups.push(ArchitectureLayerGroup {
            id: "prediction.decoder".into(),
            label: "Independent prediction depths".into(),
            physical_layer_count: prediction.units.len(),
            passes: vec![ArchitectureExecutionPass {
                index: 0,
                executions: prediction_executions,
            }],
            weight_sharing: LayerWeightSharing::None,
        });
        let streams = || {
            let mut axes = axes("hidden", target.config.hidden_size as usize);
            axes.insert(2, TensorAxis { name: "stream".into(), dimension: SymbolicDimension::Known(target.boundary.geometry.streams() as usize) });
            axes
        };
        g.get("target.embedding").output_axes = Some(streams());
        for (ordinal, unit) in target.units.iter().enumerate() {
            let id = format!("target.units.{ordinal}");
            g.get(&id).output_axes = Some(streams());
            if let UnitSpec::Lexical { spec, .. } = unit {
                g.get(&format!("{id}.table")).output_axes =
                    Some(axes("feature", spec.embedding_width as usize));
            }
        }
        for unit in &prediction.units {
            let root = format!("prediction.{}", unit.depth);
            g.get(&root).output_axes = Some(streams());
            g.get(&format!("{root}.embedding")).output_axes =
                Some(axes("hidden", target.config.hidden_size as usize));
            g.get(&format!("{root}.fusion")).output_axes = Some(streams());
        }
        for root in std::iter::once("target".to_owned()).chain(
            prediction
                .units
                .iter()
                .map(|u| format!("prediction.{}", u.depth)),
        ) {
            g.get(&format!("{root}.readout")).output_axes =
                Some(axes("hidden", target.config.hidden_size as usize));
            for suffix in ["head", "logits"] {
                g.get(&format!("{root}.{suffix}")).output_axes =
                    Some(axes("vocabulary", target.config.vocabulary as usize));
            }
        }
        for root in target
            .units
            .iter()
            .enumerate()
            .filter_map(|(i, u)| {
                matches!(u, UnitSpec::Decoder { .. }).then_some(format!("target.units.{i}"))
            })
            .chain(
                prediction
                    .units
                    .iter()
                    .map(|u| format!("prediction.{}", u.depth)),
            )
        {
            for suffix in ["mixer.residual", "feed_forward.residual"] {
                g.get(&format!("{root}.{suffix}")).output_axes = Some(streams());
            }
            for suffix in [
                "mixer",
                "feed_forward",
                "feed_forward.routed",
                "feed_forward.shared",
                "feed_forward.sum",
            ] {
                g.get(&format!("{root}.{suffix}")).output_axes =
                    Some(axes("hidden", target.config.hidden_size as usize));
            }
        }
        g
    }
    pub(super) fn with_prediction_observations(
        mut self,
        target: &TargetSpec,
        prediction: &PredictionSpec,
    ) -> Result<ArchitectureDescriptor, PreparationError> {
        let points = super::observations::prediction_points(target, prediction);
        for point in &points {
            let node = self
                .descriptor
                .nodes
                .iter_mut()
                .find(|node| node.id == point.node_id)
                .ok_or_else(|| {
                    PreparationError::Contract("prediction observation owner is absent".into())
                })?;
            node.observation_paths.push(point.path.clone());
        }
        self.descriptor.observations.points = points;
        Ok(self.descriptor)
    }

    fn node(
        &mut self,
        id: &str,
        kind: ArchitectureNodeKind,
        parent: Option<&str>,
        label: &str,
        layer_index: Option<usize>,
    ) {
        self.descriptor.nodes.push(ArchitectureNode {
            id: id.into(),
            label: label.into(),
            kind,
            parent: parent.map(str::to_owned),
            layer_index,
            parameter_groups: vec![],
            observation_paths: vec![],
            output_axes: None,
            attention: None,
            mixer: None,
            moe: None,
            completeness: DescriptionCompleteness::Complete,
        });
    }
    fn get(&mut self, id: &str) -> &mut ArchitectureNode {
        self.descriptor
            .nodes
            .iter_mut()
            .find(|n| n.id == id)
            .expect("architecture declared node")
    }
    fn parameters(&mut self, node: &str, role: &str, prefix: &str) {
        let id = format!("parameters.{node}.{role}");
        let shared_with = self.canonical.get(prefix).cloned();
        self.canonical
            .entry(prefix.into())
            .or_insert_with(|| id.clone());
        self.descriptor
            .parameter_groups
            .push(ArchitectureParameterGroup {
                id: id.clone(),
                canonical_prefix: prefix.into(),
                shared_with,
            });
        self.get(node).parameter_groups.push(id);
    }
    fn edge(&mut self, from: &str, to: &str, kind: ArchitectureEdgeKind) {
        self.descriptor.edges.push(ArchitectureEdge {
            from: from.into(),
            to: to.into(),
            kind,
        });
    }
    fn invocation(&mut self, node: &str, scope: SpeculativeCaptureScope) {
        self.descriptor
            .speculative_invocations
            .push(SpeculativeCaptureBinding {
                node_id: node.into(),
                scope,
            });
    }
    fn state(&mut self, owner: &str, label: &str) {
        let id = format!("{owner}.state");
        self.node(&id, ArchitectureNodeKind::Opaque, Some(owner), label, None);
        self.edge(&id, owner, ArchitectureEdgeKind::State);
        self.edge(owner, &id, ArchitectureEdgeKind::State);
    }
    fn readout(&mut self, root: &str, prefix: &str, input: &str, tied: bool) {
        use ArchitectureEdgeKind as E;
        use ArchitectureNodeKind as K;
        let readout = format!("{root}.readout");
        self.node(
            &readout,
            K::Projector,
            Some(root),
            "Gated residual collapse; no extra final RMSNorm",
            None,
        );
        self.parameters(&readout, "collapse", prefix);
        self.edge(input, &readout, E::Data);
        let head = format!("{root}.head");
        self.node(
            &head,
            K::OutputHead,
            Some(root),
            "Vocabulary projection",
            None,
        );
        self.parameters(
            &head,
            "vocabulary",
            if tied {
                "model.embed_tokens"
            } else {
                "lm_head"
            },
        );
        self.edge(&readout, &head, E::Data);
        let logits = format!("{root}.logits");
        self.node(
            &logits,
            K::OutputHead,
            Some(root),
            "Authoritative logits before sampling",
            None,
        );
        self.edge(&head, &logits, E::Data);
    }
    fn decoder(
        &mut self,
        root: &str,
        prefix: &str,
        mixer: &MixerSpec,
        ff: &crate::shared_routed::SharedRoutedGatedProductSpec,
    ) {
        use ArchitectureEdgeKind as E;
        use ArchitectureNodeKind as K;
        let mix = format!("{root}.mixer");
        match mixer {
            MixerSpec::Recurrent(spec) => {
                self.node(
                    &mix,
                    K::Mixer,
                    Some(root),
                    "Gated delta recurrent mixer with sigmoid output gate",
                    None,
                );
                self.get(&mix).mixer = Some(MixerAttributes {
                    mechanism: MixerMechanism::GatedDelta,
                    recurrent: true,
                    convolution_width: Some(spec.mixer.convolution.kernel_size as usize),
                });
                self.parameters(&mix, "mixer", &format!("{prefix}.linear_attn"));
                self.state(&mix, "Recurrent matrix and causal convolution history");
            }
            MixerSpec::Indexed(spec) => {
                self.node(
                    &mix,
                    K::Attention,
                    Some(root),
                    "QSA-selected original K/V attention",
                    None,
                );
                self.get(&mix).attention = Some(AttentionAttributes {
                    head_sharing: Some(if spec.heads == spec.kv_heads {
                        HeadSharing::MultiHead
                    } else if spec.kv_heads == 1 {
                        HeadSharing::MultiQuery
                    } else {
                        HeadSharing::GroupedQuery
                    }),
                    query_heads: Some(spec.heads as usize),
                    key_value_heads: Some(spec.kv_heads as usize),
                    key_head_dimension: Some(spec.head_dim as usize),
                    value_head_dimension: Some(spec.head_dim as usize),
                    mechanism: Some(AttentionMechanism::Softmax),
                    recurrent: Some(false),
                    causal: Some(true),
                    positional_encoding: Some(PositionalEncoding::Rotary),
                    ..Default::default()
                });
                self.get(&mix).completeness = DescriptionCompleteness::Partial(vec!["Generic attention attributes do not encode QSA block selection; selected original positions are declared as integer evidence".into()]);
                self.parameters(&mix, "attention", &format!("{prefix}.self_attn"));
                self.state(
                    &mix,
                    "Full original K/V, index summaries, partial block and sequence position",
                );
            }
        }
        let residual = format!("{mix}.residual");
        self.node(
            &residual,
            K::ResidualAdd,
            Some(root),
            "Gated multi-stream mixer residual",
            None,
        );
        self.parameters(
            &residual,
            "residual",
            &format!("{prefix}.attn_hyper_connection"),
        );
        let ingress = if self
            .descriptor
            .node(root)
            .is_some_and(|n| n.kind == K::Prediction)
        {
            format!("{root}.fusion")
        } else {
            root.into()
        };
        self.edge(&ingress, &mix, E::Data);
        self.edge(&mix, &residual, E::Data);
        self.edge(&ingress, &residual, E::Residual);
        let f = format!("{root}.feed_forward");
        self.node(
            &f,
            K::MixtureOfExperts,
            Some(root),
            "Shared and routed gated feed-forward",
            None,
        );
        let policy = ff.router.selection();
        self.get(&f).moe = Some(MoeAttributes {
            routed_experts: ff.experts.group_count() as usize,
            selected_experts: policy.top_k() as usize,
            shared_experts: Some(1),
            shared_expert_width: Some(ff.shared[0].output as usize),
            shared_expert_gated: Some(true),
            granularity: Some(RoutingGranularity::Token),
            score_transform: Some(match policy.scoring() {
                eredu_nn::GroupScoring::Softmax => RoutingScoreTransform::Softmax,
                eredu_nn::GroupScoring::SelectedSoftmax => RoutingScoreTransform::SelectedSoftmax,
                eredu_nn::GroupScoring::Sigmoid => RoutingScoreTransform::Sigmoid,
                eredu_nn::GroupScoring::SqrtSoftplus => RoutingScoreTransform::SqrtSoftplus,
                _ => RoutingScoreTransform::Identity,
            }),
            normalization: Some(if policy.normalize_selected() {
                RoutingNormalization::SelectedSum
            } else {
                RoutingNormalization::None
            }),
        });
        self.edge(&residual, &f, E::Data);
        for (suffix, kind, label, parameters) in [
            ("router", K::Router, "Expert selection", "gate"),
            (
                "routed",
                K::RoutedExperts,
                "Independently addressable expert bank",
                "experts",
            ),
            (
                "shared",
                K::SharedExperts,
                "Always-on gated shared branch",
                "shared_expert",
            ),
        ] {
            let node = format!("{f}.{suffix}");
            self.node(&node, kind, Some(&f), label, None);
            self.parameters(&node, suffix, &format!("{prefix}.mlp.{parameters}"));
            self.edge(&f, &node, E::Data);
        }
        self.parameters(
            &format!("{f}.shared"),
            "gate",
            &format!("{prefix}.mlp.shared_expert_gate"),
        );
        self.edge(&format!("{f}.router"), &format!("{f}.routed"), E::Routing);
        let sum = format!("{f}.sum");
        self.node(
            &sum,
            K::Sum,
            Some(&f),
            "Routed plus gated shared contribution",
            None,
        );
        self.edge(&format!("{f}.routed"), &sum, E::Data);
        self.edge(&format!("{f}.shared"), &sum, E::Data);
        let output = format!("{f}.residual");
        self.node(
            &output,
            K::ResidualAdd,
            Some(root),
            "Gated multi-stream feed-forward residual",
            None,
        );
        self.parameters(
            &output,
            "residual",
            &format!("{prefix}.mlp_hyper_connection"),
        );
        self.edge(&sum, &output, E::Data);
        self.edge(&residual, &output, E::Residual);
    }
    pub fn join(
        mut self,
        catalog: ObservationCatalog,
        edits: &[eredu_core::intervention::InterventionPoint],
        execution: &crate::speculative_execution::SpeculativeActivationExecution,
    ) -> Result<
        (ArchitectureDescriptor, Vec<SpeculativeCaptureBinding>),
        eredu_core::capture::CaptureError,
    > {
        use eredu_core::capture::CaptureError;
        let mut bindings = BTreeMap::new();
        for (node, path) in catalog
            .points
            .iter()
            .map(|p| (&p.node_id, Some(&p.path)))
            .chain(edits.iter().map(|e| (&e.node_id, None)))
        {
            let scope =
                crate::speculative_execution::speculative_capture_scope(&self.descriptor, node)?;
            execution.validate_scope(scope)?;
            bindings.insert(node.clone(), scope);
            if let Some(path) = path {
                self.get(node).observation_paths.push(path.clone());
            }
        }
        // Every retained point must be owned by an actual operation; no synthetic
        // activation node or backend selector parsing supplies missing ancestry.
        for point in &catalog.points {
            if self.descriptor.node(&point.node_id).is_none() {
                return Err(CaptureError::Invalid(
                    "unowned architecture observation".into(),
                ));
            }
        }
        self.descriptor.observations = catalog;
        Ok((
            self.descriptor,
            bindings
                .into_iter()
                .map(|(node_id, scope)| SpeculativeCaptureBinding { node_id, scope })
                .collect(),
        ))
    }
}

fn axes(name: &str, width: usize) -> Vec<TensorAxis> {
    vec![
        TensorAxis {
            name: "batch".into(),
            dimension: SymbolicDimension::Batch,
        },
        TensorAxis {
            name: "sequence".into(),
            dimension: SymbolicDimension::Sequence,
        },
        TensorAxis {
            name: name.into(),
            dimension: SymbolicDimension::Known(width),
        },
    ]
}

/// Joins retained prediction invocations to ordinary target discovery without
/// changing the target graph's public node identities or boundary catalog.
pub(crate) fn join_prediction_descriptor(
    mut target: ArchitectureDescriptor,
    prediction: ArchitectureDescriptor,
) -> Result<ArchitectureDescriptor, String> {
    let mut ids = std::collections::BTreeSet::new();
    for node in &prediction.nodes {
        if crate::speculative_execution::speculative_capture_scope(&prediction, &node.id)
            .map_err(|error| error.to_string())?
            != SpeculativeCaptureScope::Target
        {
            ids.insert(node.id.clone());
        }
    }
    let capture = target
        .edges
        .iter()
        .find(|edge| edge.to == "output.collapse")
        .map(|edge| edge.from.clone())
        .ok_or_else(|| "target residual collapse input is absent".to_owned())?;
    // The configuration-only root was explicitly opaque; replace that declaration
    // with the selected, physically executable prediction invocations.
    target.nodes.retain(|node| node.id != "prediction");
    target
        .edges
        .retain(|edge| edge.from != "prediction" && edge.to != "prediction");
    let group_ids: std::collections::BTreeSet<_> = prediction
        .nodes
        .iter()
        .filter(|node| ids.contains(&node.id))
        .flat_map(|node| node.parameter_groups.iter().cloned())
        .collect();
    for mut group in prediction
        .parameter_groups
        .iter()
        .filter(|group| group_ids.contains(&group.id))
        .cloned()
    {
        if let Some(shared) = &group.shared_with {
            if !group_ids.contains(shared) {
                let prefix = prediction
                    .parameter_groups
                    .iter()
                    .find(|candidate| &candidate.id == shared)
                    .map(|candidate| &candidate.canonical_prefix)
                    .ok_or_else(|| "prediction shared parameter target is absent".to_owned())?;
                group.shared_with = Some(
                    target
                        .parameter_groups
                        .iter()
                        .find(|candidate| &candidate.canonical_prefix == prefix)
                        .map(|candidate| candidate.id.clone())
                        .ok_or_else(|| {
                            "prediction shared parameter differs from target discovery".to_owned()
                        })?,
                );
            }
        }
        target.parameter_groups.push(group);
    }
    let source_capture = prediction
        .edges
        .iter()
        .find(|edge| edge.to == "target.readout" && edge.kind == ArchitectureEdgeKind::Data)
        .map(|edge| edge.from.as_str())
        .ok_or_else(|| "authored target residual collapse input is absent".to_owned())?;
    let mut edges = Vec::new();
    for mut edge in prediction
        .edges
        .iter()
        .filter(|edge| ids.contains(&edge.to))
        .cloned()
    {
        if !ids.contains(&edge.from) {
            if edge.from != source_capture
                || edge.to != "prediction.0.fusion"
                || edge.kind != ArchitectureEdgeKind::Data
            {
                return Err("prediction has an undeclared cross-invocation input".into());
            }
            edge.from = capture.clone();
        }
        edges.push(edge);
    }
    for node in prediction
        .nodes
        .iter()
        .filter(|node| ids.contains(&node.id))
    {
        if target.node(&node.id).is_some() {
            return Err("prediction node collides with target discovery".into());
        }
    }
    for point in &prediction.observations.points {
        if ids.contains(&point.node_id) && target.observations.get(&point.path).is_some() {
            return Err("prediction observation collides with target discovery".into());
        }
    }
    target.nodes.extend(
        prediction
            .nodes
            .into_iter()
            .filter(|node| ids.contains(&node.id)),
    );
    target.edges.extend(edges);
    target
        .layer_groups
        .extend(prediction.layer_groups.into_iter().filter(|group| {
            group
                .passes
                .iter()
                .flat_map(|pass| &pass.executions)
                .any(|execution| ids.contains(&execution.node_id))
        }));
    target.speculative_invocations.extend(
        prediction
            .speculative_invocations
            .into_iter()
            .filter(|binding| ids.contains(&binding.node_id)),
    );
    target.observations.points.extend(
        prediction
            .observations
            .points
            .into_iter()
            .filter(|point| ids.contains(&point.node_id)),
    );
    target
        .observations
        .points
        .sort_by(|a, b| a.path.cmp(&b.path));
    for (completeness, obsolete, remaining) in [
        (
            &mut target.completeness,
            crate::discovery::OPAQUE_PREDICTION_DESCRIPTION,
            "Prediction component score/readout equations remain unprojected",
        ),
        (
            &mut target.observations.completeness,
            crate::discovery::UNENUMERATED_PREDICTION_CAPTURES,
            "Prediction routed feed-forward component captures and intervention catalog remain unprojected",
        ),
    ] {
        if let DescriptionCompleteness::Partial(reasons) = completeness {
            for reason in reasons {
                if reason == obsolete {
                    *reason = remaining.into();
                }
            }
        }
    }
    Ok(target)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    fn graphs() -> (ArchitectureDescriptor, ArchitectureDescriptor) {
        let config: serde_json::Value =
            serde_json::from_str(include_str!("../config/released.json")).unwrap();
        let target = crate::configuration::MODEL_CONFIGURATIONS
            .resolve_safetensors(&config)
            .unwrap()
            .architecture_plan()
            .architecture_descriptor();
        let mut descriptor = target.clone();
        descriptor.nodes.clear();
        descriptor.edges.clear();
        descriptor.parameter_groups.clear();
        descriptor.layer_groups.clear();
        descriptor.speculative_invocations.clear();
        descriptor.observations.points.clear();
        let mut graph = Graph {
            descriptor,
            canonical: BTreeMap::new(),
        };
        graph.node("target", ArchitectureNodeKind::Opaque, None, "Target", None);
        graph.invocation("target", SpeculativeCaptureScope::Target);
        graph.node(
            "target.capture",
            ArchitectureNodeKind::ResidualAdd,
            Some("target"),
            "Residual",
            None,
        );
        graph.node(
            "target.readout",
            ArchitectureNodeKind::Projector,
            Some("target"),
            "Readout",
            None,
        );
        graph.parameters("target", "embedding", "model.embed_tokens");
        graph.parameters("target", "head", "lm_head");
        graph.edge(
            "target.capture",
            "target.readout",
            ArchitectureEdgeKind::Data,
        );
        graph.node(
            "prediction.0",
            ArchitectureNodeKind::Prediction,
            None,
            "Prediction",
            Some(0),
        );
        graph.invocation(
            "prediction.0",
            SpeculativeCaptureScope::Prediction { depth: 0 },
        );
        for suffix in ["embedding", "head", "fusion"] {
            graph.node(
                &format!("prediction.0.{suffix}"),
                ArchitectureNodeKind::Projector,
                Some("prediction.0"),
                suffix,
                None,
            );
        }
        graph.parameters("prediction.0.embedding", "embedding", "model.embed_tokens");
        graph.parameters("prediction.0.head", "head", "lm_head");
        graph.edge(
            "target.capture",
            "prediction.0.fusion",
            ArchitectureEdgeKind::Data,
        );
        graph.edge(
            "prediction.0.embedding",
            "prediction.0.fusion",
            ArchitectureEdgeKind::Data,
        );
        let mut point = target
            .observations
            .points
            .iter()
            .find(|point| point.node_id == "decoder.layers.47")
            .unwrap()
            .clone();
        point.path = "mtp.layers.0.prediction.fusion".into();
        point.node_id = "prediction.0.fusion".into();
        point
            .requirements
            .push(ObservationRequirement::PredictionExecution);
        graph
            .get("prediction.0.fusion")
            .observation_paths
            .push(point.path.clone());
        graph.descriptor.observations.points.push(point);
        (target, graph.descriptor)
    }

    #[test]
    fn retained_prediction_join_preserves_target_identity_shape_and_shared_parameters() {
        let (target, prediction) = graphs();
        let expected_axes = prediction.observations.points[0].axes.clone();
        let expected_capture = target
            .edges
            .iter()
            .find(|edge| edge.to == "output.collapse")
            .unwrap()
            .from
            .clone();
        let joined = join_prediction_descriptor(target.clone(), prediction).unwrap();
        for point in &target.observations.points {
            assert_eq!(joined.observations.get(&point.path), Some(point));
        }
        for node in target.nodes.iter().filter(|node| node.id != "prediction") {
            assert_eq!(joined.node(&node.id), Some(node));
        }
        for (original, updated, obsolete) in [
            (
                &target.completeness,
                &joined.completeness,
                crate::discovery::OPAQUE_PREDICTION_DESCRIPTION,
            ),
            (
                &target.observations.completeness,
                &joined.observations.completeness,
                crate::discovery::UNENUMERATED_PREDICTION_CAPTURES,
            ),
        ] {
            let DescriptionCompleteness::Partial(original) = original else {
                panic!("partial target");
            };
            let DescriptionCompleteness::Partial(updated) = updated else {
                panic!("remaining prediction gaps");
            };
            assert!(original.iter().any(|reason| reason == obsolete));
            assert!(!updated.iter().any(|reason| reason == obsolete));
            for reason in original.iter().filter(|reason| reason.as_str() != obsolete) {
                assert!(
                    updated.contains(reason),
                    "unrelated omission was lost: {reason}"
                );
            }
            assert_eq!(original.len(), updated.len());
        }
        let point = joined
            .observations
            .get("mtp.layers.0.prediction.fusion")
            .unwrap();
        assert_eq!(point.axes, expected_axes);
        assert_eq!(point.axes.as_ref().unwrap().len(), 4);
        assert_eq!(
            crate::speculative_execution::speculative_capture_scope(&joined, &point.node_id)
                .unwrap(),
            SpeculativeCaptureScope::Prediction { depth: 0 }
        );
        assert!(joined
            .edges
            .iter()
            .any(|edge| edge.from == expected_capture && edge.to == "prediction.0.fusion"));
        assert!(joined
            .edges
            .iter()
            .all(|edge| joined.node(&edge.from).is_some() && joined.node(&edge.to).is_some()));
        for prefix in ["model.embed_tokens", "lm_head"] {
            let original = target
                .parameter_groups
                .iter()
                .find(|group| group.canonical_prefix == prefix)
                .unwrap();
            let shared = joined
                .parameter_groups
                .iter()
                .find(|group| group.shared_with.as_ref() == Some(&original.id))
                .unwrap();
            assert_eq!(shared.canonical_prefix, prefix);
        }
        assert_eq!(
            joined
                .nodes
                .iter()
                .map(|node| &node.id)
                .collect::<BTreeSet<_>>()
                .len(),
            joined.nodes.len()
        );
        assert_eq!(
            joined
                .observations
                .points
                .iter()
                .map(|point| &point.path)
                .collect::<BTreeSet<_>>()
                .len(),
            joined.observations.points.len()
        );
    }

    #[test]
    fn retained_prediction_join_rejects_unowned_inputs_and_aliases() {
        let (target, mut prediction) = graphs();
        prediction.edges.push(ArchitectureEdge {
            from: "target".into(),
            to: "prediction.0.fusion".into(),
            kind: ArchitectureEdgeKind::Data,
        });
        assert!(join_prediction_descriptor(target, prediction)
            .unwrap_err()
            .contains("cross-invocation"));
        let (mut target, prediction) = graphs();
        target
            .parameter_groups
            .retain(|group| group.canonical_prefix != "model.embed_tokens");
        assert!(join_prediction_descriptor(target, prediction)
            .unwrap_err()
            .contains("shared parameter"));
        let (mut target, prediction) = graphs();
        target
            .observations
            .points
            .push(prediction.observations.points[0].clone());
        assert!(join_prediction_descriptor(target, prediction)
            .unwrap_err()
            .contains("observation collides"));
    }
}
