//! Logical architecture projection from the same normalized plans used for construction.

use crate::configuration::{GgufModelConfig, SafetensorsModelConfig};
use crate::processor_plan::ArtifactArchitecturePlan;
use eredu_core::*;

mod families;

impl ArtifactArchitecturePlan {
    /// Describes admitted family semantics and implemented observation points.
    /// This only reads the retained plan; it never opens weights or creates a device.
    pub fn architecture_descriptor(&self) -> ArchitectureDescriptor {
        let mut graph = Builder::new();
        if let Some(projector) = self.gguf_media_projector() {
            families::gguf_composite(&mut graph, projector.model());
            return graph.finish();
        }
        if let Some(plan) = self.safetensors_architecture() {
            match plan.model() {
                SafetensorsModelConfig::Llama(c) => families::dense(&mut graph, c, None),
                SafetensorsModelConfig::Qwen(c) => families::qwen(&mut graph, c),
                SafetensorsModelConfig::GptOss(c) => families::gpt_oss(&mut graph, c),
                SafetensorsModelConfig::QwenHybrid(c) => families::qwen_hybrid(&mut graph, c),
                other => families::remaining_safetensors(&mut graph, other),
            }
        } else if let Some(plan) = self.gguf_plan() {
            match plan.model() {
                GgufModelConfig::Llama(c) => families::dense(&mut graph, c, None),
                GgufModelConfig::Qwen(c) => families::qwen(&mut graph, c),
                GgufModelConfig::GptOss(c) => families::gpt_oss(&mut graph, c),
                GgufModelConfig::QwenHybrid(c) => families::qwen_hybrid(&mut graph, c),
                other => families::remaining_gguf(&mut graph, other),
            }
        }
        graph.finish()
    }
}

fn axes(width: usize) -> Vec<TensorAxis> {
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
            name: "hidden".into(),
            dimension: SymbolicDimension::Known(width),
        },
    ]
}

struct Builder {
    descriptor: ArchitectureDescriptor,
}

impl Builder {
    fn new() -> Self {
        Self {
            descriptor: ArchitectureDescriptor {
                schema_version: DISCOVERY_SCHEMA_VERSION,
                nodes: vec![],
                edges: vec![],
                parameter_groups: vec![],
                observations: ObservationCatalog {
                    schema_version: DISCOVERY_SCHEMA_VERSION,
                    points: vec![],
                    completeness: DescriptionCompleteness::Complete,
                },
                completeness: DescriptionCompleteness::Complete,
            },
        }
    }

    fn partial(&mut self, reason: impl Into<String>) {
        let reason = reason.into();
        match &mut self.descriptor.completeness {
            DescriptionCompleteness::Partial(reasons) => reasons.push(reason),
            status => *status = DescriptionCompleteness::Partial(vec![reason]),
        }
    }

    fn catalog_partial(&mut self, reason: impl Into<String>) {
        let reason = reason.into();
        match &mut self.descriptor.observations.completeness {
            DescriptionCompleteness::Partial(reasons) => reasons.push(reason),
            status => *status = DescriptionCompleteness::Partial(vec![reason]),
        }
    }

    fn node(
        &mut self,
        id: &str,
        kind: ArchitectureNodeKind,
        parent: Option<&str>,
        parameter: Option<&str>,
        width: Option<usize>,
    ) {
        let mut groups = vec![];
        if let Some(prefix) = parameter {
            let group = format!("parameters:{prefix}");
            if !self
                .descriptor
                .parameter_groups
                .iter()
                .any(|g| g.id == group)
            {
                self.descriptor
                    .parameter_groups
                    .push(ArchitectureParameterGroup {
                        id: group.clone(),
                        canonical_prefix: prefix.into(),
                    });
            }
            groups.push(group);
        }
        self.descriptor.nodes.push(ArchitectureNode {
            id: id.into(),
            label: id.into(),
            kind,
            parent: parent.map(str::to_owned),
            layer_index: None,
            parameter_groups: groups,
            observation_paths: vec![],
            output_axes: width.map(axes),
            attention: None,
            mixer: None,
            moe: None,
            completeness: DescriptionCompleteness::Complete,
        });
    }

    fn get_mut(&mut self, id: &str) -> &mut ArchitectureNode {
        self.descriptor
            .nodes
            .iter_mut()
            .find(|n| n.id == id)
            .expect("declared node")
    }

    fn edge(&mut self, from: &str, to: &str, kind: ArchitectureEdgeKind) {
        self.descriptor.edges.push(ArchitectureEdge {
            from: from.into(),
            to: to.into(),
            kind,
        });
    }

    fn observation(
        &mut self,
        node: &str,
        path: String,
        meaning: &str,
        dtype: ObservationDtype,
        shape: Option<Vec<TensorAxis>>,
        routing: bool,
    ) {
        self.get_mut(node).observation_paths.push(path.clone());
        self.descriptor.observations.points.push(ObservationPoint {
            path,
            node_id: node.into(),
            meaning: meaning.into(),
            value_type: ObservationValueType::Tensor,
            dtype,
            axes: shape,
            prefill: true,
            decode: true,
            requirements: vec![if routing {
                ObservationRequirement::RoutingEvents
            } else {
                ObservationRequirement::ActivationHooks
            }],
            position: if routing {
                ObservationPosition::ReadOnly
            } else {
                ObservationPosition::BeforeIntervention
            },
            retained_bytes: None,
            host_bytes: None,
        });
    }

    fn start(&mut self, embedding: &str, width: usize) -> String {
        self.node(
            "embedding",
            ArchitectureNodeKind::Embedding,
            None,
            Some(embedding),
            Some(width),
        );
        "embedding".into()
    }

    fn output(&mut self, previous: &str, norm: &str, head: &str, width: usize, vocabulary: usize) {
        self.node(
            "output.norm",
            ArchitectureNodeKind::Normalization,
            None,
            Some(norm),
            Some(width),
        );
        self.node(
            "output",
            ArchitectureNodeKind::OutputHead,
            None,
            Some(head),
            None,
        );
        let mut shape = axes(vocabulary);
        shape[2].name = "vocabulary".into();
        self.get_mut("output").output_axes = Some(shape.clone());
        self.edge(previous, "output.norm", ArchitectureEdgeKind::Data);
        self.edge("output.norm", "output", ArchitectureEdgeKind::Data);
        self.observation(
            "output",
            MODEL_LOGITS_OBSERVATION_PATH.into(),
            "Final vocabulary logits",
            ObservationDtype::Floating,
            Some(shape),
            false,
        );
    }

    fn block(&mut self, previous: &str, index: usize, path: &str, width: usize) -> String {
        let id = format!("decoder.layers.{index}");
        self.node(
            &id,
            ArchitectureNodeKind::DecoderBlock,
            None,
            Some(path),
            Some(width),
        );
        self.get_mut(&id).layer_index = Some(index);
        self.edge(previous, &id, ArchitectureEdgeKind::Data);
        self.observation(
            &id,
            UnitObservation::Input.path(path),
            "Block input",
            ObservationDtype::Floating,
            Some(axes(width)),
            false,
        );
        self.observation(
            &id,
            UnitObservation::Output.path(path),
            "Block output including residual contributions",
            ObservationDtype::Floating,
            Some(axes(width)),
            false,
        );
        id
    }

    /// One pre-normalized residual sublayer. The bypass is a distinct data-flow edge.
    fn sublayer(
        &mut self,
        block: &str,
        input: &str,
        name: &str,
        norm: &str,
        parameter: &str,
        width: usize,
        kind: ArchitectureNodeKind,
    ) -> (String, String) {
        let normalizer = format!("{block}.{name}.norm");
        let op = format!("{block}.{name}");
        let join = format!("{block}.{name}.residual");
        self.node(
            &normalizer,
            ArchitectureNodeKind::Normalization,
            Some(block),
            Some(norm),
            Some(width),
        );
        self.node(&op, kind, Some(block), Some(parameter), Some(width));
        self.node(
            &join,
            ArchitectureNodeKind::ResidualAdd,
            Some(block),
            None,
            Some(width),
        );
        self.edge(input, &normalizer, ArchitectureEdgeKind::Data);
        self.edge(&normalizer, &op, ArchitectureEdgeKind::Data);
        self.edge(&op, &join, ArchitectureEdgeKind::Data);
        self.edge(input, &join, ArchitectureEdgeKind::Residual);
        (op, join)
    }

    fn moe(
        &mut self,
        node: &str,
        parameter: &str,
        attributes: MoeAttributes,
        routing_path: Option<&str>,
        width: usize,
    ) {
        self.get_mut(node).kind = ArchitectureNodeKind::MixtureOfExperts;
        self.get_mut(node).moe = Some(attributes.clone());
        let router = format!("{node}.router");
        let routed = format!("{node}.routed");
        let sum = format!("{node}.sum");
        self.node(
            &router,
            ArchitectureNodeKind::Router,
            Some(node),
            None,
            None,
        );
        self.node(
            &routed,
            ArchitectureNodeKind::RoutedExperts,
            Some(node),
            None,
            Some(width),
        );
        self.node(
            &sum,
            ArchitectureNodeKind::Sum,
            Some(node),
            None,
            Some(width),
        );
        // Group ownership is inherited from the enclosing MoE node; individual
        // checkpoint field names differ by family and are not guessed here.
        let _ = parameter;
        self.edge(node, &router, ArchitectureEdgeKind::Data);
        self.edge(node, &routed, ArchitectureEdgeKind::Data);
        self.edge(&router, &routed, ArchitectureEdgeKind::Routing);
        self.edge(&routed, &sum, ArchitectureEdgeKind::Data);
        if attributes.shared_experts.is_some_and(|n| n > 0) {
            let shared = format!("{node}.shared");
            self.node(
                &shared,
                ArchitectureNodeKind::SharedExperts,
                Some(node),
                None,
                Some(width),
            );
            self.edge(node, &shared, ArchitectureEdgeKind::Data);
            self.edge(&shared, &sum, ArchitectureEdgeKind::Data);
        }
        // The enclosing sublayer's output is the branch sum.
        for edge in &mut self.descriptor.edges {
            if edge.from == node && edge.to.ends_with(".residual") {
                edge.from = sum.clone();
            }
        }
        if let Some(path) = routing_path {
            use RoutingObservationField as F;
            let mut fields = vec![
                F::SelectedExperts,
                F::SelectedScores,
                F::Coefficients,
                F::RoutedOutput,
            ];
            if attributes.shared_experts.is_some_and(|n| n > 0) {
                fields.extend([F::SharedOutput, F::CombinedOutput]);
            }
            for field in fields {
                let (dtype, shape) = match field {
                    F::SelectedExperts | F::SelectedScores | F::Coefficients => {
                        let shape = vec![
                            TensorAxis {
                                name: "token".into(),
                                dimension: SymbolicDimension::TokenRows,
                            },
                            TensorAxis {
                                name: "selected_expert".into(),
                                dimension: SymbolicDimension::Known(attributes.selected_experts),
                            },
                        ];
                        (
                            if field == F::SelectedExperts {
                                ObservationDtype::Integer
                            } else {
                                ObservationDtype::Floating
                            },
                            shape,
                        )
                    }
                    _ => (ObservationDtype::Floating, axes(width)),
                };
                self.observation(
                    node,
                    field.path(path),
                    &format!("{field:?}; reported after expert dispatch"),
                    dtype,
                    Some(shape),
                    true,
                );
            }
            self.observation(
                node,
                UnitObservation::Output.path(path),
                "Expert contribution before its output intervention",
                ObservationDtype::Floating,
                Some(axes(width)),
                false,
            );
        }
    }

    fn finish(mut self) -> ArchitectureDescriptor {
        self.descriptor
            .observations
            .points
            .sort_by(|a, b| a.path.cmp(&b.path));
        self.descriptor
    }
}

fn attention(heads: i32, kv: i32, dim: i32, policy: AttentionPolicy) -> AttentionAttributes {
    AttentionAttributes {
        head_sharing: Some(if heads == kv {
            HeadSharing::MultiHead
        } else if kv == 1 {
            HeadSharing::MultiQuery
        } else {
            HeadSharing::GroupedQuery
        }),
        query_heads: Some(heads as usize),
        key_value_heads: Some(kv as usize),
        key_head_dimension: Some(dim as usize),
        value_head_dimension: Some(dim as usize),
        receptive_field: Some(match policy {
            AttentionPolicy::Full => ReceptiveField::Full,
            AttentionPolicy::Sliding { window } => ReceptiveField::Sliding {
                window: window.get() as usize,
            },
        }),
        mechanism: Some(AttentionMechanism::Softmax),
        recurrent: Some(false),
        causal: Some(true),
        positional_encoding: Some(PositionalEncoding::Rotary),
    }
}

fn moe(
    experts: i32,
    selected: i32,
    shared: i32,
    normalize: bool,
    score: RoutingScoreTransform,
) -> MoeAttributes {
    MoeAttributes {
        routed_experts: experts as usize,
        selected_experts: selected as usize,
        shared_experts: Some(shared as usize),
        shared_expert_width: None,
        shared_expert_gated: Some(false),
        granularity: Some(RoutingGranularity::Token),
        score_transform: Some(score),
        normalization: Some(if normalize {
            RoutingNormalization::SelectedSum
        } else {
            RoutingNormalization::None
        }),
    }
}

#[cfg(test)]
mod tests;
