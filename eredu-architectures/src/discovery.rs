//! Logical architecture projection from the same normalized plans used for construction.

use crate::configuration::{GgufModelConfig, SafetensorsModelConfig};
use crate::processor_plan::ArtifactArchitecturePlan;
use eredu_core::*;

mod components;
mod families;
mod routed_components;

impl ArtifactArchitecturePlan {
    /// Describes admitted family semantics and implemented observation points.
    /// This only reads the retained plan; it never opens weights or creates a device.
    pub fn architecture_descriptor(&self) -> ArchitectureDescriptor {
        self.discovery_builder().finish()
    }

    /// Genuine mutable hooks declared alongside their instrumentation geometry.
    /// This catalog is separate from read-only observations and graph nodes.
    pub fn intervention_points(&self) -> Vec<eredu_core::intervention::InterventionPoint> {
        let mut graph = self.discovery_builder();
        routed_components::declare_captures(&mut graph);
        graph.scope_requirements();
        graph
            .interventions
            .into_iter()
            .filter(|point| {
                graph
                    .descriptor
                    .observations
                    .get(&if point.routing.is_some() {
                        RoutingObservationField::SelectedExperts.path(&point.path)
                    } else {
                        point.path.clone()
                    })
                    .is_some()
            })
            .collect()
    }

    fn discovery_builder(&self) -> Builder {
        let mut graph = Builder::new();
        if let Some(projector) = self.gguf_media_projector() {
            families::gguf_composite(&mut graph, projector.model());
            return graph;
        }
        if let Some(plan) = self.safetensors_architecture() {
            match plan.model() {
                SafetensorsModelConfig::Llama(c) => families::dense(&mut graph, c, None),
                SafetensorsModelConfig::Gemma2(c) => families::dense(&mut graph, c, None),
                SafetensorsModelConfig::Qwen(c) => families::qwen(&mut graph, c),
                SafetensorsModelConfig::GptOss(c) => families::gpt_oss(&mut graph, c),
                SafetensorsModelConfig::QwenHybrid(c) => families::qwen_hybrid(&mut graph, c),
                other => families::remaining_safetensors(&mut graph, other),
            }
        } else if let Some(plan) = self.gguf_plan() {
            match plan.model() {
                GgufModelConfig::Llama(c) => families::dense(&mut graph, c, None),
                GgufModelConfig::Gemma2(c) => families::dense(&mut graph, c, None),
                GgufModelConfig::Nanbeige(c) => families::nanbeige(&mut graph, c),
                GgufModelConfig::Qwen(c) => families::qwen(&mut graph, c),
                GgufModelConfig::GptOss(c) => families::gpt_oss(&mut graph, c),
                GgufModelConfig::QwenHybrid(c) => families::qwen_hybrid(&mut graph, c),
                other => families::remaining_gguf(&mut graph, other),
            }
        }
        graph
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
    interventions: Vec<eredu_core::intervention::InterventionPoint>,
}

impl Builder {
    fn routing_control(
        &mut self,
        node: &str,
        path: &str,
        spec: eredu_nn::TopKGroupSelectionSpec,
        shared_experts: u32,
        learned_coefficient_scale: bool,
    ) {
        use eredu_core::intervention::*;
        use eredu_nn::GroupScoring as S;
        let scoring = match spec.scoring() {
            S::Softmax => RoutingScoring::Softmax,
            S::SelectedSoftmax => RoutingScoring::SelectedSoftmax,
            S::Sigmoid => RoutingScoring::Sigmoid,
            S::SqrtSoftplus => RoutingScoring::SqrtSoftplus,
            _ => return,
        };
        self.interventions.push(InterventionPoint {
            path: path.into(), node_id: node.into(), stage: InterventionStage::RoutingBeforeDispatch,
            axes: vec![TensorAxis { name: "token".into(), dimension: SymbolicDimension::TokenRows }, TensorAxis { name: "selected_expert".into(), dimension: SymbolicDimension::Known(spec.top_k() as usize) }],
            dtypes: vec![], operations: vec![InterventionKind::ExcludeExperts, InterventionKind::ZeroExpertContribution, InterventionKind::BiasRoutingScores, InterventionKind::ForceExperts],
            score_stages: vec![RoutingScoreStage::RawLogits, RoutingScoreStage::TransformedScores, RoutingScoreStage::RankingScores],
            prefill: ObservationSupportStatus::Unverified("requires loaded-session routing mechanisms".into()),
            decode: ObservationSupportStatus::Unverified("requires loaded-session routing mechanisms".into()),
            conditions: vec!["Global routed-expert IDs; shared experts are unchanged".into(),
                "Force selects exactly top-k distinct IDs per selected token row and uses architecture weights".into(),
                "Zero contribution preserves IDs and other coefficient magnitudes; expert computation may still occur".into()],
            routed_units: None,
            routing: Some(InterventionRoutingPolicy {
                expert_count: spec.group_count() as u32, top_k: spec.top_k() as u32, scoring,
                normalize_selected: spec.normalize_selected(), normalization_epsilon: spec.normalization_epsilon(),
                coefficient_scale: spec.coefficient_scale(), groups: spec.selection_partitions() as u32,
                selected_groups: spec.selected_groups() as u32, learned_coefficient_scale, shared_experts,
            }),
        });
    }

    fn new() -> Self {
        Self {
            interventions: vec![],
            descriptor: ArchitectureDescriptor {
                schema_version: ARCHITECTURE_DESCRIPTOR_SCHEMA_VERSION,
                nodes: vec![],
                edges: vec![],
                parameter_groups: vec![],
                layer_groups: vec![],
                components: vec![],
                component_transforms: vec![],
                routed_components: vec![],
                component_readout: None,
                component_scopes: vec![],
                speculative_invocations: vec![],
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
            groups.push(self.parameter_group(prefix));
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

    fn parameter_group(&mut self, prefix: &str) -> String {
        let group = format!("parameters:{prefix}");
        if !self
            .descriptor
            .parameter_groups
            .iter()
            .any(|candidate| candidate.id == group)
        {
            self.descriptor
                .parameter_groups
                .push(ArchitectureParameterGroup {
                    id: group.clone(),
                    canonical_prefix: prefix.into(),
                    shared_with: None,
                });
        }
        group
    }

    fn attach_parameters(&mut self, node: &str, prefix: &str) {
        let group = self.parameter_group(prefix);
        let node = self.get_mut(node);
        if !node.parameter_groups.contains(&group) {
            node.parameter_groups.push(group);
        }
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
        // The activation-hook declaration registers both surfaces. Read-only
        // routing events deliberately create no mutable intervention target.
        if !routing {
            use eredu_core::intervention::*;
            let logits = path == MODEL_LOGITS_OBSERVATION_PATH;
            let mut operations = vec![
                InterventionKind::Zero,
                InterventionKind::Scale,
                InterventionKind::Mask,
                InterventionKind::Replace,
                InterventionKind::Add,
            ];
            if shape
                .as_ref()
                .and_then(|s| s.last())
                .is_some_and(|a| a.name == "component")
            {
                operations.push(InterventionKind::MaskComponents);
            }
            if logits {
                operations.push(InterventionKind::MaskLogits);
            }
            self.interventions.push(InterventionPoint {
                path: path.clone(), node_id: node.into(),
                stage: if logits { InterventionStage::LogitsBeforeSampling } else { InterventionStage::Activation },
                axes: shape.clone().unwrap_or_default(),
                dtypes: vec![InterventionDtype::Float32, InterventionDtype::Float16, InterventionDtype::Bfloat16],
                operations, score_stages: vec![],
                prefill: ObservationSupportStatus::Unverified("requires loaded-session support".into()),
                decode: ObservationSupportStatus::Unverified("requires loaded-session support".into()),
                conditions: vec!["Runtime dtype must exactly match the plan; no payload broadcasting or dtype conversion".into()],
                routed_units: None,
                routing: None,
            });
        }
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

    /// Declares both sides of a real component hook. Effective values are read-only
    /// and never acquire a second intervention target.
    fn component_observation(
        &mut self,
        node: &str,
        path: String,
        meaning: &str,
        shape: Vec<TensorAxis>,
    ) {
        self.observation(
            node,
            path.clone(),
            meaning,
            ObservationDtype::Floating,
            Some(shape),
            false,
        );
        let mut effective = self
            .descriptor
            .observations
            .points
            .last()
            .expect("inserted observation")
            .clone();
        effective.path = format!("{path}.effective");
        effective.meaning =
            format!("Effective {meaning}; consumed after all interventions at this boundary");
        effective.position = ObservationPosition::AfterIntervention;
        self.get_mut(node)
            .observation_paths
            .push(effective.path.clone());
        self.descriptor.observations.points.push(effective);
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
        self.decoder_execution(&id, index);
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

    /// Ordinary stacks have one physical layer for each logical invocation.
    fn decoder_execution(&mut self, node: &str, physical_layer_index: usize) {
        if self.descriptor.layer_groups.is_empty() {
            self.descriptor.layer_groups.push(ArchitectureLayerGroup {
                id: "decoder".into(),
                label: "Decoder".into(),
                physical_layer_count: 0,
                passes: vec![ArchitectureExecutionPass {
                    index: 0,
                    executions: vec![],
                }],
                weight_sharing: LayerWeightSharing::None,
            });
        }
        let group = &mut self.descriptor.layer_groups[0];
        group.physical_layer_count += 1;
        group.passes[0].executions.push(ArchitectureLayerExecution {
            node_id: node.into(),
            physical_layer_index,
        });
    }

    /// Retains invocation identities while projecting a shared stack's lowering.
    fn repeated_decoder(&mut self, root: &str, physical_layers: usize, passes: usize) {
        let group = &mut self.descriptor.layer_groups[0];
        let executions = std::mem::take(&mut group.passes[0].executions);
        assert_eq!(executions.len(), physical_layers * passes);
        group.physical_layer_count = physical_layers;
        group.weight_sharing = if passes > 1 {
            LayerWeightSharing::SharedAcrossPasses
        } else {
            LayerWeightSharing::None
        };
        group.passes = executions
            .chunks(physical_layers)
            .enumerate()
            .map(|(index, executions)| ArchitectureExecutionPass {
                index,
                executions: executions
                    .iter()
                    .enumerate()
                    .map(
                        |(physical_layer_index, execution)| ArchitectureLayerExecution {
                            node_id: execution.node_id.clone(),
                            physical_layer_index,
                        },
                    )
                    .collect(),
            })
            .collect();

        for group in &mut self.descriptor.components {
            group.shared_write_weight =
                crate::decoder::repeated::source_name(root, physical_layers, &group.write_weight);
            if let Some(projection) = &mut group.write_input_projection {
                projection.shared_weight = crate::decoder::repeated::source_name(
                    root,
                    physical_layers,
                    &projection.weight,
                );
            }
            for read in group
                .reads
                .iter_mut()
                .chain(group.output_gate.iter_mut().map(|gate| &mut gate.read))
            {
                read.shared_weight =
                    crate::decoder::repeated::source_name(root, physical_layers, &read.weight);
                for projection in &mut read.input_projections {
                    projection.shared_weight = crate::decoder::repeated::source_name(
                        root,
                        physical_layers,
                        &projection.weight,
                    );
                }
            }
        }

        // Use the same exact alias mapping as checkpoint lowering, including
        // inter-pass normalization. Logical prefixes and capture paths stay intact.
        for parameters in &mut self.descriptor.parameter_groups {
            let logical = format!("{}.weight", parameters.canonical_prefix);
            let source = crate::decoder::repeated::source_name(root, physical_layers, &logical);
            if source != logical {
                parameters.shared_with = Some(format!(
                    "parameters:{}",
                    source.strip_suffix(".weight").expect("weight module")
                ));
            }
        }
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

    fn projection_input_observation(&mut self, node: &str, path: String, shape: Vec<TensorAxis>) {
        self.read_only_observation(
            node,
            path,
            "Actual multiplication input after the selected projection input transform; read-only evidence",
            shape,
        );
    }

    fn read_only_observation(
        &mut self,
        node: &str,
        path: String,
        meaning: &str,
        shape: Vec<TensorAxis>,
    ) {
        self.get_mut(node).observation_paths.push(path.clone());
        self.descriptor.observations.points.push(ObservationPoint {
            path,
            node_id: node.into(),
            meaning: meaning.into(),
            value_type: ObservationValueType::Tensor,
            dtype: ObservationDtype::Floating,
            axes: Some(shape),
            prefill: true,
            decode: true,
            requirements: vec![ObservationRequirement::ActivationHooks],
            position: ObservationPosition::ReadOnly,
            retained_bytes: None,
            host_bytes: None,
        });
    }

    fn finish(mut self) -> ArchitectureDescriptor {
        routed_components::declare_captures(&mut self);
        let inputs: Vec<_> = self
            .descriptor
            .components
            .iter()
            .chain(
                self.descriptor
                    .component_scopes
                    .iter()
                    .flat_map(|scope| &scope.components),
            )
            .filter_map(|group| {
                group.write_input.as_ref().map(|path| {
                    let mut shape = axes(group.count);
                    shape[2].name = "component".into();
                    (group.node_id.clone(), path.clone(), shape)
                })
            })
            .collect();
        for (node, path, shape) in inputs {
            self.projection_input_observation(&node, path, shape);
        }
        let grouped: Vec<_> = self
            .descriptor
            .components
            .iter()
            .chain(
                self.descriptor
                    .component_scopes
                    .iter()
                    .flat_map(|scope| &scope.components),
            )
            .filter_map(|group| {
                group
                    .write_input_projection
                    .as_ref()
                    .map(|stage| (group.node_id.clone(), group.count, stage.clone()))
            })
            .collect();
        for (node, count, stage) in grouped {
            let Some(width) = stage.groups.checked_mul(stage.rank).filter(|n| *n > 0) else {
                self.catalog_partial("invalid grouped write extent");
                continue;
            };
            if count == 0 || !count.is_multiple_of(stage.groups) {
                self.catalog_partial("invalid grouped component extent");
                continue;
            }
            let mut input = axes(count / stage.groups);
            input[2].name = "channel".into();
            input.insert(
                1,
                TensorAxis {
                    name: "group".into(),
                    dimension: SymbolicDimension::Known(stage.groups),
                },
            );
            self.projection_input_observation(&node, stage.input, input);
            let mut output = axes(width);
            output[2].name = "projection".into();
            self.read_only_observation(
                &node,
                stage.output,
                "Actual concatenated grouped projection output before final input transformation",
                output.clone(),
            );
            self.projection_input_observation(&node, stage.final_input, output);
        }
        if let Some(readout) = &self.descriptor.component_readout {
            if let Some(path) = readout.projection_input.clone() {
                let shape = self
                    .descriptor
                    .observations
                    .get(&readout.normalized)
                    .and_then(|point| point.axes.clone())
                    .expect("declared readout geometry");
                self.projection_input_observation("output", path, shape);
            }
        }
        let mut scope_inputs = Vec::new();
        for scope in &self.descriptor.component_scopes {
            if let Some(path) = &scope.readout.projection_input {
                let point = self
                    .descriptor
                    .observations
                    .get(&scope.readout.linear_scores)
                    .expect("declared scoped head");
                let shape = self
                    .descriptor
                    .observations
                    .get(&scope.readout.normalized)
                    .and_then(|point| point.axes.clone())
                    .expect("declared scoped normalization");
                scope_inputs.push((point.node_id.clone(), path.clone(), shape));
            }
            match &scope.residual_base {
                eredu_core::component::ComponentResidualBase::Source { .. } => {}
                eredu_core::component::ComponentResidualBase::LinearFusion {
                    inputs,
                    projection_input,
                    output,
                    ..
                } => {
                    let point = self
                        .descriptor
                        .observations
                        .get(output)
                        .expect("declared fusion output");
                    let width = inputs
                        .iter()
                        .map(|input| input.columns.end)
                        .max()
                        .expect("fusion inputs");
                    scope_inputs.push((
                        point.node_id.clone(),
                        projection_input.clone(),
                        axes(width),
                    ));
                }
                eredu_core::component::ComponentResidualBase::ProjectedSum { inputs, .. } => {
                    for input in inputs {
                        let point = self
                            .descriptor
                            .observations
                            .get(&input.output)
                            .expect("declared projected fusion output");
                        let shape = self
                            .descriptor
                            .observations
                            .get(&input.normalized)
                            .and_then(|point| point.axes.clone())
                            .expect("declared projected fusion input geometry");
                        scope_inputs.push((
                            point.node_id.clone(),
                            input.projection_input.clone(),
                            shape,
                        ));
                    }
                }
            }
        }
        for (node, path, shape) in scope_inputs {
            self.projection_input_observation(&node, path, shape);
        }
        self.scope_requirements();
        self.descriptor
            .observations
            .points
            .sort_by(|a, b| a.path.cmp(&b.path));
        self.descriptor
    }

    fn scope_requirements(&mut self) {
        let scoped: std::collections::BTreeSet<_> = self
            .descriptor
            .nodes
            .iter()
            .filter(|node| {
                let mut current = Some(*node);
                while let Some(node) = current {
                    if let Some(binding) = self
                        .descriptor
                        .speculative_invocations
                        .iter()
                        .find(|binding| binding.node_id == node.id)
                    {
                        return binding.scope
                            != eredu_core::speculative::SpeculativeCaptureScope::Target;
                    }
                    if self
                        .descriptor
                        .component_scopes
                        .iter()
                        .any(|scope| scope.node_id == node.id)
                    {
                        return true;
                    }
                    current = node
                        .parent
                        .as_ref()
                        .and_then(|parent| self.descriptor.node(parent));
                }
                false
            })
            .map(|node| node.id.clone())
            .collect();
        for point in &mut self.descriptor.observations.points {
            if scoped.contains(&point.node_id)
                && !point
                    .requirements
                    .contains(&ObservationRequirement::PredictionExecution)
            {
                point
                    .requirements
                    .push(ObservationRequirement::PredictionExecution);
            }
        }
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
        sink_logits: None,
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
