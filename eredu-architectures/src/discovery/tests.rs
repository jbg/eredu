use super::*;
use eredu_core::component::{ComponentNormalizationKind, ComponentOutputTransform};
use std::collections::BTreeSet;

#[path = "tests/deepseek_v3.rs"]
mod compressed_v3;

#[path = "tests/deepseek_v4.rs"]
mod compressed_v4;

#[path = "tests/muse.rs"]
mod muse;

fn describe(json: serde_json::Value) -> ArchitectureDescriptor {
    crate::configuration::MODEL_CONFIGURATIONS
        .resolve_safetensors(&json)
        .unwrap()
        .architecture_plan()
        .architecture_descriptor()
}

#[test]
fn kimi_components_preserve_recurrent_equations_and_physical_latent_layout() {
    use eredu_core::component::*;
    for split in [false, true] {
        for query_rank in [None, Some(3)] {
            let mut args = crate::kimi_linear::model_args_from_config_value(&serde_json::json!({
                "model_type":"kimi_linear", "vocab_size":16, "hidden_size":8,
                "num_hidden_layers":2, "num_attention_heads":2, "num_key_value_heads":2,
                "intermediate_size":10, "head_dim":4, "model_max_length":64,
                "linear_attn_config":{"kda_layers":[1], "full_attn_layers":[2],
                    "num_heads":2, "head_dim":4, "short_conv_kernel_size":3},
                "num_experts":2, "moe_intermediate_size":6, "kv_lora_rank":4,
                "q_lora_rank":query_rank, "qk_nope_head_dim":4,
                "qk_rope_head_dim":2, "v_head_dim":4, "mla_use_nope":true,
                "num_experts_per_token":1, "num_shared_experts":1,
                "routed_scaling_factor":1.0, "tie_word_embeddings":false,
                "first_k_dense_replace":1, "num_expert_group":1, "topk_group":1
            }))
            .unwrap();
            args.split_kv_b = split;
            let mut builder = Builder::new();
            families::remaining_safetensors(
                &mut builder,
                &SafetensorsModelConfig::KimiLinear(args),
            );
            let graph = builder.finish();
            validate(&graph);
            assert_eq!(graph.components.len(), 4);
            assert_eq!(graph.component_transforms.len(), 3);
            let recurrent = &graph.components[0];
            assert!(matches!(
                recurrent.activation_equation,
                ComponentActivation::GatedDeltaAttention {
                    key_heads: 2,
                    value_heads: 2,
                    key_head_width: 4,
                    value_head_width: 4,
                    ..
                }
            ));
            let latent = graph
                .components
                .iter()
                .find(|g| {
                    g.layer_index == 1
                        && matches!(g.activation_equation, ComponentActivation::Attention { .. })
                })
                .unwrap();
            assert_eq!(
                latent.reads[0].input_projections.len(),
                usize::from(query_rank.is_some())
            );
            assert!(latent.reads[1].weight.ends_with(if split {
                ".k_b_proj.weight"
            } else {
                ".kv_b_proj.weight"
            }));
            assert!(latent.reads[3].weight.ends_with(if split {
                ".v_b_proj.weight"
            } else {
                ".kv_b_proj.weight"
            }));
            for group in &graph.components {
                assert!(graph.observations.get(&group.input).is_some());
                assert!(graph.observations.get(&group.activation).is_some());
                assert!(graph
                    .observations
                    .get(&group.effective_activation)
                    .is_some());
                for read in &group.reads {
                    assert!(graph
                        .parameter_groups
                        .iter()
                        .any(|g| g.id == read.parameter_group));
                    for stage in &read.input_projections {
                        assert!(graph.observations.get(&stage.output).is_some());
                    }
                }
            }
            let other = &graph.component_readout.as_ref().unwrap().other_writes;
            assert_eq!(other.len(), 1);
            assert_eq!(other[0].output, "model.layers.1.feed_forward.contribution");
        }
    }
}

#[test]
fn nemotron_prediction_scopes_preserve_physical_schedules_and_parameter_roots() {
    use eredu_core::component::*;
    for pattern in ["*", "*E", "*E*E"] {
        for tied in [false, true] {
            let graph = describe(serde_json::json!({
                "model_type":"nemotron_h", "vocab_size":32, "hidden_size":16,
                "intermediate_size":24, "num_hidden_layers":1, "hybrid_override_pattern":"*",
                "num_attention_heads":4, "num_key_value_heads":2, "head_dim":4,
                "mamba_num_heads":4, "mamba_head_dim":4, "ssm_state_size":3,
                "n_groups":2, "conv_kernel":3, "n_routed_experts":4, "n_group":2,
                "topk_group":1, "num_experts_per_tok":2, "moe_intermediate_size":6,
                "n_shared_experts":1, "moe_shared_expert_intermediate_size":8,
                "num_nextn_predict_layers":2, "mtp_hybrid_override_pattern":pattern,
                "tie_word_embeddings":tied, "attention_bias":true, "mlp_bias":true
            }));
            validate(&graph);
            assert_eq!(graph.component_scopes.len(), 2);
            assert_eq!(
                graph.observations.completeness,
                DescriptionCompleteness::Complete
            );
            assert_eq!(
                graph.components.len(),
                1,
                "prediction components have separate scopes"
            );
            let mut identities = BTreeSet::new();
            for (depth, scope) in graph.component_scopes.iter().enumerate() {
                let start = depth * pattern.len();
                let end = (depth + 1) * pattern.len() - 1;
                let first = format!("model.mtp.layers.{start}");
                assert_eq!(
                    scope.kind,
                    ComponentExecutionScopeKind::Prediction { depth }
                );
                assert_eq!(scope.components.len(), pattern.len());
                assert_eq!(scope.routed_components.len(), pattern.matches('E').count());
                assert_eq!(
                    scope.readout.other_writes.len(),
                    pattern.matches('E').count()
                );
                assert_eq!(
                    scope.readout.weight,
                    if tied {
                        "model.embeddings.weight"
                    } else {
                        "lm_head.weight"
                    }
                );
                assert_eq!(
                    scope.readout.normalization.gain,
                    Some(format!("model.mtp.layers.{end}.final_layernorm.weight"))
                );
                let ComponentResidualBase::LinearFusion {
                    inputs,
                    weight,
                    projection_input,
                    effective_output,
                    ..
                } = &scope.residual_base
                else {
                    panic!("prediction fusion")
                };
                assert_eq!(weight, &format!("{first}.eh_proj.weight"));
                assert_eq!(inputs[0].columns, 0..16);
                assert_eq!(inputs[1].columns, 16..32);
                assert_eq!(
                    inputs[0].normalization.gain,
                    Some(format!("{first}.enorm.weight"))
                );
                assert_eq!(
                    inputs[1].normalization.gain,
                    Some(format!("{first}.hnorm.weight"))
                );
                let group = graph
                    .layer_groups
                    .iter()
                    .find(|group| group.id == format!("mtp.{depth}"))
                    .unwrap();
                assert_eq!(group.physical_layer_count, pattern.len());
                assert_eq!(group.passes[0].executions.len(), pattern.len());
                for (relative, component) in scope.components.iter().enumerate() {
                    assert!(identities.insert(&component.id));
                    assert_eq!(component.layer_index, 1 + start + relative);
                    let root = format!("model.mtp.layers.{}.mixer", start + relative);
                    assert!(component
                        .reads
                        .iter()
                        .all(|read| read.weight.starts_with(&root)));
                    assert!(component.write_weight.starts_with(&root));
                    assert!(component.write_bias.is_some());
                    for point in [&component.activation, &component.effective_activation] {
                        assert_eq!(
                            crate::speculative_execution::speculative_capture_scope(
                                &graph,
                                &graph.observations.get(point).unwrap().node_id
                            )
                            .unwrap(),
                            eredu_runtime::capture::SpeculativeCaptureScope::Prediction { depth }
                        );
                    }
                }
                for point in [
                    projection_input,
                    effective_output,
                    &scope.readout.residual,
                    &scope.readout.normalized,
                    scope.readout.projection_input.as_ref().unwrap(),
                    &scope.readout.linear_scores,
                    &scope.readout.logits,
                ] {
                    assert!(graph.observations.get(point).is_some(), "missing {point}");
                }
            }
        }
    }
}

#[test]
fn routed_topology_declares_gated_and_relu_squared_parameter_coordinates() {
    use eredu_core::component::*;
    let qwen = describe(serde_json::json!({
        "model_type":"qwen3_moe", "hidden_size":16, "num_hidden_layers":2,
        "intermediate_size":0, "moe_intermediate_size":6, "num_experts":4,
        "num_experts_per_tok":2, "num_attention_heads":4, "num_key_value_heads":2,
        "head_dim":4, "vocab_size":32, "rms_norm_eps":1e-5,
        "max_position_embeddings":128, "norm_topk_prob":true, "tie_word_embeddings":false
    }));
    validate(&qwen);
    assert_eq!(qwen.routed_components.len(), 2);
    for group in &qwen.routed_components {
        assert_eq!(group.units_per_expert, 6);
        assert_eq!(
            group.reads.iter().map(|r| r.role).collect::<Vec<_>>(),
            [ComponentReadRole::Gate, ComponentReadRole::Value]
        );
        assert_eq!(group.reads[1].rows.row(5), Some(11));
        let RoutedComponentParameter::Packed { name } = &group.write_weight else {
            panic!("packed bank")
        };
        assert_eq!(
            name.parameter,
            format!("model.layers.{}.mlp.experts.down_proj", group.layer_index)
        );
    }
    let relu = describe(serde_json::json!({
        "model_type":"nemotron_h", "vocab_size":32, "hidden_size":16,
        "intermediate_size":24, "num_hidden_layers":4, "hybrid_override_pattern":"M*-E",
        "num_attention_heads":4, "num_key_value_heads":2, "head_dim":4,
        "mamba_num_heads":4, "mamba_head_dim":4, "ssm_state_size":3,
        "n_groups":2, "conv_kernel":3, "n_routed_experts":4,
        "n_group":2, "topk_group":1,
        "num_experts_per_tok":2, "moe_intermediate_size":6, "n_shared_experts":1,
        "moe_shared_expert_intermediate_size":8
    }));
    validate(&relu);
    assert_eq!(relu.routed_components.len(), 1);
    let group = &relu.routed_components[0];
    assert_eq!(
        group.activation_equation,
        ComponentActivation::Unary {
            activation: ComponentNonlinearity::ReluSquared
        }
    );
    assert_eq!(group.reads.len(), 1);
    assert_eq!(group.reads[0].role, ComponentReadRole::Input);
    assert_eq!(group.reads[0].rows.row(5), Some(5));
    assert_eq!(group.reads[0].projection_rows, 6);
    let shared = relu
        .components
        .iter()
        .find(|group| group.node_id == "decoder.layers.3.operator.shared")
        .unwrap();
    assert_eq!(shared.count, 8);
    assert_eq!(shared.activation_equation, group.activation_equation);
    assert_eq!(
        shared.activation,
        "model.layers.3.shared.feed_forward.units"
    );
    assert_eq!(
        shared.input_normalization.gain.as_deref(),
        Some("model.layers.3.norm.weight")
    );
    assert_eq!(shared.reads.len(), 1);
    assert_eq!(shared.reads[0].role, ComponentReadRole::Input);
    assert_eq!(shared.reads[0].rows.row(7), Some(7));
    assert_eq!(
        shared.reads[0].weight,
        "model.layers.3.moe.shared_experts.up_proj.weight"
    );
    assert_eq!(
        shared.write_weight,
        "model.layers.3.moe.shared_experts.down_proj.weight"
    );
    assert_eq!(
        shared.write_parameter_group,
        "parameters:model.layers.3.moe"
    );
    assert!(relu
        .parameter_groups
        .iter()
        .any(|p| p.id == shared.write_parameter_group));
    // The sparse write already includes this constituent; readout adds it once.
    let readout = relu.component_readout.as_ref().unwrap();
    assert!(readout.other_writes.iter().any(|w| w.layer_index == 3));
}

fn validate(graph: &ArchitectureDescriptor) {
    let ids = graph
        .nodes
        .iter()
        .map(|n| n.id.as_str())
        .collect::<BTreeSet<_>>();
    assert_eq!(ids.len(), graph.nodes.len());
    let mut component_ids = graph
        .components
        .iter()
        .map(|g| g.id.as_str())
        .collect::<BTreeSet<_>>();
    for group in graph.routed_components.iter().chain(
        graph
            .component_scopes
            .iter()
            .flat_map(|scope| &scope.routed_components),
    ) {
        assert!(component_ids.insert(&group.id));
        assert!(ids.contains(group.node_id.as_str()));
        assert!(group.expert_count > 0 && group.units_per_expert > 0);
        assert!(
            matches!(
                graph
                    .observations
                    .get(&group.activation)
                    .map(|p| &p.value_type),
                Some(ObservationValueType::RoutedUnits { .. })
            ),
            "routed topology declares its sparse value contract"
        );
        let mut parameters = vec![&group.write_weight];
        parameters.extend(group.write_bias.iter());
        for read in &group.reads {
            parameters.push(&read.weight);
            parameters.extend(read.bias.iter());
        }
        for parameter in parameters {
            use eredu_core::component::RoutedComponentParameter;
            let names: Vec<_> = match parameter {
                RoutedComponentParameter::Packed { name } => vec![name],
                RoutedComponentParameter::Independent { names } => {
                    assert_eq!(names.len(), group.expert_count);
                    names.iter().flatten().collect()
                }
            };
            for name in names {
                assert!(graph
                    .parameter_groups
                    .iter()
                    .any(|g| g.id == name.parameter_group));
                assert!(!name.parameter.is_empty() && !name.shared_parameter.is_empty());
            }
        }
    }
    for node in &graph.nodes {
        if let Some(parent) = &node.parent {
            assert!(ids.contains(parent.as_str()));
        }
        for group in &node.parameter_groups {
            assert!(graph.parameter_groups.iter().any(|g| &g.id == group));
        }
        for path in &node.observation_paths {
            assert_eq!(graph.observations.get(path).unwrap().node_id, node.id);
        }
    }
    for group in &graph.parameter_groups {
        if let Some(shared) = &group.shared_with {
            assert_ne!(shared, &group.id);
            let source = graph
                .parameter_groups
                .iter()
                .find(|g| &g.id == shared)
                .unwrap();
            assert!(
                source.shared_with.is_none(),
                "sharing points directly to its source"
            );
        }
    }
    let mut executions = BTreeSet::new();
    for group in &graph.layer_groups {
        for (index, pass) in group.passes.iter().enumerate() {
            assert_eq!(pass.index, index);
            assert_eq!(pass.executions.len(), group.physical_layer_count);
            for (physical, execution) in pass.executions.iter().enumerate() {
                assert_eq!(execution.physical_layer_index, physical);
                let node = graph.node(&execution.node_id).unwrap();
                assert_eq!(node.kind, ArchitectureNodeKind::DecoderBlock);
                assert!(
                    executions.insert(&node.id),
                    "execution belongs to exactly one pass"
                );
            }
        }
    }
    assert_eq!(
        executions.len(),
        graph
            .nodes
            .iter()
            .filter(|n| n.layer_index.is_some())
            .count()
    );
    for edge in &graph.edges {
        assert!(ids.contains(edge.from.as_str()));
        assert!(ids.contains(edge.to.as_str()));
    }
    // Resolve the actual data-flow graph, independent of its vector/file ordering.
    let mut visited = BTreeSet::new();
    loop {
        let before = visited.len();
        for node in &graph.nodes {
            if graph
                .edges
                .iter()
                .filter(|e| e.to == node.id)
                .all(|e| visited.contains(&e.from))
            {
                visited.insert(node.id.clone());
            }
        }
        if before == visited.len() {
            break;
        }
    }
    assert_eq!(visited.len(), ids.len(), "logical graph contains a cycle");
    let paths = graph
        .observations
        .points
        .iter()
        .map(|p| &p.path)
        .collect::<BTreeSet<_>>();
    assert_eq!(paths.len(), graph.observations.points.len());
    for point in &graph.observations.points {
        assert!(ids.contains(point.node_id.as_str()));
        assert!(point.prefill || point.decode);
        assert_eq!(point.host_bytes, None);
    }
    let encoded = serde_json::to_string(graph).unwrap();
    assert_eq!(*graph, serde_json::from_str(&encoded).unwrap());
}

#[test]
fn dense_graph_has_bypass_edges_tied_parameters_and_only_emitted_points() {
    let graph = describe(serde_json::json!({
        "model_type":"llama", "hidden_size":16, "num_hidden_layers":2,
        "intermediate_size":32, "num_attention_heads":4, "num_key_value_heads":2,
        "head_dim":4, "rms_norm_eps":1e-6, "vocab_size":32, "tie_word_embeddings":true,
        "max_position_embeddings":128, "rope_theta":10000.0
    }));
    validate(&graph);
    assert_eq!(graph.completeness, DescriptionCompleteness::Complete);
    assert_eq!(graph.components.len(), 4);
    let readout = graph.component_readout.as_ref().unwrap();
    assert!(readout.tied_embeddings);
    assert_eq!(readout.weight, readout.embedding_weight);
    assert_eq!(readout.normalization.kind, ComponentNormalizationKind::Rms);
    assert_eq!(readout.normalization.epsilon.value(), 1e-6);
    assert_eq!(readout.output_transform, ComponentOutputTransform::Identity);
    for path in [
        &readout.embedding,
        &readout.residual,
        &readout.normalized,
        &readout.linear_scores,
    ] {
        assert_eq!(
            graph.observations.get(path).unwrap().position,
            ObservationPosition::BeforeIntervention
        );
        assert_eq!(
            graph
                .observations
                .get(&format!("{path}.effective"))
                .unwrap()
                .position,
            ObservationPosition::AfterIntervention
        );
    }
    let group = &graph.layer_groups[0];
    assert_eq!(group.physical_layer_count, 2);
    assert_eq!(group.passes.len(), 1);
    assert_eq!(group.weight_sharing, LayerWeightSharing::None);
    assert_eq!(
        graph.node("embedding").unwrap().parameter_groups,
        graph.node("output").unwrap().parameter_groups
    );
    assert!(graph.edges.iter().any(|e| e.from == "decoder.layers.0"
        && e.to == "decoder.layers.0.attention.residual"
        && e.kind == ArchitectureEdgeKind::Residual));
    assert_eq!(
        graph
            .node("decoder.layers.0.attention")
            .unwrap()
            .attention
            .as_ref()
            .unwrap()
            .head_sharing,
        Some(HeadSharing::GroupedQuery)
    );
    assert!(graph
        .observations
        .get("model.layers.0.self_attn.probabilities")
        .is_none());
    assert_eq!(
        graph
            .observations
            .get("model.layers.0.output")
            .unwrap()
            .position,
        ObservationPosition::BeforeIntervention
    );
}

#[test]
fn nanbeige_groups_physical_layers_and_passes_without_collapsing_execution_identity() {
    let published: serde_json::Value = serde_json::from_str(include_str!(
        "../../tests/fixtures/configs/nanbeige4.2-3b-0e137298.json"
    ))
    .unwrap();
    for passes in [1, 2, 3] {
        for skip_norm in [false, true] {
            let mut config = published.clone();
            config["num_loops"] = passes.into();
            config["skip_loop_final_norm"] = skip_norm.into();
            let graph = describe(config);
            validate(&graph);
            assert_eq!(graph.schema_version, ARCHITECTURE_DESCRIPTOR_SCHEMA_VERSION);
            assert_eq!(graph.observations.schema_version, DISCOVERY_SCHEMA_VERSION);
            assert_eq!(graph.layer_groups.len(), 1);
            let group = &graph.layer_groups[0];
            assert_eq!(group.physical_layer_count, 22);
            assert_eq!(group.passes.len(), passes);
            assert_eq!(
                group.weight_sharing,
                if passes > 1 {
                    LayerWeightSharing::SharedAcrossPasses
                } else {
                    LayerWeightSharing::None
                }
            );
            for pass in &group.passes {
                for execution in &pass.executions {
                    let physical = execution.physical_layer_index;
                    let logical = pass.index * 22 + physical;
                    let node = graph.node(&execution.node_id).unwrap();
                    assert_eq!(node.layer_index, Some(logical));
                    assert_eq!(
                        node.observation_paths,
                        [
                            format!("model.layers.{logical}.input"),
                            format!("model.layers.{logical}.input.effective"),
                            format!("model.layers.{logical}.output"),
                            format!("model.layers.{logical}.output.effective"),
                        ]
                    );
                    for suffix in [
                        "",
                        ".self_attn",
                        ".mlp",
                        ".input_layernorm",
                        ".post_attention_layernorm",
                    ] {
                        let parameters = graph
                            .parameter_groups
                            .iter()
                            .find(|g| {
                                g.canonical_prefix == format!("model.layers.{logical}{suffix}")
                            })
                            .unwrap();
                        assert_eq!(
                            parameters.shared_with,
                            (pass.index > 0)
                                .then(|| format!("parameters:model.layers.{physical}{suffix}"))
                        );
                    }
                }
            }
            for pass in 1..passes {
                let norm = graph.node(&format!("decoder.layers.{}.output_norm", pass * 22 - 1));
                if skip_norm {
                    assert!(norm.is_none());
                } else {
                    let parameters = &norm.unwrap().parameter_groups[0];
                    let parameters = graph
                        .parameter_groups
                        .iter()
                        .find(|g| &g.id == parameters)
                        .unwrap();
                    assert_eq!(
                        parameters.shared_with.as_deref(),
                        Some("parameters:model.norm")
                    );
                }
            }
        }
    }
}

#[test]
fn older_descriptors_leave_layer_structure_and_parameter_sharing_undeclared() {
    let graph = describe(
        serde_json::from_str(include_str!(
            "../../tests/fixtures/configs/nanbeige4.2-3b-0e137298.json"
        ))
        .unwrap(),
    );
    let mut legacy = serde_json::to_value(&graph).unwrap();
    legacy["schema_version"] = 1.into();
    legacy.as_object_mut().unwrap().remove("layer_groups");
    legacy.as_object_mut().unwrap().remove("component_scopes");
    for group in legacy["parameter_groups"].as_array_mut().unwrap() {
        group.as_object_mut().unwrap().remove("shared_with");
    }
    let decoded: ArchitectureDescriptor = serde_json::from_value(legacy).unwrap();
    assert!(decoded.layer_groups.is_empty());
    assert!(decoded.component_scopes.is_empty());
    assert_eq!(decoded.component_readout, graph.component_readout);
    assert!(decoded
        .parameter_groups
        .iter()
        .all(|g| g.shared_with.is_none()));
    assert_eq!(decoded.nodes, graph.nodes);
    assert_eq!(decoded.observations, graph.observations);
    assert_eq!(decoded.schema_version, 1);
}

#[test]
fn gqa_and_sliding_attention_are_independent_with_per_layer_schedules() {
    let graph = describe(serde_json::json!({
        "model_type":"gpt_oss", "hidden_size":32,"intermediate_size":32,
        "num_hidden_layers":2,"num_attention_heads":4,"num_key_value_heads":2,
        "head_dim":8,"vocab_size":32,"num_local_experts":4,"num_experts_per_tok":2,
        "sliding_window":16,"max_position_embeddings":64,"rms_norm_eps":1e-5,
        "quantization_config":{"quant_method":"mxfp4"}
    }));
    validate(&graph);
    assert_eq!(graph.routed_components.len(), 2);
    for group in &graph.routed_components {
        assert!(group.write_bias.is_some());
        assert!(group.reads.iter().all(|read| read.bias.is_some()));
        let eredu_core::component::ComponentActivation::Gated {
            gate_upper_bound,
            value_absolute_bound,
            value_offset,
            ..
        } = &group.activation_equation
        else {
            panic!("gated experts")
        };
        assert_eq!(gate_upper_bound.unwrap().value(), 7.0);
        assert_eq!(value_absolute_bound.unwrap().value(), 7.0);
        assert_eq!(value_offset.value(), 1.0);
    }
    let first = graph
        .node("decoder.layers.0.attention")
        .unwrap()
        .attention
        .as_ref()
        .unwrap();
    let second = graph
        .node("decoder.layers.1.attention")
        .unwrap()
        .attention
        .as_ref()
        .unwrap();
    assert_eq!(first.head_sharing, Some(HeadSharing::GroupedQuery));
    assert_eq!(
        first.receptive_field,
        Some(ReceptiveField::Sliding { window: 16 })
    );
    assert_eq!(second.receptive_field, Some(ReceptiveField::Full));
    assert_eq!(
        graph
            .node("decoder.layers.0.feed_forward")
            .unwrap()
            .moe
            .as_ref()
            .unwrap()
            .score_transform,
        Some(RoutingScoreTransform::SelectedSoftmax)
    );
}

#[test]
fn hybrid_shared_experts_are_parallel_and_backend_support_does_not_change_graph() {
    let graph = describe(serde_json::json!({
        "model_type":"qwen3_next", "vocab_size":16,"hidden_size":8,
        "num_hidden_layers":2,"intermediate_size":12,"num_attention_heads":4,
        "num_key_value_heads":2,"head_dim":2,"max_position_embeddings":64,
        "linear_conv_kernel_dim":3,"linear_key_head_dim":2,"linear_value_head_dim":2,
        "linear_num_key_heads":2,"linear_num_value_heads":2,"num_experts":4,
        "moe_intermediate_size":6,"shared_expert_intermediate_size":8,
        "num_experts_per_tok":2,"norm_topk_prob":true,
        "layer_types":["linear_attention","full_attention"],"tie_word_embeddings":false
    }));
    validate(&graph);
    assert_eq!(graph.routed_components.len(), 2);
    for group in &graph.routed_components {
        assert_eq!((group.expert_count, group.units_per_expert), (4, 6));
        assert_eq!(group.reads[1].rows.row(3), Some(9));
        assert!(group.write_bias.is_none());
        assert_eq!(
            group
                .input_normalization
                .as_ref()
                .unwrap()
                .gain_offset
                .value(),
            1.0
        );
    }
    let linear = graph.node("decoder.layers.0.mixer").unwrap();
    assert_eq!(
        linear.mixer.as_ref().unwrap().mechanism,
        MixerMechanism::GatedDelta
    );
    assert_eq!(
        linear.attention.as_ref().unwrap().mechanism,
        Some(AttentionMechanism::Linear)
    );
    let moe = graph
        .node("decoder.layers.0.feed_forward")
        .unwrap()
        .moe
        .as_ref()
        .unwrap();
    assert_eq!(moe.shared_experts, Some(1));
    assert_eq!(moe.shared_expert_gated, Some(true));
    let sum = "decoder.layers.0.feed_forward.sum";
    assert_eq!(graph.edges.iter().filter(|e| e.to == sum).count(), 2);
    assert!(graph
        .observations
        .get("model.layers.0.mlp.routing.shared_output")
        .is_some());
    let before = graph.clone();
    use eredu_runtime::inspection::{observation_support, ObservationExecutionContext};
    let context = ObservationExecutionContext {
        activation_inspection: true,
        selected: true,
        prediction_inspection: false,
        partitioned: false,
        mechanisms: ObservationMechanisms {
            activation_tensors: true,
            routing_tensors: true,
            routed_unit_tensors: true,
            floating_to_f32: true,
        },
    };
    let supported = observation_support(&graph.observations, context);
    assert!(supported
        .points
        .iter()
        .all(|p| p.prefill == ObservationSupportStatus::Supported
            && p.decode == ObservationSupportStatus::Supported));
    let unsupported = observation_support(
        &graph.observations,
        ObservationExecutionContext {
            activation_inspection: false,
            ..context
        },
    );
    assert!(unsupported
        .points
        .iter()
        .all(|p| matches!(p.prefill, ObservationSupportStatus::Unsupported(_))));
    let no_routing = observation_support(
        &graph.observations,
        ObservationExecutionContext {
            mechanisms: ObservationMechanisms {
                routing_tensors: false,
                routed_unit_tensors: true,
                ..context.mechanisms
            },
            ..context
        },
    );
    assert!(no_routing
        .points
        .iter()
        .any(|p| matches!(p.prefill, ObservationSupportStatus::Unsupported(_))));
    assert_eq!(graph, before);
}

#[test]
fn unknown_architecture_details_and_execution_support_are_explicit() {
    let graph = describe(serde_json::json!({
        "model_type":"gemma4", "text_config": {"model_type":"gemma4_text", "hidden_size":16,"num_hidden_layers":2,
        "intermediate_size":32,"num_attention_heads":2,"rms_norm_eps":1e-6,
        "vocab_size":64,"num_key_value_heads":1,"max_position_embeddings":128,
        "head_dim":8,"layer_types":["sliding_attention","full_attention"],"sliding_window":16},
        "image_token_id":4,
        "vision_config":{"hidden_size":16,"intermediate_size":32,"num_hidden_layers":1,
            "num_attention_heads":2,"num_key_value_heads":2,"head_dim":8,"patch_size":4,
            "pooling_kernel_size":2,"position_embedding_size":16,"rms_norm_eps":1e-6}
    }));
    validate(&graph);
    assert!(matches!(
        graph.completeness,
        DescriptionCompleteness::Partial(_)
    ));
    assert!(matches!(
        graph.observations.completeness,
        DescriptionCompleteness::Partial(_)
    ));
    let support = eredu_runtime::inspection::observation_support(
        &graph.observations,
        eredu_runtime::inspection::ObservationExecutionContext {
            activation_inspection: false,
            selected: false,
            prediction_inspection: false,
            partitioned: false,
            mechanisms: Default::default(),
        },
    );
    assert!(support
        .points
        .iter()
        .all(|p| matches!(p.prefill, ObservationSupportStatus::Unverified(_))));
    let media = graph
        .observations
        .get(VISION_PROJECTOR_OUTPUT_OBSERVATION_PATH)
        .unwrap();
    assert!(media.prefill);
    assert!(!media.decode);
    assert!(media
        .requirements
        .contains(&ObservationRequirement::MediaInput));
    let supported = eredu_runtime::inspection::observation_support(
        &graph.observations,
        eredu_runtime::inspection::ObservationExecutionContext {
            activation_inspection: true,
            selected: true,
            prediction_inspection: false,
            partitioned: false,
            mechanisms: ObservationMechanisms {
                activation_tensors: true,
                routing_tensors: true,
                routed_unit_tensors: true,
                floating_to_f32: true,
            },
        },
    );
    let media = supported
        .points
        .iter()
        .find(|p| p.path == VISION_PROJECTOR_OUTPUT_OBSERVATION_PATH)
        .unwrap();
    assert!(matches!(
        media.prefill,
        ObservationSupportStatus::Conditional(_)
    ));
    assert!(matches!(
        media.decode,
        ObservationSupportStatus::Unsupported(_)
    ));
}

#[test]
fn components_join_gated_reads_gqa_rows_and_separate_effective_observations() {
    use eredu_core::component::*;
    let graph = describe(serde_json::json!({
        "model_type":"qwen2", "hidden_size":16, "num_hidden_layers":2,
        "intermediate_size":32, "num_attention_heads":4, "num_key_value_heads":2,
        "head_dim":4, "rms_norm_eps":1e-6, "vocab_size":32, "tie_word_embeddings":true,
        "max_position_embeddings":128, "rope_theta":10000.0
    }));
    assert_eq!(graph.components.len(), 4);
    for group in &graph.components {
        assert!(graph.node(&group.node_id).is_some());
        assert!(graph
            .parameter_groups
            .iter()
            .any(|g| g.id == group.write_parameter_group));
        assert_eq!(
            graph
                .observations
                .get(&group.effective_activation)
                .unwrap()
                .position,
            ObservationPosition::AfterIntervention
        );
        assert_eq!(
            graph.observations.get(&group.activation).unwrap().position,
            ObservationPosition::BeforeIntervention
        );
        for read in &group.reads {
            assert!(graph
                .parameter_groups
                .iter()
                .any(|g| g.id == read.parameter_group));
        }
    }
    let attn = &graph.components[0];
    let values = attn
        .reads
        .iter()
        .find(|r| r.role == ComponentReadRole::Value)
        .unwrap();
    assert_eq!(
        (0..16)
            .map(|i| values.rows.row(i).unwrap())
            .collect::<Vec<_>>(),
        vec![0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7, 4, 5, 6, 7]
    );
    assert!(values.bias.is_some());
    for (role, sharing) in [(ComponentReadRole::Query, 1), (ComponentReadRole::Key, 2)] {
        let read = attn.reads.iter().find(|read| read.role == role).unwrap();
        for component in 0..attn.count {
            let head = component / 4 / sharing;
            assert_eq!(read.rows.row_range(component), Some(head * 4..head * 4 + 4));
            assert_eq!(read.rows.row(component), None);
        }
    }
    let ffn = &graph.components[1];
    assert_eq!(ffn.count, 32);
    assert_eq!(
        ffn.reads.iter().map(|r| r.role).collect::<Vec<_>>(),
        vec![ComponentReadRole::Gate, ComponentReadRole::Value]
    );
    assert!(matches!(
        ffn.activation_equation,
        ComponentActivation::Gated { .. }
    ));
}

#[test]
fn qwen_prediction_scopes_keep_shared_fusion_and_distinct_decoder_invocations() {
    use eredu_core::component::*;
    for model in ["qwen3_next", "qwen3_5_text", "qwen3_5_moe_text"] {
        for tied in [false, true] {
            let graph = describe(serde_json::json!({
                "model_type":model, "vocab_size":16,"hidden_size":8,
                "num_hidden_layers":2,"intermediate_size":12,"num_attention_heads":4,
                "num_key_value_heads":2,"head_dim":2,"max_position_embeddings":64,
                "linear_conv_kernel_dim":3,"linear_key_head_dim":2,"linear_value_head_dim":2,
                "linear_num_key_heads":2,"linear_num_value_heads":2,"num_experts":4,
                "moe_intermediate_size":6,"shared_expert_intermediate_size":8,
                "num_experts_per_tok":2,"norm_topk_prob":true,
                "layer_types":["linear_attention","full_attention"],
                "tie_word_embeddings":tied, "mtp_num_hidden_layers":2
            }));
            validate(&graph);
            assert_eq!(graph.component_scopes.len(), 2);
            assert_eq!(
                graph.observations.completeness,
                DescriptionCompleteness::Complete
            );
            let mut identities = BTreeSet::new();
            for (depth, scope) in graph.component_scopes.iter().enumerate() {
                let path = format!("mtp.layers.{depth}");
                assert_eq!(
                    scope.kind,
                    ComponentExecutionScopeKind::Prediction { depth }
                );
                assert_eq!(scope.components.len(), 2);
                assert_eq!(
                    scope.routed_components.len(),
                    usize::from(model != "qwen3_5_text")
                );
                assert_eq!(
                    scope.readout.weight,
                    if tied {
                        "model.embed_tokens.weight"
                    } else {
                        "lm_head.weight"
                    }
                );
                assert_eq!(
                    scope.readout.normalization.gain.as_deref(),
                    Some("mtp.norm.weight")
                );
                assert_eq!(scope.readout.normalization.gain_offset.value(), 1.0);
                let ComponentResidualBase::LinearFusion {
                    inputs,
                    weight,
                    shared_weight,
                    parameter_group,
                    projection_input,
                    output,
                    effective_output,
                    ..
                } = &scope.residual_base
                else {
                    panic!("fused residual")
                };
                assert_eq!(weight, "mtp.fc.weight");
                assert_eq!(weight, shared_weight);
                assert_eq!(parameter_group, "parameters:mtp.fc");
                assert_eq!(inputs[0].columns, 0..8);
                assert_eq!(inputs[1].columns, 8..16);
                assert_eq!(
                    inputs[0].normalization.gain.as_deref(),
                    Some("mtp.pre_fc_norm_embedding.weight")
                );
                assert_eq!(
                    inputs[1].normalization.gain.as_deref(),
                    Some("mtp.pre_fc_norm_hidden.weight")
                );
                for component in &scope.components {
                    assert!(identities.insert(&component.id));
                    assert_eq!(component.layer_index, 2 + depth);
                    for point in [&component.activation, &component.effective_activation] {
                        assert_eq!(
                            crate::speculative_execution::speculative_capture_scope(
                                &graph,
                                &graph.observations.get(point).unwrap().node_id
                            )
                            .unwrap(),
                            eredu_runtime::capture::SpeculativeCaptureScope::Prediction { depth }
                        );
                    }
                }
                for point in [
                    projection_input,
                    output,
                    effective_output,
                    &scope.readout.residual,
                    &scope.readout.normalized,
                    scope.readout.projection_input.as_ref().unwrap(),
                    &scope.readout.linear_scores,
                    &scope.readout.logits,
                ] {
                    assert!(
                        graph
                            .observations
                            .get(point)
                            .unwrap()
                            .requirements
                            .contains(&ObservationRequirement::PredictionExecution),
                        "{point}"
                    );
                }
                assert_eq!(
                    scope.components[0].activation,
                    format!("{path}.attention.channels")
                );
            }
            assert_eq!(
                graph
                    .parameter_groups
                    .iter()
                    .filter(|p| p.canonical_prefix == "mtp.fc")
                    .count(),
                1
            );
        }
    }
}

#[test]
fn inkling_prediction_scopes_preserve_static_ownership_and_repeated_normalization() {
    use eredu_core::capture::CaptureInvocationBounds;
    use eredu_core::component::*;
    use eredu_core::intervention::*;
    for chain_norm in [false, true] {
        let configuration = crate::configuration::MODEL_CONFIGURATIONS.resolve_safetensors(&serde_json::json!({
            "model_type": "inkling_mm_model", "image_token_id": 5,
            "text_config": {"hidden_size": 8, "num_hidden_layers": 2, "vocab_size": 19,
                "num_attention_heads": 2, "num_key_value_heads": 2, "head_dim": 4,
                "sliding_window_size": 4, "layer_types": ["full_attention", "sliding_attention"],
                "mlp_layer_types": ["dense", "moe"], "sconv_kernel_size": 3, "d_rel": 2,
                "rel_extent": 8, "intermediate_size": 12, "dense_intermediate_size": 12,
                "moe_intermediate_size": 6, "n_routed_experts": 4, "num_experts_per_tok": 2,
                "n_shared_experts": 1, "unpadded_vocab_size": 13},
            "mtp_config": {"num_nextn_predict_layers": 2, "local_layer_ids": [1],
                "chain_hidden_post_norm": chain_norm}
        })).unwrap();
        let architecture = configuration.architecture_plan();
        let graph = architecture.architecture_descriptor();
        validate(&graph);
        let discovery = InterventionDiscovery {
            schema_version: INTERVENTION_SCHEMA_VERSION,
            artifact_identity: "neutral-inkling-source".into(),
            session_identity: None,
            points: architecture.intervention_points(),
        };
        let bounds = CaptureInvocationBounds {
            batch: 1,
            max_sequence: 3,
            max_context: None,
            max_predictions: 4,
        };
        for component in graph.components.iter().chain(
            graph
                .component_scopes
                .iter()
                .flat_map(|scope| &scope.components),
        ) {
            for keep_selected in [false, true] {
                let plan = InterventionPlan {
                    schema_version: INTERVENTION_SCHEMA_VERSION,
                    operations: vec![InterventionOperation {
                        id: "component-mask".into(),
                        target: component.activation.clone(),
                        schedule: Default::default(),
                        slices: vec![],
                        action: InterventionAction::MaskComponents {
                            dtype: InterventionDtype::Float32,
                            indices: vec![0, (component.count - 1) as u32],
                            keep_selected,
                        },
                        evidence: InterventionEvidence::None,
                    }],
                };
                // Cold topology alone never authorizes execution.
                assert!(plan
                    .clone()
                    .admit_invocations(&discovery, bounds, "run")
                    .is_err());
                let mut loaded = discovery.clone();
                loaded.session_identity = Some("neutral-loaded-session".into());
                for point in &mut loaded.points {
                    point.prefill = ObservationSupportStatus::Supported;
                    point.decode = ObservationSupportStatus::Supported;
                }
                let admitted = plan
                    .clone()
                    .admit_invocations(&loaded, bounds, "run")
                    .unwrap();
                assert_eq!(admitted.points()[0].axes[2].name, "component");
                assert!(!loaded
                    .points
                    .iter()
                    .any(|point| point.path == component.effective_activation));
                let mut invalid = plan;
                let InterventionAction::MaskComponents { indices, .. } =
                    &mut invalid.operations[0].action
                else {
                    unreachable!()
                };
                indices.push(component.count as u32);
                assert!(invalid.admit_invocations(&loaded, bounds, "run").is_err());
            }
        }
        assert_eq!(graph.component_scopes.len(), 2);
        for scope in &graph.component_scopes {
            assert!(scope.execution_groups.is_empty());
            assert_eq!(scope.components.len(), 2);
            assert!(scope.routed_components.is_empty());
            assert!(graph
                .layer_groups
                .iter()
                .any(|group| group.id == scope.id && group.physical_layer_count == 1));
            let ComponentResidualBase::LinearFusion { inputs, .. } = &scope.residual_base else {
                panic!("prediction fusion")
            };
            assert_eq!(inputs[0].columns, 0..8);
            assert_eq!(inputs[1].columns, 8..16);
            let ComponentFusionSource::Observation { path } = &inputs[0].source else {
                panic!("hidden first")
            };
            let first = graph
                .component_transforms
                .iter()
                .find(|transform| transform.effective_output.as_ref() == Some(path))
                .unwrap();
            let ComponentTensorTransformEquation::Normalization { normalization } = &first.equation
            else {
                panic!("first hidden normalization")
            };
            assert_eq!(normalization, &inputs[0].normalization);
            let ComponentFusionSource::TokenEmbedding { normalization, .. } = &inputs[1].source
            else {
                panic!("embedding second")
            };
            assert_eq!(
                normalization.as_ref().unwrap().gain.as_deref(),
                Some("model.embed_norm.weight")
            );
            assert_ne!(
                normalization.as_ref().unwrap().gain,
                inputs[1].normalization.gain
            );
            assert_eq!(
                scope.readout.normalization.kind,
                if chain_norm {
                    ComponentNormalizationKind::Rms
                } else {
                    ComponentNormalizationKind::Identity
                }
            );
            assert_eq!(scope.readout.other_writes.len(), 2);
        }
    }
}

#[test]
fn gemma4_components_declare_actual_publishers_branch_norms_and_residual_scales() {
    use eredu_core::component::*;
    for sparse in [false, true] {
        for reuse_key in [false, true] {
            let graph = describe(serde_json::json!({
                "model_type":"gemma4",
                "text_config":{
                    "model_type":"gemma4_text","hidden_size":16,"num_hidden_layers":4,
                    "intermediate_size":32,"num_attention_heads":2,"num_key_value_heads":1,
                    "head_dim":8,"rms_norm_eps":1e-5,"vocab_size":64,"max_position_embeddings":64,
                    "layer_types":["sliding_attention","full_attention","sliding_attention","full_attention"],
                    "sliding_window":16,"num_kv_shared_layers":1,"attention_k_eq_v":reuse_key,
                    "enable_moe_block":sparse,"num_experts":sparse.then_some(4),"top_k_experts":sparse.then_some(2),"moe_intermediate_size":sparse.then_some(16),
                    "hidden_size_per_layer_input":4,"vocab_size_per_layer_input":64,
                    "final_logit_softcapping":7.0
                }
            }));
            validate(&graph);
            let consumer = graph
                .components
                .iter()
                .find(|group| group.id == "decoder.3.attention.channels")
                .unwrap();
            assert!(consumer.reads[0].source.is_none());
            for read in &consumer.reads[1..] {
                assert_eq!(
                    read.source,
                    Some(ComponentReadSource::PublishedAttentionState {
                        component_group: "decoder.1.attention.channels".into()
                    })
                );
                assert_eq!(read.rows.row_range(0), Some(0..8));
            }
            assert_eq!(
                consumer.reads[2].weight,
                format!(
                    "model.language_model.layers.1.self_attn.{}.weight",
                    if reuse_key { "k_proj" } else { "v_proj" }
                )
            );
            let value_norm = &consumer.reads[2]
                .head_normalization
                .as_ref()
                .unwrap()
                .normalization;
            assert_eq!(value_norm.epsilon.value(), 1e-6);
            assert!(value_norm.gain.is_none());
            let readout = graph.component_readout.as_ref().unwrap();
            assert_eq!(readout.block_transforms.len(), 4);
            for (layer, reference) in readout.block_transforms.iter().enumerate() {
                assert_eq!(reference.layer_index, layer);
                let transform = graph
                    .component_transforms
                    .iter()
                    .find(|t| t.id == reference.transform_id)
                    .unwrap();
                let ComponentTensorTransformEquation::LearnedScale { scale } = &transform.equation
                else {
                    panic!("whole-residual learned scale")
                };
                assert_eq!(
                    scale.parameter,
                    format!("model.language_model.layers.{layer}.layer_scalar")
                );
                assert!(graph.observations.get(&transform.input).is_some());
                assert!(graph
                    .observations
                    .get(transform.effective_output.as_ref().unwrap())
                    .is_some());
            }
            assert_eq!(readout.other_writes.len(), if sparse { 8 } else { 4 });
            for group in graph
                .components
                .iter()
                .filter(|group| group.id.ends_with(".units"))
            {
                assert!(matches!(
                    group.activation_equation,
                    ComponentActivation::Gated {
                        activation: ComponentNonlinearity::Gelu,
                        ..
                    }
                ));
                if sparse {
                    assert_eq!(
                        graph.node(&group.node_id).unwrap().parent,
                        Some(format!("decoder.{}.feed_forward", group.layer_index))
                    );
                }
            }
            assert_eq!(graph.routed_components.len(), if sparse { 4 } else { 0 });
            for group in &graph.routed_components {
                let write = group
                    .write_output
                    .as_ref()
                    .expect("complete routed bank write");
                let output = group.output.as_ref().expect("post-normalized routed bank");
                let transform = graph
                    .component_transforms
                    .iter()
                    .find(|transform| {
                        transform.input == format!("{write}.effective")
                            && transform.output == *output
                    })
                    .expect("routed bank normalization equation");
                assert_eq!(transform.node_id, group.node_id);
                assert!(matches!(
                    transform.equation,
                    ComponentTensorTransformEquation::Normalization { .. }
                ));
                assert!(matches!(
                    group.activation_equation,
                    ComponentActivation::Gated {
                        activation: ComponentNonlinearity::GeluApproximate,
                        ..
                    }
                ));
                assert!(group.residual_scale.is_none());
            }
            assert_eq!(
                graph,
                serde_json::from_slice(&serde_json::to_vec(&graph).unwrap()).unwrap()
            );
        }
    }
}

/// A real unit hook has one mutable input surface and one read-only result.
fn assert_unit_boundary_pair(
    graph: &ArchitectureDescriptor,
    interventions: &[eredu_core::intervention::InterventionPoint],
    path: &str,
    expected_axes: &[TensorAxis],
) {
    let effective_path = format!("{path}.effective");
    let original = graph
        .observations
        .get(path)
        .expect("original unit boundary");
    let effective = graph
        .observations
        .get(&effective_path)
        .expect("effective unit boundary");
    assert_eq!(original.position, ObservationPosition::BeforeIntervention);
    assert_eq!(effective.position, ObservationPosition::AfterIntervention);
    assert_eq!(original.axes.as_deref(), Some(expected_axes));
    assert_eq!(effective.axes, original.axes);
    assert_eq!(effective.dtype, original.dtype);
    assert_eq!(effective.value_type, original.value_type);
    assert_eq!(effective.node_id, original.node_id);
    assert_eq!(effective.requirements, original.requirements);
    assert_eq!(effective.prefill, original.prefill);
    assert_eq!(effective.decode, original.decode);
    assert_eq!(
        original.requirements,
        [ObservationRequirement::ActivationHooks]
    );
    assert!(original.prefill && original.decode);
    for name in [path, effective_path.as_str()] {
        assert_eq!(
            graph
                .observations
                .points
                .iter()
                .filter(|p| p.path == name)
                .count(),
            1
        );
        assert_eq!(
            graph
                .node(&original.node_id)
                .unwrap()
                .observation_paths
                .iter()
                .filter(|p| p.as_str() == name)
                .count(),
            1
        );
    }
    let mutable = interventions
        .iter()
        .filter(|p| p.path == path)
        .collect::<Vec<_>>();
    assert_eq!(mutable.len(), 1);
    assert_eq!(mutable[0].axes.as_slice(), expected_axes);
    assert!(!interventions.iter().any(|p| p.path == effective_path));
    assert!(graph
        .observations
        .get(&format!("{effective_path}.effective"))
        .is_none());
}

#[test]
fn dense_unit_catalog_pairs_effective_values_without_an_extra_mutable_hook() {
    let json = serde_json::json!({
        "model_type":"llama", "hidden_size":16, "num_hidden_layers":2,
        "intermediate_size":32, "num_attention_heads":4, "num_key_value_heads":2,
        "head_dim":4, "rms_norm_eps":1e-6, "vocab_size":32,
        "max_position_embeddings":128, "rope_theta":10000.0
    });
    let prepared = crate::configuration::MODEL_CONFIGURATIONS
        .resolve_safetensors(&json)
        .unwrap();
    let plan = prepared.architecture_plan();
    let graph = plan.architecture_descriptor();
    let interventions = plan.intervention_points();
    validate(&graph);
    assert_eq!(
        graph.observations.completeness,
        DescriptionCompleteness::Complete
    );
    for layer in 0..2 {
        for boundary in ["input", "output"] {
            assert_unit_boundary_pair(
                &graph,
                &interventions,
                &format!("model.layers.{layer}.{boundary}"),
                &axes(16),
            );
        }
    }
}
