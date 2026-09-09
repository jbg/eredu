use super::*;
use std::collections::BTreeSet;

fn describe(json: serde_json::Value) -> ArchitectureDescriptor {
    crate::configuration::MODEL_CONFIGURATIONS
        .resolve_safetensors(&json)
        .unwrap()
        .architecture_plan()
        .architecture_descriptor()
}

fn validate(graph: &ArchitectureDescriptor) {
    let ids = graph
        .nodes
        .iter()
        .map(|n| n.id.as_str())
        .collect::<BTreeSet<_>>();
    assert_eq!(ids.len(), graph.nodes.len());
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
        "head_dim":4, "rms_norm_eps":1e-6, "vocab_size":32, "tie_word_embeddings":true
    }));
    validate(&graph);
    assert_eq!(graph.completeness, DescriptionCompleteness::Complete);
    assert_eq!(graph.observations.points.len(), 5);
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
                            format!("model.layers.{logical}.output"),
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
    for group in legacy["parameter_groups"].as_array_mut().unwrap() {
        group.as_object_mut().unwrap().remove("shared_with");
    }
    let decoded: ArchitectureDescriptor = serde_json::from_value(legacy).unwrap();
    assert!(decoded.layer_groups.is_empty());
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
        partitioned: false,
        mechanisms: ObservationMechanisms {
            activation_tensors: true,
            routing_tensors: true,
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
            partitioned: false,
            mechanisms: ObservationMechanisms {
                activation_tensors: true,
                routing_tensors: true,
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
