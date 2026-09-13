use super::*;
use eredu_core::component::*;

#[test]
fn v3_latent_read_chains_preserve_full_head_and_rotary_dependencies() {
    for rank in [None, Some(3)] {
        for prediction_layers in [0, 1, 2] {
            let graph = describe(serde_json::json!({
                "model_type":"deepseek_v3", "hidden_size":8, "vocab_size":16,
                "num_hidden_layers":2, "num_attention_heads":2,
                "intermediate_size":10, "moe_intermediate_size":4,
                "q_lora_rank":rank, "kv_lora_rank":3, "qk_nope_head_dim":2,
                "qk_rope_head_dim":2, "v_head_dim":3, "first_k_dense_replace":1,
                "n_routed_experts":2, "n_shared_experts":1, "num_experts_per_tok":1,
                "n_group":1, "topk_group":1, "max_position_embeddings":64,
                "num_nextn_predict_layers":prediction_layers
            }));
            validate(&graph);
            assert_eq!(graph.schema_version, ARCHITECTURE_DESCRIPTOR_SCHEMA_VERSION);
            assert_eq!(graph.components.len(), 4);
            assert_eq!(graph.routed_components.len(), 1);
            assert_eq!(
                graph.observations.completeness,
                DescriptionCompleteness::Complete
            );
            assert_eq!(graph.component_scopes.len(), prediction_layers);
            validate_prediction_scopes(&graph);
            for point in &graph.observations.points {
                let actual =
                    crate::speculative_execution::speculative_capture_scope(&graph, &point.node_id)
                        .unwrap();
                assert_eq!(
                    matches!(
                        actual,
                        eredu_runtime::capture::SpeculativeCaptureScope::Prediction { .. }
                    ),
                    point
                        .requirements
                        .contains(&ObservationRequirement::PredictionExecution),
                    "{}",
                    point.path,
                );
            }
            let shared = graph
                .components
                .iter()
                .find(|group| group.node_id.ends_with(".shared"))
                .unwrap();
            assert_eq!(shared.count, 4);
            assert_eq!(
                shared.write_partition,
                ComponentWritePartition::TensorParallelSum
            );
            assert_eq!(
                shared.activation,
                "model.layers.1.feed_forward.shared.units"
            );
            assert_eq!(
                shared.write_parameter_group,
                "parameters:model.layers.1.mlp"
            );
            assert_eq!(shared.reads.len(), 2);
            assert_eq!(shared.reads[0].role, ComponentReadRole::Gate);
            assert_eq!(shared.reads[1].role, ComponentReadRole::Value);
            assert_eq!(
                graph
                    .nodes
                    .iter()
                    .find(|node| node.id == shared.node_id)
                    .unwrap()
                    .parent
                    .as_deref(),
                Some("decoder.layers.1.feed_forward")
            );
            let attention = graph
                .components
                .iter()
                .find(|g| {
                    g.layer_index == 0
                        && matches!(g.activation_equation, ComponentActivation::Attention { .. })
                })
                .unwrap();
            assert_eq!(attention.count, 6);
            assert_eq!(attention.write_partition, ComponentWritePartition::Complete);
            assert_eq!(
                attention.output.as_deref(),
                Some("model.layers.0.compressed_attention.output")
            );
            let [query, key_latent, key_rotary, value] = attention.reads.as_slice() else {
                panic!("complete MLA dependencies")
            };
            assert_eq!(query.role, ComponentReadRole::Query);
            assert_eq!(query.rows.row_range(4), Some(4..8));
            assert_eq!(query.input_projections.len(), usize::from(rank.is_some()));
            if let Some(stage) = query.input_projections.first() {
                assert_eq!(stage.rows, 0..3);
                assert_eq!(stage.weight, "model.layers.0.self_attn.q_a_proj.weight");
                assert_eq!(
                    stage.normalization.as_ref().unwrap().gain.as_deref(),
                    Some("model.layers.0.self_attn.q_a_layernorm.weight")
                );
            }
            assert_eq!(key_latent.role, ComponentReadRole::Key);
            assert_eq!(key_rotary.role, ComponentReadRole::Key);
            assert_eq!(key_latent.rows.row_range(4), Some(5..7));
            for component in 0..6 {
                assert_eq!(key_rotary.rows.row_range(component), Some(3..5));
            }
            assert!(key_rotary.input_projections.is_empty());
            assert_eq!(value.rows.row(2), Some(4));
            assert_eq!(value.rows.row(4), Some(8));
            assert_eq!(value.input_projections, key_latent.input_projections);
            let stage = &value.input_projections[0];
            assert_eq!(stage.rows, 0..3);
            assert_eq!(stage.weight, key_rotary.weight);
            assert_eq!(
                stage.normalization.as_ref().unwrap().gain.as_deref(),
                Some("model.layers.0.self_attn.kv_a_layernorm.weight")
            );
            for read in &attention.reads {
                for stage in &read.input_projections {
                    assert!(graph
                        .parameter_groups
                        .iter()
                        .any(|p| p.id == stage.parameter_group));
                    let point = graph.observations.get(&stage.output).unwrap();
                    assert_eq!(point.position, ObservationPosition::AfterIntervention);
                    assert_eq!(
                        point.axes.as_ref().unwrap()[2].dimension,
                        SymbolicDimension::Known(stage.rows.len())
                    );
                }
            }
            let readout = graph.component_readout.as_ref().unwrap();
            assert_eq!(readout.other_writes.len(), 1);
            assert_eq!(readout.other_writes[0].layer_index, 1);
            assert_eq!(
                readout.other_writes[0].output,
                "model.layers.1.feed_forward.contribution"
            );
            assert!(graph
                .observations
                .get(&readout.other_writes[0].effective_output)
                .is_some());
            assert_eq!(
                graph.routed_components[0].input.as_deref(),
                Some("model.layers.1.feed_forward.input")
            );
        }
    }
}

#[test]
fn legacy_direct_component_read_has_no_implicit_latent_stage() {
    let read: ComponentRead = serde_json::from_value(serde_json::json!({
        "role":"query", "weight":"q.weight", "shared_weight":"q.weight",
        "parameter_group":"parameters:q", "bias":null,
        "rows":{"kind":"direct", "offset":0}, "head_normalization":null
    }))
    .unwrap();
    assert!(read.input_projections.is_empty());
    assert!(
        read.source.is_none(),
        "legacy reads use their own invocation input"
    );
    assert!(serde_json::to_value(&read)
        .unwrap()
        .get("input_projections")
        .is_none());
}

fn validate_prediction_scopes(graph: &ArchitectureDescriptor) {
    let support = eredu_runtime::inspection::observation_support(
        &graph.observations,
        eredu_runtime::inspection::ObservationExecutionContext {
            activation_inspection: true,
            prediction_inspection: false,
            selected: true,
            partitioned: false,
            mechanisms: ObservationMechanisms {
                activation_tensors: true,
                routing_tensors: true,
                routed_unit_tensors: true,
                floating_to_f32: true,
            },
        },
    );
    let mut identities: BTreeSet<_> = graph.components.iter().map(|g| &g.id).collect();
    for (depth, scope) in graph.component_scopes.iter().enumerate() {
        for point in scope
            .components
            .iter()
            .map(|group| graph.observations.get(&group.activation).unwrap())
            .chain(std::iter::once(
                graph.observations.get(&scope.readout.normalized).unwrap(),
            ))
        {
            assert_eq!(
                crate::speculative_execution::speculative_capture_scope(graph, &point.node_id)
                    .unwrap(),
                eredu_runtime::capture::SpeculativeCaptureScope::Prediction { depth }
            );
        }
        assert_eq!(
            scope.kind,
            ComponentExecutionScopeKind::Prediction { depth }
        );
        assert_eq!(
            graph.node(&scope.node_id).unwrap().kind,
            ArchitectureNodeKind::Prediction
        );
        let group = graph
            .layer_groups
            .iter()
            .find(|g| scope.execution_groups.contains(&g.id))
            .unwrap();
        assert_eq!(group.physical_layer_count, 1);
        let layer = graph
            .node(&group.passes[0].executions[0].node_id)
            .unwrap()
            .layer_index
            .unwrap();
        assert_eq!(layer, 2 + depth);
        assert_eq!(scope.components.len(), 2);
        assert_eq!(scope.routed_components.len(), 1);
        let readout = &scope.readout;
        assert_ne!(
            readout.weight,
            graph.component_readout.as_ref().unwrap().weight
        );
        assert_eq!(
            readout.weight,
            format!("model.layers.{layer}.shared_head.head.weight")
        );
        assert_eq!(readout.other_writes.len(), 1);
        for component in &scope.components {
            assert!(identities.insert(&component.id));
            assert_eq!(component.layer_index, layer);
            assert!(graph
                .parameter_groups
                .iter()
                .any(|p| p.id == component.write_parameter_group));
            for path in [
                &component.activation,
                &component.effective_activation,
                component.write_input.as_ref().unwrap(),
            ] {
                let point = graph.observations.get(path).unwrap();
                assert!(point
                    .requirements
                    .contains(&ObservationRequirement::PredictionExecution));
            }
        }
        let ComponentResidualBase::LinearFusion {
            inputs,
            weight,
            shared_weight,
            parameter_group,
            projection_input,
            output,
            effective_output,
            bias,
        } = &scope.residual_base
        else {
            panic!("V3 fixture declares concatenated linear fusion")
        };
        assert_eq!(weight, shared_weight);
        assert_eq!(weight, &format!("model.layers.{layer}.eh_proj.weight"));
        assert!(graph
            .parameter_groups
            .iter()
            .any(|p| &p.id == parameter_group));
        assert_eq!(bias, &None);
        assert_eq!(inputs.len(), 2);
        assert_eq!(inputs[0].columns, 0..8);
        assert_eq!(inputs[1].columns, 8..16);
        let ComponentFusionSource::TokenEmbedding {
            weight,
            shared_weight,
            parameter_group,
            scale,
            normalization,
        } = &inputs[0].source
        else {
            panic!("embedding input")
        };
        assert_eq!(weight, "model.embed_tokens.weight");
        assert_eq!(shared_weight, weight);
        assert_eq!(scale.value(), 1.0);
        assert!(normalization.is_none());
        assert!(graph
            .parameter_groups
            .iter()
            .any(|p| &p.id == parameter_group));
        let ComponentFusionSource::Observation { path } = &inputs[1].source else {
            panic!("supplied hidden input")
        };
        assert_eq!(path, &format!("mtp.{depth}.capture"));
        assert_eq!(
            graph.observations.get(path).unwrap().position,
            ObservationPosition::ReadOnly
        );
        for input in inputs {
            assert_eq!(input.normalization.kind, ComponentNormalizationKind::Rms);
            assert_eq!(
                graph.observations.get(&input.output).unwrap().position,
                ObservationPosition::AfterIntervention
            );
        }
        assert_eq!(
            graph.observations.get(projection_input).unwrap().position,
            ObservationPosition::ReadOnly
        );
        assert_eq!(
            graph
                .observations
                .get(projection_input)
                .unwrap()
                .axes
                .as_ref()
                .unwrap()[2]
                .dimension,
            SymbolicDimension::Known(16)
        );
        assert_eq!(
            graph.observations.get(output).unwrap().position,
            ObservationPosition::BeforeIntervention
        );
        assert_eq!(
            graph.observations.get(effective_output).unwrap().position,
            ObservationPosition::AfterIntervention
        );
        assert_eq!(
            graph
                .observations
                .get(readout.projection_input.as_ref().unwrap())
                .unwrap()
                .position,
            ObservationPosition::ReadOnly
        );
        for point in &graph.observations.points {
            let mut node = graph.node(&point.node_id).unwrap();
            while let Some(parent) = &node.parent {
                if node.id == scope.node_id {
                    break;
                }
                node = graph.node(parent).unwrap();
            }
            if node.id == scope.node_id {
                assert!(
                    point
                        .requirements
                        .contains(&ObservationRequirement::PredictionExecution),
                    "{}",
                    point.path
                );
                let selected = support
                    .points
                    .iter()
                    .find(|p| p.path == point.path)
                    .unwrap();
                assert!(matches!(
                    selected.prefill,
                    ObservationSupportStatus::Unsupported(_)
                ));
                assert!(matches!(
                    selected.decode,
                    ObservationSupportStatus::Unsupported(_)
                ));
            }
        }
    }
    let wire = serde_json::to_value(graph).unwrap();
    assert!(wire["component_readout"].get("equation").is_none());
    assert_eq!(wire["component_readout"]["weight"], "lm_head.weight");
    assert_eq!(
        serde_json::from_value::<ArchitectureDescriptor>(wire).unwrap(),
        *graph
    );
}
