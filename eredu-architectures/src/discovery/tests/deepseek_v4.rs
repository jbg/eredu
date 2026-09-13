use super::*;

fn v4_prediction_config() -> serde_json::Value {
    serde_json::json!({
        "model_type":"deepseek_v4", "hidden_size":8, "moe_intermediate_size":6,
        "num_hidden_layers":3, "num_attention_heads":4, "num_key_value_heads":1,
        "head_dim":4, "qk_rope_head_dim":2, "q_lora_rank":3,
        "o_groups":2, "o_lora_rank":5, "vocab_size":16,
        "max_position_embeddings":128, "sliding_window":4, "compress_ratios":[0,4,128],
        "index_n_heads":2, "index_head_dim":4, "index_topk":1,
        "hc_mult":2, "hc_sinkhorn_iters":2, "n_routed_experts":2,
        "n_shared_experts":1, "num_experts_per_tok":1, "num_hash_layers":1,
        "scoring_func":"sqrtsoftplus", "topk_method":"noaux_tc",
        "norm_topk_prob":true, "routed_scaling_factor":1.0, "swiglu_limit":4.0
    })
}

#[test]
fn fused_v4_declares_one_multiblock_score_scope_and_separate_context_invocations() {
    use eredu_core::{
        component::*,
        speculative::{SpeculativeCaptureBinding, SpeculativeCaptureScope as Scope},
    };
    let mut config = v4_prediction_config();
    config["num_nextn_predict_layers"] = 2.into();
    config["compress_ratios"] = serde_json::json!([0, 4, 128, 0, 0]);
    config["dspark_block_size"] = 3.into();
    config["dspark_noise_token_id"] = 0.into();
    config["dspark_target_layer_ids"] = serde_json::json!([0, 2]);
    config["dspark_markov_rank"] = 5.into();
    let graph = describe(config);
    validate(&graph);
    assert_eq!(graph.components.len(), 6);
    let [scope] = graph.component_scopes.as_slice() else {
        panic!("one fused score scope")
    };
    assert_eq!(scope.kind, ComponentExecutionScopeKind::FusedPrediction);
    assert_eq!(scope.execution_groups, ["mtp.0", "mtp.1"]);
    assert_eq!(scope.components.len(), 4);
    assert_eq!(scope.routed_components.len(), 2);
    assert_eq!(
        scope.readout.stream_residual.as_ref().unwrap().cycles.len(),
        4
    );
    assert!(
        matches!(&scope.residual_base, ComponentResidualBase::Source {
        source: ComponentFusionSource::TokenEmbedding { weight, .. },
        expansion: ComponentFusionExpansion::BroadcastAxis { axis: 2, extent: 2, .. }, ..
    } if weight == "embed.weight")
    );
    let [write] = scope.readout.score_writes.as_slice() else {
        panic!("one dynamic score addition")
    };
    assert_eq!(write.broadcast_axes, ["sequence"]);
    assert_eq!(write.weight, "mtp.1.markov_head.markov_w2.weight");
    assert!(
        matches!(&write.source, ComponentFusionSource::TokenEmbedding { weight, .. } if weight == "mtp.1.markov_head.markov_w1.weight")
    );
    assert_eq!(
        graph
            .observations
            .get(&write.output)
            .unwrap()
            .axes
            .as_ref()
            .unwrap()[1]
            .dimension,
        SymbolicDimension::Known(1)
    );
    assert!(scope.readout.other_writes.is_empty());
    for point in &graph.observations.points {
        let actual =
            crate::speculative_execution::speculative_capture_scope(&graph, &point.node_id)
                .unwrap();
        let expected = if point.path.starts_with("dspark.context.") {
            Scope::PredictionContext
        } else if point.path.starts_with("dspark.proposal.") {
            Scope::FusedProposal
        } else {
            Scope::Target
        };
        assert_eq!(actual, expected, "{}", point.path);
        assert_eq!(
            point
                .requirements
                .contains(&ObservationRequirement::PredictionExecution),
            expected != Scope::Target
        );
    }
    assert!(graph.components.iter().all(|g| g.layer_index < 3));
    assert!(scope.components.iter().all(|g| g.layer_index >= 3));
    let encoded = serde_json::to_vec(&graph).unwrap();
    assert_eq!(
        serde_json::from_slice::<ArchitectureDescriptor>(&encoded).unwrap(),
        graph
    );
    // Explicit roots are validated even when the selected point is elsewhere.
    for invalid in 0..2 {
        let mut other = graph.clone();
        other.speculative_invocations.push(if invalid == 0 {
            other.speculative_invocations[0].clone()
        } else {
            SpeculativeCaptureBinding {
                node_id: "missing".into(),
                scope: Scope::FusedProposal,
            }
        });
        assert!(crate::speculative_execution::speculative_capture_scope(&other, "output").is_err());
    }
}

#[test]
fn v4_channels_join_actual_grouped_factors_and_distinct_hyper_geometry() {
    let graph = describe(v4_prediction_config());
    validate(&graph);
    assert_eq!(graph.components.len(), 6);
    assert_eq!(graph.routed_components.len(), 3);
    assert_eq!(graph.schema_version, ARCHITECTURE_DESCRIPTOR_SCHEMA_VERSION);
    for layer in 0..3 {
        let group = graph
            .components
            .iter()
            .find(|g| g.layer_index == layer && g.write_input_projection.is_some())
            .unwrap();
        assert_eq!(group.count, 16);
        assert!(group.write_input.is_none());
        assert_eq!(
            group.write_weight,
            format!("layers.{layer}.attn.wo_b.weight")
        );
        let stage = group.write_input_projection.as_ref().unwrap();
        assert_eq!(stage.weight, format!("layers.{layer}.attn.wo_a.weight"));
        assert_eq!((stage.groups, stage.rank), (2, 5));
        for (path, expected) in [
            (&stage.input, vec!["batch", "group", "sequence", "channel"]),
            (&stage.output, vec!["batch", "sequence", "projection"]),
            (&stage.final_input, vec!["batch", "sequence", "projection"]),
        ] {
            let point = graph.observations.get(path).unwrap();
            assert_eq!(point.position, ObservationPosition::ReadOnly);
            assert_eq!(
                point
                    .axes
                    .as_ref()
                    .unwrap()
                    .iter()
                    .map(|a| a.name.as_str())
                    .collect::<Vec<_>>(),
                expected
            );
        }
        assert_eq!(group.reads[0].input_projections[0].rows, 0..3);
        assert!(group.reads[0]
            .head_normalization
            .as_ref()
            .unwrap()
            .normalization
            .gain
            .is_none());
        assert_eq!(group.reads[1].weight, group.reads[2].weight);
        assert_eq!(group.reads[2].rows.row_range(15), Some(0..4));
        let point = graph
            .observations
            .get(&format!("layers.{layer}.input.effective"))
            .unwrap();
        assert_eq!(
            point
                .axes
                .as_ref()
                .unwrap()
                .iter()
                .map(|a| a.name.as_str())
                .collect::<Vec<_>>(),
            ["batch", "sequence", "stream", "hidden"]
        );
        for phase in ["attention", "feed_forward"] {
            let point = graph
                .observations
                .get(&format!("layers.{layer}.hyper.{phase}.combination"))
                .unwrap();
            assert_eq!(point.position, ObservationPosition::ReadOnly);
            assert_eq!(
                point
                    .axes
                    .as_ref()
                    .unwrap()
                    .iter()
                    .map(|a| a.name.as_str())
                    .collect::<Vec<_>>(),
                ["batch", "sequence", "input_stream", "output_stream"]
            );
        }
    }
}

#[test]
fn v4_sequential_prediction_declares_separate_fusion_and_stream_readout() {
    use eredu_core::component::{
        ComponentFusionExpansion, ComponentResidualBase, ComponentStreamBase,
    };
    let mut config = v4_prediction_config();
    config["num_nextn_predict_layers"] = 2.into();
    config["compress_ratios"] = serde_json::json!([0, 4, 128, 0, 0]);
    let graph = describe(config);
    validate(&graph);
    assert_eq!(graph.components.len(), 6);
    assert_eq!(graph.component_scopes.len(), 2);
    for (depth, scope) in graph.component_scopes.iter().enumerate() {
        assert_eq!(scope.execution_groups, vec![format!("mtp.{depth}")]);
        assert_eq!(scope.components.len(), 2);
        assert_eq!(scope.routed_components.len(), 1);
        let ComponentResidualBase::ProjectedSum {
            inputs,
            output,
            effective_output,
        } = &scope.residual_base
        else {
            panic!("V4 has independently projected embedding and stream inputs")
        };
        assert_eq!(inputs.len(), 2);
        assert!(
            matches!(&inputs[0].expansion, ComponentFusionExpansion::BroadcastAxis {axis:2,name,extent:2} if name=="stream")
        );
        assert_eq!(inputs[1].expansion, ComponentFusionExpansion::Identity);
        for (term, rank) in inputs.iter().zip([3, 4]) {
            assert_eq!(
                graph
                    .observations
                    .get(&term.projection_input)
                    .unwrap()
                    .axes
                    .as_ref()
                    .unwrap()
                    .len(),
                rank
            );
            assert_eq!(
                graph.observations.get(&term.normalized).unwrap().position,
                ObservationPosition::AfterIntervention
            );
            assert!(term.weight.starts_with(&format!("mtp.{depth}.")));
        }
        assert_eq!(
            graph.observations.get(output).unwrap().position,
            ObservationPosition::BeforeIntervention
        );
        let streams = scope.readout.stream_residual.as_ref().unwrap();
        assert!(
            matches!(&streams.base, ComponentStreamBase::Streams{input} if input==effective_output)
        );
        assert_eq!(streams.cycles.len(), 2);
        assert_eq!(
            scope.readout.weight,
            graph.component_readout.as_ref().unwrap().weight
        );
        for cycle in &streams.cycles {
            assert_eq!(cycle.layer_index, 3 + depth);
            for path in [
                &cycle.input,
                &cycle.collapsed,
                &cycle.write,
                &cycle.output,
                &cycle.pre,
                &cycle.post,
                &cycle.combination,
            ] {
                assert!(graph.observations.get(path).is_some(), "{path}");
            }
        }
    }
}
