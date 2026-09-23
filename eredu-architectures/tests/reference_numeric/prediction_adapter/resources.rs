use super::*;
use eredu_runtime::prediction_resources::PredictionExecutionMode;

#[test]
fn embedded_resource_topology_covers_sequential_and_fused_families() {
    let mut v4 = tiny_v4_config();
    v4["num_nextn_predict_layers"] = 1.into();
    v4["num_hash_layers"] = 0.into();
    v4["compress_ratios"] = serde_json::json!([0, 4, 128, 0]);
    let mut dspark = v4.clone();
    dspark["dspark_block_size"] = 3.into();
    dspark["dspark_noise_token_id"] = 0.into();
    dspark["dspark_target_layer_ids"] = serde_json::json!([0, 1]);
    dspark["dspark_markov_rank"] = 2.into();
    let mut inkling = dense_inkling_partition_fixture();
    inkling.as_object_mut().unwrap().remove("vision_config");
    inkling["text_config"]["model_max_length"] = 128.into();
    inkling["mtp_config"] = serde_json::json!({"num_nextn_predict_layers":2,"local_layer_ids":[1],"chain_hidden_post_norm":true});
    let mut qwen = heterogeneous_replicated_configs()
        .into_iter()
        .find(|config| config["model_type"] == "qwen3_5_text")
        .unwrap();
    qwen["mtp_num_hidden_layers"] = 2.into();
    for (name, config, fused) in [
        ("v4", v4, false),
        ("dspark", dspark, true),
        ("inkling", inkling, false),
        ("qwen", qwen, false),
        ("nemotron", nemotron_prediction_config(), false),
    ] {
        let (root, _) = prepared_adapter::payload_fixture_config(&config, 1.0);
        let inspection = eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
        let plan = prepared_adapter::plan(None).with_drafting(eredu_core::DraftingPlan::Embedded {
            max_draft_tokens: 1,
            lookahead: false,
            adaptive_lookahead: false,
        });
        let request = eredu_runtime::NormalizedLoadRequest::from_execution_plan(
            &plan,
            eredu_runtime::ResidencyDiagnostics::new(false, false),
            None,
        )
        .unwrap();
        let before = last_reference_stage_evidence();
        let selected = eredu_architectures::select_preparation(&inspection, &request, &Provider)
            .unwrap_or_else(|error| panic!("{name}: {error:?}"));
        let topology = selected.embedded_prediction_topology().unwrap().unwrap();
        assert_eq!(
            topology.mode,
            if fused {
                PredictionExecutionMode::Fused
            } else {
                PredictionExecutionMode::Sequential
            },
            "{name}"
        );
        assert_eq!(
            topology.mode.prefill_sequence_len(7),
            if fused { 7 } else { 6 }
        );
        assert!(!topology.nodes.is_empty(), "{name}");
        assert!(!topology.state.is_empty(), "{name}");
        assert!(!topology.target_features.entries().is_empty(), "{name}");
        assert_eq!(topology.nodes.len(), topology.invocations.len(), "{name}");
        assert_eq!(
            before,
            last_reference_stage_evidence(),
            "{name}: cold queries must not execute or read payloads"
        );
        // A selected record remains usable independently of checkpoint access.
        std::fs::remove_file(root.path().join("model.safetensors")).unwrap();
        assert_eq!(
            selected.embedded_prediction_topology().unwrap().unwrap(),
            topology
        );
    }
}

pub(super) fn assert_prepared_prediction_resources(
    discovery: &eredu_architectures::prepared_sources::PreparedModelDiscovery,
) {
    use eredu_core::resources::{ResourceIdentity, ResourceRole};
    use eredu_runtime::prediction_resources::PredictionResourceQuery;
    let before = last_reference_stage_evidence();
    let topology = discovery.embedded_prediction_topology().unwrap().unwrap();
    let placement = discovery.prediction_placement().unwrap();
    assert!(!placement.modules().is_empty());
    let ordinals = placement
        .modules()
        .iter()
        .map(|module| module.ordinal)
        .collect::<BTreeSet<_>>();
    assert_eq!(ordinals.len(), placement.modules().len());
    assert!(placement
        .modules()
        .iter()
        .all(|module| !module.parameters.is_empty()));
    assert_eq!(placement.state().len(), topology.state.len());
    let feature_shapes = topology
        .target_features
        .entries()
        .iter()
        .map(|entry| {
            let mut shape = entry.shape().to_vec();
            shape[0] = 1;
            shape[1] = 3;
            shape
        })
        .collect();
    let query = PredictionResourceQuery {
        prepared: eredu_runtime::execution_resources::PreparedResourceQuery {
            scope: ResourceIdentity {
                scope: "reference".into(),
                key: "embedded".into(),
            },
            batch_size: 1,
            prefix_positions: 3,
            additional_positions: 2,
            device_pool: eredu_core::Observed::exact(
                ResourceIdentity {
                    scope: "reference".into(),
                    key: "pool".into(),
                },
                "neutral fixture",
            ),
        },
        floating_state_bytes: Some(4),
        feature_scalar_bytes: Some(4),
        feature_shapes,
        feature_backings: BTreeMap::new(),
    };
    let first = discovery
        .describe_embedded_prediction_resources(&[], None, &query)
        .unwrap()
        .unwrap();
    let second = discovery
        .describe_embedded_prediction_resources(&[], None, &query)
        .unwrap()
        .unwrap();
    assert_eq!(first, second);
    assert!(first.allocations.iter().any(|allocation| allocation
        .uses
        .iter()
        .any(|usage| usage.role == ResourceRole::MutableState)));
    // Without prepared backing slots, borrowed target weights are not invented.
    assert!(!first.allocations.iter().any(|allocation| allocation
        .uses
        .iter()
        .any(|usage| usage.role == ResourceRole::Parameters)));
    assert_eq!(
        before,
        last_reference_stage_evidence(),
        "resource queries must not load, acquire or execute"
    );
}
