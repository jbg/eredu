//! Resource producers consume the same cold selection as ordinary construction.
use super::*;
use eredu_core::resources::{ResourceCoverage, ResourceDescription, ResourceRole, ResourceSize};
use eredu_core::{resources::ResourceIdentity, Observed};
use eredu_runtime::execution_resources::PreparedResourceQuery;

fn state_payloads(description: &ResourceDescription) -> (u64, u64, usize) {
    let mut current = 0;
    let mut peak = 0;
    let mut count = 0;
    for allocation in &description.allocations {
        if !allocation
            .uses
            .iter()
            .any(|usage| usage.role == ResourceRole::MutableState)
        {
            continue;
        }
        let (now, future) = match &allocation.size {
            ResourceSize::Fixed { extent } => (extent, extent),
            ResourceSize::ContextDependent {
                current,
                horizon_peak,
            } => (current, horizon_peak),
        };
        assert_eq!(now.payload.upper_bytes, Some(now.payload.lower_bytes));
        assert_eq!(future.payload.upper_bytes, Some(future.payload.lower_bytes));
        assert!(now.capacity.upper_bytes.is_none());
        assert!(future.capacity.upper_bytes.is_none());
        current += now.payload.lower_bytes;
        peak += future.payload.lower_bytes;
        count += 1;
    }
    (current, peak, count)
}

#[test]
fn selected_dense_and_hybrid_state_resources_follow_ordinary_layouts() {
    let dense = inspected_llama();
    let hybrid = inspected_config(serde_json::json!({
        "model_type": "lfm2", "vocab_size": 64, "hidden_size": 16,
        "intermediate_size": 32, "num_hidden_layers": 2,
        "num_attention_heads": 4, "num_key_value_heads": 2,
        "max_position_embeddings": 64,
        "layer_types": ["conv", "full_attention"], "conv_L_cache": 3,
        "block_multiple_of": 8, "block_ffn_dim_multiplier": 1.0,
        "block_auto_adjust_ff_dim": true, "tie_word_embeddings": false
    }));
    let routed = inspected_config(routed_config());
    let mechanisms = BoundedIndependentAdapter::default();
    let query = PreparedResourceQuery {
        scope: ResourceIdentity {
            scope: "independent-model-instance".into(),
            key: "text".into(),
        },
        batch_size: 1,
        prefix_positions: 5,
        additional_positions: 3,
        device_pool: Observed::exact(
            ResourceIdentity {
                scope: "machine".into(),
                key: "unified".into(),
            },
            "test pool",
        ),
    };
    // Dense: two layers, two tensors, one head, width four, float32.
    // Hybrid: one 2x16 convolution history plus one layer of two 2x4 KV tensors.
    for ((root, inspection), expected) in [
        (dense, (320, 512, 4)),
        (hybrid, (448, 640, 3)),
        (routed, (320, 512, 4)),
    ] {
        let selected =
            select_preparation(&inspection, &NormalizedLoadRequest::default(), &mechanisms)
                .unwrap();
        // Once selected, the producer needs only retained contracts, not artifacts.
        drop(root);
        let description = selected
            .text_realization()
            .describe_prepared_resources(&query)
            .unwrap();
        description.validate().unwrap();
        assert_eq!(state_payloads(&description), expected);
        assert!(matches!(
            description.coverage,
            ResourceCoverage::Partial { .. }
        ));
        let repeated = selected
            .text_realization()
            .describe_prepared_resources(&query)
            .unwrap();
        assert_eq!(description, repeated);
        let mut other = query.clone();
        other.scope.scope = "another-independent-model-instance".into();
        let other = selected
            .text_realization()
            .describe_prepared_resources(&other)
            .unwrap();
        assert_eq!(state_payloads(&other), expected);
        assert!(description
            .allocations
            .iter()
            .all(|a| other.allocations.iter().all(|b| a.identity != b.identity)));
    }
    mechanisms.assert_cold_only();
}

#[test]
fn selected_sliding_attention_does_not_claim_native_storage_truncation() {
    let (_root, inspection) = inspected_config(serde_json::json!({
        "model_type": "mistral", "architectures": ["MistralForCausalLM"],
        "hidden_size": 8, "num_hidden_layers": 2, "intermediate_size": 16,
        "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 4,
        "rms_norm_eps": 0.00001, "vocab_size": 16, "max_position_embeddings": 32,
        "rope_theta": 10000.0, "tie_word_embeddings": false, "sliding_window": 3
    }));
    let mechanisms = BoundedIndependentAdapter::default();
    let selected =
        select_preparation(&inspection, &NormalizedLoadRequest::default(), &mechanisms).unwrap();
    let query = PreparedResourceQuery {
        scope: ResourceIdentity {
            scope: "mistral-instance".into(),
            key: "text".into(),
        },
        batch_size: 1,
        prefix_positions: 5,
        additional_positions: 3,
        device_pool: Observed::unavailable("no physical pool selected for this query"),
    };
    let description = selected
        .text_realization()
        .describe_prepared_resources(&query)
        .unwrap();
    description.validate().unwrap();
    let mut totals = [0; 4];
    for allocation in &description.allocations {
        if !allocation
            .uses
            .iter()
            .any(|usage| usage.role == ResourceRole::MutableState)
        {
            continue;
        }
        let ResourceSize::ContextDependent {
            current,
            horizon_peak,
        } = &allocation.size
        else {
            panic!("state should retain context");
        };
        for (sum, value) in totals.iter_mut().zip([
            current.payload.lower_bytes,
            current.payload.upper_bytes.unwrap(),
            horizon_peak.payload.lower_bytes,
            horizon_peak.payload.upper_bytes.unwrap(),
        ]) {
            *sum += value;
        }
        assert!(current.capacity.upper_bytes.is_none());
        assert!(allocation.placement.value().is_none());
    }
    // Visibility bounds the necessary window, while the selected mechanism may
    // retain all keys: four tensors, four float32 channels, window three.
    assert_eq!(totals, [192, 320, 192, 512]);
    mechanisms.assert_cold_only();
}

#[test]
fn cold_embedded_topology_reuses_invocation_scopes_capture_and_state_without_native_work() {
    use eredu_core::speculative::SpeculativeCaptureScope;
    use eredu_runtime::prediction_resources::PredictionExecutionMode;
    let (root, inspection) = inspected_config(prediction_config());
    let mechanisms = BoundedIndependentAdapter::default();
    let selected =
        select_preparation(&inspection, &NormalizedLoadRequest::default(), &mechanisms).unwrap();
    drop(root);
    let topology = selected.embedded_prediction_topology().unwrap().unwrap();
    assert_eq!(topology.mode, PredictionExecutionMode::Sequential);
    assert_eq!(topology.proposal_capacity, 1);
    assert_eq!(
        topology.target_features,
        *selected
            .prediction_realization()
            .unwrap()
            .requirements()
            .capture()
    );
    assert!(!topology
        .nodes_for_scope(SpeculativeCaptureScope::Prediction { depth: 0 })
        .collect::<Vec<_>>()
        .is_empty());
    assert!(topology
        .nodes_for_scope(SpeculativeCaptureScope::Target)
        .next()
        .is_none());
    assert_eq!(topology.state.len(), 1);
    assert_eq!(topology.state[0].processed_token_offset, -1);
    assert_eq!(topology.state[0].layer, 2);
    assert!(topology
        .parameters
        .iter()
        .any(|group| group.canonical_prefix == "model.embed_tokens"));
    assert_eq!(
        selected.embedded_prediction_topology().unwrap().unwrap(),
        topology
    );
    mechanisms.assert_cold_only();
}

#[test]
fn selected_prediction_topology_keeps_shared_readout_and_actual_fusion_specs() {
    use eredu_runtime::execution_topology::{FeedForwardTopology, TokenMixerTopology};
    let (root, inspection) = inspected_config(serde_json::json!({
        "model_type": "qwen3_5_text", "vocab_size": 16, "hidden_size": 8,
        "num_hidden_layers": 2, "mtp_num_hidden_layers": 2,
        "num_attention_heads": 1, "num_key_value_heads": 1, "head_dim": 8,
        "max_position_embeddings": 32, "intermediate_size": 16, "num_experts": 0,
        "tie_word_embeddings": true, "layer_types": ["full_attention", "full_attention"]
    }));
    let mechanisms = BoundedIndependentAdapter::default();
    let selected =
        select_preparation(&inspection, &NormalizedLoadRequest::default(), &mechanisms).unwrap();
    drop(root);
    let target = selected
        .text_realization()
        .requirements()
        .execution_topology()
        .unwrap();
    let prediction = selected.embedded_prediction_topology().unwrap().unwrap();
    let topology = prediction.execution_topology.as_ref().unwrap();
    assert!(prediction.missing.is_empty());
    assert_eq!(topology.layers.len(), 2);
    assert_eq!(topology.output_invocations, 2);
    assert_eq!(target.output_invocations, 1);
    assert_eq!(topology.output.parameter, target.output.parameter);
    assert_eq!(topology.output.parameter, "model.embed_tokens.weight");
    assert_eq!(prediction.state.len(), 2);
    assert!(prediction
        .state
        .iter()
        .all(|s| s.processed_token_offset == -1));
    for (index, layer) in topology.layers.iter().enumerate() {
        assert_eq!(layer.input_projections.len(), 1);
        let fusion = &layer.input_projections[0];
        assert_eq!((fusion.input, fusion.output), (16, 8));
        assert_eq!(fusion.parameter, "mtp.fc.weight");
        let TokenMixerTopology::Attention {
            projections,
            output_gate,
            ..
        } = &layer.mixer
        else {
            panic!("prediction uses full attention")
        };
        assert!(*output_gate);
        assert_eq!(projections[0].output, 16);
        assert_eq!(
            projections[0].parameter,
            format!("mtp.layers.{index}.self_attn.q_proj.weight")
        );
        assert!(matches!(
            layer.feed_forward,
            FeedForwardTopology::Gated { .. }
        ));
    }
    assert_eq!(
        selected.embedded_prediction_topology().unwrap().unwrap(),
        prediction
    );
    mechanisms.assert_cold_only();
}
