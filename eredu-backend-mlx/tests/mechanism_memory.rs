//! Reusable mechanism descriptions agree with selected native execution.
use eredu_backend_mlx::backend::{
    nn::{attention::attention_with_softcap, shared::MlxNeuralBackend},
    runtime::generation::MlxSamplingBackend,
};
use eredu_nn::{mechanism_memory::*, AttentionArithmetic, NeuralBackend, TensorElementType};
use eredu_runtime::{GenerationSampler, SamplingBackend};
use safemlx::{Array, Device, DeviceType, Stream};

fn attention(queries: u64, keys: u64) -> MechanismInvocation {
    MechanismInvocation::Attention {
        batch: 1,
        query_heads: 2,
        kv_heads: 1,
        queries,
        keys,
        key_width: 2,
        value_width: 2,
        element: TensorElementType::F32,
        arithmetic: AttentionArithmetic::InputScores,
        softcap: false,
        sinks: false,
    }
}

#[test]
fn cold_attention_keeps_aliases_and_native_unknowns_explicit() {
    for (queries, keys) in [(4, 8), (128, 128), (1, 9000)] {
        let contract = MlxNeuralBackend::mechanism_memory(&attention(queries, keys)).unwrap();
        contract.validate().unwrap();
        let output = contract
            .storage
            .iter()
            .find(|storage| storage.name == "output")
            .unwrap();
        assert_eq!(output.payload, MechanismBytes::exact(2 * queries * 2 * 4));
        assert_eq!(output.capacity.upper, None);
        let scores = contract
            .storage
            .iter()
            .find(|storage| storage.name == "score_tile")
            .unwrap();
        assert!(scores.payload.upper.unwrap() <= 2 * queries * keys * 4);
        if queries * keys > 8192 {
            assert!(scores.payload.upper.unwrap() < 2 * queries * keys * 4);
        }
        assert!(contract
            .storage
            .iter()
            .any(|storage| storage.name == "prepared_keys"
                && storage.backing == MechanismBacking::Unknown));
        assert!(contract
            .storage
            .iter()
            .any(|storage| storage.name == "native_workspace" && storage.payload.upper.is_none()));
        assert!(!contract.missing.is_empty());
    }
}

#[test]
fn selected_cpu_sdpa_reports_full_score_payloads_and_preserves_unknowns() {
    use eredu_backend_mlx::backend::nn::memory::describe_for_device;
    let invocation = MechanismInvocation::Attention {
        batch: 1,
        query_heads: 2,
        kv_heads: 1,
        queries: 128,
        keys: 129,
        key_width: 2,
        value_width: 3,
        element: TensorElementType::Bf16,
        arithmetic: AttentionArithmetic::Fused,
        softcap: false,
        sinks: true,
    };
    let cpu = describe_for_device(&invocation, DeviceType::Cpu).unwrap();
    cpu.validate().unwrap();
    let storage = |name: &str| cpu.storage.iter().find(|s| s.name == name).unwrap();
    assert_eq!(
        storage("sdpa_scaled_queries").payload,
        MechanismBytes::exact(1 * 2 * 128 * 2 * 2)
    );
    assert_eq!(
        storage("sdpa_full_scores").payload,
        MechanismBytes::exact(1 * 2 * 128 * 129 * 2)
    );
    assert_eq!(
        storage("sdpa_masked_scores").payload,
        MechanismBytes {
            lower: 0,
            upper: Some(1 * 2 * 128 * 129 * 2)
        }
    );
    for name in ["sdpa_sink_scores", "sdpa_probabilities"] {
        assert_eq!(
            storage(name).payload,
            MechanismBytes::exact(1 * 2 * 128 * 130 * 2)
        );
        assert_eq!(storage(name).capacity.upper, None);
        assert_eq!(storage(name).retention, StorageRetention::Evaluation);
    }
    assert_eq!(
        storage("sdpa_probabilities").backing,
        MechanismBacking::Unknown
    );
    assert_eq!(storage("native_workspace").payload.upper, None);
    for report in [
        MlxNeuralBackend::mechanism_memory(&invocation).unwrap(),
        describe_for_device(&invocation, DeviceType::Gpu).unwrap(),
    ] {
        assert!(!report.storage.iter().any(|s| s.name == "sdpa_full_scores"));
        assert!(!report.missing.is_empty());
    }
}

#[test]
fn described_cpu_sdpa_matches_independent_masked_gqa_with_sinks() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let queries = [0.3f32, -0.2, 0.7, 0.6, -0.5, 0.9, 0.4, 0.2];
    let keys = [0.2f32, 0.8, -0.3, 0.5, 0.7, -0.4];
    let values = [1.0f32, -2.0, 0.5, 3.0, -0.75, 2.0];
    let sinks = [0.4f32, -0.2];
    let scale = 0.7f32;
    let q = Array::from_slice(&queries, &[1, 2, 2, 2]);
    let k = Array::from_slice(&keys, &[1, 1, 3, 2]);
    let v = Array::from_slice(&values, &[1, 1, 3, 2]);
    let mask = Array::from_slice(&[true, false, true, true, true, false], &[2, 3]);
    let sink_array = Array::from_slice(&sinks, &[2]);
    for use_sinks in [false, true] {
        let invocation = MechanismInvocation::Attention {
            batch: 1,
            query_heads: 2,
            kv_heads: 1,
            queries: 2,
            keys: 3,
            key_width: 2,
            value_width: 2,
            element: TensorElementType::F32,
            arithmetic: AttentionArithmetic::Fused,
            softcap: false,
            sinks: use_sinks,
        };
        let report = MlxNeuralBackend::mechanism_memory_in_context(&invocation, &stream).unwrap();
        assert_eq!(
            report.storage.iter().any(|s| s.name == "sdpa_sink_scores"),
            use_sinks
        );
        let actual = attention_with_softcap(
            &q,
            &k,
            &v,
            scale,
            Some(&mask),
            use_sinks.then_some(&sink_array),
            None,
            AttentionArithmetic::Fused,
            &stream,
        )
        .unwrap()
        .into_evaluated()
        .unwrap();
        let actual = actual.as_slice::<f32>();
        let output = report.values.iter().find(|v| v.name == "output").unwrap();
        assert_eq!(output.logical_bytes().unwrap(), actual.len() as u64 * 4);
        for head in 0..2 {
            for row in 0..2 {
                let query = &queries[(head * 2 + row) * 2..][..2];
                let weights: Vec<_> = (0..3)
                    .map(|key| {
                        if (row == 0 && key == 1) || (row == 1 && key == 2) {
                            0.0
                        } else {
                            (scale * (query[0] * keys[key * 2] + query[1] * keys[key * 2 + 1]))
                                .exp()
                        }
                    })
                    .collect();
                let denominator =
                    weights.iter().sum::<f32>() + if use_sinks { sinks[head].exp() } else { 0.0 };
                for channel in 0..2 {
                    let expected = (0..3)
                        .map(|key| weights[key] * values[key * 2 + channel])
                        .sum::<f32>()
                        / denominator;
                    assert!((actual[(head * 2 + row) * 2 + channel] - expected).abs() < 2e-6);
                }
            }
        }
    }
}

#[test]
fn described_attention_paths_preserve_uniform_attention_results() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    for (query_count, key_count) in [(4, 8), (128, 128), (1, 9000)] {
        let contract =
            MlxNeuralBackend::mechanism_memory(&attention(query_count, key_count)).unwrap();
        let queries = Array::from_slice(
            &vec![0.0f32; (2 * query_count * 2) as usize],
            &[1, 2, query_count as i32, 2],
        );
        let keys = Array::from_slice(
            &vec![0.5f32; (key_count * 2) as usize],
            &[1, 1, key_count as i32, 2],
        );
        let values = (0..key_count)
            .flat_map(|i| {
                if i % 2 == 0 {
                    [1.0f32, 2.0]
                } else {
                    [3.0, 6.0]
                }
            })
            .collect::<Vec<_>>();
        let values = Array::from_slice(&values, &[1, 1, key_count as i32, 2]);
        assert_eq!(queries.shape(), &[1, 2, query_count as i32, 2]);
        assert_eq!(keys.shape(), &[1, 1, key_count as i32, 2]);
        assert_eq!(values.shape(), &[1, 1, key_count as i32, 2]);
        let output = attention_with_softcap(
            &queries,
            &keys,
            &values,
            1.0,
            None,
            None,
            None,
            AttentionArithmetic::InputScores,
            &stream,
        )
        .unwrap()
        .into_evaluated()
        .unwrap();
        let expected_bytes = contract
            .values
            .iter()
            .find(|value| value.name == "output")
            .unwrap()
            .logical_bytes()
            .unwrap();
        assert_eq!(output.as_slice::<f32>().len() as u64 * 4, expected_bytes);
        for pair in output.as_slice::<f32>().as_chunks::<2>().0 {
            assert!(
                (pair[0] - 2.0).abs() < 2.0e-5 && (pair[1] - 4.0).abs() < 4.0e-5,
                "queries={query_count}, keys={key_count}: {pair:?}"
            );
        }
    }
}

#[test]
fn shared_vision_report_uses_selected_segment_facts_without_a_physical_fit_claim() {
    use eredu_architectures::qwen::vision::VisionConfigSource;
    let source: VisionConfigSource = serde_json::from_value(serde_json::json!({
        "depth":2, "hidden_size":8, "intermediate_size":12, "num_heads":2,
        "num_position_embeddings":64, "in_channels":3, "patch_size":2,
        "spatial_merge_size":2, "temporal_patch_size":2, "out_hidden_size":32,
        "deepstack_visual_indexes":[]
    }))
    .unwrap();
    let config = source.normalize_qwen3_vl().unwrap();
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let report = config
        .memory_report(
            &[(2, 4, 4)],
            TensorElementType::F32,
            |invocation| MlxNeuralBackend::mechanism_memory_in_context(invocation, &stream),
            |invocation| MlxNeuralBackend::segmented_attention_memory(invocation, &stream),
        )
        .unwrap();
    let segments: Vec<_> = report
        .invocations
        .iter()
        .filter(|group| matches!(group.invocation, MechanismInvocation::Attention { .. }))
        .collect();
    assert_eq!(segments.len(), 2);
    for group in segments {
        assert_eq!(group.repetitions, 2);
        let score = group
            .memory
            .storage
            .iter()
            .find(|storage| storage.name == "sdpa_full_scores")
            .unwrap();
        assert_eq!(score.payload, MechanismBytes::exact(2 * 16 * 16 * 4));
        assert!(group
            .memory
            .storage
            .iter()
            .any(|storage| storage.name == "native_workspace" && storage.payload.upper.is_none()));
        let mut impossible = group.invocation.clone();
        if let MechanismInvocation::Attention { sinks, .. } = &mut impossible {
            *sinks = true;
        }
        assert!(MlxNeuralBackend::segmented_attention_memory(&impossible, &stream).is_err());
    }
    assert_eq!(report.retained_output_logical_bytes, 8 * 32 * 4);
    assert!(!report.missing.is_empty());
}

#[test]
fn sampling_uses_resolved_policy_and_distinguishes_host_storage() {
    let mut sampler = GenerationSampler::default();
    sampler.repeat_penalty = 1.1;
    sampler.accept_token(3);
    let invocation = sampler
        .sampling_invocation(2, 128, TensorElementType::F32, 0.0)
        .unwrap();
    let contract = MlxSamplingBackend::mechanism_memory(&invocation).unwrap();
    contract.validate().unwrap();
    let mask = contract
        .storage
        .iter()
        .find(|storage| storage.name == "penalty_mask")
        .unwrap();
    assert_eq!(mask.placement, MechanismPlacement::Host);
    assert_eq!(mask.payload, MechanismBytes::exact(2 * 128));
    let additive = contract
        .storage
        .iter()
        .find(|storage| storage.name == "penalty_additive")
        .unwrap();
    assert_eq!(additive.payload, MechanismBytes::exact(2 * 128 * 4));
    assert_eq!(sampler.generated_tokens(), &[3]);
    assert!(!contract.missing.is_empty());
}

#[test]
fn quantized_conversion_facts_follow_selected_device_without_loading_weights() {
    use eredu_backend_mlx::backend::nn::memory::describe_for_device;
    let invocation = MechanismInvocation::Projection {
        rows: 3,
        input: 256,
        output: 64,
        format: eredu_checkpoint::LinearFormat::GgufIQuant {
            ggml_type: eredu_gguf::GgmlType::Q4K,
            endian: eredu_gguf::Endian::Little,
        },
        element: TensorElementType::Bf16,
        weight_element: None,
        bias: false,
        bias_element: None,
    };
    let cold = MlxNeuralBackend::mechanism_memory(&invocation).unwrap();
    assert!(cold
        .storage
        .iter()
        .find(|value| value.name == "output")
        .unwrap()
        .payload
        .upper
        .is_none());
    let cpu = describe_for_device(&invocation, DeviceType::Cpu).unwrap();
    assert_eq!(
        cpu.values
            .iter()
            .find(|value| value.name == "output")
            .unwrap()
            .element,
        TensorElementType::F32
    );
    let row = cpu
        .storage
        .iter()
        .find(|value| value.name == "decoded_weight_row")
        .unwrap();
    assert_eq!(row.payload, MechanismBytes::exact(256 * 4));
    assert_eq!(row.placement, MechanismPlacement::Host);
    #[cfg(not(feature = "cuda"))]
    {
        let metal = describe_for_device(&invocation, DeviceType::Gpu).unwrap();
        assert_eq!(
            metal
                .values
                .iter()
                .find(|value| value.name == "output")
                .unwrap()
                .element,
            TensorElementType::Bf16
        );
        assert!(!metal
            .storage
            .iter()
            .any(|value| value.name == "decoded_weight_row"));
    }
}
