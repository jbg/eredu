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
        for pair in output.as_slice::<f32>().chunks_exact(2) {
            assert!(
                (pair[0] - 2.0).abs() < 2.0e-5 && (pair[1] - 4.0).abs() < 4.0e-5,
                "queries={query_count}, keys={key_count}: {pair:?}"
            );
        }
    }
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
