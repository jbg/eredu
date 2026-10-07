//! Exact retained perception parameters execute through native residency policies.
use eredu_architectures::voicechat::{
    perception::{PerceptionError, PerceptionState},
    perception_layered::PerceptionExecution,
};
use eredu_backend_mlx::{
    backend::runtime::checkpoint::binding::binding_bytes, native::materialize_voicechat_perception,
    MlxTensor,
};
use eredu_core::residency::{MemoryTier, OffloadConfig};
use eredu_nn::{RetainedTensorRoots, Tensor};
use eredu_runtime::{DenseDiskStreamLoadOptions, LayerWeightResidency, LayerwiseLoadOptions};
use safemlx::{Device, DeviceType, Stream};
use serde_json::Value;

#[path = "support/voicechat_perception.rs"]
mod fixture_support;
use fixture_support::{checkpoint, dimensions, floats, prepared, tensor};

fn state_values(state: &PerceptionState<MlxTensor>, stream: &Stream) -> Vec<Vec<f32>> {
    let mut values = Vec::new();
    state.visit_retained_tensors(&mut |tensor| values.push(tensor.to_f32_vec(stream).unwrap()));
    values
}

#[test]
fn native_perception_residency_preserves_streaming_state_and_casts_after_abort() {
    let fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::PERCEPTION_REFERENCE_JSON).unwrap();
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let weights = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let directory = tempfile::tempdir().unwrap();
    let dimensions = dimensions(&fixture);
    for f16 in [false, true] {
        let source = checkpoint(
            &fixture,
            f16,
            &directory.path().join(if f16 {
                "f16.safetensors"
            } else {
                "f32.safetensors"
            }),
        );
        let (_, _, restricted, units) = prepared(dimensions, source.clone()).into_parts();
        assert!(!restricted
            .source_keys()
            .iter()
            .any(|key| key == "unrelated.weight"));
        let learned_elements: u64 = ["subsampling", "conformer", "perception"]
            .into_iter()
            .flat_map(|part| fixture[part]["parameters"].as_object().unwrap().values())
            .map(|value| value["values"].as_array().unwrap().len() as u64)
            .sum();
        let largest = units
            .iter()
            .map(|unit| binding_bytes(unit).unwrap())
            .max()
            .unwrap();
        let policies = [
            LayerWeightResidency::FullyResident,
            LayerWeightResidency::LayerwiseHost(LayerwiseLoadOptions::new(
                OffloadConfig::new(Some(largest), Some(1 << 20), 1).unwrap(),
            )),
            LayerWeightResidency::DenseDiskStream(
                DenseDiskStreamLoadOptions::new(largest, 0, 0, 0).unwrap(),
            ),
        ];
        let mut expected = None;
        for residency in policies {
            let (mut model, metadata) = materialize_voicechat_perception(
                prepared(dimensions, source.clone()),
                residency,
                &weights,
                &stream,
            )
            .unwrap();
            assert_eq!(metadata.residency(), residency.execution_residency());
            assert_eq!(metadata.static_device_bytes(), 0);
            assert_eq!(model.static_lease_count(), 0);
            assert_eq!(
                metadata.layer_parameter_bytes(),
                learned_elements * 4,
                "persistent model bytes follow F32 execution after stored precision conversion"
            );
            let mut state = PerceptionState::default();
            let initial = tensor(&fixture["subsampling"]["steps"][0]["input"]);
            // The subsampling unit has executed before excessive overlap is
            // rejected. Its lease must retire safely and permit ordinary reuse.
            assert!(matches!(
                model.forward(&initial, 100, &state, &stream),
                Err(PerceptionError::State(_))
            ));
            assert_eq!(state.frames(), 0);
            assert!(state_values(&state, &stream).is_empty());
            let mut actual = Vec::new();
            for (index, step) in fixture["subsampling"]["steps"]
                .as_array()
                .unwrap()
                .iter()
                .enumerate()
            {
                let result = model
                    .forward(
                        &tensor(&step["input"]),
                        step["drop"].as_u64().unwrap() as usize,
                        &state,
                        &stream,
                    )
                    .unwrap_or_else(|error| {
                        panic!("{residency:?}, F16={f16}, frame={index}: {error}")
                    });
                let encoded = result.encoder.to_f32_vec(&stream).unwrap();
                let projected = result.projected.to_f32_vec(&stream).unwrap();
                assert_eq!(state.frames(), index as u64);
                state = result.state;
                if !f16 {
                    let tolerance = &fixture["tolerance"];
                    let absolute = tolerance["absolute"].as_f64().unwrap() as f32;
                    let relative = tolerance["relative"].as_f64().unwrap() as f32;
                    for (got, reference) in projected.iter().zip(floats(
                        &fixture["perception"]["steps"][index]["projected_output"],
                    )) {
                        assert!(
                            (got - reference).abs() <= absolute + relative * reference.abs(),
                            "{got} versus NVIDIA {reference}"
                        );
                    }
                }
                actual.push((
                    encoded,
                    projected,
                    state_values(&state, &stream),
                    state.frames(),
                    state.blocks()[0].valid_channel(),
                ));
            }
            if let Some(expected) = &expected {
                assert_eq!(&actual, expected);
            } else {
                expected = Some(actual);
            }
            let report = model.residency_report().unwrap();
            assert!(report.weight_store().physical_reads > 0);
            if !residency.is_fully_resident() {
                assert!(
                    report
                        .offload()
                        .peak_resident_bytes()
                        .get(MemoryTier::Device)
                        <= largest
                );
            }
            if matches!(residency, LayerWeightResidency::DenseDiskStream(_)) {
                let dense = model.dense_stream_report().unwrap().unwrap();
                assert!(dense.decode_forwards() > 0);
            }
        }
    }
}

#[path = "voicechat_perception_residency/parallel.rs"]
mod parallel;

#[path = "voicechat_perception_residency/pipeline.rs"]
mod pipeline;
