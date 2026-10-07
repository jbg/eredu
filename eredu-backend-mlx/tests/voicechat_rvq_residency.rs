//! EAR-TTS RVQ code prefixes and feedback use exact bounded native providers.
#[path = "support/component_coordinator.rs"]
mod component_coordinator;
use std::{collections::BTreeMap, sync::Arc};

use eredu_architectures::{
    artifact_preparation::PhysicalArtifactId,
    voicechat::{
        artifact::ParameterAdmission,
        checkpoint::Component,
        tts_mog::TtsError,
        tts_rvq::{RefinementPolicy, RvqDimensions},
        tts_rvq_layered::RvqExecution,
        tts_rvq_preparation::PreparedRvqBindings,
    },
};
use eredu_backend_mlx::{
    backend::runtime::checkpoint::binding::binding_bytes, native::materialize_voicechat_rvq,
    MlxTensor,
};
use eredu_checkpoint::{
    recipe::DerivedWeightRecipe,
    store::{SafetensorsWeightStore, SharedCheckpointSource, TensorSelection},
};
use eredu_core::residency::{MemoryTier, OffloadConfig};
use eredu_nn::{Index, Tensor};
use eredu_runtime::{DenseDiskStreamLoadOptions, LayerWeightResidency, LayerwiseLoadOptions};
use safemlx::{Array, Device, DeviceType, Stream};
use safetensors::tensor::{serialize_to_file, Dtype, TensorView};
use serde_json::Value;

fn floats(value: &Value) -> Vec<f32> {
    value["values"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect()
}
fn tensor(value: &Value) -> MlxTensor {
    let shape: Vec<i32> = value["shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|n| n.as_i64().unwrap() as i32)
        .collect();
    MlxTensor::from_array(Array::from_slice(&floats(value), &shape))
}
fn dimensions(value: &Value) -> RvqDimensions {
    let i = |name: &str| value[name].as_i64().unwrap() as i32;
    RvqDimensions {
        cardinality: i("codebook_size"),
        latent: i("latent"),
        hidden: i("hidden"),
    }
}
fn checkpoint(fixture: &Value, f16: bool, path: &std::path::Path) -> SharedCheckpointSource {
    let mut values: Vec<_> = fixture["parameters"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(name, value)| {
            let shape: Vec<usize> = value["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();
            let bytes: Vec<u8> = if f16 {
                floats(value)
                    .into_iter()
                    .flat_map(|v| half::f16::from_f32(v).to_bits().to_le_bytes())
                    .collect()
            } else {
                floats(value)
                    .into_iter()
                    .flat_map(f32::to_le_bytes)
                    .collect()
            };
            (name.clone(), shape, bytes)
        })
        .collect();
    values.push((
        "unrelated.weight".into(),
        vec![1],
        vec![0; if f16 { 2 } else { 4 }],
    ));
    let dtype = if f16 { Dtype::F16 } else { Dtype::F32 };
    serialize_to_file(
        values.iter().map(|(name, shape, bytes)| {
            (
                name.as_str(),
                TensorView::new(dtype, shape.clone(), bytes).unwrap(),
            )
        }),
        None,
        path,
    )
    .unwrap();
    Arc::new(SafetensorsWeightStore::open(path).unwrap())
}
fn prepared(dimensions: RvqDimensions, source: SharedCheckpointSource) -> PreparedRvqBindings {
    let parameters: BTreeMap<_, _> = source
        .source_keys()
        .into_iter()
        .map(|name| {
            let metadata = source.source_metadata(&name).unwrap();
            let recipe = DerivedWeightRecipe::Source {
                key: name.clone(),
                selection: TensorSelection::Full,
            };
            let output = recipe.infer(source.as_ref()).unwrap();
            let peak = recipe.peak_materialization_bytes(source.as_ref()).unwrap()
                + metadata.encoded_byte_len;
            (
                name.clone(),
                ParameterAdmission {
                    name,
                    component: Component::Tts,
                    artifact: PhysicalArtifactId::Primary,
                    output,
                    stored_bytes: metadata.encoded_byte_len,
                    conversion_workspace_bytes: 0,
                    materialization_peak_bytes: peak,
                    recipe,
                },
            )
        })
        .collect();
    let reads = source.source_diagnostics().unwrap().physical_reads;
    let bindings =
        PreparedRvqBindings::from_parameters(dimensions, source.clone(), &parameters).unwrap();
    assert_eq!(
        source.source_diagnostics().unwrap().physical_reads,
        reads,
        "cold binding projection reads no payload"
    );
    bindings
}
fn compare(actual: &[f32], expected: &Value, fixture: &Value) {
    let expected = floats(expected);
    assert_eq!(actual.len(), expected.len());
    let absolute = fixture["comparison"]["absolute_tolerance"]
        .as_f64()
        .unwrap() as f32;
    let relative = fixture["comparison"]["relative_tolerance"]
        .as_f64()
        .unwrap() as f32;
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= absolute + relative * expected.abs(),
            "{index}: {actual} versus pinned NVIDIA {expected}"
        );
    }
}
fn codes(value: &Value) -> Vec<i32> {
    value["values"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap() as i32)
        .collect()
}

#[test]
fn native_rvq_residency_preserves_every_prefix_feedback_and_casts_after_rejection() {
    let fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::MOG_REFERENCE_JSON).unwrap();
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let weights = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let directory = tempfile::tempdir().unwrap();
    let dimensions = dimensions(&fixture["dimensions"]);
    let case = &fixture["released_policy"];
    let steps = case["steps"].as_array().unwrap();
    let hidden = tensor(&case["hidden"]);
    let conditional = hidden.index(&[Index::Range(0, 1)], &stream).unwrap();
    let unconditional = hidden.index(&[Index::Range(1, 2)], &stream).unwrap();
    let policy = RefinementPolicy {
        guidance_scale: 0.2,
        noise_scale: 0.001,
        top_p: 0.95,
    };
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
        let (_, _, layout, restricted, units) = prepared(dimensions, source.clone()).into_parts();
        let expected_keys = [
            "tts_model.tts_model.rvq_embs",
            "tts_model.tts_model.embed_code.weight",
        ];
        assert_eq!(restricted.source_keys().len(), expected_keys.len());
        assert!(
            restricted
                .source_keys()
                .iter()
                .all(|key| expected_keys.contains(&key.as_str())),
            "neighboring MoG and unrelated tensors must be inaccessible"
        );
        assert_eq!(
            units.len(),
            2,
            "the stacked bank has one canonical parameter owner"
        );
        assert!(
            (0..layout.group_count()).all(|group| layout.group_range(group).unwrap().len() == 1)
        );
        let learned_elements: u64 = expected_keys
            .iter()
            .map(|key| {
                fixture["parameters"][*key]["values"]
                    .as_array()
                    .unwrap()
                    .len() as u64
            })
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
            // Each group contains one unit, so the disk window needs only that unit.
            LayerWeightResidency::DenseDiskStream(
                DenseDiskStreamLoadOptions::new(largest, 0, 0, 0).unwrap(),
            ),
        ];
        let mut expected = None;
        for residency in policies {
            let (mut model, metadata) = materialize_voicechat_rvq(
                prepared(dimensions, source.clone()),
                residency,
                &weights,
                &stream,
            )
            .unwrap();
            assert_eq!(metadata.static_device_bytes(), 0);
            assert_eq!(model.static_lease_count(), 0);
            assert_eq!(
                metadata.layer_parameter_bytes(),
                learned_elements * 4,
                "execution storage includes F16 to F32 conversion"
            );
            let initial = model.start(1, 1, &stream).unwrap();
            assert_eq!(
                initial.codes().to_i32_vec(&stream).unwrap(),
                vec![dimensions.cardinality; 31]
            );
            let invalid = MlxTensor::full_f32(0., &[1, 1, dimensions.latent - 1], &stream).unwrap();
            assert!(matches!(
                model.advance(&initial, &invalid, &stream),
                Err(TtsError::Shape(_))
            ));
            let first = model
                .advance(&initial, &tensor(&steps[0]["latent"]), &stream)
                .unwrap();
            if !f16 {
                assert_eq!(
                    first.codes().to_i32_vec(&stream).unwrap(),
                    codes(&steps[0]["codes_after"])
                );
            }
            assert_eq!(initial.completed_depth(), 0);
            assert_eq!(
                initial.codes().to_i32_vec(&stream).unwrap(),
                vec![dimensions.cardinality; 31]
            );
            if let Some(report) = model.dense_stream_report().unwrap() {
                assert_eq!(
                    report
                        .execution_groups()
                        .iter()
                        .filter(|group| group.completed_executions() > 0)
                        .count(),
                    1,
                    "advance executes only the stacked bank"
                );
                assert_eq!(
                    report
                        .execution_groups()
                        .iter()
                        .filter(|group| group.peak_device_bytes() == 0)
                        .count(),
                    1,
                    "advance does not acquire the projection"
                );
            }
            let malformed =
                MlxTensor::full_i32(dimensions.cardinality, &[1, 1, 30], &stream).unwrap();
            assert!(matches!(
                model.depthsum(&malformed, &stream),
                Err(TtsError::Shape(_))
            ));
            assert_eq!(
                model
                    .depthsum(initial.codes(), &stream)
                    .unwrap()
                    .to_f32_vec(&stream)
                    .unwrap(),
                vec![0.; dimensions.latent as usize]
            );
            let mut state = initial.clone();
            let mut actual = Vec::new();
            for (index, step) in steps.iter().enumerate() {
                let parent = state.clone();
                let before_codes = parent.codes().to_i32_vec(&stream).unwrap();
                let embedding = model
                    .embed_codes(state.codes(), &stream)
                    .unwrap()
                    .to_f32_vec(&stream)
                    .unwrap();
                let input = model
                    .input(&state, &conditional, Some(&unconditional), policy, &stream)
                    .unwrap()
                    .to_f32_vec(&stream)
                    .unwrap();
                if !f16 {
                    compare(&input, &step["input"], &fixture);
                }
                state = model
                    .advance(&state, &tensor(&step["latent"]), &stream)
                    .unwrap_or_else(|error| panic!("{residency:?} F16={f16} slot{index}: {error}"));
                let after_codes = state.codes().to_i32_vec(&stream).unwrap();
                let sum = model
                    .depthsum(state.codes(), &stream)
                    .unwrap()
                    .to_f32_vec(&stream)
                    .unwrap();
                assert_eq!(
                    parent.codes().to_i32_vec(&stream).unwrap(),
                    before_codes,
                    "failed or successful successor work cannot change a retained parent"
                );
                assert_eq!(
                    &after_codes[..parent.completed_depth()],
                    &before_codes[..parent.completed_depth()],
                    "refinement preserves assigned prefix"
                );
                if !f16 {
                    assert_eq!(after_codes, codes(&step["codes_after"]));
                    compare(&sum, &step["summed_feedback"], &fixture);
                }
                actual.push((
                    embedding,
                    input,
                    after_codes,
                    sum,
                    state.completed_depth(),
                    state.next_slot(),
                ));
            }
            assert_eq!(state.completed_depth(), 31);
            assert_eq!(state.next_slot(), None);
            if let Some(expected) = &expected {
                assert_eq!(&actual, expected, "residency parity F16={f16}");
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
            if let Some(report) = model.dense_stream_report().unwrap() {
                assert!(report
                    .execution_groups()
                    .iter()
                    .all(|group| group.completed_executions() > 0));
                assert!(report.decode_forwards() > 0);
            }
        }
    }
}

#[path = "voicechat_rvq_residency/parallel.rs"]
mod parallel;
