//! PRVQ range selection and residual arithmetic use independently leased codebooks.
use std::{collections::BTreeMap, sync::Arc};

use eredu_backend_mlx::{
    backend::runtime::checkpoint::binding::binding_bytes, native::materialize_prvq, MlxTensor,
};
use eredu_checkpoint::{
    recipe::DerivedWeightRecipe,
    store::{SafetensorsWeightStore, SharedCheckpointSource, StoreError, TensorSelection},
};
use eredu_codec::{
    prvq::{DistanceMetric, Geometry},
    prvq_layered::QuantizerExecution,
    prvq_preparation::{PreparedQuantizerBindings, QuantizerPreparationError},
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
fn recipes(source: &SharedCheckpointSource) -> BTreeMap<String, DerivedWeightRecipe> {
    source
        .source_keys()
        .into_iter()
        .map(|name| {
            (
                name.clone(),
                DerivedWeightRecipe::Source {
                    key: name,
                    selection: TensorSelection::Full,
                },
            )
        })
        .collect()
}
fn prepared(geometry: Geometry, source: SharedCheckpointSource) -> PreparedQuantizerBindings {
    let recipes = recipes(&source);
    let reads = source.source_diagnostics().unwrap().physical_reads;
    let prepared = PreparedQuantizerBindings::from_recipes(
        geometry,
        "tts_model.audio_codec.prvq",
        source.clone(),
        &recipes,
    )
    .unwrap();
    assert_eq!(
        source.source_diagnostics().unwrap().physical_reads,
        reads,
        "cold quantizer preparation reads no payload"
    );
    prepared
}
fn compare(actual: &[f32], expected: &Value, fixture: &Value) {
    let expected = floats(expected);
    assert_eq!(actual.len(), expected.len());
    let absolute = fixture["tolerance"]["absolute"].as_f64().unwrap() as f32;
    let relative = fixture["tolerance"]["relative"].as_f64().unwrap() as f32;
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
fn prvq_preparation_requires_variance_and_exact_mean_geometry_without_payload_reads() {
    let fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::PRVQ_REFERENCE_JSON).unwrap();
    let directory = tempfile::tempdir().unwrap();
    let source = checkpoint(&fixture, false, &directory.path().join("cold.safetensors"));
    let geometry = Geometry {
        depth: 3,
        cardinality: 5,
        channels: 4,
    };
    let mut recipes = recipes(&source);
    let variance = "tts_model.audio_codec.prvq._variance_list.0.variance";
    let mean = "tts_model.audio_codec.prvq.mus_list.0";
    let reads = source.source_diagnostics().unwrap().physical_reads;
    let prepare = |recipes: &BTreeMap<String, DerivedWeightRecipe>| {
        PreparedQuantizerBindings::from_recipes(
            geometry,
            "tts_model.audio_codec.prvq",
            source.clone(),
            recipes,
        )
    };

    let variance_recipe = recipes.remove(variance).unwrap();
    assert!(matches!(prepare(&recipes),
        Err(QuantizerPreparationError::MissingParameter(name)) if name == variance));
    assert_eq!(source.source_diagnostics().unwrap().physical_reads, reads);

    recipes.insert(variance.into(), variance_recipe.clone());
    let mean_recipe = recipes.insert(mean.into(), variance_recipe).unwrap();
    assert!(matches!(prepare(&recipes),
        Err(QuantizerPreparationError::Geometry(name)) if name == mean));
    assert_eq!(source.source_diagnostics().unwrap().physical_reads, reads);

    recipes.insert(mean.into(), mean_recipe);
    let (_, _, _, _, restricted, _) = prepare(&recipes).unwrap().into_parts();
    assert_eq!(
        restricted.source_metadata(mean).unwrap(),
        source.source_metadata(mean).unwrap()
    );
    assert_eq!(
        restricted.source_metadata(variance).unwrap(),
        source.source_metadata(variance).unwrap()
    );
    assert!(matches!(restricted.source_metadata("unrelated.weight"),
        Err(StoreError::UnauthorizedTensor { key, .. }) if key == "unrelated.weight"));
    assert_eq!(source.source_diagnostics().unwrap().physical_reads, reads);
}

#[test]
fn native_prvq_residency_preserves_ranges_casts_and_skips_unselected_depths() {
    let fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::PRVQ_REFERENCE_JSON).unwrap();
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let weights = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let directory = tempfile::tempdir().unwrap();
    let geometry = Geometry {
        depth: 3,
        cardinality: 5,
        channels: 4,
    };
    let latent = tensor(&fixture["input"]);
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
        let (_, _, _, layout, restricted, units) = prepared(geometry, source.clone()).into_parts();
        let expected_keys = fixture["parameters"].as_object().unwrap();
        assert_eq!(restricted.source_keys().len(), expected_keys.len());
        assert!(restricted
            .source_keys()
            .iter()
            .all(|key| expected_keys.contains_key(key)));
        assert_eq!(units.len(), geometry.depth);
        assert!(
            (0..layout.group_count()).all(|group| layout.group_range(group).unwrap().len() == 1)
        );
        let learned_elements: u64 = expected_keys
            .values()
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
            let (mut model, metadata) = materialize_prvq(
                prepared(geometry, source.clone()),
                residency,
                &weights,
                &stream,
            )
            .unwrap();
            assert_eq!(metadata.static_device_bytes(), 0);
            assert_eq!(model.static_lease_count(), 0);
            assert_eq!(metadata.layer_parameter_bytes(), learned_elements * 4);
            assert!(matches!(
                model.quantize_range(&latent, 1..4, DistanceMetric::SquaredEuclidean, &stream),
                Err(eredu_codec::Error::InvalidShape(_))
            ));
            let middle = model
                .quantize_range(
                    &latent,
                    1..2,
                    DistanceMetric::CodebookNormMinusTwiceDot,
                    &stream,
                )
                .unwrap();
            let middle_decoded = model.decode_range(&middle.codes, 1..2, &stream).unwrap();
            if let Some(report) = model.dense_stream_report().unwrap() {
                assert_eq!(
                    report
                        .execution_groups()
                        .iter()
                        .filter(|group| group.completed_executions() > 0)
                        .count(),
                    1,
                    "middle-range execution acquires only its codebook"
                );
                assert_eq!(
                    report
                        .execution_groups()
                        .iter()
                        .filter(|group| group.peak_device_bytes() == 0)
                        .count(),
                    2,
                    "unselected depths stay unloaded"
                );
            }
            let full = model.quantize(&latent, &stream).unwrap();
            let decoded = model.decode(&full.codes, &stream).unwrap();
            let prefix_codes = full
                .codes
                .index(&[Index::Full, Index::Full, Index::Range(0, 2)], &stream)
                .unwrap();
            let prefix = model.decode_range(&prefix_codes, 0..2, &stream).unwrap();
            if !f16 {
                assert_eq!(
                    full.codes.to_i32_vec(&stream).unwrap(),
                    codes(&fixture["codes"])
                );
                for (actual, key) in [
                    (&full.quantized, "quantized"),
                    (&full.residual, "residual"),
                    (&decoded, "decoded"),
                    (&prefix, "prefix_decoded"),
                ] {
                    compare(
                        &actual.to_f32_vec(&stream).unwrap(),
                        &fixture[key],
                        &fixture,
                    );
                }
            }
            let actual = (
                middle.codes.to_i32_vec(&stream).unwrap(),
                middle.residual.to_f32_vec(&stream).unwrap(),
                middle.quantized.to_f32_vec(&stream).unwrap(),
                middle_decoded.to_f32_vec(&stream).unwrap(),
                full.codes.to_i32_vec(&stream).unwrap(),
                full.residual.to_f32_vec(&stream).unwrap(),
                full.quantized.to_f32_vec(&stream).unwrap(),
                decoded.to_f32_vec(&stream).unwrap(),
                prefix.to_f32_vec(&stream).unwrap(),
            );
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
        }
    }
}

#[path = "voicechat_codec_quantizer_residency/parallel.rs"]
mod parallel;
