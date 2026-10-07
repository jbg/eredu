//! Finite EAR waveform encoding uses independently leased neural units and borrowed codebooks.
use std::{collections::BTreeMap, sync::Arc};

use eredu_backend_mlx::{
    backend::nn::shared::MlxNeuralBackend as Backend,
    backend::runtime::checkpoint::binding::binding_bytes,
    native::{materialize_ear_encoder, materialize_prvq},
    MlxTensor,
};
use eredu_checkpoint::{
    recipe::DerivedWeightRecipe,
    store::{SafetensorsWeightStore, SharedCheckpointSource, StoreError, TensorSelection},
};
use eredu_codec::{
    ear::{AudioEncoder, AudioEncoderConfig},
    ear_encoder::EncoderConfig,
    ear_encoder_preparation::{EncoderPreparationError, PreparedEncoderBindings},
    ear_spectral::SpectralConfig,
    prvq::Geometry,
    prvq_layered::QuantizerExecution,
    prvq_preparation::PreparedQuantizerBindings,
};
use eredu_core::residency::{MemoryTier, OffloadConfig};
use eredu_nn::Tensor;
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
fn geometry(v: &Value) -> EncoderConfig {
    let i = |key: &str| v[key].as_i64().unwrap() as i32;
    EncoderConfig {
        latent_size: i("latent_size"),
        base_hidden_size: i("base_hidden_size"),
        channel_mult: serde_json::from_value(v["channel_mult"].clone()).unwrap(),
        rates: serde_json::from_value(v["rates"].clone()).unwrap(),
        blocks_per_stage: i("num_blocks") as usize,
        kernel_size: i("kernel_size"),
        fft_size: i("n_fft"),
        groups: i("groups"),
    }
}
fn spectral(v: &Value) -> SpectralConfig {
    SpectralConfig {
        fft_size: v["n_fft"].as_i64().unwrap() as i32,
        hop_length: v["hop_length"].as_i64().unwrap() as i32,
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
fn prepared(config: EncoderConfig, source: SharedCheckpointSource) -> PreparedEncoderBindings {
    let recipes = recipes(&source);
    let reads = source.source_diagnostics().unwrap().physical_reads;
    let prepared = PreparedEncoderBindings::from_recipes(
        config,
        "tts_model.audio_codec.encoder",
        source.clone(),
        &recipes,
    )
    .unwrap();
    assert_eq!(
        source.source_diagnostics().unwrap().physical_reads,
        reads,
        "cold codec preparation reads no payload"
    );
    prepared
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
fn native_codec_encoder_residency_preserves_finite_latent_and_codes_after_rejection() {
    let fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::CODEC_ENCODER_REFERENCE_JSON).unwrap();
    let case = &fixture["encoders"]["compact"];
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let weights = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let directory = tempfile::tempdir().unwrap();
    let config = AudioEncoderConfig {
        quantizer: Geometry {
            depth: 3,
            cardinality: 5,
            channels: 4,
        },
        neural: geometry(&case["geometry"]),
        spectral: spectral(&case["geometry"]),
    };
    let pcm = tensor(&case["pcm"]);
    for f16 in [false, true] {
        let source = checkpoint(
            case,
            f16,
            &directory.path().join(if f16 {
                "f16.safetensors"
            } else {
                "f32.safetensors"
            }),
        );
        let (_, _, _, layout, restricted, units) =
            prepared(config.neural.clone(), source.clone()).into_parts();
        let expected_keys: Vec<_> = case["parameters"]
            .as_object()
            .unwrap()
            .keys()
            .filter(|key| key.starts_with("tts_model.audio_codec.encoder."))
            .collect();
        assert_eq!(restricted.source_keys().len(), expected_keys.len());
        assert!(
            restricted
                .source_keys()
                .iter()
                .all(|key| expected_keys.contains(&key)),
            "codebooks and neighbors stay outside the encoder source authority"
        );
        let learned_elements: u64 = expected_keys
            .iter()
            .map(|key| {
                case["parameters"][key.as_str()]["values"]
                    .as_array()
                    .unwrap()
                    .len() as u64
            })
            .sum();
        let unit_bytes: Vec<_> = units
            .iter()
            .map(|unit| binding_bytes(unit).unwrap())
            .collect();
        let largest = *unit_bytes.iter().max().unwrap();
        let depth = LayerWeightResidency::DenseDiskStream(
            DenseDiskStreamLoadOptions::new(u64::MAX, 0, 0, 0).unwrap(),
        )
        .device_depth(units.len());
        let disk_budget = (0..layout.group_count())
            .flat_map(|group| {
                let range = layout.group_range(group).unwrap();
                unit_bytes[range.clone()]
                    .windows(depth.min(range.len()))
                    .map(|window| window.iter().sum::<u64>())
                    .collect::<Vec<_>>()
            })
            .max()
            .unwrap();
        let quantizer_prepared = || {
            PreparedQuantizerBindings::from_recipes(
                config.quantizer,
                "tts_model.audio_codec.prvq",
                source.clone(),
                &recipes(&source),
            )
            .unwrap()
        };
        let (_, _, _, _, quantizer_source, quantizer_units) = quantizer_prepared().into_parts();
        assert_eq!(
            quantizer_source.source_keys().len(),
            config.quantizer.depth * 2
        );
        assert!(quantizer_source
            .source_keys()
            .iter()
            .all(|key| key.starts_with("tts_model.audio_codec.prvq.")));
        let quantizer_budget = quantizer_units
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
                DenseDiskStreamLoadOptions::new(disk_budget, 0, 0, 0).unwrap(),
            ),
        ];
        let mut expected = None;
        for residency in policies {
            let (neural, metadata) = materialize_ear_encoder(
                prepared(config.neural.clone(), source.clone()),
                residency,
                &weights,
                &stream,
            )
            .unwrap();
            assert_eq!(metadata.static_device_bytes(), 0);
            assert_eq!(neural.static_lease_count(), 0);
            assert_eq!(metadata.layer_parameter_bytes(), learned_elements * 4);
            let quantizer_residency = match residency {
                LayerWeightResidency::FullyResident => LayerWeightResidency::FullyResident,
                LayerWeightResidency::LayerwiseHost(_) => {
                    LayerWeightResidency::LayerwiseHost(LayerwiseLoadOptions::new(
                        OffloadConfig::new(Some(quantizer_budget), Some(1 << 20), 1).unwrap(),
                    ))
                }
                _ => LayerWeightResidency::DenseDiskStream(
                    DenseDiskStreamLoadOptions::new(quantizer_budget, 0, 0, 0).unwrap(),
                ),
            };
            let (mut quantizer, quantizer_metadata) =
                materialize_prvq(quantizer_prepared(), quantizer_residency, &weights, &stream)
                    .unwrap();
            assert_eq!(quantizer_metadata.static_device_bytes(), 0);
            assert_eq!(quantizer.static_lease_count(), 0);
            assert_eq!(
                quantizer_metadata.layer_parameter_bytes(),
                config.quantizer.depth as u64
                    * (config.quantizer.cardinality as u64 * config.quantizer.channels as u64 + 1)
                    * 4
            );
            let mut model =
                AudioEncoder::<Backend, _>::from_parts(config.clone(), neural, &stream).unwrap();
            let invalid = MlxTensor::full_f32(
                0.,
                &[1, 1, config.samples_per_frame().unwrap() - 1],
                &stream,
            )
            .unwrap();
            let before = source.source_diagnostics().unwrap().physical_reads;
            assert!(matches!(
                model.encode(&invalid, &mut quantizer, &stream),
                Err(eredu_codec::Error::InvalidShape(_))
            ));
            assert_eq!(
                source.source_diagnostics().unwrap().physical_reads,
                before,
                "malformed finite PCM is rejected before encoder or codebook payload reads"
            );
            let latent = model.latent(&pcm, &stream).unwrap();
            let encoded = model.encode(&pcm, &mut quantizer, &stream).unwrap();
            let decoded = quantizer.decode(&encoded, &stream).unwrap();
            let actual = (
                latent.to_f32_vec(&stream).unwrap(),
                encoded.to_i32_vec(&stream).unwrap(),
                decoded.to_f32_vec(&stream).unwrap(),
            );
            if !f16 {
                compare(&actual.0, &case["latent"], &fixture);
                assert_eq!(actual.1, codes(&case["codes"]));
                compare(&actual.2, &case["decoded"], &fixture);
            }
            if let Some(expected) = &expected {
                assert_eq!(
                    &actual, expected,
                    "finite encoder and borrowed quantizer residency parity"
                );
            } else {
                expected = Some(actual);
            }
            let report = model.neural.residency_report().unwrap();
            assert!(report.weight_store().physical_reads > 0);
            if let Some(dense) = model.neural.dense_stream_report().unwrap() {
                assert_eq!(dense.prefill_forwards() + dense.decode_forwards(), 2);
            }
            if !residency.is_fully_resident() {
                let budget = if model.neural.dense_stream_report().unwrap().is_some() {
                    disk_budget
                } else {
                    largest
                };
                assert!(
                    report
                        .offload()
                        .peak_resident_bytes()
                        .get(MemoryTier::Device)
                        <= budget
                );
                assert!(
                    quantizer
                        .residency_report()
                        .unwrap()
                        .offload()
                        .peak_resident_bytes()
                        .get(MemoryTier::Device)
                        <= quantizer_budget
                );
            }
        }
    }
}

#[test]
fn encoder_preparation_rejects_missing_or_wrong_shaped_layers_without_payload_reads() {
    let fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::CODEC_ENCODER_REFERENCE_JSON).unwrap();
    let case = &fixture["encoders"]["compact"];
    let directory = tempfile::tempdir().unwrap();
    let source = checkpoint(case, false, &directory.path().join("cold.safetensors"));
    let mut recipes = recipes(&source);
    let required = "tts_model.audio_codec.encoder.layers.0.weight";
    let saved = recipes.remove(required).unwrap();
    let reads = source.source_diagnostics().unwrap().physical_reads;
    let prepare = |recipes: &BTreeMap<String, DerivedWeightRecipe>| {
        PreparedEncoderBindings::from_recipes(
            geometry(&case["geometry"]),
            "tts_model.audio_codec.encoder",
            source.clone(),
            recipes,
        )
    };
    assert!(
        matches!(prepare(&recipes), Err(EncoderPreparationError::MissingParameter(name)) if name == required)
    );
    assert_eq!(source.source_diagnostics().unwrap().physical_reads, reads);
    recipes.insert(
        required.into(),
        DerivedWeightRecipe::Source {
            key: "unrelated.weight".into(),
            selection: TensorSelection::Full,
        },
    );
    assert!(
        matches!(prepare(&recipes), Err(EncoderPreparationError::Geometry(name)) if name == required)
    );
    assert_eq!(source.source_diagnostics().unwrap().physical_reads, reads);
    recipes.insert(required.into(), saved);
    let (_, _, _, _, restricted, _) = prepare(&recipes).unwrap().into_parts();
    assert_eq!(
        restricted.source_metadata(required).unwrap(),
        source.source_metadata(required).unwrap()
    );
    assert!(matches!(restricted.source_metadata("unrelated.weight"),
        Err(StoreError::UnauthorizedTensor { key, .. }) if key == "unrelated.weight"));
    assert_eq!(source.source_diagnostics().unwrap().physical_reads, reads);
}

#[path = "voicechat_codec_encoder_residency/parallel.rs"]
mod parallel;

#[path = "voicechat_codec_encoder_residency/pipeline.rs"]
mod pipeline;
