//! EAR neural decoder providers preserve cached convolution and waveform drain behavior.
use std::sync::Arc;

use eredu_backend_mlx::{
    backend::nn::shared::MlxNeuralBackend as Backend,
    backend::runtime::checkpoint::binding::binding_bytes,
    native::{materialize_ear_decoder, materialize_prvq},
    MlxTensor,
};
use eredu_checkpoint::{
    recipe::DerivedWeightRecipe,
    store::{SafetensorsWeightStore, SharedCheckpointSource, TensorSelection},
};
use eredu_codec::{
    ear::{AudioDecoder, AudioDecoderConfig, AudioDecoderExecution, AudioDecoderState},
    ear_decoder::DecoderConfig,
    ear_decoder_layered::DecoderExecution,
    ear_decoder_preparation::PreparedDecoderBindings,
    ear_spectral::SpectralConfig,
    prvq::Geometry,
    prvq_preparation::PreparedQuantizerBindings,
};
use eredu_core::residency::{MemoryTier, OffloadConfig};
use eredu_nn::{Index, RetainedTensorRoots, Tensor};
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
fn geometry(v: &Value) -> DecoderConfig {
    let i = |key: &str| v[key].as_i64().unwrap() as i32;
    DecoderConfig {
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
fn prepared(config: DecoderConfig, source: SharedCheckpointSource) -> PreparedDecoderBindings {
    let recipes = source
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
        .collect();
    let reads = source.source_diagnostics().unwrap().physical_reads;
    let prepared = PreparedDecoderBindings::from_recipes(
        config,
        "tts_model.audio_codec.decoder",
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

fn retained(state: &AudioDecoderState<MlxTensor>, context: &Stream) -> Vec<(Vec<i32>, Vec<f32>)> {
    let mut values = Vec::new();
    state.visit_retained_tensors(&mut |tensor| {
        values.push((tensor.shape().to_vec(), tensor.to_f32_vec(context).unwrap()))
    });
    values
}
fn code_tensor(value: &Value, context: &Stream) -> MlxTensor {
    let shape: Vec<i32> = value["shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap() as i32)
        .collect();
    MlxTensor::from_i32_slice(&codes(value), &shape, context).unwrap()
}

#[test]
fn native_codec_decoder_residency_preserves_caches_waveforms_and_flush_after_rejection() {
    let fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::CODEC_REFERENCE_JSON).unwrap();
    let case = &fixture["composition"];
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let weights = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let directory = tempfile::tempdir().unwrap();
    let config = AudioDecoderConfig {
        quantizer: Geometry {
            depth: 3,
            cardinality: 5,
            channels: 4,
        },
        neural: geometry(&case["geometry"]),
        spectral: spectral(&case["geometry"]),
    };
    let all = code_tensor(&case["codes"], &stream);
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
            .filter(|key| key.starts_with("tts_model.audio_codec.decoder."))
            .collect();
        assert_eq!(restricted.source_keys().len(), expected_keys.len());
        assert!(
            restricted
                .source_keys()
                .iter()
                .all(|key| expected_keys.contains(&key)),
            "quantizer and unrelated parameters stay outside neural decoder source authority"
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
                unit_bytes[layout.group_range(group).unwrap()]
                    .windows(depth.min(layout.group_range(group).unwrap().len()))
                    .map(|window| window.iter().sum::<u64>())
                    .collect::<Vec<_>>()
            })
            .max()
            .unwrap();
        let quantizer_prepared = || {
            let recipes = source
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
                .collect();
            PreparedQuantizerBindings::from_recipes(
                config.quantizer,
                "tts_model.audio_codec.prvq",
                source.clone(),
                &recipes,
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
            let (neural, metadata) = materialize_ear_decoder(
                prepared(config.neural.clone(), source.clone()),
                residency,
                &weights,
                &stream,
            )
            .unwrap();
            assert_eq!(metadata.static_device_bytes(), 0);
            assert_eq!(neural.static_lease_count(), 0);
            assert_eq!(metadata.layer_parameter_bytes(), learned_elements * 4);
            // Neural units and codebooks have independent providers and ledgers;
            // the combined waveform path reuses both without retaining all depths.
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
            let (quantizer, quantizer_metadata) =
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
            let mut model = AudioDecoder::<Backend, _, _>::from_parts(
                config.clone(),
                quantizer,
                neural,
                &stream,
            )
            .unwrap();
            let raw = model
                .neural
                .forward(&tensor(&case["latent"]), &stream)
                .unwrap();
            let full = model.forward(&all, true, &stream).unwrap();
            if !f16 {
                compare(
                    &raw.to_f32_vec(&stream).unwrap(),
                    &case["full"][0]["raw"],
                    &fixture,
                );
                compare(
                    &full.to_f32_vec(&stream).unwrap(),
                    &case["full"][0]["waveform"],
                    &fixture,
                );
            }
            let initial = model.initial_state(1).unwrap();
            let invalid =
                MlxTensor::full_f32(0., &[1, 1, config.neural.latent_size + 1], &stream).unwrap();
            assert!(matches!(
                model.neural.step(&invalid, initial.neural(), &stream),
                Err(eredu_codec::Error::InvalidShape(_))
            ));
            let mut state = initial.clone();
            let mut frames = 0;
            let mut actual = Vec::new();
            for (index, step) in case["streaming_unflushed"]
                .as_array()
                .unwrap()
                .iter()
                .enumerate()
            {
                let count = step["input"]["shape"][1].as_i64().unwrap() as i32;
                let input = all
                    .index(
                        &[
                            Index::Full,
                            Index::Range(frames, frames + count),
                            Index::Full,
                        ],
                        &stream,
                    )
                    .unwrap();
                let parent = state.clone();
                let before = retained(&parent, &stream);
                let (pcm, next) = model
                    .step(&input, &state, true, &stream)
                    .unwrap_or_else(|error| panic!("{residency:?} F16={f16} step{index}: {error}"));
                assert_eq!(
                    retained(&parent, &stream),
                    before,
                    "candidate execution preserves parent caches"
                );
                if !f16 {
                    compare(
                        &pcm.to_f32_vec(&stream).unwrap(),
                        &step["waveform"],
                        &fixture,
                    );
                    for (index, carry) in next
                        .neural()
                        .units()
                        .iter()
                        .filter_map(|state| state.carry())
                        .enumerate()
                    {
                        compare(
                            &carry.to_f32_vec(&stream).unwrap(),
                            &step["cache"][index.to_string()],
                            &fixture,
                        );
                    }
                }
                frames += count;
                state = next;
                actual.push((
                    pcm.to_f32_vec(&stream).unwrap(),
                    retained(&state, &stream),
                    state.frames_seen(),
                    state.samples_emitted(),
                ));
            }
            let reads = model
                .neural
                .residency_report()
                .unwrap()
                .weight_store()
                .physical_reads;
            let (tail, finished) = model.finish(&state, true, &stream).unwrap();
            let tail = tail.unwrap();
            assert_eq!(
                model
                    .neural
                    .residency_report()
                    .unwrap()
                    .weight_store()
                    .physical_reads,
                reads,
                "drain acquires no learned decoder weights"
            );
            if !f16 {
                compare(
                    &tail.to_f32_vec(&stream).unwrap(),
                    &case["finish_tail"],
                    &fixture,
                );
            }
            assert!(finished.is_finished());
            assert_eq!(finished.frames_seen(), frames as u64);
            assert_eq!(initial.frames_seen(), 0);
            assert!(retained(&initial, &stream).is_empty());
            assert!(model.finish(&finished, true, &stream).unwrap().0.is_none());
            assert!(matches!(
                model.step(&all, &finished, true, &stream),
                Err(eredu_codec::Error::StreamFinished)
            ));
            let actual = (
                raw.to_f32_vec(&stream).unwrap(),
                full.to_f32_vec(&stream).unwrap(),
                actual,
                tail.to_f32_vec(&stream).unwrap(),
                retained(&finished, &stream),
                finished.samples_emitted(),
            );
            if let Some(expected) = &expected {
                assert_eq!(&actual, expected, "residency parity F16={f16}");
            } else {
                expected = Some(actual);
            }
            let report = model.neural.residency_report().unwrap();
            assert!(report.weight_store().physical_reads > 0);
            if !residency.is_fully_resident() {
                let budget = if matches!(residency, LayerWeightResidency::DenseDiskStream(_)) {
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
            }
            let quantizer_report = model.quantizer.residency_report().unwrap();
            assert!(quantizer_report.weight_store().physical_reads > 0);
            if !quantizer_residency.is_fully_resident() {
                assert!(
                    quantizer_report
                        .offload()
                        .peak_resident_bytes()
                        .get(MemoryTier::Device)
                        <= quantizer_budget
                );
            }
            if let Some(report) = model.neural.dense_stream_report().unwrap() {
                // Shared dense telemetry classifies physical axis 1. EAR units
                // use [batch, channels, time], so those text-oriented labels do
                // not identify codec frame cadence. Count completed invocations:
                // two finite forwards plus cached steps, with no pass for drain
                // or the rejected malformed latent.
                assert_eq!(
                    report.prefill_forwards() + report.decode_forwards(),
                    2 + case["streaming_unflushed"].as_array().unwrap().len() as u64
                );
            }
        }
    }
}

#[path = "voicechat_codec_decoder_residency/parallel.rs"]
mod parallel;

#[path = "voicechat_codec_decoder_residency/pipeline.rs"]
mod pipeline;

#[path = "voicechat_codec_decoder_residency/waveform.rs"]
mod waveform;
