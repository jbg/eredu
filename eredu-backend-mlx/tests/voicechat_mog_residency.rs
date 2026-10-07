//! EAR-TTS MoG prediction and selected banks use exact bounded native providers.
#[path = "support/component_coordinator.rs"]
mod component_coordinator;
use std::{collections::BTreeMap, sync::Arc};

use eredu_architectures::{
    artifact_preparation::PhysicalArtifactId,
    voicechat::{
        artifact::ParameterAdmission,
        checkpoint::Component,
        tts_mog::{MogDimensions, TtsError},
        tts_mog_layered::MogExecution,
        tts_mog_preparation::PreparedMogBindings,
    },
};
use eredu_backend_mlx::{
    backend::runtime::checkpoint::binding::binding_bytes, native::materialize_voicechat_mog,
    MlxTensor,
};
use eredu_checkpoint::{
    recipe::DerivedWeightRecipe,
    store::{SafetensorsWeightStore, SharedCheckpointSource, TensorSelection},
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
fn dimensions(value: &Value) -> MogDimensions {
    let i = |name: &str| value[name].as_i64().unwrap() as i32;
    MogDimensions {
        hidden: i("hidden"),
        intermediate: i("intermediate"),
        latent: i("latent"),
        mixtures: i("mixtures"),
        rank: i("low_rank"),
        layers: i("mlp_layers") as usize,
        epsilon: value["eps"].as_f64().unwrap() as f32,
        min_log_std: value["min_log_std"].as_f64().unwrap() as f32,
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
fn prepared(dimensions: MogDimensions, source: SharedCheckpointSource) -> PreparedMogBindings {
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
        PreparedMogBindings::from_parameters(dimensions, source.clone(), &parameters).unwrap();
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
fn indices(value: &Value, shape: &[i32], stream: &Stream) -> MlxTensor {
    let values: Vec<i32> = value["values"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap() as i32)
        .collect();
    // NVIDIA records flattened decisions; the execution API retains batch/time axes.
    MlxTensor::from_i32_slice(&values, shape, stream).unwrap()
}

#[test]
fn native_mog_residency_preserves_guidance_selected_draws_and_casts_after_rejection() {
    let fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::MOG_REFERENCE_JSON).unwrap();
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let weights = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let directory = tempfile::tempdir().unwrap();
    let dimensions = dimensions(&fixture["dimensions"]);
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
        let expected_keys: Vec<_> = fixture["parameters"]
            .as_object()
            .unwrap()
            .keys()
            .filter(|key| key.starts_with("tts_model.tts_model.mog_head."))
            .collect();
        assert_eq!(restricted.source_keys().len(), expected_keys.len());
        assert!(
            restricted
                .source_keys()
                .iter()
                .all(|key| expected_keys.contains(&key)),
            "RVQ and unrelated tensors must be inaccessible"
        );
        let learned_elements: u64 = expected_keys
            .iter()
            .map(|key| {
                fixture["parameters"][key.as_str()]["values"]
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
        // Price the shared streamer's actual device window within each group;
        // prediction and selected-bank parameters never share a transfer window.
        let disk_depth = LayerWeightResidency::DenseDiskStream(
            DenseDiskStreamLoadOptions::new(u64::MAX, 0, 0, 0).unwrap(),
        )
        .device_depth(units.len());
        let disk_budget = (0..layout.group_count())
            .flat_map(|group| {
                let bytes: Vec<_> = units[layout.group_range(group).unwrap()]
                    .iter()
                    .map(|unit| binding_bytes(unit).unwrap())
                    .collect();
                bytes
                    .windows(disk_depth.min(bytes.len()))
                    .map(|window| window.iter().sum::<u64>())
                    .collect::<Vec<_>>()
            })
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
            let (mut model, metadata) = materialize_voicechat_mog(
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
            let mut actual = Vec::new();
            for (case_index, name) in ["released_policy", "top_k_without_guidance"]
                .into_iter()
                .enumerate()
            {
                let case = &fixture[name];
                let noise = case["policy"]["noise_scale"].as_f64().unwrap() as f32;
                for (index, step) in case["steps"].as_array().unwrap().iter().enumerate() {
                    let prediction = model
                        .predict(
                            &tensor(&step["input"]),
                            step["guidance_scale"].as_f64().unwrap() as f32,
                            &stream,
                        )
                        .unwrap_or_else(|error| {
                            panic!("{residency:?} F16={f16} {name} step{index} predict: {error}")
                        });
                    let mixture = indices(
                        &step["mixture_indices"],
                        &prediction.hidden.shape()[..2],
                        &stream,
                    );
                    if case_index == 0 && index == 0 {
                        if let Some(report) = model.dense_stream_report().unwrap() {
                            assert_eq!(
                                report
                                    .execution_groups()
                                    .iter()
                                    .filter(|group| group.completed_executions() > 0)
                                    .count(),
                                2,
                                "prediction executes only residual and small-head groups"
                            );
                            assert_eq!(
                                report
                                    .execution_groups()
                                    .iter()
                                    .filter(|group| group.peak_device_bytes() == 0)
                                    .count(),
                                1,
                                "selected bank stays unmaterialized until decisions arrive"
                            );
                        }
                        let mut shape = prediction.hidden.shape().to_vec();
                        shape[2] = dimensions.latent - 1;
                        let invalid = MlxTensor::full_f32(0., &shape, &stream).unwrap();
                        assert!(matches!(
                            model.selected(&prediction, &mixture, &invalid, noise, &stream),
                            Err(TtsError::Shape(_))
                        ));
                    }
                    let before_selection = (case_index == 0 && index == 0)
                        .then(|| model.dense_stream_report().unwrap())
                        .flatten();
                    let sample = model
                        .selected(
                            &prediction,
                            &mixture,
                            &tensor(&step["gaussian_draw"]),
                            noise,
                            &stream,
                        )
                        .unwrap_or_else(|error| {
                            panic!("{residency:?} F16={f16} {name} step{index} selected: {error}")
                        });
                    if let Some(before) = before_selection {
                        let after = model.dense_stream_report().unwrap().unwrap();
                        let changed = before
                            .execution_groups()
                            .iter()
                            .filter(|previous| {
                                let next = after
                                    .execution_groups()
                                    .iter()
                                    .find(|group| group.id() == previous.id())
                                    .unwrap();
                                next.completed_executions() != previous.completed_executions()
                            })
                            .count();
                        assert_eq!(
                            changed, 1,
                            "selected readout executes only its bank, without rerunning prediction"
                        );
                    }
                    let values = [
                        &prediction.hidden,
                        &prediction.logits,
                        &sample.mean,
                        &sample.log_std,
                        &sample.latent,
                    ]
                    .map(|tensor| tensor.to_f32_vec(&stream).unwrap());
                    if !f16 {
                        for (value, key) in
                            values
                                .iter()
                                .zip(["guided_hidden", "logits", "mu", "logs", "latent"])
                        {
                            compare(value, &step[key], &fixture);
                        }
                    }
                    actual.push(values);
                }
            }
            if let Some(expected) = &expected {
                assert_eq!(&actual, expected, "residency parity F16={f16}");
            } else {
                expected = Some(actual);
            }
            let report = model.residency_report().unwrap();
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

#[path = "voicechat_mog_residency/parallel.rs"]
mod parallel;
