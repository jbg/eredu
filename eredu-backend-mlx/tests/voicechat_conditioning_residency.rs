//! Character, prompt/decode ingress and fusion use independent native parameter leases.
use std::{collections::BTreeMap, sync::Arc};

use eredu_architectures::{
    artifact_preparation::PhysicalArtifactId,
    voicechat::{
        artifact::ParameterAdmission,
        checkpoint::Component,
        tts_conditioning::{CharacterBatch, CharacterDimensions, CharacterFlags, CharacterMapping},
        tts_conditioning_layered::{ConditioningDimensions, ConditioningExecution},
        tts_conditioning_preparation::PreparedConditioningBindings,
        tts_ingress::{PromptLayout, SpeakerPrompt},
        tts_mog::TtsError,
        tts_rvq::RvqDimensions,
        tts_rvq_preparation::PreparedRvqBindings,
    },
};
use eredu_backend_mlx::{
    backend::runtime::checkpoint::binding::binding_bytes,
    native::{materialize_voicechat_conditioning, materialize_voicechat_rvq},
    MlxTensor,
};
use eredu_checkpoint::{
    recipe::DerivedWeightRecipe,
    store::{SafetensorsWeightStore, SharedCheckpointSource, TensorSelection},
};
use eredu_core::residency::{MemoryTier, OffloadConfig};
use eredu_nn::{Tensor, TensorElementType};
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
fn parameters(source: &SharedCheckpointSource) -> BTreeMap<String, ParameterAdmission> {
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
    parameters
}
fn prepared(
    dimensions: ConditioningDimensions,
    source: SharedCheckpointSource,
) -> PreparedConditioningBindings {
    let reads = source.source_diagnostics().unwrap().physical_reads;
    let bindings = PreparedConditioningBindings::from_parameters(
        dimensions,
        source.clone(),
        &parameters(&source),
    )
    .unwrap();
    assert_eq!(
        source.source_diagnostics().unwrap().physical_reads,
        reads,
        "cold preparation reads no payload"
    );
    bindings
}
fn compare(actual: &MlxTensor, expected: &Value, absolute: f32, relative: f32, context: &Stream) {
    let expected = tensor(expected)
        .broadcast_to(actual.shape(), context)
        .unwrap()
        .to_f32_vec(context)
        .unwrap();
    for (index, (actual, expected)) in actual
        .to_f32_vec(context)
        .unwrap()
        .into_iter()
        .zip(expected)
        .enumerate()
    {
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

fn code_tensor(value: &Value, context: &Stream) -> MlxTensor {
    let shape: Vec<i32> = value["shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap() as i32)
        .collect();
    MlxTensor::from_i32_slice(&codes(value), &shape, context).unwrap()
}
fn residency(kind: usize, budget: u64) -> LayerWeightResidency {
    match kind {
        0 => LayerWeightResidency::FullyResident,
        1 => LayerWeightResidency::LayerwiseHost(LayerwiseLoadOptions::new(
            OffloadConfig::new(Some(budget), Some(1 << 20), 1).unwrap(),
        )),
        _ => LayerWeightResidency::DenseDiskStream(
            DenseDiskStreamLoadOptions::new(budget, 0, 0, 0).unwrap(),
        ),
    }
}

#[path = "voicechat_conditioning_residency/parallel.rs"]
mod parallel;

fn fixture() -> (Value, Value, Value, CharacterBatch, ConditioningDimensions) {
    let character: Value =
        serde_json::from_str(eredu_evaluation::voicechat::CONDITIONING_REFERENCE_JSON).unwrap();
    let ingress: Value =
        serde_json::from_str(eredu_evaluation::voicechat::INGRESS_REFERENCE_JSON).unwrap();
    // The independent oracles share width but use different fusion weights.
    // Keep the character oracle's fusion; compare ingress lanes independently.
    let mut all_parameters = ingress["parameters"].as_object().unwrap().clone();
    all_parameters.extend(character["parameters"].as_object().unwrap().clone());
    all_parameters.insert(
        "tts_model.audio_prompt_latents.Aria".into(),
        ingress["cases"][0]["supplied_speaker"].clone(),
    );
    let mut combined = serde_json::Map::new();
    combined.insert("parameters".into(), Value::Object(all_parameters));
    let combined = Value::Object(combined);
    let vocabulary = character["vocabulary"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
        .map(|(id, token)| (id as u32, token.as_str().unwrap().to_owned()))
        .collect();
    let mapping = CharacterMapping::new(
        &vocabulary,
        CharacterFlags {
            continuation: codes(&character["continuation_flags"]),
            continuation_pad: character["continuation_pad"].as_i64().unwrap() as i32,
            special: codes(&character["special_flags"]),
            special_pad: character["special_pad"].as_i64().unwrap() as i32,
        },
    )
    .unwrap();
    let selected: Vec<bool> = character["selected"]["values"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_bool().unwrap())
        .collect();
    let batch = mapping
        .prepare(&codes(&character["tokens"]), Some(&selected), 2, 4, 1024)
        .unwrap();
    let dimensions = ConditioningDimensions::new(
        CharacterDimensions {
            characters: mapping.characters().len() as i32,
            width: 4,
            output_width: 4,
            use_continuation: true,
            use_special: true,
        },
        5,
    )
    .unwrap();
    (character, ingress, combined, batch, dimensions)
}

#[test]
fn native_conditioning_residency_preserves_character_prompt_decode_fusion_and_casts() {
    let (character, ingress, combined, batch, dimensions) = fixture();
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let weights = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let directory = tempfile::tempdir().unwrap();
    for f16 in [false, true] {
        let source = checkpoint(
            &combined,
            f16,
            &directory.path().join(if f16 {
                "f16.safetensors"
            } else {
                "f32.safetensors"
            }),
        );
        let (_, _, layout, restricted, units) = prepared(dimensions, source.clone()).into_parts();
        let mut expected_keys: Vec<String> = character["parameters"]
            .as_object()
            .unwrap()
            .keys()
            .cloned()
            .collect();
        expected_keys.extend(
            [
                "tts_model.tts_model.bos_emb",
                "tts_model.tts_model.null_emb",
                "tts_model.tts_model.audio_prompt_projection_W",
                "tts_model.audio_prompt_latents.Aria",
            ]
            .map(str::to_owned),
        );
        assert_eq!(restricted.source_keys().len(), expected_keys.len());
        assert!(
            restricted
                .source_keys()
                .iter()
                .all(|key| expected_keys.contains(key)),
            "RVQ and unrelated parameters stay inaccessible"
        );
        assert_eq!(units.len(), 3);
        assert!(
            (0..layout.group_count()).all(|group| layout.group_range(group).unwrap().len() == 1)
        );
        let learned_elements: u64 = expected_keys
            .iter()
            .map(|key| {
                combined["parameters"][key.as_str()]["values"]
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
        let rvq_dimensions = RvqDimensions {
            cardinality: 5,
            latent: 4,
            hidden: 4,
        };
        let rvq_prepared = || {
            PreparedRvqBindings::from_parameters(
                rvq_dimensions,
                source.clone(),
                &parameters(&source),
            )
            .unwrap()
        };
        let (_, _, _, _, rvq_units) = rvq_prepared().into_parts();
        let rvq_largest = rvq_units
            .iter()
            .map(|unit| binding_bytes(unit).unwrap())
            .max()
            .unwrap();
        let mut expected = None;
        for kind in 0..3 {
            let policy = residency(kind, largest);
            let (mut model, metadata) = materialize_voicechat_conditioning(
                prepared(dimensions, source.clone()),
                policy,
                &weights,
                &stream,
            )
            .unwrap();
            let (mut feedback, _) = materialize_voicechat_rvq(
                rvq_prepared(),
                residency(kind, rvq_largest),
                &weights,
                &stream,
            )
            .unwrap();
            assert_eq!(metadata.static_device_bytes(), 0);
            assert_eq!(model.static_lease_count(), 0);
            assert_eq!(metadata.layer_parameter_bytes(), learned_elements * 4);
            let mut actual = Vec::new();
            let embedded = model.character_embed(&batch, &stream).unwrap();
            if let Some(report) = model.dense_stream_report().unwrap() {
                assert_eq!(
                    report
                        .execution_groups()
                        .iter()
                        .filter(|group| group.completed_executions() > 0)
                        .count(),
                    1
                );
                assert_eq!(
                    report
                        .execution_groups()
                        .iter()
                        .filter(|group| group.peak_device_bytes() == 0)
                        .count(),
                    2,
                    "character phase leaves ingress and fusion unmaterialized"
                );
            }
            let output = model
                .character_finish(&batch, &tensor(&character["hidden"]), &stream)
                .unwrap();
            let fused = model
                .fuse(&tensor(&character["audio"]), &output, &stream)
                .unwrap();
            if !f16 {
                let a = character["tolerance"]["absolute"].as_f64().unwrap() as f32;
                let r = character["tolerance"]["relative"].as_f64().unwrap() as f32;
                compare(&embedded, &character["embedded"], a, r, &stream);
                compare(&output, &character["character_output"], a, r, &stream);
                compare(&fused, &character["fusion_output"], a, r, &stream);
            }
            actual.extend([embedded, output, fused].map(|t| t.to_f32_vec(&stream).unwrap()));
            let invalid = tensor(&character["audio"])
                .cast_float(TensorElementType::F16, &stream)
                .unwrap();
            assert!(matches!(
                model.fuse(&invalid, &tensor(&character["character_output"]), &stream),
                Err(TtsError::Shape(_))
            ));
            for case in ingress["cases"].as_array().unwrap() {
                let code = code_tensor(&case["code"], &stream);
                let condition = tensor(&case["supplied_subwords"]);
                let any = case["subword_mask"]["values"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|v| v.as_bool().unwrap());
                let condition = any.then_some(&condition);
                let lanes = if case["audio_mask"].is_null() {
                    model
                        .decode(&mut feedback, &code, condition, true, &stream)
                        .unwrap()
                } else {
                    let mask: Vec<bool> = case["audio_mask"]["values"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|v| v.as_bool().unwrap())
                        .collect();
                    let layout = PromptLayout::new(code.dim(0), code.dim(1), &mask).unwrap();
                    let speaker = if case["supplied_speaker"].is_null() {
                        SpeakerPrompt::ProjectedCodes
                    } else {
                        SpeakerPrompt::Aria
                    };
                    model
                        .prefill(
                            &mut feedback,
                            &code,
                            &layout,
                            speaker,
                            condition,
                            true,
                            &stream,
                        )
                        .unwrap()
                };
                if !f16 {
                    let a = ingress["comparison"]["absolute_tolerance"]
                        .as_f64()
                        .unwrap() as f32;
                    let r = ingress["comparison"]["relative_tolerance"]
                        .as_f64()
                        .unwrap() as f32;
                    compare(&lanes.audio, &case["audio_embedding"], a, r, &stream);
                    compare(&lanes.text, &case["conditioning"], a, r, &stream);
                }
                let fused = model.fuse(&lanes.audio, &lanes.text, &stream).unwrap();
                actual.extend(
                    [lanes.audio, lanes.text, fused].map(|t| t.to_f32_vec(&stream).unwrap()),
                );
            }
            if let Some(expected) = &expected {
                assert_eq!(&actual, expected, "residency parity F16={f16}");
            } else {
                expected = Some(actual);
            }
            let report = model.residency_report().unwrap();
            assert!(report.weight_store().physical_reads > 0);
            if !policy.is_fully_resident() {
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
            }
        }
    }
}
