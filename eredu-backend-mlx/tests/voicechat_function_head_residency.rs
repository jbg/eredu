//! Independent function vocabulary projections use bounded native parameter providers.
#[path = "support/component_coordinator.rs"]
mod component_coordinator;
use std::{collections::BTreeMap, sync::Arc};

use eredu_architectures::{
    artifact_preparation::PhysicalArtifactId,
    voicechat::{
        artifact::ParameterAdmission,
        channels::ChannelError,
        checkpoint::Component,
        function_head_layered::{FunctionHeadDimensions, FunctionHeadExecution},
        function_head_preparation::{FunctionHeadPreparationError, PreparedFunctionHeadBindings},
        parameters::SpeechParameterError,
    },
};
use eredu_backend_mlx::{
    backend::runtime::checkpoint::binding::binding_bytes,
    native::materialize_voicechat_function_head, MlxTensor,
};
use eredu_checkpoint::{
    recipe::{DerivedWeightRecipe, RecipeDtype},
    store::{SafetensorsWeightStore, SharedCheckpointSource, StoreError, TensorSelection},
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
fn checkpoint(fixture: &Value, f16: bool, path: &std::path::Path) -> SharedCheckpointSource {
    let mut values: Vec<_> = fixture["parameters"]
        .as_object()
        .unwrap()
        .iter()
        .filter(|(name, _)| {
            name.as_str() == "stt_model.function_head.weight"
                || name.as_str() == "stt_model.lm_head.weight"
        })
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
    source
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
                    linear_format: eredu_checkpoint::LinearFormat::Dense,
                    name,
                    component: Component::FunctionHead,
                    artifact: PhysicalArtifactId::Primary,
                    output,
                    stored_bytes: metadata.encoded_byte_len,
                    conversion_workspace_bytes: 0,
                    materialization_peak_bytes: peak,
                    recipe,
                },
            )
        })
        .collect()
}
fn prepared(
    dimensions: FunctionHeadDimensions,
    source: SharedCheckpointSource,
) -> PreparedFunctionHeadBindings {
    let parameters = parameters(&source);
    let reads = source.source_diagnostics().unwrap().physical_reads;
    let bindings =
        PreparedFunctionHeadBindings::from_parameters(dimensions, source.clone(), &parameters)
            .unwrap();
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
    let absolute = fixture["tolerance"]["absolute"].as_f64().unwrap() as f32;
    let relative = fixture["tolerance"]["relative"].as_f64().unwrap() as f32;
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= absolute + relative * expected.abs(),
            "{index}: {actual} versus reference {expected}"
        );
    }
}

#[test]
fn native_function_head_residency_preserves_forced_logits_and_casts_after_rejection() {
    let fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::BACKBONE_REFERENCE_JSON).unwrap();
    let dimensions = FunctionHeadDimensions::new(4, 24).unwrap();
    let hidden = tensor(&fixture["f32"]["normalized_hidden"]);
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let weights = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let directory = tempfile::tempdir().unwrap();
    let execution_bytes = 24 * 4 * 4;
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
        let plan = prepared(dimensions, source.clone());
        assert_eq!(
            plan.parameter().stored_bytes,
            if f16 {
                execution_bytes / 2
            } else {
                execution_bytes
            }
        );
        assert_eq!(plan.parameter().output.dtype, RecipeDtype::F32);
        assert_eq!(plan.parameter().output.byte_len(), execution_bytes);
        // Half input + widened output coexist while casting, in addition to
        // the retained source bytes. A direct F32 read has no cast temporary.
        assert_eq!(
            plan.parameter().materialization_peak_bytes,
            2 * execution_bytes
        );
        assert_eq!(plan.parameter().conversion_workspace_bytes, 0);
        let (_, _, _, layout, restricted, units) = plan.into_parts();
        assert_eq!(layout.len(), 1);
        assert_eq!(units.len(), 1);
        assert_eq!(binding_bytes(&units[0]).unwrap(), execution_bytes);
        assert_eq!(restricted.source_keys(), ["stt_model.function_head.weight"]);
        assert!(matches!(
            restricted.source_metadata("stt_model.lm_head.weight"),
            Err(StoreError::UnauthorizedTensor { .. })
        ));
        let policies = [
            LayerWeightResidency::FullyResident,
            LayerWeightResidency::LayerwiseHost(LayerwiseLoadOptions::new(
                OffloadConfig::new(Some(execution_bytes), Some(1 << 20), 1).unwrap(),
            )),
            LayerWeightResidency::DenseDiskStream(
                DenseDiskStreamLoadOptions::new(execution_bytes, 0, 0, 0).unwrap(),
            ),
        ];
        let mut expected = None;
        for residency in policies {
            let (mut head, metadata) = materialize_voicechat_function_head(
                prepared(dimensions, source.clone()),
                residency,
                &weights,
                &stream,
            )
            .unwrap();
            assert_eq!(metadata.static_device_bytes(), 0);
            assert_eq!(head.static_lease_count(), 0);
            assert_eq!(metadata.layer_parameter_bytes(), execution_bytes);
            let reads = source.source_diagnostics().unwrap().physical_reads;
            let invalid = MlxTensor::full_f32(0., &[1, 1, dimensions.width + 1], &stream).unwrap();
            assert!(matches!(
                head.project(&invalid, &stream),
                Err(ChannelError::Shape(_))
            ));
            assert_eq!(
                source.source_diagnostics().unwrap().physical_reads,
                reads,
                "wrong hidden width is rejected before the head is acquired"
            );
            let logits = head.project(&hidden, &stream).unwrap();
            assert_eq!(logits.shape(), &[1, 3, dimensions.vocabulary]);
            let actual = logits.to_f32_vec(&stream).unwrap();
            if !f16 {
                compare(&actual, &fixture["f32"]["function_logits"], &fixture);
            }
            if let Some(expected) = &expected {
                assert_eq!(&actual, expected, "function projection residency parity");
            } else {
                expected = Some(actual);
            }
            let report = head.residency_report().unwrap();
            assert!(report.weight_store().physical_reads > 0);
            if !residency.is_fully_resident() {
                assert!(
                    report
                        .offload()
                        .peak_resident_bytes()
                        .get(MemoryTier::Device)
                        <= execution_bytes
                );
            }
            if let Some(report) = head.dense_stream_report().unwrap() {
                assert_eq!(report.prefill_forwards() + report.decode_forwards(), 1);
            }
        }
    }
}

#[test]
fn function_head_preparation_rejects_missing_wrong_role_and_wrong_shape_without_reads() {
    let fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::BACKBONE_REFERENCE_JSON).unwrap();
    let directory = tempfile::tempdir().unwrap();
    let source = checkpoint(&fixture, false, &directory.path().join("cold.safetensors"));
    let dimensions = FunctionHeadDimensions::new(4, 24).unwrap();
    let mut parameters = parameters(&source);
    let key = "stt_model.function_head.weight";
    let saved = parameters.remove(key).unwrap();
    let reads = source.source_diagnostics().unwrap().physical_reads;
    let prepare = |parameters: &BTreeMap<String, ParameterAdmission>| {
        PreparedFunctionHeadBindings::from_parameters(dimensions, source.clone(), parameters)
    };
    assert!(
        matches!(prepare(&parameters), Err(FunctionHeadPreparationError::Parameters(
        SpeechParameterError::MissingParameter { component: Component::FunctionHead, name }
    )) if name == key)
    );
    let mut wrong_role = saved.clone();
    wrong_role.component = Component::Backbone;
    parameters.insert(key.into(), wrong_role);
    assert!(
        matches!(prepare(&parameters), Err(FunctionHeadPreparationError::Parameters(
        SpeechParameterError::MissingParameter { component: Component::FunctionHead, name }
    )) if name == key)
    );
    let mut wrong_shape = saved.clone();
    wrong_shape.recipe = DerivedWeightRecipe::Source {
        key: "unrelated.weight".into(),
        selection: TensorSelection::Full,
    };
    wrong_shape.output = wrong_shape.recipe.infer(source.as_ref()).unwrap();
    parameters.insert(key.into(), wrong_shape);
    assert!(
        matches!(prepare(&parameters), Err(FunctionHeadPreparationError::Geometry(name)) if name == key)
    );
    parameters.insert(key.into(), saved);
    let (_, _, _, _, restricted, _) = prepare(&parameters).unwrap().into_parts();
    assert!(matches!(
        restricted.source_metadata("unrelated.weight"),
        Err(StoreError::UnauthorizedTensor { .. })
    ));
    assert_eq!(source.source_diagnostics().unwrap().physical_reads, reads);
}

#[path = "voicechat_function_head_residency/parallel.rs"]
mod parallel;

#[test]
fn affine_function_head_preserves_companions_across_residency_and_partitioning() {
    for (bits, columns, packed, bytes) in [(4, 8, 0x33333333, 160), (8, 16, 0x03030303, 288)] {
        assert_affine_function_head(bits, columns, packed, bytes);
    }
}

fn assert_affine_function_head(bits: i32, columns: usize, packed: u32, bytes: u64) {
    use eredu_checkpoint::{AffineQuantization, LinearFormat};
    use eredu_core::{ParallelRankTopology, ParallelTopology};
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("affine.safetensors");
    let weight: Vec<u8> = (0..4 * columns)
        .flat_map(|_| packed.to_le_bytes())
        .collect();
    let scales: Vec<u8> = [0.5f32, 1., 2., 4.]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();
    let biases: Vec<u8> = [-1f32, -2., -3., -8.]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();
    serialize_to_file(
        [
            (
                "stt_model.function_head.weight",
                TensorView::new(Dtype::U32, vec![4, columns], &weight).unwrap(),
            ),
            (
                "stt_model.function_head.scales",
                TensorView::new(Dtype::F32, vec![4, 1], &scales).unwrap(),
            ),
            (
                "stt_model.function_head.biases",
                TensorView::new(Dtype::F32, vec![4, 1], &biases).unwrap(),
            ),
        ],
        None,
        &path,
    )
    .unwrap();
    let source: SharedCheckpointSource = Arc::new(SafetensorsWeightStore::open(&path).unwrap());
    let mut parameters = parameters(&source);
    parameters
        .get_mut("stt_model.function_head.weight")
        .unwrap()
        .linear_format = LinearFormat::Affine(AffineQuantization::new(64, bits).unwrap());
    let dimensions = FunctionHeadDimensions::new(64, 4).unwrap();
    let prepare = || {
        PreparedFunctionHeadBindings::from_parameters(dimensions, source.clone(), &parameters)
            .unwrap()
    };
    let (_, _, _, _, _, units) = prepare().into_parts();
    assert_eq!(binding_bytes(&units[0]).unwrap(), bytes);
    for rank in 0..2 {
        let topology =
            ParallelRankTopology::new(ParallelTopology::new(2, 1, 1, 1).unwrap(), rank).unwrap();
        let (local, _) = prepare().select_parallel(topology).unwrap();
        let (_, _, _, _, _, units) = local.into_parts();
        assert_eq!(binding_bytes(&units[0]).unwrap(), bytes / 2);
        assert_eq!(units[0].len(), 3);
    }
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let weights = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let hidden = MlxTensor::full_f32(1., &[1, 1, 64], &stream).unwrap();
    for residency in [
        LayerWeightResidency::FullyResident,
        LayerWeightResidency::LayerwiseHost(LayerwiseLoadOptions::new(
            OffloadConfig::new(Some(1 << 20), Some(1 << 20), 1).unwrap(),
        )),
        LayerWeightResidency::DenseDiskStream(
            DenseDiskStreamLoadOptions::new(1 << 20, 0, 0, 0).unwrap(),
        ),
    ] {
        let (mut head, metadata) =
            materialize_voicechat_function_head(prepare(), residency, &weights, &stream).unwrap();
        assert_eq!(metadata.layer_parameter_bytes(), bytes);
        let actual = head
            .project(&hidden, &stream)
            .unwrap()
            .to_f32_vec(&stream)
            .unwrap();
        // q=3, one affine group: each row is respectively 0.5,1,3,4.
        assert_eq!(actual, [32., 64., 192., 256.]);
    }
}
