use crate::backend::runtime::checkpoint::gguf::GgufCheckpoint;
use std::collections::BTreeMap;
#[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
use std::collections::HashMap;

#[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
use eredu_backend_mlx_macros::PhysicalParameters;
#[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
use eredu_checkpoint::AffineQuantization;
use eredu_checkpoint::WeightQuantization;
use eredu_gguf::{Endian, GgmlType, TensorInput, Writer, WriterOptions};
#[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
use safemlx::{Array, Device, DeviceType, Dtype};

#[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
use crate::{
    backend::ExecutionContext,
    module::{PhysicalParam, PhysicalParameters as _},
};

use super::gguf_quantization_configs;
#[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
use super::{
    load_array_strict, load_arrays_quantized_strict, Error, QuantizedLoadRecipe, StrictLoadReport,
};
use crate::backend::runtime::checkpoint::quantization::io::safetensors_files;

#[test]
fn safetensors_file_discovery_rejects_untrusted_index_paths_and_duplicates() {
    let traversal = tempfile::tempdir().unwrap();
    std::fs::write(
        traversal.path().join("model.safetensors.index.json"),
        r#"{"weight_map":{"weight":"../outside.safetensors"}}"#,
    )
    .unwrap();
    let traversal_error = safetensors_files(traversal.path()).unwrap_err();
    assert!(matches!(
        traversal_error,
        crate::backend::Error::CheckpointShards(
            eredu_checkpoint::safetensors::SafetensorsShardError::UnsafeShardPath { .. }
        )
    ));

    let duplicate = tempfile::tempdir().unwrap();
    std::fs::write(
        duplicate.path().join("model.safetensors.index.json"),
        r#"{"weight_map":{"weight":"one.safetensors","weight":"two.safetensors"}}"#,
    )
    .unwrap();
    let duplicate_error = safetensors_files(duplicate.path()).unwrap_err();
    assert!(matches!(
        duplicate_error,
        crate::backend::Error::CheckpointShards(
            eredu_checkpoint::safetensors::SafetensorsShardError::MalformedIndex { .. }
        )
    ));
}

#[cfg(unix)]
#[test]
fn safetensors_file_discovery_rejects_external_payload_symlinks() {
    let parent = tempfile::tempdir().unwrap();
    let checkpoint = parent.path().join("checkpoint");
    std::fs::create_dir(&checkpoint).unwrap();
    let outside = parent.path().join("outside.safetensors");
    std::fs::write(&outside, []).unwrap();
    std::os::unix::fs::symlink(&outside, checkpoint.join("linked.safetensors")).unwrap();
    std::fs::write(
        checkpoint.join("model.safetensors.index.json"),
        r#"{"weight_map":{"weight":"linked.safetensors"}}"#,
    )
    .unwrap();

    assert!(matches!(
        safetensors_files(&checkpoint),
        Err(crate::backend::Error::CheckpointShards(
            eredu_checkpoint::safetensors::SafetensorsShardError::UnsafeShardPath { .. }
        ))
    ));
}

#[test]
fn gguf_runtime_configs_preserve_every_native_affine_format() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("native.gguf");
    let formats = [GgmlType::Q4K, GgmlType::Q5_1, GgmlType::Q8_0];
    let names = formats
        .iter()
        .map(|format| format!("{format:?}.weight"))
        .collect::<Vec<_>>();
    let dimensions = formats
        .iter()
        .map(|format| [format.block_and_bytes().unwrap().0])
        .collect::<Vec<_>>();
    let payloads = formats
        .iter()
        .map(|format| vec![0; format.block_and_bytes().unwrap().1 as usize])
        .collect::<Vec<_>>();
    let tensors = formats
        .iter()
        .enumerate()
        .map(|(index, format)| TensorInput {
            name: &names[index],
            dimensions: &dimensions[index],
            ggml_type: *format,
            data: &payloads[index],
        })
        .collect::<Vec<_>>();
    Writer::default()
        .write(
            std::fs::File::create(&path).unwrap(),
            &BTreeMap::new(),
            &tensors,
        )
        .unwrap();

    let checkpoint = GgufCheckpoint::open(path).unwrap();
    let tensor_mapping = checkpoint
        .catalog()
        .translated_outputs(str::to_string)
        .unwrap();
    let configs = gguf_quantization_configs(&checkpoint, &tensor_mapping).unwrap();
    for (name, expected) in names.iter().zip(formats) {
        assert!(matches!(
            configs[name],
            WeightQuantization::GgufIQuant { ggml_type, .. } if ggml_type == expected
        ));
    }
}

#[test]
fn big_endian_native_affine_format_keeps_portable_runtime_config() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("big-endian.gguf");
    let payload = vec![0; GgmlType::Q4K.block_and_bytes().unwrap().1 as usize];
    Writer::new(WriterOptions {
        endian: Endian::Big,
        ..WriterOptions::default()
    })
    .unwrap()
    .write(
        std::fs::File::create(&path).unwrap(),
        &BTreeMap::new(),
        &[TensorInput {
            name: "projection.weight",
            dimensions: &[256],
            ggml_type: GgmlType::Q4K,
            data: &payload,
        }],
    )
    .unwrap();

    let checkpoint = GgufCheckpoint::open(path).unwrap();
    let tensor_mapping = checkpoint
        .catalog()
        .translated_outputs(str::to_string)
        .unwrap();
    let configs = gguf_quantization_configs(&checkpoint, &tensor_mapping).unwrap();
    assert!(matches!(
        configs["projection.weight"],
        WeightQuantization::Affine(config) if config.group_size == 32 && config.bits == 4
    ));
}

#[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
#[derive(Debug, Clone, PhysicalParameters)]
struct PackedExperts {
    #[param]
    experts: PhysicalParam<Array>,
    #[param]
    experts_scales: PhysicalParam<Option<Array>>,
    #[param]
    experts_biases: PhysicalParam<Option<Array>>,
}

#[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
#[derive(Debug, Clone, PhysicalParameters)]
struct ExactInnerWeight {
    #[param]
    projection: ExactInnerProjection,
}

#[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
#[derive(Debug, Clone, PhysicalParameters)]
struct ExactInnerProjection {
    #[param]
    inner: ExactPackedLinear,
}

#[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
#[derive(Debug, Clone, PhysicalParameters)]
struct ExactPackedLinear {
    #[param]
    weight: PhysicalParam<Array>,
    #[param]
    scales: PhysicalParam<Array>,
    #[param]
    biases: PhysicalParam<Option<Array>>,
}

#[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
#[test]
fn strict_load_accepts_exact_inner_weight_identity() {
    let context = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
    let stream = context.stream();
    let mut model = ExactInnerWeight {
        projection: ExactInnerProjection {
            inner: ExactPackedLinear {
                weight: PhysicalParam::<Array>::unloaded(&[8, 64], Dtype::Float32, stream).unwrap(),
                scales: PhysicalParam::<Array>::unloaded(&[8, 2], Dtype::Float32, stream).unwrap(),
                biases: PhysicalParam::new(None),
            },
        },
    };
    let value = Array::from_slice(&vec![0.25f32; 8 * 64], &[8, 64]);
    let mut report = StrictLoadReport::default();
    load_array_strict(
        &mut model.parameters_mut().flatten(),
        "projection.inner.weight".into(),
        value,
        &mut report,
    );
    report
        .finish_excluding(&model, |name| name != "projection.inner.weight")
        .unwrap();
}

#[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
#[test]
fn strict_load_does_not_translate_canonical_weight_to_private_slot() {
    let context = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
    let stream = context.stream();
    let mut model = ExactInnerWeight {
        projection: ExactInnerProjection {
            inner: ExactPackedLinear {
                weight: PhysicalParam::<Array>::unloaded(&[8, 64], Dtype::Float32, stream).unwrap(),
                scales: PhysicalParam::<Array>::unloaded(&[8, 2], Dtype::Float32, stream).unwrap(),
                biases: PhysicalParam::new(None),
            },
        },
    };
    let value = Array::from_slice(&vec![0.25f32; 8 * 64], &[8, 64]);
    let mut report = StrictLoadReport::default();
    load_array_strict(
        &mut model.parameters_mut().flatten(),
        "projection.weight".into(),
        value,
        &mut report,
    );
    assert!(matches!(
        report.finish_excluding(&model, |name| name != "projection.inner.weight"),
        Err(Error::StrictLoadValidation { missing, unused })
            if missing == ["projection.inner.weight"] && unused == ["projection.weight"]
    ));
}

#[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
#[test]
fn named_array_quantization_packs_rank_three_experts() {
    let context = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
    let stream = context.stream();
    let mut model = PackedExperts {
        experts: PhysicalParam::<Array>::unloaded(&[3, 8, 8], Dtype::Uint32, stream).unwrap(),
        experts_scales: PhysicalParam::<Option<Array>>::unloaded_some(
            &[3, 8, 2],
            Dtype::Uint8,
            stream,
        )
        .unwrap(),
        experts_biases: PhysicalParam::new(None),
    };
    let dense = Array::from_slice(&vec![0.25f32; 3 * 8 * 64], &[3, 8, 64]);
    let mut report = StrictLoadReport::default();
    load_arrays_quantized_strict(
        &mut model,
        HashMap::from([("experts".into(), dense)]),
        stream,
        WeightQuantization::MxFp4,
        &HashMap::from([(
            "experts".into(),
            QuantizedLoadRecipe::new("experts", "experts_scales", None),
        )]),
        &mut report,
    )
    .unwrap();
    report.finish(&model).unwrap();
    assert_eq!(model.experts.shape(), &[3, 8, 8]);
    assert_eq!(
        model.experts_scales.value.as_ref().unwrap().shape(),
        &[3, 8, 2]
    );
    assert!(model.experts_biases.value.is_none());
}

#[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
#[test]
fn quantized_strict_load_preserves_exact_inner_weight_and_companions() {
    let context = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
    let stream = context.stream();
    let quantization = AffineQuantization::default();
    let mut model = ExactInnerWeight {
        projection: ExactInnerProjection {
            inner: ExactPackedLinear {
                weight: PhysicalParam::<Array>::unloaded(&[8, 8], Dtype::Uint32, stream).unwrap(),
                scales: PhysicalParam::<Array>::unloaded(&[8, 1], Dtype::Uint8, stream).unwrap(),
                biases: PhysicalParam::<Option<Array>>::unloaded_some(
                    &[8, 1],
                    Dtype::Float32,
                    stream,
                )
                .unwrap(),
            },
        },
    };
    let dense = Array::from_slice(&vec![0.25f32; 8 * 64], &[8, 64]);
    let mut report = StrictLoadReport::default();
    load_arrays_quantized_strict(
        &mut model,
        HashMap::from([("projection.inner.weight".into(), dense)]),
        stream,
        quantization.into(),
        &HashMap::from([(
            "projection.inner.weight".into(),
            QuantizedLoadRecipe::new(
                "projection.inner.weight",
                "projection.inner.scales",
                Some("projection.inner.biases".into()),
            ),
        )]),
        &mut report,
    )
    .unwrap();
    report.finish(&model).unwrap();

    assert_eq!(model.projection.inner.weight.shape(), &[8, 8]);
    assert_eq!(model.projection.inner.scales.shape(), &[8, 1]);
    assert_eq!(
        model
            .projection
            .inner
            .biases
            .value
            .as_ref()
            .unwrap()
            .shape(),
        &[8, 1]
    );
}
