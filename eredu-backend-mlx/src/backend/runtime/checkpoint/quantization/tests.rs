use std::time::{SystemTime, UNIX_EPOCH};

#[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
use crate::backend::ExecutionContext;
use eredu_architectures::checkpoint_conversion::{
    SafetensorsQuantizationPlan, SafetensorsQuantizationTarget,
};
use eredu_gguf::{Endian, GgmlType};
#[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
use safemlx::{Device, DeviceType, Dtype};

use super::*;

#[test]
fn affine_config_uses_mlx_spelling() {
    let value = serde_json::to_value(AffineQuantization::default()).unwrap();
    assert_eq!(value["group_size"], 64);
    assert_eq!(value["bits"], 4);
    assert_eq!(value["mode"], "affine");
}

#[test]
fn mxfp4_metadata_is_fixed_and_round_trips() {
    let value = serde_json::to_value(WeightQuantization::MxFp4).unwrap();
    assert_eq!(value, json!({"group_size": 32, "bits": 4, "mode": "mxfp4"}));
    assert_eq!(
        serde_json::from_value::<WeightQuantization>(value).unwrap(),
        WeightQuantization::MxFp4
    );
    assert!(serde_json::from_value::<WeightQuantization>(
        json!({"group_size": 64, "bits": 4, "mode": "mxfp4"})
    )
    .is_err());
    assert!(serde_json::from_value::<WeightQuantization>(
        json!({"group_size": 32, "bits": 8, "mode": "mxfp4"})
    )
    .is_err());
}

#[test]
fn omitted_quantization_mode_defaults_to_affine() {
    let quantization =
        serde_json::from_value::<WeightQuantization>(json!({"group_size": 64, "bits": 4})).unwrap();
    assert_eq!(
        quantization,
        WeightQuantization::Affine(AffineQuantization::new(64, 4).unwrap())
    );
}

#[test]
fn valid_gguf_quantization_is_not_a_producible_mlx_format() {
    let quantization = WeightQuantization::GgufIQuant {
        ggml_type: GgmlType::Q4_0,
        endian: Endian::Little,
    };
    quantization.validate().unwrap();

    let error = mlx_quantization_mode(quantization).unwrap_err();
    assert!(matches!(error, Error::Quantization(_)));
    assert!(error
        .to_string()
        .contains("cannot be produced by dense quantization"));
}

#[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
#[test]
fn quantize_tensor_returns_an_error_for_valid_gguf_quantization() {
    let context = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
    let weight = Array::from_slice(&[0.25_f32; 64], &[2, 32]);
    let quantization = WeightQuantization::GgufIQuant {
        ggml_type: GgmlType::Q4_0,
        endian: Endian::Little,
    };

    let error = quantize_tensor(&weight, quantization, context.stream()).unwrap_err();
    assert!(matches!(error, Error::Quantization(_)));
    assert!(error
        .to_string()
        .contains("cannot be produced by dense quantization"));
}

#[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
#[test]
fn quantize_tensor_rejects_zero_width_without_panicking() {
    let context = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
    let weight = Array::from_slice(&[] as &[f32], &[2, 0]);

    let error =
        quantize_tensor(&weight, AffineQuantization::default(), context.stream()).unwrap_err();
    assert!(error
        .to_string()
        .contains("input dimension must be nonzero"));
}

#[test]
fn planned_config_is_written_without_backend_metadata_rewrites() {
    let suffix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = std::env::temp_dir().join(format!(
        "eredu-mlx-neutral-quantization-config-{}-{suffix}",
        std::process::id()
    ));
    let output = root.join("output");
    fs::create_dir_all(&output).unwrap();
    let planned = json!({
        "model_type": "opaque",
        "artifact_filename": "custom.safetensors",
        "architecture_quantization": {"encoding": "mxfp4"}
    });
    write_planned_config(&output, &planned).unwrap();
    let config: Value =
        serde_json::from_slice(&fs::read(output.join("config.json")).unwrap()).unwrap();
    assert_eq!(config["artifact_filename"], "custom.safetensors");
    assert_eq!(config, planned);
    assert!(config.get("quantization").is_none());
    assert!(config.get("quantization_config").is_none());
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn options_have_no_tensor_name_selection_policy() {
    let plan = SafetensorsQuantizationPlan::new(
        WeightQuantization::MxFp4,
        [SafetensorsQuantizationTarget::new(
            "architecture.exact.source",
            "architecture.exact.packed",
            "architecture.exact.scale",
            None::<String>,
        )],
        json!({"model_type": "test"}),
    )
    .unwrap();
    let options = CheckpointQuantizationOptions::new(plan);
    assert_eq!(options.plan.targets().len(), 1);
    assert_eq!(
        options.plan.targets()[0].source_name(),
        "architecture.exact.source"
    );
}

#[test]
fn on_load_resolution_reuses_matching_metadata_and_rejects_mismatch() {
    let q4 = AffineQuantization::default();
    assert!(should_quantize_on_load("test", None, q4.into()).unwrap());
    assert!(!should_quantize_on_load("test", Some(q4.into()), q4.into()).unwrap());

    let q8 = AffineQuantization::new(64, 8).unwrap();
    let error = should_quantize_on_load("test", Some(q4.into()), q8.into()).unwrap_err();
    assert!(error.to_string().contains("already quantized"));
    assert!(error.to_string().contains("implicit dequantization"));
    assert!(!should_quantize_on_load(
        "test",
        Some(WeightQuantization::MxFp4),
        WeightQuantization::MxFp4
    )
    .unwrap());
}

#[test]
fn affine_config_accepts_mlx_non_power_of_two_widths() {
    assert!(AffineQuantization::new(32, 3).is_ok());
    assert!(AffineQuantization::new(32, 5).is_ok());
    assert!(AffineQuantization::new(32, 6).is_ok());
    assert!(AffineQuantization::new(32, 7).is_err());
}

#[test]
fn quantization_geometry_rejects_empty_dimensions() {
    let error = QuantizationMatrixGeometry::from_shape(&[2, 0]).unwrap_err();
    assert!(error
        .to_string()
        .contains("input dimension must be nonzero"));

    let error = QuantizationMatrixGeometry::from_shape(&[0, 2, 32]).unwrap_err();
    assert!(error
        .to_string()
        .contains("leading dimensions must be nonzero"));
}

#[test]
fn quantization_geometry_preserves_large_expert_banks() {
    let shape = [256, 65_536, 65_536];
    let geometry = QuantizationMatrixGeometry::from_shape(&shape).unwrap();

    assert!(
        shape
            .iter()
            .map(|&dimension| i64::from(dimension))
            .product::<i64>()
            > i64::from(i32::MAX)
    );
    assert_eq!(geometry.leading_size.get(), 16_777_216);
    assert_eq!(geometry.input_dims.get(), 65_536);
}

#[test]
fn quantization_geometry_rejects_unrepresentable_flattened_rows() {
    let error = QuantizationMatrixGeometry::from_shape(&[46_341, 46_341, 32]).unwrap_err();
    assert!(error.to_string().contains("cannot be flattened"));
}

#[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
#[test]
fn mxfp4_quantizes_rank_three_expert_banks() {
    let context = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
    let experts = Array::from_slice(&vec![0.25f32; 3 * 8 * 64], &[3, 8, 64]);
    let quantized = quantize_tensor(&experts, WeightQuantization::MxFp4, context.stream()).unwrap();
    assert_eq!(quantized.weight.shape(), &[3, 8, 8]);
    assert_eq!(quantized.scales.shape(), &[3, 8, 2]);
    assert!(quantized.biases.is_none());
}

#[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
#[test]
fn saved_mxfp4_checkpoint_has_no_affine_bias_tensors() {
    let suffix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = std::env::temp_dir().join(format!(
        "eredu-mlx-mxfp4-save-test-{}-{suffix}",
        std::process::id()
    ));
    let source = root.join("source");
    let output = root.join("output");
    fs::create_dir_all(&source).unwrap();
    fs::write(source.join("config.json"), br#"{"model_type":"test"}"#).unwrap();
    let weight = Array::from_slice(&vec![0.25f32; 2 * 64], &[2, 64]);
    Array::save_safetensors(
        [("model.proj.weight", &weight)],
        None,
        source.join("model.safetensors"),
    )
    .unwrap();

    let context = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
    let stream = context.stream();
    let weights_context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let plan = SafetensorsQuantizationPlan::new(
        WeightQuantization::MxFp4,
        [SafetensorsQuantizationTarget::new(
            "model.proj.weight",
            "model.proj.weight",
            "model.proj.scales",
            None::<String>,
        )],
        json!({
            "model_type": "test",
            "architecture_quantization": {"mode": "mxfp4"}
        }),
    )
    .unwrap();
    let options = CheckpointQuantizationOptions::new(plan);
    quantize_checkpoint(&source, &output, &options, stream).unwrap();
    let arrays =
        Array::load_safetensors(output.join("model.safetensors"), weights_context.stream())
            .unwrap();
    assert!(arrays.contains_key("model.proj.weight"));
    assert!(arrays.contains_key("model.proj.scales"));
    assert!(!arrays.contains_key("model.proj.biases"));
    let config: Value =
        serde_json::from_slice(&fs::read(output.join("config.json")).unwrap()).unwrap();
    assert_eq!(config["architecture_quantization"]["mode"], "mxfp4");
    assert!(config.get("quantization").is_none());
    assert!(config.get("quantization_config").is_none());
    fs::remove_dir_all(root).unwrap();
}

#[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
#[test]
fn saved_checkpoint_matches_direct_tensor_quantization() {
    let suffix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = std::env::temp_dir().join(format!(
        "eredu-mlx-quantization-test-{}-{suffix}",
        std::process::id()
    ));
    let source = root.join("source");
    let output = root.join("output");
    fs::create_dir_all(&source).unwrap();
    fs::write(source.join("config.json"), br#"{"model_type":"test"}"#).unwrap();

    let values = (0..(8 * 64))
        .map(|index| (index as f32 - 255.5) / 64.0)
        .collect::<Vec<_>>();
    let weight = Array::from_slice(&values, &[8, 64]);
    let embedding = Array::from_slice(&vec![1.0f32; 3 * 64], &[3, 64]);
    Array::save_safetensors(
        [
            ("model.proj.weight", &weight),
            ("model.embed_tokens.weight", &embedding),
        ],
        None,
        source.join("model.safetensors"),
    )
    .unwrap();
    let norm = Array::from_slice(&vec![1.0f32; 64], &[64]);
    Array::save_safetensors(
        [("auxiliary.weight", &norm)],
        None,
        source.join("auxiliary.safetensors"),
    )
    .unwrap();

    let context = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
    let stream = context.stream();
    let weights_context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let weights_stream = weights_context.stream();
    let expected = quantize_tensor(&weight, AffineQuantization::default(), stream).unwrap();
    let plan = SafetensorsQuantizationPlan::new(
        AffineQuantization::default(),
        [SafetensorsQuantizationTarget::new(
            "model.proj.weight",
            "runtime.projection.packed",
            "runtime.projection.scale",
            Some("runtime.projection.zero"),
        )],
        json!({
            "model_type": "test",
            "architecture_quantization": {"mode": "affine"}
        }),
    )
    .unwrap();
    let mut options = CheckpointQuantizationOptions::new(plan);
    options.shard_size_bytes = 1;
    let report = quantize_checkpoint(&source, &output, &options, stream).unwrap();
    assert_eq!(report.quantized_tensors, 1);
    assert_eq!(report.copied_tensors, 1);
    assert_eq!(report.shards, 2);

    let mut saved = HashMap::new();
    for file in safetensors_files(&output).unwrap() {
        saved.extend(Array::load_safetensors(file, weights_stream).unwrap());
    }
    let saved_weight = &saved["runtime.projection.packed"];
    assert_eq!(saved_weight.dtype(), Dtype::Uint32);
    assert_eq!(
        saved_weight.evaluated().unwrap().as_slice::<u32>(),
        expected.weight.evaluated().unwrap().as_slice::<u32>()
    );
    assert_eq!(
        saved["runtime.projection.scale"]
            .evaluated()
            .unwrap()
            .as_slice::<f32>(),
        expected.scales.evaluated().unwrap().as_slice::<f32>()
    );
    assert_eq!(
        saved["runtime.projection.zero"]
            .evaluated()
            .unwrap()
            .as_slice::<f32>(),
        expected
            .biases
            .as_ref()
            .unwrap()
            .evaluated()
            .unwrap()
            .as_slice::<f32>()
    );
    assert_eq!(
        saved["model.embed_tokens.weight"]
            .evaluated()
            .unwrap()
            .as_slice::<f32>(),
        embedding.evaluated().unwrap().as_slice::<f32>()
    );

    let config: Value =
        serde_json::from_slice(&fs::read(output.join("config.json")).unwrap()).unwrap();
    assert_eq!(config["architecture_quantization"]["mode"], "affine");
    assert!(config.get("quantization").is_none());
    assert!(config.get("quantization_config").is_none());
    assert!(output.join("auxiliary.safetensors").exists());
    fs::remove_dir_all(root).unwrap();
}
