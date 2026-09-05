use super::*;
use crate::backend::nn::shared::MlxNeuralBackend;
use crate::backend::ExecutionContext;
use eredu_checkpoint::store::MemoryWeightStore;
use eredu_checkpoint::AffineQuantization;
use eredu_checkpoint::LinearFormat;
use eredu_nn::{LinearFormatSpec, LinearSpec, NeuralBackend, ParameterSpec};
use safemlx::{Device, DeviceType};

fn parameter(name: &str) -> ParameterSpec {
    ParameterSpec::trainable(name).unwrap()
}

#[test]
fn quantized_store_preserves_nonconventional_architecture_companion_names() {
    let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let weight_name = "encoder.blocks.3.projection.kernel";
    let source = MlxNeuralBackend::linear(
        LinearSpec {
            input: 64,
            output: 8,
            weight: parameter(weight_name),
            bias: None,
            format: LinearFormatSpec::unscaled(LinearFormat::Dense).unwrap(),
        },
        context.stream(),
    )
    .unwrap();
    let format = LinearFormatSpec::affine(
        WeightQuantization::Affine(AffineQuantization::default()).into(),
        parameter("encoder.blocks.3.quantization.scale-table"),
        parameter("encoder.blocks.3.quantization.zero-points"),
    )
    .unwrap();
    let linear = MlxNeuralBackend::linear(
        LinearSpec {
            input: 64,
            output: 8,
            weight: parameter(weight_name),
            bias: None,
            format,
        },
        context.stream(),
    )
    .unwrap();

    let companions = packed_weight_companions(
        &linear,
        WeightQuantization::Affine(AffineQuantization::default()),
    )
    .unwrap();
    let target = companions
        .get("encoder.blocks.3.projection.kernel")
        .unwrap();
    assert_eq!(target.weight_name, "encoder.blocks.3.projection.kernel");
    assert_eq!(
        target.scales_name,
        "encoder.blocks.3.quantization.scale-table"
    );
    assert_eq!(
        target.biases_name.as_deref(),
        Some("encoder.blocks.3.quantization.zero-points")
    );

    let store: SharedCheckpointSource = Arc::new(
        MemoryWeightStore::from_safetensors([(
            weight_name.to_owned(),
            safetensors::Dtype::F32,
            vec![8, 64],
            vec![0; 8 * 64 * size_of::<f32>()],
        )])
        .unwrap(),
    );
    let (quantized, _) = quantize_parameterized_module_store(
        store,
        &source,
        &linear,
        WeightQuantization::Affine(AffineQuantization::default()),
        context.stream(),
    )
    .unwrap();
    assert!(quantized
        .source_metadata("encoder.blocks.3.quantization.scale-table")
        .is_ok());
    assert!(quantized
        .source_metadata("encoder.blocks.3.quantization.zero-points")
        .is_ok());
    assert!(quantized
        .source_metadata("encoder.blocks.3.projection.kernel_scales")
        .is_err());
}

#[test]
fn exact_consumption_rejects_missing_and_extra_targets() {
    let requested = BTreeSet::from(["selected.weight"]);
    let error = validate_exact_consumption(&requested, &BTreeMap::new()).unwrap_err();
    assert!(error.to_string().contains("not consumed exactly once"));

    let extra = BTreeMap::from([(
        "extra.weight".to_owned(),
        (
            DerivedWeightRecipe::source("extra.weight", TensorSelection::Full),
            PackedWeightCompanions {
                weight_name: "extra.weight".into(),
                scales_name: "extra.scales".into(),
                biases_name: None,
                affine_companion_dtype: RecipeDtype::F32,
            },
        ),
    )]);
    let error = validate_exact_consumption(&BTreeSet::new(), &extra).unwrap_err();
    assert!(error.to_string().contains("unselected targets"));
}
