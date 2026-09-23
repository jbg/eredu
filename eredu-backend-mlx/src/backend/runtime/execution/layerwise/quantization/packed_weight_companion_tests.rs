use super::*;
use crate::backend::nn::shared::MlxNeuralBackend;
use crate::backend::ExecutionContext;
use eredu_checkpoint::store::MemoryWeightStore;
use eredu_checkpoint::AffineQuantization;
use eredu_checkpoint::LinearFormat;
use eredu_nn::{LinearFormatSpec, LinearSpec, NeuralBackend, ParameterSpec};
use safemlx::{Array, Device, DeviceType};

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

#[test]
fn exact_affine_tasks_preserve_source_precision_through_unloaded_f32_slots() {
    use crate::backend::runtime::checkpoint::quantization::quantize_tensor;
    use crate::backend::runtime::checkpoint::store::MlxParameterMaterializationContext;
    use eredu_checkpoint::store::{ReadPolicy, TensorReadRequest};
    use eredu_checkpoint::{SourceTensorEncoding, StoredDtype};
    use eredu_runtime::{
        ReplicatedTextOutputCompanion, ReplicatedTextParameterOwner, ReplicatedTextParameterRole,
        ReplicatedTextPhysicalSource, WeightLoweringDescriptor,
    };

    let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = context.stream();
    let values: Vec<f32> = (0..512)
        .map(|i| ((i * 37 % 257) as f32 - 128.0) / 71.0)
        .collect();
    for (safe_dtype, stored_dtype, native_dtype) in [
        (safetensors::Dtype::F16, StoredDtype::F16, Dtype::Float16),
        (safetensors::Dtype::BF16, StoredDtype::BF16, Dtype::Bfloat16),
        (safetensors::Dtype::F32, StoredDtype::F32, Dtype::Float32),
    ] {
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|&value| match safe_dtype {
                safetensors::Dtype::F16 => half::f16::from_f32(value).to_le_bytes().to_vec(),
                safetensors::Dtype::BF16 => half::bf16::from_f32(value).to_le_bytes().to_vec(),
                _ => value.to_le_bytes().to_vec(),
            })
            .collect();
        let encoding = SourceTensorEncoding::Safetensors(stored_dtype.clone());
        let quantization = AffineQuantization::default();
        let format = LinearFormat::Affine(quantization);
        let owner = ReplicatedTextParameterOwner::ExecutionUnit {
            group: "layers".into(),
            unit: 0,
        };
        let task = ReplicatedTextMaterializationTask::from_exact_source(
            "projection.kernel",
            ReplicatedTextPhysicalSource::new(
                "projection.kernel",
                "projection.kernel",
                "weights.safetensors",
                "projection.kernel",
                encoding.clone(),
                bytes.len() as u64,
            )
            .unwrap(),
            vec![],
            vec![8, 64],
            vec![8, 64],
            ReplicatedTextParameterRole::LinearWeight,
            owner.clone(),
            format,
            WeightLoweringKind::Transform,
            WeightLoweringDescriptor::new(encoding, format, vec![8, 64], vec![8, 64], Some(1))
                .unwrap(),
        )
        .unwrap()
        .with_output_companions(vec![
            ReplicatedTextOutputCompanion::new(
                "projection.scale",
                LinearCompanionRole::Scale,
                vec![8, 1],
                owner.parameter_group_owner().unwrap(),
            )
            .unwrap(),
            ReplicatedTextOutputCompanion::new(
                "projection.zero",
                LinearCompanionRole::AffineBias,
                vec![8, 1],
                owner.parameter_group_owner().unwrap(),
            )
            .unwrap(),
        ])
        .unwrap();
        let source = MlxNeuralBackend::linear(
            LinearSpec {
                input: 64,
                output: 8,
                weight: parameter("projection.kernel"),
                bias: None,
                format: LinearFormatSpec::unscaled(LinearFormat::Dense).unwrap(),
            },
            stream,
        )
        .unwrap();
        let target = MlxNeuralBackend::linear(
            LinearSpec {
                input: 64,
                output: 8,
                weight: parameter("projection.kernel"),
                bias: None,
                format: LinearFormatSpec::affine(
                    format,
                    parameter("projection.scale"),
                    parameter("projection.zero"),
                )
                .unwrap(),
            },
            stream,
        )
        .unwrap();
        let store: SharedCheckpointSource = Arc::new(
            MemoryWeightStore::from_safetensors([(
                "projection.kernel".into(),
                safe_dtype,
                vec![8, 64],
                bytes,
            )])
            .unwrap(),
        );
        let empty: Vec<eredu_nn::Parameter<crate::MlxTensor>> = vec![];
        let (transformed, _) = quantize_exact_replicated_text_tasks(
            Arc::clone(&store),
            &empty,
            &empty,
            &[source],
            &[target],
            None,
            WeightQuantization::Affine(quantization),
            &[&task],
            stream,
        )
        .unwrap();
        for name in ["projection.scale", "projection.zero"] {
            assert_eq!(
                transformed.source_metadata(name).unwrap().stored_dtype,
                stored_dtype
            );
        }
        let read = |name: &str| {
            let lease = transformed
                .acquire_lease(TensorReadRequest {
                    key: name.into(),
                    selection: TensorSelection::Full,
                    policy: ReadPolicy::RequireBounded,
                })
                .unwrap();
            MlxParameterMaterializationContext::new(stream, stream)
                .weight_lease(lease)
                .unwrap()
                .materialize(stream, stream)
                .unwrap()
                .synchronize()
                .unwrap()
        };
        let dense = Array::from_slice(&values, &[8, 64])
            .as_dtype(native_dtype, stream)
            .unwrap();
        let expected = quantize_tensor(&dense, quantization, stream).unwrap();
        assert_eq!(
            read("projection.kernel")
                .evaluated()
                .unwrap()
                .as_slice::<u32>(),
            expected.weight.evaluated().unwrap().as_slice::<u32>()
        );
        for (name, expected) in [
            ("projection.scale", expected.scales),
            ("projection.zero", expected.biases.unwrap()),
        ] {
            let actual = read(name);
            assert_eq!(actual.dtype(), native_dtype);
            assert_eq!(
                actual
                    .as_dtype(Dtype::Float32, stream)
                    .unwrap()
                    .evaluated()
                    .unwrap()
                    .as_slice::<f32>(),
                expected
                    .as_dtype(Dtype::Float32, stream)
                    .unwrap()
                    .evaluated()
                    .unwrap()
                    .as_slice::<f32>()
            );
        }
        // Placeholder precision is flexible; source shape and floating category
        // remain mandatory even when the admitted checkpoint is floating.
        let companions = BTreeMap::from([(
            "projection.kernel".into(),
            PackedWeightCompanions {
                weight_name: "projection.kernel".into(),
                scales_name: "projection.scale".into(),
                biases_name: Some("projection.zero".into()),
                affine_companion_dtype: RecipeDtype::F32,
            },
        )]);
        let requested = BTreeMap::from([("projection.kernel", &task)]);
        for (array, message) in [
            (
                Array::from_slice(&[0i32; 512], &[8, 64]),
                "unsupported dtype",
            ),
            (
                Array::from_slice(&[0f32; 256], &[4, 64]),
                "native source module requires",
            ),
        ] {
            let invalid = eredu_nn::Parameter::new(
                parameter("projection.kernel"),
                crate::MlxTensor::from_array(array),
            );
            let error = collect_exact_quantization_recipes(
                store.as_ref(),
                &invalid,
                None,
                &companions,
                &requested,
                &mut BTreeMap::new(),
            )
            .unwrap_err();
            assert!(error.to_string().contains(message), "{error}");
        }
    }
}
