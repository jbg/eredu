use super::*;
use crate::backend::nn::shared::MlxNeuralBackend;
use crate::backend::ExecutionContext;
use eredu_checkpoint::store::MemoryWeightStore;
use eredu_checkpoint::AffineQuantization;
use eredu_checkpoint::LinearFormat;
use eredu_nn::{LinearFormatSpec, LinearSpec, NeuralBackend, ParameterSpec};
use eredu_runtime::WeightLoweringKind;
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

#[test]
fn exact_affine_tp_input_slices_convert_before_binding_without_double_placement() {
    use crate::backend::runtime::checkpoint::{
        binding::{build_mlx_exact_replicated_text_bindings, materialize_module_bindings},
        quantization::quantize_tensor,
    };
    use eredu_checkpoint::{SourceTensorEncoding, StoredDtype};
    use eredu_runtime::{
        ArchitectureParameterDescription, ExecutionGraph, ExecutionGroupId, ExecutionUnitLayout,
        MemberSharding, OwnedParameterGroupSpec, ParameterGroupOwner, ParameterGroupSpec,
        ParameterMemberSpec, ParameterRole, ReplicatedTextOutputCompanion,
        ReplicatedTextParameterOwner, ReplicatedTextParameterRole, ReplicatedTextPhysicalSource,
        WeightLoweringDescriptor,
    };
    use safemlx::ops::indexing::TryIndexOp;
    let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = context.stream();
    let quantization = AffineQuantization::default();
    let format = LinearFormat::Affine(quantization);
    let native_format = LinearFormatSpec::affine(
        format,
        parameter("projection.scale"),
        parameter("projection.zero"),
    )
    .unwrap();
    let group = ParameterGroupSpec::partitioned(
        "projection.intermediate",
        ParameterRole::FeedForwardIntermediate,
        192,
        [ParameterMemberSpec::new(
            "projection.kernel",
            vec![8, 192],
            MemberSharding::Partitioned { axis: 1 },
        )],
    )
    .unwrap();
    let groups = eredu_runtime::expand_linear_format_parameter_groups(vec![group], |_| {
        Ok(Some(native_format.clone()))
    })
    .unwrap();
    let graph = ExecutionGraph::chain(["decoder"]).unwrap();
    let units = ExecutionUnitLayout::new(&graph, [1]).unwrap();
    let owned = groups
        .iter()
        .cloned()
        .map(|group| {
            OwnedParameterGroupSpec::new(
                ParameterGroupOwner::execution_unit(ExecutionGroupId::new("decoder").unwrap(), 0),
                group,
            )
        })
        .collect::<Vec<_>>();
    let parameters = ArchitectureParameterDescription::new(&graph, &units, groups, owned).unwrap();
    let values = (0..8 * 192)
        .map(|i| ((i * 37 % 257) as f32 - 128.) / 71.)
        .collect::<Vec<_>>();
    let bytes = values
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<_>>();
    let directory = tempfile::tempdir().unwrap();
    safetensors::tensor::serialize_to_file(
        [(
            "projection.kernel",
            safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![8, 192], &bytes)
                .unwrap(),
        )],
        None,
        &directory.path().join("model.safetensors"),
    )
    .unwrap();
    let full =
        quantize_tensor(&Array::from_slice(&values, &[8, 192]), quantization, stream).unwrap();
    for rank in 0..2 {
        let layout = eredu_runtime::derive_partitioned_local_layout(
            &parameters,
            eredu_core::ParallelRankTopology::new(
                eredu_core::ParallelTopology::new(2, 1, 1, 1).unwrap(),
                rank,
            )
            .unwrap(),
        )
        .unwrap();
        let columns = [128, 64][rank];
        let first = [0, 128][rank];
        let end = first + columns;
        let store: SharedCheckpointSource = Arc::new(
            SafetensorsWeightStore::open_with_max_cached_shards(directory.path(), 1).unwrap(),
        );
        let provenance = store.source_provenance("projection.kernel").unwrap();
        let owner = ReplicatedTextParameterOwner::ExecutionUnit {
            group: "decoder".into(),
            unit: 0,
        };
        let task = ReplicatedTextMaterializationTask::from_exact_source(
            "projection.kernel",
            ReplicatedTextPhysicalSource::new(
                provenance.catalog_key,
                provenance.physical_tensor,
                provenance.backing_shard.unwrap(),
                provenance.output,
                provenance.source_encoding,
                bytes.len() as u64,
            )
            .unwrap(),
            Vec::new(),
            vec![8, 192],
            vec![8, 192],
            ReplicatedTextParameterRole::LinearWeight,
            owner.clone(),
            format,
            WeightLoweringKind::Transform,
            WeightLoweringDescriptor::new(
                SourceTensorEncoding::Safetensors(StoredDtype::F32),
                format,
                vec![8, 192],
                vec![8, 192],
                Some(1),
            )
            .unwrap(),
        )
        .unwrap()
        .with_output_companions(vec![
            ReplicatedTextOutputCompanion::new(
                "projection.scale",
                LinearCompanionRole::Scale,
                vec![8, 3],
                owner.parameter_group_owner().unwrap(),
            )
            .unwrap(),
            ReplicatedTextOutputCompanion::new(
                "projection.zero",
                LinearCompanionRole::AffineBias,
                vec![8, 3],
                owner.parameter_group_owner().unwrap(),
            )
            .unwrap(),
        ])
        .unwrap();
        let source = MlxNeuralBackend::linear(
            LinearSpec {
                input: columns,
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
                input: columns,
                output: 8,
                weight: parameter("projection.kernel"),
                bias: None,
                format: native_format.clone(),
            },
            stream,
        )
        .unwrap();
        let empty: Vec<eredu_nn::Parameter<crate::MlxTensor>> = Vec::new();
        let (converted, report) = quantize_exact_replicated_text_tasks(
            Arc::clone(&store),
            &empty,
            &empty,
            &[source],
            &[target.clone()],
            Some(&layout),
            quantization.into(),
            &[&task],
            stream,
        )
        .unwrap();
        // The shared persistent cache may avoid conversion entirely. A fresh
        // conversion reads exactly the admitted local columns, never the full matrix.
        assert!(
            report.source_bytes_read == 0 || report.source_bytes_read == 8 * columns as u64 * 4
        );
        assert_eq!(
            report.output_bytes,
            8 * columns as u64 / 2 + 8 * (columns as u64 / 64) * 8
        );
        let bindings = build_mlx_exact_replicated_text_bindings(
            &target,
            converted.as_ref(),
            &[&task],
            &BTreeSet::new(),
            Some(&layout),
        )
        .unwrap();
        let arrays =
            materialize_module_bindings(converted.as_ref(), &bindings, stream, stream).unwrap();
        let expected = full
            .weight
            .try_index_device((.., first / 8..end / 8), stream)
            .unwrap()
            .contiguous(false, stream)
            .unwrap();
        assert_eq!(arrays["projection.kernel"].shape(), &[8, columns / 8]);
        assert_eq!(
            arrays["projection.kernel"]
                .evaluated()
                .unwrap()
                .as_slice::<u32>(),
            expected.evaluated().unwrap().as_slice::<u32>()
        );
        for (name, full) in [
            ("projection.scale", &full.scales),
            ("projection.zero", full.biases.as_ref().unwrap()),
        ] {
            let expected = full
                .try_index_device((.., first / 64..end / 64), stream)
                .unwrap()
                .contiguous(false, stream)
                .unwrap();
            assert_eq!(
                arrays[name].evaluated().unwrap().as_slice::<f32>(),
                expected.evaluated().unwrap().as_slice::<f32>()
            );
        }
        println!("exact affine TP rank {rank}: source columns {first}..{end}, bounded source bytes {}, packed output bytes {}",report.source_bytes_read,report.output_bytes);
    }
}
