use super::*;

#[test]
fn mlx_pipeline_allocator_preserves_exact_u32_and_i32_boundary_roles() {
    use eredu_architectures::partitioned_execution::PartitionTensorAllocator;

    let (stream, _) = execution_streams();
    for (source, logical, expected) in [
        (
            MlxTensor::from_array(Array::from_slice(&[7_u32], &[1, 1])),
            eredu_runtime::BoundaryTensorDtype::Uint32,
            Dtype::Uint32,
        ),
        (
            MlxTensor::from_array(Array::from_slice(&[-7_i32], &[1, 1])),
            eredu_runtime::BoundaryTensorDtype::Int32,
            Dtype::Int32,
        ),
    ] {
        let converted = MlxPartitionTensorAllocator
            .tensor_to_wire(
                source,
                logical,
                eredu_runtime::PipelineActivationDtype::Float32,
                &stream,
            )
            .unwrap();
        assert_eq!(converted.as_array().dtype(), expected);
        let placeholder = MlxPartitionTensorAllocator
            .tensor_placeholder(
                &[1, 1],
                logical,
                eredu_runtime::PipelineActivationDtype::Float32,
                &stream,
            )
            .unwrap();
        assert_eq!(placeholder.as_array().dtype(), expected);
    }
}

#[test]
fn mlx_pipeline_allocator_rejects_nonfloating_source_before_wire_submission() {
    use eredu_architectures::partitioned_execution::PartitionTensorAllocator;

    let (stream, _) = execution_streams();
    let source = MlxTensor::from_array(Array::from_slice(&[1_i32], &[1, 1, 1]));
    let error = MlxPartitionTensorAllocator
        .tensor_to_wire(
            source,
            eredu_runtime::BoundaryTensorDtype::Activation,
            eredu_runtime::PipelineActivationDtype::Float32,
            &stream,
        )
        .expect_err("integer source activation must not enter the pipeline wire");
    assert!(error.to_string().contains("logical boundary dtype"));
}

#[test]
fn mlx_pipeline_allocator_converts_f16_and_bf16_sources_to_exact_f32_wire_values() {
    use eredu_architectures::partitioned_execution::PartitionTensorAllocator;

    let (stream, _) = execution_streams();
    let expected = [1.5_f32, -2.0, 0.25];
    for source_dtype in [Dtype::Float16, Dtype::Bfloat16] {
        let source = Array::from_slice(&expected, &[1, 1, 3])
            .as_dtype(source_dtype, &stream)
            .unwrap();
        let converted = MlxPartitionTensorAllocator
            .tensor_to_wire(
                MlxTensor::from_array(source),
                eredu_runtime::BoundaryTensorDtype::Activation,
                eredu_runtime::PipelineActivationDtype::Float32,
                &stream,
            )
            .unwrap();
        let evaluated = converted.as_array().evaluated().unwrap();
        assert_eq!(evaluated.as_array().dtype(), Dtype::Float32);
        assert_eq!(evaluated.as_slice::<f32>(), expected.as_slice());
    }
}

#[test]
fn dense_decoder_partition_classifier_is_architecture_owned_and_exhaustive() {
    for model_type in ["llama", "qwen2", "qwen3"] {
        let root = tiny_artifact(model_type, false);
        let inspection = eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
        assert!(
            eredu_architectures::partitioned_execution::is_supported_dense_decoder_partition(
                inspection.architecture_plan(),
            ),
            "{model_type} must enter typed dense-decoder dispatch"
        );
    }
    for model_type in ["qwen3_moe", "gpt_oss"] {
        let root = tiny_artifact(model_type, false);
        let inspection = eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
        assert!(
            !eredu_architectures::partitioned_execution::is_supported_dense_decoder_partition(
                inspection.architecture_plan(),
            ),
            "{model_type} must remain outside dense-decoder dispatch"
        );
    }
}

#[test]
fn report_distinguishes_native_and_transforming_lowerings() {
    let parameter = ReplicatedTextParameterRequirement::new(
        "projection.weight",
        vec!["projection.weight".into()],
        vec![eredu_runtime::ReplicatedTextPhysicalSource::new(
            "projection.weight",
            "projection.weight",
            "/checkpoint/model.safetensors",
            "projection.weight",
            SourceTensorEncoding::Safetensors(StoredDtype::F16),
            64 * 64 * 2,
        )
        .unwrap()],
        Vec::new(),
        Some(SourceTensorEncoding::Safetensors(StoredDtype::F16)),
        Some(vec![64, 64]),
        vec![64, 64],
        LinearFormat::Dense,
        eredu_runtime::ReplicatedTextParameterRole::LinearWeight,
        eredu_runtime::ReplicatedTextParameterOwner::ExecutionUnit {
            group: "decoder".into(),
            unit: 0,
        },
        eredu_runtime::ReplicatedTextParameterPresence::Required,
        ParameterTransformConstraint::Linear { packed_axis: 1 },
    )
    .unwrap();
    let graph = eredu_runtime::ExecutionGraph::chain(["decoder"]).unwrap();
    let requirements = ReplicatedTextRequirements::new(
        "test.generic-binding",
        eredu_nn::NeuralOperatorCapabilities::NONE,
        graph.clone(),
        eredu_runtime::ExecutionUnitLayout::new(&graph, [1]).unwrap(),
        vec![eredu_runtime::ArchitectureGroupTransport {
            placement: eredu_runtime::ArchitectureGroupPlacement::Pipeline,
            kind: eredu_runtime::ArchitectureGroupKind::Decoder,
            first_owner_static_roles: vec!["embedding".into()],
            last_owner_static_roles: vec!["output".into()],
            merge_destination: eredu_runtime::ArchitectureMergeDestination::LastOwner,
            parallel_subgroup: None,
            request_optional: false,
        }],
        StateLayout::new(
            LayerSchedule::new(
                1,
                vec![LayerCachePolicy::key_value(AttentionPolicy::Full, 1, 8).unwrap()],
            )
            .unwrap(),
        )
        .unwrap(),
        eredu_runtime::ReplicatedTextStateAccess::KeyValue,
        vec![parameter],
    )
    .unwrap()
    .with_floating_state_source(eredu_core::checkpoint::TensorDtype::F32);
    let request = eredu_runtime::ReplicatedTextSelectionRequest::new(
        eredu_runtime::LayerWeightResidency::FullyResident,
        CacheResidencyPolicy::Device,
    )
    .with_quantization(eredu_core::QuantizationRequest::Affine {
        group_size: 64,
        bits: 4,
    });
    let report = capabilities(&requirements, &request);
    assert!(report.weight_lowerings().iter().any(|lowering| {
        lowering.executable() == LinearFormat::Dense
            && lowering.kind() == WeightLoweringKind::Direct
    }));
    assert!(report.weight_lowerings().iter().any(|lowering| {
        matches!(lowering.executable(), LinearFormat::Affine(_))
            && lowering.kind() == WeightLoweringKind::Transform
    }));
}

#[test]
fn selected_graph_mismatch_rejects_before_module_construction() {
    crate::tests::support::path_instrumentation::reset();
    let artifact = tiny_artifact("llama", false);
    let inspection = eredu_architectures::configuration::inspect_artifact(artifact.path()).unwrap();
    let requirements =
        eredu_architectures::replicated_text::replicated_text_requirements(&inspection).unwrap();
    let forged_graph = eredu_runtime::ExecutionGraph::chain(["forged-decoder"]).unwrap();
    let forged_units = eredu_runtime::ExecutionUnitLayout::new(
        &forged_graph,
        [requirements.execution_units().len()],
    )
    .unwrap();
    let forged_requirements = ReplicatedTextRequirements::new(
        requirements.architecture_identity().to_owned(),
        requirements.operators(),
        forged_graph,
        forged_units,
        requirements.group_transports().to_vec(),
        requirements.state_layout().clone(),
        requirements.state_access(),
        requirements.parameters().to_vec(),
    )
    .unwrap()
    .with_floating_state_source(eredu_core::checkpoint::TensorDtype::F32)
    .with_derived_recipes(
        requirements.derived_recipes().clone(),
        requirements.derived_recipe_outputs().clone(),
    )
    .unwrap();
    let request = eredu_runtime::ReplicatedTextSelectionRequest::new(
        eredu_runtime::LayerWeightResidency::FullyResident,
        CacheResidencyPolicy::Device,
    );
    let selected = eredu_runtime::select_replicated_text_realization(
        &forged_requirements,
        &request,
        &capabilities(&forged_requirements, &request),
    )
    .unwrap();
    let source: eredu_checkpoint::store::SharedCheckpointSource =
        Arc::new(eredu_checkpoint::store::SafetensorsWeightStore::open(artifact.path()).unwrap());
    let (stream, weights_stream) = execution_streams();
    let error = super::super::loading::bind_replicated_text(
        inspection.architecture_plan(),
        selected,
        source,
        &stream,
        &weights_stream,
    )
    .err()
    .expect("forged graph must fail");
    assert!(error.to_string().contains("structure differs"), "{error}");
    assert_eq!(
        crate::tests::support::path_instrumentation::snapshot(),
        crate::tests::support::path_instrumentation::Counts::default()
    );
}

#[test]
fn selected_store_geometry_mismatch_rejects_before_construction_or_payload_open() {
    crate::tests::support::path_instrumentation::reset();
    let artifact = tiny_heterogeneous_artifact(lfm2_config());
    let inspection = eredu_architectures::configuration::inspect_artifact(artifact.path()).unwrap();
    let requirements =
        eredu_architectures::replicated_text::replicated_text_requirements(&inspection).unwrap();
    let request = eredu_runtime::ReplicatedTextSelectionRequest::new(
        eredu_runtime::LayerWeightResidency::FullyResident,
        eredu_runtime::CacheResidencyPolicy::Device,
    );
    let selected = eredu_runtime::select_replicated_text_realization(
        &requirements,
        &request,
        &capabilities(&requirements, &request),
    )
    .unwrap();
    let source: eredu_checkpoint::store::SharedCheckpointSource =
        Arc::new(eredu_checkpoint::store::SafetensorsWeightStore::open(artifact.path()).unwrap());
    let key = requirements
        .parameters()
        .iter()
        .find_map(|parameter| parameter.sources().first())
        .unwrap()
        .clone();
    let forged: eredu_checkpoint::store::SharedCheckpointSource =
        Arc::new(ForgedShapeSource { inner: source, key });
    let (stream, weights_stream) = execution_streams();
    let error = super::super::loading::bind_replicated_text(
        inspection.architecture_plan(),
        selected,
        forged,
        &stream,
        &weights_stream,
    )
    .err()
    .expect("forged source geometry must fail");
    assert!(error.to_string().contains("physical geometry"));
    assert_eq!(
        crate::tests::support::path_instrumentation::snapshot(),
        crate::tests::support::path_instrumentation::Counts::default()
    );
}

#[test]
fn exact_lowering_rejects_unsupported_encodings_and_incoherent_physical_geometry() {
    let affine = LinearFormat::Affine(eredu_checkpoint::AffineQuantization::new(32, 4).unwrap());
    let gguf = |physical_shape| {
        WeightLoweringDescriptor::new(
            SourceTensorEncoding::Gguf {
                ggml_type: eredu_gguf::GgmlType::Q4_0,
                endian: eredu_gguf::Endian::Little,
            },
            affine,
            physical_shape,
            vec![64, 8],
            Some(1),
        )
        .unwrap()
    };
    assert!(supports_direct(&gguf(vec![64, 32])));
    assert!(!supports_direct(&gguf(vec![63, 32])));
    assert!(!supports_direct(&gguf(vec![64, 31])));
    assert!(!supports_direct(&gguf(vec![64, 64])));

    let safetensors = |source, physical_shape| {
        WeightLoweringDescriptor::new(source, affine, physical_shape, vec![64, 64], Some(1))
            .unwrap()
    };
    assert!(supports_direct(&safetensors(
        SourceTensorEncoding::Safetensors(StoredDtype::U32),
        vec![64, 8],
    )));
    assert!(!supports_direct(&safetensors(
        SourceTensorEncoding::Safetensors(StoredDtype::U32),
        vec![64, 7],
    )));
    assert!(!supports_direct(&safetensors(
        SourceTensorEncoding::Safetensors(StoredDtype::U8),
        vec![64, 64],
    )));

    let transform = safetensors(
        SourceTensorEncoding::Safetensors(StoredDtype::F16),
        vec![64, 32],
    );
    assert!(!supports_transform(&transform));
    let non_final_axis = WeightLoweringDescriptor::new(
        SourceTensorEncoding::Safetensors(StoredDtype::F16),
        affine,
        vec![64, 64],
        vec![64, 64],
        Some(0),
    )
    .unwrap();
    assert!(!supports_direct(&non_final_axis));
    assert!(!supports_transform(&non_final_axis));
    let final_axis = WeightLoweringDescriptor::new(
        SourceTensorEncoding::RecipeOutput(StoredDtype::F16),
        affine,
        vec![64, 64],
        vec![64, 64],
        Some(1),
    )
    .unwrap();
    assert!(supports_transform(&final_axis));
}

#[test]
fn exact_local_transform_output_is_not_sharded_twice() {
    use eredu_checkpoint::store::{MemoryWeightStore, TensorSelection};
    use eredu_runtime::{LocalModelLayout, LocalTensorLayout, ParameterRole, TensorPlacement};

    let store = MemoryWeightStore::from_safetensors([
        (
            "transformed.weight".to_owned(),
            safetensors::Dtype::F32,
            vec![2, 2],
            vec![0; 2 * 2 * size_of::<f32>()],
        ),
        (
            "untouched.weight".to_owned(),
            safetensors::Dtype::F32,
            vec![4, 2],
            vec![0; 4 * 2 * size_of::<f32>()],
        ),
    ])
    .unwrap();
    let mut layout = LocalModelLayout::default();
    for name in ["transformed.weight", "untouched.weight"] {
        layout.insert(
            name.to_owned(),
            LocalTensorLayout::new(
                "projection",
                ParameterRole::ColumnProjection,
                vec![4, 2],
                vec![2, 2],
                TensorPlacement::Shard {
                    axis: 0,
                    index: 0,
                    parts: 2,
                },
                None,
                None,
                false,
            ),
        );
    }
    let binding = |name: &str, bytes| {
        WeightBinding::new(name, name, TensorSelection::Full, bytes)
            .unwrap()
            .with_logical_target(name)
            .unwrap()
    };
    let bindings = vec![
        binding("transformed.weight", 16),
        binding("untouched.weight", 32),
    ];
    let output = shard_unmaterialized_bindings(
        bindings,
        &store,
        &layout,
        &["transformed.weight".to_owned()].into_iter().collect(),
    )
    .unwrap();

    assert_eq!(
        output[0].source_recipe().infer(&store).unwrap().shape(),
        [2, 2]
    );
    assert_eq!(
        output[1].source_recipe().infer(&store).unwrap().shape(),
        [2, 2]
    );
    assert_eq!(output[0].expected_bytes(), 16);
    assert_eq!(output[1].expected_bytes(), 16);
}

#[test]
fn prediction_target_forks_preserve_kv_and_compressed_paging_sessions() {
    let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = context.stream();
    let key_value_layout = StateLayout::new(
        LayerSchedule::new(
            1,
            vec![LayerCachePolicy::key_value(AttentionPolicy::Full, 1, 8).unwrap()],
        )
        .unwrap(),
    )
    .unwrap();
    let canonical =
        MlxKeyValueState::paged(key_value_layout, prediction_cache_manager(), None).unwrap();
    let canonical_checkpoint = canonical.deep_checkpoint().unwrap();
    let mut fork = fork_mlx_prediction_target_state(&canonical, stream).unwrap();
    let fork_checkpoint = fork.deep_checkpoint().unwrap();
    fork.restore_checkpoint(&fork_checkpoint, stream).unwrap();
    assert_eq!(fork.offset(), canonical.offset());
    assert!(fork
        .restore_checkpoint(&canonical_checkpoint, stream)
        .unwrap_err()
        .to_string()
        .contains("does not belong to the same paged layer"));

    let compressed_layout = StateLayout::new(
        LayerSchedule::new(
            1,
            vec![LayerCachePolicy::compressed_latent_rotary(AttentionPolicy::Full, 8, 4).unwrap()],
        )
        .unwrap(),
    )
    .unwrap();
    let canonical =
        MlxHybridState::paged(compressed_layout, prediction_cache_manager(), None).unwrap();
    let canonical_checkpoint = canonical.deep_checkpoint().unwrap();
    let mut fork = fork_mlx_prediction_target_state(&canonical, stream).unwrap();
    let fork_checkpoint = fork.deep_checkpoint().unwrap();
    fork.restore_checkpoint(&fork_checkpoint, stream).unwrap();
    assert_eq!(fork.offset(), canonical.offset());
    assert!(fork
        .restore_checkpoint(&canonical_checkpoint, stream)
        .unwrap_err()
        .to_string()
        .contains("does not belong to the same paged layer"));
}
