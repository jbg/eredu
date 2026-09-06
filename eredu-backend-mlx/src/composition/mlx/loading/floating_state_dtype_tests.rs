use super::{
    inspected_floating_state_dtype_bytes, mlx_floating_state_dtype_bytes,
    select_preparation_with_mechanism_capabilities,
};
use crate::backend::{ExecutionContext, MlxBackend};
use eredu_core::{
    checkpoint::TensorDtype, residency::OffloadConfig, ModelLoadingBackend as _, ParallelTopology,
};
use eredu_gguf::{GgmlType, MetadataValue, TensorInput, Writer};

#[test]
fn deepseek_v4_prediction_selection_retains_neutral_extension_contract() {
    let root = tempfile::tempdir().unwrap();
    crate::tests::distributed_pipeline_ring::write_deepseek_v4_fixture(root.path(), 1);
    let inspection = eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
    let topology = crate::test_parallel_rank(0, 2, 1, 1);
    let options = crate::MlxLoadRequest::with_parallel(
        topology,
        crate::backend::DeviceAssignment::new(safemlx::DeviceType::Cpu, 0),
        eredu_runtime::PipelineWireContract::new(eredu_runtime::PipelineActivationDtype::Float32),
        4,
        4096,
        crate::MlxLoadRequest::test_communication_completion_policy(),
    )
    .unwrap();

    let selected = super::select_preparation(&inspection, options).unwrap();

    let (target, extension) = inspection
        .architecture_plan()
        .prediction_target_projection()
        .unwrap()
        .unwrap();
    let extension_sources = extension
        .source_keys(target.safetensors_architecture().unwrap())
        .unwrap();
    assert_eq!(extension_sources.len(), 41);
    let target_sources = target
        .safetensors_architecture()
        .unwrap()
        .checkpoint_resolution()
        .unwrap()
        .source_keys();
    let complete_sources = extension
        .complete_architecture()
        .checkpoint_resolution()
        .unwrap()
        .source_keys();
    assert!(target_sources.is_disjoint(&extension_sources));
    assert_eq!(
        target_sources
            .union(&extension_sources)
            .cloned()
            .collect::<std::collections::BTreeSet<_>>(),
        *complete_sources
    );
    assert!(selected.neutral().prediction_extension().is_some());
    assert!(selected.neutral().communication_manifest().is_some());
    assert!(selected.rank_context().is_some());
}

#[test]
fn disabled_plan_projects_one_ordinary_target_without_selecting_extension_payloads() {
    let root = tempfile::tempdir().unwrap();
    crate::tests::distributed_pipeline_ring::write_deepseek_v4_fixture(root.path(), 1);
    let inspection = eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
    let options = crate::MlxLoadRequest::from_normalized(
        eredu_runtime::NormalizedLoadRequest::default()
            .with_drafting(eredu_runtime::DraftingLoadRequest::Disabled),
    );

    let selected = super::select_preparation(&inspection, options).unwrap();

    assert!(selected.neutral().prediction_extension().is_none());
    assert!(selected.neutral().communication_manifest().is_none());
}
use safemlx::{Device, DeviceType};
use std::collections::BTreeMap;

fn write_minimal_llama_gguf(path: &std::path::Path, dtype: GgmlType) {
    let metadata = BTreeMap::from([
        (
            "general.architecture".into(),
            MetadataValue::String("llama".into()),
        ),
        ("llama.block_count".into(), MetadataValue::Uint32(2)),
        ("llama.embedding_length".into(), MetadataValue::Uint32(16)),
        (
            "llama.attention.head_count".into(),
            MetadataValue::Uint32(2),
        ),
        (
            "llama.feed_forward_length".into(),
            MetadataValue::Uint32(16),
        ),
        (
            "llama.attention.layer_norm_rms_epsilon".into(),
            MetadataValue::Float32(1e-5),
        ),
        ("llama.vocab_size".into(), MetadataValue::Uint32(16)),
        ("llama.context_length".into(), MetadataValue::Uint32(1)),
    ]);
    let vector_data = [0_u8; 32];
    let matrix_data = [0_u8; 512];
    let tensor = |name, dimensions| TensorInput {
        name,
        dimensions,
        ggml_type: dtype,
        data: if dimensions.len() == 1 {
            &vector_data
        } else {
            &matrix_data
        },
    };
    let tensors = [
        tensor("token_embd.weight", &[16, 16]),
        tensor("output_norm.weight", &[16]),
        tensor("blk.0.attn_norm.weight", &[16]),
        tensor("blk.0.ffn_norm.weight", &[16]),
        tensor("blk.0.attn_q.weight", &[16, 16]),
        tensor("blk.0.attn_k.weight", &[16, 16]),
        tensor("blk.0.attn_v.weight", &[16, 16]),
        tensor("blk.0.attn_output.weight", &[16, 16]),
        tensor("blk.0.ffn_gate.weight", &[16, 16]),
        tensor("blk.0.ffn_up.weight", &[16, 16]),
        tensor("blk.0.ffn_down.weight", &[16, 16]),
        tensor("blk.1.attn_norm.weight", &[16]),
        tensor("blk.1.ffn_norm.weight", &[16]),
        tensor("blk.1.attn_q.weight", &[16, 16]),
        tensor("blk.1.attn_k.weight", &[16, 16]),
        tensor("blk.1.attn_v.weight", &[16, 16]),
        tensor("blk.1.attn_output.weight", &[16, 16]),
        tensor("blk.1.ffn_gate.weight", &[16, 16]),
        tensor("blk.1.ffn_up.weight", &[16, 16]),
        tensor("blk.1.ffn_down.weight", &[16, 16]),
    ];
    Writer::default()
        .write(std::fs::File::create(path).unwrap(), &metadata, &tensors)
        .unwrap();
}

#[test]
fn partitioned_capability_failure_precedes_native_or_payload_work() {
    let root = tempfile::tempdir().unwrap();
    let model = root.path().join("model.gguf");
    write_minimal_llama_gguf(&model, GgmlType::F16);
    let inspection = eredu_architectures::configuration::inspect_artifact(&model).unwrap();
    let topology = crate::test_parallel_rank(0, 2, 1, 1);
    let options = crate::MlxLoadRequest::with_parallel(
        topology,
        crate::backend::DeviceAssignment::new(safemlx::DeviceType::Cpu, 0),
        eredu_runtime::PipelineWireContract::new(eredu_runtime::PipelineActivationDtype::Float32),
        1,
        1,
        crate::MlxLoadRequest::test_communication_completion_policy(),
    )
    .unwrap();
    crate::tests::support::path_instrumentation::reset();

    let error = select_preparation_with_mechanism_capabilities(
        &inspection,
        options,
        &super::super::replicated_text::GROUPED_OPERATION_CAPABILITIES,
        &eredu_runtime::CommunicationCapabilities::new([]).unwrap(),
    )
    .unwrap_err();

    assert!(
        error.to_string().contains("communication"),
        "unexpected selection failure: {error}"
    );
    assert_eq!(
        crate::tests::support::path_instrumentation::communication_realization_attempts(),
        0
    );
    assert_eq!(
        crate::tests::support::path_instrumentation::snapshot(),
        Default::default()
    );
}

#[test]
fn data_parallel_request_reaches_neutral_selection_and_fails_before_native_or_payload_work() {
    let root = tempfile::tempdir().unwrap();
    let model = root.path().join("model.gguf");
    write_minimal_llama_gguf(&model, GgmlType::F16);
    let inspection = eredu_architectures::configuration::inspect_artifact(&model).unwrap();
    let topology =
        eredu_core::ParallelRankTopology::new(ParallelTopology::new(1, 1, 1, 2).unwrap(), 0)
            .unwrap();
    let options = crate::MlxLoadRequest::with_parallel(
        topology,
        crate::backend::DeviceAssignment::new(safemlx::DeviceType::Cpu, 0),
        eredu_runtime::PipelineWireContract::new(eredu_runtime::PipelineActivationDtype::Float32),
        1,
        1,
        crate::MlxLoadRequest::test_communication_completion_policy(),
    )
    .unwrap();
    crate::tests::support::path_instrumentation::reset();

    let error = super::select_preparation(&inspection, options).unwrap_err();

    assert!(
        error
            .to_string()
            .contains("data-parallel execution is not supported"),
        "unexpected data-parallel rejection: {error}"
    );
    assert_eq!(
        crate::tests::support::path_instrumentation::communication_realization_attempts(),
        0
    );
    assert_eq!(
        crate::tests::support::path_instrumentation::manifest_communication_realization_attempts(),
        0
    );
    assert_eq!(
        crate::tests::support::path_instrumentation::snapshot(),
        Default::default()
    );
}

#[test]
fn public_preparation_rejects_every_oversubscribed_pipeline_rank_before_native_work() {
    let root = tempfile::tempdir().unwrap();
    let model = root.path().join("model.gguf");
    write_minimal_llama_gguf(&model, GgmlType::F16);
    let inspection = eredu_architectures::configuration::inspect_artifact(&model).unwrap();
    let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let backend = MlxBackend::new(execution.stream(), execution.stream());

    for rank in 0..3 {
        let topology = crate::test_parallel_rank(rank, 1, 3, 1);
        let options = crate::MlxLoadRequest::with_parallel(
            topology,
            crate::backend::DeviceAssignment::new(safemlx::DeviceType::Cpu, 0),
            eredu_runtime::PipelineWireContract::new(
                eredu_runtime::PipelineActivationDtype::Float32,
            ),
            1,
            1,
            crate::MlxLoadRequest::test_communication_completion_policy(),
        )
        .unwrap();
        crate::tests::support::path_instrumentation::reset();

        let error = backend
            .select_preparation(&inspection, &options)
            .unwrap_err();

        assert!(
            error
                .to_string()
                .contains("decoder execution group 0 has 2 units for 3 pipeline stages"),
            "rank {rank} returned an unexpected selection failure: {error}"
        );
        assert_eq!(
            crate::tests::support::path_instrumentation::communication_realization_attempts(),
            0,
            "rank {rank} reached native communication realization"
        );
        assert_eq!(
            crate::tests::support::path_instrumentation::manifest_communication_realization_attempts(),
            0,
            "rank {rank} reached opaque manifest realization"
        );
        assert_eq!(
            crate::tests::support::path_instrumentation::snapshot(),
            Default::default(),
            "rank {rank} performed payload, construction, state, or execution work"
        );
    }
}

#[test]
fn gguf_llama_tp_selects_neutral_partitioned_execution() {
    let root = tempfile::tempdir().unwrap();
    let model = root.path().join("model.gguf");
    write_minimal_llama_gguf(&model, GgmlType::F16);
    let inspection = eredu_architectures::configuration::inspect_artifact(&model).unwrap();
    let topology = crate::test_parallel_rank(0, 2, 1, 1);
    let options = crate::MlxLoadRequest::with_parallel(
        topology,
        crate::backend::DeviceAssignment::new(safemlx::DeviceType::Cpu, 0),
        eredu_runtime::PipelineWireContract::new(eredu_runtime::PipelineActivationDtype::Float32),
        1,
        1,
        crate::MlxLoadRequest::test_communication_completion_policy(),
    )
    .unwrap();

    let selected = super::select_preparation(&inspection, options).unwrap();
    let manifest = selected
        .neutral()
        .communication_manifest()
        .expect("prediction-free Llama TP must retain neutral communication");
    assert_eq!(manifest.world_size(), 2);
    assert_eq!(manifest.rank(), 0);
    assert!(!manifest.groups().is_empty());
    assert!(selected.rank_context().is_some());
}

#[test]
fn gguf_llama_pp_selects_neutral_partitioned_execution() {
    let root = tempfile::tempdir().unwrap();
    let model = root.path().join("model.gguf");
    write_minimal_llama_gguf(&model, GgmlType::F16);
    let inspection = eredu_architectures::configuration::inspect_artifact(&model).unwrap();
    let topology = crate::test_parallel_rank(0, 1, 2, 1);
    let options = crate::MlxLoadRequest::with_parallel(
        topology,
        crate::backend::DeviceAssignment::new(safemlx::DeviceType::Cpu, 0),
        eredu_runtime::PipelineWireContract::new(eredu_runtime::PipelineActivationDtype::Float32),
        1,
        1,
        crate::MlxLoadRequest::test_communication_completion_policy(),
    )
    .unwrap();
    let selected = super::select_preparation(&inspection, options).unwrap();
    let manifest = selected
        .neutral()
        .communication_manifest()
        .expect("prediction-free Llama PP must retain neutral communication");
    assert_eq!(manifest.world_size(), 2);
    assert!(!manifest.routes().is_empty());
    let session_group = selected
        .neutral()
        .partitioned_session_group()
        .expect("PP admission must select one session-wide publication group");
    let descriptor = manifest
        .groups()
        .iter()
        .find(|group| group.id() == session_group)
        .expect("selected session group must be present in the manifest");
    assert_eq!(descriptor.creation_order(), 0);
    assert_eq!(descriptor.members(), [0, 1]);
    assert_eq!(descriptor.local_index(), Some(0));
    assert_eq!(
        descriptor
            .requirements()
            .operations()
            .iter()
            .map(eredu_runtime::CommunicationOperationRequirement::operation)
            .collect::<Vec<_>>(),
        [
            eredu_runtime::CommunicationOperation::Broadcast,
            eredu_runtime::CommunicationOperation::FailureAgreement,
        ]
    );
    let broadcast = &descriptor.requirements().operations()[0];
    assert_eq!(
        broadcast.dtypes(),
        [eredu_core::checkpoint::TensorDtype::F32]
    );
    assert!(broadcast.exact_completion());
    assert_eq!(broadcast.limits().unwrap().max_tensors(), 1);
    assert_eq!(broadcast.limits().unwrap().max_tensor_rank(), 3);
    assert!(descriptor.requirements().operations()[1].limits().is_none());
    assert!(selected.rank_context().is_some());
    assert!(selected.neutral().communication_manifest().is_some());
}

#[test]
fn gguf_llama_bounded_residency_selects_neutral_partitioned_execution() {
    let root = tempfile::tempdir().unwrap();
    let model = root.path().join("model.gguf");
    write_minimal_llama_gguf(&model, GgmlType::F16);
    let inspection = eredu_architectures::configuration::inspect_artifact(&model).unwrap();
    let layerwise = eredu_runtime::WeightResidency::layerwise_host(
        eredu_runtime::LayerwiseLoadOptions::new(OffloadConfig::new(None, None, 1).unwrap()),
    );
    let dense = eredu_runtime::WeightResidency::dense_disk_stream(
        eredu_runtime::DenseDiskStreamLoadOptions::new(u64::MAX, u64::MAX, 1, 1).unwrap(),
    );

    for residency in [layerwise, dense] {
        let topology = crate::test_parallel_rank(0, 2, 1, 1);
        let options = crate::MlxLoadRequest::from_normalized(
            eredu_runtime::NormalizedLoadRequest::default().with_weight_residency(residency),
        )
        .with_parallel_topology(
            topology,
            crate::backend::DeviceAssignment::new(safemlx::DeviceType::Cpu, 0),
            eredu_runtime::PipelineWireContract::new(
                eredu_runtime::PipelineActivationDtype::Float32,
            ),
            1,
            1,
            crate::MlxLoadRequest::test_communication_completion_policy(),
        )
        .unwrap();

        let selected = super::select_preparation(&inspection, options).unwrap();
        assert!(selected.neutral().communication_manifest().is_some());
        assert!(selected.rank_context().is_some());
    }
}

#[test]
fn gguf_llama_tp_transform_selects_the_immutable_neutral_route() {
    let root = tempfile::tempdir().unwrap();
    let model = root.path().join("model.gguf");
    write_minimal_llama_gguf(&model, GgmlType::F16);
    let inspection = eredu_architectures::configuration::inspect_artifact(&model).unwrap();
    let topology = crate::test_parallel_rank(0, 2, 1, 1);
    let options = crate::MlxLoadRequest::from_normalized(
        eredu_runtime::NormalizedLoadRequest::with_quantization(
            eredu_core::QuantizationRequest::Affine {
                group_size: 16,
                bits: 4,
            },
        ),
    )
    .with_parallel_topology(
        topology,
        crate::backend::DeviceAssignment::new(safemlx::DeviceType::Cpu, 0),
        eredu_runtime::PipelineWireContract::new(eredu_runtime::PipelineActivationDtype::Float32),
        1,
        1,
        crate::MlxLoadRequest::test_communication_completion_policy(),
    )
    .unwrap();
    crate::tests::support::path_instrumentation::reset();

    let selected = super::select_preparation(&inspection, options).unwrap();
    assert!(selected.neutral().communication_manifest().is_some());
    assert!(selected.rank_context().is_some());
}

#[cfg(feature = "image")]
#[test]
fn mlx_processor_consumes_retained_qwen_plan_after_sidecar_removal() {
    use eredu_core::{ArtifactFormat, ModelConfigurationResolver};

    let root = tempfile::tempdir().unwrap();
    let sidecar = root
        .path()
        .join(eredu_architectures::processor_plan::PROCESSOR_CONFIG_FILENAME);
    std::fs::write(
        &sidecar,
        br#"{
            "size":{"shortest_edge":16,"longest_edge":64},
            "patch_size":2,"temporal_patch_size":2,"merge_size":2,
            "image_mean":[0.0,0.0,0.0],"image_std":[1.0,1.0,1.0]
        }"#,
    )
    .unwrap();
    let config = serde_json::json!({
        "model_type": "qwen3_vl", "image_token_id": 61, "video_token_id": 62,
        "vision_start_token_id": 44, "vision_end_token_id": 45,
        "tie_word_embeddings": true,
        "text_config": {
            "model_type": "qwen3_vl_text", "hidden_size": 32,
            "num_hidden_layers": 3, "intermediate_size": 64,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
            "rms_norm_eps": 0.000001, "vocab_size": 64,
            "max_position_embeddings": 128, "rope_theta": 1000000.0,
            "rope_scaling": {"mrope_section": [2, 1, 1], "mrope_interleaved": true}
        },
        "vision_config": {
            "depth": 4, "hidden_size": 16, "intermediate_size": 24,
            "num_heads": 4, "num_position_embeddings": 16, "in_channels": 3,
            "patch_size": 2, "spatial_merge_size": 2, "temporal_patch_size": 2,
            "out_hidden_size": 32, "deepstack_visual_indexes": [1, 3]
        }
    });
    let (configuration, resolved_plan) = eredu_architectures::configuration::MODEL_CONFIGURATIONS
        .resolve_safetensors(&config)
        .unwrap()
        .into_parts();
    let checkpoint = resolved_plan
        .safetensors_architecture()
        .unwrap()
        .checkpoint();
    let catalog = eredu_core::checkpoint::TensorCatalog::new(
        checkpoint
            .common_tensors
            .iter()
            .chain(
                checkpoint
                    .layout_groups
                    .iter()
                    .filter(|group| group.required)
                    .filter_map(|group| group.variants.first())
                    .flat_map(|variant| variant.tensors.iter()),
            )
            .filter(|tensor| {
                tensor.requirement == eredu_checkpoint::schema::TensorRequirement::Required
            })
            .map(|tensor| eredu_core::checkpoint::TensorDescriptor {
                name: tensor.key.clone(),
                shape: tensor.shape.clone(),
                dtype: eredu_core::checkpoint::TensorDtype::F32,
                storage: None,
            }),
    )
    .unwrap();
    let architecture_plan = eredu_architectures::configuration::MODEL_CONFIGURATIONS
        .artifact_plan(
            root.path(),
            ArtifactFormat::SafeTensors,
            &configuration,
            &catalog,
            None,
            resolved_plan,
        )
        .unwrap();
    std::fs::remove_file(sidecar).unwrap();

    assert_eq!(
        architecture_plan.model_kind(),
        eredu_architectures::ModelKind::Qwen3Vl
    );
    assert!(crate::composition::mlx::ModelProcessor::from_plan(&architecture_plan).is_some());
}

#[test]
fn resolved_floating_dtype_selects_runtime_state_width() {
    for (dtype, bytes) in [
        (TensorDtype::F16, 2),
        (TensorDtype::Bf16, 2),
        (TensorDtype::F32, 4),
        (TensorDtype::F64, 8),
    ] {
        assert_eq!(mlx_floating_state_dtype_bytes(&dtype).unwrap().get(), bytes);
    }
}

#[test]
fn dense_half_gguf_embeddings_select_two_byte_runtime_state() {
    for dtype in [GgmlType::F16, GgmlType::Bf16] {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("model.gguf");
        write_minimal_llama_gguf(&path, dtype);
        let inspection = eredu_core::inspect_artifact(
            &path,
            &eredu_architectures::configuration::MODEL_CONFIGURATIONS,
        )
        .unwrap();

        assert_eq!(
            inspected_floating_state_dtype_bytes(&inspection)
                .unwrap()
                .get(),
            2
        );
    }
}

#[test]
fn packed_embedding_dtype_uses_known_mlx_materialization_width() {
    assert_eq!(
        mlx_floating_state_dtype_bytes(&TensorDtype::Encoded("F8_E4M3".into()))
            .unwrap()
            .get(),
        4
    );
    assert_eq!(
        mlx_floating_state_dtype_bytes(&TensorDtype::U32)
            .unwrap()
            .get(),
        4
    );
}

#[test]
fn invalid_activation_dtype_does_not_silently_default() {
    assert_eq!(
        mlx_floating_state_dtype_bytes(&TensorDtype::U8),
        Err(TensorDtype::U8)
    );
}
