//! The complete execution-plan path owns its selected native world and state.
use super::*;
use eredu_core::BackendProvider;
use eredu_nn::Tensor as _;
use eredu_runtime::{ParallelLoadRequest, PipelineActivationDtype, PipelineWireContract};

pub(super) const WORKER_FLAG: &str = "EREDU_RING_EXECUTION_PLAN_TARGET";
mod media;
pub(super) const MEDIA_WORKER_FLAG: &str = "EREDU_RING_EXECUTION_PLAN_MEDIA";

#[test]
#[ignore = "spawns CPU Ring ranks and opens loopback sockets; run explicitly"]
fn native_spark_parallel_execution_plan_target() {
    assert!(distributed::is_available(Backend::Ring));
    for (axes, residency) in [
        ("tp", WorkerResidency::FullyResident),
        ("pp", WorkerResidency::LayerwiseHost),
        ("ep", WorkerResidency::DenseDiskStream),
        ("tp-pp-ep", WorkerResidency::DenseDiskStream),
    ] {
        let checkpoint = tempfile::tempdir().unwrap();
        write_spark_fixture(checkpoint.path(), Dtype::F32);
        let path = checkpoint.path().to_path_buf();
        run_ring_pipeline_processes(
            residency,
            FixtureFamily::MuseSpark,
            WorkerMode::ExecutionPlanTarget,
            checkpoint,
            path,
            Some(axes),
        );
    }
}

#[test]
#[ignore = "spawns local CPU Ring ranks and opens loopback sockets; run explicitly"]
fn native_spark_tp4_repeated_kv_execution_plan_target() {
    assert!(distributed::is_available(Backend::Ring));
    let checkpoint = tempfile::tempdir().unwrap();
    write_spark_fixture(checkpoint.path(), Dtype::F32);
    let path = checkpoint.path().to_path_buf();
    run_ring_pipeline_processes(
        WorkerResidency::DenseDiskStream,
        FixtureFamily::MuseSpark,
        WorkerMode::ExecutionPlanTarget,
        checkpoint,
        path,
        Some("tp4"),
    );
}

fn spark_text_config() -> serde_json::Value {
    let mut config: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../validation/fixtures/muse-spark/bf16-config.json"
    ))
    .unwrap();
    config.as_object_mut().unwrap().remove("vision_config");
    config.as_object_mut().unwrap().remove("audio_config");
    for (name, id) in [
        ("image_token_id", 42),
        ("video_token_id", 43),
        ("audio_token_id", 44),
        ("eos_token_id", 63),
    ] {
        config[name] = id.into();
    }
    let text = &mut config["text_config"];
    for (name, value) in [
        ("hidden_size", 16),
        ("num_hidden_layers", 2),
        ("intermediate_size", 32),
        ("num_attention_heads", 4),
        ("num_key_value_heads", 2),
        ("head_dim", 4),
        ("vocab_size", 64),
        ("max_position_embeddings", 128),
        // Cached comparison crosses eviction as well as the full-attention clock.
        ("sliding_window", 3),
        ("moe_hidden_size", 8),
        ("moe_intermediate_size", 12),
        ("num_local_experts", 4),
        ("num_experts_per_tok", 2),
    ] {
        text[name] = value.into();
    }
    text["layer_types"] = serde_json::json!(["sliding_attention", "full_attention"]);
    text["layer_rope_theta"] = serde_json::json!([500000.0, 0.0]);
    text["mlp_layer_types"] = serde_json::json!(["sparse", "sparse"]);
    config
}

pub(super) fn write_spark_fixture(directory: &Path, dtype: Dtype) {
    write_spark_config(directory, dtype, &spark_text_config());
}

fn write_spark_config(directory: &Path, dtype: Dtype, config: &serde_json::Value) {
    let resolved = eredu_architectures::configuration::resolve_model_config(config).unwrap();
    let schema = resolved.architecture.checkpoint();
    let tensors = schema
        .common_tensors
        .iter()
        .chain(
            schema
                .layout_groups
                .iter()
                .flat_map(|g| &g.variants[0].tensors),
        )
        .map(|tensor| {
            let seed = tensor
                .key
                .bytes()
                .fold(0u32, |s, b| s.wrapping_mul(31).wrapping_add(u32::from(b)));
            let values = (0..tensor.shape.iter().product::<usize>())
                .map(|i| {
                    let delta = ((i * 7 + (seed % 97) as usize) % 41) as f32 - 20.0;
                    let value = if tensor.key.contains("norm") && tensor.key.ends_with("weight") {
                        0.9 + delta * 0.002
                    } else {
                        delta * 0.008
                    };
                    value
                })
                .collect::<Vec<_>>();
            let bytes = match dtype {
                Dtype::BF16 => values
                    .iter()
                    .flat_map(|value| half::bf16::from_f32(*value).to_bits().to_le_bytes())
                    .collect::<Vec<_>>(),
                Dtype::F32 => values
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect::<Vec<_>>(),
                _ => panic!("fixture requires F32 or BF16"),
            };
            (tensor.key.clone(), tensor.shape.clone(), bytes)
        })
        .collect::<Vec<_>>();
    let views = tensors.iter().map(|(name, shape, bytes)| {
        (
            name.as_str(),
            TensorView::new(dtype, shape.clone(), bytes).unwrap(),
        )
    });
    serialize_to_file(views, None, &directory.join("model.safetensors")).unwrap();
    std::fs::write(
        directory.join("config.json"),
        serde_json::to_vec(&config).unwrap(),
    )
    .unwrap();
}

#[test]
#[ignore = "spawns CPU Ring ranks and opens loopback sockets; run explicitly"]
fn native_parallel_execution_plan_target() {
    assert!(distributed::is_available(Backend::Ring));
    for (axes, residency) in [
        ("tp", WorkerResidency::FullyResident),
        ("pp", WorkerResidency::LayerwiseHost),
        ("tp-pp", WorkerResidency::DenseDiskStream),
    ] {
        let checkpoint = tempfile::tempdir().unwrap();
        write_muse_glimmer_component_fixture(checkpoint.path(), false, false);
        let path = checkpoint.path().to_path_buf();
        run_ring_pipeline_processes(
            residency,
            FixtureFamily::MuseGlimmer,
            WorkerMode::ExecutionPlanTarget,
            checkpoint,
            path,
            Some(axes),
        );
    }
}

pub(super) fn worker(checkpoint: &Path, rank: eredu_core::ParallelRankTopology) {
    use crate::composition::mlx::automatic::MlxBackendFactory;
    let parallel = ParallelLoadRequest::new(
        rank,
        PipelineWireContract::new(PipelineActivationDtype::Float32),
        1,
        32,
        ring_completion_policy(),
    )
    .unwrap();
    let factory = MlxBackendFactory::default().with_parallel_execution(parallel, Backend::Ring);
    let mut plan = ExecutionPlan::fully_resident(DevicePlan::new("mlx", "cpu:0").unwrap())
        .with_topology(rank.topology())
        .with_max_cached_shards(1);
    if std::env::var_os(DENSE_STREAM).is_some() {
        plan = plan.with_residency(eredu_core::ResidencyPlan::DenseDiskStream {
            device_budget_bytes: 1 << 20,
            host_budget_bytes: 1 << 20,
            host_lookahead: 1,
            background_queue: 1,
        });
    } else if std::env::var_os(LAYERWISE_HOST).is_some() {
        plan = plan.with_residency(eredu_core::ResidencyPlan::LayerwiseHost {
            device_layer_window: 1,
            device_budget_bytes: Some(1 << 20),
            host_budget_bytes: Some(1 << 20),
        });
    }
    let inspection = eredu_architectures::configuration::inspect_artifact(checkpoint).unwrap();
    crate::tests::support::path_instrumentation::reset();
    let selected =
        eredu_core::select_execution_plan_target(&factory, &plan, inspection.clone()).unwrap();
    assert_eq!(
        selected
            .selected_preparation()
            .neutral()
            .execution()
            .parallel_topology(),
        Some(rank)
    );
    assert_eq!(
        crate::tests::support::path_instrumentation::snapshot(),
        Default::default()
    );
    assert_eq!(
        crate::tests::support::path_instrumentation::target_native_resource_realization_attempts(),
        0
    );
    // A different factory cannot replace the initializer retained in selection.
    let realizing_factory =
        MlxBackendFactory::default().with_parallel_execution(parallel, Backend::Mpi);
    let mut actual = eredu_core::realize_execution_plan_target(&realizing_factory, &plan, selected)
        .unwrap()
        .into_runtime()
        .unwrap();
    assert!(actual.backend().devices().unwrap()[0].1.collectives());
    let assistant_stream = actual.backend().stream().clone();
    crate::composition::mlx::speculative::verify_retained_assistant_tensor_group(
        actual.session_mut(),
        &assistant_stream,
        rank,
    );
    let local_plan = ExecutionPlan::fully_resident(DevicePlan::new("mlx", "cpu:0").unwrap());
    let local_factory = MlxBackendFactory::default();
    let selected =
        eredu_core::select_execution_plan_target(&local_factory, &local_plan, inspection).unwrap();
    let mut ordinary =
        eredu_core::realize_execution_plan_target(&local_factory, &local_plan, selected)
            .unwrap()
            .into_runtime()
            .unwrap();
    assert!(!ordinary.backend().devices().unwrap()[0].1.collectives());
    let owns_output = rank.global_rank()
        == rank
            .topology()
            .rank_for(eredu_core::ParallelCoordinates::new(
                0,
                rank.pipeline_parallel_size() - 1,
                0,
                0,
            ))
            .unwrap();
    let (parts, prefix) = if std::env::var_os(MEDIA_WORKER_FLAG).is_some() {
        media::input()
    } else {
        let tokens = Array::from_slice(&[1u32, 2], &[1, 2]);
        (vec![text_input_part(&tokens)], vec![1, 2])
    };
    let mut output = actual
        .prefill(synthetic_prediction_input(&parts, &prefix))
        .unwrap()
        .wait()
        .unwrap();
    let mut reference = ordinary
        .prefill(synthetic_prediction_input(&parts, &prefix))
        .unwrap()
        .wait()
        .unwrap();
    for step in 0..4 {
        assert_eq!(output.logits().is_some(), owns_output);
        if let Some(logits) = output.logits() {
            let expected = reference.logits().unwrap();
            assert_eq!(logits.shape(), expected.shape());
            let values = logits.to_f32_vec(actual.backend().stream()).unwrap();
            let expected = expected.to_f32_vec(ordinary.backend().stream()).unwrap();
            assert!(expected.iter().any(|v| *v != 0.));
            for (value, expected) in values.iter().zip(expected) {
                assert!(
                    value.is_finite() && (value - expected).abs() <= 2e-4,
                    "rank {} step {step}: {value} vs {expected}",
                    rank.global_rank()
                );
            }
        }
        if step < 3 {
            let token = Array::from_slice(&[3u32 + step as u32], &[1, 1]);
            output = actual.decode(token.clone()).unwrap().wait().unwrap();
            reference = ordinary.decode(token).unwrap().wait().unwrap();
        }
    }
    verify_distributed_control_branches(&mut actual, owns_output);
    actual.backend().synchronize().unwrap();
}
