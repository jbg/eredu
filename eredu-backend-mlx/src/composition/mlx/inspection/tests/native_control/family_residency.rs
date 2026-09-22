//! Actual family and residency selections through the funded snapshot provider.

use super::*;
use eredu_core::residency::{MemoryTier, ResidencyPolicy};
use eredu_core::{ExecutionPlan, ResidencyPlan};

fn enter() -> bool {
    if !crate::tests::support::native_process::enter("prepared-native-continuation") {
        return false;
    }
    crate::tests::support::test_utils::initialize_original_sources();
    true
}

fn runtime(root: &Path, residency: ResidencyPlan) -> ModelRuntime<Backend> {
    let plan =
        ExecutionPlan::fully_resident(eredu_core::DevicePlan::new("mlx", "metal:0").unwrap())
            .with_residency(residency);
    let inspection =
        eredu_architectures::configuration::inspect_artifact_with_prepared_gguf_headers(root)
            .unwrap();
    let factory = crate::MlxBackendFactory::default();
    let selected = eredu_core::select_execution_plan_target(&factory, &plan, inspection).unwrap();
    let target = eredu_core::realize_execution_plan_target(&factory, &plan, selected).unwrap();
    target.backend().validate_original_stream_owners().unwrap();
    target.into_runtime().unwrap()
}

fn resident(root: &Path, routed: bool, inspect: impl Fn(&ModelRuntime<Backend>)) {
    sampled_conformance_using(
        routed,
        || runtime(root, ResidencyPlan::FullyResident),
        inspect,
    );
}

#[test]
fn canonical_routed_qwen3_tied_snapshots_preserve_stochastic_future() {
    if !enter() {
        return;
    }
    let root = crate::composition::mlx::replicated_text::tests::tiny_artifact("qwen3_moe", true);
    resident(root.path(), true, |_| {});
}

#[test]
fn canonical_sliding_gemma4_snapshots_cross_the_window_with_equal_future() {
    if !enter() {
        return;
    }
    let root = tempfile::tempdir().unwrap();
    let config = serde_json::json!({
        "model_type": "gemma4", "tie_word_embeddings": false,
        "text_config": {
            "model_type": "gemma4_text", "hidden_size": 16,
            "num_hidden_layers": 3, "intermediate_size": 16,
            "num_attention_heads": 4, "num_key_value_heads": 2,
            "head_dim": 4, "rms_norm_eps": 0.00001,
            "vocab_size": 64, "max_position_embeddings": 128,
            "tie_word_embeddings": false, "attention_k_eq_v": false,
            "layer_types": ["full_attention", "sliding_attention", "full_attention"],
            "sliding_window": 8
        }
    });
    write_configured_safetensors_fixture(root.path(), &config, fixture_value);
    resident(root.path(), false, |_| {});
}

#[test]
fn canonical_affine_qwen3_snapshots_preserve_packed_readout_and_future() {
    if !enter() {
        return;
    }
    let root =
        crate::composition::mlx::replicated_text::tests::tiny_packed_safetensors_artifact("qwen3");
    resident(root.path(), false, |runtime| {
        let head = runtime
            .session()
            .fixture_selected_parameter("lm_head.weight")
            .unwrap();
        assert!(matches!(
            head.executable(),
            eredu_checkpoint::LinearFormat::Affine(_)
        ));
        assert!(matches!(
            head.source_encoding(),
            eredu_checkpoint::SourceTensorEncoding::Safetensors(eredu_checkpoint::StoredDtype::U32)
        ));
    });
}

#[test]
fn canonical_compressed_deepseek_snapshots_preserve_routed_future() {
    if !enter() {
        return;
    }
    let root = tempfile::tempdir().unwrap();
    let config = crate::composition::mlx::replicated_text::tests::routed_deepseek_v3_config();
    write_configured_safetensors_fixture(root.path(), &config, fixture_value);
    resident(root.path(), true, |_| {});
}

#[test]
fn canonical_pooling_snapshots_preserve_partial_windows_and_future() {
    if !enter() {
        return;
    }
    let root = crate::composition::mlx::replicated_text::tests::tiny_heterogeneous_artifact(
        crate::composition::mlx::replicated_text::tests::routed_deepseek_v4_config(),
    );
    resident(root.path(), true, |_| {});
}

fn bounded(residency: ResidencyPlan, tier: MemoryTier) {
    if !enter() {
        return;
    }
    let root = tempfile::tempdir().unwrap();
    let config = serde_json::json!({
        "model_type": "llama", "hidden_size": 16, "num_hidden_layers": 3,
        "intermediate_size": 32, "num_attention_heads": 4, "num_key_value_heads": 2,
        "rms_norm_eps": 0.00001, "vocab_size": 64, "max_position_embeddings": 128
    });
    write_configured_safetensors_fixture(root.path(), &config, fixture_value);
    sampled_conformance_using(
        false,
        || runtime(root.path(), residency.clone()),
        |runtime| {
            let report = runtime.session().residency_report().unwrap().unwrap();
            let layers = report
                .units()
                .iter()
                .filter(|unit| unit.planned_tier() == tier)
                .collect::<Vec<_>>();
            assert_eq!(layers.len(), 3);
            let depth = runtime
                .session()
                .fixture_device_unit_window(layers.len())
                .unwrap();
            assert!(layers
                .iter()
                .all(|unit| unit.policy() != ResidencyPolicy::Pinned));
            assert!(layers
                .iter()
                .all(|unit| unit.host_resident() == (tier == MemoryTier::Host)));
            let live = layers.iter().filter(|unit| unit.device_resident()).count();
            assert!(live > 0 && live <= depth);
            let pinned = report
                .units()
                .iter()
                .filter(|unit| unit.policy() == ResidencyPolicy::Pinned)
                .count();
            assert!(
                report
                    .offload()
                    .peak_resident_units()
                    .get(MemoryTier::Device)
                    <= pinned + depth
            );
            assert!(report.offload().tier_evictions(MemoryTier::Device).count() > 0);
            if tier == MemoryTier::Disk {
                assert!(
                    runtime
                        .session()
                        .fixture_detached_payload_read_bytes()
                        .unwrap()
                        > 0,
                    "the selected disk source must perform payload reads"
                );
            }
        },
    );
}

#[test]
fn canonical_host_window_one_snapshots_keep_bounded_residency_and_future() {
    bounded(
        ResidencyPlan::LayerwiseHost {
            device_layer_window: 1,
            device_budget_bytes: Some(1 << 26),
            host_budget_bytes: Some(1 << 26),
        },
        MemoryTier::Host,
    );
}

#[test]
fn canonical_host_window_two_snapshots_keep_bounded_residency_and_future() {
    bounded(
        ResidencyPlan::LayerwiseHost {
            device_layer_window: 2,
            device_budget_bytes: Some(1 << 26),
            host_budget_bytes: Some(1 << 26),
        },
        MemoryTier::Host,
    );
}

#[test]
fn canonical_direct_disk_snapshots_keep_real_payload_reads_and_future() {
    bounded(
        ResidencyPlan::DenseDiskStream {
            device_budget_bytes: 1 << 26,
            host_budget_bytes: 0,
            host_lookahead: 0,
            background_queue: 0,
        },
        MemoryTier::Disk,
    );
}
