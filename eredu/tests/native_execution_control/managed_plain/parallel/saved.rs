//! Fresh two-rank controlled saved-state lifecycle through the shared driver.
use super::*;
const CASE:&str="managed_plain::parallel::saved::native_managed_tensor_parallel_saved_state_matches_uninterrupted_future";
const MODE: &str = "EREDU_PUBLIC_MANAGED_TP_SAVED_MODE";
const RESULT: &str = "PUBLIC_MANAGED_TP_SAVED_RESULT:";
pub(super) fn lifecycle(
    model: LoadedModel<eredu_backend_mlx::backend::MlxBackend<'_>>,
    root: Fixture,
) -> serde_json::Value {
    super::super::snapshot::check_resume_loaded(
        model,
        root,
        1,
        &[8, 38, 26, 1],
        Some(&|| eprintln!("PUBLIC_TP_SAVED_PHASE committed")),
        "PUBLIC_TP_SAVED_PHASE",
    )
}
fn run(mode: &str) -> serde_json::Value {
    assert!(matches!(mode, "serial" | "lifecycle"));
    run_partitioned_with_lifecycle(
        mode,
        eredu_core::ParallelTopology::new(2, 1, 1, 1).unwrap(),
        Some(lifecycle),
    )
}
#[test]
#[ignore = "requires Metal and two local Ring processes"]
fn native_managed_tensor_parallel_saved_state_matches_uninterrupted_future() {
    // Existing TP generation parity is a separate case. This runs only the new
    // two-rank snapshot/cancel/refuse/restore/fork path against a serial oracle.
    compare_selected_modes(CASE, MODE, RESULT, "TP saved state", 2, &["lifecycle"], run)
}


fn run_pipeline(mode: &str) -> serde_json::Value {
    assert!(matches!(mode, "serial" | "lifecycle"));
    run_partitioned_with_lifecycle(mode,
        eredu_core::ParallelTopology::new(1, 2, 1, 1).unwrap(), Some(lifecycle))
}

#[test]
#[ignore = "requires Metal and two local Ring processes"]
fn native_managed_pipeline_parallel_saved_state_matches_uninterrupted_future() {
    compare_selected_modes(
        "managed_plain::parallel::saved::native_managed_pipeline_parallel_saved_state_matches_uninterrupted_future",
        "EREDU_PUBLIC_MANAGED_PP_SAVED_MODE", "PUBLIC_MANAGED_PP_SAVED_RESULT:",
        "PP saved state", 2, &["lifecycle"], run_pipeline,
    )
}

fn run_combined(mode: &str) -> serde_json::Value {
    assert!(matches!(mode, "serial" | "lifecycle"));
    run_partitioned_with_lifecycle(mode,
        eredu_core::ParallelTopology::new(2, 2, 1, 1).unwrap(), Some(lifecycle))
}

#[test]
#[ignore = "requires Metal and four local Ring processes"]
fn native_managed_combined_parallel_saved_state_matches_uninterrupted_future() {
    compare_selected_modes(
        "managed_plain::parallel::saved::native_managed_combined_parallel_saved_state_matches_uninterrupted_future",
        "EREDU_PUBLIC_MANAGED_COMBINED_SAVED_MODE", "PUBLIC_MANAGED_COMBINED_SAVED_RESULT:",
        "combined TP/PP saved state", 4, &["lifecycle"], run_combined,
    )
}

// This fixed oracle was established by the existing nonzero routed fixture's
// ordinary run. The parent harness independently reruns its serial oracle.
const EXPERT_IDS: &[u32] = &[8, 38, 39, 56];

fn expert_saved_settings() -> PreparedChatGenerationSettings {
    let mut settings = super::expert::positive_settings();
    // The pending restore retains its original request and saved source while
    // admitting a new request. Recorded held + next demand is 19,006,893,119
    // bytes, exceeding the generation-only 16 GiB positive fixture. Leave room
    // for the later independent branch; one-byte refusal/copy limits stay in
    // the shared lifecycle helper and production admission remains unchanged.
    settings.inference.managed_memory_capacity_bytes = Some(32 * 1024 * 1024 * 1024);
    settings
}


fn expert_pending(
    model: LoadedModel<eredu_backend_mlx::backend::MlxBackend<'_>>,
    root: Fixture,
) -> serde_json::Value {
    super::super::snapshot::check_resume_loaded_with_settings(
        model, root, 0, EXPERT_IDS,
        Some(&|| eprintln!("PUBLIC_EXPERT_SAVED_PHASE pending")),
        "PUBLIC_EXPERT_SAVED_PHASE", expert_saved_settings(),
    )
}

fn expert_committed(
    model: LoadedModel<eredu_backend_mlx::backend::MlxBackend<'_>>,
    root: Fixture,
) -> serde_json::Value {
    super::super::snapshot::check_resume_loaded_with_settings(
        model, root, 1, EXPERT_IDS,
        Some(&|| eprintln!("PUBLIC_EXPERT_SAVED_PHASE committed")),
        "PUBLIC_EXPERT_SAVED_PHASE", expert_saved_settings(),
    )
}

fn run_expert_topology(mode: &str, tensor: usize, pipeline: usize) -> serde_json::Value {
    let lifecycle = match mode {
        "pending" => expert_pending,
        "serial" | "committed" => expert_committed,
        _ => panic!("unexpected expert saved-state mode {mode}"),
    };
    run_partitioned_with_fixture(
        mode,
        eredu_core::ParallelTopology::new(tensor, pipeline, 2, 1).unwrap(),
        Some(lifecycle),
        crate::routed_components::routed_fixture,
    )
}

fn run_expert(mode: &str) -> serde_json::Value {
    run_expert_topology(mode, 1, 1)
}

#[test]
#[ignore = "requires Metal and two local Ring processes"]
fn native_managed_expert_parallel_saved_state_matches_uninterrupted_future() {
    compare_selected_modes(
        "managed_plain::parallel::saved::native_managed_expert_parallel_saved_state_matches_uninterrupted_future",
        "EREDU_PUBLIC_MANAGED_EP_SAVED_MODE", "PUBLIC_MANAGED_EP_SAVED_RESULT:",
        "EP pending/committed saved state", 2, &["pending", "committed"], run_expert,
    )
}

fn run_tensor_pipeline_expert(mode: &str) -> serde_json::Value {
    run_expert_topology(mode, 2, 2)
}

#[test]
#[ignore = "requires Metal and eight local Ring processes"]
fn native_managed_tensor_pipeline_expert_saved_state_matches_uninterrupted_future() {
    compare_selected_modes(
        "managed_plain::parallel::saved::native_managed_tensor_pipeline_expert_saved_state_matches_uninterrupted_future",
        "EREDU_PUBLIC_MANAGED_TP_PP_EP_SAVED_MODE", "PUBLIC_MANAGED_TP_PP_EP_SAVED_RESULT:",
        "TP/PP/EP pending/committed saved state", 8, &["pending", "committed"],
        run_tensor_pipeline_expert,
    )
}
