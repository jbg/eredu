//! Retained CPU equations and transport use the same public parallel driver.
use super::*;

fn run(mode: &str, tp: usize, pp: usize, saved_state: bool) -> serde_json::Value {
    run_partitioned_with_load_on(
        mode,
        eredu_core::ParallelTopology::new(tp, pp, 1, 1).unwrap(),
        saved_state.then_some(saved::lifecycle),
        || fixture(false),
        eredu_runtime::NormalizedLoadRequest::default(),
        safemlx::DeviceType::Cpu,
    )
}
#[test]
#[ignore = "requires native CPU sources and two local Ring processes"]
fn native_cpu_tensor_parallel_matches_ordinary_and_controlled() {
    compare_modes(
        "managed_plain::parallel::cpu::native_cpu_tensor_parallel_matches_ordinary_and_controlled",
        "EREDU_PUBLIC_CPU_TP_MODE",
        "PUBLIC_CPU_TP_RESULT:",
        "CPU TP",
        |mode| run(mode, 2, 1, false),
    )
}
#[test]
#[ignore = "requires native CPU sources and two local Ring processes"]
fn native_cpu_tensor_parallel_saved_state_preserves_future_and_limits() {
    compare_selected_modes(
        "managed_plain::parallel::cpu::native_cpu_tensor_parallel_saved_state_preserves_future_and_limits",
        "EREDU_PUBLIC_CPU_TP_SAVED_MODE", "PUBLIC_CPU_TP_SAVED_RESULT:", "CPU TP saved state",
        2, &["lifecycle"], |mode| run(mode, 2, 1, true),
    )
}
#[test]
#[ignore = "requires native CPU sources and four local Ring processes"]
fn native_cpu_combined_parallel_matches_ordinary_and_controlled() {
    compare_modes_with_world(
        "managed_plain::parallel::cpu::native_cpu_combined_parallel_matches_ordinary_and_controlled",
        "EREDU_PUBLIC_CPU_COMBINED_MODE", "PUBLIC_CPU_COMBINED_RESULT:", "CPU TP/PP", 4,
        |mode| run(mode, 2, 2, false),
    )
}
