//! Actual composite text execution with shared KV across semantic pipeline cuts.
use super::*;
fn source() -> Fixture {
    crate::components::gemma4_component_fixture(false)
}
fn run_tp(mode: &str) -> serde_json::Value {
    run_partitioned_with_fixture(
        mode,
        eredu_core::ParallelTopology::new(2, 1, 1, 1).unwrap(),
        None,
        source,
    )
}
fn run_pp(mode: &str) -> serde_json::Value {
    run_partitioned_with_fixture(
        mode,
        eredu_core::ParallelTopology::new(1, 2, 1, 1).unwrap(),
        None,
        source,
    )
}
fn run_combined(mode: &str) -> serde_json::Value {
    run_partitioned_with_fixture(
        mode,
        eredu_core::ParallelTopology::new(2, 2, 1, 1).unwrap(),
        None,
        source,
    )
}
#[test]
#[ignore = "requires Metal and two local Ring processes"]
fn native_managed_composite_tensor_parallel_matches_ordinary_and_controlled() {
    compare_modes(
        "managed_plain::parallel::composite::native_managed_composite_tensor_parallel_matches_ordinary_and_controlled",
        "EREDU_PUBLIC_COMPOSITE_TP_MODE", "PUBLIC_COMPOSITE_TP_RESULT:", "composite TP", run_tp);
}
#[test]
#[ignore = "requires Metal and two local Ring processes"]
fn native_managed_composite_pipeline_matches_ordinary_and_controlled() {
    compare_modes(
        "managed_plain::parallel::composite::native_managed_composite_pipeline_matches_ordinary_and_controlled",
        "EREDU_PUBLIC_COMPOSITE_PP_MODE", "PUBLIC_COMPOSITE_PP_RESULT:", "composite PP", run_pp);
}
#[test]
#[ignore = "requires Metal and four local Ring processes"]
fn native_managed_composite_combined_matches_ordinary_and_controlled() {
    compare_modes_with_world(
        "managed_plain::parallel::composite::native_managed_composite_combined_matches_ordinary_and_controlled",
        "EREDU_PUBLIC_COMPOSITE_COMBINED_MODE", "PUBLIC_COMPOSITE_COMBINED_RESULT:", "composite TP/PP", 4, run_combined);
}
