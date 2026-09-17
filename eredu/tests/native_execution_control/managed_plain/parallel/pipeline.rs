//! Real two-rank pure PP: the neural TP option stays absent throughout.
use super::*;
const CASE:&str="managed_plain::parallel::pipeline::native_managed_pipeline_parallel_matches_ordinary_and_controlled";
const MODE: &str = "EREDU_PUBLIC_MANAGED_PP_MODE";
const RESULT: &str = "PUBLIC_MANAGED_PP_RESULT:";
fn run(mode: &str) -> serde_json::Value {
    super::run_partitioned(mode, eredu_core::ParallelTopology::new(1, 2, 1, 1).unwrap())
}
#[test]
#[ignore = "requires Metal and two local Ring processes"]
fn native_managed_pipeline_parallel_matches_ordinary_and_controlled() {
    super::compare_modes(CASE, MODE, RESULT, "PP", run)
}
