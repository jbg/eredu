//! Full four-rank public model path: two logical TP pairs and two PP stages.
use super::*;
const CASE:&str="managed_plain::parallel::combined::native_managed_combined_parallel_matches_ordinary_and_controlled";
const MODE: &str = "EREDU_PUBLIC_MANAGED_COMBINED_MODE";
const RESULT: &str = "PUBLIC_MANAGED_COMBINED_RESULT:";
fn run(mode: &str) -> serde_json::Value {
    run_partitioned(mode, eredu_core::ParallelTopology::new(2, 2, 1, 1).unwrap())
}
#[test]
#[ignore = "requires Metal and four local Ring processes"]
fn native_managed_combined_parallel_matches_ordinary_and_controlled() {
    compare_modes_with_world(CASE, MODE, RESULT, "combined TP/PP", 4, run)
}
