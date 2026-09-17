//! Exact paged local state through the same two-rank public generation driver.
use super::*;
const CASE: &str = "managed_plain::parallel::paged::native_managed_paged_tensor_parallel_matches_ordinary_and_controlled";
const MODE: &str = "EREDU_PUBLIC_MANAGED_PAGED_TP_MODE";
const RESULT: &str = "PUBLIC_MANAGED_PAGED_TP_RESULT:";
fn run(mode: &str) -> serde_json::Value {
    // Same finite state recipe as the public replicated paging case: pages
    // seal across 2/2/1 prefill chunks, followed by a live tail and four outputs.
    let state = eredu_runtime::CacheResidencyPolicy::Paged(
        eredu_runtime::PagedCacheOptions::new(3, 8 << 20, 0, 1)
            .unwrap()
            .with_full_attention(true),
    );
    run_partitioned_with_state(
        mode,
        eredu_core::ParallelTopology::new(2, 1, 1, 1).unwrap(),
        None,
        || fixture(false),
        state,
    )
}
#[test]
#[ignore = "requires Metal and two local Ring processes"]
fn native_managed_paged_tensor_parallel_matches_ordinary_and_controlled() {
    compare_modes(CASE, MODE, RESULT, "paged TP", run)
}
