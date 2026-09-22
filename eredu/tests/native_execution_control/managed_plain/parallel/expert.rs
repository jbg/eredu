//! Full public expert parallel dispatch/return through the existing text driver.
use super::*;

pub(super) fn positive_settings() -> PreparedChatGenerationSettings {
    let mut value = settings(0.0);
    // Recorded rank-local live owners + next admission require 10.64 GB for
    // TP/EP and 10.42 GB for saved EP. The positive path has a 16 GiB total;
    // exact low-capacity refusals remain in the shared lifecycle driver.
    value.inference.memory_limits = eredu_core::MemoryLimitDeclarations::new([(
        "host".into(),
        eredu_core::MemoryLimit::Finite(16 * 1024 * 1024 * 1024),
    )]);
    value
}

fn run_topology(mode: &str, tensor: usize, pipeline: usize) -> serde_json::Value {
    run_partitioned_with_fixture_settings(
        mode,
        eredu_core::ParallelTopology::new(tensor, pipeline, 2, 1).unwrap(),
        None,
        crate::routed_components::routed_fixture,
        if tensor > 1 || pipeline > 1 {
            positive_settings()
        } else {
            settings(0.0)
        },
    )
}

fn run(mode: &str) -> serde_json::Value {
    run_topology(mode, 1, 1)
}

#[test]
#[ignore = "requires Metal and two local Ring processes"]
fn native_managed_expert_parallel_matches_ordinary_and_controlled() {
    compare_modes(
        "managed_plain::parallel::expert::native_managed_expert_parallel_matches_ordinary_and_controlled",
        "EREDU_PUBLIC_MANAGED_EP_MODE",
        "PUBLIC_MANAGED_EP_RESULT:",
        "expert parallel",
        run,
    );
}

// Expert rank is the minor coordinate. In world4 these EP groups are [0,1]
// and [2,3], so the retained Ring realization selects logical local-peer
// transport, while TP groups [0,2]/[1,3] carry the existing world-wave source.
fn run_tensor_expert(mode: &str) -> serde_json::Value {
    run_topology(mode, 2, 1)
}

#[test]
#[ignore = "requires Metal and four local Ring processes"]
fn native_managed_tensor_expert_parallel_local_peers_match_all_drivers() {
    compare_modes_with_world(
        "managed_plain::parallel::expert::native_managed_tensor_expert_parallel_local_peers_match_all_drivers",
        "EREDU_PUBLIC_MANAGED_TP_EP_MODE",
        "PUBLIC_MANAGED_TP_EP_RESULT:",
        "tensor/expert local-peer parallel",
        4,
        run_tensor_expert,
    )
}

// Both nonzero decoder units reside on distinct PP stages. Every rank uses
// the retained routed schedule, including inactive count/transfer/provider
// waves for the other stage and the actual framed hidden-state boundary.
fn run_pipeline_expert(mode: &str) -> serde_json::Value {
    run_topology(mode, 1, 2)
}

#[test]
#[ignore = "requires Metal and four local Ring processes"]
fn native_managed_pipeline_expert_parallel_inactive_waves_match_all_drivers() {
    compare_modes_with_world(
        "managed_plain::parallel::expert::native_managed_pipeline_expert_parallel_inactive_waves_match_all_drivers",
        "EREDU_PUBLIC_MANAGED_PP_EP_MODE",
        "PUBLIC_MANAGED_PP_EP_RESULT:",
        "pipeline/expert inactive-wave parallel",
        4,
        run_pipeline_expert,
    )
}

fn run_tensor_pipeline_expert(mode: &str) -> serde_json::Value {
    run_topology(mode, 2, 2)
}

#[test]
#[ignore = "requires Metal and eight local Ring processes"]
fn native_managed_tensor_pipeline_expert_parallel_matches_all_drivers() {
    compare_modes_with_world(
        "managed_plain::parallel::expert::native_managed_tensor_pipeline_expert_parallel_matches_all_drivers",
        "EREDU_PUBLIC_MANAGED_TP_PP_EP_MODE",
        "PUBLIC_MANAGED_TP_PP_EP_RESULT:",
        "tensor/pipeline/expert parallel",
        8,
        run_tensor_pipeline_expert,
    )
}
