//! Both independent models use the selected one-member cache in the existing driver.
use super::*;

const MEMBER_BYTES: u64 = 3 * 16 * 6 * 4;
const CASE: &str = "managed_plain::speculative::expert_cache::native_original_independent_cache_speculation_matches_ordinary_and_controlled";

fn plan(execution: ExecutionPlan) -> ExecutionPlan {
    // Independent drafting copies this selected plan; both actual providers use
    // the same one-member capacity policy with separate physical bank owners.
    execution.with_expert_cache(Some(eredu_core::ExpertCachePlan::new(
        Some(MEMBER_BYTES),
        Some(MEMBER_BYTES),
        MEMBER_BYTES,
        MEMBER_BYTES,
        eredu_core::residency::CacheEvictionPolicy::LeastRecentlyUsed,
    )))
}
fn pressure(model: &LoadedModel<eredu_backend_mlx::backend::MlxBackend<'_>>) {
    let report = model
        .expert_cache_telemetry()
        .unwrap()
        .expect("selected independent bank");
    assert_eq!(report.owned_experts, 4);
    assert_eq!(report.owned_bytes, 4 * MEMBER_BYTES);
    assert!(report.device_resident_experts <= 1, "{report:?}");
    assert!(report.device_resident_bytes <= MEMBER_BYTES, "{report:?}");
    assert_eq!(
        report.peak_device_resident_bytes, MEMBER_BYTES,
        "{report:?}"
    );
    eprintln!("PUBLIC_SPECULATIVE_CACHE_PRESSURE:{report:?}");
}

#[test]
#[ignore = "run after original independent-cache generation and source activation pass"]
fn native_original_independent_cache_speculation_matches_ordinary_and_controlled() {
    verify_mode_results(CASE, "EREDU_PUBLIC_SPECULATIVE_CACHE_MODE", |mode| {
        let make = || super::super::super::components::lfm2_fixture_with_banks(true);
        let (value, rounds) = run_artifacts_on_inspected(
            mode,
            make(),
            make(),
            eredu_core::DevicePlan::new("mlx", "metal:0").unwrap(),
            DraftPlacementPlan::Target,
            settings(0.0),
            plan,
            Some(pressure),
        );
        // Five prompt rows at chunk=2 and four committed outputs require both
        // uneven prefill and cached target/assistant verification work.
        assert!(
            rounds >= 2,
            "expected multiple completed speculative rounds, got {rounds}"
        );
        value
    });
}
