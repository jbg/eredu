//! The existing public text and saved-state drivers with actual independent banks.
use super::*;

const MEMBER_BYTES: u64 = 3 * 16 * 6 * 4;
#[derive(Clone, Copy)]
enum Residency { Resident, Host, ForegroundDisk }
impl Residency {
    fn execution(self) -> ExecutionPlan {
        let residency = match self {
            Self::Resident => eredu_core::ResidencyPlan::FullyResident,
            Self::Host => eredu_core::ResidencyPlan::LayerwiseHost {
                device_layer_window: 1, device_budget_bytes: Some(8 << 20),
                host_budget_bytes: Some(8 << 20),
            },
            Self::ForegroundDisk => eredu_core::ResidencyPlan::DenseDiskStream {
                device_budget_bytes: 8 << 20, host_budget_bytes: 0,
                host_lookahead: 0, background_queue: 0,
            },
        };
        ExecutionPlan::fully_resident(eredu_core::DevicePlan::new("mlx", "metal:0").unwrap())
            .with_residency(residency)
            .with_required_session_capabilities(SessionCapabilities::new(true, true, true))
            .with_expert_cache(Some(eredu_core::ExpertCachePlan::new(
                Some(MEMBER_BYTES), Some(MEMBER_BYTES), MEMBER_BYTES, MEMBER_BYTES,
                eredu_core::residency::CacheEvictionPolicy::LeastRecentlyUsed,
            )))
    }
}
fn fixture() -> Fixture { super::super::components::lfm2_fixture_with_banks(true) }
fn pressure(model: &LoadedModel<eredu_backend_mlx::backend::MlxBackend<'_>>) {
    pressure_with_member_bytes(model, MEMBER_BYTES);
}
fn pressure_with_member_bytes(model: &LoadedModel<eredu_backend_mlx::backend::MlxBackend<'_>>, member_bytes: u64) {
    let report = model.expert_cache_telemetry().unwrap().expect("independent expert banks");
    // Both nonzero routed units execute. Their different bank identities cannot
    // coexist in this one-member device cache, so transitions and later decode
    // require real eviction/acquisition even if routing repeats the same member.
    assert_eq!(report.owned_experts, 4);
    assert_eq!(report.owned_bytes, 4 * member_bytes);
    assert!(report.device_resident_experts <= 1, "{report:?}");
    assert!(report.device_resident_bytes <= member_bytes, "{report:?}");
    assert_eq!(report.peak_device_resident_bytes, member_bytes, "{report:?}");
    eprintln!("PUBLIC_INDEPENDENT_CACHE_PRESSURE:{report:?}");
}
fn run(mode: &str, residency: Residency) -> serde_json::Value {
    run_mode_with_execution_observed(
        mode, fixture(), 0.0, eredu_core::TextSamplingStrategy::Standard,
        residency.execution(), eredu_runtime::CacheResidencyPolicy::Device, Some(&pressure),
    )
}
fn verify(case: &str, environment: &str, residency: Residency) {
    verify_mode_results(case, environment, |mode| run(mode, residency));
}
fn verify_saved(case: &str, environment: &str, residency: Residency) {
    verify_mode_results_with_reference(case, environment, |mode, reference| {
        if mode == "ordinary" { return run(mode, residency); }
        assert!(matches!(mode, "managed" | "controlled"));
        let expected = reference.expect("parent supplies the completed ordinary reference");
        let ids: Vec<u32> = expected["ids"].as_array().unwrap().iter()
            .map(|value| u32::try_from(value.as_u64().unwrap()).unwrap()).collect();
        assert_eq!(ids.len(), 4);
        let root = managed_fixture(fixture());
        let (model, _) = LoadedModel::load_execution_plan(
            &MlxBackendFactory::default(), &root.0, &residency.execution(),
        ).unwrap().into_parts();
        // Two commits include cached decode. The other fresh process snapshots
        // pending prefill; both continue through the same restore/fork driver.
        let committed = if mode == "managed" { 2 } else { 0 };
        let actual = snapshot::check_resume_loaded(
            model, root, committed, &ids, Some(&|| {}), "PUBLIC_EXPERT_CACHE_SAVED",
        );
        assert_eq!(actual, expected);
        actual
    });
}

#[test]
#[ignore = "requires Metal and complete original independent-cache qualification"]
fn native_managed_independent_cache_evicts_and_decodes_in_all_drivers() {
    verify("managed_plain::expert_cache::native_managed_independent_cache_evicts_and_decodes_in_all_drivers",
        "EREDU_PUBLIC_INDEPENDENT_CACHE_MODE", Residency::Resident);
}
#[test]
#[ignore = "run after original independent-cache generation passes"]
fn native_managed_independent_cache_saved_pending_and_decode_restore_fork_release() {
    verify_saved("managed_plain::expert_cache::native_managed_independent_cache_saved_pending_and_decode_restore_fork_release",
        "EREDU_PUBLIC_INDEPENDENT_CACHE_SAVED_MODE", Residency::Resident);
}
#[test]
#[ignore = "run after resident original independent-cache generation passes"]
fn native_managed_host_independent_cache_evicts_and_decodes_in_all_drivers() {
    verify("managed_plain::expert_cache::native_managed_host_independent_cache_evicts_and_decodes_in_all_drivers",
        "EREDU_PUBLIC_HOST_INDEPENDENT_CACHE_MODE", Residency::Host);
}
#[test]
#[ignore = "run after resident original independent-cache generation passes"]
fn native_managed_foreground_disk_independent_cache_evicts_and_decodes_in_all_drivers() {
    verify("managed_plain::expert_cache::native_managed_foreground_disk_independent_cache_evicts_and_decodes_in_all_drivers",
        "EREDU_PUBLIC_DISK_INDEPENDENT_CACHE_MODE", Residency::ForegroundDisk);
}
#[test]
#[ignore = "run after Host original independent-cache generation and resident saved lifecycle pass"]
fn native_managed_host_independent_cache_saved_restore_fork_release() {
    verify_saved("managed_plain::expert_cache::native_managed_host_independent_cache_saved_restore_fork_release",
        "EREDU_PUBLIC_HOST_INDEPENDENT_CACHE_SAVED_MODE", Residency::Host);
}
#[test]
#[ignore = "run after Disk original independent-cache generation and resident saved lifecycle pass"]
fn native_managed_foreground_disk_independent_cache_saved_restore_fork_release() {
    verify_saved("managed_plain::expert_cache::native_managed_foreground_disk_independent_cache_saved_restore_fork_release",
        "EREDU_PUBLIC_DISK_INDEPENDENT_CACHE_SAVED_MODE", Residency::ForegroundDisk);
}

#[path = "expert_cache/transformed.rs"]
mod transformed;
