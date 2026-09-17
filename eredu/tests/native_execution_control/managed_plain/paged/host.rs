//! Same public runner, with actual Host traffic and separate physical limits.
use super::*;
use eredu_runtime::{CachePoolLimits, CacheResidencyPool};

const CANONICAL_DEVICE: u64 = 768;
const PHYSICAL_DEVICE: u64 = 1 << 20;
const PHYSICAL_HOST: u64 = 1 << 20;
const PHYSICAL_TRANSFER: u64 = 1 << 20;

#[test]
#[ignore = "requires an accessible Metal device"]
fn native_managed_host_paged_matches_ordinary_and_controlled_with_real_transfers() {
    const CASE: &str = "managed_plain::paged::host::native_managed_host_paged_matches_ordinary_and_controlled_with_real_transfers";
    const ENV: &str = "EREDU_PUBLIC_MANAGED_HOST_PAGED_MODE";
    if let Ok(mode) = std::env::var(ENV) {
        let pool = CacheResidencyPool::new(
            CachePoolLimits::new(PHYSICAL_DEVICE, PHYSICAL_HOST, PHYSICAL_TRANSFER, 0).unwrap(),
        );
        // The actual fixture has two layers, two KV heads, width four and F32:
        // each three-token sealed page is 192 bytes. Final eight cached tokens
        // need 1024 canonical bytes, so this 768-byte limit forces real paging.
        // The larger finite pool covers actual roots and replaced backing that
        // remain alive through completion; no mid-role physical refund is assumed.
        let options = PagedCacheOptions::new(3, CANONICAL_DEVICE, PHYSICAL_HOST, 1)
            .unwrap()
            .with_full_attention(true)
            .with_pool(pool.clone())
            .unwrap();
        let inspect = |model: &LoadedModel<eredu_backend_mlx::backend::MlxBackend<'_>>| {
            model.synchronize().unwrap();
            let cache = model
                .cache_residency_telemetry()
                .unwrap()
                .expect("actual paged manager");
            let physical = pool.report().unwrap();
            assert!(
                cache.host_demotions > 0,
                "no actual Device-to-Host publication: {cache:?}"
            );
            assert!(
                cache.host_promotions > 0,
                "no actual Host reload: {cache:?}"
            );
            assert!(cache.transfer_bytes > 0);
            assert!(cache.peak_host_bytes > 0);
            assert!(cache.current_device_bytes <= CANONICAL_DEVICE, "{cache:?}");
            assert!(cache.current_host_bytes <= PHYSICAL_HOST, "{cache:?}");
            assert!(physical.peak_host_bytes > 0);
            assert!(
                physical.peak_device_bytes <= PHYSICAL_DEVICE,
                "{physical:?}"
            );
            assert!(physical.peak_host_bytes <= PHYSICAL_HOST, "{physical:?}");
            assert!(
                physical.peak_transfer_in_flight_bytes <= PHYSICAL_TRANSFER,
                "{physical:?}"
            );
            println!(
                "\nPUBLIC_HOST_PAGING canonical_device={} physical_device_peak={} host_demotions={} host_promotions={}",
                cache.current_device_bytes,
                physical.peak_device_bytes,
                cache.host_demotions,
                cache.host_promotions
            );
        };
        // The shared managed/controlled runner also checks cancellation and an
        // actual one-byte inference admission refusal before the successful run.
        let result = run_mode_with_state_residency_observed(
            &mode,
            fixture(false),
            0.0,
            eredu_core::TextSamplingStrategy::Standard,
            None,
            CacheResidencyPolicy::Paged(options),
            Some(&inspect),
        );
        println!("\n{RESULT_PREFIX}{result}");
        return;
    }
    let mut expected = None;
    for mode in ["ordinary", "managed", "controlled"] {
        let result = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", CASE, "--ignored", "--nocapture"])
            .env(ENV, mode)
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&result.stdout);
        assert!(
            result.status.success(),
            "{mode}: {stdout}\n{}",
            String::from_utf8_lossy(&result.stderr)
        );
        assert!(
            stdout
                .lines()
                .any(|line| line.starts_with("PUBLIC_HOST_PAGING ")),
            "missing actual transfer report: {mode}: {stdout}"
        );
        let actual: serde_json::Value = serde_json::from_str(
            stdout
                .lines()
                .find_map(|line| line.strip_prefix(RESULT_PREFIX))
                .unwrap_or_else(|| {
                    panic!("missing positive Host execution marker: {mode}: {stdout}")
                }),
        )
        .unwrap();
        if let Some(expected) = &expected {
            assert_eq!(&actual, expected, "{mode}");
        } else {
            expected = Some(actual);
        }
    }
}

#[test]
#[ignore = "requires an accessible Metal device"]
fn native_managed_host_paged_snapshot_restore_fork_and_reset_preserve_custody() {
    let pool = CacheResidencyPool::new(
        CachePoolLimits::new(PHYSICAL_DEVICE, PHYSICAL_HOST, PHYSICAL_TRANSFER, 0).unwrap(),
    );
    let state = || {
        CacheResidencyPolicy::Paged(
            PagedCacheOptions::new(3, CANONICAL_DEVICE, PHYSICAL_HOST, 1)
                .unwrap()
                .with_full_attention(true)
                .with_pool(pool.clone())
                .unwrap(),
        )
    };
    let inspect = || {
        let actual = pool.report().unwrap();
        assert!(
            actual.current_host_bytes > 0,
            "saved source has no live Host backing: {actual:?}"
        );
        // This aggregate includes prepaid destinations and completed aliases
        // retained by the source bank. Snapshot admission separately requires
        // idle session authority, releasable token completion and stable manager
        // storage; a zero reservation counter is not that completion witness.
        assert!(
            actual.current_transfer_in_flight_bytes <= PHYSICAL_TRANSFER,
            "retained transfer storage exceeds its finite pool: {actual:?}"
        );
        assert!(actual.peak_device_bytes <= PHYSICAL_DEVICE, "{actual:?}");
        assert!(actual.peak_host_bytes <= PHYSICAL_HOST, "{actual:?}");
        println!(
            "\nPUBLIC_HOST_SAVED_SOURCE current_host={} physical_device_peak={}",
            actual.current_host_bytes, actual.peak_device_bytes
        );
    };
    // A sampled token is pending, so three commitments leave seven cached
    // positions: 7 * 128 = 896 logical bytes against the 768-byte Device cap.
    // This includes sealed Host pages plus a mutable tail, with one future
    // token to compare after both restore and fork.
    super::super::snapshot::check_resume_after_commits(3, false, state(), Some(&inspect));
    let after_resume = pool.report().unwrap();
    assert_eq!(
        after_resume.current_host_bytes, 0,
        "escaped saved owners did not retire: {after_resume:?}"
    );
    assert_eq!(after_resume.current_device_bytes, 0, "{after_resume:?}");
    assert_eq!(
        after_resume.current_transfer_in_flight_bytes, 0,
        "{after_resume:?}"
    );

    // Reuse the existing admitted-reset driver, including one-byte refusal,
    // preserved old output, fresh generation and escaped output ownership.
    super::super::lifecycle::reset_fixture_with_state(fixture(false), state());
    eredu_backend_mlx::backend::nn::shared::MlxNeuralBackend::reclaim_retired_resources();
    let retired = pool.report().unwrap();
    assert_eq!(retired.current_host_bytes, 0, "{retired:?}");
    assert_eq!(retired.current_device_bytes, 0, "{retired:?}");
    assert_eq!(retired.current_transfer_in_flight_bytes, 0, "{retired:?}");
    println!("\nPUBLIC_HOST_SAVED_LIFECYCLE_COMPLETE");
}
