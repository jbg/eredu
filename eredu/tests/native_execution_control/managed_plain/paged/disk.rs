//! Public Disk paging uses the existing runners and actual live file owners.
use super::*;
use eredu_runtime::{CachePoolLimits, CacheResidencyPool};

const DEVICE: u64 = 768;
// Each actual F32 three-token KV page has two 16 KiB Host allocations.
const HOST: u64 = 64 << 10;
const PHYSICAL_DEVICE: u64 = 1 << 20;
const PHYSICAL_HOST: u64 = 16 << 20;
const PHYSICAL_TRANSFER: u64 = 16 << 20;
const DISK: u64 = 4 << 20;

fn source(directory: &Path) -> (CacheResidencyPool, CacheResidencyPolicy) {
    let pool = CacheResidencyPool::new(
        CachePoolLimits::new(PHYSICAL_DEVICE, PHYSICAL_HOST, PHYSICAL_TRANSFER, DISK).unwrap(),
    );
    let options = PagedCacheOptions::new(3, DEVICE, HOST, 1)
        .unwrap()
        .with_full_attention(true)
        .with_live_disk(directory, DISK, 2)
        .unwrap()
        .with_pool(pool.clone())
        .unwrap();
    (pool, CacheResidencyPolicy::Paged(options))
}

#[test]
#[ignore = "requires an accessible Metal device"]
fn native_managed_disk_paged_matches_ordinary_and_controlled_with_real_io() {
    const CASE: &str = "managed_plain::paged::disk::native_managed_disk_paged_matches_ordinary_and_controlled_with_real_io";
    const ENV: &str = "EREDU_PUBLIC_MANAGED_DISK_PAGED_MODE";
    if let Ok(mode) = std::env::var(ENV) {
        let directory = tempfile::tempdir().unwrap();
        let (pool, policy) = source(directory.path());
        let inspect = |model: &LoadedModel<eredu_backend_mlx::backend::MlxBackend<'_>>| {
            let cache = settled_report(model);
            let physical = pool.report().unwrap();
            assert!(
                cache.disk_demotions > 0,
                "no completed Disk publication: {cache:?}"
            );
            assert!(
                cache.disk_promotions > 0,
                "no actual Disk reload: {cache:?}"
            );
            // Device and Host rows may retain immutable files too. A zero
            // Disk-only row count does not mean the backing was discarded.
            assert!(
                cache.peak_disk_bytes > 0,
                "no actual Disk residency: {cache:?}"
            );
            assert!(cache.transfer_bytes > 0);
            assert!(cache.current_device_bytes <= DEVICE, "{cache:?}");
            assert!(cache.current_host_bytes <= HOST, "{cache:?}");
            assert!(cache.current_disk_bytes <= DISK, "{cache:?}");
            assert!(
                physical.peak_device_bytes <= PHYSICAL_DEVICE,
                "{physical:?}"
            );
            assert!(physical.peak_host_bytes <= PHYSICAL_HOST, "{physical:?}");
            assert!(
                physical.peak_transfer_in_flight_bytes <= PHYSICAL_TRANSFER,
                "{physical:?}"
            );
            assert!(
                std::fs::read_dir(directory.path())
                    .unwrap()
                    .next()
                    .is_some()
            );
            println!(
                "\nPUBLIC_DISK_PAGING canonical_device={} canonical_host={} physical_device_peak={} disk_demotions={} disk_promotions={}",
                cache.current_device_bytes,
                cache.current_host_bytes,
                physical.peak_device_bytes,
                cache.disk_demotions,
                cache.disk_promotions
            );
        };
        // The shared managed/control runner checks pre-work cancellation and
        // one-byte inference refusal, and retains output beyond the model.
        let result = run_mode_with_state_residency_observed(
            &mode,
            fixture(false),
            0.0,
            eredu_core::TextSamplingStrategy::Standard,
            None,
            policy,
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
                .any(|line| line.starts_with("PUBLIC_DISK_PAGING ")),
            "missing actual I/O report: {mode}: {stdout}"
        );
        let actual: serde_json::Value = serde_json::from_str(
            stdout
                .lines()
                .find_map(|line| line.strip_prefix(RESULT_PREFIX))
                .unwrap_or_else(|| panic!("missing successful Disk result: {mode}: {stdout}")),
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
fn native_managed_disk_paged_snapshot_restore_fork_and_reset_preserve_files() {
    let directory = tempfile::tempdir().unwrap();
    let (pool, policy) = source(directory.path());
    let inspect = || {
        let actual = pool.report().unwrap();
        assert!(
            actual.current_disk_bytes > 0,
            "saved source has no live paid Disk file: {actual:?}"
        );
        assert!(actual.current_disk_bytes <= DISK, "{actual:?}");
        assert!(actual.peak_device_bytes <= PHYSICAL_DEVICE, "{actual:?}");
        assert!(actual.peak_host_bytes <= PHYSICAL_HOST, "{actual:?}");
        assert!(
            actual.peak_transfer_in_flight_bytes <= PHYSICAL_TRANSFER,
            "{actual:?}"
        );
        assert!(
            std::fs::read_dir(directory.path())
                .unwrap()
                .next()
                .is_some()
        );
        println!(
            "\nPUBLIC_DISK_SAVED_SOURCE physical_disk={} physical_device_peak={}",
            actual.current_disk_bytes, actual.peak_device_bytes
        );
    };
    // Seven cached positions retain sealed file-backed pages and a mutable
    // tail. The shared driver checks original/restored/forked continuation,
    // refusal/cancellation, cumulative spending, and final escaped ownership.
    super::super::snapshot::check_resume_after_commits(3, false, policy.clone(), Some(&inspect));
    assert_retired(&pool, directory.path());
    super::super::lifecycle::reset_fixture_with_state(fixture(false), policy);
    eredu_backend_mlx::backend::nn::shared::MlxNeuralBackend::reclaim_retired_resources();
    assert_retired(&pool, directory.path());
    println!("\nPUBLIC_DISK_SAVED_LIFECYCLE_COMPLETE");
}

fn assert_retired(pool: &CacheResidencyPool, directory: &Path) {
    let actual = pool.report().unwrap();
    assert_eq!(
        actual.current_disk_bytes, 0,
        "file owner survived final state: {actual:?}"
    );
    assert_eq!(actual.current_host_bytes, 0, "{actual:?}");
    assert_eq!(actual.current_device_bytes, 0, "{actual:?}");
    assert_eq!(actual.current_transfer_in_flight_bytes, 0, "{actual:?}");
    assert!(
        std::fs::read_dir(directory).unwrap().next().is_none(),
        "published file outlived its final owner"
    );
}

// Poll the existing ordinary report/reap worker after model synchronization.
// It settles real completed writes instead of treating a scalar counter as a
// completion witness or changing canonical cache state in the fixture.
fn settled_report(
    model: &LoadedModel<eredu_backend_mlx::backend::MlxBackend<'_>>,
) -> eredu_runtime::CacheResidencyReport {
    model.synchronize().unwrap();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    loop {
        let report = model
            .cache_residency_telemetry()
            .unwrap()
            .expect("actual paged manager");
        if report.in_flight_write_blocks == 0 && report.in_flight_host_demotion_blocks == 0 {
            return report;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "cache workers did not settle: {report:?}"
        );
        std::thread::yield_now();
    }
}
