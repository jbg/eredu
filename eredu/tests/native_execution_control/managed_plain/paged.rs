//! Actual public paging through the shared modes and control-state drivers.
use super::*;
use eredu_runtime::{CacheResidencyPolicy, PagedCacheOptions};

fn policy() -> CacheResidencyPolicy {
    // Per-cache finite limits. Three-token pages are sealed within and across
    // actual 2/2/1 prefill chunks, with a live tail and several cached decodes.
    CacheResidencyPolicy::Paged(
        PagedCacheOptions::new(3, 8 << 20, 0, 1)
            .unwrap()
            .with_full_attention(true),
    )
}

#[test]
#[ignore = "requires an accessible Metal device"]
fn native_managed_paged_matches_ordinary_and_controlled_generation() {
    const CASE: &str =
        "managed_plain::paged::native_managed_paged_matches_ordinary_and_controlled_generation";
    const ENV: &str = "EREDU_PUBLIC_MANAGED_PAGED_MODE";
    if let Ok(mode) = std::env::var(ENV) {
        let result = run_mode_with_state_residency(
            if mode == "resident" {
                "ordinary"
            } else {
                &mode
            },
            fixture(false),
            0.0,
            eredu_core::TextSamplingStrategy::Standard,
            None,
            if mode == "resident" {
                CacheResidencyPolicy::Device
            } else {
                policy()
            },
        );
        println!("\n{RESULT_PREFIX}{result}");
        return;
    }
    let mut expected = None;
    for mode in ["resident", "ordinary", "managed", "controlled"] {
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
        let actual: serde_json::Value = serde_json::from_str(
            stdout
                .lines()
                .find_map(|line| line.strip_prefix(RESULT_PREFIX))
                .unwrap_or_else(|| {
                    panic!("missing positive paged execution marker: {mode}: {stdout}")
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
fn native_managed_paged_snapshot_restore_fork_and_reset_preserve_custody() {
    // These are the existing native control drivers: completed snapshot with
    // sealed pages plus live tail, fresh restore/fork, one-byte refusal,
    // cumulative copy spending and escaped saved/output owners. Reset is a
    // separately admitted fresh table/manager, with old output retained.
    super::snapshot::check_resume_with_state(true, false, policy());
    super::lifecycle::reset_fixture_with_state(fixture(false), policy());
}

#[path = "paged/host.rs"]
mod host;

#[path = "paged/disk.rs"]
mod disk;
