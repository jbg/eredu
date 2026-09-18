//! Exact CPU target / GPU assistant through the existing shared public driver.
//! Generation and observation share the retained placement and source selection.
use super::*;
const CASE: &str = "managed_plain::speculative::cpu_target::native_original_cpu_target_gpu_assistant_matches_ordinary_and_controlled";
const MODE: &str = "EREDU_PUBLIC_CPU_TARGET_MODE";
const RESULT: &str = "PUBLIC_CPU_TARGET_RESULT:";
fn target_device() -> eredu_core::DevicePlan {
    eredu_core::DevicePlan::new("mlx", "cpu:0").unwrap()
}
fn draft_placement() -> DraftPlacementPlan {
    DraftPlacementPlan::Device {
        device: eredu_core::DevicePlan::new("mlx", "metal:0").unwrap(),
    }
}
#[test]
#[ignore = "requires complete retained CPU target sources and an accessible Metal assistant"]
fn native_original_cpu_target_gpu_assistant_matches_ordinary_and_controlled() {
    if let Ok(mode) = std::env::var(MODE) {
        let (target, draft) = super::super::super::speculative::artifacts();
        let output = run_artifacts_on(
            &mode,
            target,
            draft,
            target_device(),
            draft_placement(),
            cpu_assistant::cpu_settings(),
        );
        println!("\n{RESULT}{output}");
        return;
    }
    let mut expected = None;
    for mode in ["ordinary", "managed", "controlled"] {
        let result = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", CASE, "--ignored", "--nocapture"])
            .env(MODE, mode)
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&result.stdout);
        assert!(
            result.status.success(),
            "{mode}: {stdout}\n{}",
            String::from_utf8_lossy(&result.stderr)
        );
        let output: serde_json::Value = serde_json::from_str(
            stdout
                .lines()
                .find_map(|line| line.strip_prefix(RESULT))
                .expect("completed public CPU target marker"),
        )
        .unwrap();
        assert_eq!(output["ids"].as_array().unwrap().len(), 4);
        if let Some(expected) = &expected {
            assert_eq!(&output, expected, "{mode}");
        } else {
            expected = Some(output);
        }
    }
}
#[test]
#[ignore = "requires complete retained CPU target copy sources and an accessible Metal assistant"]
fn native_original_cpu_target_gpu_assistant_restores_forks_and_preserves_limits() {
    const CASE: &str = "managed_plain::speculative::cpu_target::native_original_cpu_target_gpu_assistant_restores_forks_and_preserves_limits";
    const MODE: &str = "EREDU_PUBLIC_CPU_TARGET_REPLAY_MODE";
    const RESULT: &str = "PUBLIC_CPU_TARGET_REPLAY_RESULT:";
    if let Ok(mode) = std::env::var(MODE) {
        let (target, draft) = super::super::super::speculative::artifacts();
        let output = run_continuation_artifacts_on(
            &mode,
            target,
            draft,
            target_device(),
            draft_placement(),
            cpu_assistant::cpu_settings(),
        );
        println!("\n{RESULT}{output}");
        return;
    }
    // Same snapshot/restore/fork/exchange, mutation isolation and nonrefund
    // checks as the validated opposite placement; only the retained plan changes.
    compare_continuation_modes(CASE, MODE, RESULT);
}

#[test]
#[ignore = "requires an accessible Metal assistant and retained CPU capture sources"]
fn native_original_cpu_target_selected_scores_capture_matches_ordinary_and_controlled() {
    capture::compare_cpu_target(
        capture::CaptureKind::Readouts,
        "managed_plain::speculative::cpu_target::native_original_cpu_target_selected_scores_capture_matches_ordinary_and_controlled",
    );
}

#[test]
#[ignore = "requires an accessible Metal assistant and retained CPU partial intervention sources"]
fn native_original_cpu_target_partial_scale_capture_matches_ordinary_and_controlled() {
    capture::compare_cpu_target(
        capture::CaptureKind::PartialScale,
        "managed_plain::speculative::cpu_target::native_original_cpu_target_partial_scale_capture_matches_ordinary_and_controlled",
    );
}

#[path = "cpu_target/half.rs"]
mod half;
