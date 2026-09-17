//! Real external assistant with ordered vocabulary and nonzero centroid readout.
use super::*;
const CASE: &str = "managed_plain::speculative::assistant::native_original_ordered_assistant_matches_ordinary_and_controlled";
const MODE: &str = "EREDU_PUBLIC_ORDERED_ASSISTANT_MODE";
const RESULT: &str = "PUBLIC_ORDERED_ASSISTANT_RESULT:";
#[test]
#[ignore = "requires an accessible Metal device"]
fn native_original_ordered_assistant_matches_ordinary_and_controlled() {
    compare_modes(CASE,MODE,RESULT,false)
}
#[test]
#[ignore = "requires an accessible Metal device"]
fn native_original_ordered_assistant_same_device_split_matches_ordinary_and_controlled() {
    compare_modes(
        "managed_plain::speculative::assistant::native_original_ordered_assistant_same_device_split_matches_ordinary_and_controlled",
        "EREDU_PUBLIC_ORDERED_ASSISTANT_SPLIT_MODE","PUBLIC_ORDERED_ASSISTANT_SPLIT_RESULT:",true)
}
fn compare_modes(case:&str,mode_env:&str,result_marker:&str,split:bool) {
    if let Ok(mode) = std::env::var(mode_env) {
        let (target, draft) = super::super::super::speculative::ordered_artifacts();
        let result=if split {
            run_artifacts_at(&mode,target,draft,DraftPlacementPlan::Device {
                device:eredu_core::DevicePlan::new("mlx","metal:0").unwrap()
            },0.7)
        }else{run_artifacts(&mode,target,draft)};
        println!("\n{result_marker}{result}");
        return;
    }
    let mut expected = None;
    for mode in ["ordinary", "managed", "controlled"] {
        let result = std::process::Command::new(std::env::current_exe().unwrap())
             .args(["--exact", case, "--ignored", "--nocapture"])
            .env(mode_env, mode)
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
                .find_map(|line| line.strip_prefix(result_marker))
                .expect("positive assistant execution marker"),
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
fn native_original_ordered_assistant_split_restores_forks_and_preserves_limits() {
    const CASE: &str = "managed_plain::speculative::assistant::native_original_ordered_assistant_split_restores_forks_and_preserves_limits";
    const MODE: &str = "EREDU_PUBLIC_ORDERED_ASSISTANT_REPLAY_MODE";
    const RESULT: &str = "PUBLIC_ORDERED_ASSISTANT_REPLAY_RESULT:";
    if let Ok(mode) = std::env::var(MODE) {
        let (target, draft) = super::super::super::speculative::ordered_artifacts();
        let result = run_continuation_artifacts(&mode, target, draft, DraftPlacementPlan::Device {
            device: eredu_core::DevicePlan::new("mlx", "metal:0").unwrap(),
        });
        println!("\n{RESULT}{result}");
        return;
    }
    compare_continuation_modes(CASE, MODE, RESULT);
}
