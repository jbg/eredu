//! Distinct external request sources through the ordinary fair batch scheduler.
use super::*;
const CASE: &str = "managed_plain::speculative::batch::cpu::native_original_cpu_external_two_lane_batch_preserves_sources_and_order";
const MODE: &str = "EREDU_PUBLIC_CPU_EXTERNAL_BATCH_MODE";
const SIDE: &str = "EREDU_PUBLIC_CPU_EXTERNAL_BATCH_SIDE";
const RESULT: &str = "PUBLIC_CPU_EXTERNAL_BATCH_RESULT:";

fn settings(index: usize, original: bool) -> PreparedChatGenerationSettings {
    let mut value = super::super::cpu_assistant::cpu_settings();
    // Reuse the selected plain probability/categorical workers with independent
    // lane keys; optional top-k/top-p and penalty policies stay disabled.
    value.overrides.temperature = Some(if index == 0 { 0.65 } else { 0.9 });
    value.overrides.max_new_tokens = Some(OUTPUTS[index]);
    value.seed = SEEDS[index];
    if !original { value.inference.managed_memory_capacity_bytes = None; }
    value
}
fn run(mode: &str, cpu_side: &str) -> serde_json::Value {
    let (target, draft) = super::super::super::super::speculative::artifacts();
    let (target_device, draft_device) = match cpu_side {
        "target" => ("cpu:0", "metal:0"),
        "draft" => ("metal:0", "cpu:0"),
        _ => panic!("unexpected CPU side"),
    };
    run_artifacts(mode, target, draft,
        eredu_core::DevicePlan::new("mlx", target_device).unwrap(),
        DraftPlacementPlan::Device {
            device: eredu_core::DevicePlan::new("mlx", draft_device).unwrap(),
        }, settings)
}
#[test]
#[ignore = "requires an accessible Metal device and retained external CPU lane sources"]
fn native_original_cpu_external_two_lane_batch_preserves_sources_and_order() {
    if let Ok(mode) = std::env::var(MODE) {
        let side = std::env::var(SIDE).expect("selected CPU side");
        println!("\n{RESULT}{}", run(&mode, &side));
        return;
    }
    for side in ["target", "draft"] {
        let mut expected = None;
        for mode in ["ordinary", "managed"] {
            let result = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", CASE, "--ignored", "--nocapture"])
                .env(MODE, mode).env(SIDE, side).output().unwrap();
            let stdout = String::from_utf8_lossy(&result.stdout);
            assert!(result.status.success(), "{side} {mode}: {stdout}\n{}",
                String::from_utf8_lossy(&result.stderr));
            let actual: serde_json::Value = serde_json::from_str(stdout.lines()
                .find_map(|line| line.strip_prefix(RESULT))
                .unwrap_or_else(|| panic!("missing batch result: {side} {mode}: {stdout}"))).unwrap();
            assert_eq!(actual.as_array().unwrap().len(), 2);
            if let Some(expected) = &expected {
                assert_eq!(&actual, expected, "{side}: lane source, random key and order parity");
            } else { expected = Some(actual); }
        }
    }
}
