//! Actual GPU target / selected CPU assistant through all three shared modes.
use super::*;
const CASE:&str="managed_plain::speculative::cpu_assistant::native_original_cpu_assistant_matches_ordinary_and_controlled";
const MODE:&str="EREDU_PUBLIC_CPU_ASSISTANT_MODE";
const RESULT:&str="PUBLIC_CPU_ASSISTANT_RESULT:";
#[test]
#[ignore="requires an accessible Metal target and qualified CPU assistant"]
fn native_original_cpu_assistant_matches_ordinary_and_controlled() {
    if let Ok(mode)=std::env::var(MODE) {
        let (target,draft)=super::super::super::speculative::artifacts();
        let generation=cpu_settings();
        let value=run_artifacts_configured(&mode,target,draft,DraftPlacementPlan::Device {
            device:eredu_core::DevicePlan::new("mlx","cpu:0").unwrap(),
        },generation);
        println!("\n{RESULT}{value}");return;
    }
    let mut expected=None;
    for mode in ["ordinary","managed","controlled"] {
        let result=std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact",CASE,"--ignored","--nocapture"]).env(MODE,mode).output().unwrap();
        let stdout=String::from_utf8_lossy(&result.stdout);
        assert!(result.status.success(),"{mode}: {stdout}\n{}",String::from_utf8_lossy(&result.stderr));
        let value:serde_json::Value=serde_json::from_str(stdout.lines()
            .find_map(|line|line.strip_prefix(RESULT)).expect("completed public CPU assistant marker")).unwrap();
        assert_eq!(value["ids"].as_array().unwrap().len(),4);
        if let Some(expected)=&expected {assert_eq!(&value,expected,"{mode}");}else{expected=Some(value);}
    }
}

pub(super) fn cpu_settings()->PreparedChatGenerationSettings {
    let mut generation=settings(0.0);
    // Preserve the same selected greedy mechanism in uninterrupted and every
    // continuation. Other policies require their own actual source facts.
    generation.overrides.top_k=Some(0);
    generation.overrides.top_p=Some(1.0);
    generation.overrides.min_p=Some(0.0);
    generation.overrides.repetition_penalty=Some(1.0);
    generation.overrides.frequency_penalty=Some(0.0);
    generation.overrides.presence_penalty=Some(0.0);
    generation
}

#[test]
#[ignore="requires an accessible Metal target and qualified CPU assistant"]
fn native_original_cpu_assistant_restores_forks_and_preserves_limits() {
    const CASE:&str="managed_plain::speculative::cpu_assistant::native_original_cpu_assistant_restores_forks_and_preserves_limits";
    const MODE:&str="EREDU_PUBLIC_CPU_ASSISTANT_REPLAY_MODE";
    const RESULT:&str="PUBLIC_CPU_ASSISTANT_REPLAY_RESULT:";
    if let Ok(mode)=std::env::var(MODE) {
        let (target,draft)=super::super::super::speculative::artifacts();
        let result=run_continuation_artifacts_configured(&mode,target,draft,DraftPlacementPlan::Device {
            device:eredu_core::DevicePlan::new("mlx","cpu:0").unwrap(),
        },cpu_settings());
        println!("\n{RESULT}{result}");return;
    }
    // The existing driver compares ordinary/managed output and exercises saved
    // restore, inactive fork/exchange, forced-token isolation, release, and the
    // calibrated cumulative copy and trace refusals. No second engine exists.
    compare_continuation_modes(CASE,MODE,RESULT);
}
