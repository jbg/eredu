//! End-to-end original Embedded parity using the existing nonzero family fixtures.
use super::*;
use eredu_core::DraftingPlan;

const MODE: &str = "EREDU_PUBLIC_MANAGED_EMBEDDED_MODE";
const RESULT: &str = "PUBLIC_MANAGED_EMBEDDED_RESULT:";

fn run(kind: &str, mode: &str) -> serde_json::Value {
    let fixture = super::super::speculative::captured_prefill::source(kind);
    if mode == "ordinary" {
        return run_mode("ordinary", fixture, 0.0);
    }
    let target = managed_fixture(fixture);
    let execution =
        ExecutionPlan::fully_resident(eredu_core::DevicePlan::new("mlx", "metal:0").unwrap())
            .with_required_session_capabilities(SessionCapabilities::new(true, true, true))
            .with_drafting(DraftingPlan::Embedded {
                max_draft_tokens: 2,
                lookahead: false,
                adaptive_lookahead: false,
            });
    let mut loaded =
        LoadedModel::load_execution_plan(&MlxBackendFactory::default(), &target.0, &execution)
            .unwrap();
    let options = loaded.speculative_generation_options().unwrap().unwrap();
    let (model, drafting) = loaded.parts_mut();
    let source = model
        .compile_managed_plain_text_source(
            std::fs::File::open(target.0.join("tokenizer.json")).unwrap(),
        )
        .unwrap_or_else(report_failure);
    let mut visible = String::new();
    let mut observed = Vec::new();
    let on_event = |event| {
        if let SemanticEvent::TextDelta(text) = event {
            visible.push_str(text.as_str());
        }
    };
    let request = ManagedPlainTextSpeculativeRequest {
        text: ManagedPlainTextRequest::new(PROMPT, settings(0.0)),
        drafting: drafting.as_speculative_draft().unwrap(),
        options,
        cancellation: GenerationCancellationToken::new(),
        on_event,
    };
    let output = if mode == "controlled" {
        model.with_controlled_managed_plain_text_speculative(
            &source,
            request,
            ControlledSpeculativeOptions::default(),
            |session| {
                assert!(session.token_ids().is_empty());
                while let Some(step) = session.step()? {
                    observed.extend(step.committed_token_ids);
                }
                assert_eq!(observed, session.token_ids());
                Ok(())
            },
        )
    } else {
        assert_eq!(mode, "managed");
        model.generate_managed_plain_text_speculative(&source, request)
    }
    .unwrap_or_else(report_failure);
    assert_eq!(output.token_ids().len(), 4);
    assert!(output.stats().rounds() >= 1);
    assert!(output.timing().time_to_first_token().is_some());
    if mode == "controlled" {
        assert_eq!(observed, output.token_ids());
    }
    let address = output.token_ids().as_ptr();
    drop((source, loaded));
    assert_eq!(output.token_ids().as_ptr(), address);
    serde_json::json!({"ids":output.token_ids(),"text":visible})
}
fn parity(kind: &str, case: &str) {
    parity_with(kind, case, &["ordinary", "managed", "controlled"], run)
}
fn parity_with(kind: &str, case: &str, modes: &[&str], run: fn(&str, &str) -> serde_json::Value) {
    if let Ok(mode) = std::env::var(MODE) {
        println!("\n{RESULT}{}", run(kind, &mode));
        return;
    }
    let mut expected = None;
    for mode in modes {
        let result = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", case, "--ignored", "--nocapture"])
            .env(MODE, mode)
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&result.stdout);
        assert!(
            result.status.success(),
            "{kind}/{mode}: {stdout}\n{}",
            String::from_utf8_lossy(&result.stderr)
        );
        let actual: serde_json::Value = serde_json::from_str(
            stdout
                .lines()
                .find_map(|line| line.strip_prefix(RESULT))
                .unwrap_or_else(|| {
                    panic!("missing positive execution marker: {kind}/{mode}: {stdout}")
                }),
        )
        .unwrap();
        if let Some(expected) = &expected {
            assert_eq!(&actual, expected, "{kind}/{mode}");
        } else {
            expected = Some(actual);
        }
    }
}
macro_rules! family {
    ($test:ident,$kind:literal) => {
        #[test]
        #[ignore = "requires an accessible Metal device"]
        fn $test() {
            parity(
                $kind,
                concat!("managed_plain::embedded::", stringify!($test)),
            );
        }
    };
}
family!(
    native_original_qwen_embedded_matches_plain_and_controlled,
    "qwen"
);
family!(
    native_original_v3_embedded_matches_plain_and_controlled,
    "v3"
);
family!(
    native_original_v4_embedded_matches_plain_and_controlled,
    "v4"
);
family!(
    native_original_dspark_embedded_matches_plain_and_controlled,
    "dspark"
);
family!(
    native_original_inkling_embedded_matches_plain_and_controlled,
    "inkling"
);
family!(
    native_original_nemotron_embedded_matches_plain_and_controlled,
    "nemotron"
);

#[path = "embedded/warm.rs"]
mod warm;

#[path = "embedded/activations.rs"]
mod activations;

#[path = "embedded/snapshots.rs"]
mod snapshots;

#[path = "embedded/batch.rs"]
mod batch;

#[path = "embedded/semantic.rs"]
mod semantic;
