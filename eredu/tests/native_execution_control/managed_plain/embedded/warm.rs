//! A fresh prediction lane starts from a real populated target, not an imported lane snapshot.
use super::*;

const CASE: &str =
    "managed_plain::embedded::warm::native_original_qwen_embedded_continues_populated_target";
const MODE: &str = "EREDU_PUBLIC_EMBEDDED_WARM_TARGET_MODE";
const RESULT: &str = "PUBLIC_EMBEDDED_WARM_TARGET_RESULT:";
const PREFIX: &str = "abc";
const SUFFIX: &str = "defgh";

fn run(mode: &str) -> serde_json::Value {
    let fixture = managed_fixture(super::super::super::speculative::captured_prefill::source(
        "qwen",
    ));
    let execution =
        ExecutionPlan::fully_resident(eredu_core::DevicePlan::new("mlx", "metal:0").unwrap())
            .with_required_session_capabilities(SessionCapabilities::new(true, true, true))
            .with_drafting(DraftingPlan::Embedded {
                max_draft_tokens: 2,
                lookahead: false,
                adaptive_lookahead: false,
            });
    let mut loaded =
        LoadedModel::load_execution_plan(&MlxBackendFactory::default(), &fixture.0, &execution)
            .unwrap_or_else(report_failure);
    let options = loaded.speculative_generation_options().unwrap().unwrap();
    let (model, drafting) = loaded.parts_mut();
    assert_eq!(model.encode(PREFIX, false).unwrap(), [0, 1, 2]);
    assert_eq!(model.encode(SUFFIX, false).unwrap(), [3, 4, 5, 6, 7]);

    let mut warm_settings = settings(0.0);
    warm_settings.overrides.max_new_tokens = Some(1);
    if matches!(mode, "full" | "ordinary") {
        let warm_ids = if mode == "ordinary" {
            let config = model
                .resolve_generation_config(warm_settings.overrides)
                .unwrap();
            let ids: Vec<u32> = model
                .generate_tokens(
                    model.encode(PREFIX, false).unwrap(),
                    TextGenerationConfig::new(config).with_seed(warm_settings.seed),
                )
                .unwrap_or_else(report_failure)
                .map(|token| token.unwrap_or_else(report_failure).token_id().unwrap())
                .collect();
            assert_eq!(ids.len(), 1);
            model.synchronize().unwrap_or_else(report_failure);
            Some(ids)
        } else {
            None
        };
        // One final sampled token is pending, not cached. Discarding that token
        // deliberately leaves only PREFIX in state; SUFFIX is the next input.
        // No reset occurs between these two ordinary public requests.
        let prompt = if mode == "full" {
            format!("{PREFIX}{SUFFIX}")
        } else {
            SUFFIX.to_owned()
        };
        let settings = settings(0.0);
        let config = model.resolve_generation_config(settings.overrides).unwrap();
        let ids: Vec<u32> = model
            .generate_tokens(
                model.encode(&prompt, false).unwrap(),
                TextGenerationConfig::new(config).with_seed(settings.seed),
            )
            .unwrap_or_else(report_failure)
            .map(|token| token.unwrap_or_else(report_failure).token_id().unwrap())
            .collect();
        assert_eq!(ids.len(), 4);
        model.synchronize().unwrap_or_else(report_failure);
        let text = model.decode(&ids, true).unwrap();
        drop(loaded);
        return serde_json::json!({"ids":ids,"text":text,"warm_ids":warm_ids});
    }

    assert!(matches!(mode, "managed" | "controlled"));
    let source = model
        .compile_managed_plain_text_source(
            std::fs::File::open(fixture.0.join("tokenizer.json")).unwrap(),
        )
        .unwrap_or_else(report_failure);
    let warm = model
        .generate_managed_plain_text(
            &source,
            ManagedPlainTextRequest::new(PREFIX, warm_settings),
            &GenerationCancellationToken::new(),
            &mut |_| {},
        )
        .unwrap_or_else(|failure| {
            eprintln!("PUBLIC_EMBEDDED_WARM_TARGET_FAILURE_STAGE: managed warm-up");
            report_failure(failure)
        })
        .expect("uncancelled warm-up");
    assert_eq!(warm.finish_reason, FinishReason::MaxTokens);
    let warm_ids = warm.token_ids.as_ref().to_vec();
    assert_eq!(warm_ids.len(), 1);
    // The target cache must retain its own original copy/publication custody.
    // The warm output does not stay alive to lend authority to the new request.
    drop(warm);
    model.synchronize().unwrap_or_else(report_failure);

    let mut visible = String::new();
    let mut committed = Vec::new();
    let on_event = |event| {
        if let SemanticEvent::TextDelta(text) = event {
            visible.push_str(text.as_str());
        }
    };
    let request = ManagedPlainTextSpeculativeRequest {
        text: ManagedPlainTextRequest::new(SUFFIX, settings(0.0)),
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
                    committed.extend(step.committed_token_ids);
                }
                assert_eq!(committed, session.token_ids());
                Ok(())
            },
        )
    } else {
        model.generate_managed_plain_text_speculative(&source, request)
    }
    .unwrap_or_else(|failure| {
        eprintln!("PUBLIC_EMBEDDED_WARM_TARGET_FAILURE_STAGE: populated-target speculation");
        report_failure(failure)
    });
    assert_eq!(output.token_ids().len(), 4);
    assert!(output.stats().rounds() >= 1);
    assert!(output.timing().time_to_first_token().is_some());
    if mode == "controlled" {
        assert_eq!(committed, output.token_ids());
    }
    model.synchronize().unwrap_or_else(report_failure);
    let address = output.token_ids().as_ptr();
    drop((source, loaded));
    assert_eq!(output.token_ids().as_ptr(), address);
    serde_json::json!({"ids":output.token_ids(),"text":visible,"warm_ids":warm_ids})
}

#[test]
#[ignore = "requires an accessible Metal device"]
fn native_original_qwen_embedded_continues_populated_target() {
    if let Ok(mode) = std::env::var(MODE) {
        println!("\n{RESULT}{}", run(&mode));
        return;
    }
    let mut expected = None;
    let mut expected_warm = None;
    for mode in ["full", "ordinary", "managed", "controlled"] {
        let result = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", CASE, "--ignored", "--nocapture"])
            .env(MODE, mode)
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&result.stdout);
        assert!(
            result.status.success(),
            "warm target/{mode}: {stdout}\n{}",
            String::from_utf8_lossy(&result.stderr)
        );
        let actual: serde_json::Value = serde_json::from_str(
            stdout
                .lines()
                .find_map(|line| line.strip_prefix(RESULT))
                .unwrap_or_else(|| panic!("missing positive execution marker: {mode}: {stdout}")),
        )
        .unwrap();
        let generated = serde_json::json!({"ids":actual["ids"],"text":actual["text"]});
        if let Some(expected) = &expected {
            assert_eq!(&generated, expected, "warm target/{mode}");
        } else {
            expected = Some(generated);
        }
        if mode != "full" {
            let warm = actual["warm_ids"].as_array().expect("warm-up token ids");
            assert_eq!(warm.len(), 1);
            if let Some(expected) = &expected_warm {
                assert_eq!(warm, expected, "warm-up/{mode}");
            } else {
                expected_warm = Some(warm.clone());
            }
        }
    }
}
