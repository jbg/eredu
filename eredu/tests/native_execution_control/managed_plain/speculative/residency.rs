//! Same selected host/disk residency for both independent models.
use super::*;

const MODE: &str = "EREDU_PUBLIC_SPECULATIVE_LAYERWISE_MODE";
const RESULT: &str = "PUBLIC_SPECULATIVE_LAYERWISE_RESULT:";

#[derive(Clone, Copy, Debug)]
enum Residency {
    Host,
    ForegroundDisk,
}
impl Residency {
    fn plan(self) -> eredu_core::ResidencyPlan {
        match self {
            Self::Host => eredu_core::ResidencyPlan::LayerwiseHost {
                device_layer_window: 1,
                device_budget_bytes: Some(8 << 20),
                host_budget_bytes: Some(8 << 20),
            },
            Self::ForegroundDisk => eredu_core::ResidencyPlan::DenseDiskStream {
                device_budget_bytes: 8 << 20,
                host_budget_bytes: 0,
                host_lookahead: 0,
                background_queue: 0,
            },
        }
    }
}

fn source_fixture() -> Fixture {
    let fixture = fixture(false);
    let path = fixture.0.join("config.json");
    let mut config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    // More nonzero decoder units than either selected device window.
    config["num_hidden_layers"] = 3.into();
    std::fs::write(path, serde_json::to_vec(&config).unwrap()).unwrap();
    let resolved = eredu_architectures::configuration::resolve_model_config(&config).unwrap();
    write_tensor_plan(&fixture.0, resolved.architecture.checkpoint());
    managed_fixture(fixture)
}

fn run_selected(residency: Residency, mode: &str) -> serde_json::Value {
    let target = source_fixture();
    let draft = source_fixture();
    // Neutral independent-draft preparation clones this plan and disables only
    // drafting, preserving the selected residency for the actual draft as well.
    let execution =
        ExecutionPlan::fully_resident(eredu_core::DevicePlan::new("mlx", "metal:0").unwrap())
            .with_residency(residency.plan())
            .with_required_session_capabilities(SessionCapabilities::new(true, true, true))
            .with_drafting(DraftingPlan::External {
                model: draft.0.display().to_string(),
                placement: DraftPlacementPlan::Target,
                max_draft_tokens: 1,
                lookahead: false,
                adaptive_lookahead: false,
            });
    let mut loaded =
        LoadedModel::load_execution_plan(&MlxBackendFactory::default(), &target.0, &execution)
            .unwrap_or_else(report_failure);
    let options = loaded.speculative_generation_options().unwrap().unwrap();
    let (model, drafting) = loaded.parts_mut();
    let mut visible = String::new();
    let on_event = |event| {
        if let SemanticEvent::TextDelta(text) = event {
            visible.push_str(text.as_str());
        }
    };
    let mut settings = settings(0.7);
    assert_eq!(settings.inference.prefill_chunk_positions.unwrap().get(), 2);
    assert_eq!(settings.overrides.max_new_tokens, Some(4));
    if mode == "ordinary" {
        let chat = model
            .source_chat_with_capacity(
                ChatTemplateRequest {
                    messages: vec![serde_json::json!({"role":"user", "content":PROMPT})],
                    add_generation_prompt: false,
                    tool_choice: ToolChoice::None,
                    ..Default::default()
                },
                8 * 1024 * 1024 * 1024,
            )
            .unwrap();
        let ids = model.encode(PROMPT, false).unwrap();
        assert_eq!(ids, [0, 1, 2, 3, 4]); // chunk2 has a one-token final span.
        let output = model
            .generate_prepared_chat_speculative(PreparedChatSpeculativeRequest {
                chat: &chat,
                input: eredu::api::PreparedChatPrompt::TokenIds(&ids),
                output_mode: eredu::api::PreparedChatOutputMode::Text,
                skip_special_tokens: true,
                drafting: drafting.as_speculative_draft().unwrap(),
                settings: chat_settings(&chat, settings.clone()),
                options,
                caller_stop_sequences: &[],
                cancellation: Default::default(),
                on_event,
            })
            .unwrap_or_else(report_failure);
        assert_eq!(output.token_ids().len(), 4);
        assert!(output.stats().rounds() >= 2);
        return serde_json::json!({"ids":output.token_ids(), "text":visible});
    }
    let source = model
        .compile_managed_plain_text_source(
            std::fs::File::open(target.0.join("tokenizer.json")).unwrap(),
        )
        .unwrap_or_else(report_failure);
    let request = ManagedPlainTextSpeculativeRequest {
        text: ManagedPlainTextRequest::new(PROMPT, settings),
        drafting: drafting.as_speculative_draft().unwrap(),
        options,
        cancellation: Default::default(),
        on_event,
    };
    let mut delivered = Vec::new();
    let output = match mode {
        "managed" => model.generate_managed_plain_text_speculative(&source, request),
        "controlled" => model.with_controlled_managed_plain_text_speculative(
            &source,
            request,
            ControlledSpeculativeOptions::default(),
            |session| {
                assert!(session.token_ids().is_empty());
                while let Some(step) = session.step()? {
                    delivered.extend(step.committed_token_ids);
                }
                assert_eq!(delivered, session.token_ids());
                Ok(())
            },
        ),
        _ => panic!("unknown mode {mode}"),
    }
    .unwrap_or_else(report_failure);
    assert_eq!(output.token_ids().len(), 4);
    assert!(output.stats().rounds() >= 2);
    assert!(output.timing().time_to_first_token().is_some());
    if mode == "controlled" {
        assert_eq!(delivered, output.token_ids());
    }
    let address = output.token_ids().as_ptr();
    drop((source, loaded));
    assert_eq!(output.token_ids().as_ptr(), address);
    serde_json::json!({"ids":output.token_ids(), "text":visible})
}

fn verify(residency: Residency, case: &str) {
    if let Ok(mode) = std::env::var(MODE) {
        println!("\n{RESULT}{}", run_selected(residency, &mode));
        return;
    }
    let mut expected = None;
    for mode in ["ordinary", "managed", "controlled"] {
        let result = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", case, "--ignored", "--nocapture"])
            .env(MODE, mode)
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&result.stdout);
        assert!(
            result.status.success(),
            "{residency:?}/{mode}: {stdout}\n{}",
            String::from_utf8_lossy(&result.stderr)
        );
        let actual: serde_json::Value = serde_json::from_str(
            stdout
                .lines()
                .find_map(|line| line.strip_prefix(RESULT))
                .unwrap_or_else(|| {
                    panic!("missing positive execution marker: {residency:?}/{mode}: {stdout}")
                }),
        )
        .unwrap();
        if let Some(expected) = &expected {
            assert_eq!(&actual, expected, "{residency:?}/{mode}");
        } else {
            expected = Some(actual);
        }
    }
}

#[test]
#[ignore = "requires an accessible Metal device"]
fn native_original_host_speculation_matches_ordinary_and_controlled() {
    verify(
        Residency::Host,
        "managed_plain::speculative::residency::native_original_host_speculation_matches_ordinary_and_controlled",
    );
}

#[test]
#[ignore = "requires an accessible Metal device"]
fn native_original_foreground_disk_speculation_matches_ordinary_and_controlled() {
    verify(
        Residency::ForegroundDisk,
        "managed_plain::speculative::residency::native_original_foreground_disk_speculation_matches_ordinary_and_controlled",
    );
}
