//! Independent original lane sources match the existing ordinary fair scheduler.
use super::*;
use eredu::api::{
    ManagedPlainTextSpeculativeBatchLane, ManagedPlainTextSpeculativeBatchRequest,
    PreparedChatSpeculativeBatchLane, PreparedChatSpeculativeBatchRequest,
};
use std::{cell::RefCell, num::NonZeroUsize, rc::Rc};

const CASE: &str = "managed_plain::embedded::batch::native_original_qwen_two_lane_batch_matches_ordinary_sources_and_order";
const MODE: &str = "EREDU_PUBLIC_EMBEDDED_BATCH_MODE";
const RESULT: &str = "PUBLIC_EMBEDDED_BATCH_RESULT:";
const PROMPTS: [&str; 2] = ["abc", "efghij"];
const OUTPUTS: [usize; 2] = [3, 4];
const DRAFTS: [usize; 2] = [1, 2];
const SEEDS: [u64; 2] = [827, 1949];

fn lane_settings(index: usize, original: bool) -> PreparedChatGenerationSettings {
    let mut value = settings(if index == 0 { 0.65 } else { 0.9 });
    value.overrides.max_new_tokens = Some(OUTPUTS[index]);
    value.seed = SEEDS[index];
    if !original {
        value.inference.managed_memory_capacity_bytes = None;
    }
    value
}
fn run(mode: &str) -> serde_json::Value {
    let original = match mode {
        "ordinary" => false,
        "managed" => true,
        _ => panic!("unexpected mode"),
    };
    let target = managed_fixture(super::super::super::speculative::captured_prefill::source(
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
        LoadedModel::load_execution_plan(&MlxBackendFactory::default(), &target.0, &execution)
            .unwrap_or_else(report_failure);
    let options = loaded.speculative_generation_options().unwrap().unwrap();
    let (model, drafting) = loaded.parts_mut();
    let visible = Rc::new(RefCell::new([String::new(), String::new()]));
    let event = |index: usize| -> Box<dyn FnMut(SemanticEvent)> {
        let visible = visible.clone();
        Box::new(move |event| {
            if let SemanticEvent::TextDelta(text) = event {
                visible.borrow_mut()[index].push_str(text.as_str());
            }
        })
    };
    let mut source = None;
    let output = if original {
        source = Some(
            model
                .compile_managed_plain_text_source(
                    std::fs::File::open(target.0.join("tokenizer.json")).unwrap(),
                )
                .unwrap_or_else(report_failure),
        );
        let lanes = (0..2)
            .map(|index| ManagedPlainTextSpeculativeBatchLane {
                text: ManagedPlainTextRequest::new(PROMPTS[index], lane_settings(index, true)),
                max_draft_tokens: NonZeroUsize::new(DRAFTS[index]).unwrap(),
                cancellation: Default::default(),
                on_event: event(index),
            })
            .collect();
        model
            .generate_managed_plain_text_speculative_batch(
                source.as_ref().unwrap(),
                ManagedPlainTextSpeculativeBatchRequest {
                    drafting: drafting.as_speculative_draft().unwrap(),
                    lanes,
                    scheduler: options.scheduler,
                },
            )
            .unwrap_or_else(report_failure)
    } else {
        let chat = model
            .source_chat_with_capacity(
                ChatTemplateRequest {
                    messages: vec![serde_json::json!({"role":"user","content":"batch"})],
                    add_generation_prompt: false,
                    tool_choice: ToolChoice::None,
                    ..Default::default()
                },
                8 * 1024 * 1024 * 1024,
            )
            .unwrap();
        let prefixes = PROMPTS.map(|prompt| model.encode(prompt, true).unwrap());
        let lanes = (0..2)
            .map(|index| PreparedChatSpeculativeBatchLane {
                chat: &chat,
                input: eredu::api::PreparedChatPrompt::TokenIds(&prefixes[index]),
                output_mode: eredu::api::PreparedChatOutputMode::Text,
                skip_special_tokens: true,
                settings: chat_settings(&chat, lane_settings(index, false)),
                max_draft_tokens: NonZeroUsize::new(DRAFTS[index]).unwrap(),
                caller_stop_sequences: &[],
                cancellation: Default::default(),
                on_event: event(index),
            })
            .collect();
        model
            .generate_prepared_chat_speculative_batch(PreparedChatSpeculativeBatchRequest {
                drafting: drafting.as_speculative_draft().unwrap(),
                lanes,
                scheduler: options.scheduler,
            })
            .unwrap_or_else(report_failure)
    };
    assert_eq!(output.requests().len(), 2);
    for (index, request) in output.requests().iter().enumerate() {
        assert_eq!(request.token_ids().len(), OUTPUTS[index]);
        assert!(request.stats().rounds() > 0);
        assert!(request.timing().time_to_first_token().is_some());
    }
    let addresses: Vec<_> = output
        .requests()
        .iter()
        .map(|request| request.token_ids().as_ptr())
        .collect();
    drop((source, loaded));
    let rows: Vec<_> = output
        .requests()
        .iter()
        .enumerate()
        .map(|(index, request)| {
            assert_eq!(
                request.token_ids().as_ptr(),
                addresses[index],
                "escaped lane output retains its actual custody"
            );
            serde_json::json!({"input":PROMPTS[index], "seed":SEEDS[index], "draft":DRAFTS[index],
            "ids":request.token_ids(), "text":visible.borrow()[index]})
        })
        .collect();
    rows.into()
}
#[test]
#[ignore = "requires an accessible Metal device and original Embedded batch sources"]
fn native_original_qwen_two_lane_batch_matches_ordinary_sources_and_order() {
    if let Ok(mode) = std::env::var(MODE) {
        println!("\n{RESULT}{}", run(&mode));
        return;
    }
    let mut expected = None;
    for mode in ["ordinary", "managed"] {
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
        let actual: serde_json::Value = serde_json::from_str(
            stdout
                .lines()
                .find_map(|line| line.strip_prefix(RESULT))
                .unwrap_or_else(|| panic!("missing positive batch marker: {mode}: {stdout}")),
        )
        .unwrap();
        if let Some(expected) = &expected {
            assert_eq!(&actual, expected, "{mode} lane source/order parity");
        } else {
            expected = Some(actual);
        }
    }
}
