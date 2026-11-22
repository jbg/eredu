use super::*;
use eredu::api::{ObservedGenerationEvent, OutputMode};
use eredu::runtime::chat::ToolChoice;
use eredu_core::execution_control::SnapshotLimits;
use std::ops::ControlFlow;

const TEMPLATE: &str =
    include_str!("../fixtures/chat_templates/deepseek-r1-distill-qwen-1.5b-ad9f0ae0.jinja");
const TAGGED_TEMPLATE: &str =
    include_str!("../fixtures/chat_templates/qwen3.8-flash-next-de4b8e4d.jinja");
const EMULATED_TEXT: &str = "Answer<tool_call>emulated</tool_call>";

fn setup(
    method: usize,
    tool_choice: Option<ToolChoice>,
) -> (
    LoadedModel<MockBackend, MockSession>,
    PreparedChat,
    Vec<u32>,
) {
    let mut request = ChatTemplateRequest {
        messages: vec![serde_json::json!({"role":"user", "content":"hello"})],
        enable_thinking: Some(true),
        add_generation_prompt: true,
        ..Default::default()
    };
    let (template, tail) = if let Some(choice) = tool_choice {
        request.tool_choice = choice;
        request.tools = vec![serde_json::json!({"type":"function","function":{
            "name":"lookup", "parameters":{"type":"object"}
        }})];
        (TAGGED_TEMPLATE, "nk>Answer")
    } else {
        (TEMPLATE, "nk>Answer<tool_call>emulated</tool_call>")
    };
    let (mut model, ids) =
        text::output_model_for_request(method, ["Check</thi", tail], template, request.clone());
    let chat = model.prepare_chat(request).unwrap();
    assert!(chat.capabilities().reasoning_parser.is_supported());
    assert_eq!(
        chat.native_tool_support().is_supported(),
        tool_choice.is_some()
    );
    (model, chat, ids)
}

fn settings() -> PreparedChatGenerationSettings {
    PreparedChatGenerationSettings {
        overrides: GenerationConfigOverrides {
            max_new_tokens: Some(3),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn channels(events: &[SemanticEvent]) -> (String, String) {
    let mut result = (String::new(), String::new());
    for event in events {
        match event {
            SemanticEvent::ReasoningDelta(s) => result.0.push_str(s),
            SemanticEvent::TextDelta(s) => result.1.push_str(s),
            _ => {}
        }
    }
    result
}

#[test]
fn reasoning_generation_matches_controlled_and_speculative_delivery() {
    for (native_tools, method) in [(false, 0), (false, 1), (true, 0), (true, 1)] {
        let mut expected = None;
        for controlled in [false, true] {
            let (mut model, chat, ids) = setup(method, native_tools.then_some(ToolChoice::Auto));
            let mut events = vec![];
            let request = GenerationRequest {
                input: PreparedChatInput::rendered_prompt(&chat),
                output_mode: OutputMode::Semantic,
                settings: PreparedChatGenerationSettings {
                    decoding_mode: Some(if method == 0 {
                        eredu_core::decoding::DecodingMode::Autoregressive
                    } else {
                        eredu_core::decoding::DecodingMode::SelfSpeculation(Default::default())
                    }),
                    ..settings()
                },
                caller_stop_sequences: &[],
                cancellation: Default::default(),
                control_handle: None,
                on_event: |event| events.push(event),
            };
            let tokens = if method == 0 {
                if controlled {
                    model
                        .session()
                        .with_controlled(request, Default::default(), |run| run.run())
                        .unwrap()
                        .token_ids()
                        .to_vec()
                } else {
                    model.session().generate(request).unwrap().token_ids
                }
            } else {
                let mut session = model.drafting_session(SpeculativeDraft::Embedded, None);
                if controlled {
                    session
                        .with_controlled(request, Default::default(), |run| run.run())
                        .unwrap()
                        .token_ids()
                        .to_vec()
                } else {
                    session.generate(request).unwrap().token_ids
                }
            };
            assert_eq!(tokens, ids);
            assert_eq!(
                channels(&events),
                (
                    "Check".into(),
                    if native_tools {
                        "Answer"
                    } else {
                        EMULATED_TEXT
                    }
                    .into()
                )
            );
            assert_eq!(
                events
                    .iter()
                    .filter(|event| matches!(event, SemanticEvent::Finished { .. }))
                    .count(),
                1
            );
            if let Some(expected) = &expected {
                assert_eq!(&events, expected);
            } else {
                expected = Some(events);
            }
        }
    }
}

#[test]
fn reasoning_snapshot_restores_a_partial_closing_marker() {
    // A forbidden-call plan retains a declarative parser with bounded controller state.
    for (choice, visible) in [(None, EMULATED_TEXT), (Some(ToolChoice::None), "Answer")] {
        let (mut model, chat, ids) = setup(0, choice);
        let prepared = model.prepare_controlled_chat(&chat, settings()).unwrap();
        let mut events = vec![];
        let mut collect = |record: eredu::api::ControlledGenerationRecord| {
            if let ObservedGenerationEvent::Semantic { event, .. } = record.generation.event {
                events.push(event);
            }
            ControlFlow::Continue(())
        };
        let mut session = model
            .start_controlled(
                prepared,
                &[],
                Default::default(),
                &mut collect,
                OutputMode::Semantic,
            )
            .unwrap();
        session
            .enable_snapshots(SnapshotLimits {
                max_snapshots: 1,
                max_branches: 0,
                retained_bytes: 4_000_000,
                cumulative_copy_bytes: 32_000_000,
            })
            .unwrap();
        session.step(&mut collect).unwrap();
        let saved = session.snapshot(&mut collect).unwrap();
        session.run(&mut collect).unwrap();
        assert_eq!(session.token_ids(), ids);
        session.restore(&saved, &mut collect).unwrap();
        session.run(&mut collect).unwrap();
        assert_eq!(session.token_ids(), ids);
        assert_eq!(channels(&events), ("Check".into(), visible.repeat(2)));
    }
}

#[test]
fn retained_text_inspection_and_loading_agree_on_reasoning_only_support() {
    use eredu::api::{ArtifactSource, PreparedTextModel, TextInspectionOptions, TextModelOptions};
    let root = TestDirectory::new();
    write_loadable_text_artifact(root.path());
    let artifact = eredu::api::inspect_artifact(&ArtifactSource::new(root.path())).unwrap();
    let mut prepared = PreparedTextModel::from_artifact(
        artifact,
        TextModelOptions {
            chat_template: Some(TEMPLATE.into()),
        },
    )
    .unwrap();
    let request = ChatTemplateRequest {
        messages: vec![serde_json::json!({"role":"user", "content":"hello"})],
        enable_thinking: Some(true),
        add_generation_prompt: true,
        ..Default::default()
    };
    let report = prepared.inspect(TextInspectionOptions {
        chat_request: Some(request.clone()),
    });
    assert_eq!(
        report.semantic_streaming,
        eredu_core::InspectionReadiness::Ready
    );
    assert_eq!(
        report.native_tools,
        eredu_core::InspectionReadiness::Unsupported
    );
    let mut loaded = prepared
        .load_on::<_, super::FixtureLoadSelection>(MockBackend, Default::default())
        .unwrap();
    let chat = loaded.prepare_chat(request).unwrap();
    assert!(chat.capabilities().reasoning_parser.is_supported());
    assert!(!chat.native_tool_support().is_supported());
}
