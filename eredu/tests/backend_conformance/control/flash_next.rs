//! Checkpoint-native protocol through the real shared facade drivers. The small
//! sequential mock chooses tokens; the pinned template and tool parser are real.
use super::*;
use eredu::runtime::chat::{SemanticSupport, ToolChoice};
use serde_json::{json, Value};

const TEMPLATE: &str =
    include_str!("../../fixtures/chat_templates/qwen3.8-flash-next-de4b8e4d.jinja");
const VOCABULARY: u32 = 4096;

fn request(thinking: bool) -> ChatTemplateRequest {
    ChatTemplateRequest {
        messages: vec![json!({"role":"user", "content":"Find two places near Paris."})],
        tools: vec![json!({"type":"function", "function": {
            "name":"lookup", "description":"Find places", "parameters": {
                "type":"object", "properties": {
                    "query":{"type":"string"}, "limit":{"type":"integer"}
                }, "required":["query", "limit"], "additionalProperties":false
            }
        }})],
        tool_choice: ToolChoice::Auto,
        enable_thinking: Some(thinking),
        add_generation_prompt: true,
        ..Default::default()
    }
}

fn model(first: Option<u32>, pieces: &[&str]) -> LoadedModel<MockBackend> {
    let mut vocabulary = std::iter::once(("[UNK]".to_owned(), 0))
        .chain((1..VOCABULARY).map(|i| (format!("ordinary_{i}"), i)))
        .collect::<std::collections::HashMap<_, _>>();
    let eos = first.map_or(VOCABULARY - 1, |id| id + pieces.len() as u32);
    vocabulary.retain(|_, id| *id != eos && !first.is_some_and(|first| *id >= first && *id < eos));
    if let Some(first) = first {
        for (index, piece) in pieces.iter().enumerate() {
            // The ByteLevel decoder concatenates pieces without inventing spaces.
            let encoded = piece.replace(' ', "Ġ").replace('\n', "Ċ");
            vocabulary.insert(encoded, first + index as u32);
        }
    }
    vocabulary.insert("<|im_end|>".into(), eos);
    let words = WordLevel::builder()
        .vocab(vocabulary.into_iter().collect())
        .unk_token("[UNK]".into())
        .build()
        .unwrap();
    let mut tokenizer = Tokenizer::new(words);
    tokenizer.with_pre_tokenizer(Some(Whitespace));
    tokenizer.with_decoder(Some(ByteLevel::default()));
    tokenizer
        .add_special_tokens([AddedToken::from("<|im_end|>", true).normalized(false)])
        .unwrap();
    LoadedModel::from_runtime(
        ModelRuntime::prepare(MockBackend, Default::default()).unwrap(),
        ChatTokenizer::from_tokenizer(tokenizer),
        LoadedTextModelConfig {
            model_family: ModelKind::Qwen4Exp,
            effective_model_type: "qwen4_exp".into(),
            model_id: "flash-next-protocol-fixture".into(),
            chat_template: Some(TEMPLATE.into()),
            eos_token_ids: vec![eos],
            checkpoint_generation_config: None,
        },
    )
}

fn setup(
    request: ChatTemplateRequest,
    pieces: &[&str],
) -> (LoadedModel<MockBackend>, PreparedChat, Vec<u32>) {
    let mut probe = model(None, pieces);
    let chat = probe.prepare_chat(request.clone()).unwrap();
    let first = probe.encode(chat.rendered_prompt(), false).unwrap().len() as u32;
    assert!(first + pieces.len() as u32 + 1 < VOCABULARY);
    let mut model = model(Some(first), pieces);
    let chat = model.prepare_chat(request).unwrap();
    assert_eq!(
        model.encode(chat.rendered_prompt(), false).unwrap().len(),
        first as usize
    );
    assert!(matches!(
        chat.semantic_support(),
        SemanticSupport::Supported
    ));
    assert!(chat.native_tool_support().is_supported());
    (model, chat, (first..=first + pieces.len() as u32).collect())
}

fn settings() -> PreparedChatGenerationSettings {
    PreparedChatGenerationSettings {
        overrides: GenerationConfigOverrides {
            max_new_tokens: Some(16),
            ..Default::default()
        },
        seed: 17,
        ..Default::default()
    }
}

fn arguments(events: &[SemanticEvent]) -> Value {
    let fragments: String = events
        .iter()
        .filter_map(|event| match event {
            SemanticEvent::ToolArgumentsDelta {
                index: 0,
                json_fragment,
            } => Some(json_fragment.as_str()),
            _ => None,
        })
        .collect();
    serde_json::from_str(&fragments).unwrap()
}

#[test]
fn flash_next_tagged_tools_pause_partial_arguments_and_replay_tool_results() {
    for thinking in [false, true] {
        let mut pieces = vec![];
        if thinking {
            pieces.extend(["Check nearby places.", "\n</think>\n\n"]);
        }
        pieces.extend([
            "<tool_call>\n<function=lookup>\n<parameter=query>\npar",
            "is and nearby\n</parameter>\n<parameter=limit>\n",
            "2\n</parameter>\n</function>\n</tool_call>",
        ]);
        let partial_steps = if thinking { 3 } else { 1 };
        let (mut model, chat, ids) = setup(request(thinking), &pieces);
        let mut ordinary_events = vec![];
        let ordinary = model
            .generate_prepared_chat(PreparedChatGenerationRequest {
                input: PreparedChatInput::rendered_prompt(&chat),
                settings: settings(),
                caller_stop_sequences: &[],
                cancellation: Default::default(),
                on_event: |event| ordinary_events.push(event),
            })
            .unwrap();
        // With one call allowed, closing the tagged call completes the grammar;
        // the following EOS must remain unconsumed.
        assert_eq!(ordinary.token_ids, ids[..pieces.len()]);
        assert_eq!(ordinary.finish_reason, FinishReason::GrammarComplete);
        let reasoning: String = ordinary_events
            .iter()
            .filter_map(|event| match event {
                SemanticEvent::ReasoningDelta(text) => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(
            reasoning,
            if thinking { "Check nearby places." } else { "" }
        );
        let expected_arguments = json!({"query":"paris and nearby", "limit":2});
        assert_eq!(arguments(&ordinary_events), expected_arguments);
        assert_eq!(
            ordinary_events
                .iter()
                .filter(|event| matches!(event, SemanticEvent::ToolCallEnd))
                .count(),
            1
        );
        let call_id = ordinary_events
            .iter()
            .find_map(|event| match event {
                SemanticEvent::ToolCallStart { index: 0, id, name } if name == "lookup" => {
                    Some(id.clone())
                }
                _ => None,
            })
            .unwrap();
        let prepared = model
            .prepare_observed_chat(&chat, settings(), CapturePlan::none(), limits())
            .unwrap();
        let mut records = vec![];
        let mut session = model
            .start_controlled_chat(prepared, &[], Default::default(), |record| {
                records.push(record);
                ControlFlow::Continue(())
            })
            .unwrap();
        for _ in 0..partial_steps {
            session
                .step(|record| {
                    records.push(record);
                    ControlFlow::Continue(())
                })
                .unwrap();
        }
        assert_eq!(session.token_ids(), &ids[..partial_steps]);
        assert!(!semantic(&records)
            .iter()
            .any(|event| matches!(event, SemanticEvent::ToolCallEnd)));
        // Shared production grammars currently have no storage estimate. Keep
        // this existing typed capability gap visible, without blocking ordinary
        // controlled advancement or inventing a family-specific bypass.
        let before = (
            session.token_ids().to_vec(),
            session.next_prediction(),
            session.emitted_bytes(),
            session.output_checkpoint(),
            session.status(),
        );
        let error = session
            .enable_snapshots(SnapshotLimits {
                max_snapshots: 1,
                max_branches: 0,
                retained_bytes: 4_000_000,
                cumulative_copy_bytes: 32_000_000,
            })
            .unwrap_err();
        assert!(
            matches!(error, eredu::api::ControlledGenerationError::Capture(
            eredu_core::capture::CaptureError::Unsupported(reason)
        ) if reason == "complete active grammar storage estimate is unavailable")
        );
        assert!(matches!(
            session.capabilities().snapshot,
            ControlSupport::Unsupported { .. }
        ));
        assert!(session.snapshot_usage().is_none());
        assert_eq!(
            (
                session.token_ids().to_vec(),
                session.next_prediction(),
                session.emitted_bytes(),
                session.output_checkpoint(),
                session.status()
            ),
            before
        );
        session
            .run(|record| {
                records.push(record);
                ControlFlow::Continue(())
            })
            .unwrap();
        assert_eq!(session.status(), GenerationStatus::Completed);
        assert_eq!(session.token_ids(), ordinary.token_ids);
        assert_eq!(semantic(&records), ordinary_events);
        drop(session);

        // Feed the emitted call back as a checkpoint-native assistant message,
        // then execute the tool-result continuation through the same drivers.
        let mut next = request(thinking);
        next.messages.extend([
            json!({"role":"assistant", "content":"", "reasoning_content":reasoning,
                "tool_calls":[{"id":call_id, "type":"function", "function":{"name":"lookup", "arguments":expected_arguments}}]}),
            json!({"role":"tool", "tool_call_id":call_id, "content":"Found two places."}),
        ]);
        let answer = if thinking {
            vec!["Use those results.", "\n</think>\n\n", "Found two places."]
        } else {
            vec!["Found two places."]
        };
        let (mut model, chat, ids) = setup(next, &answer);
        assert!(chat
            .rendered_prompt()
            .contains("<parameter=query>\nparis and nearby\n</parameter>"));
        assert!(chat
            .rendered_prompt()
            .contains("<tool_response>\nFound two places.\n</tool_response>"));
        let mut events = vec![];
        let ordinary = model
            .generate_prepared_chat(PreparedChatGenerationRequest {
                input: PreparedChatInput::rendered_prompt(&chat),
                settings: settings(),
                caller_stop_sequences: &[],
                cancellation: Default::default(),
                on_event: |event| events.push(event),
            })
            .unwrap();
        assert_eq!(ordinary.token_ids, ids);
        let text: String = events
            .iter()
            .filter_map(|event| match event {
                SemanticEvent::TextDelta(text) => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(
            text,
            if thinking {
                "\n\nFound two places."
            } else {
                "Found two places."
            }
        );
        assert!(!events
            .iter()
            .any(|event| matches!(event, SemanticEvent::ToolCallStart { .. })));
        let prepared = model
            .prepare_observed_chat(&chat, settings(), CapturePlan::none(), limits())
            .unwrap();
        let mut records = vec![];
        let mut controlled = model
            .start_controlled_chat(prepared, &[], Default::default(), |record| {
                records.push(record);
                ControlFlow::Continue(())
            })
            .unwrap();
        controlled
            .run(|record| {
                records.push(record);
                ControlFlow::Continue(())
            })
            .unwrap();
        assert_eq!(controlled.token_ids(), ids);
        assert_eq!(semantic(&records), events);
    }
}
