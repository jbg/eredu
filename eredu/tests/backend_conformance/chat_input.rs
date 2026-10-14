use super::*;
use eredu::api::{
    ChatContentFormat, ChatInputPreparationError, ChatInputRequest, ChatMessage, ChatPart,
    ChatPreparationOptions, ChatToolArgumentFormat, ChatToolCall, TraceLimits,
};
use eredu_core::{capture::CapturePlan, MediaRequestError};
use serde_json::json;
use std::ops::ControlFlow;

const CONTENT: &str = "{% for message in messages %}{{ message.content }}{% endfor %}";

fn image() -> ChatPart {
    ChatPart::Image(RgbImage::new(vec![17, 83, 201], 1, 1).unwrap())
}

fn mixed() -> ChatInputRequest {
    ChatInputRequest {
        messages: vec![ChatMessage::User(vec![
            ChatPart::Text("ordinary_1 ".into()),
            image(),
            ChatPart::Text(" ordinary_2 ".into()),
            image(),
            ChatPart::Text(" ordinary_3".into()),
        ])],
        ..Default::default()
    }
}

fn settings() -> PreparedChatGenerationSettings {
    PreparedChatGenerationSettings {
        overrides: GenerationConfigOverrides {
            max_new_tokens: Some(2),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn limits() -> TraceLimits {
    TraceLimits {
        per_record_bytes: 65536,
        total_bytes: 1024 * 1024,
    }
}

#[test]
fn typed_chat_input_preserves_text_whitespace_and_explicit_joining() {
    let mut model = unicode_model_with_template(None, 64, CONTENT);
    let request = |content_format| ChatInputRequest {
        messages: vec![ChatMessage::User(vec![
            ChatPart::Text("ordinary_1 ".into()),
            ChatPart::Text("ordinary_2".into()),
        ])],
        options: ChatPreparationOptions {
            content_format,
            ..Default::default()
        },
        ..Default::default()
    };
    let prepared = model
        .prepare_chat_text_input(request(Default::default()))
        .unwrap();
    assert_eq!(prepared.chat().rendered_prompt(), "ordinary_1 ordinary_2");
    assert_eq!(prepared.count(), InputTokenCount::text(2));
    let joined = model
        .prepare_chat_text_input(request(ChatContentFormat::String {
            separator: "\n".into(),
        }))
        .unwrap();
    assert_eq!(joined.chat().rendered_prompt(), "ordinary_1 \nordinary_2");
    let (chat, payload) = prepared.into_parts();
    let output = model
        .generate_prepared_text(PreparedChatGenerationRequest {
            input: payload.into_input(&model, &chat).unwrap(),
            settings: settings(),
            caller_stop_sequences: &[],
            cancellation: Default::default(),
            control_handle: None,
            on_event: |_| {},
        })
        .unwrap();
    assert_eq!(output.token_ids, vec![2, 3]);
}

#[test]
fn typed_chat_input_media_alignment_and_controlled_output_match_ordinary() {
    let mut expected = None;
    for controlled in [false, true] {
        let mut model = unicode_model_with_template(None, 2200, CONTENT);
        let rendered = model.prepare_typed_chat(mixed()).unwrap();
        assert!(rendered.has_media());
        let prepared = model.materialize_chat_input(rendered).unwrap();
        let count = prepared.count();
        assert_eq!(
            (
                count.text_tokens,
                count.media_positions,
                count.model_positions
            ),
            (4, 2, 6)
        );
        let prepared = model
            .prepare_observed_chat_input(prepared, settings(), CapturePlan::none(), limits())
            .unwrap();
        // Two independently meaningful image positions interleave with the actual text IDs.
        // The mock processor adds its final "hello" ID (unknown in this vocabulary).
        assert_eq!(prepared.prompt_token_ids(), &[2, 2001, 3, 2001, 4, 0]);
        let mut events = Vec::new();
        let output = if controlled {
            let mut session = model
                .start_controlled_text(prepared, &[], Default::default(), |record| {
                    events.push(record.generation.event);
                    ControlFlow::Continue(())
                })
                .unwrap();
            session
                .run(|record| {
                    events.push(record.generation.event);
                    ControlFlow::Continue(())
                })
                .unwrap();
            session.token_ids().to_vec()
        } else {
            model
                .generate_observed_text(prepared, &[], Default::default(), |record| {
                    events.push(record.event);
                    ControlFlow::Continue(())
                })
                .unwrap()
                .token_ids
        };
        assert_eq!(output, vec![6, 7]);
        // Timing is intentionally not compared; committed semantic output must agree.
        let visible: Vec<_> = events
            .into_iter()
            .filter_map(|event| match event {
                eredu::api::ObservedGenerationEvent::Semantic { event, .. } => Some(event),
                _ => None,
            })
            .collect();
        if let Some(expected) = &expected {
            assert_eq!(&visible, expected);
        } else {
            expected = Some(visible);
        }
    }
}

#[test]
fn typed_chat_input_rejects_lost_duplicated_transformed_or_reordered_media() {
    for template in [
        "{{ messages[0].content[0].text }}",
        "{% for part in messages[0].content %}{{ part.text }}{{ part.text }}{% endfor %}",
        "{% for part in messages[0].content %}{{ part.text|upper }}{% endfor %}",
        "{% for part in messages[0].content|reverse %}{{ part.text }}{% endfor %}",
    ] {
        let mut model = unicode_model_with_template(None, 64, template);
        let mut request = mixed();
        request.options.content_format = ChatContentFormat::TextParts;
        let error = model.prepare_typed_chat(request).err().unwrap();
        assert!(
            matches!(
                error,
                ChatInputPreparationError::Media(
                    MediaRequestError::PlaceholderCount { .. }
                        | MediaRequestError::PlaceholderOrder { .. }
                )
            ),
            "{error}"
        );
    }
    // Literal text resembling the implementation's private binding stays literal.
    let mut model = unicode_model_with_template(None, 64, CONTENT);
    let request = ChatInputRequest {
        messages: vec![ChatMessage::User(vec![
            ChatPart::Text("<eredu_".into()),
            ChatPart::Text("media_0_0>".into()),
            image(),
        ])],
        ..Default::default()
    };
    let first = model.prepare_typed_chat(request.clone()).unwrap();
    let second = model.prepare_typed_chat(request).unwrap();
    assert!(first
        .chat()
        .rendered_prompt()
        .starts_with("<eredu_media_0_0>"));
    assert_eq!(
        first.chat().rendered_prompt(),
        second.chat().rendered_prompt()
    );
    model.materialize_chat_input(first).unwrap();
}

#[test]
fn typed_chat_input_renders_parsed_historical_calls_and_linked_results() {
    let call = ChatToolCall::from_json_arguments(
        Some("call_1".into()),
        "lookup",
        r#"{"count":3,"nested":{"ok":true},"literal":"{\"x\":1}"}"#,
    )
    .unwrap();
    let request = ChatInputRequest {
        messages: vec![
            ChatMessage::User(vec![ChatPart::Text("hello".into())]),
            ChatMessage::Assistant {
                content: vec![],
                reasoning_content: None,
                tool_calls: vec![call.clone()],
            },
            ChatMessage::Tool {
                tool_call_id: "call_1".into(),
                name: Some("lookup".into()),
                content: vec![ChatPart::Text("result".into())],
            },
        ],
        ..Default::default()
    };
    let mut model = unicode_model(None);
    let prepared = model.prepare_typed_chat(request.clone()).unwrap();
    let rendered = prepared.chat().rendered_prompt();
    let body = rendered
        .split("<tool_call>\n")
        .nth(1)
        .unwrap()
        .split("\n</tool_call>")
        .next()
        .unwrap();
    let value: serde_json::Value = serde_json::from_str(body).unwrap();
    assert_eq!(value, json!({"name":"lookup", "arguments":call.arguments}));
    assert!(rendered.contains("<tool_response>\nresult\n</tool_response>"));

    // A custom template can explicitly require a string without double-encoding its object.
    let template = "{% set call = messages[1].tool_calls[0] %}{% if call.function.arguments is not string %}{{ raise_exception('expected string') }}{% endif %}{{ call.id }}|{{ call.function.arguments }}|{{ messages[2].tool_call_id }}|{{ messages[2].name }}";
    let mut model = unicode_model_with_template(None, 64, template);
    let mut string_request = request;
    string_request.options.tool_argument_format = ChatToolArgumentFormat::JsonString;
    let prepared = model.prepare_typed_chat(string_request).unwrap();
    let parts: Vec<_> = prepared.chat().rendered_prompt().split('|').collect();
    assert_eq!(parts[0], "call_1");
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(parts[1]).unwrap(),
        json!(call.arguments)
    );
    assert_eq!(&parts[2..], &["call_1", "lookup"]);
    assert!(matches!(
        ChatToolCall::from_json_arguments(None, "lookup", "[]"),
        Err(ChatInputPreparationError::ToolArgumentsNotObject { .. })
    ));
    assert!(matches!(
        ChatToolCall::from_json_arguments(None, "lookup", "{"),
        Err(ChatInputPreparationError::InvalidToolArguments { .. })
    ));
}

#[test]
fn typed_chat_input_auto_selects_established_string_argument_support() {
    // Keep the checkpoint's protocol while requiring serialized historical arguments.
    let template = QWEN_TEMPLATE.replace(
        "{{- tool_call.arguments | tojson }}",
        concat!(
            "{%- if tool_call.arguments is not string %}",
            "{{ raise_exception('serialized arguments required') }}{% endif %}",
            "{{- tool_call.arguments }}",
        ),
    );
    let mut model = unicode_model_with_template(None, 64, &template);
    let call = ChatToolCall::from_json_arguments(None, "lookup", r#"{"count":3}"#).unwrap();
    let prepared = model
        .prepare_typed_chat(ChatInputRequest {
            messages: vec![
                ChatMessage::User(vec![ChatPart::Text("hello".into())]),
                ChatMessage::Assistant {
                    content: vec![],
                    reasoning_content: None,
                    tool_calls: vec![call],
                },
            ],
            ..Default::default()
        })
        .unwrap();
    assert!(!prepared
        .chat()
        .capabilities()
        .mapping_tool_arguments
        .is_supported());
    assert!(prepared
        .chat()
        .capabilities()
        .string_tool_arguments
        .is_supported());
    let body = prepared
        .chat()
        .rendered_prompt()
        .split("<tool_call>\n")
        .nth(1)
        .unwrap()
        .split("\n</tool_call>")
        .next()
        .unwrap();
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(body).unwrap(),
        json!({"name":"lookup", "arguments":{"count":3}})
    );
    // The raw API continues passing historical strings unchanged.
    let raw = model.prepare_chat(ChatTemplateRequest {
        messages: vec![json!({"role":"user", "content":"hello"}), json!({
            "role":"assistant", "content":"", "tool_calls":[{
                "type":"function", "function":{"name":"lookup", "arguments":r#"{"count":3}"#},
            }],
        })], ..Default::default()
    }).unwrap();
    assert_eq!(raw.rendered_prompt(), prepared.chat().rendered_prompt());
}

#[test]
fn typed_chat_input_counted_media_uses_existing_speculative_and_forecast_routes() {
    let mut model = unicode_model_with_template(None, 2200, CONTENT);
    let prepared = model.prepare_chat_input(mixed()).unwrap();
    let count = prepared.count();
    let (chat, payload) = prepared.into_parts();
    let input = payload.into_input(&model, &chat).unwrap();
    let request = PreparedChatSpeculativeGenerationRequest {
        input,
        drafting: SpeculativeDraft::Embedded,
        settings: settings(),
        caller_stop_sequences: &[],
        cancellation: Default::default(),
        control_handle: None,
        on_event: |_| {},
    };
    let forecast = model
        .forecast_prepared_speculative_generation(&request, &Default::default())
        .unwrap();
    assert_eq!(forecast.request.input, count);
    let output = model.generate_prepared_text_speculative(request).unwrap();
    assert_eq!(output.token_ids, vec![7, 11]);
}

#[test]
fn typed_chat_input_cannot_cross_sessions_or_change_rendered_chat() {
    let mut first = unicode_model_with_template(None, 64, CONTENT);
    let mut other = unicode_model_with_template(None, 64, CONTENT);
    let request = || ChatInputRequest {
        messages: vec![ChatMessage::User(vec![ChatPart::Text("ordinary_1".into())])],
        ..Default::default()
    };
    let rendered = first.prepare_typed_chat(request()).unwrap();
    assert!(matches!(
        other.materialize_chat_text_input(rendered),
        Err(ChatInputPreparationError::SessionMismatch)
    ));
    let prepared = first.prepare_chat_text_input(request()).unwrap();
    let (chat, payload) = prepared.into_parts();
    assert!(matches!(
        payload.into_input(&other, &chat),
        Err(ChatInputPreparationError::SessionMismatch)
    ));
    let prepared = first.prepare_chat_text_input(request()).unwrap();
    let (_, payload) = prepared.into_parts();
    let changed = other
        .prepare_chat(ChatTemplateRequest {
            messages: vec![json!({"role":"user", "content":"ordinary_2"})],
            ..Default::default()
        })
        .unwrap();
    assert!(matches!(
        payload.into_input(&first, &changed),
        Err(ChatInputPreparationError::ChatMismatch)
    ));
    let prepared = first.prepare_chat_text_input(request()).unwrap();
    let (chat, payload) = prepared.into_parts();
    let input = payload.into_input(&first, &chat).unwrap();
    let error = other
        .generate_prepared_text(PreparedChatGenerationRequest {
            input,
            settings: settings(),
            caller_stop_sequences: &[],
            cancellation: Default::default(),
            control_handle: None,
            on_event: |_| panic!("mismatched input must not generate"),
        })
        .unwrap_err();
    assert!(
        matches!(error, eredu::api::PreparedChatError::Backend(ref error)
        if error.kind() == eredu_core::BackendFailureKind::InvalidSession)
    );
}
