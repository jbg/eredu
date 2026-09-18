use super::*;
use eredu::api::{
    GenerationOutput, PreparedChatOutputMode, PreparedChatPrompt, PreparedChatRequest,
};
use eredu::runtime::chat::{SemanticSupport, ToolChoice};
use eredu_core::GenerationCancellationToken;
use std::time::{Duration, Instant};

const TEMPLATE: &str = "{% for message in messages %}{{ message.role }} says: {{ message.content }}\n{% endfor %}{% if add_generation_prompt %}reply begins: {% endif %}";

fn request() -> ChatTemplateRequest {
    ChatTemplateRequest {
        messages: vec![serde_json::json!({"role": "user", "content": "hello"})],
        add_generation_prompt: true,
        ..Default::default()
    }
}

type Terminal = (
    eredu_core::SpeculativeTokenIds,
    FinishReason,
    Option<Duration>,
);

fn terminal<S, T: TerminalTokenStorage>(output: GenerationOutput<S, T>) -> Terminal {
    let ttft = output.timing().time_to_first_token();
    (
        output.token_ids.into_terminal_tokens(),
        output.finish_reason,
        ttft,
    )
}

// Exercise all public text entry points against the neutral backend and real
// scheduler. Ordinary prefill predicts prompt length; speculation predicts 7,
// then accepts 11 and commits bonus 17.
fn generate<F: FnMut(SemanticEvent)>(
    method: usize,
    model: &mut original_sources::Fixture<MockBackend>,
    chat: &PreparedChat,
    input: PreparedChatPrompt<'_, <MockBackend as TextGenerationBackend>::Prompt>,
    stops: &[String],
    cancellation: GenerationCancellationToken,
    on_event: F,
) -> Result<Terminal, Box<dyn std::error::Error>> {
    let settings = original_sources::settings(PreparedChatGenerationSettings {
        overrides: GenerationConfigOverrides {
            max_new_tokens: Some(3),
            ..Default::default()
        },
        ..Default::default()
    });
    Ok(match method {
        0 => {
            let mut settings = settings;
            settings.inference.managed_memory_capacity_bytes = Some(original_sources::CAPACITY);
            let mut request = PreparedChatRequest::new(chat, settings);
            request.input = input;
            request.output_mode = PreparedChatOutputMode::Text;
            request.stop_sequences = stops;
            let mut on_event = on_event;
            match model.start_prepared_chat(request, &cancellation)? {
                Some(session) => terminal(session.run(&cancellation, &mut on_event)?),
                None => {
                    on_event(SemanticEvent::Finished {
                        reason: FinishReason::Cancelled,
                    });
                    (
                        Vec::<u32>::new().into_terminal_tokens(),
                        FinishReason::Cancelled,
                        None,
                    )
                }
            }
        }
        1 => terminal(model.generate_prepared_chat_speculative(
            PreparedChatSpeculativeRequest {
                chat,
                input,
                output_mode: PreparedChatOutputMode::Text,
                skip_special_tokens: true,
                drafting: SpeculativeDraft::Embedded,
                settings,
                options: Default::default(),
                caller_stop_sequences: stops,
                cancellation,
                on_event,
            },
        )?),
        2 => {
            let output = model.generate_prepared_chat_speculative_batch(
                PreparedChatSpeculativeBatchRequest {
                    drafting: SpeculativeDraft::Embedded,
                    lanes: vec![PreparedChatSpeculativeBatchLane {
                        chat,
                        input,
                        output_mode: PreparedChatOutputMode::Text,
                        skip_special_tokens: true,
                        settings,
                        max_draft_tokens: NonZeroUsize::new(2).unwrap(),
                        caller_stop_sequences: stops,
                        cancellation,
                        on_event: Box::new(on_event),
                    }],
                    scheduler: Default::default(),
                },
            )?;
            let mut lanes = output.into_requests();
            assert_eq!(lanes.len(), 1);
            terminal(lanes.remove(0).expect("one validated request"))
        }
        3 => terminal(model.with_controlled_prepared_chat_speculative(
            PreparedChatSpeculativeRequest {
                chat,
                input,
                output_mode: PreparedChatOutputMode::Text,
                skip_special_tokens: true,
                drafting: SpeculativeDraft::Embedded,
                settings,
                options: Default::default(),
                caller_stop_sequences: stops,
                cancellation,
                on_event,
            },
            Default::default(),
            |session| {
                while session.step()?.is_some() {}
                Ok(())
            },
        )?),
        _ => unreachable!(),
    })
}

fn output_model(
    method: usize,
    pieces: [&str; 2],
) -> (original_sources::Fixture<MockBackend>, Vec<u32>) {
    let tokenizer = unicode_tokenizer(None, 64);
    let mut value: serde_json::Value =
        serde_json::from_str(&tokenizer.to_string(false).unwrap()).unwrap();
    let mut probe =
        unicode_model_from_tokenizer(ChatTokenizer::from_tokenizer(tokenizer), TEMPLATE);
    let chat = {
        let request = request();
        let cancellation = eredu_core::GenerationCancellationToken::new();
        let source = probe
            .chat_source(!request.tools.is_empty(), &cancellation)
            .unwrap()
            .unwrap();
        probe
            .prepare_chat(&source, &request, original_sources::CAPACITY, &cancellation)
            .unwrap()
            .unwrap()
    };
    let first = probe.encode(chat.rendered_prompt(), false).unwrap().len() as u32;
    let ids = if method == 0 {
        vec![first, first + 1, first + 2]
    } else {
        vec![7, 11, 17]
    };
    let vocab = value["model"]["vocab"].as_object_mut().unwrap();
    vocab
        .retain(|token, id| token != "<|im_end|>" && !ids.contains(&(id.as_u64().unwrap() as u32)));
    for (id, token) in ids.iter().zip([pieces[0], pieces[1], "<|im_end|>"]) {
        vocab.insert(token.into(), serde_json::json!(id));
    }
    value["added_tokens"] = serde_json::json!([]);
    let mut tokenizer = Tokenizer::from_bytes(serde_json::to_vec(&value).unwrap()).unwrap();
    tokenizer
        .add_special_tokens([AddedToken::from("<|im_end|>", true).normalized(false)])
        .unwrap();
    let model = original_sources::Fixture::from_runtime(
        ModelRuntime::prepare(MockBackend, ()).unwrap(),
        ChatTokenizer::from_tokenizer(tokenizer),
        LoadedTextModelConfig {
            model_family: ModelKind::Qwen2,
            effective_model_type: "qwen2".into(),
            model_id: "unrecognized-text".into(),
            chat_template: Some(TEMPLATE.into()),
            eos_token_ids: vec![ids[2]],
            checkpoint_generation_config: None,
        },
    )
    .unwrap();
    (model, ids)
}

#[test]
fn unrecognized_text_preserves_prompt_unicode_literal_protocol_and_cross_token_stops() {
    for method in 0..4 {
        for backend_input in [false, true] {
            for unicode in [false, true] {
                for stop in [false, true] {
                    let pieces = if unicode {
                        ["Ã", "©"]
                    } else {
                        ["<think>literal", "<tool_call>"]
                    };
                    let (mut model, expected) = output_model(method, pieces);
                    let chat = {
                        let request = request();
                        let cancellation = eredu_core::GenerationCancellationToken::new();
                        let source = model
                            .chat_source(!request.tools.is_empty(), &cancellation)
                            .unwrap()
                            .unwrap();
                        model
                            .prepare_chat(
                                &source,
                                &request,
                                original_sources::CAPACITY,
                                &cancellation,
                            )
                            .unwrap()
                            .unwrap()
                    };
                    assert!(matches!(
                        chat.semantic_support(),
                        SemanticSupport::Unsupported { .. }
                    ));
                    assert!(chat.text_generation_support().is_supported());
                    assert_eq!(chat.rendered_prompt(), "user says: hello\nreply begins: ");
                    let prompt = model.encode(chat.rendered_prompt(), false).unwrap();
                    let input = if backend_input {
                        PreparedChatPrompt::TokenIds(&prompt)
                    } else {
                        PreparedChatPrompt::Rendered
                    };
                    let stops = if stop {
                        vec![if unicode {
                            "é".into()
                        } else {
                            "literal<tool_call>".into()
                        }]
                    } else {
                        vec![]
                    };
                    let mut events = vec![];
                    let started = Instant::now();
                    let mut first_delivery = None;
                    let (ids, reason, ttft) = generate(
                        method,
                        &mut model,
                        &chat,
                        input,
                        &stops,
                        Default::default(),
                        |event| {
                            first_delivery.get_or_insert_with(|| started.elapsed());
                            events.push(event);
                        },
                    )
                    .unwrap();
                    assert_eq!(ids.as_ref(), &expected[..if stop { 2 } else { 3 }]);
                    assert_eq!(
                        reason,
                        if stop {
                            FinishReason::StopSequence
                        } else {
                            FinishReason::Eos
                        }
                    );
                    assert!(ttft.unwrap() <= first_delivery.unwrap());
                    let mut text = String::new();
                    for event in &events[..events.len() - 1] {
                        let SemanticEvent::TextDelta(delta) = event else {
                            panic!("unexpected semantic event: {event:?}")
                        };
                        text.push_str(delta);
                    }
                    assert_eq!(
                        text,
                        match (unicode, stop) {
                            (true, true) => "",
                            (true, false) => "é",
                            (false, true) => "<think>",
                            (false, false) => "<think>literal<tool_call>",
                        }
                    );
                    assert_eq!(events.last(), Some(&SemanticEvent::Finished { reason }));
                }
            }
        }
    }
}

#[test]
fn text_ttft_counts_invisible_eos_and_stop_tokens_and_excludes_pre_cancellation() {
    for method in 0..4 {
        for eos in [false, true] {
            for pre_cancel in [false, true] {
                let mut model = unicode_model_with_template(eos.then_some(5), 64, TEMPLATE);
                let chat = {
                    let request = request();
                    let cancellation = eredu_core::GenerationCancellationToken::new();
                    let source = model
                        .chat_source(!request.tools.is_empty(), &cancellation)
                        .unwrap()
                        .unwrap();
                    model
                        .prepare_chat(&source, &request, original_sources::CAPACITY, &cancellation)
                        .unwrap()
                        .unwrap()
                };
                let cancellation = GenerationCancellationToken::new();
                if pre_cancel {
                    cancellation.cancel();
                }
                let stops = if eos {
                    vec![]
                } else {
                    vec!["ordinary_6".into()]
                };
                let mut events = vec![];
                let (ids, reason, ttft) = generate(
                    method,
                    &mut model,
                    &chat,
                    PreparedChatPrompt::TokenIds(&[0; 7]),
                    &stops,
                    cancellation,
                    |event| events.push(event),
                )
                .unwrap();
                assert_eq!(ids, if pre_cancel { vec![] } else { vec![7] });
                assert_eq!(ttft.is_some(), !pre_cancel);
                assert_eq!(
                    reason,
                    if pre_cancel {
                        FinishReason::Cancelled
                    } else if eos {
                        FinishReason::Eos
                    } else {
                        FinishReason::StopSequence
                    }
                );
                assert_eq!(events, [SemanticEvent::Finished { reason }]);
            }
        }
        let mut model = unicode_model_with_template(None, 64, TEMPLATE);
        let chat = {
            let request = request();
            let cancellation = eredu_core::GenerationCancellationToken::new();
            let source = model
                .chat_source(!request.tools.is_empty(), &cancellation)
                .unwrap()
                .unwrap();
            model
                .prepare_chat(&source, &request, original_sources::CAPACITY, &cancellation)
                .unwrap()
                .unwrap()
        };
        let cancellation = GenerationCancellationToken::new();
        let (ids, reason, ttft) = generate(
            method,
            &mut model,
            &chat,
            PreparedChatPrompt::TokenIds(&[0; 7]),
            &[],
            cancellation.clone(),
            |_| cancellation.cancel(),
        )
        .unwrap();
        assert_eq!(ids, [7]);
        assert_eq!(reason, FinishReason::Cancelled);
        assert!(ttft.is_some());
    }
}

#[test]
fn text_generation_rejects_native_tools_and_requires_unparsed_thinking_opt_in() {
    for method in 0..4 {
        for template in [TEMPLATE, QWEN_TEMPLATE] {
            let mut model = unicode_model_with_template(None, 64, template);
            for choice in [ToolChoice::None, ToolChoice::Auto, ToolChoice::Required] {
                let chat = {
                    let request = ChatTemplateRequest {
                        tools: vec![
                            serde_json::json!({"type":"function","function":{"name":"lookup","parameters":{"type":"object","properties":{}}}}),
                        ],
                        tool_choice: choice,
                        ..request()
                    };
                    let cancellation = eredu_core::GenerationCancellationToken::new();
                    let source = model
                        .chat_source(!request.tools.is_empty(), &cancellation)
                        .unwrap()
                        .unwrap();
                    model
                        .prepare_chat(&source, &request, original_sources::CAPACITY, &cancellation)
                        .unwrap()
                        .unwrap()
                };
                let error = generate(
                    method,
                    &mut model,
                    &chat,
                    PreparedChatPrompt::Rendered,
                    &[],
                    Default::default(),
                    |_| panic!("rejected tools emitted"),
                )
                .unwrap_err();
                assert!(error.to_string().contains("tool declarations"));
            }
            let requested = ChatTemplateRequest {
                tool_choice: ToolChoice::Required,
                ..request()
            };
            let cancellation = eredu_core::GenerationCancellationToken::new();
            let source = model.chat_source(false, &cancellation).unwrap().unwrap();
            let required = model
                .prepare_chat(
                    &source,
                    &requested,
                    original_sources::CAPACITY,
                    &cancellation,
                )
                .map(|prepared| prepared.expect("uncancelled preparation"));
            match required {
                Ok(required) => assert!(
                    generate(
                        method,
                        &mut model,
                        &required,
                        PreparedChatPrompt::Rendered,
                        &[],
                        Default::default(),
                        |_| panic!("required call admitted")
                    )
                    .is_err()
                ),
                Err(error) => assert!(error.to_string().contains("no tools were supplied")),
            }
        }
        let mut model = unicode_model_with_template(None, 64, TEMPLATE);
        let requested = ChatTemplateRequest {
            enable_thinking: Some(true),
            ..request()
        };
        let cancellation = eredu_core::GenerationCancellationToken::new();
        let source = model.chat_source(false, &cancellation).unwrap().unwrap();
        assert!(
            model
                .prepare_chat(
                    &source,
                    &requested,
                    original_sources::CAPACITY,
                    &cancellation
                )
                .is_err()
        );
        let chat = {
            let request = ChatTemplateRequest {
                enable_thinking: Some(true),
                allow_unparsed_reasoning: true,
                ..request()
            };
            let cancellation = eredu_core::GenerationCancellationToken::new();
            let source = model
                .chat_source(!request.tools.is_empty(), &cancellation)
                .unwrap()
                .unwrap();
            model
                .prepare_chat(&source, &request, original_sources::CAPACITY, &cancellation)
                .unwrap()
                .unwrap()
        };
        assert!(
            generate(
                method,
                &mut model,
                &chat,
                PreparedChatPrompt::Rendered,
                &[],
                Default::default(),
                |_| {}
            )
            .is_ok()
        );
    }
}

#[test]
fn unrecognized_templates_still_require_explicit_text_generation() {
    let mut model = unicode_model_with_template(None, 64, TEMPLATE);
    let chat = {
        let request = request();
        let cancellation = eredu_core::GenerationCancellationToken::new();
        let source = model
            .chat_source(!request.tools.is_empty(), &cancellation)
            .unwrap()
            .unwrap();
        model
            .prepare_chat(&source, &request, original_sources::CAPACITY, &cancellation)
            .unwrap()
            .unwrap()
    };
    let error = {
        let cancellation = eredu_core::GenerationCancellationToken::new();
        let mut on_event = |_| panic!("unsupported chat emitted");
        let mut request = eredu::api::PreparedChatRequest::new(
            &chat,
            original_sources::settings(Default::default()),
        );
        request.stop_sequences = &[];
        model
            .start_prepared_chat(request, &cancellation)
            .and_then(|session| {
                session
                    .expect("uncancelled original request")
                    .run(&cancellation, &mut on_event)
            })
    }
    .unwrap_err();
    assert!(matches!(
        error.semantic_output_rejection(),
        Some(SemanticSupport::Unsupported { .. })
    ));
    let source = model.tokenizer_source().clone();
    let error = model
        .generate_prepared_chat_speculative(eredu::api::PreparedChatSpeculativeRequest {
            chat: &chat,
            input: eredu::api::PreparedChatPrompt::Rendered,
            drafting: SpeculativeDraft::Embedded,
            settings: original_sources::settings(Default::default()),
            options: Default::default(),
            caller_stop_sequences: &[],
            cancellation: Default::default(),
            on_event: |_| panic!("unsupported chat emitted"),
            output_mode: eredu::api::PreparedChatOutputMode::Semantic,
            skip_special_tokens: true,
        })
        .unwrap_err();
    assert!(matches!(
        error.semantic_output_rejection(),
        Some(SemanticSupport::Unsupported { .. })
    ));
}
