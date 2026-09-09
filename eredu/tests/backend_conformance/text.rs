use super::*;
use eredu::api::GenerationOutput;
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

type Terminal = (Vec<u32>, FinishReason, Option<Duration>);

fn terminal<S>(output: GenerationOutput<S>) -> Terminal {
    let ttft = output.timing().time_to_first_token();
    (output.token_ids, output.finish_reason, ttft)
}

// Exercise all public text entry points against the neutral backend and real
// scheduler. Ordinary prefill predicts prompt length; speculation predicts 7,
// then accepts 11 and commits bonus 17.
fn generate<F: FnMut(SemanticEvent)>(
    method: usize,
    model: &mut LoadedModel<MockBackend>,
    input: PreparedChatInput<'_, MockBackend>,
    stops: &[String],
    cancellation: GenerationCancellationToken,
    on_event: F,
) -> Result<Terminal, Box<dyn std::error::Error>> {
    let settings = PreparedChatGenerationSettings {
        overrides: GenerationConfigOverrides {
            max_new_tokens: Some(3),
            ..Default::default()
        },
        ..Default::default()
    };
    Ok(match method {
        0 => terminal(model.generate_prepared_text(PreparedChatGenerationRequest {
            input,
            settings,
            caller_stop_sequences: stops,
            cancellation,
            on_event,
        })?),
        1 => terminal(model.generate_prepared_text_speculative(
            PreparedChatSpeculativeGenerationRequest {
                input,
                drafting: SpeculativeDraft::Embedded,
                settings,
                options: Default::default(),
                caller_stop_sequences: stops,
                cancellation,
                on_event,
            },
        )?),
        2 => {
            let output = model.generate_prepared_text_speculative_batch(
                PreparedChatSpeculativeBatchRequest {
                    drafting: SpeculativeDraft::Embedded,
                    lanes: vec![PreparedChatSpeculativeBatchLane {
                        input,
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
            terminal(lanes.remove(0))
        }
        3 => terminal(model.with_controlled_text_speculative(
            PreparedChatSpeculativeGenerationRequest {
                input,
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

fn output_model(method: usize, pieces: [&str; 2]) -> (LoadedModel<MockBackend>, Vec<u32>) {
    let mut probe = unicode_model_with_template(None, 64, TEMPLATE);
    let chat = probe.prepare_chat(request()).unwrap();
    let first = probe.encode(chat.rendered_prompt(), false).unwrap().len() as u32;
    let ids = if method == 0 {
        vec![first, first + 1, first + 2]
    } else {
        vec![7, 11, 17]
    };
    let mut value: serde_json::Value =
        serde_json::from_str(&probe.tokenizer().to_string(false).unwrap()).unwrap();
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
    let model = LoadedModel::from_runtime(
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
    );
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
                    let chat = model.prepare_chat(request()).unwrap();
                    assert!(matches!(
                        chat.semantic_support(),
                        SemanticSupport::Unsupported { .. }
                    ));
                    assert!(chat.text_generation_support().is_supported());
                    assert_eq!(chat.rendered_prompt(), "user says: hello\nreply begins: ");
                    let prompt = model.encode(chat.rendered_prompt(), false).unwrap();
                    let input = if backend_input {
                        PreparedChatInput::prepared_backend_input(&chat, prompt)
                    } else {
                        PreparedChatInput::rendered_prompt(&chat)
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
                        input,
                        &stops,
                        Default::default(),
                        |event| {
                            first_delivery.get_or_insert_with(|| started.elapsed());
                            events.push(event);
                        },
                    )
                    .unwrap();
                    assert_eq!(ids, expected[..if stop { 2 } else { 3 }]);
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
                let chat = model.prepare_chat(request()).unwrap();
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
                    PreparedChatInput::prepared_backend_input(&chat, vec![0; 7]),
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
        let chat = model.prepare_chat(request()).unwrap();
        let cancellation = GenerationCancellationToken::new();
        let (ids, reason, ttft) = generate(
            method,
            &mut model,
            PreparedChatInput::prepared_backend_input(&chat, vec![0; 7]),
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
                let chat = model.prepare_chat(ChatTemplateRequest {
                    tools: vec![serde_json::json!({"type":"function","function":{"name":"lookup","parameters":{"type":"object","properties":{}}}})],
                    tool_choice: choice,
                    ..request()
                }).unwrap();
                let error = generate(
                    method,
                    &mut model,
                    PreparedChatInput::rendered_prompt(&chat),
                    &[],
                    Default::default(),
                    |_| panic!("rejected tools emitted"),
                )
                .unwrap_err();
                assert!(error.to_string().contains("tool declarations"));
            }
            let required = model.prepare_chat(ChatTemplateRequest {
                tool_choice: ToolChoice::Required,
                ..request()
            });
            match required {
                Ok(required) => assert!(generate(
                    method,
                    &mut model,
                    PreparedChatInput::rendered_prompt(&required),
                    &[],
                    Default::default(),
                    |_| panic!("required call admitted")
                )
                .is_err()),
                Err(error) => assert!(error.to_string().contains("no tools were supplied")),
            }
        }
        let mut model = unicode_model_with_template(None, 64, TEMPLATE);
        assert!(model
            .prepare_chat(ChatTemplateRequest {
                enable_thinking: Some(true),
                ..request()
            })
            .is_err());
        let chat = model
            .prepare_chat(ChatTemplateRequest {
                enable_thinking: Some(true),
                allow_unparsed_reasoning: true,
                ..request()
            })
            .unwrap();
        assert!(generate(
            method,
            &mut model,
            PreparedChatInput::rendered_prompt(&chat),
            &[],
            Default::default(),
            |_| {}
        )
        .is_ok());
    }
}

#[test]
fn unrecognized_templates_still_require_explicit_text_generation() {
    let mut model = unicode_model_with_template(None, 64, TEMPLATE);
    let chat = model.prepare_chat(request()).unwrap();
    let error = model
        .generate_prepared_chat(PreparedChatGenerationRequest {
            input: PreparedChatInput::rendered_prompt(&chat),
            settings: Default::default(),
            caller_stop_sequences: &[],
            cancellation: Default::default(),
            on_event: |_| panic!("unsupported chat emitted"),
        })
        .unwrap_err();
    assert!(matches!(error, PreparedChatError::Semantic(_)));
    let error = model
        .generate_prepared_chat_speculative(PreparedChatSpeculativeGenerationRequest {
            input: PreparedChatInput::rendered_prompt(&chat),
            drafting: SpeculativeDraft::Embedded,
            settings: Default::default(),
            options: Default::default(),
            caller_stop_sequences: &[],
            cancellation: Default::default(),
            on_event: |_| panic!("unsupported chat emitted"),
        })
        .unwrap_err();
    assert!(matches!(error, PreparedChatSpeculativeError::Semantic(_)));
}
