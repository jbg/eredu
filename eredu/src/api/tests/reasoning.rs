use super::*;

const R1: &str = include_str!(
    "../../../tests/fixtures/chat_templates/deepseek-r1-distill-qwen-1.5b-ad9f0ae0.jinja"
);
const QWQ: &str = include_str!("../../../tests/fixtures/chat_templates/qwq-32b-976055f8.jinja");
// A tool-free template exercises generated reasoning and verified disable policy.
const HISTORY: &str = "{% for message in messages %}{{message.role}}: {% if message.reasoning_content %}<think>{{message.reasoning_content}}</think>{% endif %}{{message.content}}\n{% endfor %}{% if add_generation_prompt %}assistant: {% if enable_thinking is defined and not enable_thinking %}<think></think>{% endif %}{% endif %}";

fn tokenizer(added: Option<bool>) -> ChatTokenizer {
    let words = WordLevel::builder()
        .vocab([("[UNK]".into(), 0)].into_iter().collect())
        .unk_token("[UNK]".into())
        .build()
        .unwrap();
    let mut raw = Tokenizer::new(words);
    if let Some(special) = added {
        raw.add_special_tokens([
            AddedToken::from("<think>", special),
            AddedToken::from("</think>", special),
        ])
        .unwrap();
    }
    ChatTokenizer::from_tokenizer(raw)
}

fn request() -> ChatTemplateRequest {
    ChatTemplateRequest {
        messages: vec![
            json!({"role":"user", "content":"Literal <think> and </think> in a question."}),
        ],
        enable_thinking: Some(true),
        add_generation_prompt: true,
        ..Default::default()
    }
}

fn prepare(
    template: &str,
    request: ChatTemplateRequest,
    added: Option<bool>,
) -> Result<crate::runtime::chat::PreparedChat, TextModelError> {
    prepare_chat_from_parts(
        &mut tokenizer(added),
        template.into(),
        "unrelated-architecture",
        &[],
        None,
        request,
    )
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
fn reasoning_only_admits_released_templates_without_tools_or_a_constraint_compiler() {
    for template in [R1, QWQ, HISTORY] {
        for added in [None, Some(false), Some(true)] {
            let chat = prepare(template, request(), added).unwrap();
            assert!(chat.capabilities().reasoning_parser.is_supported());
            assert!(!chat.native_tool_support().is_supported());
            assert!(!chat.capabilities().tool_output_parser.is_supported());
            let plan = chat.generation_runtime_plan().unwrap();
            assert!(plan.generation_constraint().is_none());
            let output = if template == HISTORY {
                "<think>Check 🦀</think>Answer <think>literal</think>"
            } else {
                "Check 🦀</think>Answer <think>literal</think>"
            };
            // Independent expected channels across every UTF-8 boundary, including
            // partial delimiters copied into speculative/controlled parser forks.
            for split in (0..=output.len()).filter(|&i| output.is_char_boundary(i)) {
                let mut parser = plan.create_parser().unwrap();
                parser.push(&output[..split]).unwrap();
                let mut fork = parser.fork().unwrap();
                parser.push(&output[split..]).unwrap();
                parser.finish(FinishReason::Eos).unwrap();
                fork.push(&output[split..]).unwrap();
                fork.finish(FinishReason::Eos).unwrap();
                assert_eq!(
                    channels(parser.events()),
                    ("Check 🦀".into(), "Answer <think>literal</think>".into())
                );
                assert_eq!(parser.events(), fork.events());
            }
        }
    }
}

#[test]
fn reasoning_initial_state_follows_the_actual_generation_suffix() {
    let mut no_suffix = request();
    no_suffix.add_generation_prompt = false;
    let chat = prepare(R1, no_suffix, None).unwrap();
    let mut parser = chat
        .generation_runtime_plan()
        .unwrap()
        .create_parser()
        .unwrap();
    parser
        .push("Answer quoting <think>literal</think>")
        .unwrap();
    parser.finish(FinishReason::Eos).unwrap();
    assert_eq!(
        channels(parser.events()),
        (
            String::new(),
            "Answer quoting <think>literal</think>".into()
        )
    );

    let mut disabled = request();
    disabled.enable_thinking = Some(false);
    let chat = prepare(HISTORY, disabled.clone(), None).unwrap();
    let mut parser = chat
        .generation_runtime_plan()
        .unwrap()
        .create_parser()
        .unwrap();
    parser.push("<think>literal</think>").unwrap();
    parser.finish(FinishReason::Eos).unwrap();
    assert_eq!(
        channels(parser.events()),
        (String::new(), "<think>literal</think>".into())
    );
    assert!(matches!(
        prepare(R1, disabled, None),
        Err(TextModelError::UnsupportedReasoning { .. })
    ));
}

#[test]
fn incomplete_reasoning_keeps_its_channel_and_cancellation_does_not_flush_lookbehind() {
    let chat = prepare(R1, request(), None).unwrap();
    let plan = chat.generation_runtime_plan().unwrap();
    let mut parser = plan.create_parser().unwrap();
    parser.push("Check </thi").unwrap();
    let mut cancelled = parser.fork().unwrap();
    parser.finish(FinishReason::MaxTokens).unwrap();
    cancelled.cancel();
    assert_eq!(
        channels(parser.events()),
        ("Check </thi".into(), String::new())
    );
    assert_eq!(
        channels(cancelled.events()),
        ("Check ".into(), String::new())
    );
}

#[test]
fn literal_tags_and_ignored_reasoning_history_do_not_grant_reasoning_support() {
    let template = "{% for message in messages %}{{message.content}}{% endfor %}{% if add_generation_prompt %}assistant: {% endif %}";
    assert!(matches!(
        prepare(template, request(), None),
        Err(TextModelError::UnsupportedReasoning { .. })
    ));
}

#[test]
fn qwq_composes_reasoning_with_existing_native_xml_tools() {
    let compiler = Ok(ConstraintCompiler::synthetic_for_tests());
    let mut tokenizer = production_chat_tokenizer(50);
    let mut request = request();
    request.tools = vec![
        json!({"type":"function", "function":{"name":"lookup", "parameters":{"type":"object", "properties":{"value":{"type":"integer"}}, "required":["value"]}}}),
    ];
    let chat = prepare_chat_from_parts(
        &mut tokenizer,
        QWQ.into(),
        "unrelated",
        &[],
        Some(&compiler),
        request,
    )
    .unwrap();
    assert!(chat.native_tool_support().is_supported());
    assert!(chat.capabilities().reasoning_parser.is_supported());
    let mut parser = chat
        .generation_runtime_plan()
        .unwrap()
        .create_parser()
        .unwrap();
    parser.push("Check</think>\n<tool_call>\n{\"name\":\"lookup\",\"arguments\":{\"value\":7}}\n</tool_call>").unwrap();
    parser.finish(FinishReason::Eos).unwrap();
    assert_eq!(channels(parser.events()).0, "Check");
    assert!(parser.events().iter().any(|event| matches!(event, SemanticEvent::ToolCallCompleted { name, arguments, .. } if name == "lookup" && arguments["value"] == 7)));
}

#[test]
fn reasoning_only_does_not_satisfy_requested_native_tools() {
    let mut request = request();
    request.tools = vec![
        json!({"type":"function", "function":{"name":"lookup", "parameters":{"type":"object"}}}),
    ];
    request.tool_choice = ToolChoice::Required;
    request.allow_unparsed_reasoning = true;
    let chat = prepare(HISTORY, request, None).unwrap();
    assert!(!chat.native_tool_support().is_supported());
    assert!(matches!(
        chat.semantic_support(),
        crate::runtime::chat::SemanticSupport::Unsupported { .. }
    ));
    assert!(!chat.text_generation_support().is_supported());
}

#[test]
fn reasoning_only_keeps_structural_stops_from_a_recognized_tool_envelope() {
    let mut tokenizer = production_chat_tokenizer(50);
    let stop = tokenizer.token_to_id("<|im_end|>").unwrap();
    let chat = prepare_chat_from_parts(
        &mut tokenizer,
        QWQ.into(),
        "unrelated",
        &[],
        None,
        request(),
    )
    .unwrap();
    let mut parser = chat
        .generation_runtime_plan()
        .unwrap()
        .create_parser_with_stops([])
        .unwrap();
    parser.push("Check</think>Answer").unwrap();
    parser.push_structural(stop, "<|im_end|>").unwrap();
    assert_eq!(channels(parser.events()), ("Check".into(), "Answer".into()));
    assert_eq!(
        parser.events().last(),
        Some(&SemanticEvent::Finished {
            reason: FinishReason::StopSequence
        })
    );
}

#[test]
fn every_reasoning_protocol_survives_rejected_tools_missing_tool_tokens_and_compiler_failure() {
    let cases: &[(&str, &str, &[&str], &str)] = &[
        ("gemma", GEMMA4_EDGE_FIXTURE, &["<|channel>", "<channel|>", "<turn|>"],
            "<|channel>thought\nCheck<channel|>Answer<turn|>"),
        ("inkling", INKLING_SMALL_FIXTURE,
            &["<|message_model|>", "<|content_text|>", "<|content_thinking|>", "<|end_message|>", "<|content_model_end_sampling|>"],
            "<|content_thinking|>Check<|end_message|><|message_model|><|content_text|>Answer<|end_message|><|content_model_end_sampling|>"),
        ("atem", MUSE_GLIMMER_FIXTURE_WITH_TERMINATOR, &["<|start|>", "<|message|>", "<|eom|>", "<|eot|>"],
            " to=self<|message|>Check<|eom|><|start|>assistant to=user<|message|>Answer<|eot|>"),
        ("harmony", GPT_OSS_HARMONY_CURRENT_FIXTURE_WITH_TERMINATOR,
            &["<|start|>", "<|end|>", "<|message|>", "<|channel|>", "<|return|>"],
            "<|channel|>analysis<|message|>Check<|end|><|start|>assistant<|channel|>final<|message|>Answer<|return|>"),
        ("qwen", QWEN3_CURRENT_FIXTURE_WITH_TERMINATOR, &[], "<think>Check</think>Answer"),
        ("nemotron", NEMOTRON_NANO_V2_FIXTURE_WITH_TERMINATOR, &[], "Check</think>Answer"),
        ("deepseek", DEEPSEEK_V31_TOOL_FIXTURE, &[], "Check</think>Answer"),
    ];
    let compiler = Err("grammar compilation is unavailable".into());
    for &(name, template, tokens, output) in cases {
        let template = without_tool_rendering(template);
        let chat = prepare_chat_from_parts(
            &mut llama_chat_tokenizer(32, tokens),
            template.into(),
            "unrelated-model",
            &[],
            Some(&compiler),
            request(),
        )
        .unwrap_or_else(|error| panic!("{name}: {error}"));
        assert!(
            chat.capabilities().reasoning_parser.is_supported(),
            "{name}"
        );
        assert!(
            !chat.capabilities().tool_output_parser.is_supported(),
            "{name}"
        );
        assert!(
            !chat.capabilities().tool_input_rendering.is_supported(),
            "{name}"
        );
        assert!(chat
            .generation_runtime_plan()
            .unwrap()
            .generation_constraint()
            .is_none());
        assert_eq!(
            parse_response(&chat, output),
            ("Check".into(), "Answer".into()),
            "{name}"
        );
    }
    // IFM effort changes the wire delimiters, independently of every tool format.
    for (effort, closing) in [
        ("high", "</ifm|think>"),
        ("medium", "</ifm|think_fast>"),
        ("low", "</ifm|think_faster>"),
    ] {
        let template = without_tool_rendering(include_str!(
            "../../../tests/fixtures/k2_horizon/text/dense-gguf.jinja"
        ));
        let mut req = request();
        req.reasoning_effort = Some(effort.into());
        let chat = prepare_chat_from_parts(
            &mut llama_chat_tokenizer(32, &["<|ifm|im_start|>", "<|ifm|im_end|>"]),
            template.into(),
            "unrelated-model",
            &[],
            Some(&compiler),
            req,
        )
        .unwrap();
        assert!(chat.capabilities().reasoning_parser.is_supported());
        assert!(!chat.capabilities().tool_output_parser.is_supported());
        assert!(chat
            .generation_runtime_plan()
            .unwrap()
            .generation_constraint()
            .is_none());
        assert_eq!(
            parse_response(&chat, &format!("Check{closing}Answer<|ifm|im_end|>")),
            ("Check".into(), "Answer".into())
        );
    }
}

fn without_tool_rendering(template: &str) -> String {
    format!(concat!(
        "{{% if tools %}}{{{{ raise_exception('tools are unavailable') }}}}{{% endif %}}",
        "{{% for m in messages %}}{{% if m.role == 'tool' or m.tool_calls %}}",
        "{{{{ raise_exception('tool history is unavailable') }}}}{{% endif %}}{{% endfor %}}{template}"
    ), template = template)
}

fn parse_response(chat: &crate::runtime::chat::PreparedChat, mut output: &str) -> (String, String) {
    let plan = chat.generation_runtime_plan().unwrap();
    let mut parser = plan.create_parser_with_stops([]).unwrap();
    let tokens = plan.structural_tokens().collect::<Vec<_>>();
    while !output.is_empty() {
        if let Some((id, spelling)) = tokens
            .iter()
            .find(|(_, spelling)| output.starts_with(*spelling))
        {
            parser.push_structural(*id, spelling).unwrap();
            output = &output[spelling.len()..];
        } else {
            let len = output.chars().next().unwrap().len_utf8();
            parser.push(&output[..len]).unwrap();
            output = &output[len..];
        }
    }
    parser.finish(FinishReason::Eos).unwrap();
    channels(parser.events())
}

#[test]
fn delimited_reasoning_and_effort_are_admitted_without_a_tool_dialect() {
    let thinking = HISTORY
        .replace("<think>", "<thinking>")
        .replace("</think>", "</thinking>");
    let chat = prepare(&thinking, request(), None).unwrap();
    assert_eq!(
        parse_response(&chat, "<thinking>Check</thinking>Answer"),
        ("Check".into(), "Answer".into())
    );
    let mut req = request();
    req.reasoning_effort = Some("low".into());
    let template = without_tool_rendering(include_str!(
        "../../../tests/fixtures/chat_templates/qwen3.8-flash-next-de4b8e4d.jinja"
    ));
    let chat = prepare(&template, req, None).unwrap();
    assert!(chat
        .rendered_prompt()
        .contains("Reasoning effort is set to low."));
    assert_eq!(
        parse_response(&chat, "Check</think>Answer"),
        ("Check".into(), "Answer".into())
    );
}

#[test]
fn response_only_parsing_does_not_activate_a_known_native_tool_envelope() {
    for tool_rendering in [false, true] {
        let template: String = if tool_rendering {
            QWEN3_CURRENT_FIXTURE_WITH_TERMINATOR.into()
        } else {
            without_tool_rendering(QWEN3_CURRENT_FIXTURE_WITH_TERMINATOR)
        };
        for compiler_available in [false, true] {
            let compiler = if compiler_available {
                Ok(ConstraintCompiler::synthetic_for_tests())
            } else {
                Err("constraint compiler failed".into())
            };
            let chat = prepare_chat_from_parts(
                &mut production_chat_tokenizer(32),
                template.clone().into(),
                "unrelated-model",
                &[],
                Some(&compiler),
                request(),
            )
            .unwrap();
            assert_eq!(
                chat.native_tool_support().is_supported(),
                tool_rendering && compiler_available
            );
            assert_eq!(
                chat.capabilities().tool_output_parser.is_supported(),
                tool_rendering
            );
            let visible = "<tool_call>\n{\"name\":\"lookup\",\"arguments\":{}}\n</tool_call>";
            assert_eq!(
                parse_response(&chat, &format!("<think>\nCheck\n</think>{visible}")),
                ("\nCheck\n".into(), visible.into())
            );
        }
    }
}

#[test]
fn tool_protocol_recognition_does_not_grant_unverified_response_reasoning() {
    let template = format!(
        "{{% filter replace('<think>', '') | replace('</think>', '') %}}{}{{% endfilter %}}",
        NEMOTRON_NANO_V2_FIXTURE_WITH_TERMINATOR
    );
    let compiler = Ok(ConstraintCompiler::synthetic_for_tests());
    let mut req = request();
    req.enable_thinking = None;
    let chat = prepare_chat_from_parts(
        &mut llama_chat_tokenizer(32, &["<SPECIAL_12>"]),
        template.into(),
        "unrelated-model",
        &[],
        Some(&compiler),
        req,
    )
    .unwrap();
    assert!(chat.native_tool_support().is_supported());
    assert!(!chat.capabilities().reasoning_parser.is_supported());
    assert!(matches!(
        chat.semantic_support(),
        crate::runtime::chat::SemanticSupport::Unsupported { .. }
    ));
}
