use super::*;
use crate::tokenizer::{Tokenizer, load_model_chat_template_from_str};
use serde_json::json;
use std::error::Error as _;

const RELEASED: &str = include_str!("tests/smollm2-tokenizer_config.json");

fn config(template: serde_json::Value) -> String {
    json!({"chat_template": template}).to_string()
}
fn ordinary() -> Tokenizer {
    Tokenizer::from_tokenizer(tokenizers::Tokenizer::new(
        tokenizers::models::bpe::BPE::default(),
    ))
}

#[test]
fn request_selected_sources_match_named_tool_rendering_and_reject_wrong_selection() {
    use crate::tokenizer::ModelChatTemplate;
    let plain = "{% for message in messages %}plain:{{ message.content }}{% endfor %}";
    let tool =
        "{% for message in messages %}tool:{{ message.content }}{% endfor %}|{{ tools|tojson }}";
    let template = ModelChatTemplate::Named(std::collections::BTreeMap::from([
        ("default".into(), plain.into()),
        ("tool_use".into(), tool.into()),
    ]));
    let messages = vec![json!({"role":"user", "content":"λ<value>"})];
    let declarations = vec![json!({"type":"function","function":{"name":"reading"}})];
    for tools in [&[][..], declarations.as_slice()] {
        let has_tools = !tools.is_empty();
        let plan = ChatTemplatePlan::prepare_model(&template, "model.html", has_tools).unwrap();
        let source = plan.compile().unwrap();
        assert!(source.matches_configuration(&template, "model.html"));
        assert!(source.matches_selection(&template, "model.html", has_tools));
        assert!(!source.matches_selection(&template, "model.html", !has_tools));
        let context = ChatRenderContext::from_json(&messages, None, None)
            .unwrap()
            .with_tools(ChatInputArray::Json(tools));
        let rendered = source
            .render_plan_with_context(context)
            .unwrap()
            .render()
            .unwrap();
        for generation in [false, true] {
            let expected = ordinary()
                .apply_chat_template_json(
                    template.clone(),
                    [messages.clone()],
                    Some(tools),
                    "model.html",
                    generation,
                    None,
                )
                .unwrap();
            assert_eq!(rendered.prompt(generation), expected[0]);
        }
    }
    let fallback = ModelChatTemplate::Named(std::collections::BTreeMap::from([(
        "default".into(),
        plain.into(),
    )]));
    let source = ChatTemplatePlan::prepare_model(&fallback, "model", true)
        .unwrap()
        .compile()
        .unwrap();
    assert!(source.matches_selection(&fallback, "model", true));
    assert!(source.matches_selection(&fallback, "model", false));
    let missing = ModelChatTemplate::Named(std::collections::BTreeMap::from([(
        "tool_use".into(),
        tool.into(),
    )]));
    assert!(matches!(
        ChatTemplatePlan::prepare_model(&missing, "model", false),
        Err(ChatSourceError::MissingTemplate)
    ));
}

#[test]
fn config_selection_shares_named_policy_and_decoded_entry_identity() {
    let plain = "plain:{{ messages[0].content }}";
    let tool = "tools:{{ messages[0].content }}|{{ tools|tojson }}";
    let messages = vec![json!({"role":"user", "content":"λ<value>"})];
    let declarations = vec![json!({"type":"function","function":{"name":"reading"}})];
    for entries in [
        json!([{"name":"default","template":plain},{"name":"tool_use","template":tool}]),
        json!([{"name":"tool_use","template":tool},{"name":"default","template":plain}]),
        json!([{"name":"default","template":plain}]),
        json!([{"name":"tool_use","template":tool}]),
    ] {
        // Source selection compares decoded keys/names without materializing
        // config values; escaped spellings preserve the same ordinary identity.
        let input = config(entries).replace("tool_use", "tool_\\u0075se");
        let model = load_model_chat_template_from_str(&input).unwrap().unwrap();
        for has_tools in [false, true] {
            let tools = if has_tools {
                declarations.as_slice()
            } else {
                &[]
            };
            let expected = model.select(Some(tools));
            let planned = ChatTemplatePlan::prepare_config(input.as_bytes(), "named", has_tools);
            let Ok(expected) = expected else {
                assert!(matches!(planned, Err(ChatSourceError::MissingTemplate)));
                continue;
            };
            let source = planned.unwrap().compile().unwrap();
            assert!(source.matches_selection(&model, "named", has_tools));
            assert!(source.matches_configuration(&model, "named"));
            assert_eq!(source.inner.source(), expected.template());
            let context = ChatRenderContext::from_json(&messages, None, None)
                .unwrap()
                .with_tools(ChatInputArray::Json(tools));
            let rendered = source
                .render_plan_with_context(context)
                .unwrap()
                .render()
                .unwrap();
            for generation in [false, true] {
                let expected = ordinary()
                    .apply_chat_template_json(
                        model.clone(),
                        [messages.clone()],
                        Some(tools),
                        "named",
                        generation,
                        None,
                    )
                    .unwrap();
                assert_eq!(rendered.prompt(generation), expected[0]);
            }
        }
    }
    // Invalid unselected entries remain invalid; tool selection is not a way
    // to omit duplicate-name and schema validation for the rest of the file.
    for has_tools in [false, true] {
        let duplicate = config(json!([
            {"name":"default","template":plain},
            {"name":"tool_use","template":tool},
            {"name":"tool_use","template":"other"},
        ]));
        assert!(matches!(
            ChatTemplatePlan::prepare_config(duplicate.as_bytes(), "named", has_tools),
            Err(ChatSourceError::DuplicateName)
        ));
    }
}

#[test]
fn released_config_borrow_fresh_source_and_render_match_ordinary_chat_pipeline() {
    let input = RELEASED.to_owned();
    let model_template = load_model_chat_template_from_str(&input).unwrap().unwrap();
    let plan = ChatTemplatePlan::prepare_config(input.as_bytes(), "smol", false).unwrap();
    match &plan.input {
        Input::Config { document, selected } => {
            assert_eq!(document.source().as_ptr(), input.as_ptr());
            assert_eq!(document.source().len(), input.len());
            assert!(
                selected
                    .expect("selected source")
                    .is(model_template.select(None).unwrap().template())
            );
        }
        Input::Utf8(_) => panic!("config source lost its original borrow"),
    }
    let required = plan.requirements();
    assert!(required.buffer_bytes() > 0 && required.control_bytes() > 0);
    assert_eq!(
        required.required_bytes(),
        required.buffer_bytes() + required.control_bytes()
    );
    let source = plan.compile().unwrap();
    drop(input);
    let other = ChatTemplatePlan::prepare_utf8(vm::supported_source(), "smol")
        .unwrap()
        .compile()
        .unwrap();
    let cases = [
        json!([]),
        json!([{"role":"user","content":"Hello 世界 e\u{301}\0\n"}]),
        json!([{"role":"system","content":"Stay precise"},{"role":"user","content":"<|im_end|> really?"},{"role":"assistant","content":"Yes"}]),
    ];
    for input in cases {
        let messages = input.as_array().unwrap();
        let plan = source
            .render_plan(ChatMessages::from_json(messages).unwrap())
            .unwrap();
        assert!(plan.is_for(&source));
        assert!(
            !plan.is_for(&other),
            "equal text is not source object identity"
        );
        let rendered = plan.render().unwrap();
        for generation in [false, true] {
            let expected = ordinary()
                .apply_chat_template_json(
                    model_template.clone(),
                    [messages.clone()],
                    None,
                    "smol",
                    generation,
                    None,
                )
                .unwrap();
            assert_eq!(rendered.prompt(generation), expected[0]);
        }
        assert_eq!(rendered.generation_suffix(), "<|im_start|>assistant\n");
        assert_eq!(
            rendered.prompt(true),
            format!("{}{}", rendered.prompt(false), rendered.generation_suffix())
        );
    }
}

#[test]
fn decoded_named_selection_and_invalid_metadata_follow_ordinary_contracts() {
    let single = config(json!(vm::supported_source()));
    let escaped = single.replace("chat_template", "chat_\\u0074emplate");
    let named = config(json!([
        {"name":"tool_use", "template":"unselected ordinary source"},
        {"name":"default", "template":vm::supported_source()},
    ]));
    for (input, model, expected_name) in [
        (single.as_str(), "chat", "chat"),
        (escaped.as_str(), "chat", "chat"),
        (
            named.as_str(),
            "chat.html",
            "chat.html::chat_template::default",
        ),
    ] {
        let actual = ChatTemplatePlan::prepare_config(input.as_bytes(), model, false)
            .unwrap()
            .compile()
            .unwrap();
        let selected = load_model_chat_template_from_str(input).unwrap().unwrap();
        assert_eq!(
            selected.select(None).unwrap().template(),
            vm::supported_source()
        );
        assert_eq!(actual.name(), expected_name);
    }
    for metadata in [
        json!([]),
        json!([{"name":"", "template":vm::supported_source()}]),
        json!([{"name":"default", "template":vm::supported_source(), "ignored":true}]),
        json!([{"name":"default", "template":vm::supported_source()}, {"name":"default", "template":vm::supported_source()}]),
        json!(42),
    ] {
        let input = config(metadata);
        assert!(load_model_chat_template_from_str(&input).is_err());
        assert!(
            match ChatTemplatePlan::prepare_config(input.as_bytes(), "chat", false) {
                Ok(plan) => plan.compile().is_err(),
                Err(_) => true,
            }
        );
    }
    let missing = config(json!([{"name":"tool_use", "template":vm::supported_source()}]));
    assert!(
        load_model_chat_template_from_str(&missing)
            .unwrap()
            .unwrap()
            .select(None)
            .is_err()
    );
    assert!(matches!(
        ChatTemplatePlan::prepare_config(missing.as_bytes(), "chat", false),
        Err(ChatSourceError::MissingTemplate)
    ));
    let different = config(json!(format!("{} ", vm::supported_source())));
    assert!(load_model_chat_template_from_str(&different).is_ok());
    let different_source = ChatTemplatePlan::prepare_config(different.as_bytes(), "chat", false)
        .unwrap()
        .compile()
        .unwrap();
    assert_eq!(
        different_source.inner.source(),
        format!("{} ", vm::supported_source())
    );
    assert!(matches!(
        ChatTemplatePlan::prepare_config(single.as_bytes(), "chat.html", false),
        Err(ChatSourceError::Source(vm::SourceError::Settings))
    ));
}

#[test]
fn numeric_config_uses_ordinary_scalar_semantics_after_source_planning() {
    let source = serde_json::to_string(vm::supported_source()).unwrap();
    for number in [
        "0",
        "-0",
        "-9223372036854775808",
        "18446744073709551615",
        "1.25",
        "-2e-3",
        // Full mantissa reconstruction, halfway rounding and exponent edges.
        "1.00000000000000011102230246251565404236316680908203125",
        "1.00000000000000011102230246251565404236316680908203126",
        "2.2250738585072012595828020632615924102173170860556483202493214926802275770343682699968542685797460828291415522517e-308",
        "4.940656458412465441765687928682213723650598026143247644255856825e-324",
        "18446744073709551616",
        "-9223372036854775809",
        "1e-999",
        "0e999999999999999999999999999",
        "-0e999999999999999999999999999",
        "1.7976931348623157e308",
        "1.7976931348623159e308",
        "1e309",
        "1e999",
    ] {
        let input = format!(
            "{{\n  \"unused\": [0, {{\"nested\":{number}}}],\n \"chat_template\":{source}}}"
        );
        let expected = serde_json::from_str::<serde_json::Value>(&input);
        let plan = ChatTemplatePlan::prepare_config(input.as_bytes(), "chat", false).unwrap();
        let required = plan.requirements().buffer_bytes();
        match expected {
            Ok(_) => {
                let expected = serde_json::from_str::<serde_json::Number>(number).unwrap();
                let scalar = serde_json::bounded_number::Plan::prepare(number, 0..number.len())
                    .unwrap()
                    .parse()
                    .unwrap();
                assert_eq!(
                    scalar.as_f64().unwrap().to_bits(),
                    expected.as_f64().unwrap().to_bits(),
                    "{number}"
                );
                assert!(load_model_chat_template_from_str(&input).is_ok());
                let template = plan.compile().unwrap();
                assert!(template.retained_buffer_bytes() <= required);
                assert_eq!(template.inner.source(), vm::supported_source());
            }
            Err(expected) => {
                assert!(load_model_chat_template_from_str(&input).is_err());
                let failure = plan.compile().unwrap_err();
                let actual = failure.numeric_error().unwrap().error();
                assert_eq!(actual.classify(), expected.classify(), "{number}");
                assert_eq!(
                    (actual.line(), actual.column()),
                    (expected.line(), expected.column()),
                    "{number}"
                );
                assert_eq!(actual.to_string(), expected.to_string(), "{number}");
                assert!(failure.retained_buffer_bytes() > 0);
                assert!(failure.retained_buffer_bytes() <= required);
                assert!(failure.source().is_some());
            }
        }
    }
    // A numeric-looking key or string never enters the numeric scalar parser.
    let strings = format!("{{\"1e999\":\"-9223372036854775809\",\"chat_template\":{source}}}");
    assert!(
        ChatTemplatePlan::prepare_config(strings.as_bytes(), "chat", false)
            .unwrap()
            .compile()
            .is_ok()
    );
    // Numeric errors retain ordinary precedence over template selection errors.
    for input in [
        r#"{"unused":1e999}"#,
        r#"{"unused":1e999,"chat_template":42}"#,
    ] {
        let failure = ChatTemplatePlan::prepare_config(input.as_bytes(), "chat", false)
            .unwrap()
            .compile()
            .unwrap_err();
        assert!(failure.numeric_error().is_some());
    }
    let failure =
        ChatTemplatePlan::prepare_config(br#"{"unused":1.25,"chat_template":42}"#, "chat", false)
            .unwrap()
            .compile()
            .unwrap_err();
    assert!(matches!(
        failure.source_error(),
        Some(ChatSourceError::TemplateProfile)
    ));
    let invalid = format!("{{\"unused\":01,\"chat_template\":{source}}}");
    assert!(matches!(
        ChatTemplatePlan::prepare_config(invalid.as_bytes(), "chat", false),
        Err(ChatSourceError::Json(_))
    ));
}

#[cfg(feature = "tokenizer-compiler-test-support")]
#[test]
fn text_adapters_retain_actual_source_and_render_reserve_prefix_errors() {
    for (index, buffer) in [
        ChatSourceBuffer::Instructions,
        ChatSourceBuffer::Bytes,
        ChatSourceBuffer::Locations,
    ]
    .into_iter()
    .enumerate()
    {
        let error = ChatTemplatePlan::prepare_config(RELEASED.as_bytes(), "chat", false)
            .unwrap()
            .fail_reservation(buffer)
            .compile()
            .unwrap_err();
        assert_eq!(error.retained_buffer_bytes() > 0, index > 0);
        assert!(error.source().is_some());
        assert!(
            matches!(error.cause(), Some(vm::CompileCause::Reserve(actual, _)) if *actual == buffer)
        );
    }
    let source = ChatTemplatePlan::prepare_config(RELEASED.as_bytes(), "chat", false)
        .unwrap()
        .compile()
        .unwrap();
    let messages = [TextMessage {
        role: "user",
        content: "nonempty 世界",
    }];
    for (index, buffer) in [
        ChatRenderBuffer::Operands,
        ChatRenderBuffer::Frames,
        ChatRenderBuffer::Locals,
        ChatRenderBuffer::Concat,
        ChatRenderBuffer::WithoutPrompt,
        ChatRenderBuffer::WithPrompt,
    ]
    .into_iter()
    .enumerate()
    {
        let error = source
            .render_plan(ChatMessages::from_text(&messages))
            .unwrap()
            .fail_reservation(buffer)
            .render()
            .unwrap_err();
        assert_eq!(error.retained_buffer_bytes() > 0, index > 0);
        assert!(error.source().is_some());
        assert!(matches!(error.cause(), vm::RenderCause::Reserve(actual, _) if *actual == buffer));
    }
}

#[test]
fn generic_source_compilation_matches_ordinary_selected_text_and_owns_decoded_source() {
    let sources = [
        "{{ messages | length }}",
        "{% for a in messages %}{% for b in messages %}{{ b.content }}{% endfor %}{% endfor %}",
        "{% for m in messages %}{{ m.content }}{% endfor %}{% if add_generation_prompt %} word3 word7{% endif %}",
        "前\n{%- for entry in messages %}{% if loop.first and entry.role != 'system' %}開始{% else %}|{% endif %}{{ entry['role'] + ':' + entry.content }}{% endfor %}{% if add_generation_prompt %}答:{% endif %}",
        "{% for a in messages %}{{ a.role }}{% endfor %}/{% for b in messages %}{{ b.content }}{% endfor %}{% if add_generation_prompt %}> {% endif %}",
    ];
    for text in sources {
        let encoded = config(json!([{"name":"default","template":text}]));
        let selected = load_model_chat_template_from_str(&encoded)
            .unwrap()
            .unwrap();
        let plan = ChatTemplatePlan::prepare_config(encoded.as_bytes(), "general", false).unwrap();
        let required = plan.requirements();
        let source = plan.compile().unwrap();
        drop(encoded);
        assert_eq!(source.inner.source(), text);
        assert_eq!(source.name(), "general::chat_template::default");
        assert!(source.retained_buffer_bytes() <= required.buffer_bytes());
        assert!(source.accepts_default_variables(&serde_json::Map::new()));
        let equal_source = ChatTemplatePlan::prepare_utf8(text, "general")
            .unwrap()
            .compile()
            .unwrap();
        for input in [
            json!([]),
            json!([{"role":"user","content":"e\u{301} 世界\0"}]),
            json!([{"role":"system","content":"rules"},{"role":"assistant","content":"answer"}]),
        ] {
            let messages = input.as_array().unwrap();
            let plan = source
                .render_plan(ChatMessages::from_json(messages).unwrap())
                .unwrap();
            assert!(plan.is_for(&source));
            assert!(!plan.is_for(&equal_source));
            let output = plan.render().unwrap();
            for generation in [false, true] {
                let expected = ordinary()
                    .apply_chat_template_json(
                        selected.clone(),
                        [messages.clone()],
                        None,
                        "general",
                        generation,
                        None,
                    )
                    .unwrap();
                assert_eq!(output.prompt(generation), expected[0]);
            }
        }
    }
    let invalid = ChatTemplatePlan::prepare_utf8("{% if %}", "general")
        .unwrap()
        .compile()
        .unwrap_err();
    assert!(matches!(
        invalid.cause(),
        Some(vm::CompileCause::Source(vm::SourceError::Syntax))
    ));
}

#[cfg(feature = "tokenizer-compiler-test-support")]
#[test]
fn generic_source_reserve_failure_retains_its_real_decoded_and_output_prefix() {
    let text = "{% for m in messages %}{{ m.content }}{% endfor %}";
    for buffer in [ChatSourceBuffer::Instructions, ChatSourceBuffer::Locations] {
        let error = ChatTemplatePlan::prepare_utf8(text, "general")
            .unwrap()
            .fail_reservation(buffer)
            .compile()
            .unwrap_err();
        assert!(error.retained_buffer_bytes() >= text.len());
        assert!(
            matches!(error.cause(),Some(vm::CompileCause::Reserve(actual,_)) if *actual==buffer)
        );
    }
}

#[test]
fn source_trim_filter_matches_ordinary_unicode_and_concatenated_text() {
    // Generic shapes used by released chat sources, including a filter before
    // and after concatenation. No complete template is selected by its text.
    let sources = [
        "{{ 'x'|upper }}",
        "{% for message in messages %}{{ message['content']|trim }}{% endfor %}",
        "{% for message in messages %}{{ ' ' + message['content']|trim + '\\n' }}{% endfor %}",
        "{% for message in messages %}{{ ('\\t' + message['content'] + '\\n')|trim|trim }}{% endfor %}",
        "{% for message in messages %}{{ '[' + (message['content'] + ' ')|trim + ']' }}{% endfor %}{% if add_generation_prompt %}{{ ' next '|trim }}{% endif %}",
    ];
    let contents = [
        "",
        " \t\n",
        "\u{2003}\u{a0}é世界\u{2002}",
        " \0 x \0 ",
        "e\u{301}",
        "\u{200b} x \u{200b}",
    ];
    for source in sources {
        let plan = ChatTemplatePlan::prepare_utf8(source, "trim").unwrap();
        let required = plan.requirements().buffer_bytes();
        let template = plan.compile().unwrap();
        assert!(template.retained_buffer_bytes() <= required);
        for content in contents {
            let messages = vec![
                json!({"role":"user", "content":content}),
                json!({"role":"assistant", "content":"  tail  "}),
            ];
            let plan = template
                .render_plan(ChatMessages::from_json(&messages).unwrap())
                .unwrap();
            let required = plan.requirements().buffer_bytes();
            let rendered = plan.render().unwrap();
            assert!(rendered.retained_buffer_bytes() <= required);
            for generation in [false, true] {
                let expected = ordinary()
                    .apply_chat_template_json(
                        crate::tokenizer::ModelChatTemplate::Single(source.into()),
                        [messages.clone()],
                        None,
                        "trim",
                        generation,
                        None,
                    )
                    .unwrap();
                assert_eq!(
                    rendered.prompt(generation),
                    expected[0],
                    "{source:?} {content:?}"
                );
            }
        }
    }
    // Character trimming of a concatenation still requires a source profile.
    for source in ["{{ ('x' + 'x')|trim('x') }}"] {
        let error = ChatTemplatePlan::prepare_utf8(source, "trim")
            .unwrap()
            .compile()
            .unwrap_err();
        assert!(matches!(
            error.cause(),
            Some(vm::CompileCause::Source(vm::SourceError::Profile))
        ));
    }
}

#[test]
fn source_trim_character_arguments_share_ordinary_set_semantics_and_bounds() {
    let sources = [
        "{{ '1212foo12bar1212'|trim('12') }}",
        "{% for message in messages %}{{ message.content|trim('é\\u2003') }}{% endfor %}",
        "{% for message in messages %}{{ message['content']|trim('') }}{% endfor %}",
        "{% for message in messages %}{{ message.content|trim(message.role) }}{% endfor %}",
        "{% for message in messages %}{{ '[' + message.content|trim('xyé') + ']' }}{% endfor %}",
        "{% for message in messages %}{{ message.content|trim|trim('xyé')|trim }}{% endfor %}",
        "{% for message in messages %}{{ message.content|trim('xyé')|trim('x') }}{% endfor %}",
    ];
    let contents = [
        "",
        "xyxyé",
        "xyfooxybaré",
        "é世界é",
        "\u{2003}é hi é\u{2003}",
        "e\u{301}",
        "\0xy\0",
    ];
    for source in sources {
        let plan = ChatTemplatePlan::prepare_utf8(source, "trim-characters").unwrap();
        let required = plan.requirements().buffer_bytes();
        let template = plan.compile().unwrap();
        assert!(template.retained_buffer_bytes() <= required);
        for content in contents {
            let messages = vec![
                json!({"role":"user", "content":content}),
                json!({"role":"assistant", "content":"a test a"}),
            ];
            let plan = template
                .render_plan(ChatMessages::from_json(&messages).unwrap())
                .unwrap();
            let required = plan.requirements().buffer_bytes();
            let rendered = plan.render().unwrap();
            assert!(rendered.retained_buffer_bytes() <= required);
            for generation in [false, true] {
                let expected = ordinary()
                    .apply_chat_template_json(
                        crate::tokenizer::ModelChatTemplate::Single(source.into()),
                        [messages.clone()],
                        None,
                        "trim-characters",
                        generation,
                        None,
                    )
                    .unwrap();
                assert_eq!(
                    rendered.prompt(generation),
                    expected[0],
                    "{source:?} {content:?}"
                );
            }
        }
    }
    for source in [
        "{{ ('x' + 'y')|trim('xy') }}",
        "{{ 'xy'|trim('x' + 'y') }}",
        "{{ 'xy'|trim('x', 'y') }}",
        "{{ 'xy'|trim(chars='xy') }}",
    ] {
        let error = ChatTemplatePlan::prepare_utf8(source, "trim-characters")
            .unwrap()
            .compile()
            .unwrap_err();
        assert!(matches!(
            error.cause(),
            Some(vm::CompileCause::Source(vm::SourceError::Profile))
        ));
    }
}

#[test]
fn shared_filter_dispatch_keeps_ordinary_registration_arguments_and_errors() {
    let mut env = minijinja::Environment::new();
    env.add_filter("trim", |value: String| format!("custom:{value}"));
    env.add_template("custom", "{{ ' x '|trim }}{{ ' y '|trim }}")
        .unwrap();
    assert_eq!(
        env.get_template("custom").unwrap().render(()).unwrap(),
        "custom: x custom: y "
    );
    let mut env = minijinja::Environment::new();
    env.add_template("args", "{{ '1212foo12bar1212'|trim('12') }}")
        .unwrap();
    assert_eq!(
        env.get_template("args").unwrap().render(()).unwrap(),
        "foo12bar"
    );
    env.add_template("missing", "{{ 'x'|missing_filter }}")
        .unwrap();
    assert_eq!(
        env.get_template("missing")
            .unwrap()
            .render(())
            .unwrap_err()
            .kind(),
        minijinja::ErrorKind::UnknownFilter
    );
}

mod context;
