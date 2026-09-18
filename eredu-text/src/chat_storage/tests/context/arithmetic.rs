use super::*;

#[test]
fn released_mistral_role_alternation_uses_exact_numeric_modulo_and_shared_namespace() {
    // Byte-exact released Mistral role-check block; only the input alias and
    // output probe are test setup. Tool rows remain excluded by that block.
    let template = concat!(
        "{% set loop_messages=messages %}",
        include_str!("arithmetic_mistral.jinja"),
        "{{ns.index}}|{% for row in messages %}{{row.content}};{% endfor %}"
    );
    let rows = json!([
        {"role":"user","content":"é"},
        {"role":"assistant","content":"界"},
        {"role":"tool","content":"🦀\u{0}"},
        {"role":"assistant","content":"call","tool_calls":[]},
        {"role":"user","content":"again"},
        {"role":"assistant","content":"done"}
    ]);
    let output = compare(
        template,
        rows.as_array().unwrap(),
        &serde_json::Map::new(),
        &serde_json::Map::new(),
    );
    drop(rows);
    assert_eq!(output.prompt(false), "\n4|é;界;🦀\u{0};call;again;done;");
    let invalid = json!([{"role":"user","content":"first"},{"role":"user","content":"second"}]);
    let source = ChatTemplatePlan::prepare_utf8(template, "role-check")
        .unwrap()
        .compile()
        .unwrap();
    assert!(
        source
            .render_plan_with_context(
                ChatRenderContext::from_json(invalid.as_array().unwrap(), None, None).unwrap()
            )
            .is_err()
    );
    assert!(
        ordinary()
            .apply_chat_template_json(
                crate::tokenizer::ModelChatTemplate::Single(template.into()),
                [invalid.as_array().unwrap().to_vec()],
                None,
                "role-check",
                false,
                None
            )
            .is_err()
    );
}

#[test]
fn scalar_arithmetic_preserves_signed_float_boolean_and_checked_power_results() {
    let template = "{{a-b}}|{{a*b}}|{{a/b}}|{{a//b}}|{{a%b}}|{{a**exponent}}|{{true*b}}|{{ -(a-b) }}";
    let caller = json!({"a":-7,"b":3,"exponent":2});
    let output = compare(
        template,
        &[],
        &serde_json::Map::new(),
        caller.as_object().unwrap(),
    );
    assert_eq!(
        output.prompt(false),
        "-10|-21|-2.3333333333333335|-3|2|49|3|10"
    );
    for caller in [
        json!({"a":7,"b":-3,"exponent":2}),
        json!({"a":-7.5,"b":2.0,"exponent":2.0}),
        json!({"a":0.0,"b":-3.0,"exponent":2.0}),
        json!({"a":-0.0,"b":3.0,"exponent":2.0}),
        json!({"a":i64::MIN,"b":3,"exponent":1}),
        json!({"a":u64::MAX,"b":3,"exponent":1}),
    ] {
        compare(
            template,
            &[],
            &serde_json::Map::new(),
            caller.as_object().unwrap(),
        );
    }
    // Floating zero division retains the existing ordinary IEEE result,
    // independently of checked integer floor-division/remainder refusal.
    let caller = json!({"a":1.0,"b":0.0});
    compare(
        "{{a/b}}|{{a//b}}|{{a%b}}",
        &[],
        &serde_json::Map::new(),
        caller.as_object().unwrap(),
    );
}

#[test]
fn arithmetic_errors_and_partial_render_destinations_keep_original_custody() {
    for (template, caller) in [
        ("{{a//b}}", json!({"a":7,"b":0})),
        ("{{a%b}}", json!({"a":7,"b":0})),
        ("{{a*b}}", json!({"a":u64::MAX,"b":u64::MAX})),
        ("{{a**b}}", json!({"a":7,"b":-1})),
        ("{{a-b}}", json!({"a":null,"b":1})),
    ] {
        let source = ChatTemplatePlan::prepare_utf8(template, "numeric-refusal")
            .unwrap()
            .compile()
            .unwrap();
        assert!(
            source
                .render_plan_with_context(
                    ChatRenderContext::from_json(&[], None, caller.as_object()).unwrap()
                )
                .is_err()
        );
        assert!(
            ordinary()
                .apply_chat_template_json(
                    crate::tokenizer::ModelChatTemplate::Single(template.into()),
                    [Vec::<serde_json::Value>::new()],
                    None,
                    "numeric-refusal",
                    false,
                    caller.as_object()
                )
                .is_err()
        );
    }
    let template = "{{prefix}}:{{a*b}}/{{a%b}}";
    let source = ChatTemplatePlan::prepare_utf8(template, "numeric-custody")
        .unwrap()
        .compile()
        .unwrap();
    let caller = json!({"prefix":"é\u{0}界🦀","a":37,"b":11});
    let plan = source
        .render_plan_with_context(
            ChatRenderContext::from_json(&[], None, caller.as_object()).unwrap(),
        )
        .unwrap();
    let failure = plan
        .fail_reservation(ChatRenderBuffer::WithPrompt)
        .render()
        .unwrap_err();
    drop((source, caller));
    assert!(failure.retained_buffer_bytes() > 0);

    // Dispatch preserves ordinary dynamic multiplication; it grants
    // no repeated-string storage to the closed scalar producer.
    let template = "{{value*count}}";
    let caller = json!({"value":"é界","count":3});
    assert_eq!(
        ordinary()
            .apply_chat_template_json(
                crate::tokenizer::ModelChatTemplate::Single(template.into()),
                [Vec::<serde_json::Value>::new()],
                None,
                "ordinary-repeat",
                false,
                caller.as_object()
            )
            .unwrap()[0],
        "é界é界é界"
    );
    let source = ChatTemplatePlan::prepare_utf8(template, "ordinary-repeat")
        .unwrap()
        .compile()
        .unwrap();
    assert!(
        source
            .render_plan_with_context(
                ChatRenderContext::from_json(&[], None, caller.as_object()).unwrap()
            )
            .is_err()
    );
}
