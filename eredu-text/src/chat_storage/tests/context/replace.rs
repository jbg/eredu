use super::*;

#[test]
fn borrowed_replace_preserves_unicode_empty_patterns_registers_and_retained_results() {
    let template = r#"{{ 'aaa'|replace('aa', '界') }}|{{ ''|replace('', '·') }}|{{ cfg.value|replace(cfg.from, cfg.to) }}|{{ cfg.value|replace('', cfg.to) }}|{{ cfg.value|replace(cfg.from, '') }}|{{ cfg.value|replace('absent', 'wrong') }}|{% for message in messages %}{{ message.content|replace(cfg.from, cfg.to) }}:{{ message.content|replace('', '·')|length }}:{{ (message.content|replace(cfg.from, cfg.to) + cfg.suffix)|trim }};{% endfor %}{{ cfg.blank|replace('x', '')|trim|count }}|{{ cfg.spaced|trim|replace(cfg.from, cfg.to)|trim }}|{% if cfg.empty|replace('', '') %}wrong{% else %}empty{% endif %}|{{ cfg.empty|replace('', '')|default('fallback', true) }}{% if add_generation_prompt %}|{{ cfg.tail|replace(cfg.from, cfg.to) }}{% endif %}"#;
    let defaults = json!({"cfg":{"value":"wrong default"}});
    let caller = json!({"cfg":{
        "value":"é\u{301}界\u{0}🦀é", "from":"é", "to":"終\u{0}",
        "suffix":"\u{2003}", "blank":" \t x\u{2003}",
        "empty":"", "spaced":"\u{2003} é \t", "tail":"é🦀"
    }});
    let messages = [
        json!({"role":"user","content":"é\u{0}🦀"}),
        json!({"role":"assistant","content":"  第二é "}),
    ];
    let rendered = compare(
        template,
        &messages,
        defaults.as_object().unwrap(),
        caller.as_object().unwrap(),
    );
    assert!(
        rendered
            .prompt(false)
            .starts_with("界a|·|終\u{0}\u{301}界\u{0}🦀終\u{0}|")
    );
    assert!(rendered.prompt(false).ends_with("|empty|fallback"));
    assert!(rendered.prompt(true).ends_with("|終\u{0}🦀"));
    let source = ChatTemplatePlan::prepare_utf8(template, "replace")
        .unwrap()
        .compile()
        .unwrap();
    let plan = source
        .render_plan_with_context(
            ChatRenderContext::from_json(&messages, defaults.as_object(), caller.as_object())
                .unwrap(),
        )
        .unwrap();
    assert!(rendered.retained_buffer_bytes() <= plan.requirements().buffer_bytes());
    let failure = plan
        .fail_reservation(ChatRenderBuffer::WithPrompt)
        .render()
        .unwrap_err();
    drop((messages, defaults, caller, source));
    assert!(failure.retained_buffer_bytes() > 0);
    assert!(rendered.prompt(false).contains("第二終\u{0}"));
    drop((rendered, failure));
}

#[test]
fn replace_matches_dynamic_coercion_and_argument_errors() {
    // Arbitrary scalar/container string coercion remains a distinct producer.
    for value in [json!(17), json!(null), json!(["x"]), json!({"x":"y"})] {
        for template in [
            "{{ value|replace('x', 'y') }}",
            "{{ 'x'|replace(value, 'y') }}",
            "{{ 'x'|replace('x', value) }}",
        ] {
            let caller = json!({"value":value});
            compare_outcome(template, caller.as_object().unwrap());
        }
    }
    for template in [
        "{{ 'x'|replace }}",
        "{{ 'x'|replace('x') }}",
        "{{ 'x'|replace('x', 'y', 1) }}",
        "{{ 'x'|replace(from='x', to='y') }}",
    ] {
        compare_outcome(template, &serde_json::Map::new());
    }
}

#[test]
fn nested_generated_filter_inputs_use_the_existing_paid_text_materializer() {
    let caller = json!({"left":"É","right":"界🙂","from":"界","to":"é"});
    for template in [
        "{{ ('x' + 'y')|replace('x', 'z') }}",
        "{{ 'x'|replace('x'|join, 'y') }}",
        "{{ 'x'|replace('x', 'y'|join) }}",
        "{{ 'x'|replace('x', 'y')|replace('y', 'z') }}",
        "{{ (left+right)|replace(from,to)|replace('🙂','!') }}",
        "{{ [left,right]|join(from+to) }}",
    ] {
        compare(
            template,
            &[],
            &serde_json::Map::new(),
            caller.as_object().unwrap(),
        );
    }
}
