use super::*;

#[test]
fn borrowed_join_preserves_scalar_spelling_unicode_registers_and_retained_results() {
    let template = r#"{{ cfg.types|join(' | ') }}|{{ cfg.scalars|join(cfg.separator) }}|{{ cfg.empty|join }}|{{ absent|join }}|{{ none|join }}|{{ []|join('unused') }}|{% for message in messages %}{{ (cfg.left|join + message.content + cfg.right|join)|trim }}:{{ (cfg.left|join + message.content + cfg.right|join)|trim|length }}:{{ message.content|join('·') }};{% endfor %}{{ (cfg.blank|join + cfg.blank|join)|trim|count }}|{{ (cfg.blank|join + cfg.right|join)|trim }}{% if add_generation_prompt %}|{{ cfg.tail|join }}{% endif %}"#;
    let defaults = json!({"cfg":{"types":["wrong default"]}});
    let caller = json!({"cfg":{
        "types":["string","integer","null"],
        "scalars":["界",true,false,null,0,-9223372036854775808i64,u64::MAX,-0.0,1.0,1.25,f64::MIN_POSITIVE,f64::MAX,f64::from_bits(1)],
        "separator":"‖", "empty":[], "left":[" \t", "é\u{301}"],
        "right":["終", "\u{2003}"], "blank":["", " \t", "\u{2003}"],
        "tail":["a", "\u{0}", "🦀"]
    }});
    let messages = [
        json!({"role":"user","content":"a\u{0}🦀"}),
        json!({"role":"assistant","content":"  第二 "}),
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
            .starts_with("string | integer | null|界‖True‖False‖None‖0‖")
    );
    assert!(rendered.prompt(true).ends_with("|a\u{0}🦀"));
    let source = ChatTemplatePlan::prepare_utf8(template, "join")
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
    assert!(rendered.prompt(false).contains("a·\u{0}·🦀"));
    drop((rendered, failure));

    for (value, expected) in [
        (json!(["prefix", [1, 2]]), "prefix,[1, 2]"),
        (json!(["prefix", {"x":1}]), "prefix,{\"x\": 1}"),
        (json!({"x":1}), "x"),
    ] {
        let caller = json!({"value":value});
        let rendered = compare(
            "{{ value|join(',') }}",
            &[],
            &serde_json::Map::new(),
            caller.as_object().unwrap(),
        );
        assert_eq!(rendered.prompt(false), expected);
    }
}
