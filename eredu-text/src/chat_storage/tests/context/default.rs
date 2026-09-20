use super::*;

#[test]
fn defaults_preserve_eager_arguments_truth_modes_borrowed_selection_and_retirement() {
    let template = r#"{{ absent|default }}|{{ absent|d([])|length }}|{{ (cfg.missing|default(cfg.fallback)).text + ':' + (cfg.primary|d(cfg.fallback)).text }}|{% for message in messages %}{{ (cfg.empty|default(cfg.fallback.text,cfg.flag)) + message.content + (cfg.missing|d(cfg.other)).text }};{% endfor %}{{ cfg.none|default('wrong') }}|{{ cfg.zero|default('wrong') }}|{{ cfg.empty|default('wrong') }}{% if add_generation_prompt %}{{ absent|default(cfg.fallback.text) }}{% endif %}"#;
    let caller = json!({"cfg":{"fallback":{"text":"fallback 界"},"primary":{"text":"primary"},"other":{"text":"other"},"empty":"","none":null,"zero":0,"flag":"truthy"}});
    let base = [
        json!({"role":"user","content":"a"}),
        json!({"role":"assistant","content":"b"}),
    ];
    let rendered = compare(
        template,
        &base,
        &serde_json::Map::new(),
        caller.as_object().unwrap(),
    );
    assert!(
        rendered
            .prompt(false)
            .starts_with("|0|fallback 界:primary|fallback 界aother;fallback 界bother;")
    );
    assert!(rendered.prompt(true).ends_with("fallback 界"));
    let source = ChatTemplatePlan::prepare_utf8(template, "defaults")
        .unwrap()
        .compile()
        .unwrap();
    let plan = source
        .render_plan_with_context(
            ChatRenderContext::from_json(&base, None, caller.as_object()).unwrap(),
        )
        .unwrap();
    let failure = plan
        .fail_reservation(ChatRenderBuffer::WithPrompt)
        .render()
        .unwrap_err();
    drop((base, caller, source));
    assert!(failure.retained_buffer_bytes() > 0);
    assert!(rendered.prompt(true).ends_with("fallback 界"));
    drop((failure, rendered));

    for flag in [
        json!(null),
        json!(false),
        json!(true),
        json!(0),
        json!(-0.0),
        json!(2),
        json!(""),
        json!("yes"),
        json!([]),
        json!([0]),
        json!({}),
        json!({"key":false}),
    ] {
        let caller = json!({"flag":flag,"empty":"","none":null,"zero":0,"truthy":"kept"});
        compare(
            "{{ empty|default('fallback',flag) }}:{{ none|d('fallback',flag) }}:{{ zero|default('fallback',flag) }}:{{ truthy|default('fallback',flag) }}:{{ absent|d('fallback',flag) }}",
            &[],
            &serde_json::Map::new(),
            caller.as_object().unwrap(),
        );
    }
    compare(
        "{{ ''|default('fallback',undefined_flag) }}|{{ absent|d('fallback',undefined_flag) }}",
        &[],
        &serde_json::Map::new(),
        &serde_json::Map::new(),
    );
    // Arguments remain eager even when the primary value is selected.
    let template = "{{ 'kept'|default(missing.child) }}";
    let source = ChatTemplatePlan::prepare_utf8(template, "defaults")
        .unwrap()
        .compile()
        .unwrap();
    assert!(
        source
            .render_plan_with_context(ChatRenderContext::from_json(&[], None, None).unwrap())
            .unwrap()
            .render()
            .is_err()
    );
    assert!(
        ordinary()
            .apply_chat_template_json(
                crate::tokenizer::ModelChatTemplate::Single(template.into()),
                [vec![]],
                None,
                "defaults",
                false,
                None
            )
            .is_err()
    );
}
