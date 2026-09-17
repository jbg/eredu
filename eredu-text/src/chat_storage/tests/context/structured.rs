use super::*;

#[test]
fn nested_context_selection_conditions_and_register_reuse_match_ordinary() {
    let base = json!([{"role":"user","content":"first"},{"role":"assistant","content":"second"}]);
    let defaults = json!({"cfg":{"ignored":"replaced whole object"}});
    let caller = json!({"cfg":{"flags":{"enabled":true,"disabled":false},"tokens":["  α\u{0}  ","[β]"],"chars":"[]", "empty":null,"entries":[{"name":"entry0"},{"name":"entry1"}]},"which":1,"absent_base":null});
    let template = r#"{% if cfg is defined and cfg.flags.enabled and not cfg.flags.disabled %}{{ cfg.tokens[-1] + ':' + cfg.tokens[0]|trim }}{% endif %}|{{ cfg.tokens[-1]|trim(cfg.chars) }}|{{ cfg.entries[which].name }}|{{ cfg.missing is undefined }}|{{ cfg.empty is none }}|{{ cfg != none }}|{{ cfg.tokens[which] == '[β]' }}|{{ cfg.flags.enabled == true }}|{{ cfg.entries[99] is not defined }}|{{ absent_base.child is undefined }}|{{ messages[-1].content }}|{% for message in messages %}{{ cfg.tokens[0] + message.content + cfg.tokens[1] }};{% endfor %}{% if add_generation_prompt %}{{ cfg.tokens[-1] + cfg.tokens[0] }}{% endif %}"#;
    let rendered = compare(
        template,
        base.as_array().unwrap(),
        defaults.as_object().unwrap(),
        caller.as_object().unwrap(),
    );
    assert!(rendered.prompt(false).contains("[β]:α\u{0}"));
    assert!(
        rendered
            .prompt(false)
            .contains("entry1|True|True|True|True|True|True|True|second")
    );
    drop((base, defaults, caller));
    assert!(rendered.prompt(true).ends_with("[β]  α\u{0}  "));

    // Each selection replaces one borrowed operand. JSON nesting does not
    // require a recursive renderer stack, a path Vec or a new nesting limit.
    let mut nested = json!({"leaf":"深い"});
    let mut template = "{{ root".to_owned();
    for _ in 0..40 {
        nested = json!({"child":nested});
        template.push_str(".child");
    }
    template.push_str(".leaf + '!' }}");
    let caller = json!({"root":nested});
    compare(
        &template,
        &[],
        &serde_json::Map::new(),
        caller.as_object().unwrap(),
    );
}

#[test]
fn borrowed_selection_missing_and_index_conversion_match_ordinary() {
    let caller = json!({"values":[10,20,30],"map":{"present":false},"invalid":"bad"});
    for index in [
        json!(-4),
        json!(-3),
        json!(-1),
        json!(0),
        json!(2),
        json!(3),
        json!(true),
        json!(1.0),
        json!(1.5),
        json!(u64::MAX),
        json!(null),
        json!("1"),
    ] {
        let mut caller = caller.as_object().unwrap().clone();
        caller.insert("index".into(), index);
        compare(
            "{{ values[index] }}|{{ values[index] is defined }}",
            &[],
            &serde_json::Map::new(),
            &caller,
        );
    }
    compare(
        "{{ map.absent }}|{{ map['absent'] is undefined }}|{{ map[0] is undefined }}|{{ map.present == false }}",
        &[],
        &serde_json::Map::new(),
        caller.as_object().unwrap(),
    );
    for template in ["{{ map.absent.child }}", "{{ missing.child }}"] {
        let source = ChatTemplatePlan::prepare_utf8(template, "context")
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
        let mut ordinary = ordinary();
        assert!(
            ordinary
                .apply_chat_template_json(
                    crate::tokenizer::ModelChatTemplate::Single(template.into()),
                    [vec![]],
                    None,
                    "context",
                    false,
                    caller.as_object()
                )
                .is_err()
        );
    }
}

#[test]
fn selected_text_storage_is_measured_and_all_error_prefixes_outlive_the_context() {
    let source = ChatTemplatePlan::prepare_utf8(
        "{{ cfg.a + cfg.b }}{% if add_generation_prompt %}{{ cfg.a + cfg.a }}{% endif %}",
        "context",
    )
    .unwrap()
    .compile()
    .unwrap();
    let caller = json!({"cfg":{"a":"é","b":"非空"}});
    let context = ChatRenderContext::from_json(&[], None, caller.as_object()).unwrap();
    let inner = vm::RenderPlan::prepare_context(&source.inner, context).unwrap();
    assert_eq!(
        inner.requirements().capacity(ChatRenderBuffer::ContextText),
        "é非空éé".len()
    );
    let rendered = source
        .render_plan_with_context(context)
        .unwrap()
        .render()
        .unwrap();
    assert_eq!(rendered.prompt(false), "é非空");
    assert_eq!(rendered.prompt(true), "é非空éé");
    let failure = source
        .render_plan_with_context(context)
        .unwrap()
        .fail_reservation(ChatRenderBuffer::ContextText)
        .render()
        .unwrap_err();
    drop(caller);
    assert_eq!(rendered.prompt(true), "é非空éé");
    assert!(failure.retained_buffer_bytes() > 0);
    drop((failure, rendered));
}
