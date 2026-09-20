use super::*;

#[test]
fn borrowed_membership_matches_ordinary_search_keys_and_scalar_equality() {
    let template = "{{ needle in values }}|{{ needle not in values }}|{{ cfg.needle in cfg.values }}|{{ cfg.needle not in cfg.values }}{% if add_generation_prompt %}|done{% endif %}";
    for (needle, values) in [
        (json!(null), json!([false, 0, null])),
        (json!(true), json!([0, 1])),
        (json!(false), json!([0])),
        (json!(i64::MIN), json!([0, i64::MIN, u64::MAX])),
        (json!(u64::MAX), json!([18446744073709551616.0, u64::MAX])),
        (json!(9007199254740993u64), json!([9007199254740992.0])),
        (json!(-0.0), json!([0.0])),
        (
            json!("tail\u{0}界"),
            json!([null,{"nested":[1]},["nested"],"tail\u{0}界"]),
        ),
        (json!("absent"), json!([])),
        (json!("items"), json!({"items":null,"other":2})),
        (json!("absent"), json!({"items":[]})),
        (json!(1), json!({"1":"string key"})),
        (
            json!("é\u{301}界\u{0}"),
            json!("prefixé\u{301}界\u{0}🦀suffix"),
        ),
        (json!("é"), json!("e\u{301}")),
        (json!(""), json!("")),
        (json!(""), json!("nonempty")),
        (json!("longer"), json!("short")),
        (
            json!("z".repeat(65) + "界"),
            json!("z".repeat(300) + "界tail"),
        ),
    ] {
        let caller =
            json!({"needle":needle,"values":values,"cfg":{"needle":needle,"values":values}});
        let rendered = compare(
            template,
            &[],
            &serde_json::Map::new(),
            caller.as_object().unwrap(),
        );
        drop(caller);
        assert!(rendered.prompt(true).ends_with("|done"));
    }
    let rendered = compare(
        "{{ missing in absent }}|{{ missing not in absent }}|{{ none in [] }}",
        &[],
        &serde_json::Map::new(),
        &serde_json::Map::new(),
    );
    assert_eq!(rendered.prompt(false), "False|True|False");
}

#[test]
fn released_membership_conditions_preserve_short_circuit_and_escaped_custody() {
    // Retained Nanbeige uses '</think>' in content; Kimi K2 uses 'items' in spec.
    let template = "{% if 'items' in spec %}{{ spec.title + suffix }}{% endif %}|{% if 'missing' not in spec or missing.child %}map{% endif %}{% for message in messages %}|{% if '</think>' in message.content %}thinking{% else %}plain{% endif %}:{% if 'role' in message and 'unknown' not in message %}keys{% endif %}:{{ message.content }}{% endfor %}{% if add_generation_prompt and 'items' in spec %}|next{% endif %}";
    let defaults = json!({"spec":{"wrong":1},"suffix":"default"});
    let caller = json!({"spec":{"items":null,"title":"é\u{0}"},"suffix":"界"});
    let messages = [
        json!({"role":"user","content":"before</think>after"}),
        json!({"role":"assistant","content":"plain🦀"}),
    ];
    let rendered = compare(
        template,
        &messages,
        defaults.as_object().unwrap(),
        caller.as_object().unwrap(),
    );
    assert_eq!(
        rendered.prompt(false),
        "é\u{0}界|map|thinking:keys:before</think>after|plain:keys:plain🦀"
    );
    assert!(rendered.prompt(true).ends_with("|next"));
    let source = ChatTemplatePlan::prepare_utf8(template, "membership")
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
    drop((source, caller, defaults, messages));
    assert!(failure.retained_buffer_bytes() > 0);
    assert!(
        rendered
            .prompt(false)
            .contains("thinking:keys:before</think>after")
    );
    assert!(rendered.prompt(true).ends_with("|next"));
}

#[test]
fn membership_matches_structured_equality_and_dynamic_coercion() {
    for caller in [
        json!({"needle":{"x":1},"values":[{"x":1}]}),
        json!({"needle":[1],"values":[[1]]}),
        json!({"needle":17,"values":"17"}),
        json!({"needle":"x","values":17}),
        json!({"needle":"x","values":null}),
    ] {
        compare_outcome("{{ needle in values }}", caller.as_object().unwrap());
    }
}

#[test]
fn generated_membership_and_empty_macro_values_share_paid_string_consumers() {
    let caller = json!({"needle":"界","left":"É","right":"界🙂","map":{"界界":7}});
    for template in [
        "{{ needle in (left+right) }}|{{ (needle+needle) in map }}",
        "{{ (left+right) in [left+right,'absent'] }}|{{ needle in ((left+right)|replace('🙂','!')) }}",
        "{% macro empty() %}{% endmacro %}{{ empty() == '' }}|{{ empty().endswith('') }}|{{ empty()|tojson }}",
        "{% macro content() %}{{ left }}{{ right }}{% endmacro %}{{ needle in content() }}|{{ content() == left+right }}",
    ] {
        compare(
            template,
            &[],
            &serde_json::Map::new(),
            caller.as_object().unwrap(),
        );
    }
}
