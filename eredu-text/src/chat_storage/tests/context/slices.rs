use super::*;

#[test]
fn shared_slices_preserve_reverse_bounds_nested_views_and_borrowed_messages() {
    let empty = serde_json::Map::new();
    // Independent expected coordinates include empty reverse slices and MIN
    // step; neither VM may inherit the old reverse-index underflow behavior.
    for (start, stop, step, expected) in [
        (None, None, -1, "40;30;20;10;"),
        (Some(0), Some(0), -1, ""),
        (Some(-9), None, -1, ""),
        (None, Some(-9), -2, "40;20;"),
        (None, Some(-1), -1, ""),
        (None, None, i64::MIN, "40;"),
        (Some(1), Some(9), 2, "20;40;"),
        (Some(9), None, 1, ""),
    ] {
        let args = json!({"rows":[10,20,30,40],"start":start,"stop":stop,"step":step});
        let rendered = compare(
            "{% for n in rows[start:stop:step] %}{{ n }};{% endfor %}",
            &[],
            &empty,
            args.as_object().unwrap(),
        );
        assert_eq!(rendered.prompt(false), expected);
    }
    let messages = json!([{"role":"system","content":"S"},{"role":"user","content":"É"},{"role":"assistant","content":"界"}]);
    let rendered = compare(
        "{% set selected=messages[::-1][1:][::-1] %}{% for message in selected %}{{ loop.index }}:{{ message.role }}={{ message.content }};{% endfor %}|{{ selected[1].content }}|{{ selected|length }}|{% for n in range(1,8)[::-2] %}{{ n }};{% endfor %}",
        messages.as_array().unwrap(),
        &empty,
        &empty,
    );
    assert_eq!(rendered.prompt(false), "1:system=S;2:user=É;|É|2|7;5;3;1;");
    drop(messages);
    assert!(rendered.prompt(true).contains("user=É"));
}

#[test]
fn shared_unicode_slices_consume_generated_json_and_concatenated_prefixes() {
    let args = json!({"value":"É🙂界x","payload":{"label":"É🙂","count":7}});
    let empty = serde_json::Map::new();
    let rendered = compare(
        "{{ value[::-1] }}|{{ value[1:3] }}|{% set out=payload|tojson %}{{ out[:-1] }}|{% set text='A'+value+'Z' %}{{ text[::-2] }}",
        &[],
        &empty,
        args.as_object().unwrap(),
    );
    assert!(rendered.prompt(false).starts_with("x界🙂É|🙂界|{"));
    assert!(rendered.prompt(false).ends_with("|Z界É"));
    drop(args);
    assert!(rendered.prompt(true).contains("count"));
    for value in ["", "a", "🙂"] {
        let args = json!({"value":value});
        assert_eq!(
            compare(
                "{{ value[0:0:-1] }}",
                &[],
                &empty,
                args.as_object().unwrap()
            )
            .prompt(false),
            ""
        );
    }
}

#[test]
fn slice_zero_step_refuses_during_render_and_generated_prefix_retains_output() {
    let source = ChatTemplatePlan::prepare_utf8("{{ value[::step] }}", "slice")
        .unwrap()
        .compile()
        .unwrap();
    let args = json!({"value":"abc","step":0});
    let context = ChatRenderContext::from_json(&[], None, args.as_object()).unwrap();
    assert!(
        source
            .render_plan_with_context(context)
            .unwrap()
            .render()
            .is_err()
    );
    let template = "{% set out=payload|tojson %}{{ out[:-1] }}";
    let source = ChatTemplatePlan::prepare_utf8(template, "slice")
        .unwrap()
        .compile()
        .unwrap();
    let args = json!({"payload":{"text":"actual bytes"}});
    let context = ChatRenderContext::from_json(&[], None, args.as_object()).unwrap();
    let plan = source.render_plan_with_context(context).unwrap();
    let rendered = plan.render().unwrap();
    drop(args);
    assert_eq!(rendered.prompt(false), "{\"text\": \"actual bytes\"");
}
