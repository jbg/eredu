use super::*;

#[test]
fn borrowed_lengths_count_unicode_collections_and_trimmed_prefixes_with_retained_outputs() {
    let template = r#"{{ messages|length }}|{% for message in messages %}{{ message|count }}:{{ message.content|length }}:{{ ((cfg.left.text|trim) + message.content + cfg.right.text)|trim|length }}:{% if cfg.options|count == 2 %}two{% endif %};{% endfor %}{{ (cfg.left.text + cfg.right.text)|trim|count }}|{{ cfg.values[-1]|length }}|{{ cfg.empty|length }}|{{ (cfg.left.text|trim(' é'))|length }}|{{ ((cfg.blank + cfg.blank)|trim + cfg.right.text)|trim|length }}{% if add_generation_prompt %}|{{ cfg.values|count }}{% endif %}"#;
    let defaults = json!({"cfg":{"left":{"text":"wrong default"}}, "messages":[{"role":"user","content":"default"}]});
    let caller = json!({"cfg":{"left":{"text":"  é\u{301}界  "},"right":{"text":"\u{2003}終\u{2003}"},"blank":" \t\u{2003}","options":{"one":1,"two":2},"values":["x",[1,2,3]],"empty":[]},"messages":[{"role":"user","content":"a\u{0}🦀"},{"role":"assistant","content":"  第二 "}]});
    let base = [json!({"role":"user","content":"discarded"})];
    let rendered = compare(
        template,
        &base,
        defaults.as_object().unwrap(),
        caller.as_object().unwrap(),
    );
    assert!(rendered.prompt(false).starts_with("2|2:3:"));
    assert!(rendered.prompt(true).ends_with("|2"));
    let source = ChatTemplatePlan::prepare_utf8(template, "lengths")
        .unwrap()
        .compile()
        .unwrap();
    let context =
        ChatRenderContext::from_json(&base, defaults.as_object(), caller.as_object()).unwrap();
    let plan = source.render_plan_with_context(context).unwrap();
    let capacity = plan.requirements().buffer_bytes();
    assert!(rendered.retained_buffer_bytes() <= capacity);
    let failure = plan
        .fail_reservation(ChatRenderBuffer::ContextText)
        .render()
        .unwrap_err();
    drop((base, defaults, caller, source));
    assert!(failure.retained_buffer_bytes() > 0);
    assert!(rendered.prompt(false).starts_with("2|2:3:"));
    drop((failure, rendered));

    for value in [json!(null), json!(false), json!(0), json!(-0.0)] {
        let caller = json!({"value":value});
        let source = ChatTemplatePlan::prepare_utf8("{{ value|length }}", "lengths")
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
                    crate::tokenizer::ModelChatTemplate::Single("{{ value|length }}".into()),
                    [vec![]],
                    None,
                    "lengths",
                    false,
                    caller.as_object()
                )
                .is_err()
        );
    }
    let source = ChatTemplatePlan::prepare_utf8("{{ absent|count }}", "lengths")
        .unwrap()
        .compile()
        .unwrap();
    assert!(
        source
            .render_plan_with_context(ChatRenderContext::from_json(&[], None, None).unwrap())
            .is_err()
    );
}
