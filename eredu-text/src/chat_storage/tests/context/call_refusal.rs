use super::*;
#[test]
fn absent_calls_preserve_ordinary_refusal_and_do_not_hide_known_worker_gaps() {
    let template = "prefix{% if reject %}{{ raise_exception(message) }}{% else %}ok{% endif %}";
    let source = ChatTemplatePlan::prepare_utf8(template, "call-refusal")
        .unwrap()
        .compile()
        .unwrap();
    let caller = json!({"reject":true,"message":"É界\u{0}"});
    let ordinary = ordinary().apply_chat_template_json(
        crate::tokenizer::ModelChatTemplate::Single(template.into()),
        [vec![]],
        None,
        "call-refusal",
        false,
        caller.as_object(),
    );
    assert!(
        ordinary
            .unwrap_err()
            .to_string()
            .contains("unknown function")
    );
    let failure = source
        .render_plan_with_context(
            ChatRenderContext::from_json(&[], None, caller.as_object()).unwrap(),
        )
        .unwrap_err();
    assert!(failure.is_unknown_function());
    drop((caller, source));
    assert!(failure.is_unknown_function());
    let success = compare(
        "{% macro raise_exception(value) %}{{ value }}{% endmacro %}{{ raise_exception(message) }}",
        &[],
        &serde_json::Map::new(),
        json!({"message":"local É界"}).as_object().unwrap(),
    );
    assert_eq!(success.prompt(false), "local É界");
    for template in ["{{ debug() }}", "{{ dict() }}"] {
        let source = ChatTemplatePlan::prepare_utf8(template, "known-call")
            .unwrap()
            .compile()
            .unwrap();
        if let Err(failure) = source.render_plan(ChatMessages::from_text(&[])) {
            assert!(!failure.is_unknown_function());
        }
    }
}
