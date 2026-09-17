use super::*;

#[test]
fn nested_macros_share_lexical_updates_and_capture_the_actual_outer_loop() {
    let template = "{% macro outer(row) %}{% macro value(suffix='!') %}{{row.text}}{{suffix}}{% endmacro %}{% set alias=value %}{% set row=row.next|default(row) %}{{alias()}}{% endmacro %}{% for row in rows %}{{outer(row)}}:{% macro loop_value() %}{{row.text}}{{loop.index}}{% endmacro %}{% set alias=loop_value %}{% for item in row.parts %}{{alias()}};{% endfor %}|{% endfor %}";
    let caller = json!({"rows":[{"text":"é","next":{"text":"界"},"parts":[1,2]},{"text":"🦀\u{0}","parts":[1]}]});
    let output = compare(
        template,
        &[],
        &serde_json::Map::new(),
        caller.as_object().unwrap(),
    );
    drop(caller);
    assert_eq!(output.prompt(false), "界!:é1;é1;|🦀\u{0}!:🦀\u{0}2;|");
}

#[test]
fn call_blocks_keep_caller_arguments_and_nested_closures_in_their_lexical_frame() {
    let template = "{% macro box(items,prefix='<') %}{{prefix}}{% for part in items %}{{caller(part)}};{% endfor %}>{% endmacro %}{% for row in rows %}{% call(part) box(row.parts,prefix='[') %}{% macro local() %}{{row.name}}:{{part}}:{{loop.index}}{% endmacro %}{{local()}}{% endcall %}{% endfor %}";
    let caller =
        json!({"rows":[{"name":"é","parts":["界","🦀"]},{"name":"NUL\u{0}","parts":["終"]}]});
    let output = compare(
        template,
        &[],
        &serde_json::Map::new(),
        caller.as_object().unwrap(),
    );
    drop(caller);
    assert_eq!(output.prompt(false), "[é:界:1;é:🦀:1;>[NUL\u{0}:終:2;>");
}

#[test]
fn nested_closure_destinations_keep_partial_custody_and_reject_unfunded_escape() {
    let template = "{% macro m(value) %}{% macro n() %}{{value.text}}{% endmacro %}{{n()}}{% endmacro %}{{m(row)}}";
    let source = ChatTemplatePlan::prepare_utf8(template, "nested-closures")
        .unwrap()
        .compile()
        .unwrap();
    let caller = json!({"row":{"text":"é\u{0}界🦀"}});
    let plan = source
        .render_plan_with_context(
            ChatRenderContext::from_json(&[], None, caller.as_object()).unwrap(),
        )
        .unwrap();
    let failure = plan
        .fail_reservation(ChatRenderBuffer::MacroCaptures)
        .render()
        .unwrap_err();
    drop((source, caller));
    assert!(failure.retained_buffer_bytes() > 0);
    let escaped = "{% set ns=namespace() %}{% macro m() %}{% macro n() %}inner{% endmacro %}{% set ns.saved=n %}{% endmacro %}{{m()}}";
    let source = ChatTemplatePlan::prepare_utf8(escaped, "escaping-closure")
        .unwrap()
        .compile()
        .unwrap();
    assert!(
        source
            .render_plan_with_context(ChatRenderContext::from_json(&[], None, None).unwrap())
            .is_err()
    );
}

#[test]
fn nested_macro_local_loop_shadows_captured_loop_until_it_retires() {
    let template = "{% for row in rows %}{% macro indexes() %}{{loop.index}}:{% for part in row.parts %}{{loop.index}}{% endfor %}:{{loop.index}}{% endmacro %}{{indexes()}};{% endfor %}";
    let caller = json!({"rows":[{"parts":[1,2]},{"parts":[3]}]});
    let output = compare(
        template,
        &[],
        &serde_json::Map::new(),
        caller.as_object().unwrap(),
    );
    assert_eq!(output.prompt(false), "1:12:1;2:1:2;");
}
