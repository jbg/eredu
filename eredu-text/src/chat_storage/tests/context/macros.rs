use super::*;

#[test]
fn macros_released_muse_content_and_reasoning_match_ordinary_unicode() {
    // The two declarations are byte-for-byte released 97c77dff macro bodies.
    let template = format!(
        "{}{{% for row in rows %}}{{{{ render_content(row) }}}}|{{% endfor %}}{{{{ render_reasoning() }}}}",
        include_str!("macros_muse.jinja")
    );
    let caller = json!({"rows":["é\u{0}界",[{"type":"image"},{"type":"text","text":"🦀 tail"},{"type":"video"}],null],"reasoning_strength":"強"});
    let output = compare(
        &template,
        &[],
        &serde_json::Map::new(),
        caller.as_object().unwrap(),
    );
    drop((template, caller));
    assert_eq!(
        output.prompt(false),
        "é\u{0}界|<|patch|>🦀 tail<|video|>||Reasoning strength: 強."
    );
}

#[test]
fn macros_share_late_captures_but_keep_arguments_locals_and_outputs_private() {
    let template = "{% set prefix='old' %}{% set ns=namespace(count=0) %}{% macro leaf(value, suffix=tail) %}{% set local='private' %}{% set ns.count=ns.count+1 %} {{ prefix }}{{ value.text }}{{ suffix }} {% endmacro %}{% macro outer(value) %}{% set prefix='caller-local' %}{{ leaf(value) }}{% endmacro %}{% set alias=leaf %}{% set prefix='新' %}{% for row in rows %}{{ outer(row)|trim }}|{{ alias(value=row,suffix='!')|trim }};{% endfor %}{{ ns.count }}:{{ local|default('gone') }}:{{ leaf.name }}:{{ leaf.caller }}";
    let caller = json!({"rows":[{"text":"é"},{"text":"界\u{0}"}],"tail":"🦀"});
    let output = compare(
        template,
        &[],
        &serde_json::Map::new(),
        caller.as_object().unwrap(),
    );
    drop(caller);
    assert_eq!(
        output.prompt(false),
        "新é🦀|新é!;新界\u{0}🦀|新界\u{0}!;4:gone:leaf:False"
    );
}

#[test]
fn macros_recursive_inputs_keywords_namespace_loans_and_retired_arguments_match() {
    let template = "{% macro tree(node, ns) %}{% set ns.count=ns.count+1 %}[{{ node.text }}{% for child in node.children|default([]) %}{{ tree(ns=ns,node=child) }}{% endfor %}]{% endmacro %}{% macro replace(ns, value='défaut') %}{% set ns.text=value %}{{ ns.text }}{% endmacro %}{% set result=namespace(count=0,text='') %}{{ tree(root, result) }}|{{ replace(ns=result)|length }}|{{ result.count }}:{{ result.text }}";
    let caller = json!({"root":{"text":"é","children":[{"text":"界","children":[{"text":"🦀"}]},{"text":"\u{0}"}]}});
    let output = compare(
        template,
        &[],
        &serde_json::Map::new(),
        caller.as_object().unwrap(),
    );
    drop(caller);
    assert_eq!(output.prompt(false), "[é[界[🦀]][\u{0}]]|6|4:défaut");
}

#[test]
fn macro_destinations_and_argument_refusals_preserve_paid_prefix_custody() {
    let template = "{% macro value(row, suffix='!') %}{{ row.text }}{{ suffix }}{% endmacro %}{{ value(row, suffix='界') }}";
    let source = ChatTemplatePlan::prepare_utf8(template, "macro-custody")
        .unwrap()
        .compile()
        .unwrap();
    let caller = json!({"row":{"text":"é\u{0}🦀"}});
    for buffer in [
        ChatRenderBuffer::MacroCalls,
        ChatRenderBuffer::MacroArguments,
        ChatRenderBuffer::MacroCaptures,
        ChatRenderBuffer::ContextText,
        ChatRenderBuffer::Borrowed,
    ] {
        let context = ChatRenderContext::from_json(&[], None, caller.as_object()).unwrap();
        let failure = source
            .render_plan_with_context(context)
            .unwrap()
            .fail_reservation(buffer)
            .render()
            .unwrap_err();
        assert!(failure.retained_buffer_bytes() > 0);
    }
    let context = ChatRenderContext::from_json(&[], None, caller.as_object()).unwrap();
    let failed = source
        .render_plan_with_context(context)
        .unwrap()
        .fail_reservation(ChatRenderBuffer::MacroArguments)
        .render()
        .unwrap_err();
    drop((source, caller));
    assert!(failed.retained_buffer_bytes() > 0);
    for template in [
        "{% macro m(x) %}{{x}}{% endmacro %}{{m(1,2)}}",
        "{% macro m(x) %}{{x}}{% endmacro %}{{m(1,x=2)}}",
        "{% macro m(x) %}{{x}}{% endmacro %}{{m(unexpected=2)}}",
        "{% macro m() %}{{m()}}{% endmacro %}{{m()}}",
    ] {
        let source = ChatTemplatePlan::prepare_utf8(template, "macro-refusal")
            .unwrap()
            .compile()
            .unwrap();
        assert!(
            source
                .render_plan_with_context(ChatRenderContext::from_json(&[], None, None).unwrap())
                .is_err()
        );
        assert!(
            ordinary()
                .apply_chat_template_json(
                    crate::tokenizer::ModelChatTemplate::Single(template.into()),
                    [Vec::<serde_json::Value>::new()],
                    None,
                    "macro-refusal",
                    false,
                    None
                )
                .is_err()
        );
    }
}
