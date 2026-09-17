use super::*;

#[test]
fn released_gemma_range_scans_preserve_unicode_messages_and_outer_loop_positions() {
    // The pre-scan and backward-scan declarations/loop bounds are the released
    // Gemma 4 3e22461f / Unsloth Gemma 4 94899c0f template expressions.
    let template = r#"{%- set loop_messages=messages -%}
{%- set ns_turn = namespace(last_user_idx=-1) -%}
{%- for i in range(loop_messages | length) -%}
    {%- if loop_messages[i]['role'] == 'user' -%}
        {%- set ns_turn.last_user_idx = i -%}
    {%- endif -%}
{%- endfor -%}
{{ns_turn.last_user_idx}}|
{%- for message in loop_messages -%}
    {%- set prev_nt = namespace(role=None, found=false) -%}
    {%- if loop.index0 > 0 -%}
        {%- for j in range(loop.index0 - 1, -1, -1) -%}
            {%- if not prev_nt.found -%}
                {%- if loop_messages[j]['role'] != 'tool' -%}
                    {%- set prev_nt.role = loop_messages[j]['role'] -%}
                    {%- set prev_nt.found = true -%}
                {%- endif -%}
            {%- endif -%}
        {%- endfor -%}
    {%- endif -%}
    {{loop.index0}}:{{prev_nt.role}}:{{message.content}};
{%- endfor -%}"#;
    let base = json!([
        {"role":"user","content":"é界"},
        {"role":"assistant","content":"first"},
        {"role":"tool","content":"🦀"},
        {"role":"assistant","content":"again"},
        {"role":"user","content":"nul\u{0}end"}
    ]);
    let output = compare(
        template,
        base.as_array().unwrap(),
        &serde_json::Map::new(),
        &serde_json::Map::new(),
    );
    drop(base);
    assert_eq!(
        output.prompt(false),
        "4|0:None:é界;1:user:first;2:assistant:🦀;3:assistant:again;4:assistant:nul\u{0}end;"
    );
}

#[test]
fn range_signed_progressions_list_views_and_scalar_arguments_match_ordinary() {
    let template = r#"{% set values=range(start,stop,step) %}{{values|length}}:{% if values %}yes{% else %}no{% endif %}:{{values is iterable}}:{{values is sequence}}:{{values|list is sequence}}:{% for i in values %}{{loop.index0}}={{i}};{% endfor %}|{{values|join('界')}}|{{needle in values}}|{{(values|list)[-1]}}|{{values[0]}}"#;
    for (start, stop, step, needle) in [
        (json!(-7), json!(9), json!(3), json!(2)),
        (json!(9), json!(-7), json!(-3), json!(-3)),
        (json!(7), json!(7), json!(1), json!(7)),
        (json!(2), json!(-9), json!(3), json!(2)),
        (json!(-3), json!(5), json!(null), json!(false)),
        (json!(false), json!(4.0), json!(true), json!(3.0)),
        (json!(i64::MAX), json!(i64::MIN), json!(i64::MIN), json!(-1)),
        (json!(i64::MIN), json!(i64::MAX), json!(i64::MAX), json!(-1)),
    ] {
        let caller = json!({"start":start,"stop":stop,"step":step,"needle":needle});
        compare(
            template,
            &[],
            &serde_json::Map::new(),
            caller.as_object().unwrap(),
        );
    }
    let output = compare(
        "{{range(5)|join(',')}}|{{range(5,none,none)|join(',')}}|{{range(-3)|length}}|{{range(100000)|length}}",
        &[],
        &serde_json::Map::new(),
        &serde_json::Map::new(),
    );
    assert_eq!(output.prompt(false), "0,1,2,3,4|0,1,2,3,4|0|100000");
    let caller = json!({"start":i64::MAX,"stop":i64::MIN,"step":i64::MIN});
    let output = compare(
        "{{range(start,stop,step)|join(',')}}",
        &[],
        &serde_json::Map::new(),
        caller.as_object().unwrap(),
    );
    assert_eq!(output.prompt(false), "9223372036854775807,-1");
}

#[test]
fn range_refusals_and_paid_output_keep_source_custody_after_input_drop() {
    let source = ChatTemplatePlan::prepare_utf8(
        "{% for i in range(count) %}{{prefix}}:{{i}};{% endfor %}",
        "range-custody",
    )
    .unwrap()
    .compile()
    .unwrap();
    let caller = json!({"count":7,"prefix":"é\u{0}界🦀"});
    let plan = source
        .render_plan_with_context(
            ChatRenderContext::from_json(&[], None, caller.as_object()).unwrap(),
        )
        .unwrap();
    let failure = plan
        .fail_reservation(ChatRenderBuffer::WithoutPrompt)
        .render()
        .unwrap_err();
    drop((source, caller));
    assert!(failure.retained_buffer_bytes() > 0);

    for template in [
        "{{range(1,7,0)|length}}",
        "{{range(100001)|length}}",
        "{{range()|length}}",
        "{{range(1,2,3,4)|length}}",
        "{{range('3')|length}}",
        "{{range(3.5)|length}}",
        "{{range(stop=3)|length}}",
        "{% set range=3 %}{{range(5)|length}}",
    ] {
        let source = ChatTemplatePlan::prepare_utf8(template, "invalid-range")
            .unwrap()
            .compile()
            .unwrap();
        assert!(
            source
                .render_plan_with_context(ChatRenderContext::from_json(&[], None, None).unwrap())
                .is_err(),
            "{template}"
        );
        assert!(
            ordinary()
                .apply_chat_template_json(
                    crate::tokenizer::ModelChatTemplate::Single(template.into()),
                    [Vec::<serde_json::Value>::new()],
                    None,
                    "invalid-range",
                    false,
                    None
                )
                .is_err(),
            "{template}"
        );
    }
}
