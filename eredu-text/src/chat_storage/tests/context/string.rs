use super::*;

#[test]
fn released_lfm_string_expression_preserves_aliases_unicode_and_nonzero_scalars() {
    // The assignment is byte-exact LFM2.5-1.2B ba551d58 parse_content syntax.
    let template = r#"{%- set _ns=namespace(result='') -%}{%- for item in content -%}{%- set _ns.result = _ns.result + ((item.get("text") or "") | string) -%}{%- endfor -%}{{_ns.result}}"#;
    let caller = json!({"content":[{"text":"é\u{0}界🦀"},{"text":37},{"text":3.5},{"text":false},{"text":null},{"text":"end"}]});
    let output = compare(
        template,
        &[],
        &serde_json::Map::new(),
        caller.as_object().unwrap(),
    );
    drop(caller);
    assert_eq!(output.prompt(false), "é\u{0}界🦀373.5end");
}

#[test]
fn string_filter_reuses_text_and_formats_scalar_truth_lengths_and_macro_results() {
    let template = "{% macro text(value) %}{{value|string}}{% endmacro %}{% set alias=value|string %}{{alias}}|{{alias|length}}|{% if alias %}text{% endif %}|{{alias is string}}|{{text(value)|string}}|{{missing|string|length}}";
    for value in [
        json!("  é\u{0}界🦀  "),
        json!(37),
        json!(-19),
        json!(u64::MAX),
        json!(true),
        json!(false),
        json!(null),
        json!(0.0),
        json!(-0.0),
        json!(1.5),
    ] {
        let caller = json!({"value":value});
        compare(
            template,
            &[],
            &serde_json::Map::new(),
            caller.as_object().unwrap(),
        );
    }
    let caller = json!({"value":false});
    let output = compare(
        "{{value|string}}|{{value|string|length}}|{% if value|string %}nonempty{% endif %}",
        &[],
        &serde_json::Map::new(),
        caller.as_object().unwrap(),
    );
    assert_eq!(output.prompt(false), "False|5|nonempty");
}

#[test]
fn string_filter_paid_generated_destination_and_refusals_keep_source_custody() {
    let source = ChatTemplatePlan::prepare_utf8("{{prefix}}:{{value|string}}", "string-custody")
        .unwrap()
        .compile()
        .unwrap();
    let caller = json!({"prefix":"é\u{0}界","value":12345});
    let plan = source
        .render_plan_with_context(
            ChatRenderContext::from_json(&[], None, caller.as_object()).unwrap(),
        )
        .unwrap();
    let failure = plan
        .fail_reservation(ChatRenderBuffer::WithPrompt)
        .render()
        .unwrap_err();
    drop((source, caller));
    assert!(failure.retained_buffer_bytes() > 0);

    let caller = json!({"value":{"nonempty":"structured formatting"}});
    let rendered = compare(
        "{{value|string}}",
        &[],
        &serde_json::Map::new(),
        caller.as_object().unwrap(),
    );
    assert_eq!(
        rendered.prompt(false),
        "{\"nonempty\": \"structured formatting\"}"
    );
    compare_outcome("{{value|string(1)}}", caller.as_object().unwrap());
}

#[test]
fn unicode_uppercase_generated_and_borrowed_strings_match_ordinary_custody() {
    let template = "{{ value|upper }}|{{ ('pre-' + value)|upper }}|{% macro text(x) %}{{ x }}{% endmacro %}{{ text(value)|upper }}|{{ text('')|upper }}";
    for input in ["Straße", "ıiİſﬃ", "é\\u{301}界\\0🦀", "σςΣ", ""] {
        let caller = json!({"value":input});
        let rendered = compare(
            template,
            &[],
            &serde_json::Map::new(),
            caller.as_object().unwrap(),
        );
        assert_eq!(
            rendered.prompt(false),
            format!(
                "{}|PRE-{}|{}|",
                input.to_uppercase(),
                input.to_uppercase(),
                input.to_uppercase()
            )
        );
        drop(caller);
        assert!(rendered.retained_buffer_bytes() > 0);
    }
}

#[test]
fn mapped_uppercase_preserves_order_coercion_generated_sources_and_retained_output() {
    let template = "{{ values|map('upper')|list|join(',') }}|{{ ['str'+'ing','array','object']|map('up'+'per')|join('/') }}|{{ []|map('upper')|list|length }}|{{ absent|map('upper')|join(',') }}";
    let caller = json!({"values":["Straße","ﬃ",true,false,null,-3,1.25,""]});
    let rendered = compare(
        template,
        &[],
        &serde_json::Map::new(),
        caller.as_object().unwrap(),
    );
    drop(caller);
    assert_eq!(
        rendered.prompt(false),
        "STRASSE,FFI,TRUE,FALSE,NONE,-3,1.25,|STRING/ARRAY/OBJECT|0|"
    );
    assert!(rendered.retained_buffer_bytes() > 0);
}
