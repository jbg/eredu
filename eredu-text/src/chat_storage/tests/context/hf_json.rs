use super::*;

#[test]
fn hf_tojson_nested_tools_options_and_python_spelling_match_ordinary() {
    let input = json!([{"role":"user","content":"É界🙂\u{0}<>&'\"\\","tool_calls":[{"name":"nonzero","arguments":{"z":7,"a":-11}}]}]);
    let defaults = json!({"tools":[{"z":7,"a":{"界":"É🙂","x":[true,null,-11,2.5]},"empty":{}}],"value":"É界🙂\u{7f}\n<>&'\"\\"});
    for template in [
        "{{ tools|tojson }}",
        "{{ messages|tojson(indent=2) }}",
        "{{ messages[0]|tojson(sort_keys=true) }}",
        "{{ tools[0]|tojson(sort_keys=true, separators=(',',':')) }}",
        "{{ tools|tojson(true, '界', [' ; ', ' => '], true) }}",
        "{{ value|tojson(ensure_ascii=true) }}",
        "{{ value|tojson(false) }}",
    ] {
        compare(
            template,
            input.as_array().unwrap(),
            defaults.as_object().unwrap(),
            &serde_json::Map::new(),
        );
    }
    let scalar = json!({"values":[-0.0,1e-7,1e20,7,-11,true,null,"É🙂\u{7f}"]});
    let rendered = compare(
        "{{ values|tojson(ensure_ascii=true) }}",
        &[],
        scalar.as_object().unwrap(),
        &serde_json::Map::new(),
    );
    // Independent Python json.dumps spelling.
    assert_eq!(
        rendered.prompt(false),
        r#"[-0.0, 1e-07, 1e+20, 7, -11, true, null, "\u00c9\ud83d\ude42\u007f"]"#
    );
    let keys = json!({"value":{"z":7,"a":"É🙂"}});
    let rendered = compare(
        "{{ value|tojson(sort_keys=true,separators=(',',':'),ensure_ascii=true) }}",
        &[],
        keys.as_object().unwrap(),
        &serde_json::Map::new(),
    );
    assert_eq!(
        rendered.prompt(false),
        r#"{"a":"\u00c9\ud83d\ude42","z":7}"#
    );
}

#[test]
fn hf_tojson_depth_and_sorted_keys_request_actual_fixed_scratch_before_growth() {
    let source = ChatTemplatePlan::prepare_utf8(
        "{{ tools|tojson(sort_keys=true,indent='界') }}",
        "json-depth",
    )
    .unwrap()
    .compile()
    .unwrap();
    let input = json!({"tools":[{"z":7,"a":[{"z":-11,"a":{"z":23,"a":[{"z":31,"a":"É🙂"}]}}]}]});
    let context = ChatRenderContext::from_json(&[], None, input.as_object()).unwrap();
    let mut capacity = ChatJsonCapacity::default();
    let mut attempts = 0;
    let plan = loop {
        let before = source
            .render_prefix_bytes_with_json_capacity(capacity)
            .unwrap();
        match source.render_plan_attempt(context, capacity) {
            Err(ChatRenderPlanError::JsonCapacity(next)) => {
                assert!(
                    next.frames >= capacity.frames
                        && next.keys >= capacity.keys
                        && next != capacity
                );
                assert!(source.render_prefix_bytes_with_json_capacity(next).unwrap() > before);
                capacity = next;
                attempts += 1;
            }
            Ok(plan) => break plan,
            Err(error) => panic!("unexpected JSON attempt failure: {error}"),
        }
    };
    assert!(attempts >= 4 && capacity.frames >= 4 && capacity.keys >= 6);
    let rendered = plan.render().unwrap();
    let expected = compare(
        "{{ tools|tojson(sort_keys=true,indent='界') }}",
        &[],
        &serde_json::Map::new(),
        input.as_object().unwrap(),
    );
    assert_eq!(rendered.prompt(false), expected.prompt(false));
    drop((input, source));
    assert!(rendered.prompt(false).contains("É🙂"));
}

#[test]
fn hf_tojson_preserves_keyword_evaluation_order_and_refuses_invalid_options() {
    let template = "{% set ns=namespace(value=0) %}{% macro mark(n) %}{% set ns.value=ns.value*10+n %}{{ '' }}{% endmacro %}{{ value|tojson(sort_keys=mark(1),ensure_ascii=mark(2)) }}|{{ ns.value }}";
    let values = json!({"value":{"z":7,"a":"É"}});
    let rendered = compare(
        template,
        &[],
        &serde_json::Map::new(),
        values.as_object().unwrap(),
    );
    assert!(rendered.prompt(false).ends_with("|12"));
    for template in [
        "{{ value|tojson(no_such_option=true) }}",
        "{{ value|tojson(true,ensure_ascii=false) }}",
        "{{ value|tojson(separators=(',',)) }}",
    ] {
        assert!(
            ChatTemplatePlan::prepare_utf8(template, "json-invalid")
                .unwrap()
                .compile()
                .is_err()
        );
    }
    for template in [
        "{{ value|tojson(indent=1.5) }}",
        "{{ value|tojson(separators=parts) }}",
    ] {
        let source = ChatTemplatePlan::prepare_utf8(template, "json-invalid-value")
            .unwrap()
            .compile()
            .unwrap();
        let values = json!({"value":[7,-11],"parts":[",",false]});
        assert!(
            source
                .render_plan_with_context(
                    ChatRenderContext::from_json(&[], None, values.as_object()).unwrap()
                )
                .is_err()
        );
    }
}

#[test]
fn hf_tojson_reads_generated_unicode_concatenation_and_sliced_json_prefixes() {
    let values = json!({"value": "É🙂\\\n", "object": {"station": "界", "n": -11}});
    for template in [
        "{{ ('  ' ~ value ~ ' x  ')|trim|tojson(ensure_ascii=true) }}",
        "{{ (value ~ '界')[::-1]|tojson }}",
        "{{ object|tojson|tojson(ensure_ascii=true) }}",
        "{{ (value ~ -11 ~ true ~ none ~ missing)|tojson }}",
        "{% set rendered = object|tojson %}{{ rendered[:-1]|tojson }}",
    ] {
        compare(template, &[], &serde_json::Map::new(), values.as_object().unwrap());
    }
    let rendered = compare("{{ ('É' ~ '🙂')|tojson(ensure_ascii=true) }}",
        &[], &serde_json::Map::new(), &serde_json::Map::new());
    assert_eq!(rendered.prompt(false), r#""\u00c9\ud83d\ude42""#);
}

#[test]
fn hf_tojson_generated_containers_share_depth_sorting_and_borrowed_views() {
    let input=json!([{"role":"user","content":"É🙂","arguments":{"z":7,"a":-11}},{"role":"assistant","content":"界"}]);
    let empty=serde_json::Map::new();
    for template in [
        "{% set item={'z':messages[0],'a':[messages[1].content,-11,true,none]} %}{{ item|tojson }}",
        "{% set item={'z':messages[0],'a':[messages[1].content,-11,true,none]} %}{{ item|tojson(sort_keys=true,indent='界') }}",
        "{% set ns=namespace(values=[]) %}{% for message in messages %}{% set ns.values=ns.values+[{'z':message.content~'尾','a':message.arguments}] %}{% endfor %}{{ ns.values[::-1]|tojson(ensure_ascii=true,sort_keys=true) }}",
        "{% set values={'z':[7,-11],'a':messages} %}{{ values.items()|list|tojson }}|{{ values.keys()|tojson }}",
        "{{ messages[0].arguments.items()|list|tojson }}|{{ messages[0].arguments.keys()|tojson }}",
        "{{ ['É 界'|tojson,range(3),messages[::-1],('a::界::b').split('::')[::-1]]|tojson(indent=2) }}",
        "{% set parts=[',',':'] %}{{ {'z':[-11,7],'a':'É🙂'}|tojson(separators=parts,sort_keys=true) }}",
    ] { compare(template,input.as_array().unwrap(),&empty,&empty); }
    let output=compare("{{ {'z':7,'a':[-11,'É🙂']}|tojson(sort_keys=true,separators=(',',':'),ensure_ascii=true) }}",&[],&empty,&empty);
    assert_eq!(output.prompt(false),r#"{"a":[-11,"\u00c9\ud83d\ude42"],"z":7}"#);
}
