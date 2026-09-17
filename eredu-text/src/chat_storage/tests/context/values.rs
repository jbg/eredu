use super::*;

#[test]
fn generated_lists_retain_borrowed_values_nested_views_and_prior_versions() {
    let messages = json!([
        {"role":"user","content":"É"},
        {"role":"assistant","content":"界🙂"},
        {"role":"user","content":"last"}
    ]);
    let empty = serde_json::Map::new();
    let source="{% set ns=namespace(values=[]) %}{% for message in messages %}{% set ns.values=ns.values+[message] %}{% endfor %}{% set old=ns.values %}{% set current=old+[messages[0]] %}{{ old|length }}:{{ current|length }}|{% for item in current[::-1][1:] %}{{ item.content }};{% endfor %}|{% set nested=[current,[-11,7,true,none]] %}{{ nested[0][1].content }}|{{ nested[1][0] }}|{{ -11 in nested[1] }}|{{ old is sequence }}";
    let rendered = compare(source, messages.as_array().unwrap(), &empty, &empty);
    assert_eq!(
        rendered.prompt(false),
        "3:4|last;界🙂;É;|界🙂|-11|True|False"
    );
    // Ordinary MergeSeq materializes the same sized chain when its shared
    // depth threshold is crossed. Preserve that observable sequence test.
    let deep=compare("{% set ns=namespace(values=[]) %}{% for n in range(33) %}{% set ns.values=ns.values+[n] %}{% endfor %}{{ ns.values is sequence }}|{{ ns.values|length }}|{{ (ns.values+[]) is sequence }}",&[],&empty,&empty);
    assert_eq!(deep.prompt(false),"True|33|False");
    drop(messages);
    assert!(rendered.prompt(true).contains("界🙂"));
}

#[test]
fn generated_list_attempts_request_exact_value_and_borrowed_register_storage() {
    let source = ChatTemplatePlan::prepare_utf8(
        "{% set values=[messages[0],messages[1]] %}{{ values[-1].content }}",
        "values",
    )
    .unwrap()
    .compile()
    .unwrap();
    let messages = json!([{"role":"user","content":"first"},{"role":"assistant","content":"É🙂"}]);
    let context = ChatRenderContext::from_json(messages.as_array().unwrap(), None, None).unwrap();
    let no_values = ChatValueCapacity::default();
    let json = ChatJsonCapacity::default();
    let text = ChatTextCapacity::default();
    let before = source
        .render_prefix_bytes_with_values(json, text, no_values)
        .unwrap();
    let capacity = match source.render_plan_attempt_with_values(context, json, text, no_values) {
        Err(ChatRenderPlanError::ValueCapacity(next)) => next,
        result => panic!("expected actual fixed value destination: {result:?}"),
    };
    assert_eq!(capacity.slots, 2);
    assert!(
        source
            .render_prefix_bytes_with_values(json, text, capacity)
            .unwrap()
            > before
    );
    let rendered = source
        .render_plan_attempt_with_values(context, json, text, capacity)
        .unwrap()
        .render()
        .unwrap();
    drop(messages);
    assert_eq!(rendered.prompt(false), "É🙂");
}

#[test]
fn generated_maps_preserve_duplicate_order_borrowed_members_and_mapping_views() {
    let messages=json!([{"role":"user","content":"É🙂"}]);
    let empty=serde_json::Map::new();
    let rendered=compare("{% set value={'z':-11,'a':messages[0],'z':23} %}{{ value.z }}|{{ value.a.content }}|{{ 'z' in value }}|{{ value is mapping }}|{{ value.get(\"absent\",-5) }}|{% for key,item in value.items() %}{{ key }}:{% if key=='z' %}{{ item }}{% else %}{{ item.role }}{% endif %};{% endfor %}|{% for key in value.keys()|list %}{{ key }};{% endfor %}",messages.as_array().unwrap(),&empty,&empty);
    assert_eq!(rendered.prompt(false),"23|É🙂|True|True|-5|z:23;a:user;|z;a;");
    drop(messages);
    assert!(rendered.prompt(true).contains("É🙂"));
}

#[test]
fn float_filter_preserves_numeric_and_text_conversion() {
    let empty=serde_json::Map::new();
    let rendered=compare("{{ 7|float }}|{{ -11|float }}|{{ none|float }}|{{ true|float }}|{{ '0.7'|float }}|{{ 'NaN'|float }}|{{ missing|float }}",&[],&empty,&empty);
    assert_eq!(rendered.prompt(false),"7.0|-11.0|0.0|1.0|0.7|NaN|0.0");
}

#[test]
fn generated_list_join_preserves_segmented_unicode_and_separator_prefixes(){
    let empty=serde_json::Map::new();
    let input=json!([{"role":"user","content":"É🙂"}]);
    let output=compare("{% set ns=namespace(values=['self']) %}{% for message in messages %}{% set ns.values=ns.values+['\"'+message.content+'.*\"'] %}{% endfor %}{% set ns.values=ns.values+[7,-11,true,none] %}{% set separator={'s':'界'}|tojson %}{{ ns.values|join(separator) }}",input.as_array().unwrap(),&empty,&empty);
    assert!(output.prompt(false).contains("\"É🙂.*\""));
    assert!(output.prompt(false).ends_with("True{\"s\": \"界\"}None"));
    compare("{{ ('É🙂'|tojson)|join(':') }}",&[],&empty,&empty);
    let flattened=compare("{{ (' pre '~messages[0].content~' tail ').replace('tail','end').strip() }}|{{ ('0.'~'7')|float }}",input.as_array().unwrap(),&empty,&empty);
    assert_eq!(flattened.prompt(false),"pre É🙂 end|0.7");
}

#[test]
fn limited_split_preserves_whitespace_remainder_empty_separator_and_generated_views(){
    let empty=serde_json::Map::new();
    for template in [
        "{{ 'prefix</think>É</think>界'.split('</think>',1)|tojson }}",
        "{{ ' a  É   界 '.split(none,1)|tojson }}",
        "{{ ' a  É   界 '.split(none,0)|join('|') }}",
        "{{ 'É🙂'.split('',2)|tojson }}",
        "{{ (' a '~'É  界 ').split(none,-1)[::-1]|tojson }}",
    ] {compare(template,&[],&empty,&empty);}
}

#[test]
fn selection_preserves_order_attribute_paths_generated_candidates_and_escaped_sources(){
    let caller=json!({"items":[{"role":"user","detail":{"score":7}},{"role":"assistant","detail":{"score":-11}},{"role":"user","detail":{"score":7}}],"names":["browser","code_interpreter","search"],"left":"bro","right":"wser"});
    for template in [
        "{{ names|reject('equalto','code_interpreter')|join(', ') }}",
        "{% for item in items|selectattr('role','equalto','user') %}{{ item.detail.score }};{% endfor %}",
        "{{ items|selectattr('detail.score','equalto',7)|length }}",
        "{{ [0,1,false,true,none,'',left+right]|select|tojson }}",
        "{{ [left+right,'code_interpreter']|reject('eq','code_interpreter')|join(',') }}",
        "{{ names|reject('equalto','absent')|list|tojson }}",
        "{{ items|rejectattr('role','equalto','user')|tojson }}",
    ] { compare(template,&[],&serde_json::Map::new(),caller.as_object().unwrap()); }
}
