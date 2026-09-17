use super::*;

#[test]
fn argument_mapping_loops_share_ordinary_unpacking_and_lexical_scopes() {
    // This is the released Nanbeige argument-iteration idiom; the surrounding
    // macro/set/serialization language remains separately qualified.
    let template = "{{ args_name }}|{% for tool_call in calls %}{{ loop.index }}:{{ tool_call.name }}{% for args_name, args_value in tool_call.arguments|items %}[{{ args_name }}={{ args_value }}:{{ loop.index0 }}:{{ loop.revindex }}:{{ loop.first }}:{{ loop.last }}]{% endfor %}:{{ tool_call.name }}:{{ loop.length }};{% endfor %}|{{ args_name }}";
    let caller = json!({"args_name":"outside","calls":[{"name":"first","arguments":{"é\u{0}界":"borrowed 🦀","flag":false,"nothing":null,"number":2.5}},{"name":"empty","arguments":{}},{"name":"last","arguments":{"z":"tail"}}]});
    compare(
        template,
        &[],
        &serde_json::Map::new(),
        caller.as_object().unwrap(),
    );
    let template = "{% for x in groups %}{{ x.title }}:{% for x in x.values %}{{ x }}:{{ loop.index }};{% endfor %}{{ x.title }}:{{ loop.index }}|{% endfor %}{{ x }}";
    let caller = json!({"x":"external","groups":[{"title":"A","values":["é","界"]},{"title":"B","values":[]},{"title":"C","values":[false,3]}]});
    compare(
        template,
        &[],
        &serde_json::Map::new(),
        caller.as_object().unwrap(),
    );
}

#[test]
fn loop_workspace_follows_source_depth_and_assignment_arity() {
    let caller = json!({"rows":[["one","two","三","four","five"],["a","b","c","d","e"]],"items":["kept"],"map":{"z":1,"a":2}});
    compare(
        "{% for a,b,c,d,e in rows %}{{ a }}:{{ b }}:{{ c }}:{{ d }}:{{ e }};{% endfor %}|{% for key in map %}{{ key }};{% endfor %}",
        &[],
        &serde_json::Map::new(),
        caller.as_object().unwrap(),
    );
    let mut template = String::new();
    for depth in 0..12 {
        template.push_str(&format!("{{% for v{depth} in items %}}"));
    }
    template.push_str("{{ v0 }}:{{ v11 }}:{{ loop.length }}");
    for _ in 0..12 {
        template.push_str("{% endfor %}");
    }
    let source = ChatTemplatePlan::prepare_utf8(&template, "nested")
        .unwrap()
        .compile()
        .unwrap();
    let small = ChatTemplatePlan::prepare_utf8("{% for v in items %}{{ v }}{% endfor %}", "small")
        .unwrap()
        .compile()
        .unwrap();
    assert!(source.render_prefix_bytes().unwrap() > small.render_prefix_bytes().unwrap());
    compare(
        &template,
        &[],
        &serde_json::Map::new(),
        caller.as_object().unwrap(),
    );
    compare(
        "{% for part in text.split(',') %}{{ part }}:{{ loop.revindex0 }};{% endfor %}{% for v in none %}bad{% endfor %}{% for v in missing %}bad{% endfor %}",
        &[],
        &serde_json::Map::new(),
        json!({"text":"é,,界"}).as_object().unwrap(),
    );
}

#[test]
fn nested_loop_outputs_and_failed_destinations_outlive_all_borrowed_inputs() {
    let template = "{% for group in groups %}{% for name, value in group.items() %}{{ name + ':' + value }};{% endfor %}{% endfor %}{% if add_generation_prompt %}!{% endif %}";
    let source = ChatTemplatePlan::prepare_utf8(template, "loans")
        .unwrap()
        .compile()
        .unwrap();
    let caller = json!({"groups":[{"é\u{0}":"first"},{"界":"second","tail":"third"}]});
    let rendered = compare(
        template,
        &[],
        &serde_json::Map::new(),
        caller.as_object().unwrap(),
    );
    let failure = source
        .render_plan_with_context(
            ChatRenderContext::from_json(&[], None, caller.as_object()).unwrap(),
        )
        .unwrap()
        .fail_reservation(ChatRenderBuffer::Borrowed)
        .render()
        .unwrap_err();
    assert!(failure.retained_buffer_bytes() > 0);
    drop((source, caller));
    assert_eq!(rendered.prompt(true), "é\0:first;界:second;tail:third;!");
    assert!(failure.retained_buffer_bytes() > 0);
}

#[test]
fn wrong_unpack_lengths_and_unsupported_loop_protocols_refuse() {
    let source = ChatTemplatePlan::prepare_utf8(
        "{% for k,v in rows %}{{ k }}={{ v }}{% endfor %}",
        "unpack",
    )
    .unwrap()
    .compile()
    .unwrap();
    for rows in [
        json!([[1]]),
        json!([[1, 2, 3]]),
        json!([3]),
        json!([null]),
        json!(["ab"]),
    ] {
        let caller = json!({"rows": rows});
        assert!(
            source
                .render_plan_with_context(
                    ChatRenderContext::from_json(&[], None, caller.as_object()).unwrap()
                )
                .is_err()
        );
    }
    for template in [
        "{% for x in rows recursive %}{{ x }}{% endfor %}",
        "{% for x in rows if x %}{{ x }}{% endfor %}",
        "{% for x in rows %}{{ x }}{% else %}empty{% endfor %}",
        "{% for k,(a,b) in rows %}{{ k }}{% endfor %}",
    ] {
        assert!(match ChatTemplatePlan::prepare_utf8(template, "protocol") {
            Ok(plan) => plan.compile().is_err(),
            Err(_) => true,
        });
    }
}
