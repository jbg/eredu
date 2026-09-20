use super::*;

#[test]
fn released_reasoning_assignments_use_the_shared_string_workers() {
    // Actual Nanbeige content/reasoning assignment idiom. Its macros,
    // namespace mutation and tool serialization are separate language features.
    // Borrow full input records through the existing caller collection; the
    // compact reserved messages adapter remains role/content-only.
    let template = "{% for message in entries %}{% set content = message.content %}{% set reasoning_content = '' %}{% if message.reasoning_content is string %}{% set reasoning_content = message.reasoning_content %}{% else %}{% if '</think>' in content %}{% set reasoning_content = content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n') %}{% set content = content.split('</think>')[-1].lstrip('\\n').rstrip('\\n') %}{% endif %}{% endif %}{% set reasoning_content = reasoning_content|trim %}{{ '<think>\\n' + reasoning_content + '\\n</think>\\n\\n' + content }}|{% endfor %}";
    let caller = json!({"entries":[
        {"role":"assistant","content":"<think>\né\u{0}界\n</think>\n\nanswer\n"},
        {"role":"assistant","content":"body","reasoning_content":" supplied 🦀 "},
        {"role":"assistant","content":"no reasoning"}
    ]});
    let rendered = compare(
        template,
        &[],
        &serde_json::Map::new(),
        caller.as_object().unwrap(),
    );
    assert!(
        rendered
            .prompt(false)
            .contains("<think>\né\u{0}界\n</think>\n\nanswer|")
    );
    assert!(
        rendered
            .prompt(false)
            .contains("<think>\n\n</think>\n\nno reasoning|")
    );
}

#[test]
fn assignments_shadow_restore_and_clear_only_the_current_iteration() {
    let caller = json!({"value":"caller", "rows":[true,false,true],"nested":["inner"],"object":{"function":{"name":"borrowed"}}});
    let template = "{{ value }}|{% set value = 'root' %}{% set total = 1 %}{% for active in rows %}{% if active %}{% set value = 'loop' %}{% set temporary = 'present' %}{% endif %}{{ value }}:{{ temporary|default('absent') }}:{% for child in nested %}{% set value = child %}{{ value }};{% endfor %}{{ value }}:{% set total = total + 1 %}{{ total }}|{% endfor %}{{ value }}:{{ total }}:{{ temporary|default('absent') }}|{% set tool_call = object %}{% set tool_call = tool_call.function %}{{ tool_call.name }}";
    let rendered = compare(
        template,
        &[],
        &serde_json::Map::new(),
        caller.as_object().unwrap(),
    );
    assert_eq!(
        rendered.prompt(false),
        "caller|loop:present:inner;loop:2|root:absent:inner;root:2|loop:present:inner;loop:2|root:1:absent|borrowed"
    );
}

#[test]
fn flat_assignments_preserve_input_loans_and_generated_output_custody() {
    let template = "{% set a,b,c,d,e = row %}{% set generated = a + ':' + b %}{% set saved = mapping %}{% set mapped = saved.items() %}{% set left,right = pair %}{% set a = e %}{{ generated }}:{{ c }}:{{ d }}:{{ a }}|{% for key,value in mapped %}{% set text = key + '=' + value %}{{ text }};{% endfor %}|{{ saved.tail }}:{{ left }}:{{ right }}";
    let caller = json!({"row":["é\u{0}","界",false,2.5,"last"],"mapping":{"first":"one","tail":"two"},"pair":["L","R"]});
    let rendered = compare(
        template,
        &[],
        &serde_json::Map::new(),
        caller.as_object().unwrap(),
    );
    let source = ChatTemplatePlan::prepare_utf8(template, "assignments")
        .unwrap()
        .compile()
        .unwrap();
    let context =
        ChatRenderContext::from_json(&[], None, Some(caller.as_object().unwrap())).unwrap();
    let failure = source
        .render_plan_with_context(context)
        .unwrap()
        .fail_reservation(ChatRenderBuffer::WithPrompt)
        .render()
        .unwrap_err();
    drop((source, caller));
    assert_eq!(
        rendered.prompt(false),
        "é\u{0}:界:False:2.5:last|first=one;tail=two;|two:L:R"
    );
    assert!(failure.retained_buffer_bytes() > 0);
}

#[test]
fn assignments_preserve_many_bindings_nested_targets_and_arity_errors() {
    let mut template = String::new();
    for index in 0..24 {
        template.push_str(&format!("{{% set v{index} = value %}}"));
    }
    template.push_str("{{ v0 }}:{{ v23 }}");
    let caller = json!({"value":"kept"});
    compare(
        &template,
        &[],
        &serde_json::Map::new(),
        caller.as_object().unwrap(),
    );
    for template in ["{% set a,b = value %}", "{% set a,b,c = value %}"] {
        let source = ChatTemplatePlan::prepare_utf8(template, "arity")
            .unwrap()
            .compile()
            .unwrap();
        let caller = json!({"value":[1]});
        let context =
            ChatRenderContext::from_json(&[], None, Some(caller.as_object().unwrap())).unwrap();
        assert!(
            source
                .render_plan_with_context(context)
                .unwrap()
                .render()
                .is_err()
        );
    }
    let caller = json!({"value":[7,[11,13]]});
    let rendered = compare(
        "{% set a,(b,c) = value %}{{ a }}:{{ b }}:{{ c }}",
        &[],
        &serde_json::Map::new(),
        caller.as_object().unwrap(),
    );
    assert_eq!(rendered.prompt(false), "7:11:13");
}

#[test]
fn block_output_capture_preserves_scope_nested_macros_loops_and_empty_text() {
    let template = "{% macro render(x) %}<{{ x }}>{% endmacro %}{% set outer %}{% set visible='yes' %}A{% set inner %}{{ render(value) }}{% for x in values %}{{ x }}{% endfor %}{% endset %}{{ inner|upper }}Z{% endset %}{{ outer }}|{{ visible }}|{% set empty %}{{ absent }}{% endset %}{{ empty|length }}|{% macro captured(x) %}{% set local %}{{ render(x) }}{% endset %}{{ local }}{% endmacro %}{{ captured('last') }}";
    let caller = json!({"value":"Straße","values":[7,-11,2.5]});
    let rendered = compare(
        template,
        &[],
        &serde_json::Map::new(),
        caller.as_object().unwrap(),
    );
    drop(caller);
    assert_eq!(rendered.prompt(false), "A<STRASSE>7-112.5Z|yes|0|<last>");
    assert!(rendered.retained_buffer_bytes() > 0);
}

#[test]
fn filtered_loops_preserve_outer_loop_scope_unpacking_and_selected_length() {
    let template = "{% set item='root' %}{% for outer in outers %}{% for key,value in mapping.items() if key != 'return' and loop.index == outer %}{{ loop.index }}/{{ loop.length }}:{{ key }}={{ value }}{% if not loop.last %},{% endif %}{% endfor %}|{% endfor %}{{ item }}";
    let caller = json!({"outers":[1,2],"mapping":{"a":7,"return":99,"z":-11}});
    let rendered = compare(
        template,
        &[],
        &serde_json::Map::new(),
        caller.as_object().unwrap(),
    );
    drop(caller);
    assert_eq!(
        rendered.prompt(false),
        "1/2:a=7,2/2:z=-11|1/2:a=7,2/2:z=-11|root"
    );
}
#[test]
fn nested_filtered_macro_calls_keep_each_collection_and_borrowed_input_alive() {
    let template = "{% macro f(xs) %}{% for x in xs if x>0 %}{{ x }}:{{ loop.index }}/{{ loop.length }};{% endfor %}{% endmacro %}{% for xs in rows if f(xs)|length>0 %}{{ f(xs) }}|{% endfor %}";
    let caller = json!({"rows":[[1,-2,3],[-7],[],[4]]});
    let rendered = compare(
        template,
        &[],
        &serde_json::Map::new(),
        caller.as_object().unwrap(),
    );
    drop(caller);
    assert_eq!(rendered.prompt(false), "1:1/2;3:2/2;|4:1/1;|");
}
