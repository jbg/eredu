use super::*;

#[test]
fn mapping_get_preserves_absence_null_and_scalar_keys_like_installed_methods() {
    let template = "{{ map.get(key, fallback) }}|{{ map.get(key) is none }}|{{ map.get('null', fallback) is none }}|{{ map.get('missing', absent) is none }}|{{ map.get('false', true) }}|{{ map.get('zero', 8) }}|{{ map.get('empty', 'fallback') }}";
    for key in [
        json!("present"),
        json!("missing"),
        json!("null"),
        json!("é\u{0}界"),
        json!(1),
        json!(false),
        json!(null),
    ] {
        let caller = json!({"map":{"present":"actual","null":null,"false":false,"zero":0,"empty":"","é\u{0}界":"unicode"},"key":key,"fallback":"selected"});
        compare(
            template,
            &[],
            &serde_json::Map::new(),
            caller.as_object().unwrap(),
        );
    }
    compare("{{ map.get('missing', fallback).get('nested').value }}|{{ map.get(key_map.key).value }}",
        &[], &serde_json::Map::new(), json!({"map":{"found":{"value":"from-map"}},"fallback":{"nested":{"value":"from-fallback"}},"key_map":{"key":"found"}}).as_object().unwrap());
}

#[test]
fn released_message_get_calls_keep_eager_arguments_and_escaped_output_custody() {
    // Existing Nanbeige/Muse templates call these exact mapping methods.
    let template = "{% for message in messages %}{{ message.get('role', '') }}:{{ message.get('content', '') }}|{{ message.get('absent') is none }};{% endfor %}{{ map.get('missing', defaults.title) + suffix }}{% if add_generation_prompt %}|{{ map.get('present', 'wrong') }}{% endif %}";
    let messages = [
        json!({"role":"user","content":"é\u{0}界"}),
        json!({"role":"assistant","content":"🦀"}),
    ];
    let caller = json!({"map":{"present":"next"},"defaults":{"title":"borrowed"},"suffix":"!"});
    let rendered = compare(
        template,
        &messages,
        &serde_json::Map::new(),
        caller.as_object().unwrap(),
    );
    assert_eq!(
        rendered.prompt(true),
        "user:é\u{0}界|True;assistant:🦀|True;borrowed!|next"
    );
    let source = ChatTemplatePlan::prepare_utf8(template, "methods")
        .unwrap()
        .compile()
        .unwrap();
    let plan = source
        .render_plan_with_context(
            ChatRenderContext::from_json(&messages, None, caller.as_object()).unwrap(),
        )
        .unwrap();
    assert!(rendered.retained_buffer_bytes() <= plan.requirements().buffer_bytes());
    let failure = plan
        .fail_reservation(ChatRenderBuffer::WithPrompt)
        .render()
        .unwrap_err();
    drop((source, caller, messages));
    assert!(failure.retained_buffer_bytes() > 0);
    assert!(rendered.prompt(false).ends_with("borrowed!"));

    let caller = json!({"map":{"present":"found"}});
    let source = ChatTemplatePlan::prepare_utf8("{{ map.get('present', missing.child) }}", "eager")
        .unwrap()
        .compile()
        .unwrap();
    assert!(
        source
            .render_plan_with_context(
                ChatRenderContext::from_json(&[], None, caller.as_object()).unwrap()
            )
            .unwrap()
            .render()
            .is_err()
    );
    let skipped = compare(
        "{% if false and map.get('present', missing.child) %}wrong{% else %}skipped{% endif %}",
        &[],
        &serde_json::Map::new(),
        caller.as_object().unwrap(),
    );
    assert_eq!(skipped.prompt(false), "skipped");
}

#[test]
fn mapping_get_preserves_dynamic_receiver_errors_and_generated_keys() {
    for template in [
        "{{ map.get() }}",
        "{{ map.get('key', 1, 2) }}",
        "{{ map.get(key='key') }}",
        "{{ map.pop('key') }}",
    ] {
        compare_outcome(
            template,
            json!({"map":{},"text":"text"}).as_object().unwrap(),
        );
    }
    for value in [json!([1, 2]), json!("text"), json!(null), json!(4)] {
        let source = ChatTemplatePlan::prepare_utf8("{{ value.get('key') }}", "receiver")
            .unwrap()
            .compile()
            .unwrap();
        let caller = json!({"value":value});
        assert!(
            source
                .render_plan_with_context(
                    ChatRenderContext::from_json(&[], None, caller.as_object()).unwrap()
                )
                .unwrap()
                .render()
                .is_err()
        );
    }
    let caller = json!({"map":{"ab":"found"},"left":"a","right":"b"});
    let rendered = compare(
        "{{ map.get(left + right, 'default') }}",
        &[],
        &serde_json::Map::new(),
        caller.as_object().unwrap(),
    );
    assert_eq!(rendered.prompt(false), "found");
}

#[test]
fn released_string_edges_match_borrowed_unicode_sequences_and_short_circuit() {
    let template = "{{ text.startswith(prefix) }}|{{ text.endswith(suffix) }}|{{ text.startswith([]) }}|{{ text.endswith([]) }}{% if text.startswith('<tool_response>') and text.endswith('</tool_response>') %}|tool{% endif %}";
    for (text, prefix, suffix) in [
        (json!(""), json!(""), json!("")),
        (json!("é\u{0}界🦀"), json!("é\u{0}"), json!("界🦀")),
        (json!("e\u{301}"), json!("é"), json!("é")),
        (json!("short"), json!("too long"), json!("too long")),
        (
            json!("<tool_response>nonzero</tool_response>"),
            json!(["absent", "<tool_response>", null]),
            json!(["</tool_response>", false]),
        ),
        (json!("borrowed"), json!([]), json!(["x", "y"])),
    ] {
        let caller = json!({"text":text,"prefix":prefix,"suffix":suffix});
        compare(
            template,
            &[],
            &serde_json::Map::new(),
            caller.as_object().unwrap(),
        );
    }
    let caller = json!({"text":"é\u{0}界","prefix":"é","suffix":"界"});
    let retained = compare(
        "{% if text.startswith(prefix) or missing.child %}{{ text }}{% endif %}{% if add_generation_prompt and text.endswith(suffix) %}|edge{% endif %}",
        &[],
        &serde_json::Map::new(),
        caller.as_object().unwrap(),
    );
    drop(caller);
    assert_eq!(retained.prompt(true), "é\u{0}界|edge");

    for (text, prefix) in [
        (json!(4), json!("4")),
        (json!("abc"), json!([null, "a"])),
        (json!("abc"), json!(false)),
        (json!("abc"), json!({"a":1})),
    ] {
        let caller = json!({"text":text,"prefix":prefix});
        let source =
            ChatTemplatePlan::prepare_utf8("{{ text.startswith(prefix) }}", "edge-refusal")
                .unwrap()
                .compile()
                .unwrap();
        assert!(
            source
                .render_plan_with_context(
                    ChatRenderContext::from_json(&[], None, caller.as_object()).unwrap()
                )
                .unwrap()
                .render()
                .is_err()
        );
    }
    for template in [
        "{{ text.startswith() }}",
        "{{ text.endswith('x', 1) }}",
        "{{ text.startswith(prefix='x') }}",
    ] {
        compare_outcome(
            template,
            json!({"map":{},"text":"text"}).as_object().unwrap(),
        );
    }
}

#[test]
fn generated_string_edges_preserve_unicode_and_short_circuit_sequence_candidates() {
    let caller = json!({"left":"É", "right":"界🙂", "tail":"🙂"});
    for template in [
        "{% set text=left+right %}{{ text.startswith(left) }}|{{ text.endswith(tail) }}|{{ text.startswith(left+'界') }}|{{ text.endswith('界'+tail) }}",
        "{% set text=left+right %}{{ text.startswith(['absent',left+'界',none]) }}|{{ text.endswith(['absent','界'+tail,false]) }}",
        "{% set text=[left,right]|join('') %}{{ text.startswith([left]+[none]) }}|{{ text.endswith([]) }}|{{ text.startswith('') }}",
        "{% macro content() %}{{ left }}{{ right }}{% endmacro %}{{ content().endswith('界'+tail) }}",
    ] {
        compare(
            template,
            &[],
            &serde_json::Map::new(),
            caller.as_object().unwrap(),
        );
    }
}

#[test]
fn generation_blocks_share_normalized_source_and_whitespace_controls() {
    let caller = json!({"text":"É界🙂"});
    for template in [
        " before {%- generation -%}{{ text }}{%- endgeneration -%} after ",
        "{% generation %}{% if text %}{{ text }}{% endif %}{% endgeneration %}",
        "{% generation %}{% generation %}{{ text }}{% endgeneration %}{% endgeneration %}",
        "{# {% generation %} #}{{ text }}{# {% endgeneration %} #}",
    ] {
        compare(
            template,
            &[],
            &serde_json::Map::new(),
            caller.as_object().unwrap(),
        );
    }
}

#[test]
fn normalized_templates_retain_original_configuration_identity() {
    use crate::tokenizer::ModelChatTemplate;
    let raw = "{% generation %}{{ text }}{% endgeneration %}";
    let normalized = "{% if true %}{{ text }}{% endif %}";
    let source = ChatTemplatePlan::prepare_utf8(raw, "identity")
        .unwrap()
        .compile()
        .unwrap();
    assert!(source.matches_configuration(&ModelChatTemplate::Single(raw.into()), "identity"));
    assert!(
        !source.matches_configuration(&ModelChatTemplate::Single(normalized.into()), "identity")
    );
    assert!(!source.matches_configuration(&ModelChatTemplate::Single(raw.into()), "other"));
}
