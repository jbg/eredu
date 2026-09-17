use super::*;

#[test]
fn borrowed_split_and_strip_match_ordinary_unicode_empty_and_directional_rules() {
    let template = "{{ text.split(',')[0] }}|{{ text.split(',')[-1] }}|{{ text.split(',')[12] is undefined }}|{{ text.split(',')[-12] is undefined }}|{{ text.split(',')|length }}|{{ text.split(',')|join('~') }}|{{ text.split('')|join('~') }}|{{ text.split()|join('~') }}|{{ text.split(none)|join('~') }}|{{ text.split() is sequence }}|{{ text.split() is mapping }}|{{ needle in text.split(',') }}|{% if text.split() %}nonempty{% endif %}";
    for (text, needle) in [
        ("", ""),
        (",é\u{0},界,", "界"),
        (" \t\u{2003}🦀  e\u{301}\n ", "🦀"),
        ("a,,b", ""),
        ("no separator", "absent"),
    ] {
        let caller = json!({"text":text,"needle":needle});
        compare(
            template,
            &[],
            &serde_json::Map::new(),
            caller.as_object().unwrap(),
        );
    }
    let template = "{{ text.strip() }}|{{ text.lstrip() }}|{{ text.rstrip() }}|{{ text.strip(chars) }}|{{ text.lstrip(chars) }}|{{ text.rstrip(chars) }}|{{ text.strip(none) }}|{{ text.lstrip(absent) }}|{{ text.rstrip('') }}";
    for (text, chars) in [
        ("", ""),
        (" \t\u{2003}é\u{0}界🦀\n ", "é🦀"),
        ("界ébodyé界", "é界"),
        ("éé", "é"),
        ("e\u{301}bodye\u{301}", "é"),
        ("\u{0}body\u{0}", "\u{0}"),
    ] {
        let caller = json!({"text":text,"chars":chars});
        compare(
            template,
            &[],
            &serde_json::Map::new(),
            caller.as_object().unwrap(),
        );
    }
}

#[test]
fn nanbeige_reasoning_content_chains_retain_output_after_input_and_source_retire() {
    // Exact expressions from the retained released Nanbeige template. The
    // surrounding fixture uses existing message iteration, not macro/set stubs.
    let template = "{% for message in messages %}{% if '</think>' in message.content %}{{ message.content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n')|trim }}|{{ message.content.split('</think>')[-1].lstrip('\\n').rstrip('\\n') }}{% else %}{{ message.content }}{% endif %};{% endfor %}{% if add_generation_prompt %}{{ nested.content.split('</think>')[-1].strip('\\n') + suffix }}{% endif %}";
    let messages = [
        json!({"role":"assistant","content":"<think>\n\né\u{0}界 reasoning\n</think>\n\n🦀 answer\n"}),
        json!({"role":"assistant","content":"prefix<think>discard<think>\nfinal\n</think>discard</think>\nlast\n"}),
        json!({"role":"user","content":"ordinary nonzero"}),
    ];
    let caller = json!({"nested":{"content":"<think>old</think>\nretained\n"},"suffix":"!"});
    let rendered = compare(
        template,
        &messages,
        &serde_json::Map::new(),
        caller.as_object().unwrap(),
    );
    assert_eq!(
        rendered.prompt(false),
        "é\u{0}界 reasoning|🦀 answer;final|last;ordinary nonzero;"
    );
    let source = ChatTemplatePlan::prepare_utf8(template, "nanbeige-expressions")
        .unwrap()
        .compile()
        .unwrap();
    let plan = source
        .render_plan_with_context(
            ChatRenderContext::from_json(&messages, None, caller.as_object()).unwrap(),
        )
        .unwrap();
    assert!(rendered.retained_buffer_bytes() <= plan.requirements().buffer_bytes());
    let failed = plan
        .fail_reservation(ChatRenderBuffer::WithPrompt)
        .render()
        .unwrap_err();
    drop((source, caller, messages));
    assert!(failed.retained_buffer_bytes() > 0);
    assert_eq!(
        rendered.prompt(true),
        "é\u{0}界 reasoning|🦀 answer;final|last;ordinary nonzero;retained!"
    );
}

#[test]
fn split_view_refuses_unqualified_storage_and_other_nanbeige_statements() {
    for template in [
        "{{ text.split(',', 1)[0] }}",
        "{{ text.split(sep=',')[0] }}",
        "{{ text.strip('x', 'y') }}",
    ] {
        let refused = match ChatTemplatePlan::prepare_utf8(template, "arity") {
            Ok(plan) => plan.compile().is_err(),
            Err(_) => true,
        };
        assert!(refused);
    }
    for (template, caller) in [
        (
            "{{ text.split(separator)[0] }}",
            json!({"text":"a,b","separator":","}),
        ),
        ("{{ text.split(',') }}", json!({"text":"a,b"})),
        (
            "{{ (left + right).strip() }}",
            json!({"left":" a","right":"b "}),
        ),
        ("{{ text.split(',')[0] }}", json!({"text":4})),
        (
            "{{ text.rstrip(chars) }}",
            json!({"text":"abc","chars":["a"]}),
        ),
    ] {
        let source = ChatTemplatePlan::prepare_utf8(template, "storage")
            .unwrap()
            .compile()
            .unwrap();
        assert!(
            source
                .render_plan_with_context(
                    ChatRenderContext::from_json(&[], None, caller.as_object()).unwrap()
                )
                .is_err()
        );
    }
    let skipped = compare(
        "{% if false and text.split(separator) %}wrong{% else %}skipped{% endif %}",
        &[],
        &serde_json::Map::new(),
        json!({"text":"a,b","separator":","}).as_object().unwrap(),
    );
    assert_eq!(skipped.prompt(false), "skipped");
    let full = include_str!("../../../../tests/fixtures/nanbeige/chat_template.jinja");
    // Macro/state/slice support is still absent. Method parity must not turn a
    // released-template support claim into an assertion about a reduced source.
    let refused = match ChatTemplatePlan::prepare_utf8(full, "nanbeige-full") {
        Ok(plan) => plan.compile().is_err(),
        Err(_) => true,
    };
    assert!(refused);
}
