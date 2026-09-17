use super::*;

fn source() -> PreparedTemplate {
    CompilePlan::prepare(
        recipe::SOURCE,
        TemplateName::Single("chat"),
        TemplateSettings::text_chat(),
    )
    .unwrap()
    .compile()
    .unwrap()
}

#[test]
fn exact_source_settings_name_and_real_source_reserve_prefixes() {
    assert!(matches!(
        CompilePlan::prepare(
            "",
            TemplateName::Single("chat"),
            TemplateSettings::text_chat()
        ),
        Err(SourceError::Profile)
    ));
    assert!(matches!(
        CompilePlan::prepare(
            recipe::SOURCE,
            TemplateName::Single("chat.html"),
            TemplateSettings::text_chat()
        ),
        Err(SourceError::Settings)
    ));
    let mut settings = TemplateSettings::text_chat();
    settings.lstrip_blocks = false;
    assert!(matches!(
        CompilePlan::prepare(recipe::SOURCE, TemplateName::Single("chat"), settings),
        Err(SourceError::Settings)
    ));
    let source = CompilePlan::prepare(
        recipe::SOURCE,
        TemplateName::Default("chat.html"),
        TemplateSettings::text_chat(),
    )
    .unwrap()
    .compile()
    .unwrap();
    assert_eq!(source.name(), "chat.html::chat_template::default");
    assert_eq!(source.source(), recipe::SOURCE);
    assert_eq!(source.instruction_count(), 39);
    #[cfg(feature = "development-closed-chat")]
    for (index, buffer) in [
        SourceBuffer::Instructions,
        SourceBuffer::Bytes,
        SourceBuffer::Locations,
    ]
    .into_iter()
    .enumerate()
    {
        let plan = CompilePlan::prepare(
            recipe::SOURCE,
            TemplateName::Single("chat"),
            TemplateSettings::text_chat(),
        )
        .unwrap();
        let error = plan.fail_reservation(buffer).compile().unwrap_err();
        assert!(matches!(error.cause(), CompileCause::Reserve(actual, _) if *actual==buffer));
        assert_eq!(error.retained_buffer_bytes() > 0, index != 0);
        if index == 2 {
            assert!(error.retained_buffer_bytes() > recipe::BYTES.len());
        }
        assert!(std::error::Error::source(error.cause()).is_some());
    }
}

#[cfg(feature = "json")]
#[test]
fn shared_dispatch_matches_ordinary_empty_system_unicode_and_generation_settings() {
    let source = source();
    let mut ordinary = crate::Environment::new();
    ordinary.set_trim_blocks(true);
    ordinary.set_lstrip_blocks(true);
    ordinary.set_keep_trailing_newline(false);
    ordinary.set_undefined_behavior(crate::UndefinedBehavior::Lenient);
    ordinary.add_template("chat", recipe::SOURCE).unwrap();
    let template = ordinary.get_template("chat").unwrap();
    let cases = [
        serde_json::json!([]),
        serde_json::json!([{"role":"system","content":""}]),
        serde_json::json!([{"role":"user","content":"Hello e\u{301}\0 世界\n<|im_end|>"}]),
        serde_json::json!([{"role":"system","content":"Rules"},{"role":"user","content":"question"},{"role":"assistant","content":"answer"}]),
        serde_json::json!([{"role":"","content":""},{"role":"assistant","content":"é"}]),
    ];
    for input in cases {
        let messages = Messages::from_json(input.as_array().unwrap()).unwrap();
        let plan = RenderPlan::prepare(&source, messages).unwrap();
        let requirements = plan.requirements();
        assert_eq!(
            requirements.capacity(RenderBuffer::Concat),
            20 * messages.len()
        );
        let rendered = plan.render().unwrap();
        for generation in [false, true] {
            let expected = template
                .render(serde_json::json!({"messages":input,"add_generation_prompt":generation}))
                .unwrap();
            assert_eq!(rendered.prompt(generation), expected);
            assert_eq!(
                rendered.prompt(generation).len(),
                requirements.capacity(if generation {
                    RenderBuffer::WithPrompt
                } else {
                    RenderBuffer::WithoutPrompt
                })
            );
        }
        let expected_suffix = rendered
            .prompt(true)
            .strip_prefix(rendered.prompt(false))
            .unwrap_or("");
        assert_eq!(rendered.generation_suffix(), expected_suffix);
        assert_eq!(
            rendered.retained_buffer_bytes(),
            requirements.buffer_bytes()
        );
    }
}

#[cfg(feature = "json")]
#[test]
fn actual_borrowed_json_and_typed_messages_reject_unread_fields_without_a_copy() {
    let source = source();
    let raw = serde_json::json!([{"role":"user","content":"héllo"}]);
    let messages = Messages::from_json(raw.as_array().unwrap()).unwrap();
    let typed = [TextMessage {
        role: raw[0]["role"].as_str().unwrap(),
        content: raw[0]["content"].as_str().unwrap(),
    }];
    let actual = RenderPlan::prepare(&source, messages)
        .unwrap()
        .render()
        .unwrap();
    let expected = RenderPlan::prepare(&source, Messages::from_text(&typed))
        .unwrap()
        .render()
        .unwrap();
    assert_eq!(actual.prompt(false), expected.prompt(false));
    assert_eq!(actual.prompt(true), expected.prompt(true));
    assert_eq!(
        messages.get(0).unwrap().content.as_ptr(),
        typed[0].content.as_ptr()
    );
    for invalid in [
        serde_json::json!([null]),
        serde_json::json!([42]),
        serde_json::json!([[]]),
        serde_json::json!(["user"]),
    ] {
        assert_eq!(
            Messages::from_json(invalid.as_array().unwrap()).unwrap_err(),
            InputError::MessageProfile
        );
    }
}

#[cfg(feature = "development-closed-chat")]
#[test]
fn every_real_render_reserve_keeps_completed_destination_prefixes() {
    let source = source();
    let raw = [TextMessage {
        role: "user",
        content: "nonempty",
    }];
    for (index, buffer) in [
        RenderBuffer::Operands,
        RenderBuffer::Frames,
        RenderBuffer::Locals,
        RenderBuffer::Concat,
        RenderBuffer::WithoutPrompt,
        RenderBuffer::WithPrompt,
    ]
    .into_iter()
    .enumerate()
    {
        let plan = RenderPlan::prepare(&source, Messages::from_text(&raw)).unwrap();
        let requirements = plan.requirements();
        let error = plan.fail_reservation(buffer).render().unwrap_err();
        assert!(matches!(error.cause(),RenderCause::Reserve(actual,_) if *actual==buffer));
        assert_eq!(error.retained_buffer_bytes() > 0, index != 0);
        assert!(error.retained_buffer_bytes() < requirements.buffer_bytes());
        assert!(std::error::Error::source(error.cause()).is_some());
        assert_eq!(source.source(), recipe::SOURCE);
    }
}

#[cfg(feature = "json")]
#[test]
fn ordinary_recursive_loop_restores_parent_frames_through_shared_dispatch() {
    let mut env = crate::Environment::new();
    env.add_template(
        "recursive",
        "{% for node in nodes recursive %}{{ loop.depth }}:{{ node.name }}[{{ loop(node.children) }}]{% endfor %}",
    )
    .unwrap();
    let nodes = serde_json::json!([
        {"name":"A","children":[
            {"name":"B","children":[]},
            {"name":"C","children":[{"name":"D","children":[]}]}
        ]},
        {"name":"E","children":[]}
    ]);
    assert_eq!(
        env.get_template("recursive")
            .unwrap()
            .render(serde_json::json!({"nodes":nodes}))
            .unwrap(),
        "1:A[2:B[]2:C[3:D[]]]1:E[]"
    );
}

#[test]
fn ordinary_formatter_and_strict_undefined_keep_original_emit_contract() {
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };
    let calls = Arc::new(AtomicUsize::new(0));
    let observed = Arc::clone(&calls);
    let mut env = crate::Environment::new();
    env.set_formatter(move |out, state, value| {
        observed.fetch_add(1, Ordering::SeqCst);
        crate::escape_formatter(
            out,
            state,
            &crate::value::Value::from(format!("[{}]", value)),
        )
    });
    env.add_template("ordinary.html", "{{ payload }}|{{ 2+3 }}")
        .unwrap();
    assert_eq!(
        env.get_template("ordinary.html")
            .unwrap()
            .render(crate::context!(payload => "<&>"))
            .unwrap(),
        "[&lt;&amp;&gt;]|[5]"
    );
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    env.set_undefined_behavior(crate::UndefinedBehavior::Strict);
    env.add_template("strict", "\n{{ missing }}").unwrap();
    let error = env.get_template("strict").unwrap().render(()).unwrap_err();
    assert_eq!(error.kind(), crate::ErrorKind::UndefinedError);
    assert_eq!(error.name(), Some("strict"));
    assert_eq!(error.line(), Some(2));
    // Strict validation precedes even the ordinary custom formatter callback.
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

#[cfg(feature = "fuel")]
#[test]
fn ordinary_fuel_is_consumed_before_shared_instruction_dispatch() {
    let mut env = crate::Environment::new();
    env.set_fuel(Some(1));
    env.add_template("fuel", "{{ payload }}").unwrap();
    let error = env
        .get_template("fuel")
        .unwrap()
        .render(crate::context!(payload => "nonempty"))
        .unwrap_err();
    assert_eq!(error.kind(), crate::ErrorKind::OutOfFuel);
    assert_eq!(error.name(), Some("fuel"));
    env.set_fuel(Some(100));
    assert_eq!(
        env.get_template("fuel")
            .unwrap()
            .render(crate::context!(payload => "nonempty"))
            .unwrap(),
        "nonempty"
    );
}

#[cfg(feature = "loop_controls")]
#[test]
fn ordinary_break_continue_keep_shared_loop_frame_transitions() {
    let mut env = crate::Environment::new();
    env.add_template(
        "controls",
        "{% for n in values %}{% if n == 2 %}{% continue %}{% endif %}{% if n == 4 %}{% break %}{% endif %}{{ n }}{% endfor %}",
    )
    .unwrap();
    assert_eq!(
        env.get_template("controls")
            .unwrap()
            .render(crate::context!(values => [1, 2, 3, 4, 5]))
            .unwrap(),
        "13"
    );
}
