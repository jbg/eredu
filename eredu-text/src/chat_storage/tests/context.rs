use super::*;

fn compare(
    template: &str,
    base: &[serde_json::Value],
    defaults: &serde_json::Map<String, serde_json::Value>,
    caller: &serde_json::Map<String, serde_json::Value>,
) -> PreparedChatRender {
    let source = ChatTemplatePlan::prepare_utf8(template, "context")
        .unwrap()
        .compile()
        .unwrap();
    let context = ChatRenderContext::from_json(base, Some(defaults), Some(caller)).unwrap();
    let plan = source.render_plan_with_context(context).unwrap();
    let bound = plan.requirements().buffer_bytes();
    let rendered = plan.render().unwrap();
    assert!(rendered.retained_buffer_bytes() <= bound);
    let mut ordinary = ordinary();
    ordinary.set_template_kwargs(defaults.clone());
    for generation in [false, true] {
        let expected = ordinary
            .apply_chat_template_json(
                crate::tokenizer::ModelChatTemplate::Single(template.into()),
                [base.to_vec()],
                None,
                "context",
                generation,
                Some(caller),
            )
            .unwrap();
        assert_eq!(
            rendered.prompt(generation),
            expected[0],
            "generation={generation}"
        );
    }
    rendered
}

#[test]
fn borrowed_chat_context_preserves_defaults_reserved_overrides_locals_and_scalar_spelling() {
    let base = json!([{"role":"user","content":"discarded"}]);
    let defaults = json!({"bos_token":"default", "messages":[{"role":"system","content":"default input"}],"add_generation_prompt":false,"message":"external default", "loop":"external loop"});
    let caller = json!({"bos_token":"  É界\u{0}  ","messages":[{"role":"user","content":"effective input"},{"role":"assistant","content":"second"}],"message":"external caller","add_generation_prompt":true,"suffix":"end"});
    let template = "{{ bos_token|trim }}|{{ message }}|{% for message in messages %}{% if loop.first %}first:{% endif %}{{ message.role + ':' + message.content }};{% endfor %}|{{ message }}|{{ loop }}|{% if add_generation_prompt %}{{ suffix }}{% endif %}";
    let rendered = compare(
        template,
        base.as_array().unwrap(),
        defaults.as_object().unwrap(),
        caller.as_object().unwrap(),
    );
    assert!(rendered.prompt(false).contains("effective input"));
    assert!(!rendered.prompt(false).contains("discarded"));
    assert_eq!(rendered.prompt(false), rendered.prompt(true));
    assert_eq!(rendered.generation_suffix(), "");
    drop((base, defaults, caller));
    assert!(
        rendered
            .prompt(true)
            .contains("external caller|external loop|end")
    );

    let template = "{{ value }}|{% if value %}truth{% endif %}|{{ value != none }}|{{ value != false }}|{{ missing }}|{{ missing != none }}";
    for value in [
        json!(null),
        json!(false),
        json!(true),
        json!(0),
        json!(-9223372036854775808i64),
        json!(u64::MAX),
        json!(-0.0),
        json!(1.0),
        json!(1.25),
        json!(f64::MIN_POSITIVE),
        json!(f64::MAX),
        json!(f64::from_bits(1)),
    ] {
        let mut defaults = serde_json::Map::new();
        defaults.insert("value".into(), value);
        compare(template, &[], &defaults, &serde_json::Map::new());
    }
    // Scalar reserved-key replacements are real values, not silently dropped.
    let defaults = json!({"messages":3,"add_generation_prompt":-0.0,"tools":"caller tools", "documents":false});
    compare(
        "{{ messages }}|{{ add_generation_prompt }}|{{ tools }}|{{ documents }}",
        &[],
        defaults.as_object().unwrap(),
        &serde_json::Map::new(),
    );
}

#[test]
fn structured_context_operations_remain_explicit_and_partial_buffers_outlive_borrows() {
    let source = ChatTemplatePlan::prepare_utf8("{{ prefix + suffix }}", "context")
        .unwrap()
        .compile()
        .unwrap();
    for buffer in [
        ChatRenderBuffer::Concat,
        ChatRenderBuffer::WithoutPrompt,
        ChatRenderBuffer::WithPrompt,
    ] {
        let caller = json!({"prefix":"nonempty Unicode 界", "suffix":" tail"});
        let context = ChatRenderContext::from_json(&[], None, caller.as_object()).unwrap();
        let plan = source.render_plan_with_context(context).unwrap();
        let failure = plan.fail_reservation(buffer).render().unwrap_err();
        drop(caller);
        assert!(failure.retained_buffer_bytes() > 0);
        drop(failure);
    }
    let nested = json!({"value":{"nested":"unimplemented"}});
    for template in ["{{ value }}", "{{ value['nested'][0] }}"] {
        let source = ChatTemplatePlan::prepare_utf8(template, "context")
            .unwrap()
            .compile()
            .unwrap();
        let context = ChatRenderContext::from_json(&[], None, nested.as_object()).unwrap();
        assert!(source.render_plan_with_context(context).is_err());
    }
    // A non-demanded structured binding does not alter the rendered program.
    compare(
        "{% if value %}nonempty{% endif %}",
        &[],
        &serde_json::Map::new(),
        nested.as_object().unwrap(),
    );
}

mod structured;

mod type_conditions;

mod length;

mod default;

mod join;

mod replace;

mod ordering;

mod membership;

mod methods;

mod string_views;

mod mapping_views;

mod nested_loops;

mod assignment;

mod messages;

mod namespaces;

mod macros;

mod conditional_expressions;

mod nested_macros;

mod arithmetic;

mod string;

mod range;

mod hf_json;

mod slices;

mod values;

mod clock;

mod adjacent;

mod scalar_overlay;

mod records;

#[path = "context/call_refusal.rs"]
mod call_refusal;
