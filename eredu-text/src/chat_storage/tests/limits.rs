use super::*;

fn source(template: &str, limits: ChatRenderLimits) -> PreparedChatTemplate {
    ChatTemplatePlan::prepare_utf8(template, "limits")
        .unwrap()
        .with_render_limits(limits)
        .compile()
        .unwrap()
}

#[test]
fn input_size_and_depth_are_checked_before_template_execution() {
    let context = ChatRenderContext::from_json(&[], None, None).unwrap();
    let source = source("{{ unknown_function() }}", ChatRenderLimits::default());
    let plan = source.render_plan_with_context(context).unwrap();
    assert!(plan.render().unwrap_err().cause().is_unknown_function());

    let limits = ChatRenderLimits {
        max_input_bytes: 8,
        ..Default::default()
    };
    assert!(matches!(
        self::source("", limits).render_plan_with_context(context),
        Err(ChatRenderPlanError::InputBytes)
    ));
    let values = json!({"value":[[[7]]]});
    let context = ChatRenderContext::from_json(&[], None, values.as_object()).unwrap();
    let limits = ChatRenderLimits {
        max_input_depth: 3,
        ..Default::default()
    };
    assert!(matches!(
        self::source("{{ value }}", limits).render_plan_with_context(context),
        Err(ChatRenderPlanError::InputDepth)
    ));
}

#[test]
fn output_limit_keeps_utf8_prefix_and_is_not_a_soft_template_rejection() {
    let limits = ChatRenderLimits {
        max_output_bytes: 4,
        ..Default::default()
    };
    let source = source("é{% if add_generation_prompt %}界{% endif %}", limits);
    let failure = source
        .render_plan(ChatMessages::from_text(&[]))
        .unwrap()
        .render()
        .unwrap_err();
    assert!(matches!(failure.cause(), ChatRenderCause::OutputLimit));
    assert!(!failure.cause().is_template_rejection());
    assert_eq!(failure.output[0], "é".as_bytes());
    assert_eq!(failure.output[1], "é".as_bytes());
    assert_eq!(failure.retained_buffer_bytes(), 8);
}

#[test]
fn fuel_and_recursion_refusals_are_hard_failures_with_output_custody() {
    let source = source(
        "{% for n in range(1000) %}{{ n }}{% endfor %}",
        ChatRenderLimits {
            fuel: 30,
            max_output_bytes: 128,
            ..Default::default()
        },
    );
    let failure = source
        .render_plan(ChatMessages::from_text(&[]))
        .unwrap()
        .render()
        .unwrap_err();
    assert!(
        matches!(failure.cause(), ChatRenderCause::Template(e) if e.kind() == minijinja::ErrorKind::OutOfFuel)
    );
    assert!(!failure.cause().is_template_rejection());
    assert_eq!(failure.retained_buffer_bytes(), 128);
    let source = self::source(
        "{% macro recurse(n) %}{{ n }}{% if n %}{{ recurse(n-1) }}{% endif %}{% endmacro %}{{ recurse(30) }}",
        ChatRenderLimits {
            recursion_limit: 8,
            ..Default::default()
        },
    );
    let failure = source
        .render_plan(ChatMessages::from_text(&[]))
        .unwrap()
        .render()
        .unwrap_err();
    assert!(!failure.cause().is_template_rejection());
    assert!(failure.retained_buffer_bytes() > 0);
}

#[test]
fn dependency_headroom_is_configurable_separately_from_output_limits() {
    let estimate = ChatMemoryEstimate {
        fixed_bytes: 17,
        bytes_per_source_byte: 3,
        bytes_per_input_byte: 5,
    };
    let plan = ChatTemplatePlan::prepare_utf8("abc", "id")
        .unwrap()
        .with_memory_estimate(estimate)
        .unwrap();
    assert_eq!(plan.requirements().required_bytes(), 17 + 3 * 5);
    let source = plan
        .with_render_limits(ChatRenderLimits {
            max_output_bytes: 7,
            ..Default::default()
        })
        .compile()
        .unwrap();
    let context = ChatRenderContext::from_json(&[], None, None).unwrap();
    let input = context.input_bytes(usize::MAX, 128).unwrap();
    let plan = source.render_plan_with_context(context).unwrap();
    assert_eq!(plan.requirements().buffer_bytes(), 14);
    assert_eq!(plan.requirements().required_bytes(), 14 + 17 + input * 5);
    assert_eq!(plan.render().unwrap().prompt(false), "abc");
}
