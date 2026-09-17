use super::*;

#[test]
fn source_bound_ordering_matches_ordinary_scalar_boundaries_and_borrowed_unicode() {
    let template = "{{ left < right }}|{{ left <= right }}|{{ left > right }}|{{ left >= right }}|{{ right < left }}|{{ right >= left }}{% if add_generation_prompt %}|generated{% endif %}";
    for (left, right) in [
        (json!(i64::MIN), json!(u64::MAX)),
        (json!(u64::MAX), json!(u64::MAX)),
        (json!(9007199254740993u64), json!(9007199254740992.0)),
        (json!(i64::MAX), json!(9223372036854775808.0)),
        (json!(u64::MAX), json!(18446744073709551616.0)),
        (json!(-0.0), json!(0.0)),
        (json!(f64::from_bits(1)), json!(0)),
        (json!(f64::MAX), json!(u64::MAX)),
        (json!(-1.25), json!(-1)),
        (json!(null), json!(null)),
        (json!(false), json!(true)),
        (json!(false), json!(0)),
        (json!(null), json!(false)),
        (json!(17), json!("17")),
        (json!("é\u{301}界\u{0}"), json!("é\u{301}界\u{0}🦀")),
        (json!("界"), json!("🦀")),
        (json!(""), json!("")),
    ] {
        let caller = json!({"left":left,"right":right});
        let rendered = compare(
            template,
            &[],
            &serde_json::Map::new(),
            caller.as_object().unwrap(),
        );
        drop(caller);
        assert!(rendered.prompt(true).ends_with("|generated"));
    }
    compare(
        "{{ absent < none }}|{{ absent <= missing }}|{{ missing > absent }}|{{ none >= absent }}",
        &[],
        &serde_json::Map::new(),
        &serde_json::Map::new(),
    );
}

#[test]
fn released_length_conditions_share_short_circuit_and_retain_paid_outputs_on_refusal() {
    // Existing released Nanbeige and Kimi K2 templates use these length
    // comparisons; the rest of their template programs remain separately scoped.
    let template = "{% if messages|length > 0 and spec.type|length > 0 %}{{ spec.left + spec.right }}{% endif %}|{% if spec.left < spec.right or missing.child %}ordered{% endif %}|{{ messages|count >= 2 }}|{{ spec.left|trim <= spec.right }}{% if add_generation_prompt %}|{{ spec.type|length < 3 }}{% endif %}";
    let defaults = json!({"spec":{"type":[],"left":"wrong","right":"default"}});
    let caller = json!({"spec":{"type":["string","number"],"left":"é\u{0}","right":"界"}});
    let messages = [
        json!({"role":"user","content":"nonzero"}),
        json!({"role":"assistant","content":"second"}),
    ];
    let rendered = compare(
        template,
        &messages,
        defaults.as_object().unwrap(),
        caller.as_object().unwrap(),
    );
    assert_eq!(rendered.prompt(false), "é\u{0}界|ordered|True|True");
    assert_eq!(rendered.prompt(true), "é\u{0}界|ordered|True|True|True");
    let source = ChatTemplatePlan::prepare_utf8(template, "ordering")
        .unwrap()
        .compile()
        .unwrap();
    let context =
        ChatRenderContext::from_json(&messages, defaults.as_object(), caller.as_object()).unwrap();
    let plan = source.render_plan_with_context(context).unwrap();
    assert!(rendered.retained_buffer_bytes() <= plan.requirements().buffer_bytes());
    let failure = plan
        .fail_reservation(ChatRenderBuffer::WithPrompt)
        .render()
        .unwrap_err();
    drop((source, caller, defaults, messages));
    assert!(failure.retained_buffer_bytes() > 0);
    assert_eq!(rendered.prompt(true), "é\u{0}界|ordered|True|True|True");

    let empty = compare(
        "{% if messages|length > 0 and missing.child %}wrong{% else %}empty{% endif %}",
        &[],
        &serde_json::Map::new(),
        &serde_json::Map::new(),
    );
    assert_eq!(empty.prompt(false), "empty");
}

#[test]
fn ordering_does_not_admit_recursive_containers_or_chained_comparisons() {
    for template in ["{{ left < right }}", "{{ left <= right }}"] {
        for caller in [
            json!({"left":[1],"right":[2]}),
            json!({"left":{"a":1},"right":{"a":2}}),
        ] {
            let source = ChatTemplatePlan::prepare_utf8(template, "ordering")
                .unwrap()
                .compile()
                .unwrap();
            let context = ChatRenderContext::from_json(&[], None, caller.as_object()).unwrap();
            assert!(source.render_plan_with_context(context).is_err());
        }
    }
    assert!(
        ChatTemplatePlan::prepare_utf8("{{ left < middle < right }}", "ordering")
            .unwrap()
            .compile()
            .is_err()
    );
}

#[test]
fn generated_text_comparisons_preserve_ordinary_order_and_equality(){
    let caller=json!({"left":"É","right":"界🙂"});
    for template in [
        "{{ (left+right) < right }}|{{ (left+right) == (left+right) }}|{{ left != left+right }}",
        "{{ (left|replace('É','終')) >= right }}|{{ (left+right) > 7 }}|{{ (left+right) == 7 }}",
        "{% macro text() %}{{ left }}{{ right }}{% endmacro %}{{ text() == left+right }}|{{ text() > right }}",
    ]{compare(template,&[],&serde_json::Map::new(),caller.as_object().unwrap());}
}
