use super::*;

#[test]
fn type_predicates_distinguish_actual_scalar_container_and_iterable_values() {
    let template = r#"{% if value is boolean %}B{% endif %}{% if value is number %}N{% endif %}{% if value is integer %}I{% endif %}{% if value is float %}F{% endif %}{% if value is string %}S{% endif %}{% if value is sequence %}Q{% endif %}{% if value is mapping %}M{% endif %}{% if value is iterable %}E{% endif %}{% if value is true %}T{% endif %}{% if value is false %}f{% endif %}"#;
    for (value, expected) in [
        (json!(null), "E"),
        (json!(false), "Bf"),
        (json!(true), "BT"),
        (json!(0), "NI"),
        (json!(i64::MIN), "NI"),
        (json!(u64::MAX), "NI"),
        (json!(-0.0), "NF"),
        (json!(1.0), "NF"),
        (json!(1.25), "NF"),
        (json!(""), "SE"),
        (json!("Unicode 界"), "SE"),
        (json!([]), "QE"),
        (json!([null, false, "value"]), "QE"),
        (json!({}), "ME"),
        (json!({"content":"actual map"}), "ME"),
    ] {
        let caller = json!({"value":value});
        let render = compare(
            template,
            &[],
            &serde_json::Map::new(),
            caller.as_object().unwrap(),
        );
        assert_eq!(render.prompt(false), expected);
        drop(caller);
        assert_eq!(render.prompt(true), expected);
    }
    // Undefined is iterable in MiniJinja (an empty iteration), and unlike a
    // false boolean, integer zero is neither `is true` nor `is false`.
    let render = compare(
        template,
        &[],
        &serde_json::Map::new(),
        &serde_json::Map::new(),
    );
    assert_eq!(render.prompt(false), "E");
    let caller = json!({"value":null});
    let render = compare(
        "{% for message in messages %}{{ message is mapping }}:{{ loop is mapping }}:{{ loop is iterable }};{% endfor %}",
        &[json!({"role":"user","content":"actual row"})],
        &serde_json::Map::new(),
        caller.as_object().unwrap(),
    );
    assert_eq!(render.prompt(false), "True:True:True;");
}

#[test]
fn short_circuit_conditions_preserve_selected_borrowed_operands_and_error_lifetimes() {
    let template = r#"{{ (cfg.preferred or cfg.fallback).text + ':' + (cfg.empty or cfg.other).text }}|{{ cfg.truthy or missing.child }}|{{ cfg.empty and missing.child }}|{% if cfg.argument is mapping or (cfg.argument is iterable and cfg.argument is not string) %}structured{% endif %}|{% for message in messages %}{{ ((cfg.empty or cfg.fallback).text + message.content + (cfg.preferred or cfg.other).text)|trim }};{% endfor %}{% if add_generation_prompt is true or cfg.reasoning is false %}{{ (cfg.empty or cfg.fallback).text + (cfg.preferred or cfg.other).text }}{% endif %}"#;
    let caller = json!({"cfg":{"preferred":{"text":"  first 界"},"fallback":{"text":"fallback"},"other":{"text":"other"},"empty":"","truthy":"kept","argument":{"name":"tool"},"reasoning":true}});
    let base = [
        json!({"role":"user","content":"x"}),
        json!({"role":"assistant","content":"y"}),
    ];
    let rendered = compare(
        template,
        &base,
        &serde_json::Map::new(),
        caller.as_object().unwrap(),
    );
    assert!(
        rendered
            .prompt(false)
            .starts_with("  first 界:other|kept||structured|")
    );
    assert!(rendered.prompt(true).ends_with("fallback  first 界"));
    let source = ChatTemplatePlan::prepare_utf8(template, "context")
        .unwrap()
        .compile()
        .unwrap();
    let context = ChatRenderContext::from_json(&base, None, caller.as_object()).unwrap();
    let failure = source
        .render_plan_with_context(context)
        .unwrap()
        .fail_reservation(ChatRenderBuffer::WithPrompt)
        .render()
        .unwrap_err();
    drop((caller, base, source));
    assert!(failure.retained_buffer_bytes() > 0);
    assert!(rendered.prompt(true).ends_with("fallback  first 界"));
    drop((failure, rendered));

    let template = "{{ value or missing.child }}";
    let caller = json!({"value":""});
    let source = ChatTemplatePlan::prepare_utf8(template, "context")
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
    assert!(
        ordinary()
            .apply_chat_template_json(
                crate::tokenizer::ModelChatTemplate::Single(template.into()),
                [vec![]],
                None,
                "context",
                false,
                caller.as_object()
            )
            .is_err()
    );
}

#[test]
fn ordinary_type_tests_preserve_custom_object_queries_and_test_override() {
    use minijinja::value::{Enumerator, Object, ObjectRepr, Value};
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    #[derive(Debug)]
    struct Observed {
        representations: Arc<AtomicUsize>,
        iterations: Arc<AtomicUsize>,
    }
    impl Object for Observed {
        fn repr(self: &Arc<Self>) -> ObjectRepr {
            self.representations.fetch_add(1, Ordering::SeqCst);
            ObjectRepr::Map
        }
        fn enumerate(self: &Arc<Self>) -> Enumerator {
            self.iterations.fetch_add(1, Ordering::SeqCst);
            Enumerator::NonEnumerable
        }
    }
    let representations = Arc::new(AtomicUsize::new(0));
    let iterations = Arc::new(AtomicUsize::new(0));
    let value = Value::from_object(Observed {
        representations: representations.clone(),
        iterations: iterations.clone(),
    });
    assert!(!minijinja::tests::is_integer(&value));
    assert!(!minijinja::tests::is_float(&value));
    assert!(!minijinja::tests::is_true(&value));
    assert!(!minijinja::tests::is_false(&value));
    assert_eq!(representations.load(Ordering::SeqCst), 0);
    assert_eq!(iterations.load(Ordering::SeqCst), 0);
    assert!(minijinja::tests::is_mapping(&value));
    assert_eq!(representations.load(Ordering::SeqCst), 1);
    assert_eq!(iterations.load(Ordering::SeqCst), 0);
    // A map representation alone does not make a custom object iterable.
    assert!(!minijinja::tests::is_iterable(&value));
    assert_eq!(iterations.load(Ordering::SeqCst), 1);

    let calls = Arc::new(AtomicUsize::new(0));
    let callback = calls.clone();
    let mut env = minijinja::Environment::new();
    env.add_test("mapping", move |_: &Value| {
        callback.fetch_add(1, Ordering::SeqCst);
        false
    });
    let rendered = env
        .render_str(
            "{{ item is mapping or 'override' }}",
            minijinja::context! {item=>value},
        )
        .unwrap();
    assert_eq!(rendered, "override");
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    env.set_undefined_behavior(minijinja::UndefinedBehavior::Strict);
    let error = env
        .render_str("{{ missing or 'unreachable' }}", ())
        .unwrap_err();
    assert_eq!(error.kind(), minijinja::ErrorKind::UndefinedError);
}
