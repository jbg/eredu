use super::*;
use crate::chat_storage::{ChatScalarBinding, ChatScalarBindingValue};

#[test]
fn borrowed_scalar_overlay_preserves_policy_precedence_local_scope_and_escaped_render() {
    let template = "{{ reasoning_effort }}|{{ enable_thinking }}|{% set reasoning_effort = 'local' %}{{ reasoning_effort }}|{% if add_generation_prompt %}suffix{% endif %}";
    let source = ChatTemplatePlan::prepare_utf8(template, "scalar-overlay")
        .unwrap()
        .compile()
        .unwrap();
    let caller = json!({"reasoning_effort":"caller", "enable_thinking":false, "add_generation_prompt":false});
    let defaults = json!({"reasoning_effort":"default", "enable_thinking":false});
    let actual = String::from("É界🙂 high");
    let bindings = [
        ChatScalarBinding {
            name: "reasoning_effort",
            value: ChatScalarBindingValue::Text("first"),
        },
        ChatScalarBinding {
            name: "enable_thinking",
            value: ChatScalarBindingValue::Bool(true),
        },
        ChatScalarBinding {
            name: "reasoning_effort",
            value: ChatScalarBindingValue::Text(&actual),
        },
        ChatScalarBinding {
            name: "add_generation_prompt",
            value: ChatScalarBindingValue::Bool(true),
        },
    ];
    let context = ChatRenderContext::from_json(&[], defaults.as_object(), caller.as_object())
        .unwrap()
        .with_scalar_overrides(&bindings);
    assert!(!context.is_plain());
    let plan = source.render_plan_with_context(context).unwrap();
    let bound = plan.requirements().buffer_bytes();
    let rendered = plan.render().unwrap();
    assert!(rendered.retained_buffer_bytes() <= bound);
    let mut ordinary = ordinary();
    ordinary.set_template_kwargs(defaults.as_object().unwrap().clone());
    let mut merged = caller.as_object().unwrap().clone();
    for binding in bindings {
        let value = match binding.value {
            ChatScalarBindingValue::Bool(value) => serde_json::Value::Bool(value),
            ChatScalarBindingValue::Text(value) => serde_json::Value::String(value.into()),
        };
        merged.insert(binding.name.into(), value);
    }
    for generation in [false, true] {
        let expected = ordinary
            .apply_chat_template_json(
                crate::tokenizer::ModelChatTemplate::Single(template.into()),
                [Vec::new()],
                None,
                "scalar-overlay",
                generation,
                Some(&merged),
            )
            .unwrap();
        assert_eq!(rendered.prompt(generation), expected[0]);
    }
    drop((actual, caller, defaults, merged));
    assert_eq!(rendered.prompt(true), "É界🙂 high|True|local|suffix");
    assert_eq!(rendered.prompt(false), rendered.prompt(true));
    assert_eq!(rendered.generation_suffix(), "");
}
