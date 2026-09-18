use super::*;
use crate::chat_storage::{ChatInputArray, ChatRecordField as F, ChatRecordValue as V};

#[test]
fn borrowed_json_tools_preserve_nested_numbers_and_variable_precedence() {
    let template = "{{ tools|tojson(sort_keys=true) }}|{% for tool in tools %}{{ tool.function.parameters.properties.value.minimum }};{% endfor %}{% if add_generation_prompt %}tail{% endif %}";
    let source = ChatTemplatePlan::prepare_utf8(template, "json-tools")
        .unwrap().compile().unwrap();
    let tools = vec![json!({"type":"function", "function":{"name":"reading", "parameters":{"type":"object", "properties":{"value":{"type":"number", "minimum":-17.25}}}}})];
    let defaults = json!({"tools":[{"function":{"name":"default", "parameters":{"properties":{"value":{"minimum":9}}}}}]});
    let overrides = json!({"tools":[{"function":{"name":"override", "parameters":{"properties":{"value":{"minimum":31}}}}}]});
    let mut outputs = Vec::new();
    for (default_map, override_map) in [
        (None, None),
        (defaults.as_object(), None),
        (defaults.as_object(), overrides.as_object()),
    ] {
        let context = ChatRenderContext::from_json(&[], default_map, override_map)
            .unwrap().with_tools(ChatInputArray::Json(&tools));
        let plan = source.render_plan_with_context(context).unwrap();
        let bound = plan.requirements().buffer_bytes();
        let rendered = plan.render().unwrap();
        assert!(rendered.retained_buffer_bytes() <= bound);
        let mut ordinary = ordinary();
        if let Some(defaults) = default_map {
            ordinary.set_template_kwargs(defaults.clone());
        }
        for generation in [false, true] {
            let expected = ordinary.apply_chat_template_json(
                crate::tokenizer::ModelChatTemplate::Single(template.into()),
                [Vec::new()], Some(&tools), "json-tools", generation, override_map,
            ).unwrap();
            assert_eq!(rendered.prompt(generation), expected[0]);
        }
        outputs.push(rendered);
    }
    drop((tools, defaults, overrides));
    assert!(outputs[0].prompt(true).contains("-17.25"));
    assert!(outputs[1].prompt(true).contains("default"));
    assert!(outputs[2].prompt(true).contains("override"));
}

#[test]
fn borrowed_records_share_nested_json_workers_and_retire_before_escaped_render() {
    let template = "{% for message in messages %}{{ message.role }}|{{ message.reasoning_content|default('') }}|{{ message.tool_calls[0].function.arguments|tojson }}|{{ message.keys()|list|join(',') }}|{{ message.content }}{% endfor %}|{{ tools|tojson(sort_keys=true) }}|{{ messages|tojson }}{% if add_generation_prompt %}tail{% endif %}";
    let source = ChatTemplatePlan::prepare_utf8(template, "record-source")
        .unwrap()
        .compile()
        .unwrap();
    let text = String::from("λ\0quoted\"🙂");
    let reasoning = String::from("actual thought");
    let args = [F {
        key: "value",
        value: V::Text(&text),
    }];
    let function = [
        F {
            key: "name",
            value: V::Text("probe"),
        },
        F {
            key: "arguments",
            value: V::Object(&args),
        },
    ];
    let call = [F {
        key: "function",
        value: V::Object(&function),
    }];
    let calls = [V::Object(&call)];
    let fields = [
        F {
            key: "role",
            value: V::Text("assistant"),
        },
        F {
            key: "content",
            value: V::Text("obsolete"),
        },
        F {
            key: "reasoning_content",
            value: V::Text(&reasoning),
        },
        F {
            key: "tool_calls",
            value: V::Array(&calls),
        },
        F {
            key: "content",
            value: V::Text(&text),
        },
    ];
    let messages = [V::Object(&fields)];
    let tool_fields = [
        F {
            key: "z",
            value: V::Bool(true),
        },
        F {
            key: "a",
            value: V::Text("name"),
        },
    ];
    let tools = [V::Object(&tool_fields)];
    let context = ChatRenderContext::from_messages(ChatMessages::from_records(&messages).unwrap())
        .with_tools(ChatInputArray::Record(&tools));
    let plan = source.render_plan_with_context(context).unwrap();
    let bound = plan.requirements().buffer_bytes();
    let rendered = plan.render().unwrap();
    assert!(rendered.retained_buffer_bytes() <= bound);
    let mut message = serde_json::Map::new();
    message.insert("role".into(), json!("assistant"));
    message.insert("content".into(), json!("obsolete"));
    message.insert("reasoning_content".into(), json!(reasoning));
    message.insert(
        "tool_calls".into(),
        json!([{"function":{"name":"probe","arguments":{"value":text}}}]),
    );
    message.insert("content".into(), json!(text));
    let json_messages = vec![serde_json::Value::Object(message)];
    let json_tools = vec![json!({"z":true,"a":"name"})];
    let mut ordinary = ordinary();
    for generation in [false, true] {
        let expected = ordinary
            .apply_chat_template_json(
                crate::tokenizer::ModelChatTemplate::Single(template.into()),
                [json_messages.clone()],
                Some(&json_tools),
                "record-source",
                generation,
                None,
            )
            .unwrap();
        assert_eq!(rendered.prompt(generation), expected[0]);
    }
    drop((text, reasoning, json_messages, json_tools));
    assert!(rendered.prompt(true).ends_with("tail"));
    assert!(rendered.prompt(false).contains("actual thought"));
    assert_eq!(rendered.generation_suffix(), "tail");
}
