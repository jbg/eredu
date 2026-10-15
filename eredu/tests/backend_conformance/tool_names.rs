use super::*;
use eredu::api::{
    ChatInputRequest, ChatMessage, ChatPart, ChatPreparationOptions, ChatToolCall,
    ChatToolDefinition, TextModelError,
};
use eredu::runtime::chat::ToolChoice;
use serde_json::{json, Value};

fn request(names: &[&str]) -> ChatTemplateRequest {
    ChatTemplateRequest {
        messages: vec![json!({"role":"user", "content":"hello"})],
        tools: names.iter().map(|name| json!({"type":"function", "function":{
            "name":name, "description":name, "parameters":{
                "type":"object", "properties":{"value":{"type":"string", "description":name}},
                "required":["value"], "additionalProperties":false
            }
        }})).collect(),
        tool_choice: ToolChoice::Auto,
        ..Default::default()
    }
}

fn declarations(chat: &PreparedChat) -> Vec<Value> {
    chat.rendered_prompt()
        .split("<tools>\n")
        .nth(1)
        .unwrap()
        .split("\n</tools>")
        .next()
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}

#[test]
fn tool_names_are_stable_collision_free_and_shared_by_raw_and_typed_history() {
    let original = "goose.extension.lookup";
    let mut model = unicode_model(None);
    let first = model.prepare_chat(request(&[original])).unwrap();
    let colliding = declarations(&first)[0]["function"]["name"]
        .as_str()
        .unwrap()
        .to_owned();
    let long = "x".repeat(100);
    let names = [
        original,
        "lookup",
        "search-docs",
        "9search",
        "class",
        "天气/查询",
        &long,
        &colliding,
    ];
    let raw = request(&names);
    let chat = model.prepare_chat(raw.clone()).unwrap();
    let declarations = declarations(&chat);
    let wires: Vec<_> = declarations
        .iter()
        .map(|tool| tool["function"]["name"].as_str().unwrap())
        .collect();
    assert_eq!(
        wires.iter().copied().collect::<BTreeSet<_>>().len(),
        names.len()
    );
    assert_ne!(wires[0], original);
    assert_ne!(wires[0], colliding);
    assert_eq!(&wires[1..5], &["lookup", "search-docs", "9search", "class"]);
    assert_eq!(wires[7], colliding);
    for (tool, original) in declarations.iter().zip(names) {
        let function = &tool["function"];
        let wire = function["name"].as_str().unwrap();
        assert!(!wire.is_empty() && wire.len() <= 64);
        assert!(wire
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-'));
        assert_eq!(function["description"], original);
        assert_eq!(
            function["parameters"]["properties"]["value"]["description"],
            original
        );
    }
    let mut reordered = raw.clone();
    reordered.tools.reverse();
    let reordered = model.prepare_chat(reordered).unwrap();
    let mut reordered = self::declarations(&reordered);
    reordered.reverse();
    assert_eq!(reordered, declarations);

    // A named tool result and a historical-only tool both use the same namespace.
    let template = format!("{QWEN_TEMPLATE}{{% for message in messages if message.role == 'tool' %}}RESULT_NAME={{{{ [message.tool_call_id, message.name]|tojson }}}}{{% endfor %}}");
    let mut model = unicode_model_with_template(None, 64, &template);
    let mut history = request(&[original]);
    history.messages.extend([
        json!({"role":"assistant", "content":"", "tool_calls":[
            {"id":"id.goose.extension.lookup", "type":"function", "function":{"name":original, "arguments":{"value":original}}},
            {"id":"retired", "type":"function", "function":{"name":"retired.extension.tool", "arguments":{}}}
        ]}),
        json!({"role":"tool", "name":original, "tool_call_id":"id.goose.extension.lookup", "content":original}),
    ]);
    let chat = model.prepare_chat(history.clone()).unwrap();
    let declared = self::declarations(&chat)[0]["function"]["name"]
        .as_str()
        .unwrap()
        .to_owned();
    let calls: Vec<Value> = chat
        .rendered_prompt()
        .split("<tool_call>\n")
        .skip(1)
        .filter_map(|call| serde_json::from_str(call.split("\n</tool_call>").next().unwrap()).ok())
        .collect();
    assert_eq!(calls.len(), 2);
    assert_eq!(
        calls[0],
        json!({"name":declared,"arguments":{"value":original}})
    );
    assert_ne!(calls[1]["name"], "retired.extension.tool");
    let result: Value =
        serde_json::from_str(chat.rendered_prompt().split("RESULT_NAME=").nth(1).unwrap()).unwrap();
    assert_eq!(result, json!(["id.goose.extension.lookup", declared]));
    assert!(chat
        .rendered_prompt()
        .contains(&format!("<tool_response>\n{original}\n</tool_response>")));
    let typed = model
        .prepare_typed_chat(ChatInputRequest {
            messages: vec![
                ChatMessage::User(vec![ChatPart::Text("hello".into())]),
                ChatMessage::Assistant {
                    content: vec![],
                    reasoning_content: None,
                    tool_calls: vec![
                        ChatToolCall {
                            id: Some("id.goose.extension.lookup".into()),
                            name: original.into(),
                            arguments: json!({"value":original}).as_object().unwrap().clone(),
                        },
                        ChatToolCall {
                            id: Some("retired".into()),
                            name: "retired.extension.tool".into(),
                            arguments: Default::default(),
                        },
                    ],
                },
                ChatMessage::Tool {
                    tool_call_id: "id.goose.extension.lookup".into(),
                    name: Some(original.into()),
                    content: vec![ChatPart::Text(original.into())],
                },
            ],
            tools: vec![ChatToolDefinition {
                name: original.into(),
                description: Some(original.into()),
                parameters: history.tools[0]["function"]["parameters"].clone(),
            }],
            options: ChatPreparationOptions {
                tool_choice: ToolChoice::Auto,
                ..Default::default()
            },
        })
        .unwrap();
    // JSON object field order can legitimately differ between typed and raw
    // construction. Names and historical call semantics must agree.
    assert_eq!(self::declarations(typed.chat()), self::declarations(&chat));
    let typed_calls: Vec<Value> = typed
        .chat()
        .rendered_prompt()
        .split("<tool_call>\n")
        .skip(1)
        .filter_map(|call| serde_json::from_str(call.split("\n</tool_call>").next().unwrap()).ok())
        .collect();
    assert_eq!(typed_calls, calls);
    let typed_result: Value = serde_json::from_str(
        typed
            .chat()
            .rendered_prompt()
            .split("RESULT_NAME=")
            .nth(1)
            .unwrap(),
    )
    .unwrap();
    assert_eq!(typed_result, result);
}

#[test]
fn tool_names_reject_ambiguous_or_malformed_identity_and_unknown_rules() {
    let mut model = unicode_model(None);
    assert!(matches!(
        model.prepare_chat(request(&["extension.lookup", "extension.lookup"])),
        Err(TextModelError::InvalidRequest(
            eredu::api::ChatRequestError::DuplicateToolName { .. }
        ))
    ));
    for name in [Value::Null, json!(3), json!("")] {
        let mut invalid = request(&["lookup"]);
        invalid.tools[0]["function"]["name"] = name;
        assert!(matches!(
            model.prepare_chat(invalid),
            Err(TextModelError::InvalidRequest(
                eredu::api::ChatRequestError::InvalidToolName { .. }
            ))
        ));
    }
    let mut invalid = request(&["extension.lookup"]);
    invalid.tools[0]["function"]["parameters"] = json!({"type":"not_a_type"});
    assert!(matches!(
        model.prepare_chat(invalid),
        Err(TextModelError::InvalidRequest(
            eredu::api::ChatRequestError::ToolDefinition { .. }
        ))
    ));
    let mut unknown = unicode_model_with_template(None, 64, "{{ messages[0].content }}");
    assert!(matches!(
        unknown.prepare_chat(request(&["extension.lookup"])),
        Err(TextModelError::UnsupportedCapability(
            eredu::api::ChatCapabilityRejection::NativeTools(
                eredu::runtime::chat::NativeToolRejection::ToolNameRulesUnavailable { .. }
            )
        ))
    ));
}
