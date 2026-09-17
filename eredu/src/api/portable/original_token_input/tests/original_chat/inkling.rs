use super::*;
#[test]
fn public_inkling_generated_tools_reasoning_and_history_share_ordinary_prompt_and_controlled_cursor(
) {
    const TEMPLATE: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/chat_templates/inkling-small-8cc5877b.jinja"
    ));
    let tools = serde_json::json!([
        {"type":"function","function":{"name":"reading","description":"É界🙂 readings","parameters":{"type":"object","properties":{"station":{"type":"string"},"values":{"type":"array","items":{"type":"number"}}},"required":["station"]}}},
        {"name":"status","description":"","parameters":{}}
    ]);
    for manual in [false, true] {
        for (has_tools, system, effort) in [
            (true, true, serde_json::json!(" medium ")),
            (true, false, serde_json::json!(0.2)),
            (false, false, serde_json::json!("none")),
        ] {
            let mut messages = Vec::new();
            if system {
                messages
                    .push(serde_json::json!({"role":"system","content":"Preserve É and 世界."}));
            }
            messages.extend([
                serde_json::json!({"role":"user","content":["Compare É.",{"type":"text","text":"Second 界🙂."}]}),
                serde_json::json!({"role":"assistant","reasoning_content":"Check signed readings.","content":"Looking.","tool_calls":[
                    {"id":"first","function":{"name":"reading","arguments":{"station":"É界","values":[7,-11,2.5]}}},
                    {"id":"second","function":{"name":"status","arguments":{}}}
                ]}),
                serde_json::json!({"role":"tool","tool_call_id":"first","content":"First 7"}),
                serde_json::json!({"role":"tool","name":"status","tool_call_id":"second","content":"Second -11"}),
                serde_json::json!({"role":"user","content":"Explain the difference."}),
            ]);
            let mut chat = ChatTemplateRequest {
                messages,
                add_generation_prompt: true,
                ..Default::default()
            };
            chat.extra_template_kwargs.insert(
                "tools".into(),
                if has_tools {
                    tools.clone()
                } else {
                    serde_json::json!([])
                },
            );
            chat.extra_template_kwargs
                .insert("reasoning_effort".into(), effort);
            super::released_template::run_case(
                TEMPLATE,
                ModelKind::Inkling,
                "inkling",
                chat,
                manual,
                |prompt| {
                    assert_eq!(prompt.contains("tool_declare<|content_xml|>"), has_tools);
                    assert!(prompt.contains(
                        r#"{"name":"reading","args":{"station":"É界","values":[7,-11,2.5]}}"#
                    ));
                    assert!(prompt.contains("<|message_tool|>reading<|content_text|>First 7"));
                    assert!(prompt
                        .contains("<|message_model|><|content_thinking|>Check signed readings."));
                    assert!(prompt.contains("<|message_user|><|content_text|>Second 界🙂."));
                    assert!(prompt.ends_with("<|message_model|>"));
                },
            );
        }
    }
}
