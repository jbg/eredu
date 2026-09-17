use super::*;
#[test]
fn public_deepseek31_tools_reasoning_and_limited_split_share_ordinary_prompt_and_controlled_cursor()
{
    const TEMPLATE: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/chat_templates/deepseek-v3.1-tools-ef1ab230.jinja"
    ));
    let tools = serde_json::json!([{"type":"function","function":{"name":"reading","description":"Read É界🙂","parameters":{"type":"object","properties":{"station":{"type":"string"}}}}}]);
    for manual in [false, true] {
        for (has_tools, thinking) in [(true, true), (true, false), (false, true)] {
            let messages = vec![
                serde_json::json!({"role":"system","content":"Preserve É and 世界."}),
                serde_json::json!({"role":"system","content":"Compare signed readings."}),
                serde_json::json!({"role":"user","content":"Earlier question."}),
                serde_json::json!({"role":"assistant","content":"<think>private</think>visible</think>tail"}),
                serde_json::json!({"role":"user","content":"Compare the two readings 🙂."}),
                serde_json::json!({"role":"assistant","content":null,"tool_calls":[
                    {"function":{"name":"reading","arguments":{"station":"É界","values":[7,-11,2.5]}}},
                    {"function":{"name":"reading","arguments":"{\"station\":\"text args\"}"}}
                ]}),
                serde_json::json!({"role":"tool","content":"first 7"}),
                serde_json::json!({"role":"tool","content":"second -11"}),
                serde_json::json!({"role":"assistant","content":"The difference is eighteen."}),
                serde_json::json!({"role":"user","content":"Explain the difference."}),
            ];
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
                    serde_json::Value::Null
                },
            );
            chat.extra_template_kwargs
                .insert("thinking".into(), serde_json::json!(thinking));
            super::released_template::run_case(
                TEMPLATE,
                ModelKind::DeepSeekV3,
                "deepseek_v3",
                chat,
                manual,
                |prompt| {
                    assert_eq!(prompt.contains("## Tools"), has_tools);
                    assert!(prompt.contains("visible</think>tail"));
                    assert!(!prompt.contains("private"));
                    assert!(prompt.contains(
                        "<｜tool▁sep｜>{\"station\": \"É界\", \"values\": [7, -11, 2.5]}"
                    ));
                    assert!(prompt.contains("<｜tool▁output▁begin｜>first 7<｜tool▁output▁end｜>"));
                    let (_,suffix)=prompt.rsplit_once("<｜Assistant｜>").expect("assistant prompt marker");
                    assert_eq!(suffix.trim(),if thinking {"<think>"}else{"</think>"},"actual prompt: {prompt:?}");
                },
            );
        }
    }
}
