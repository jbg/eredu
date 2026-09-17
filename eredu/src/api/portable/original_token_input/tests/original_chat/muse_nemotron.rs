use super::*;
#[test]
fn public_muse_generated_recipients_tools_and_atem_share_ordinary_prompt_and_controlled_cursor() {
    const TEMPLATE: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/chat_templates/muse-glimmer-30b-97c77dff.jinja"
    ));
    let tools = serde_json::json!([
        {"function":{"name":"weather.reading","description":"Read É界🙂","parameters":{"type":"object","properties":{"station":{"type":"string"}}}}},
        {"function":{"name":"weather.status","description":"Status","parameters":{"type":"object","properties":{}}}},
        {"function":{"name":"local","description":"Local status","parameters":{}}}
    ]);
    for manual in [false, true] {
        for (has_tools, system) in [(true, true), (true, false), (false, false)] {
            let mut messages = Vec::new();
            if system {
                messages
                    .push(serde_json::json!({"role":"system","content":"Preserve É and 世界."}));
            }
            messages.extend([
                serde_json::json!({"role":"user","content":[{"type":"text","text":"Compare É界🙂."}]}),
                serde_json::json!({"role":"assistant","reasoning_content":"Check signed readings.","content":"","tool_calls":[
                    {"id":"first","function":{"name":"weather.reading","arguments":{"station":"É界","values":[7,-11,2.5],"enabled":true,"missing":null}}},
                    {"id":"second","function":{"name":"weather.status","arguments":{"z":7,"a":-11}}}
                ]}),
                serde_json::json!({"role":"tool","tool_call_id":"first","content":"First 7"}),
                serde_json::json!({"role":"tool","name":"weather.status","tool_call_id":"second","content":"Second -11"}),
                serde_json::json!({"role":"assistant","recipient":"user","end_turn":true,"content":"Earlier conclusion."}),
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
                .insert("reasoning_strength".into(), serde_json::json!("medium"));
            chat.extra_template_kwargs
                .insert("current_date".into(), serde_json::json!("2026-09-17"));
            chat.extra_template_kwargs.insert(
                "tool_namespace_descriptions".into(),
                serde_json::json!({"weather":"É界 readings","local":"Local"}),
            );
            super::released_template::run_case(
                TEMPLATE,
                ModelKind::MuseGlimmer,
                "muse_glimmer",
                chat,
                manual,
                |prompt| {
                    assert_eq!(prompt.contains("// Function schemas"), has_tools);
                    assert!(prompt.contains(
                        "<atem:parameter name=\"values\">[7, -11, 2.5]</atem:parameter>"
                    ));
                    assert!(
                        prompt.contains("<atem:parameter name=\"enabled\">true</atem:parameter>")
                    );
                    assert!(prompt.contains("<|start|>tool weather.reading<|message|>"));
                    assert!(prompt.contains("Reasoning strength: medium."));
                    assert!(prompt.contains(if has_tools {
                        "# Valid recipients: \"self\", \"weather.*\", \"local.*\", \"user\"."
                    } else {
                        "# Valid recipients: \"self\", \"user\"."
                    }));
                    assert!(prompt.ends_with("<|start|>assistant"));
                },
            );
        }
    }
}

#[test]
fn public_nemotron_thinking_tool_groups_and_prefix_share_ordinary_prompt_and_controlled_cursor() {
    const TEMPLATE: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/chat_templates/nemotron-nano-v2-6533e8de.jinja"
    ));
    let tools = serde_json::json!([{"type":"function","function":{"name":"reading","description":"Read É界🙂","parameters":{"type":"object","properties":{"station":{"type":"string"}}}}}]);
    for manual in [false, true] {
        for (has_tools, system, thinking) in [
            (true, true, true),
            (true, false, false),
            (false, false, true),
        ] {
            let mut messages = Vec::new();
            if system {
                messages.push(serde_json::json!({"role":"system","content":"  /think Preserve É and 世界.  "}));
            }
            messages.extend([
                serde_json::json!({"role":"user","content":"Compare the two readings 🙂."}),
                serde_json::json!({"role":"assistant","content":"<think>private</think> Checking. ","tool_calls":[
                    {"function":{"name":"reading","arguments":{"station":"É界","values":[7,-11,2.5]}}},
                    {"name":"reading","arguments":"{\"station\":\"text args\"}"}
                ]}),
                serde_json::json!({"role":"tool","content":"first 7"}),
                serde_json::json!({"role":"tool","content":"second -11"}),
                serde_json::json!({"role":"user","content":if thinking {"Explain /think difference."}else{"Explain /no_think difference."}}),
                serde_json::json!({"role":"assistant","content":"  continued É🙂  "}),
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
            super::released_template::run_case(
                TEMPLATE,
                ModelKind::NemotronH,
                "nemotron_h",
                chat,
                manual,
                |prompt| {
                    assert_eq!(prompt.contains("<AVAILABLE_TOOLS>["), has_tools);
                    assert!(prompt.contains("<TOOL_RESPONSE>[first 7, second -11]</TOOL_RESPONSE>"));
                    assert!(prompt
                        .contains(r#""arguments": {"station": "É界", "values": [7, -11, 2.5]}"#));
                    assert!(!prompt.contains("private"));
                    assert!(prompt.ends_with(if thinking {
                        "<think>\ncontinued É🙂"
                    } else {
                        "<think></think>continued É🙂"
                    }));
                },
            );
        }
    }
}
