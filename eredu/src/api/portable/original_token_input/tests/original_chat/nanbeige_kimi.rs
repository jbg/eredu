use super::*;
#[test]
fn public_nanbeige_adjacent_tools_xml_json_and_visible_content_match_ordinary_and_controlled() {
    const TEMPLATE: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/chat_templates/nanbeige4.2-0e137298.jinja"
    ));
    for manual in [false, true] {
        for format in ["xml", "json"] {
            let chat=ChatTemplateRequest {
            messages:vec![
                serde_json::json!({"role":"system","content":[{"type":"text","text":"Preserve É界🙂."}]}),
                serde_json::json!({"role":"user","content":[{"type":"text","text":"Compare readings."},{"type":"input_audio"}," trailing"]}),
                serde_json::json!({"role":"assistant","content":"<think>\nUse tools.\n</think>\nDone.","tool_calls":[{"function":{"name":"measure","arguments":{"values":[7,-11,2.5],"label":"<&> É界🙂"}}},{"function":{"name":"lookup","arguments":{"label":"Straße"}}}]}),
                serde_json::json!({"role":"tool","content":[{"type":"text","text":"18"}]}),
                serde_json::json!({"role":"tool","content":"found"}),
                serde_json::json!({"role":"assistant","reasoning_content":" Old scratch. ","content":"The difference is eighteen."}),
                serde_json::json!({"role":"user","content":"Check again."}),
            ],
            extra_template_kwargs:serde_json::json!({"tool_call_format":format,"preserve_thinking":false,"enable_thinking":false,"tools":[{"type":"function","function":{"name":"measure","parameters":{"type":"object","properties":{"values":{"type":"array","items":{"type":"number"}}}}}}]}).as_object().unwrap().clone(),
            add_generation_prompt:true,..Default::default()
        };
            super::released_template::run_case(
                TEMPLATE,
                ModelKind::Nanbeige,
                "nanbeige",
                chat,
                manual,
                |prompt| {
                    assert!(prompt.contains("unable to process this audio"));
                    assert_eq!(prompt.matches("<tool_response>").count(), 2);
                    assert!(!prompt.contains("Old scratch."));
                    assert!(prompt.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
                },
            );
        }
    }
}
#[test]
fn public_kimi_linear_compact_tools_and_structured_history_match_ordinary_and_controlled() {
    const TEMPLATE: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/chat_templates/kimi-linear-48b-a3b-instruct.jinja"
    ));
    for manual in [false, true] {
        let chat=ChatTemplateRequest {
            messages:vec![
                serde_json::json!({"role":"system","content":"Preserve É界🙂."}),
                serde_json::json!({"role":"user","name":"reader","content":[{"type":"text","text":"Compare readings."},{"type":"image"}]}),
                serde_json::json!({"role":"assistant","content":"Use tools.","tool_calls":[{"id":"measure:0","function":{"name":"measure","arguments":{"values":[7,-11,2.5]}}}]}),
                serde_json::json!({"role":"tool","tool_call_id":"measure:0","content":[{"type":"text","text":"18"}]}),
                serde_json::json!({"role":"user","content":"Explain."}),
            ],
            extra_template_kwargs:serde_json::json!({"tools":[{"type":"function","function":{"name":"measure","parameters":{"type":"object","properties":{"values":{"type":"array","items":{"type":"number"}}}}}}]}).as_object().unwrap().clone(),
            add_generation_prompt:true,..Default::default()
        };
        super::released_template::run_case(
            TEMPLATE,
            ModelKind::KimiLinear,
            "kimi-linear",
            chat,
            manual,
            |prompt| {
                assert!(prompt.contains("<|im_user|>reader<|im_middle|>"));
                assert!(prompt.contains("<|media_start|>image"));
                assert!(prompt.contains("## Return of measure:0"));
                assert!(prompt.ends_with("<|im_assistant|>assistant<|im_middle|>"));
            },
        );
    }
}
