use super::*;
#[test]
fn public_gpt_oss_current_date_and_recursive_tools_match_ordinary_and_controlled() {
    const TEMPLATE: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/chat_templates/gpt-oss-harmony-a4c9919c.jinja"
    ));
    for manual in [false, true] {
        let chat=ChatTemplateRequest {
            messages:vec![
                serde_json::json!({"role":"system","content":"Preserve É界🙂."}),
                serde_json::json!({"role":"user","content":"Compare readings."}),
                serde_json::json!({"role":"assistant","thinking":"Use the tool.","tool_calls":[{"function":{"name":"measure","arguments":{"values":[7,-11,2.5]}}}]}),
                serde_json::json!({"role":"tool","content":{"value":18,"label":"É界🙂"}}),
                serde_json::json!({"role":"assistant","content":"The difference is eighteen."}),
                serde_json::json!({"role":"user","content":"Check again."}),
            ],
            extra_template_kwargs:serde_json::json!({"builtin_tools":["python"],"reasoning_effort":"high","tools":[{"type":"function","function":{"name":"measure","description":"Measure É界🙂","parameters":{"type":"object","properties":{"values":{"type":"array","items":{"type":"number"}},"label":{"type":"string","enum":["a","É界"]},"options":{"type":"object","properties":{"selected":{"type":"boolean"}}}},"required":["values"]}}}]}).as_object().unwrap().clone(),
            add_generation_prompt:true,..Default::default()
        };
        super::released_template::run_case(
            TEMPLATE,
            ModelKind::GptOss,
            "gpt-oss",
            chat,
            manual,
            |prompt| {
                assert!(prompt.contains("Current date: "));
                assert!(prompt.contains("Reasoning: high"));
                assert!(prompt.contains("type measure = "));
                assert!(prompt.contains("<|start|>functions.measure to=assistant"));
                assert!(prompt.ends_with("<|start|>assistant"));
            },
        );
    }
}
#[test]
fn public_llama4_clock_lookup_and_tool_history_match_ordinary_and_controlled() {
    const TEMPLATE: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/chat_templates/llama-4-01a91bfb.jinja"
    ));
    for manual in [false, true] {
        let chat=ChatTemplateRequest {
            messages:vec![
                serde_json::json!({"role":"system","content":[{"type":"text","text":" Preserve É界🙂. "}]}),
                serde_json::json!({"role":"user","content":"Compare readings."}),
                serde_json::json!({"role":"assistant","content":"","tool_calls":[{"function":{"name":"measure","arguments":{"values":[7,-11,2.5]}}}]}),
                serde_json::json!({"role":"tool","content":{"value":18}}),
                serde_json::json!({"role":"assistant","content":[{"type":"text","text":"The difference is eighteen."}]}),
                serde_json::json!({"role":"user","content":[{"type":"text","text":"Look again."},{"type":"image"}]}),
            ],
            extra_template_kwargs:serde_json::json!({"bos_token":"<|begin_of_text|>","tools":[{"type":"function","function":{"name":"measure","parameters":{"type":"object","properties":{"values":{"type":"array","items":{"type":"number"}}}}}}]}).as_object().unwrap().clone(),
            add_generation_prompt:true,..Default::default()
        };
        super::released_template::run_case(
            TEMPLATE,
            ModelKind::Llama,
            "llama4-template",
            chat,
            manual,
            |prompt| {
                assert!(prompt.contains("Preserve É界🙂."));
                assert!(prompt.contains("<|image|>"));
                assert!(prompt.contains("\"name\": \"measure\""));
                assert!(prompt.ends_with("<|header_start|>assistant<|header_end|>\n\n"));
            },
        );
    }
}
