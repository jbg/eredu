use super::*;
#[test]
fn public_llama31_builtin_and_custom_tools_share_ordinary_prompt_and_controlled_cursor() {
    const TEMPLATE: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/chat_templates/llama-3.1-3.3-e10ca381.jinja"
    ));
    for manual in [false, true] {
        for tools_in_user in [false, true] {
            let chat=ChatTemplateRequest{
                messages:vec![
                    serde_json::json!({"role":"system","content":"  Preserve É界🙂.  "}),
                    serde_json::json!({"role":"user","content":" Compare readings. "}),
                    serde_json::json!({"role":"assistant","tool_calls":[{"function":{"name":"browser","arguments":{"query":"É界🙂"}}}]}),
                    serde_json::json!({"role":"ipython","content":{"value":7}}),
                    serde_json::json!({"role":"assistant","tool_calls":[{"function":{"name":"measure","arguments":{"values":[7,-11,2.5]}}}]}),
                    serde_json::json!({"role":"tool","content":[7,-11]}),
                    serde_json::json!({"role":"assistant","content":" The difference is eighteen. "}),
                    serde_json::json!({"role":"user","content":" Explain it. "})
                ],
                extra_template_kwargs:serde_json::json!({
                    "custom_tools":[{"type":"function","function":{"name":"measure","parameters":{"type":"object"}}}],
                    "builtin_tools":["code_interpreter","browser","search"],
                    "tools_in_user_message":tools_in_user,
                    "date_string":"17 Sep 2026"
                }).as_object().unwrap().clone(),
                add_generation_prompt:true,..Default::default()
            };
            super::released_template::run_case(
                TEMPLATE,
                ModelKind::Llama,
                "llama",
                chat,
                manual,
                |prompt| {
                    assert!(prompt.contains("Tools: browser, search"));
                    assert!(prompt.contains("Today Date: 17 Sep 2026"));
                    assert!(prompt.contains("<|python_tag|>browser.call("));
                    assert!(prompt.contains("\"parameters\": {\"values\": [7, -11, 2.5]}"));
                    assert!(prompt.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n"));
                },
            );
        }
    }
}
