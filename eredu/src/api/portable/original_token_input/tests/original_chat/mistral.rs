use super::*;
#[test]
fn public_mistral_filtered_tools_and_nested_message_equality_match_ordinary_and_controlled() {
    const TEMPLATE: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/chat_templates/mistral-7b-instruct-v0.3-c170c708.jinja"
    ));
    for manual in [false, true] {
        let chat=ChatTemplateRequest{
            messages:vec![
                serde_json::json!({"role":"system","content":"Preserve É界🙂."}),
                serde_json::json!({"role":"user","content":"Repeat.","metadata":{"trace":{"step":1}}}),
                serde_json::json!({"role":"assistant","tool_calls":[
                    {"id":"call00001","function":{"name":"measure","arguments":{"values":[7,-11,2.5]}}},
                    {"id":"call00002","function":{"name":"lookup","arguments":{"label":"É界🙂"}}}
                ]}),
                serde_json::json!({"role":"tool","tool_call_id":"call00001","content":{"content":"18"}}),
                serde_json::json!({"role":"tool_results","tool_call_id":"call00002","content":"found"}),
                serde_json::json!({"role":"assistant","content":" The difference is eighteen. "}),
                serde_json::json!({"role":"user","content":"Repeat.","metadata":{"trace":{"step":2}}}),
            ],
            extra_template_kwargs:serde_json::json!({"bos_token":"<s>","eos_token":"</s>","tools":[
                {"type":"function","function":{"name":"measure","description":"Measure É界🙂","parameters":{"type":"object","properties":{"values":{"type":"array","items":{"type":"number"}}}},"return":"SHOULD_NOT_RENDER"}},
                {"type":"function","function":{"name":"lookup","parameters":{"type":"object","properties":{"label":{"type":"string"}}},"return":"SHOULD_NOT_RENDER"}}
            ]}).as_object().unwrap().clone(),
            add_generation_prompt:true,..Default::default()
        };
        super::released_template::run_case(
            TEMPLATE,
            ModelKind::Llama,
            "mistral",
            chat,
            manual,
            |prompt| {
                assert_eq!(prompt.matches("[AVAILABLE_TOOLS]").count(), 1);
                assert!(!prompt.contains("SHOULD_NOT_RENDER"));
                assert!(prompt.contains("\"id\": \"call00001\""));
                assert_eq!(prompt.matches("[TOOL_RESULTS]").count(), 2);
                assert!(prompt.ends_with("[INST] Preserve É界🙂.\n\nRepeat.[/INST]"));
            },
        );
    }
}
