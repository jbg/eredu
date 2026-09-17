use super::*;
#[test]
fn public_gemma4_recursive_tool_schema_and_response_turns_match_ordinary_and_controlled() {
    const TEMPLATE: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/chat_templates/gemma-4-e2b-it-3e22461f.jinja"
    ));
    for manual in [false, true] {
        for preserve in [false, true] {
            let chat=ChatTemplateRequest{
                messages:vec![
                    serde_json::json!({"role":"system","content":[{"type":"text","text":" Preserve É界🙂. "}]}),
                    serde_json::json!({"role":"user","content":[{"type":"text","text":"Compare readings."},{"type":"image"}]}),
                    serde_json::json!({"role":"assistant","reasoning_content":"Check the older values.","tool_calls":[
                        {"id":"first","function":{"name":"measure","arguments":{"Values":[7,-11,2.5],"label":"É界🙂","options":{"b":true,"a":null}}}},
                        {"id":"second","function":{"name":"lookup","arguments":{"name":"Straße"}}}
                    ]}),
                    serde_json::json!({"role":"tool","tool_call_id":"first","content":{"z":18,"a":{"values":[7,-11]}}}),
                    serde_json::json!({"role":"tool","tool_call_id":"second","content":[{"type":"text","text":" found "},{"type":"image"}]}),
                    serde_json::json!({"role":"assistant","content":"The difference is eighteen.<|channel>thought\nprivate scratch<channel|> Continue."}),
                    serde_json::json!({"role":"user","content":"Explain."}),
                    serde_json::json!({"role":"assistant","reasoning":"Use the measured values.","content":"The comparison follows."}),
                    serde_json::json!({"role":"user","content":"One more."}),
                ],
                extra_template_kwargs:serde_json::json!({
                    "enable_thinking":true,"preserve_thinking":preserve,
                    "tools":[{"type":"function","function":{"name":"measure","description":"Measure É界🙂","parameters":{
                        "type":"object","required":["Values"],"properties":{
                            "label":{"type":"string","enum":["Straße","É界"]},
                            "Values":{"type":"array","items":{"type":["number","null"],"description":"Observed"}},
                            "options":{"type":"object","properties":{"b":{"type":"boolean"},"a":{"type":"number","nullable":true}},"required":["b"]}
                        }},"response":{"type":"object","description":"Result"}}}]
                }).as_object().unwrap().clone(),
                add_generation_prompt:true,..Default::default()
            };
            super::released_template::run_case(
                TEMPLATE,
                ModelKind::Gemma4,
                "gemma4",
                chat,
                manual,
                |prompt| {
                    assert!(prompt.contains("declaration:measure"));
                    assert!(prompt.contains("type:[<|\"|>NUMBER<|\"|>,<|\"|>NULL<|\"|>]"));
                    assert!(prompt.contains("response:measure{a:{values:[7,-11]},z:18}"));
                    assert!(prompt.contains("<|image|>"));
                    assert!(!prompt.contains("private scratch"));
                    assert!(prompt.ends_with("<|turn>model\n"));
                },
            );
        }
    }
}
