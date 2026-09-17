use super::*;
#[test]
fn public_lfm25_generated_tool_arguments_structured_content_and_reasoning_match_ordinary() {
    const TEMPLATE: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/chat_templates/lfm2.5-1.2b-instruct-ba551d58.jinja"
    ));
    for manual in [false, true] {
        for preserve in [false, true] {
            let chat=ChatTemplateRequest{
                messages: vec![
                    serde_json::json!({"role":"system","content":[{"type":"text","text":"Keep É界🙂."}]}),
                    serde_json::json!({"role":"user","content":["Observe ",{"type":"image"},{"type":"text","text":" and explain."}]}),
                    serde_json::json!({"role":"assistant","thinking":"old reasoning","content":"<think>hidden</think> Visible result."}),
                    serde_json::json!({"role":"user","content":"Call both functions."}),
                    serde_json::json!({"role":"assistant","reasoning_content":"recent reasoning","content":null,"tool_calls":[
                        {"function":{"name":"measure","arguments":{"station":"É'界\n🙂","values":[7,-11,2.5],"enabled":true}}},
                        {"function":{"name":"lookup","arguments":{}}}
                    ]}),
                    serde_json::json!({"role":"tool","content":{"reading":7,"station":"É界"}}),
                    serde_json::json!({"role":"user","content":"Explain the reading."})
                ],
                extra_template_kwargs:serde_json::json!({
                    "tools":[{"type":"function","function":{"name":"measure","parameters":{"type":"object"}}},"lookup()"],
                    "preserve_thinking":preserve
                }).as_object().unwrap().clone(),
                add_generation_prompt:true,
                ..Default::default()
            };
            super::released_template::run_case(
                TEMPLATE,
                ModelKind::Lfm2,
                "lfm2",
                chat,
                manual,
                |prompt| {
                    assert!(prompt.contains("List of tools: ["));
                    assert!(prompt.contains("Observe <image> and explain."));
                    assert!(prompt.contains("<|tool_call_start|>[measure("));
                    assert!(prompt.contains("lookup()]<|tool_call_end|>"));
                    assert_eq!(prompt.contains("old reasoning"), preserve);
                    assert!(prompt.contains("<think>recent reasoning</think>"));
                    assert!(prompt.ends_with("<|im_start|>assistant\n"));
                },
            );
        }
    }
}
