use super::*;

const QWEN: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/chat_templates/qwen2.5-7b-instruct-acbd9653.jinja"
));

fn history(system: bool) -> Vec<serde_json::Value> {
    let mut messages = Vec::new();
    if system {
        messages.push(serde_json::json!({"role":"system","content":"Preserve É and 世界."}));
    }
    messages.extend([
        serde_json::json!({"role":"user","content":"Compare the two readings 🙂."}),
        serde_json::json!({"role":"assistant","content":"Earlier answer."}),
        serde_json::json!({"role":"system","content":"A later system instruction."}),
        serde_json::json!({"role":"assistant","content":"Checking.","tool_calls":[
            {"function":{"name":"reading","arguments":{"station":"É界","values":[7,-11,2.5]}}},
            {"name":"reading","arguments":{"station":"🙂","enabled":true,"missing":null}}
        ]}),
        serde_json::json!({"role":"tool","content":"first 7","tool_call_id":"first"}),
        serde_json::json!({"role":"tool","content":"second -11","tool_call_id":"second"}),
        serde_json::json!({"role":"user","content":"Explain the difference."}),
    ]);
    messages
}

#[test]
fn public_qwen25_borrowed_tools_and_history_share_ordinary_prompt_and_controlled_cursor() {
    public_tools(QWEN, ModelKind::Qwen2, "qwen2.5", true);
}
#[test]
fn public_qwen3_reversed_history_and_tools_share_ordinary_prompt_and_controlled_cursor() {
    const TEMPLATE: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/chat_templates/qwen3-0.6b-7e4ae267.jinja"));
    public_tools(TEMPLATE, ModelKind::Qwen3, "qwen3", false);
}
fn public_tools(template: &str, family: ModelKind, model_id: &str, default_system: bool) {
    // These are immutable template bindings and historical messages. Explicit
    // tool-execution declarations retain the ordinary text-request policy.
    let tools = serde_json::json!([{"type":"function","function":{
        "name":"reading","description":"Return a reading for É界🙂.",
        "parameters":{"type":"object","properties":{
            "station":{"type":"string"},"values":{"type":"array","items":{"type":"number"}}
        },"required":["station"],"additionalProperties":false}
    }}]);
    for manual in [false, true] {
        for (has_tools, system) in [(true, true), (true, false), (false, false)] {
            let mut chat = ChatTemplateRequest {
                messages: history(system),
                add_generation_prompt: true,
                ..Default::default()
            };
            if !default_system {
                // The released Qwen3 reverse scan selects the last real user,
                // retaining both literal and explicit reasoning content.
                chat.messages.push(serde_json::json!({"role":"assistant",
                    "content":"<think>\nreason É🙂\n</think>\n\nFinal answer.",
                    "tool_calls":[{"function":{"name":"reading","arguments":"{\"station\":\"text args\"}"}}]}));
            }
            chat.extra_template_kwargs.insert(
                "tools".into(),
                if has_tools {
                    tools.clone()
                } else {
                    serde_json::json!([])
                },
            );

            super::released_template::run_case(template,family,model_id,chat,manual,|prompt|{
            assert!(
                prompt.contains("\"arguments\": {\"station\": \"É界\", \"values\": [7, -11, 2.5]}")
            );
            assert!(prompt.contains("<|im_start|>user\n<tool_response>\nfirst 7\n</tool_response>\n<tool_response>\nsecond -11\n</tool_response><|im_end|>"));
            assert_eq!(prompt.contains("# Tools"), has_tools);
            assert_eq!(
                prompt.contains("You are Qwen, created by Alibaba Cloud."),
                default_system && !system
            );
            assert!(prompt.ends_with("<|im_start|>assistant\n"));
            });
        }
    }
}
