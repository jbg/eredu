use super::*;

const TEMPLATE: &str =
    include_str!("../../../tests/fixtures/chat_templates/qwen3.8-flash-next-de4b8e4d.jinja");
const TOKENIZER_CONFIG: &str = include_str!(
    "../../../tests/fixtures/chat_templates/qwen3.8-flash-next-de4b8e4d-tokenizer-config.json"
);
const GENERATION_CONFIG: &str = include_str!(
    "../../../tests/fixtures/chat_templates/qwen3.8-flash-next-de4b8e4d-generation-config.json"
);
const XHIGH: &str = "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.";
const LOW: &str = "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.";

fn prepare(
    request: ChatTemplateRequest,
) -> Result<crate::runtime::chat::PreparedChat, TextModelError> {
    let config: serde_json::Value = serde_json::from_str(TOKENIZER_CONFIG).unwrap();
    let eos = config["added_tokens_decoder"]
        .as_object()
        .unwrap()
        .iter()
        .find(|(_, value)| value["content"] == config["eos_token"])
        .unwrap()
        .0
        .parse::<u32>()
        .unwrap();
    let generation: serde_json::Value = serde_json::from_str(GENERATION_CONFIG).unwrap();
    let generation_eos = generation["eos_token_id"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| u32::try_from(value.as_u64().unwrap()).unwrap())
        .collect::<Vec<_>>();
    assert!(generation_eos.contains(&eos));
    let compiler = Ok(ConstraintCompiler::synthetic_for_tests());
    prepare_chat_from_parts(
        &mut production_chat_tokenizer(9),
        ModelChatTemplate::Single(TEMPLATE.into()),
        "behavior-selects-this-profile-without-family-dispatch",
        &generation_eos,
        Some(&compiler),
        request,
    )
}

fn user_request() -> ChatTemplateRequest {
    ChatTemplateRequest {
        messages: vec![json!({"role": "user", "content": "hello"})],
        add_generation_prompt: true,
        ..ChatTemplateRequest::default()
    }
}

#[test]
fn qwen4_exp_official_template_thinking_controls_and_effort_render_exactly() {
    for (thinking, effort, instructions) in [
        (None, None, XHIGH),
        (Some(true), None, XHIGH),
        (Some(true), Some("xhigh"), XHIGH),
        (Some(true), Some("low"), LOW),
        (Some(true), Some("medium"), ""),
        (Some(false), None, ""),
    ] {
        for add_generation_prompt in [false, true] {
            let prepared = prepare(ChatTemplateRequest {
                enable_thinking: thinking,
                reasoning_effort: effort.map(str::to_owned),
                add_generation_prompt,
                ..user_request()
            })
            .unwrap();
            let mut expected = if instructions.is_empty() {
                String::new()
            } else {
                format!("<|im_start|>system\n{instructions}<|im_end|>\n")
            };
            expected.push_str("<|im_start|>user\nhello<|im_end|>\n");
            let suffix = if thinking == Some(false) {
                "<|im_start|>assistant\n<think>\n\n</think>\n\n"
            } else {
                "<|im_start|>assistant\n<think>\n"
            };
            if add_generation_prompt {
                expected.push_str(suffix);
            }
            assert_eq!(prepared.rendered_prompt(), expected);
            assert_eq!(prepared.generation_prompt(), suffix);
            assert_eq!(
                prepared.format_profile_identity(),
                Some("qwen3.8.tagged-parameter-tools.v1")
            );
            assert_eq!(
                prepared.capabilities().reasoning_parser.is_supported(),
                thinking != Some(false)
            );
        }
    }
}

#[test]
fn qwen4_exp_official_template_preserves_old_reasoning_and_current_tool_reasoning() {
    for preserve in [None, Some(true), Some(false)] {
        let prepared = prepare(ChatTemplateRequest {
            messages: vec![
                json!({"role": "user", "content": "first"}),
                json!({"role": "assistant", "reasoning_content": " old reasoning ", "content": "old answer"}),
                json!({"role": "user", "content": "next"}),
                json!({"role": "assistant", "reasoning_content": " current reasoning ", "content": "", "tool_calls": [{"function": {"name": "lookup", "arguments": {"value": 7}}}]}),
                json!({"role": "tool", "content": "seven"}),
            ],
            tools: vec![production_tool("lookup")],
            extra_template_kwargs: preserve
                .map(|value| serde_json::Map::from_iter([("preserve_thinking".into(), json!(value))]))
                .unwrap_or_default(),
            ..user_request()
        })
        .unwrap();
        let expected_old = if preserve == Some(false) {
            "<|im_start|>assistant\nold answer<|im_end|>\n"
        } else {
            "<|im_start|>assistant\n<think>\nold reasoning\n</think>\n\nold answer<|im_end|>\n"
        };
        assert!(prepared.rendered_prompt().contains(expected_old));
        assert!(prepared.rendered_prompt().contains(concat!(
            "<|im_start|>assistant\n<think>\ncurrent reasoning\n</think>\n\n",
            "<tool_call>\n<function=lookup>\n<parameter=value>\n7\n</parameter>\n</function>\n</tool_call><|im_end|>\n",
            "<|im_start|>user\n<tool_response>\nseven\n</tool_response><|im_end|>\n"
        )));
    }
}

#[test]
fn qwen4_exp_official_template_tagged_arguments_tool_results_and_eos() {
    for (tool_choice, expected_trigger) in [
        (ToolChoice::Required, None),
        (ToolChoice::Auto, Some("<tool_call>")),
    ] {
        let prepared = prepare(ChatTemplateRequest {
        messages: vec![
            json!({"role": "user", "content": "look up two values"}),
            json!({"role": "assistant", "reasoning_content": "Inspect both.", "content": "Checking.", "tool_calls": [
                {"id": "first", "function": {"name": "lookup", "arguments": {"value": 7}}},
                {"id": "second", "function": {"name": "lookup", "arguments": {"value": 8}}}
            ]}),
            json!({"role": "tool", "tool_call_id": "first", "content": "seven"}),
            json!({"role": "tool", "tool_call_id": "second", "content": "eight"}),
        ],
        tools: vec![production_tool("lookup")],
        tool_choice,
        parallel_tool_calls: ParallelToolCallPolicy::Enabled {
            max_calls: std::num::NonZeroUsize::new(2),
        },
        ..user_request()
    })
    .unwrap();
        assert!(prepared.rendered_prompt().contains(concat!(
        "Checking.\n\n<tool_call>\n<function=lookup>\n<parameter=value>\n7\n</parameter>\n</function>\n</tool_call>\n",
        "<tool_call>\n<function=lookup>\n<parameter=value>\n8\n</parameter>\n</function>\n</tool_call><|im_end|>\n",
        "<|im_start|>user\n<tool_response>\nseven\n</tool_response>\n<tool_response>\neight\n</tool_response><|im_end|>\n",
        "<|im_start|>assistant\n<think>\n"
    )));
        assert_eq!(prepared.eos_token_ids(), &[248_046, 248_044]);
        assert_eq!(prepared.profile_stop_sequences(), ["<|im_end|>"]);
        assert_eq!(prepared.preserved_structural_token_ids(), &[9]);
        assert!(matches!(
            prepared.native_tool_support(),
            NativeToolSupport::Supported
        ));
        assert!(prepared
            .capabilities()
            .mapping_tool_arguments
            .is_supported());
        assert_eq!(
            prepared
                .tool_runtime_plan()
                .unwrap()
                .auto_activation_trigger(),
            expected_trigger
        );
    }
}

#[test]
fn qwen4_exp_official_template_rejects_unsupported_effort_and_ambiguous_history() {
    for request in [
        ChatTemplateRequest {
            reasoning_effort: Some("high".into()),
            ..user_request()
        },
        ChatTemplateRequest {
            enable_thinking: Some(false),
            reasoning_effort: Some("low".into()),
            ..user_request()
        },
        ChatTemplateRequest {
            extra_template_kwargs: serde_json::Map::from_iter([(
                "reasoning_effort".into(),
                json!(3),
            )]),
            ..user_request()
        },
    ] {
        assert!(matches!(
            prepare(request),
            Err(TextModelError::ToolConstraint(_))
        ));
    }
    for arguments in [
        json!("{\"value\":7}"),
        json!({"value": "unescaped </parameter>"}),
        json!({"bad>name": 7}),
    ] {
        let request = ChatTemplateRequest {
            messages: vec![
                json!({"role": "user", "content": "lookup"}),
                json!({"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "lookup", "arguments": arguments}}]}),
            ],
            tools: vec![production_tool("lookup")],
            ..user_request()
        };
        assert!(matches!(
            prepare(request),
            Err(TextModelError::ToolConstraint(_))
        ));
    }
}

#[test]
fn qwen4_exp_official_template_preserves_multiline_strings_and_json_parameter_values() {
    let prepared = prepare(ChatTemplateRequest {
        messages: vec![
            json!({"role": "user", "content": "inspect"}),
            json!({"role": "assistant", "content": "", "tool_calls": [{"function": {
                "name": "inspect", "arguments": {"text": "Bogotá\n東京 🦀", "values": [1, 2], "enabled": true}
            }}]}),
        ],
        tools: vec![json!({"type": "function", "function": {
            "name": "inspect", "parameters": {"type": "object", "properties": {
                "text": {"type": "string"}, "values": {"type": "array", "items": {"type": "integer"}}, "enabled": {"type": "boolean"}
            }, "required": ["text", "values", "enabled"], "additionalProperties": false}
        }})],
        ..user_request()
    })
    .unwrap();
    for parameter in [
        "<parameter=text>\nBogotá\n東京 🦀\n</parameter>",
        "<parameter=values>\n[1, 2]\n</parameter>",
        "<parameter=enabled>\ntrue\n</parameter>",
    ] {
        assert!(prepared.rendered_prompt().contains(parameter));
    }
}

#[test]
fn qwen4_exp_official_template_renders_image_video_placeholders_in_original_order() {
    let prepared = prepare(ChatTemplateRequest {
        messages: vec![json!({"role": "user", "content": [
            {"type": "text", "text": "Compare "},
            {"type": "image", "image": "prepared-image"},
            {"type": "text", "text": " with "},
            {"type": "video", "video": "prepared-video"}
        ]})],
        enable_thinking: Some(false),
        extra_template_kwargs: serde_json::Map::from_iter([("add_vision_id".into(), json!(true))]),
        ..user_request()
    })
    .unwrap();
    assert_eq!(
        prepared.rendered_prompt(),
        concat!(
            "<|im_start|>user\nCompare Picture 1: <|vision_start|><|image_pad|><|vision_end|>",
            " with Video 1: <|vision_start|><|video_pad|><|vision_end|><|im_end|>\n",
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )
    );
}
