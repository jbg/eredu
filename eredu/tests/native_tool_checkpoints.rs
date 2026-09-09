//! Opt-in smoke tests for real native-tool checkpoints.
//!
//! Tests never download data. Set the named environment variable to a local
//! model directory or GGUF and run the exact ignored test on a Metal host.

use eredu::{
    api::{
        local_device_plan, LoadedModel, LocalDevice, PreparedChatGenerationRequest,
        PreparedChatGenerationSettings, PreparedChatInput,
    },
    runtime::chat::{ChatTemplateRequest, NativeToolSupport, ToolChoice},
};
use eredu_backend_mlx::MlxBackendFactory;
use eredu_core::{ExecutionPlan, FinishReason, ResidencyPlan, SemanticEvent};
use serde_json::json;

fn profile_requires_structural_tool_tokens(identity: &str) -> bool {
    identity.starts_with("google.gemma4.")
        || identity.starts_with("openai.gpt-oss.")
        || identity.starts_with("mistral.")
        || identity.starts_with("lfm2.")
        || identity.starts_with("deepseek.")
        || identity.starts_with("meta.llama-4.")
}

fn smoke(environment: &str, expected_profile_prefix: &str) {
    smoke_with_plan(environment, expected_profile_prefix, accelerator_plan());
}

fn accelerator_plan() -> ExecutionPlan {
    ExecutionPlan::fully_resident(
        local_device_plan(LocalDevice::Accelerator(0))
            .expect("native checkpoint smoke tests require a compiled accelerator"),
    )
}

fn smoke_with_plan(environment: &str, expected_profile_prefix: &str, plan: ExecutionPlan) {
    smoke_with_tool(
        environment,
        expected_profile_prefix,
        plan,
        json!({
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Look up one integer.",
                "parameters": {
                    "type": "object",
                    "properties": {"value": {"type": "integer", "enum": [7]}},
                    "required": ["value"],
                    "additionalProperties": false
                }
            }
        }),
        json!({"value": 7}),
    );
}

fn smoke_with_tool(
    environment: &str,
    expected_profile_prefix: &str,
    plan: ExecutionPlan,
    tool: serde_json::Value,
    expected_arguments: serde_json::Value,
) {
    let tool_name = tool["function"]["name"].as_str().unwrap().to_owned();
    let path = std::env::var(environment)
        .unwrap_or_else(|_| panic!("{environment} must name a local checkpoint"));
    let planned = LoadedModel::load_execution_plan(&MlxBackendFactory::default(), &path, &plan)
        .unwrap_or_else(|error| panic!("failed to load {environment}={path:?}: {error}"));
    let (mut model, _) = planned.into_parts();
    // Tool declarations must also prepare for an ordinary text request.
    model
        .prepare_chat(ChatTemplateRequest {
            messages: vec![json!({"role": "user", "content": "Hello."})],
            tools: vec![tool.clone()],
            tool_choice: ToolChoice::Auto,
            add_generation_prompt: true,
            ..Default::default()
        })
        .unwrap();
    let prepared = model
        .prepare_chat(ChatTemplateRequest {
            messages: vec![json!({
                "role": "user",
                "content": format!("Call {tool_name} exactly once with {expected_arguments}. Do not answer with text.")
            })],
            tools: vec![tool],
            // Automatic selection deliberately exercises activation when a
            // trigger begins or ends within a tokenizer token.
            tool_choice: ToolChoice::Auto,
            enable_thinking: Some(false),
            add_generation_prompt: true,
            ..ChatTemplateRequest::default()
        })
        .unwrap_or_else(|error| panic!("failed to prepare {environment}={path:?}: {error}"));

    assert!(
        matches!(prepared.native_tool_support(), NativeToolSupport::Supported),
        "{environment} selected unsupported native tools: {:?}",
        prepared.native_tool_support()
    );
    let profile_identity = prepared
        .format_profile_identity()
        .unwrap_or_else(|| panic!("{environment} did not select a format profile"));
    assert!(
        profile_identity.starts_with(expected_profile_prefix),
        "{environment} selected unexpected profile {:?}",
        prepared.format_profile_identity()
    );
    assert!(
        !prepared.profile_stop_sequences().is_empty(),
        "{environment} profile {profile_identity} did not configure a stop sequence"
    );

    let mut events = Vec::new();
    let output = model
        .generate_prepared_chat(PreparedChatGenerationRequest {
            input: PreparedChatInput::rendered_prompt(&prepared),
            settings: PreparedChatGenerationSettings {
                overrides: eredu_core::GenerationConfigOverrides {
                    temperature: Some(0.0),
                    max_new_tokens: Some(256),
                    ..Default::default()
                },
                seed: 0,
                ..Default::default()
            },
            caller_stop_sequences: &[],
            cancellation: eredu_core::GenerationCancellationToken::new(),
            on_event: |event| events.push(event),
        })
        .unwrap_or_else(|error| {
            panic!("failed to generate from {environment}={path:?} ({profile_identity}): {error}")
        });
    let decoded = model
        .decode(&output.token_ids, false)
        .unwrap_or_else(|error| panic!("failed to decode {environment} output: {error}"));

    assert!(
        !output.token_ids.is_empty(),
        "{environment} generated no tokens"
    );
    assert!(
        matches!(
            output.finish_reason,
            FinishReason::GrammarComplete | FinishReason::StopSequence
        ),
        "{environment} stopped before completing a constrained call: {:?}; decoded={decoded:?}",
        output.finish_reason
    );
    assert_eq!(
        events.last(),
        Some(&SemanticEvent::Finished {
            reason: output.finish_reason
        }),
        "{environment} did not publish the terminal event; decoded={decoded:?}"
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, SemanticEvent::Finished { .. }))
            .count(),
        1,
        "{environment} published multiple terminal events; events={events:?}"
    );

    let starts = events
        .iter()
        .filter_map(|event| match event {
            SemanticEvent::ToolCallStart { index, id, name } => Some((*index, id, name)),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        starts.len(),
        1,
        "{environment} did not emit exactly one tool-call start; decoded={decoded:?}; events={events:?}"
    );
    assert_eq!(starts[0].0, 0, "{environment} used a nonzero call index");
    assert!(
        !starts[0].1.is_empty(),
        "{environment} emitted an empty call id"
    );
    assert_eq!(
        starts[0].2, &tool_name,
        "{environment} selected the wrong constrained tool"
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, SemanticEvent::ToolCallEnd))
            .count(),
        1,
        "{environment} did not structurally close exactly one tool call; decoded={decoded:?}; events={events:?}"
    );

    let arguments = events
        .iter()
        .filter_map(|event| match event {
            SemanticEvent::ToolArgumentsDelta {
                index,
                json_fragment,
            } => {
                assert_eq!(
                    *index, 0,
                    "{environment} emitted arguments for an unexpected call"
                );
                Some(json_fragment.as_str())
            }
            _ => None,
        })
        .collect::<String>();
    let arguments: serde_json::Value = serde_json::from_str(&arguments).unwrap_or_else(|error| {
        panic!(
            "{environment} emitted invalid semantic JSON arguments {arguments:?}: {error}; decoded={decoded:?}"
        )
    });
    assert_eq!(
        arguments, expected_arguments,
        "{environment} emitted the wrong semantic arguments; decoded={decoded:?}"
    );

    if profile_requires_structural_tool_tokens(profile_identity) {
        assert!(
            output
                .token_ids
                .iter()
                .any(|token| prepared.preserved_structural_token_ids().contains(token)),
            "{environment} did not generate a profile structural token; decoded={decoded:?}"
        );
    }
}

fn qwen_residency_smoke(environment: &str) {
    smoke_with_plan(environment, "qwen.", accelerator_plan());
    let layerwise = accelerator_plan().with_residency(ResidencyPlan::LayerwiseHost {
        device_layer_window: 1,
        device_budget_bytes: None,
        host_budget_bytes: None,
    });
    smoke_with_plan(environment, "qwen.", layerwise);
}

#[test]
#[ignore = "requires EREDU_GEMMA4_TOOL_CHECKPOINT and an MLX Metal device"]
fn gemma4_real_checkpoint_native_tool_smoke() {
    smoke("EREDU_GEMMA4_TOOL_CHECKPOINT", "google.gemma4.");
}

#[test]
#[ignore = "requires EREDU_GPT_OSS_TOOL_CHECKPOINT and an MLX Metal device"]
fn gpt_oss_real_checkpoint_native_tool_smoke() {
    smoke("EREDU_GPT_OSS_TOOL_CHECKPOINT", "openai.gpt-oss.harmony.");
}

#[test]
#[ignore = "requires EREDU_QWEN_TOOL_CHECKPOINT and an MLX Metal device"]
fn qwen_real_checkpoint_native_tool_smoke() {
    smoke("EREDU_QWEN_TOOL_CHECKPOINT", "qwen.");
}

#[test]
#[ignore = "requires EREDU_QWEN2_CHECKPOINT and an MLX Metal device"]
fn qwen2_real_checkpoint_resident_and_bounded_smoke() {
    qwen_residency_smoke("EREDU_QWEN2_CHECKPOINT");
}

#[test]
#[ignore = "requires EREDU_QWEN3_CHECKPOINT and an MLX Metal device"]
fn qwen3_real_checkpoint_resident_and_bounded_smoke() {
    qwen_residency_smoke("EREDU_QWEN3_CHECKPOINT");
}

#[test]
#[ignore = "requires EREDU_QWEN3_MOE_CHECKPOINT and an MLX Metal device"]
fn qwen3_moe_real_checkpoint_resident_and_bounded_smoke() {
    qwen_residency_smoke("EREDU_QWEN3_MOE_CHECKPOINT");
}

#[test]
#[ignore = "requires EREDU_MISTRAL_TOOL_CHECKPOINT and an MLX Metal device"]
fn mistral_real_checkpoint_native_tool_smoke() {
    smoke("EREDU_MISTRAL_TOOL_CHECKPOINT", "mistral.");
}

#[test]
#[ignore = "requires EREDU_LFM2_TOOL_CHECKPOINT and an MLX Metal device"]
fn lfm2_real_checkpoint_native_tool_smoke() {
    smoke("EREDU_LFM2_TOOL_CHECKPOINT", "lfm2.");
}

#[test]
#[ignore = "requires EREDU_LFM2_TOOL_CHECKPOINT and an MLX Metal device"]
fn lfm2_real_checkpoint_optional_keyword_argument_smoke() {
    smoke_with_tool(
        "EREDU_LFM2_TOOL_CHECKPOINT",
        "lfm2.",
        accelerator_plan(),
        json!({
            "type": "function",
            "function": {
                "name": "delegate",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "instructions": {"type": "string"},
                        "async": {"type": "boolean", "default": false}
                    }
                }
            }
        }),
        json!({"instructions": "Summarize.", "async": true}),
    );
}

#[test]
#[ignore = "requires EREDU_DEEPSEEK_TOOL_CHECKPOINT and an MLX Metal device"]
fn deepseek_real_checkpoint_native_tool_smoke() {
    smoke("EREDU_DEEPSEEK_TOOL_CHECKPOINT", "deepseek.");
}

#[test]
#[ignore = "requires EREDU_LLAMA_TOOL_CHECKPOINT and an MLX Metal device"]
fn llama_real_checkpoint_native_tool_smoke() {
    smoke("EREDU_LLAMA_TOOL_CHECKPOINT", "meta.llama-");
}

#[test]
#[ignore = "requires EREDU_NEMOTRON_TOOL_CHECKPOINT and an MLX Metal device"]
fn nemotron_real_checkpoint_native_tool_smoke() {
    smoke("EREDU_NEMOTRON_TOOL_CHECKPOINT", "nvidia.");
}

#[test]
#[ignore = "requires EREDU_NANBEIGE_CHECKPOINT and an MLX Metal device"]
fn nanbeige_real_checkpoint_executes_tool_and_answers_from_result() {
    use eredu::api::{ControlledGenerationRecord, ObservedGenerationEvent, TraceLimits};
    use eredu_core::{capture::CapturePlan, GenerationConfigOverrides};
    use std::ops::ControlFlow;

    let path = std::env::var("EREDU_NANBEIGE_CHECKPOINT").unwrap();
    let (mut model, _) =
        LoadedModel::load_execution_plan(&MlxBackendFactory::default(), &path, &accelerator_plan())
            .unwrap()
            .into_parts();
    let tool = json!({"type":"function", "function": {
        "name":"lookup", "description":"Look up the secret word for an integer.",
        "parameters":{"type":"object", "properties":{"value":{"type":"integer", "enum":[7]}},
            "required":["value"], "additionalProperties":false}
    }});
    let settings = PreparedChatGenerationSettings {
        overrides: GenerationConfigOverrides {
            temperature: Some(0.0),
            max_new_tokens: Some(512),
            ..Default::default()
        },
        seed: 0,
        ..Default::default()
    };
    let todo_content = "- [ ] Shell tool round trip\n- [ ] Inspect the request log";
    let todo_tool = json!({"type":"function", "function":{
        "name":"todo__todo_write", "description":"Save the todo list and return a secret receipt word.",
        "parameters":{"type":"object", "properties":{"content":{"type":"string"}},
            "required":["content"], "additionalProperties":false}
    }});
    let cases = [
        (tool, "Call lookup with value 7. Then reply with only the secret word returned by the tool.".to_owned(), json!({"value":7})),
        (todo_tool, format!("Use todo__todo_write to save exactly this list:\n{todo_content}\nThen reply with only the secret receipt word returned by the tool."), json!({"content":todo_content})),
    ];
    for (tool, prompt, expected_arguments) in cases {
        let tool_name = tool["function"]["name"].as_str().unwrap();
        for thinking in [false, true] {
            for controlled in [false, true] {
                model.reset().unwrap();
                let mut messages = vec![
                    json!({"role":"system", "content":"You are a concise assistant. Follow the user's instructions and use tools when requested."}),
                    json!({"role":"user", "content":prompt}),
                ];
                let mut executions = 0;
                for turn in 0..2 {
                    let prepared = model
                        .prepare_chat(ChatTemplateRequest {
                            messages: messages.clone(),
                            tools: vec![tool.clone()],
                            tool_choice: ToolChoice::Auto,
                            enable_thinking: Some(thinking),
                            add_generation_prompt: true,
                            ..Default::default()
                        })
                        .unwrap();
                    assert!(matches!(
                        prepared.native_tool_support(),
                        NativeToolSupport::Supported
                    ));
                    let mut events = Vec::new();
                    let ids = if controlled {
                        let observed = model
                            .prepare_observed_chat(
                                &prepared,
                                settings,
                                CapturePlan::none(),
                                TraceLimits {
                                    per_record_bytes: 65536,
                                    total_bytes: 4 << 20,
                                },
                            )
                            .unwrap();
                        let mut collect = |record: ControlledGenerationRecord| {
                            if let ObservedGenerationEvent::Semantic { event, .. } =
                                record.generation.event
                            {
                                events.push(event);
                            }
                            ControlFlow::Continue(())
                        };
                        let mut session = model
                            .start_controlled_chat(observed, &[], Default::default(), &mut collect)
                            .unwrap();
                        session.run(&mut collect).unwrap();
                        session.token_ids().to_vec()
                    } else {
                        model
                            .generate_prepared_chat(PreparedChatGenerationRequest {
                                input: PreparedChatInput::rendered_prompt(&prepared),
                                settings,
                                caller_stop_sequences: &[],
                                cancellation: Default::default(),
                                on_event: |e| events.push(e),
                            })
                            .unwrap()
                            .token_ids
                    };
                    let decoded = model.decode(&ids, false).unwrap();
                    eprintln!(
                        "thinking={thinking} controlled={controlled} turn={turn}: {decoded:?}"
                    );
                    if turn == 0 {
                        assert_eq!(
                            events
                                .iter()
                                .filter(|e| matches!(e, SemanticEvent::ToolCallEnd))
                                .count(),
                            1,
                            "{events:?}"
                        );
                        let (id, name) = events
                            .iter()
                            .find_map(|e| match e {
                                SemanticEvent::ToolCallStart { id, name, .. } => Some((id, name)),
                                _ => None,
                            })
                            .unwrap();
                        assert_eq!(name, tool_name);
                        let args = events
                            .iter()
                            .filter_map(|e| match e {
                                SemanticEvent::ToolArgumentsDelta { json_fragment, .. } => {
                                    Some(json_fragment.as_str())
                                }
                                _ => None,
                            })
                            .collect::<String>();
                        let args: serde_json::Value = serde_json::from_str(&args).unwrap();
                        assert_eq!(args, expected_arguments);
                        // Dispatch only after ToolCallEnd/schema validation. The
                        // returned word is absent from the original prompt.
                        let result = match (name.as_str(), args["value"].as_i64()) {
                            ("lookup", Some(7)) => {
                                executions += 1;
                                "cobalt"
                            }
                            ("todo__todo_write", _) => {
                                let saved_list = args["content"]
                                    .as_str()
                                    .unwrap()
                                    .lines()
                                    .collect::<Vec<_>>();
                                assert_eq!(saved_list, todo_content.lines().collect::<Vec<_>>());
                                executions += 1;
                                "cobalt"
                            }
                            _ => panic!("unexpected native tool request"),
                        };
                        let reasoning = events
                            .iter()
                            .filter_map(|e| match e {
                                SemanticEvent::ReasoningDelta(s) => Some(s.as_str()),
                                _ => None,
                            })
                            .collect::<String>();
                        messages.push(json!({"role":"assistant", "content":"", "reasoning_content":reasoning,
                        "tool_calls":[{"id":id, "type":"function", "function":{"name":name, "arguments":args}}]}));
                        messages.push(json!({"role":"tool", "tool_call_id":id, "content":result}));
                    } else {
                        let text = events
                            .iter()
                            .filter_map(|e| match e {
                                SemanticEvent::TextDelta(s) => Some(s.as_str()),
                                _ => None,
                            })
                            .collect::<String>();
                        assert_eq!(text.trim(), "cobalt", "{events:?}");
                        assert!(!events
                            .iter()
                            .any(|e| matches!(e, SemanticEvent::ToolCallEnd)));
                    }
                }
                assert_eq!(executions, 1);
            }
        }
    }
}
