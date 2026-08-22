//! Opt-in smoke tests for real native-tool checkpoints.
//!
//! Tests never download data. Set the named environment variable to a local
//! model directory or GGUF and run the exact ignored test on a Metal host.

use eredu::backend::mlx::ModelLoadOptions;
use eredu::{
    api::{
        LoadedModel, PreparedChatGenerationRequest, PreparedChatGenerationSettings,
        PreparedChatInput,
    },
    runtime::chat::{ChatTemplateRequest, NativeToolSupport, ToolChoice},
    FinishReason, SemanticEvent,
};
use eredu_backend_mlx::native::{Device, DeviceType, ExecutionContext};
use serde_json::json;

fn profile_requires_structural_tool_tokens(identity: &str) -> bool {
    identity.starts_with("google.gemma4.")
        || identity.starts_with("openai.gpt-oss.")
        || identity.starts_with("mistral.")
        || identity.starts_with("liquid.lfm2")
        || identity.starts_with("deepseek.")
        || identity.starts_with("meta.llama-4.")
}

fn smoke(environment: &str, expected_profile_prefix: &str) {
    smoke_with_options(
        environment,
        expected_profile_prefix,
        ModelLoadOptions::default(),
    );
}

fn smoke_with_options(environment: &str, expected_profile_prefix: &str, options: ModelLoadOptions) {
    let path = std::env::var(environment)
        .unwrap_or_else(|_| panic!("{environment} must name a local checkpoint"));
    let execution = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
    let mut model = LoadedModel::load(
        eredu::backend::mlx::MlxBackend::new(execution.stream(), execution.stream()),
        &path,
        options,
    )
    .unwrap_or_else(|error| panic!("failed to load {environment}={path:?}: {error}"));
    let prepared = model
        .prepare_chat(ChatTemplateRequest {
            messages: vec![json!({
                "role": "user",
                "content": "Call lookup exactly once with {\"value\":7}. Do not answer with text."
            })],
            tools: vec![json!({
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Look up one integer.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "value": {"type": "integer", "enum": [7]}
                        },
                        "required": ["value"],
                        "additionalProperties": false
                    }
                }
            })],
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
                overrides: eredu::GenerationConfigOverrides {
                    temperature: Some(0.0),
                    max_new_tokens: Some(256),
                    ..Default::default()
                },
                seed: 0,
            },
            caller_stop_sequences: &[],
            cancellation: eredu::GenerationCancellationToken::new(),
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
        starts[0].2, "lookup",
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
        arguments,
        json!({"value": 7}),
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
    smoke_with_options(
        environment,
        "qwen.",
        ModelLoadOptions::default()
            .with_weight_residency(eredu_runtime::WeightResidency::fully_resident()),
    );
    smoke_with_options(
        environment,
        "qwen.",
        ModelLoadOptions::default().with_weight_residency(
            eredu_runtime::WeightResidency::layerwise_host(
                eredu_runtime::LayerwiseLoadOptions::default(),
            ),
        ),
    );
}

#[test]
#[ignore = "requires SAFEMLX_GEMMA4_TOOL_CHECKPOINT and an MLX Metal device"]
fn gemma4_real_checkpoint_native_tool_smoke() {
    smoke("SAFEMLX_GEMMA4_TOOL_CHECKPOINT", "google.gemma4.");
}

#[test]
#[ignore = "requires SAFEMLX_GPT_OSS_TOOL_CHECKPOINT and an MLX Metal device"]
fn gpt_oss_real_checkpoint_native_tool_smoke() {
    smoke("SAFEMLX_GPT_OSS_TOOL_CHECKPOINT", "openai.gpt-oss.harmony.");
}

#[test]
#[ignore = "requires SAFEMLX_QWEN_TOOL_CHECKPOINT and an MLX Metal device"]
fn qwen_real_checkpoint_native_tool_smoke() {
    smoke("SAFEMLX_QWEN_TOOL_CHECKPOINT", "qwen.");
}

#[test]
#[ignore = "requires SAFEMLX_QWEN2_CHECKPOINT and an MLX Metal device"]
fn qwen2_real_checkpoint_resident_and_bounded_smoke() {
    qwen_residency_smoke("SAFEMLX_QWEN2_CHECKPOINT");
}

#[test]
#[ignore = "requires SAFEMLX_QWEN3_CHECKPOINT and an MLX Metal device"]
fn qwen3_real_checkpoint_resident_and_bounded_smoke() {
    qwen_residency_smoke("SAFEMLX_QWEN3_CHECKPOINT");
}

#[test]
#[ignore = "requires SAFEMLX_QWEN3_MOE_CHECKPOINT and an MLX Metal device"]
fn qwen3_moe_real_checkpoint_resident_and_bounded_smoke() {
    qwen_residency_smoke("SAFEMLX_QWEN3_MOE_CHECKPOINT");
}

#[test]
#[ignore = "requires SAFEMLX_MISTRAL_TOOL_CHECKPOINT and an MLX Metal device"]
fn mistral_real_checkpoint_native_tool_smoke() {
    smoke("SAFEMLX_MISTRAL_TOOL_CHECKPOINT", "mistral.");
}

#[test]
#[ignore = "requires SAFEMLX_LFM2_TOOL_CHECKPOINT and an MLX Metal device"]
fn lfm2_real_checkpoint_native_tool_smoke() {
    smoke("SAFEMLX_LFM2_TOOL_CHECKPOINT", "liquid.lfm2");
}

#[test]
#[ignore = "requires SAFEMLX_DEEPSEEK_TOOL_CHECKPOINT and an MLX Metal device"]
fn deepseek_real_checkpoint_native_tool_smoke() {
    smoke("SAFEMLX_DEEPSEEK_TOOL_CHECKPOINT", "deepseek.");
}

#[test]
#[ignore = "requires SAFEMLX_LLAMA_TOOL_CHECKPOINT and an MLX Metal device"]
fn llama_real_checkpoint_native_tool_smoke() {
    smoke("SAFEMLX_LLAMA_TOOL_CHECKPOINT", "meta.llama-");
}

#[test]
#[ignore = "requires SAFEMLX_NEMOTRON_TOOL_CHECKPOINT and an MLX Metal device"]
fn nemotron_real_checkpoint_native_tool_smoke() {
    smoke("SAFEMLX_NEMOTRON_TOOL_CHECKPOINT", "nvidia.");
}
