//! Opt-in smoke tests for real native-tool checkpoints.
//!
//! Tests never download data. Set the named environment variable to a local
//! model directory or GGUF and run the exact ignored test on a Metal host.

use safemlx::{Device, DeviceType, ExecutionContext};
use safemlx_lm::{
    chat::{ChatTemplateRequest, NativeToolSupport, ToolChoice},
    models::LoadedModel,
};
use serde_json::json;

fn smoke(environment: &str, expected_profile_prefix: &str) {
    let path = std::env::var(environment)
        .unwrap_or_else(|_| panic!("{environment} must name a local checkpoint"));
    let execution = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
    let mut model = LoadedModel::load(&path, execution.stream(), execution.stream())
        .unwrap_or_else(|error| panic!("failed to load {environment}={path:?}: {error}"));
    let prepared = model
        .prepare_chat(ChatTemplateRequest {
            messages: vec![json!({"role": "user", "content": "look up seven"})],
            tools: vec![json!({
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Look up one integer.",
                    "parameters": {
                        "type": "object",
                        "properties": {"value": {"type": "integer"}},
                        "required": ["value"],
                        "additionalProperties": false
                    }
                }
            })],
            tool_choice: ToolChoice::Required,
            add_generation_prompt: true,
            ..ChatTemplateRequest::default()
        })
        .unwrap_or_else(|error| panic!("failed to prepare {environment}={path:?}: {error}"));

    assert!(
        matches!(prepared.native_tool_support(), NativeToolSupport::Supported),
        "{environment} selected unsupported native tools: {:?}",
        prepared.native_tool_support()
    );
    assert!(
        prepared
            .format_profile_identity()
            .is_some_and(|identity| identity.starts_with(expected_profile_prefix)),
        "{environment} selected unexpected profile {:?}",
        prepared.format_profile_identity()
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
