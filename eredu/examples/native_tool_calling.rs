//! Native tool calling through the public semantic API.
//!
//! Run with a target checkpoint and, optionally, an external Gemma 4 assistant:
//! `cargo run -p eredu --example native_tool_calling -- TARGET [DRAFTER]`.

use eredu_backend_mlx::MlxBackendFactory;
use std::env;

use eredu::{
    api::{
        default_local_device, local_device_plan, GenerationRequest, LoadedModel, LocalDevice,
        PreparedChatGenerationSettings, PreparedChatInput,
    },
    runtime::chat::{ChatTemplateRequest, NativeToolSupport, ParallelToolCallPolicy, ToolChoice},
};
use eredu_core::{
    DraftPlacementPlan, DraftingPlan, ExecutionPlan, GenerationCancellationToken,
    GenerationConfigOverrides, SemanticEvent,
};
use serde_json::json;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = env::args().skip(1);
    let target_path = arguments
        .next()
        .ok_or("usage: native_tool_calling TARGET [DRAFTER]")?;
    let drafter_path = arguments.next();

    let mut plan = ExecutionPlan::fully_resident(local_device_plan(default_local_device())?);
    if let Some(path) = &drafter_path {
        plan = plan.with_drafting(DraftingPlan::External {
            model: path.clone(),
            placement: DraftPlacementPlan::Device {
                device: local_device_plan(LocalDevice::Cpu)?,
            },
            max_draft_tokens: 3,
            lookahead: true,
            adaptive_lookahead: false,
        });
    }
    let mut session =
        LoadedModel::load_execution_plan(&MlxBackendFactory::default(), &target_path, &plan)?;
    let prepared = session.model_mut().prepare_chat(ChatTemplateRequest {
        messages: vec![json!({
            "role": "user",
            "content": "What is the weather in Bogotá?"
        })],
        tools: vec![json!({
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Return current weather for one city.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                    "additionalProperties": false
                }
            }
        })],
        tool_choice: ToolChoice::Auto,
        parallel_tool_calls: ParallelToolCallPolicy::Disabled,
        add_generation_prompt: true,
        ..ChatTemplateRequest::default()
    })?;

    match prepared.native_tool_support() {
        NativeToolSupport::Supported => {
            eprintln!(
                "native tools: {}",
                prepared.format_profile_identity().unwrap_or("registered")
            );
        }
        NativeToolSupport::Unsupported { rejection } => {
            return Err(format!("native tools unavailable: {rejection}").into());
        }
    }

    let settings = PreparedChatGenerationSettings {
        overrides: GenerationConfigOverrides {
            max_new_tokens: Some(256),
            ..Default::default()
        },
        ..PreparedChatGenerationSettings::default()
    };
    let mut events = Vec::<SemanticEvent>::new();
    let output = session.generate(GenerationRequest {
        output_mode: eredu::api::OutputMode::Semantic,
        control_handle: None,
        input: PreparedChatInput::rendered_prompt(&prepared),
        settings,
        caller_stop_sequences: &[],
        cancellation: GenerationCancellationToken::new(),
        on_event: |event| events.push(event),
    })?;
    let finish_reason = output.finish_reason();

    for event in &events {
        println!("{event:?}");
    }
    eprintln!("finish_reason: {finish_reason:?}");
    Ok(())
}
