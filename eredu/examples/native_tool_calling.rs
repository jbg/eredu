//! Native tool calling through the public semantic API.
//!
//! Run with a target checkpoint and, optionally, an external Gemma 4 assistant:
//! `cargo run -p eredu --example native_tool_calling -- TARGET MEMORY_LIMITS [DRAFTER]`.

#[path = "support/physical_memory.rs"]
mod physical_memory;
use eredu_backend_mlx::MlxBackendFactory;
use std::{env, num::NonZeroUsize};

use eredu::{
    api::{
        default_local_device, local_device_plan, ChatSourceInput, LoadedModel, LocalDevice,
        PreparedChatGenerationSettings, PreparedChatRequest,
        PreparedChatSpeculativeGenerationOptions, PreparedChatSpeculativeRequest,
        TokenizerSourceInput,
    },
    runtime::chat::{ChatTemplateRequest, NativeToolSupport, ParallelToolCallPolicy, ToolChoice},
};
use eredu_core::{
    DraftPlacementPlan, DraftingPlan, ExecutionPlan, GenerationCancellationToken,
    GenerationConfigOverrides, SemanticEvent, SpeculativeSchedulerOptions, TextInferencePolicy,
};
use serde_json::json;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = env::args().skip(1);
    let target_path = arguments
        .next()
        .ok_or("usage: native_tool_calling TARGET MEMORY_LIMITS [DRAFTER]")?;
    let capacity = physical_memory::parse(&arguments.next().ok_or("missing MEMORY_LIMITS")?)?;
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
    let planned =
        LoadedModel::load_execution_plan(&MlxBackendFactory::default(), &target_path, &plan)?;
    let (mut model, mut drafting) = planned.into_parts();
    let cancellation = GenerationCancellationToken::new();
    let tokenizer =
        model.compile_managed_plain_text_source(TokenizerSourceInput::RetainedConfiguration)?;
    let source = model
        .compile_managed_chat_source(
            &tokenizer,
            ChatSourceInput::RetainedConfiguration,
            true,
            &cancellation,
        )?
        .ok_or("cancelled before source compilation")?;
    let prepared = model
        .prepare_chat(
            &source,
            &ChatTemplateRequest {
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
            },
            &capacity,
            &cancellation,
        )?
        .ok_or("cancelled before chat preparation")?;

    match prepared.native_tool_support() {
        NativeToolSupport::Supported => {
            eprintln!(
                "native tools: {}",
                prepared.format_profile_identity().unwrap_or("registered")
            );
        }
        NativeToolSupport::Unsupported { reason } => {
            return Err(format!("native tools unavailable: {reason}").into());
        }
    }

    let settings = PreparedChatGenerationSettings {
        overrides: GenerationConfigOverrides {
            max_new_tokens: Some(256),
            ..Default::default()
        },
        inference: TextInferencePolicy {
            memory_limits: capacity.clone(),
            ..Default::default()
        },
        ..PreparedChatGenerationSettings::default()
    };
    let scheduler = SpeculativeSchedulerOptions {
        max_in_flight_verifications: 1,
        max_optimistic_branches: 1,
        lookahead_blocks: 1,
        ..SpeculativeSchedulerOptions::default()
    };
    let mut events = Vec::<SemanticEvent>::new();

    let finish_reason = if drafter_path.is_some() {
        if !drafting.is_enabled() {
            return Err("external drafting plan was not realized".into());
        }
        model
            .generate_prepared_chat_speculative(PreparedChatSpeculativeRequest {
                chat: &prepared,
                input: eredu::api::PreparedChatPrompt::Rendered,
                output_mode: eredu::api::PreparedChatOutputMode::Semantic,
                skip_special_tokens: true,
                drafting: drafting
                    .as_speculative_draft()
                    .expect("drafting is enabled"),
                settings,
                options: PreparedChatSpeculativeGenerationOptions {
                    max_draft_tokens: NonZeroUsize::new(3).unwrap(),
                    scheduler,
                },
                caller_stop_sequences: &[],
                cancellation: cancellation.clone(),
                on_event: |event| events.push(event),
            })?
            .finish_reason()
    } else {
        model
            .start_prepared_chat(
                PreparedChatRequest::new(&prepared, settings.clone()),
                &cancellation,
            )?
            .ok_or("cancelled before generation")?
            .run(&cancellation, &mut |event| events.push(event))?
            .finish_reason
    };

    for event in &events {
        println!("{event:?}");
    }
    eprintln!("finish_reason: {finish_reason:?}");
    Ok(())
}
