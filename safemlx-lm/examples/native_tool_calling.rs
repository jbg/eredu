//! Native tool calling through the public semantic API.
//!
//! Run with a target checkpoint and, optionally, an external Gemma 4 assistant:
//! `cargo run -p safemlx-lm --example native_tool_calling -- TARGET [DRAFTER]`.

use std::{env, num::NonZeroUsize};

use safemlx::{Device, DeviceType, ExecutionContext};
use safemlx_lm::{
    api::{
        LoadedModel, PreparedChatDraft, PreparedChatGenerationRequest,
        PreparedChatGenerationSettings, PreparedChatInput, PreparedChatMtpGenerationOptions,
        PreparedChatMtpGenerationRequest,
    },
    backend::mlx::speculative::MlxDrafter,
    runtime::chat::{ChatTemplateRequest, NativeToolSupport, ParallelToolCallPolicy, ToolChoice},
    runtime::generation::speculative::{MtpCapability, MtpCheckpointKind},
    MtpSchedulerOptions, SemanticEvent,
};
use serde_json::json;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = env::args().skip(1);
    let target_path = arguments
        .next()
        .ok_or("usage: native_tool_calling TARGET [DRAFTER]")?;
    let drafter_path = arguments.next();

    let target = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
    let mut model = LoadedModel::load(&target_path, target.stream(), target.stream())?;
    let prepared = model.prepare_chat(ChatTemplateRequest {
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
        NativeToolSupport::Unsupported { reason } => {
            return Err(format!("native tools unavailable: {reason}").into());
        }
    }

    let settings = PreparedChatGenerationSettings {
        overrides: safemlx_lm::GenerationConfigOverrides {
            max_new_tokens: Some(256),
            ..Default::default()
        },
        ..PreparedChatGenerationSettings::default()
    };
    let scheduler = MtpSchedulerOptions {
        max_in_flight_verifications: 1,
        max_optimistic_branches: 1,
        lookahead_blocks: 1,
        ..MtpSchedulerOptions::default()
    };
    let mut events = Vec::<SemanticEvent>::new();

    let finish_reason = if let Some(drafter_path) = drafter_path {
        // Draft on CPU while the target verifies on GPU. The loaded target and
        // drafter retain their respective execution placements.
        let draft = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let mut drafter = MlxDrafter::load(&drafter_path, draft.stream(), draft.stream())?;
        model
            .generate_prepared_chat_mtp(PreparedChatMtpGenerationRequest {
                input: PreparedChatInput::rendered_prompt(&prepared),
                drafting: PreparedChatDraft::External(&mut drafter),
                settings,
                options: PreparedChatMtpGenerationOptions {
                    max_draft_tokens: NonZeroUsize::new(3).unwrap(),
                    scheduler,
                },
                caller_stop_sequences: &[],
                cancellation: safemlx_lm::GenerationCancellationToken::new(),
                on_event: |event| events.push(event),
            })?
            .finish_reason
    } else if matches!(
        model.mtp_capability(),
        MtpCapability::Ready {
            checkpoint: MtpCheckpointKind::Embedded
        }
    ) {
        model
            .generate_prepared_chat_mtp(PreparedChatMtpGenerationRequest {
                input: PreparedChatInput::rendered_prompt(&prepared),
                drafting: PreparedChatDraft::Embedded,
                settings,
                options: PreparedChatMtpGenerationOptions {
                    max_draft_tokens: NonZeroUsize::new(3).unwrap(),
                    scheduler,
                },
                caller_stop_sequences: &[],
                cancellation: safemlx_lm::GenerationCancellationToken::new(),
                on_event: |event| events.push(event),
            })?
            .finish_reason
    } else {
        model
            .generate_prepared_chat(PreparedChatGenerationRequest {
                input: PreparedChatInput::rendered_prompt(&prepared),
                settings,
                caller_stop_sequences: &[],
                cancellation: safemlx_lm::GenerationCancellationToken::new(),
                on_event: |event| events.push(event),
            })?
            .finish_reason
    };

    for event in &events {
        println!("{event:?}");
    }
    eprintln!("finish_reason: {finish_reason:?}");
    Ok(())
}
