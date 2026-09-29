//! Measure one prepared text request; no tool calls are executed.
//! Usage: prefill_memory_probe CHECKPOINT REQUEST_JSON CHUNK [CANCEL_MS]
//! CHUNK=0 disables chunking. REQUEST_JSON accepts messages/tools, optional
//! output_tokens (default 8), and positions (repeat/truncate the rendered IDs).
//! render_only checks repeated rendering without generation. capture_paths emits
//! bounded activation summaries on the original chat (not the synthetic IDs).
//! Each invocation is a fresh process; the request peak is reset after loading.
use eredu::{
    api::{
        local_device_plan, LoadedModel, PreparedChatGenerationRequest,
        PreparedChatGenerationSettings, PreparedChatInput,
    },
    runtime::chat::ChatTemplateRequest,
};
use eredu_backend_mlx::{allocator_memory, reset_allocator_peak, MlxBackendFactory};
use eredu_core::{
    ExecutionPlan, GenerationCancellationToken, GenerationConfigOverrides, PrefillChunkPolicy,
};
use serde_json::{json, Value};
use std::{
    num::NonZeroUsize,
    time::{Duration, Instant},
};

fn memory() -> anyhow::Result<Value> {
    let m = allocator_memory()?;
    Ok(
        json!({"active_bytes":m.active_bytes(), "cache_bytes":m.cached_bytes(), "peak_bytes":m.peak_bytes()}),
    )
}
fn main() -> anyhow::Result<()> {
    let args: Vec<_> = std::env::args().collect();
    anyhow::ensure!(args.len() >= 4, "CHECKPOINT REQUEST_JSON CHUNK [CANCEL_MS]");
    let chunk: usize = args[3].parse()?;
    let input: Value = serde_json::from_slice(&std::fs::read(&args[2])?)?;
    let plan =
        ExecutionPlan::fully_resident(local_device_plan(eredu::api::LocalDevice::Accelerator(0))?);
    let (mut model, _) =
        LoadedModel::load_execution_plan(&MlxBackendFactory::default(), &args[1], &plan)?
            .into_parts();
    model.synchronize()?;
    let loaded = memory()?;
    let chat_request = ChatTemplateRequest {
        messages: serde_json::from_value(input["messages"].clone())?,
        tools: input["tools"].as_array().cloned().unwrap_or_default(),
        enable_thinking: Some(false),
        add_generation_prompt: true,
        ..Default::default()
    };
    let chat = model.prepare_chat(chat_request.clone())?;

    if let Some(paths) = input["capture_paths"].as_array() {
        use eredu_core::capture::*;
        let usage = CaptureUsage {
            captures: 16,
            retained_bytes: 8 << 30,
            host_bytes: 1 << 20,
            encoded_bytes: 1 << 20,
        };
        let capture = CapturePlan {
            schema_version: CAPTURE_SCHEMA_VERSION,
            selections: paths
                .iter()
                .enumerate()
                .map(|(i, path)| CaptureSelection {
                    id: i.to_string(),
                    path: path.as_str().unwrap().into(),
                    schedule: CaptureSchedule::default(),
                    slices: vec![],
                    transform: CaptureTransform::Summary,
                })
                .collect(),
            limits: CaptureLimits {
                per_step: usage,
                cumulative: usage,
                physical_native_bytes: None,
                on_limit: CaptureLimitPolicy::Fail,
            },
        };
        let observed = model.prepare_observed_chat(
            &chat,
            PreparedChatGenerationSettings {
                overrides: GenerationConfigOverrides {
                    temperature: Some(0.0),
                    max_new_tokens: Some(1),
                    ..Default::default()
                },
                ..Default::default()
            },
            capture,
            eredu::api::TraceLimits {
                per_record_bytes: 1 << 20,
                total_bytes: 4 << 20,
            },
        )?;
        model.generate_observed_chat(observed, &[], Default::default(), |record| {
            println!("{}", serde_json::to_string(&record).unwrap());
            std::ops::ControlFlow::Continue(())
        })?;
        return Ok(());
    }
    let rendered_ids = model.encode(chat.rendered_prompt(), false)?;
    // Repeated identical rendering tests determinism without changing message or schema order.
    for _ in 0..3 {
        let repeated = model.prepare_chat(chat_request.clone())?;
        anyhow::ensure!(
            repeated.rendered_prompt() == chat.rendered_prompt(),
            "rendering changed for identical input"
        );
    }
    if input["render_only"].as_bool() == Some(true) {
        println!(
            "{}",
            json!({"rendered_positions": rendered_ids.len(), "prompt_ids":rendered_ids})
        );
        return Ok(());
    }
    let mut ids = rendered_ids.clone();
    if let Some(positions) = input["positions"].as_u64() {
        ids = ids
            .iter()
            .copied()
            .cycle()
            .take(positions as usize)
            .collect();
    }
    let cancellation = GenerationCancellationToken::new();
    let settings = PreparedChatGenerationSettings {
        overrides: GenerationConfigOverrides {
            temperature: Some(0.0),
            max_new_tokens: Some(input["output_tokens"].as_u64().unwrap_or(8) as usize),
            ..Default::default()
        },
        prefill: NonZeroUsize::new(chunk)
            .map_or(PrefillChunkPolicy::Unchunked, PrefillChunkPolicy::Bounded),
        ..Default::default()
    };
    let request = PreparedChatGenerationRequest {
        input: PreparedChatInput::TokenIds {
            prepared_chat: &chat,
            token_ids: ids.clone(),
        },
        settings,
        caller_stop_sequences: &[],
        cancellation: cancellation.clone(),
        on_event: |_| {},
    };
    let forecast = model.forecast_prepared_generation(&request, &Default::default())?;
    reset_allocator_peak()?;
    let start = Instant::now();
    let (done, wait) = std::sync::mpsc::channel::<()>();
    let cancel_ms = args.get(4).map(|s| s.parse::<u64>()).transpose()?;
    let timer = cancel_ms.map(|ms| {
        std::thread::spawn(move || {
            if wait.recv_timeout(Duration::from_millis(ms)).is_err() {
                let requested = start.elapsed().as_secs_f64();
                cancellation.cancel();
                Some(requested)
            } else {
                None
            }
        })
    });
    let output = model.generate_prepared_chat(request);
    let generation_seconds = start.elapsed().as_secs_f64();
    let _ = done.send(());
    let cancelled_at = timer.and_then(|t| t.join().unwrap());
    model.synchronize()?;
    let generated = memory()?;
    model.reset()?;
    model.synchronize()?;
    println!(
        "{}",
        serde_json::to_string(&json!({
            "checkpoint":args[1], "chunk":chunk, "rendered_positions":rendered_ids.len(),
            "prompt_ids":ids, "temperature":0, "token_ids":output.as_ref().map(|o| o.token_ids.clone()).unwrap_or_default(),
            "generation_error":output.as_ref().err().map(ToString::to_string),
            "finish_reason":output.as_ref().ok().map(|o| o.finish_reason), "generation_seconds":generation_seconds,
            "cancel_requested_seconds":cancelled_at,
            "cancel_return_latency_seconds":cancelled_at.map(|at| generation_seconds-at),
            "loaded":loaded, "request":generated, "reset":memory()?,
            "forecast":forecast, "plan":plan,
        }))?
    );
    output?;
    Ok(())
}
