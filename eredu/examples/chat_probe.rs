//! Compare semantic constraints with ordinary generation on the same chat prompt.
//! Usage: chat_probe CHECKPOINT OUTPUT_PREFIX [MAX_TOKENS] [TEMPERATURE] [cpu|metal]
use eredu::{
    api::{
        local_device_plan, ControlledGenerationRecord, LoadedModel, LocalDevice,
        ObservedGenerationEvent, PreparedChatGenerationRequest, PreparedChatGenerationSettings,
        PreparedChatInput, TraceLimits,
    },
    runtime::chat::ChatTemplateRequest,
};
use eredu_backend_mlx::MlxBackendFactory;
use eredu_core::{ExecutionPlan, GenerationConfigOverrides, SemanticEvent};
use serde_json::json;
use std::ops::ControlFlow;

fn main() -> anyhow::Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    anyhow::ensure!(
        args.len() >= 2,
        "usage: chat_probe CHECKPOINT OUTPUT_PREFIX [MAX_TOKENS] [TEMPERATURE] [cpu|metal]"
    );
    let max_tokens: usize = args.get(2).map(String::as_str).unwrap_or("2048").parse()?;
    let temperature: f32 = args.get(3).map(String::as_str).unwrap_or("0").parse()?;
    let device = match args.get(4).map(String::as_str).unwrap_or("cpu") {
        "cpu" => LocalDevice::Cpu,
        "metal" => LocalDevice::Accelerator(0),
        other => anyhow::bail!("unknown device {other}"),
    };
    let plan = ExecutionPlan::fully_resident(local_device_plan(device)?);
    let (mut model, _) =
        LoadedModel::load_execution_plan(&MlxBackendFactory::default(), &args[0], &plan)?
            .into_parts();
    let messages = vec![
        json!({"role":"system", "content":"You are a concise assistant. Follow the user's instructions and use tools when requested."}),
        json!({"role":"user", "content":"Reply with just the word hello."}),
    ];
    let prepared = model.prepare_chat(ChatTemplateRequest {
        messages: messages.clone(),
        enable_thinking: Some(true),
        allow_unparsed_reasoning: true,
        add_generation_prompt: true,
        ..Default::default()
    })?;
    let prompt_ids = model.encode(prepared.rendered_prompt(), false)?;
    let settings = PreparedChatGenerationSettings {
        overrides: GenerationConfigOverrides {
            temperature: Some(temperature),
            max_new_tokens: Some(max_tokens),
            top_k: Some(40),
            top_p: Some(0.95),
            min_p: Some(0.05),
            repetition_penalty: Some(1.0),
            ..Default::default()
        },
        seed: 0,
        ..Default::default()
    };
    for mode in ["ordinary", "semantic", "controlled"] {
        model.reset()?;
        let mut events = Vec::<SemanticEvent>::new();
        let start = std::time::Instant::now();
        let (token_ids, finish_reason) = if mode == "controlled" {
            let observed = model.prepare_observed_chat(
                &prepared,
                settings,
                eredu_core::capture::CapturePlan::none(),
                TraceLimits {
                    per_record_bytes: 65536,
                    total_bytes: 16 << 20,
                },
            )?;
            let mut collect = |record: ControlledGenerationRecord| {
                if let ObservedGenerationEvent::Semantic { event, .. } = record.generation.event {
                    events.push(event);
                }
                ControlFlow::Continue(())
            };
            let mut session =
                model.start_controlled_chat(observed, &[], Default::default(), &mut collect)?;
            while session.finish_reason().is_none() {
                session.step(&mut collect)?;
            }
            (
                session.token_ids().to_vec(),
                session.finish_reason().unwrap(),
            )
        } else {
            let request = PreparedChatGenerationRequest {
                input: PreparedChatInput::rendered_prompt(&prepared),
                settings,
                caller_stop_sequences: &[],
                cancellation: Default::default(),
                on_event: |event| events.push(event),
            };
            let output = if mode == "semantic" {
                model.generate_prepared_chat(request)?
            } else {
                model.generate_prepared_text(request)?
            };
            (output.token_ids, output.finish_reason)
        };
        let decoded = model.decode(&token_ids, false)?;
        let report = json!({
            "mode":mode, "checkpoint":args[0], "messages":messages,
            "prompt":prepared.rendered_prompt(), "prompt_ids":prompt_ids,
            "temperature":temperature, "top_k":40, "top_p":0.95, "min_p":0.05, "repetition_penalty":1.0,
            "seed":0, "max_new_tokens":max_tokens, "plan":plan,
            "profile":prepared.format_profile_identity(),
            "token_ids":token_ids, "decoded":decoded, "events":events,
            "finish_reason":finish_reason, "elapsed_seconds":start.elapsed().as_secs_f64(),
        });
        std::fs::write(
            format!("{}-{mode}.json", args[1]),
            serde_json::to_vec_pretty(&report)?,
        )?;
        eprintln!(
            "{mode}: {} tokens, {:?}, {:?}",
            token_ids.len(),
            finish_reason,
            decoded
        );
    }
    Ok(())
}
