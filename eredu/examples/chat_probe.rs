//! Compare semantic constraints with ordinary generation on the same chat prompt.
//! Usage: chat_probe CHECKPOINT OUTPUT_PREFIX [MAX_TOKENS] [TEMPERATURE] [cpu|metal] [CAPACITY_BYTES]
use anyhow::Context;
use eredu::{
    api::{
        ChatSourceInput, LoadedModel, LocalDevice, PreparedChatGenerationSettings,
        PreparedChatOutputMode, PreparedChatRequest, TokenizerSourceInput, local_device_plan,
    },
    runtime::chat::ChatTemplateRequest,
};
use eredu_backend_mlx::MlxBackendFactory;
use eredu_core::{
    ExecutionPlan, GenerationCancellationToken, GenerationConfigOverrides, SemanticEvent,
    TextInferencePolicy,
};
use serde_json::json;

fn main() -> anyhow::Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    anyhow::ensure!(
        args.len() >= 2,
        "usage: chat_probe CHECKPOINT OUTPUT_PREFIX [MAX_TOKENS] [TEMPERATURE] [cpu|metal] [CAPACITY_BYTES]"
    );
    let max_tokens: usize = args.get(2).map(String::as_str).unwrap_or("2048").parse()?;
    let temperature: f32 = args.get(3).map(String::as_str).unwrap_or("0").parse()?;
    let device = match args.get(4).map(String::as_str).unwrap_or("cpu") {
        "cpu" => LocalDevice::Cpu,
        "metal" => LocalDevice::Accelerator(0),
        other => anyhow::bail!("unknown device {other}"),
    };
    let capacity: u64 = args
        .get(5)
        .map(String::as_str)
        .unwrap_or("1073741824")
        .parse()?;
    anyhow::ensure!(capacity > 0, "capacity must be positive");
    let plan = ExecutionPlan::fully_resident(local_device_plan(device)?);
    let (mut model, _) =
        LoadedModel::load_execution_plan(&MlxBackendFactory::default(), &args[0], &plan)?
            .into_parts();
    let messages = vec![
        json!({"role":"system", "content":"You are a concise assistant. Follow the user's instructions and use tools when requested."}),
        json!({"role":"user", "content":"Reply with just the word hello."}),
    ];
    let cancellation = GenerationCancellationToken::new();
    let tokenizer =
        model.compile_managed_plain_text_source(TokenizerSourceInput::RetainedConfiguration)?;
    let source = model
        .compile_managed_chat_source(
            &tokenizer,
            ChatSourceInput::RetainedConfiguration,
            false,
            &cancellation,
        )?
        .context("cancelled before template compilation")?;
    let prepared = model
        .prepare_chat(
            &source,
            &ChatTemplateRequest {
                messages: messages.clone(),
                enable_thinking: Some(true),
                allow_unparsed_reasoning: true,
                add_generation_prompt: true,
                ..Default::default()
            },
            capacity,
            &cancellation,
        )?
        .context("cancelled before chat preparation")?;
    // Caller-owned report data; generation itself encodes the retained source.
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
        inference: TextInferencePolicy {
            managed_memory_capacity_bytes: Some(capacity),
            ..Default::default()
        },
        seed: 0,
        ..Default::default()
    };
    for mode in ["ordinary", "semantic", "controlled"] {
        model.reset()?;
        let mut events = Vec::<SemanticEvent>::new();
        let start = std::time::Instant::now();
        let mut request = PreparedChatRequest::new(&prepared, settings);
        request.output_mode = if mode == "ordinary" {
            PreparedChatOutputMode::Text
        } else {
            PreparedChatOutputMode::Semantic
        };
        let mut session = model
            .start_prepared_chat(request, &cancellation)?
            .context("cancelled before generation")?;
        let mut emit = |event| events.push(event);
        let output = if mode == "controlled" {
            while session.finish_reason().is_none() {
                session = session.advance(&cancellation, &mut emit)?;
            }
            session
                .into_output()
                .map_err(|_| anyhow::anyhow!("manual cursor did not terminate"))?
        } else {
            session.run(&cancellation, &mut emit)?
        };
        let token_ids = output.token_ids;
        let finish_reason = output.finish_reason;
        let decoded = model.decode(&token_ids, false)?;
        let report = json!({
            "mode":mode, "checkpoint":args[0], "messages":messages,
            "prompt":prepared.rendered_prompt(), "prompt_ids":prompt_ids,
            "temperature":temperature, "top_k":40, "top_p":0.95, "min_p":0.05, "repetition_penalty":1.0,
            "seed":0, "max_new_tokens":max_tokens, "capacity_bytes":capacity, "plan":plan,
            "profile":prepared.format_profile_identity(),
            "token_ids":&*token_ids, "decoded":decoded, "events":events,
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
