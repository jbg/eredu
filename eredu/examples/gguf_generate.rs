use std::path::PathBuf;

use anyhow::Context;
use eredu::{
    api::{
        ChatSourceInput, LoadedModel, ManagedPlainTextRequest, PreparedChatGenerationSettings,
        PreparedChatOutputMode, PreparedChatRequest, TokenizerSourceInput, default_local_device,
        local_device_plan,
    },
    runtime::chat::ChatTemplateRequest,
};
use eredu_backend_mlx::MlxBackendFactory;
use eredu_core::{
    ExecutionPlan, GenerationCancellationToken, GenerationConfigOverrides, TextInferencePolicy,
};

fn main() -> anyhow::Result<()> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let gguf_file = args.first().map(PathBuf::from).ok_or_else(|| {
        anyhow::anyhow!(
            "usage: cargo run -p eredu --example gguf_generate -- <model.gguf> [prompt] [max-tokens] [temperature] [capacity-bytes]"
        )
    })?;
    let prompt = args
        .get(1)
        .map(String::as_str)
        .unwrap_or("Briefly explain what MLX is.");
    let max_tokens = args
        .get(2)
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(16);
    let temperature = args
        .get(3)
        .map(|value| value.parse::<f32>())
        .transpose()?
        .unwrap_or(0.0);

    let capacity = args
        .get(4)
        .map(|value| value.parse::<u64>())
        .transpose()?
        .unwrap_or(1024 * 1024 * 1024);
    anyhow::ensure!(capacity > 0, "capacity must be positive");

    let plan = ExecutionPlan::fully_resident(local_device_plan(default_local_device())?);
    let planned =
        LoadedModel::load_execution_plan(&MlxBackendFactory::default(), &gguf_file, &plan)?;
    let (mut model, _) = planned.into_parts();

    println!("model family: {}", model.model_family().canonical_name());
    println!("effective model type: {}", model.effective_model_type());
    println!("chat template: {}", model.has_chat_template());

    let cancellation = GenerationCancellationToken::new();
    // The loader has already selected the embedded or sidecar configuration.
    // Reuse that complete owner without reopening either artifact.
    let tokenizer =
        model.compile_managed_plain_text_source(TokenizerSourceInput::RetainedConfiguration)?;
    let settings = PreparedChatGenerationSettings {
        overrides: GenerationConfigOverrides {
            temperature: Some(temperature),
            max_new_tokens: Some(max_tokens),
            ..Default::default()
        },
        inference: TextInferencePolicy {
            managed_memory_capacity_bytes: Some(capacity),
            ..Default::default()
        },
        ..Default::default()
    };
    if model.has_chat_template() {
        let source = model
            .compile_managed_chat_source(
                &tokenizer,
                ChatSourceInput::RetainedConfiguration,
                false,
                &cancellation,
            )?
            .context("cancelled before template compilation")?;
        let chat = model
            .prepare_chat(
                &source,
                &ChatTemplateRequest {
                    messages: vec![serde_json::json!({"role":"user","content":prompt})],
                    add_generation_prompt: true,
                    ..Default::default()
                },
                capacity,
                &cancellation,
            )?
            .context("cancelled before chat preparation")?;
        let mut request = PreparedChatRequest::new(&chat, settings);
        request.output_mode = PreparedChatOutputMode::Text;
        let session = model
            .start_prepared_chat(request, &cancellation)?
            .context("cancelled before generation")?;
        let mut text = String::new();
        let result = session.run(&cancellation, &mut |event| {
            if let eredu_core::generation::SemanticEvent::TextDelta(delta) = event {
                text.push_str(&delta);
            }
        })?;
        println!("output ids: {:?}", result.token_ids);
        println!("output: {text}");
    } else {
        let output = model
            .generate_managed_plain_text(
                &tokenizer,
                ManagedPlainTextRequest::new(prompt, settings),
                &cancellation,
                &mut |_| {},
            )?
            .context("cancelled before generation")?;
        println!("output ids: {:?}", output.token_ids);
        println!("output: {}", output.text.as_str());
    }
    Ok(())
}
