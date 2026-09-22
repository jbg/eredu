#[path = "support/physical_memory.rs"]
mod physical_memory;
use std::path::PathBuf;

use anyhow::{ensure, Context};

use eredu::{
    api::{
        default_local_device, local_device_plan, ChatSourceInput, LoadedModel,
        PreparedChatGenerationSettings, PreparedChatOutputMode, PreparedChatRequest,
        TokenizerSourceInput,
    },
    runtime::chat::ChatTemplateRequest,
};
use eredu_architectures::ModelKind;
use eredu_backend_mlx::{backend::MlxBackend, MlxBackendFactory};
use eredu_core::{
    ExecutionPlan, GenerationCancellationToken, GenerationConfigOverrides, TextInferencePolicy,
};

fn main() -> anyhow::Result<()> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let model_dir = args
        .first()
        .map(PathBuf::from)
        .or_else(default_e4b_snapshot)
        .expect("usage: cargo run -p eredu --example gemma4_generate -- <model-dir> [prompt] [temperature] [domain=bytes|unlimited]");
    let prompt = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "what is MLX?".to_string());
    let temp = args
        .get(2)
        .and_then(|value| value.parse::<f32>().ok())
        .unwrap_or(0.0);

    // Physical-domain limits apply during loading, preparation and generation.
    let capacity =
        physical_memory::parse(args.get(3).map(String::as_str).unwrap_or("host=unlimited"))?;

    let plan = ExecutionPlan::fully_resident(local_device_plan(default_local_device())?);
    let planned =
        LoadedModel::load_execution_plan(&MlxBackendFactory::default(), &model_dir, &plan)?;
    let (mut model, _) = planned.into_parts();

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
                messages: vec![gemma4_message(&prompt, model.model_family())],
                add_generation_prompt: true,
                ..ChatTemplateRequest::default()
            },
            &capacity,
            &cancellation,
        )?
        .context("cancelled before chat preparation")?;
    println!("\n=== prompt ===\n{}\n", prepared.rendered_prompt());
    println!("temperature: {temp}; physical memory limits: {capacity:?}");

    print_first_token_distribution(&mut model, &prepared, &capacity, &cancellation)?;
    model.reset()?;
    let maximum = model
        .resolve_generation_config(GenerationConfigOverrides {
            temperature: Some(temp),
            ..Default::default()
        })?
        .max_new_tokens
        .unwrap_or(120)
        .min(120);
    let mut request = PreparedChatRequest::new(&prepared, settings(temp, maximum, &capacity));
    request.output_mode = PreparedChatOutputMode::Text;
    request.skip_special_tokens = false;
    let output = model
        .start_prepared_chat(request, &cancellation)?
        .context("cancelled before generation")?
        .run(&cancellation, &mut |_| {})?;
    println!("=== output ids ===\n{:?}\n", output.token_ids);
    println!(
        "=== output ===\n{}",
        model.decode(&output.token_ids, false)?
    );
    Ok(())
}

fn gemma4_message(prompt: &str, model_family: ModelKind) -> serde_json::Value {
    if model_family == ModelKind::Gemma4 {
        serde_json::json!({
            "role": "user",
            "content": [{"type": "text", "text": prompt, "content": prompt}],
        })
    } else {
        serde_json::json!({"role": "user", "content": prompt})
    }
}

fn settings(
    temperature: f32,
    maximum: usize,
    capacity: &eredu_core::MemoryLimitDeclarations,
) -> PreparedChatGenerationSettings {
    PreparedChatGenerationSettings {
        overrides: GenerationConfigOverrides {
            temperature: Some(temperature),
            max_new_tokens: Some(maximum),
            ..Default::default()
        },
        inference: TextInferencePolicy {
            memory_limits: capacity.clone(),
            ..Default::default()
        },
        seed: 0,
        ..Default::default()
    }
}

fn print_first_token_distribution(
    model: &mut LoadedModel<MlxBackend<'_>>,
    chat: &eredu::runtime::chat::PreparedChat,
    capacity: &eredu_core::MemoryLimitDeclarations,
    cancellation: &GenerationCancellationToken,
) -> anyhow::Result<()> {
    let mut request = PreparedChatRequest::new(chat, settings(0.0, 1, capacity));
    request.output_mode = PreparedChatOutputMode::Text;
    request.skip_special_tokens = false;
    let output = model
        .start_prepared_chat(request, cancellation)?
        .context("cancelled before greedy inspection")?
        .run(cancellation, &mut |_| {})?;
    if let Some(&first_id) = output.token_ids.first() {
        println!(
            "first greedy id: {first_id} {:?}",
            model.decode(&[first_id], false)?
        );
    }
    Ok(())
}

fn default_e4b_snapshot() -> Option<PathBuf> {
    let home = std::env::var_os("HOME").map(PathBuf::from)?;
    let snapshots = home
        .join(".cache/huggingface/hub")
        .join("models--mlx-community--gemma-4-e4b-it-4bit")
        .join("snapshots");
    snapshots
        .read_dir()
        .ok()?
        .flatten()
        .map(|entry| entry.path())
        .find(|path| path.join("config.json").exists())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemma4_message_uses_typed_content_parts() {
        assert_eq!(
            gemma4_message("hello", ModelKind::Gemma4),
            serde_json::json!({
                "role": "user",
                "content": [{"type": "text", "text": "hello", "content": "hello"}],
            })
        );
    }

    #[test]
    fn non_gemma4_message_uses_plain_content() {
        assert_eq!(
            gemma4_message("hello", ModelKind::Llama),
            serde_json::json!({"role": "user", "content": "hello"})
        );
    }
}
