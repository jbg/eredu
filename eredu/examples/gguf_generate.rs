use std::path::PathBuf;

use eredu::{
    api::{default_local_device, local_device_plan, LocalBackendFactory, LocalModel},
    runtime::chat::ChatTemplateRequest,
};
use eredu_core::{ExecutionPlan, GenerationConfigOverrides, TextGenerationConfig};

fn generate(
    model: &mut LocalModel,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
) -> anyhow::Result<(usize, Vec<u32>)> {
    let prompt_ids = model.encode(prompt, false)?;
    let prompt_len = prompt_ids.len();
    let eos_token_ids = model.eos_token_ids().to_vec();
    let sampling = model.resolve_generation_config(GenerationConfigOverrides {
        temperature: Some(temperature),
        max_new_tokens: Some(max_tokens),
        ..Default::default()
    })?;
    let mut output_ids = Vec::new();
    let generator = model.generate_tokens(prompt_ids, TextGenerationConfig::new(sampling))?;
    for token in generator {
        let token_id = token?;
        output_ids.push(token_id);
        if eos_token_ids.contains(&token_id) {
            break;
        }
    }
    Ok((prompt_len, output_ids))
}

fn main() -> anyhow::Result<()> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let gguf_file = args.first().map(PathBuf::from).ok_or_else(|| {
        anyhow::anyhow!(
            "usage: cargo run -p eredu --example gguf_generate -- <model.gguf> [prompt] [max-tokens] [temperature]"
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

    let plan = ExecutionPlan::fully_resident(local_device_plan(default_local_device())?);
    let planned =
        LocalModel::load_execution_plan(&LocalBackendFactory::default(), &gguf_file, &plan)?;
    let (mut model, _) = planned.into_parts();

    println!("model family: {}", model.model_family().canonical_name());
    println!("effective model type: {}", model.effective_model_type());
    println!("chat template: {}", model.has_chat_template());

    let rendered = if model.has_chat_template() {
        model
            .prepare_chat(ChatTemplateRequest {
                messages: vec![serde_json::json!({
                    "role": "user",
                    "content": prompt,
                })],
                add_generation_prompt: true,
                ..ChatTemplateRequest::default()
            })?
            .rendered_prompt()
            .to_owned()
    } else {
        prompt.to_owned()
    };
    let (prompt_len, output_ids) = generate(&mut model, &rendered, max_tokens, temperature)?;

    println!("prompt tokens: {prompt_len}");
    println!("output ids: {output_ids:?}");
    println!("output: {}", model.decode(&output_ids, false)?);
    Ok(())
}
