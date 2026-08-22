use std::path::PathBuf;

use eredu::{
    api::LoadedModel, GenerationConfigOverrides, TextGenerationBackend, TextGenerationConfig,
    TokenOutput,
};
use eredu_backend_mlx::native::{Device, DeviceType, ExecutionContext};

fn generate<B: TextGenerationBackend>(
    model: &mut LoadedModel<B>,
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
        let token_id = token?.token_id()?;
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

    let ctx = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
    let weights_ctx = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = ctx.stream();
    let mut model = LoadedModel::load(
        eredu_backend_mlx::MlxBackend::new(stream, weights_ctx.stream()),
        &gguf_file,
        Default::default(),
    )?;

    println!("model type: {}", model.model_type());
    println!("chat template: {}", model.has_chat_template());

    let rendered = model
        .apply_chat_template_json(
            vec![vec![serde_json::json!({
                "role": "user",
                "content": prompt,
            })]],
            None,
            true,
        )?
        .unwrap_or_else(|| prompt.to_owned());
    let (prompt_len, output_ids) = generate(&mut model, &rendered, max_tokens, temperature)?;

    println!("prompt tokens: {prompt_len}");
    println!("output ids: {output_ids:?}");
    println!("output: {}", model.decode(&output_ids, false)?);
    Ok(())
}
