use std::path::PathBuf;

use eredu::{
    api::{default_local_device, local_device_plan, LocalBackendFactory, LocalModel},
    runtime::chat::ChatTemplateRequest,
};
use eredu_architectures::ModelKind;
use eredu_core::{ExecutionPlan, GenerationConfigOverrides, TextGenerationConfig};

fn main() -> anyhow::Result<()> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let model_dir = args
        .first()
        .map(PathBuf::from)
        .or_else(default_e4b_snapshot)
        .expect("usage: cargo run -p eredu --example gemma4_generate -- <model-dir> [prompt]");
    let prompt = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "what is MLX?".to_string());
    let temp = args
        .get(2)
        .and_then(|value| value.parse::<f32>().ok())
        .unwrap_or(0.0);

    let plan = ExecutionPlan::fully_resident(local_device_plan(default_local_device())?);
    let planned =
        LocalModel::load_execution_plan(&LocalBackendFactory::default(), &model_dir, &plan)?;
    let (mut model, _) = planned.into_parts();

    let prepared = model.prepare_chat(ChatTemplateRequest {
        messages: vec![gemma4_message(&prompt, model.model_family())],
        add_generation_prompt: true,
        ..ChatTemplateRequest::default()
    })?;
    let rendered = prepared.rendered_prompt().to_owned();
    println!("\n=== prompt ===\n{rendered}\n");
    println!("temperature: {temp}");

    let ids = model.encode(&rendered, false)?;
    let eos = model.eos_token_ids().to_vec();
    print_first_token_distribution(&mut model, ids.clone())?;
    model.reset()?;
    let mut output_ids = Vec::new();

    {
        let resolved = model.resolve_generation_config(GenerationConfigOverrides {
            temperature: Some(temp),
            ..Default::default()
        })?;
        let mut generator =
            model.generate_tokens(ids, TextGenerationConfig::new(resolved).with_seed(0))?;
        for _ in 0..120 {
            let token = match generator.next() {
                Some(token) => token?,
                None => break,
            };
            let id = token;
            output_ids.push(id);
            if eos.contains(&id) {
                break;
            }
        }
    }

    println!("=== output ids ===\n{output_ids:?}\n");
    println!("=== output ===\n{}", model.decode(&output_ids, false)?);
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

fn print_first_token_distribution(model: &mut LocalModel, tokens: Vec<u32>) -> anyhow::Result<()> {
    let resolved = model.resolve_generation_config(GenerationConfigOverrides {
        temperature: Some(0.0),
        ..Default::default()
    })?;
    let mut generator = model.generate_tokens(tokens, TextGenerationConfig::new(resolved))?;
    let Some(first) = generator.next() else {
        return Ok(());
    };
    let first_id = first?;
    drop(generator);
    println!(
        "first greedy id: {first_id} {:?}",
        model.decode(&[first_id], false)?
    );
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
