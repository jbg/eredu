//! Header-only admission of released DiffusionGemma SafeTensors or GGUF.
use eredu_architectures::diffusion_gemma::{safetensors_plan, FamilyConfig};
use eredu_checkpoint::{
    safetensors::SafetensorsMetadataCatalog,
    validation::{resolve_safetensors_plan, CheckpointValidation, SafetensorsCatalog},
};
use std::{error::Error, path::PathBuf};

fn main() -> Result<(), Box<dyn Error>> {
    let path = PathBuf::from(
        std::env::args_os()
            .nth(1)
            .ok_or("usage: inspect_diffusion_gemma CHECKPOINT_DIRECTORY")?,
    );
    let inspection = eredu_architectures::configuration::inspect_artifact(&path)?;
    let execution =
        eredu_architectures::replicated_text::replicated_text_execution_class(&inspection)?;
    let text = match &execution {
        eredu_architectures::replicated_text::ReplicatedTextExecutionClass::Replicated(text) => {
            text
        }
        eredu_architectures::replicated_text::ReplicatedTextExecutionClass::Routed(routed) => {
            routed.text()
        }
        _ => return Err("released artifact selected an unexpected execution class".into()),
    };
    let geometry = text
        .block_refinement()
        .ok_or("cold selection lost refinement geometry")?;
    println!(
        "Registry admitted {:?}; cold geometry: canvas {}, vocabulary {}, {} parameters",
        inspection.architecture_plan().model_kind(),
        geometry.canvas_length,
        geometry.vocabulary_size,
        text.parameters().len()
    );
    if path.is_file() {
        let checkpoint = eredu_gguf::Checkpoint::open(&path)?;
        let metadata = checkpoint
            .metadata()
            .iter()
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect();
        let config = FamilyConfig::from_gguf_metadata(&checkpoint, &metadata)?;
        let plan = eredu_architectures::diffusion_gemma::gguf_plan(&config)
            .map_err(std::io::Error::other)?;
        if let Err(validation) = eredu_checkpoint::validation::resolve_gguf_plan(&checkpoint, &plan)
        {
            if let CheckpointValidation::Invalid(issues) = &validation {
                eprintln!("{} admission issues", issues.len());
                for issue in issues.iter().take(8) {
                    eprintln!("{:?}: {}", issue.kind, issue.detail);
                }
            }
            return Err(std::io::Error::other(
                "DiffusionGemma GGUF does not satisfy the exact schema",
            )
            .into());
        }
        println!(
            "Exactly admitted {} GGUF tensors, {} shared text layers, canvas {}, modalities {:?}",
            checkpoint.tensors().count(),
            config.text.num_hidden_layers(),
            config.canvas_length,
            config.input_modalities()
        );
        return Ok(());
    }
    let config = FamilyConfig::from_hf_json(&std::fs::read(path.join("config.json"))?)?;
    let plan = safetensors_plan(&config).map_err(std::io::Error::other)?;
    let catalog = SafetensorsMetadataCatalog::discover(&path)?;
    if let Err(validation) = resolve_safetensors_plan(&catalog, &plan) {
        if let CheckpointValidation::Invalid(issues) = &validation {
            eprintln!("{} admission issues", issues.len());
            for issue in issues.iter().take(8) {
                eprintln!("{:?}: {}", issue.kind, issue.detail);
            }
        }
        return Err(std::io::Error::other(
            "DiffusionGemma checkpoint does not satisfy the exact schema",
        )
        .into());
    }
    println!(
        "Exactly admitted {} tensors, {} shared text layers, canvas {}, modalities {:?}",
        catalog.keys().len(),
        config.text.num_hidden_layers(),
        config.canvas_length,
        config.input_modalities()
    );
    let residency =
        eredu_architectures::diffusion_gemma::expert_residency_catalog(&catalog, &config)
            .map_err(std::io::Error::other)?;
    let admitted: u64 = residency
        .units()
        .iter()
        .map(|unit| unit.byte_len().expect("inferred expert bytes"))
        .sum();
    use eredu_checkpoint::recipe::RecipeCatalog;
    let mut physical = 0u64;
    for index in 0..config.text.num_hidden_layers() {
        for field in ["gate_up_proj", "down_proj"] {
            physical += catalog
                .tensor_metadata(&format!("model.decoder.layers.{index}.experts.{field}"))?
                .encoded_byte_len;
        }
    }
    if admitted != physical {
        return Err(
            "independent expert units duplicate or omit admitted projection storage".into(),
        );
    }
    println!(
        "{} independently resident experts, {} projection bytes, one shared stack",
        residency.units().len(),
        admitted
    );
    Ok(())
}
