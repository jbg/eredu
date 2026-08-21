//! MLX architecture binding against portable checkpoint catalogs.

use std::collections::HashMap;
#[cfg(test)]
use std::path::Path;

use safemlx::ops::{GgufCheckpoint, GgufMetadataValue};
use serde_json::Value;

use eredu_core::{GgufArchitecture, ModelKind};

use super::ModelLoadOptions;
use crate::backend::mlx::runtime::checkpoint::load::GgufTensorNames;
use crate::backend::mlx::{error::Error, runtime::checkpoint::store::SafetensorsWeightStore};
use crate::composition::llama::checkpoint as llama_checkpoint;

pub(crate) use eredu_checkpoint::validation::{
    CheckpointIssue as StructuralIssue, CheckpointIssueKind as StructuralIssueKind,
    CheckpointValidation as StructuralValidation,
};

pub(crate) trait GgufArchitectureValidation {
    fn validate_load_policy(self, options: ModelLoadOptions) -> Result<(), Error>;
    fn validate_catalog(
        self,
        checkpoint: &GgufCheckpoint,
        metadata: &HashMap<String, GgufMetadataValue>,
    ) -> Result<(), Error>;
}

impl GgufArchitectureValidation for GgufArchitecture {
    fn validate_load_policy(self, options: ModelLoadOptions) -> Result<(), Error> {
        options.validate_preparation(
            self.model_kind(),
            Some(self),
            eredu_core::ArtifactFormat::Gguf,
        )?;
        Ok(())
    }

    fn validate_catalog(
        self,
        checkpoint: &GgufCheckpoint,
        metadata: &HashMap<String, GgufMetadataValue>,
    ) -> Result<(), Error> {
        if checkpoint.catalog().physical_tensor_count() == 0 {
            return Err(Error::UnsupportedArchitecture(
                "GGUF model checkpoint contains no tensors".into(),
            ));
        }
        let prefix = self.metadata_name();
        for suffix in ["block_count", "embedding_length"] {
            let key = format!("{prefix}.{suffix}");
            let value = metadata
                .get(&key)
                .and_then(GgufMetadataValue::as_i64)
                .ok_or_else(|| {
                    Error::UnsupportedArchitecture(format!(
                        "GGUF metadata key {key:?} must be a present integer"
                    ))
                })?;
            if value <= 0 {
                return Err(Error::UnsupportedArchitecture(format!(
                    "GGUF metadata key {key:?} must be positive, got {value}"
                )));
            }
        }
        if !checkpoint
            .catalog()
            .tensors()
            .any(|tensor| tensor.descriptor().name == "token_embd.weight")
        {
            return Err(Error::UnsupportedArchitecture(
                "GGUF model checkpoint is missing required tensor \"token_embd.weight\"".into(),
            ));
        }
        if matches!(self, Self::Qwen35 | Self::Qwen35Moe | Self::Qwen3Next)
            && checkpoint.catalog().tensors().any(|tensor| {
                let name = tensor.descriptor().name.as_str();
                name.starts_with("v.") || name.starts_with("mm.")
            })
        {
            return Err(Error::UnsupportedArchitecture(
                "multimodal Qwen3-Next/Qwen3.5 GGUF checkpoints are not supported".into(),
            ));
        }
        Ok(())
    }
}

pub(crate) fn validate_safetensors(
    kind: ModelKind,
    config: &Value,
    store: &SafetensorsWeightStore,
    options: ModelLoadOptions,
) -> StructuralValidation {
    let validation = match kind {
        ModelKind::DeepSeekV3 => validate_neutral_deepseek_v3_safetensors(config, store),
        ModelKind::DeepSeekV4 => validate_neutral_deepseek_v4_safetensors(config, store),
        ModelKind::Gemma4 => validate_neutral_gemma4_safetensors(config, store),
        ModelKind::GptOss => validate_neutral_gpt_oss_safetensors(config, store),
        ModelKind::Inkling => validate_neutral_inkling_safetensors(config, store),
        ModelKind::KimiLinear => validate_neutral_kimi_safetensors(config, store),
        ModelKind::Lfm2 => validate_neutral_lfm2_safetensors(config, store),
        ModelKind::Llama => llama_checkpoint::validate_safetensors(config, store),
        ModelKind::MuseGlimmer => validate_neutral_muse_glimmer_safetensors(config, store),
        ModelKind::NemotronH => validate_neutral_nemotron_safetensors(config, store),
        ModelKind::Moshi => validate_neutral_moshi_safetensors(config, store),
        ModelKind::Qwen2 | ModelKind::Qwen3 => validate_neutral_qwen_safetensors(config, store),
        ModelKind::Qwen3Next | ModelKind::Qwen35 => {
            validate_neutral_qwen_hybrid_safetensors(config, store)
        }
        ModelKind::Qwen3Vl | ModelKind::Qwen3VlMoe => {
            validate_neutral_qwen_vl_safetensors(kind, config, store)
        }
    };
    validation.with_strict_catalog(options.weight_residency.strict_loading())
}

fn validate_neutral_moshi_safetensors(
    config: &Value,
    store: &SafetensorsWeightStore,
) -> StructuralValidation {
    let config = match eredu_architectures::moshi::MoshiConfig::from_config_value(Some(config)) {
        Ok(config) => config,
        Err(error) => return invalid_geometry(error.to_string()),
    };
    let plan = match eredu_architectures::moshi::safetensors_plan(&config) {
        Ok(plan) => plan,
        Err(error) => return invalid_geometry(error),
    };
    let validation = eredu_checkpoint::validation::validate_safetensors_plan(store, &plan);
    if validation != StructuralValidation::Exact {
        return validation;
    }
    match eredu_architectures::moshi::canonical_recipes(&config, store) {
        Ok(_) => StructuralValidation::Exact,
        Err(error) => invalid_geometry(error),
    }
}

#[cfg(test)]
pub(crate) fn validate_safetensors_load_path(
    kind: ModelKind,
    model_dir: &Path,
    options: ModelLoadOptions,
) -> Result<(), Error> {
    let config: Value = serde_json::from_slice(&std::fs::read(model_dir.join("config.json"))?)?;
    let store =
        SafetensorsWeightStore::open(model_dir).map_err(|error| Error::Other(Box::new(error)))?;
    validate_safetensors(kind, &config, &store, options)
        .into_loader_result()
        .map_err(Error::from)
}

pub(crate) fn validate_gguf(
    architecture: GgufArchitecture,
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
    options: ModelLoadOptions,
) -> StructuralValidation {
    let validation = match architecture {
        GgufArchitecture::DeepSeek2 => validate_neutral_deepseek_v3_gguf(checkpoint, metadata),
        GgufArchitecture::DeepSeek4 => validate_neutral_deepseek_v4_gguf(checkpoint, metadata),
        GgufArchitecture::GptOss => validate_neutral_gpt_oss_gguf(checkpoint, metadata),
        GgufArchitecture::Gemma4 => {
            if let Err(error) = architecture.validate_load_policy(options) {
                invalid_geometry(error.to_string())
            } else {
                validate_neutral_gemma4_gguf(checkpoint, metadata)
            }
        }
        GgufArchitecture::Inkling => {
            if let Err(error) = architecture.validate_load_policy(options) {
                invalid_geometry(error.to_string())
            } else {
                validate_neutral_inkling_gguf(checkpoint, metadata)
            }
        }
        GgufArchitecture::Lfm2 | GgufArchitecture::Lfm2Moe => {
            validate_neutral_lfm2_gguf(checkpoint, metadata)
        }
        GgufArchitecture::Llama | GgufArchitecture::Mistral => {
            llama_checkpoint::validate_gguf(checkpoint, metadata)
        }
        GgufArchitecture::MuseGlimmer => validate_neutral_muse_glimmer_gguf(checkpoint, metadata),
        GgufArchitecture::NemotronH | GgufArchitecture::NemotronHMoe => {
            if let Err(error) = architecture.validate_load_policy(options) {
                invalid_geometry(error.to_string())
            } else {
                validate_neutral_nemotron_gguf(checkpoint, metadata)
            }
        }
        GgufArchitecture::Qwen2 | GgufArchitecture::Qwen3 | GgufArchitecture::Qwen3Moe => {
            validate_neutral_qwen_gguf(checkpoint, metadata)
        }
        architecture @ (GgufArchitecture::Qwen3Vl | GgufArchitecture::Qwen3VlMoe) => {
            if let Err(error) = architecture.validate_load_policy(options) {
                invalid_geometry(error.to_string())
            } else {
                validate_neutral_qwen_vl_gguf(architecture, checkpoint, metadata)
            }
        }
        GgufArchitecture::KimiLinear => validate_neutral_kimi_gguf(checkpoint, metadata),
        GgufArchitecture::Qwen35 | GgufArchitecture::Qwen35Moe | GgufArchitecture::Qwen3Next => {
            validate_neutral_qwen_hybrid_gguf(checkpoint, metadata)
        }
    };
    validation.with_strict_catalog(options.weight_residency.strict_loading())
}

fn validate_neutral_gemma4_safetensors(
    config: &Value,
    store: &SafetensorsWeightStore,
) -> StructuralValidation {
    let bytes = match serde_json::to_vec(config) {
        Ok(bytes) => bytes,
        Err(error) => return invalid_geometry(error.to_string()),
    };
    let args = match eredu_architectures::gemma4::FamilyConfig::from_hf_json(&bytes) {
        Ok(args) => args,
        Err(error) => return invalid_geometry(error.to_string()),
    };
    let plan = match eredu_architectures::gemma4::safetensors_plan(&args) {
        Ok(plan) => plan,
        Err(error) => return invalid_geometry(error),
    };
    eredu_checkpoint::validation::validate_safetensors_plan(store, &plan)
}

fn validate_neutral_gpt_oss_safetensors(
    config: &Value,
    store: &SafetensorsWeightStore,
) -> StructuralValidation {
    let args = match eredu_architectures::gpt_oss::model_args_from_config_value(config) {
        Ok(args) => args,
        Err(error) => return invalid_geometry(error.to_string()),
    };
    let plan = match eredu_architectures::gpt_oss::safetensors_plan(&args) {
        Ok(plan) => plan,
        Err(error) => return invalid_geometry(error),
    };
    eredu_checkpoint::validation::validate_safetensors_plan(store, &plan)
}

fn validate_neutral_inkling_safetensors(
    config: &Value,
    store: &SafetensorsWeightStore,
) -> StructuralValidation {
    let bytes = match serde_json::to_vec(config) {
        Ok(bytes) => bytes,
        Err(error) => return invalid_geometry(error.to_string()),
    };
    let args = match eredu_architectures::inkling::ModelArgs::from_hf_json(&bytes) {
        Ok(args) => args,
        Err(error) => return invalid_geometry(error.to_string()),
    };
    let plan = match eredu_architectures::inkling::safetensors_plan(&args) {
        Ok(plan) => plan,
        Err(error) => return invalid_geometry(error),
    };
    eredu_checkpoint::validation::validate_safetensors_plan(store, &plan)
}

fn validate_neutral_muse_glimmer_safetensors(
    config: &Value,
    store: &SafetensorsWeightStore,
) -> StructuralValidation {
    let bytes = match serde_json::to_vec(config) {
        Ok(bytes) => bytes,
        Err(error) => return invalid_geometry(error.to_string()),
    };
    let args = match eredu_architectures::muse_glimmer::DecoderConfig::from_hf_json(&bytes) {
        Ok(args) => args,
        Err(error) => return invalid_geometry(error.to_string()),
    };
    let plan = match eredu_architectures::muse_glimmer::safetensors_plan(&args) {
        Ok(plan) => plan,
        Err(error) => return invalid_geometry(error),
    };
    eredu_checkpoint::validation::validate_safetensors_plan(store, &plan)
}

fn validate_neutral_gemma4_gguf(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
) -> StructuralValidation {
    let names = checkpoint
        .catalog()
        .tensors()
        .flat_map(|tensor| tensor.outputs())
        .map(|output| output.name.clone())
        .collect::<std::collections::BTreeSet<_>>();
    let text = match eredu_architectures::gemma4::ModelArgs::from_gguf_metadata(&names, metadata) {
        Ok(args) => args,
        Err(error) => return invalid_geometry(error.to_string()),
    };
    let family = match eredu_architectures::gemma4::family_from_gguf_metadata(text, metadata, None)
    {
        Ok(args) => args,
        Err(error) => return invalid_geometry(error.to_string()),
    };
    let plan = match eredu_architectures::gemma4::gguf_plan(&family.text) {
        Ok(plan) => plan,
        Err(error) => return invalid_geometry(error),
    };
    eredu_checkpoint::validation::validate_gguf_plan(checkpoint, &plan)
}

fn validate_neutral_gpt_oss_gguf(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
) -> StructuralValidation {
    if let Err(error) = checkpoint
        .catalog()
        .translated_outputs(eredu_architectures::gpt_oss::translate_gguf_weight_name)
    {
        return StructuralValidation::Invalid(vec![StructuralIssue {
            kind: StructuralIssueKind::ConflictingLayout,
            detail: error.to_string(),
            tensor_name: None,
            tensor_type_code: None,
            metadata_key: None,
        }]);
    }
    let args = match eredu_architectures::gpt_oss::model_args_from_gguf_catalog(metadata) {
        Ok(args) => args,
        Err(error) => return invalid_geometry(error.to_string()),
    };
    eredu_architectures::gpt_oss::validate_gguf(checkpoint, &args)
}

fn validate_neutral_inkling_gguf(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
) -> StructuralValidation {
    let args = match eredu_architectures::inkling::ModelArgs::from_gguf_metadata(metadata) {
        Ok(args) => args,
        Err(error) => return invalid_geometry(error.to_string()),
    };
    let plan = match eredu_architectures::inkling::gguf_plan(&args) {
        Ok(plan) => plan,
        Err(error) => return invalid_geometry(error),
    };
    eredu_checkpoint::validation::validate_gguf_plan(checkpoint, &plan)
}

fn validate_neutral_muse_glimmer_gguf(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
) -> StructuralValidation {
    let args = match eredu_architectures::muse_glimmer::DecoderConfig::from_gguf_metadata(
        metadata,
        checkpoint.contains_gguf_tensor("output.weight"),
    ) {
        Ok(args) => args,
        Err(error) => return invalid_geometry(error.to_string()),
    };
    let plan = match eredu_architectures::muse_glimmer::gguf_plan(&args) {
        Ok(plan) => plan,
        Err(error) => return invalid_geometry(error),
    };
    eredu_checkpoint::validation::validate_gguf_plan(checkpoint, &plan)
}

fn validate_neutral_qwen_safetensors(
    config: &Value,
    store: &SafetensorsWeightStore,
) -> StructuralValidation {
    let args = match eredu_architectures::qwen::model_args_from_config_value(config) {
        Ok(args) => args,
        Err(error) => return invalid_geometry(error.to_string()),
    };
    let plan = match eredu_architectures::qwen::safetensors_plan(&args) {
        Ok(plan) => plan,
        Err(error) => return invalid_geometry(error),
    };
    eredu_checkpoint::validation::validate_safetensors_plan(store, &plan)
}

fn validate_neutral_lfm2_safetensors(
    config: &Value,
    store: &SafetensorsWeightStore,
) -> StructuralValidation {
    let args = match eredu_architectures::lfm2::model_args_from_config_value(config) {
        Ok(args) => args,
        Err(error) => return invalid_geometry(error.to_string()),
    };
    let plan = match eredu_architectures::lfm2::safetensors_plan(&args, true) {
        Ok(plan) => plan,
        Err(error) => return invalid_geometry(error),
    };
    eredu_checkpoint::validation::validate_safetensors_plan(store, &plan)
}

fn validate_neutral_nemotron_safetensors(
    config: &Value,
    store: &SafetensorsWeightStore,
) -> StructuralValidation {
    let args = match eredu_architectures::nemotron_h::model_args_from_config_value(config) {
        Ok(args) => args,
        Err(error) => return invalid_geometry(error.to_string()),
    };
    let plan = match eredu_architectures::nemotron_h::safetensors_plan(&args) {
        Ok(plan) => plan,
        Err(error) => return invalid_geometry(error),
    };
    eredu_checkpoint::validation::validate_safetensors_plan(store, &plan)
}

fn validate_neutral_kimi_safetensors(
    config: &Value,
    store: &SafetensorsWeightStore,
) -> StructuralValidation {
    let args = match eredu_architectures::kimi_linear::model_args_from_config_value(config) {
        Ok(args) => args,
        Err(error) => return invalid_geometry(error.to_string()),
    };
    let plan = match eredu_architectures::kimi_linear::safetensors_plan(&args) {
        Ok(plan) => plan,
        Err(error) => return invalid_geometry(error),
    };
    eredu_checkpoint::validation::validate_safetensors_plan(store, &plan)
}

struct NeutralKimiGgufCatalog<'a>(&'a GgufCheckpoint);

impl eredu_architectures::kimi_linear::GgufTensorCatalog for NeutralKimiGgufCatalog<'_> {
    fn contains(&self, name: &str) -> bool {
        self.0.contains_gguf_tensor(name)
    }

    fn any(&self, predicate: impl FnMut(&str) -> bool) -> bool {
        self.0.any_gguf_tensor(predicate)
    }
}

fn validate_neutral_kimi_gguf(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
) -> StructuralValidation {
    let args = match eredu_architectures::kimi_linear::model_args_from_gguf_catalog(
        &NeutralKimiGgufCatalog(checkpoint),
        metadata,
    ) {
        Ok(args) => args,
        Err(error) => return invalid_geometry(error.to_string()),
    };
    if let Err(error) = checkpoint
        .catalog()
        .translated_outputs(eredu_architectures::kimi_linear::translate_gguf_weight_name)
    {
        return invalid_geometry(error.to_string());
    }
    let plan = match eredu_architectures::kimi_linear::gguf_plan(&args) {
        Ok(plan) => plan,
        Err(error) => return invalid_geometry(error),
    };
    eredu_checkpoint::validation::validate_gguf_plan(checkpoint, &plan)
}

struct NeutralLfm2GgufCatalog<'a>(&'a GgufCheckpoint);

impl eredu_architectures::lfm2::GgufTensorCatalog for NeutralLfm2GgufCatalog<'_> {
    fn contains(&self, name: &str) -> bool {
        self.0.contains_gguf_tensor(name)
    }

    fn any(&self, predicate: impl FnMut(&str) -> bool) -> bool {
        self.0.any_gguf_tensor(predicate)
    }
}

fn validate_neutral_lfm2_gguf(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
) -> StructuralValidation {
    let args = match eredu_architectures::lfm2::model_args_from_gguf_catalog(
        &NeutralLfm2GgufCatalog(checkpoint),
        metadata,
    ) {
        Ok(args) => args,
        Err(error) => return invalid_geometry(error.to_string()),
    };
    let is_moe = args.model_type == "lfm2_moe";
    if let Err(error) = checkpoint.catalog().translated_outputs(|name| {
        eredu_architectures::lfm2::translate_gguf_weight_name(name, is_moe)
    }) {
        return invalid_geometry(error.to_string());
    }
    let plan = match eredu_architectures::lfm2::gguf_plan(&args) {
        Ok(plan) => plan,
        Err(error) => return invalid_geometry(error),
    };
    eredu_checkpoint::validation::validate_gguf_plan(checkpoint, &plan)
}

struct NeutralNemotronGgufCatalog<'a>(&'a GgufCheckpoint);

impl eredu_architectures::nemotron_h::GgufTensorCatalog for NeutralNemotronGgufCatalog<'_> {
    fn contains(&self, name: &str) -> bool {
        self.0.contains_gguf_tensor(name)
    }

    fn any(&self, predicate: impl FnMut(&str) -> bool) -> bool {
        self.0.any_gguf_tensor(predicate)
    }
}

fn validate_neutral_nemotron_gguf(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
) -> StructuralValidation {
    let args = match eredu_architectures::nemotron_h::model_args_from_gguf_catalog(
        &NeutralNemotronGgufCatalog(checkpoint),
        metadata,
    ) {
        Ok(args) => args,
        Err(error) => return invalid_geometry(error.to_string()),
    };
    if let Err(error) = checkpoint
        .catalog()
        .translated_outputs(eredu_architectures::nemotron_h::translate_gguf_weight_name)
    {
        return invalid_geometry(error.to_string());
    }
    let plan = match eredu_architectures::nemotron_h::gguf_plan(&args) {
        Ok(plan) => plan,
        Err(error) => return invalid_geometry(error),
    };
    eredu_checkpoint::validation::validate_gguf_plan(checkpoint, &plan)
}

fn validate_neutral_deepseek_v3_safetensors(
    config: &Value,
    store: &SafetensorsWeightStore,
) -> StructuralValidation {
    let args = match eredu_architectures::deepseek::parse_v3_config(config) {
        Ok(args) => args,
        Err(error) => return invalid_geometry(error.to_string()),
    };
    let plan = match eredu_architectures::deepseek::v3_safetensors_plan(&args, true) {
        Ok(plan) => plan,
        Err(error) => return invalid_geometry(error),
    };
    eredu_checkpoint::validation::validate_safetensors_plan(store, &plan)
}

fn validate_neutral_deepseek_v4_safetensors(
    config: &Value,
    store: &SafetensorsWeightStore,
) -> StructuralValidation {
    let args = match eredu_architectures::deepseek::parse_v4_config(config) {
        Ok(args) => args,
        Err(error) => return invalid_geometry(error.to_string()),
    };
    let plan = match eredu_architectures::deepseek::v4_safetensors_plan(&args) {
        Ok(plan) => plan,
        Err(error) => return invalid_geometry(error),
    };
    eredu_checkpoint::validation::validate_safetensors_plan(store, &plan)
}

struct NeutralDeepSeekGgufCatalog<'a>(&'a GgufCheckpoint);

impl eredu_architectures::deepseek::GgufTensorCatalog for NeutralDeepSeekGgufCatalog<'_> {
    fn contains(&self, name: &str) -> bool {
        self.0.contains_gguf_tensor(name)
    }
}

fn validate_neutral_deepseek_v3_gguf(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
) -> StructuralValidation {
    let args = match eredu_architectures::deepseek::parse_v3_gguf(
        &NeutralDeepSeekGgufCatalog(checkpoint),
        metadata,
    ) {
        Ok(args) => args,
        Err(error) => return invalid_geometry(error.to_string()),
    };
    if let Err(error) = checkpoint
        .catalog()
        .translated_outputs(eredu_architectures::deepseek::translate_v3_gguf_weight_name)
    {
        return invalid_geometry(error.to_string());
    }
    let plan = match eredu_architectures::deepseek::v3_gguf_plan(&args) {
        Ok(plan) => plan,
        Err(error) => return invalid_geometry(error),
    };
    eredu_checkpoint::validation::validate_gguf_plan(checkpoint, &plan)
}

fn validate_neutral_deepseek_v4_gguf(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
) -> StructuralValidation {
    let args = match eredu_architectures::deepseek::parse_v4_gguf(metadata) {
        Ok(args) => args,
        Err(error) => return invalid_geometry(error.to_string()),
    };
    if let Err(error) = checkpoint
        .catalog()
        .translated_outputs(eredu_architectures::deepseek::translate_v4_gguf_weight_name)
    {
        return invalid_geometry(error.to_string());
    }
    let plan = match eredu_architectures::deepseek::v4_gguf_plan(&args) {
        Ok(plan) => plan,
        Err(error) => return invalid_geometry(error),
    };
    eredu_checkpoint::validation::validate_gguf_plan(checkpoint, &plan)
}

struct NeutralQwenGgufCatalog<'a>(&'a GgufCheckpoint);

impl eredu_architectures::qwen::GgufTensorCatalog for NeutralQwenGgufCatalog<'_> {
    fn contains(&self, name: &str) -> bool {
        self.0.contains_gguf_tensor(name)
    }
}

fn validate_neutral_qwen_gguf(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
) -> StructuralValidation {
    let args = match eredu_architectures::qwen::model_args_from_gguf_catalog(
        &NeutralQwenGgufCatalog(checkpoint),
        metadata,
    ) {
        Ok(args) => args,
        Err(error) => return invalid_geometry(error.to_string()),
    };
    if let Err(error) = checkpoint.catalog().translated_outputs(|name| {
        eredu_architectures::qwen::translate_gguf_weight_name(name, args.is_moe())
    }) {
        return invalid_geometry(error.to_string());
    }
    let plan = match eredu_architectures::qwen::gguf_plan(&args) {
        Ok(plan) => plan,
        Err(error) => return invalid_geometry(error),
    };
    eredu_checkpoint::validation::validate_gguf_plan(checkpoint, &plan)
}

fn validate_neutral_qwen_hybrid_safetensors(
    config: &Value,
    store: &SafetensorsWeightStore,
) -> StructuralValidation {
    let parsed = match eredu_architectures::qwen::hybrid::model_args_from_config_value(config) {
        Ok(parsed) => parsed,
        Err(error) => return invalid_geometry(error.to_string()),
    };
    let plan = match eredu_architectures::qwen::hybrid::safetensors_plan(&parsed.text) {
        Ok(plan) => plan,
        Err(error) => return invalid_geometry(error),
    };
    eredu_checkpoint::validation::validate_safetensors_plan(store, &plan)
}

fn validate_neutral_qwen_vl_safetensors(
    kind: ModelKind,
    config: &Value,
    store: &SafetensorsWeightStore,
) -> StructuralValidation {
    let args = match eredu_architectures::qwen::vl::model_args_from_config_value(config) {
        Ok(args) => args,
        Err(error) => return invalid_geometry(error.to_string()),
    };
    if args.text.is_moe() != (kind == ModelKind::Qwen3VlMoe) {
        return invalid_geometry(format!(
            "Qwen3-VL dispatch selected {}, but the nested text configuration is {}",
            if kind == ModelKind::Qwen3VlMoe {
                "MoE"
            } else {
                "dense"
            },
            if args.text.is_moe() { "MoE" } else { "dense" }
        ));
    }
    let plan = match eredu_architectures::qwen::vl::safetensors_plan(&args) {
        Ok(plan) => plan,
        Err(error) => return invalid_geometry(error),
    };
    eredu_checkpoint::validation::validate_safetensors_plan(store, &plan)
}

fn validate_neutral_qwen_hybrid_gguf(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
) -> StructuralValidation {
    let parsed = match eredu_architectures::qwen::hybrid::model_args_from_gguf_catalog(
        &NeutralQwenGgufCatalog(checkpoint),
        metadata,
    ) {
        Ok(parsed) => parsed,
        Err(error) => return invalid_geometry(error.to_string()),
    };
    if let Err(error) = checkpoint
        .catalog()
        .translated_outputs(eredu_architectures::qwen::hybrid::translate_gguf_weight_name)
    {
        return invalid_geometry(error.to_string());
    }
    let plan = match eredu_architectures::qwen::hybrid::gguf_plan(&parsed.text) {
        Ok(plan) => plan,
        Err(error) => return invalid_geometry(error),
    };
    eredu_checkpoint::validation::validate_gguf_plan(checkpoint, &plan)
}

fn validate_neutral_qwen_vl_gguf(
    architecture: GgufArchitecture,
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
) -> StructuralValidation {
    let is_moe = architecture == GgufArchitecture::Qwen3VlMoe;
    let context = if is_moe {
        eredu_architectures::qwen::TextConfigContext::Qwen3VlMoe
    } else {
        eredu_architectures::qwen::TextConfigContext::Qwen3Vl
    };
    let args = match eredu_architectures::qwen::model_args_from_gguf_catalog_with_context(
        &NeutralQwenGgufCatalog(checkpoint),
        metadata,
        context,
    ) {
        Ok(args) => args,
        Err(error) => return invalid_geometry(error.to_string()),
    };
    if args.is_moe() != is_moe {
        return invalid_geometry("Qwen3-VL GGUF architecture and expert geometry disagree".into());
    }
    if let Err(error) = checkpoint.catalog().translated_outputs(|name| {
        eredu_architectures::qwen::translate_gguf_weight_name(name, is_moe)
    }) {
        return invalid_geometry(error.to_string());
    }
    let plan = match eredu_architectures::qwen::gguf_plan(&args) {
        Ok(plan) => plan,
        Err(error) => return invalid_geometry(error),
    };
    eredu_checkpoint::validation::validate_gguf_plan(checkpoint, &plan)
}

pub(crate) fn validate_inkling_mmproj_gguf(
    model_checkpoint: &GgufCheckpoint,
    model_metadata: &HashMap<String, GgufMetadataValue>,
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
) -> StructuralValidation {
    let mut args = match eredu_architectures::inkling::ModelArgs::from_gguf_metadata(model_metadata)
    {
        Ok(args) => args,
        Err(error) => return invalid_geometry(error.to_string()),
    };
    let formats = match crate::backend::mlx::runtime::checkpoint::load::gguf_quantization_configs(
        checkpoint,
        eredu_architectures::inkling::translate_mmproj_weight_name,
    ) {
        Ok(formats) => formats,
        Err(error) => return invalid_geometry(error.to_string()),
    };
    let audio = formats
        .iter()
        .filter(|(name, _)| name.starts_with("audio."))
        .map(|(name, value)| (name.clone(), *value))
        .collect();
    let vision = formats
        .into_iter()
        .filter(|(name, _)| name.starts_with("visual."))
        .collect();
    args = match args.with_gguf_projector_metadata(model_metadata, metadata, audio, vision) {
        Ok(args) => args,
        Err(error) => return invalid_geometry(error.to_string()),
    };
    let primary = validate_neutral_inkling_gguf(model_checkpoint, model_metadata);
    if !matches!(primary, StructuralValidation::Exact) {
        return primary;
    }
    let plan = match eredu_architectures::inkling::mmproj_gguf_plan(&args) {
        Ok(plan) => plan,
        Err(error) => return invalid_geometry(error),
    };
    eredu_checkpoint::validation::validate_gguf_plan(checkpoint, &plan)
}

pub(crate) fn validate_gemma4_mmproj_gguf(
    model_checkpoint: &GgufCheckpoint,
    model_metadata: &HashMap<String, GgufMetadataValue>,
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
) -> StructuralValidation {
    let names = model_checkpoint
        .catalog()
        .tensors()
        .flat_map(|tensor| tensor.outputs())
        .map(|output| output.name.clone())
        .collect::<std::collections::BTreeSet<_>>();
    let text =
        match eredu_architectures::gemma4::ModelArgs::from_gguf_metadata(&names, model_metadata) {
            Ok(args) => args,
            Err(error) => return invalid_geometry(error.to_string()),
        };
    let family = match eredu_architectures::gemma4::family_from_gguf_metadata(
        text,
        model_metadata,
        Some(metadata),
    ) {
        Ok(args) => args,
        Err(error) => return invalid_geometry(error.to_string()),
    };
    let primary = validate_neutral_gemma4_gguf(model_checkpoint, model_metadata);
    if !matches!(primary, StructuralValidation::Exact) {
        return primary;
    }
    let plan = match eredu_architectures::gemma4::mmproj_gguf_plan(&family) {
        Ok(plan) => plan,
        Err(error) => return invalid_geometry(error),
    };
    eredu_checkpoint::validation::validate_gguf_plan(checkpoint, &plan)
}

pub(crate) fn validate_muse_glimmer_projector_gguf(
    model_checkpoint: &GgufCheckpoint,
    model_metadata: &HashMap<String, GgufMetadataValue>,
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
) -> StructuralValidation {
    let formats = match crate::backend::mlx::runtime::checkpoint::load::gguf_quantization_configs(
        checkpoint,
        eredu_architectures::muse_glimmer::translate_projector_gguf_name,
    ) {
        Ok(formats) => formats,
        Err(error) => return invalid_geometry(error.to_string()),
    };
    let args = match eredu_architectures::muse_glimmer::DecoderConfig::from_gguf_metadata(
        model_metadata,
        model_checkpoint.contains_gguf_tensor("output.weight"),
    )
    .and_then(|args| args.with_gguf_projector_metadata(metadata, formats))
    {
        Ok(args) => args,
        Err(error) => return invalid_geometry(error.to_string()),
    };
    let primary = validate_neutral_muse_glimmer_gguf(model_checkpoint, model_metadata);
    if !matches!(primary, StructuralValidation::Exact) {
        return primary;
    }
    let plan = match eredu_architectures::muse_glimmer::projector_gguf_plan(&args) {
        Ok(plan) => plan,
        Err(error) => return invalid_geometry(error),
    };
    eredu_checkpoint::validation::validate_gguf_plan(checkpoint, &plan)
}

pub(crate) fn validate_qwen3_vl_projector_gguf(
    model_checkpoint: &GgufCheckpoint,
    model_metadata: &HashMap<String, GgufMetadataValue>,
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
) -> StructuralValidation {
    let context = match model_metadata.get("general.architecture") {
        Some(GgufMetadataValue::String(value)) if value == "qwen3vl" => {
            eredu_architectures::qwen::TextConfigContext::Qwen3Vl
        }
        Some(GgufMetadataValue::String(value)) if value == "qwen3vlmoe" => {
            eredu_architectures::qwen::TextConfigContext::Qwen3VlMoe
        }
        other => {
            return invalid_geometry(format!(
                "Qwen3-VL projector requires qwen3vl text metadata, got {other:?}"
            ))
        }
    };
    let text = match eredu_architectures::qwen::model_args_from_gguf_catalog_with_context(
        &NeutralQwenGgufCatalog(model_checkpoint),
        model_metadata,
        context,
    ) {
        Ok(text) => text,
        Err(error) => return invalid_geometry(error.to_string()),
    };
    validate_neutral_qwen_projector(checkpoint, metadata, text.hidden_size, true)
}

pub(crate) fn validate_qwen35_projector_gguf(
    model_checkpoint: &GgufCheckpoint,
    model_metadata: &HashMap<String, GgufMetadataValue>,
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
) -> StructuralValidation {
    let text = match eredu_architectures::qwen::hybrid::model_args_from_gguf_catalog(
        &NeutralQwenGgufCatalog(model_checkpoint),
        model_metadata,
    ) {
        Ok(parsed) => parsed.text,
        Err(error) => return invalid_geometry(error.to_string()),
    };
    validate_neutral_qwen_projector(checkpoint, metadata, text.hidden_size, false)
}

struct NeutralQwenVisionGgufCatalog<'a>(&'a GgufCheckpoint);

impl eredu_architectures::qwen::vision::VisionGgufCatalog for NeutralQwenVisionGgufCatalog<'_> {
    fn shape(&self, name: &str) -> Option<Vec<usize>> {
        self.0
            .catalog()
            .tensors()
            .find(|tensor| tensor.descriptor().name == name)
            .map(|tensor| tensor.descriptor().mlx_shape())
            .and_then(|shape| {
                shape
                    .into_iter()
                    .map(usize::try_from)
                    .collect::<Result<Vec<_>, _>>()
                    .ok()
            })
    }
}

fn validate_neutral_qwen_projector(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
    text_hidden: i32,
    allow_deepstack: bool,
) -> StructuralValidation {
    let vision = match eredu_architectures::qwen::vision::config_from_gguf_catalog(
        &NeutralQwenVisionGgufCatalog(checkpoint),
        metadata,
    ) {
        Ok(vision) => vision,
        Err(error) => return invalid_geometry(error.to_string()),
    };
    if vision.out_hidden_size != text_hidden {
        return invalid_geometry(format!(
            "Qwen projector output {} does not match language hidden size {text_hidden}",
            vision.out_hidden_size
        ));
    }
    if !allow_deepstack && vision.deepstack_layer_count() != 0 {
        return invalid_geometry(format!(
            "Qwen3.5 projector declares {} unsupported DeepStack outputs",
            vision.deepstack_layer_count()
        ));
    }
    let deepstack = vision.deepstack_layers();
    if let Err(error) = checkpoint.catalog().translated_outputs(|name| {
        eredu_architectures::qwen::vision::translate_gguf_weight_name(name, &deepstack)
    }) {
        return invalid_geometry(error.to_string());
    }
    let plan = match eredu_architectures::qwen::vision::gguf_plan(&vision, text_hidden) {
        Ok(plan) => plan,
        Err(error) => return invalid_geometry(error),
    };
    eredu_checkpoint::validation::validate_gguf_plan(checkpoint, &plan)
}
fn invalid_geometry(detail: String) -> StructuralValidation {
    StructuralValidation::Invalid(vec![StructuralIssue {
        kind: StructuralIssueKind::InvalidGeometry,
        detail,
        tensor_name: None,
        tensor_type_code: None,
        metadata_key: None,
    }])
}

#[cfg(test)]
mod admission_policy_tests {
    use super::*;

    #[test]
    fn non_strict_catalog_ignores_only_unexpected_tensors() {
        let unexpected = StructuralIssue {
            kind: StructuralIssueKind::UnexpectedTensor,
            detail: "test catalog contains unexpected tensor \"unrelated.weight\"".into(),
            tensor_name: Some("unrelated.weight".into()),
            tensor_type_code: None,
            metadata_key: None,
        };
        let malformed = eredu_checkpoint::validation::shape_mismatch("model.weight", &[2, 2], &[1]);
        assert_eq!(
            StructuralValidation::Invalid(vec![unexpected.clone(), malformed.clone()])
                .with_strict_catalog(false),
            StructuralValidation::Invalid(vec![malformed])
        );

        let error = StructuralValidation::Invalid(vec![unexpected])
            .into_loader_result()
            .map_err(Error::from)
            .unwrap_err();
        assert!(matches!(
            error,
            Error::StrictLoadValidation { missing, unused }
                if missing.is_empty() && unused == ["unrelated.weight"]
        ));
    }
}

#[cfg(test)]
mod neutral_qwen_tests {
    use std::collections::BTreeSet;

    fn qwen2_args(tied: bool) -> eredu_architectures::qwen::ModelArgs {
        eredu_architectures::qwen::model_args_from_config_value(&serde_json::json!({
            "model_type": "qwen2", "hidden_size": 8, "num_hidden_layers": 2,
            "intermediate_size": 16, "num_attention_heads": 4,
            "num_key_value_heads": 2, "rms_norm_eps": 1e-6, "vocab_size": 32,
            "max_position_embeddings": 64, "rope_theta": 10000.0,
            "tie_word_embeddings": tied, "use_sliding_window": false
        }))
        .unwrap()
    }

    #[test]
    fn qwen2_plan_is_exactly_biased_and_has_no_qk_norms() {
        let tied = eredu_architectures::qwen::safetensors_plan(&qwen2_args(true)).unwrap();
        let names = tied
            .common_tensors
            .iter()
            .map(|tensor| tensor.key.as_str())
            .collect::<BTreeSet<_>>();
        assert!(names.contains("model.layers.0.self_attn.q_proj.bias"));
        assert!(names.contains("model.layers.0.self_attn.k_proj.bias"));
        assert!(names.contains("model.layers.0.self_attn.v_proj.bias"));
        assert!(!names.contains("model.layers.0.self_attn.q_norm.weight"));
        assert!(!names.contains("model.layers.0.self_attn.k_norm.weight"));
        assert!(!names.contains("lm_head.weight"));
        let redundant_head = tied
            .layout_groups
            .iter()
            .find(|group| group.id == "redundant tied output head")
            .expect("optional redundant tied head layout");
        assert!(!redundant_head.required);
        assert_eq!(redundant_head.variants[0].tensors[0].key, "lm_head.weight");
        let untied = eredu_architectures::qwen::safetensors_plan(&qwen2_args(false)).unwrap();
        assert!(untied
            .common_tensors
            .iter()
            .any(|tensor| tensor.key == "lm_head.weight"));
    }
}
