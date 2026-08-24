//! MLX architecture binding against portable checkpoint catalogs.

use std::collections::HashMap;

use safemlx::ops::{GgufCheckpoint, GgufMetadataValue};
use serde_json::Value;

use eredu_architectures::{GgufArchitecture, ModelKind};

use super::ModelLoadOptions;
use crate::backend::error::Error;
use crate::backend::runtime::checkpoint::load::GgufTensorNames;
use crate::composition::llama::checkpoint as llama_checkpoint;
use eredu_checkpoint::store::SafetensorsWeightStore;

pub use eredu_checkpoint::validation::{
    CheckpointIssue as StructuralIssue, CheckpointIssueKind as StructuralIssueKind,
    CheckpointValidation as StructuralValidation,
};

/// Native GGUF source admitted by composition-level structural validation.
///
/// Family composition consumes this type instead of reparsing architecture
/// metadata or reaching upward from reusable backend runtime code.
pub(crate) struct AdmittedGguf {
    architecture: GgufArchitecture,
    checkpoint: GgufCheckpoint,
    metadata: HashMap<String, GgufMetadataValue>,
}

impl AdmittedGguf {
    pub(crate) const fn architecture(&self) -> GgufArchitecture {
        self.architecture
    }

    pub(crate) fn checkpoint(&self) -> &GgufCheckpoint {
        &self.checkpoint
    }

    pub(crate) fn metadata(&self) -> &HashMap<String, GgufMetadataValue> {
        &self.metadata
    }
}

pub(crate) fn admit_gguf(
    architecture: GgufArchitecture,
    checkpoint: GgufCheckpoint,
    metadata: HashMap<String, GgufMetadataValue>,
    options: ModelLoadOptions,
) -> Result<AdmittedGguf, Error> {
    validate_gguf(architecture, &checkpoint, &metadata, options).into_loader_result()?;
    Ok(AdmittedGguf {
        architecture,
        checkpoint,
        metadata,
    })
}

const fn mlx_supports_expert_cache(kind: ModelKind) -> bool {
    matches!(
        kind,
        ModelKind::KimiLinear
            | ModelKind::DeepSeekV3
            | ModelKind::DeepSeekV4
            | ModelKind::Gemma4
            | ModelKind::GptOss
            | ModelKind::Inkling
            | ModelKind::Lfm2
            | ModelKind::MuseGlimmer
            | ModelKind::NemotronH
            | ModelKind::Qwen3
            | ModelKind::Qwen3Next
            | ModelKind::Qwen3VlMoe
            | ModelKind::Qwen35
    )
}

const fn mlx_supports_nonresident_safetensors_quantization(kind: ModelKind) -> bool {
    matches!(
        kind,
        ModelKind::DeepSeekV3
            | ModelKind::DeepSeekV4
            | ModelKind::Gemma4
            | ModelKind::GptOss
            | ModelKind::Inkling
            | ModelKind::KimiLinear
            | ModelKind::Lfm2
            | ModelKind::Llama
            | ModelKind::MuseGlimmer
            | ModelKind::NemotronH
            | ModelKind::Qwen2
            | ModelKind::Qwen3
            | ModelKind::Qwen3Next
            | ModelKind::Qwen3Vl
            | ModelKind::Qwen3VlMoe
            | ModelKind::Qwen35
    )
}

fn validate_expert_cache_capability(
    kind: ModelKind,
    capabilities: eredu_architectures::preparation::ArchitectureCapabilities,
) -> Result<(), Error> {
    if capabilities.independently_addressable_experts() && mlx_supports_expert_cache(kind) {
        return Ok(());
    }
    Err(Error::Artifact(
        eredu_core::artifact::ArtifactError::UnsupportedResidencyPolicy(format!(
            "independent expert caching is unavailable for the normalized {} architecture on MLX",
            kind.canonical_name()
        )),
    ))
}

fn validate_preparation_capability_intersection(
    kind: ModelKind,
    format: eredu_core::ArtifactFormat,
    policy: eredu_core::PreparationPolicy,
    capabilities: eredu_architectures::preparation::ArchitectureCapabilities,
) -> Result<(), Error> {
    if format == eredu_core::ArtifactFormat::SafeTensors
        && policy.quantization.is_some()
        && policy.residency != eredu_core::ResidencyRequest::FullyResident
        && !(capabilities.nonresident_safetensors_quantization()
            && mlx_supports_nonresident_safetensors_quantization(kind))
    {
        return Err(Error::Artifact(
            eredu_core::artifact::ArtifactError::UnsupportedQuantizationPolicy(format!(
                "load-time quantization is unavailable for the normalized {} architecture with nonresident weights on MLX",
                kind.canonical_name()
            )),
        ));
    }
    if policy.residency == eredu_core::ResidencyRequest::ExpertCache {
        validate_expert_cache_capability(kind, capabilities)?;
    }
    Ok(())
}

fn requires_architecture_capabilities(
    format: eredu_core::ArtifactFormat,
    policy: eredu_core::PreparationPolicy,
) -> bool {
    policy.residency == eredu_core::ResidencyRequest::ExpertCache
        || (format == eredu_core::ArtifactFormat::SafeTensors
            && policy.quantization.is_some()
            && policy.residency != eredu_core::ResidencyRequest::FullyResident)
}

pub(crate) fn validate_safetensors_preparation(
    kind: ModelKind,
    config: &Value,
    options: ModelLoadOptions,
) -> Result<(), Error> {
    let policy = options.preparation_policy()?;
    eredu_core::validate_preparation_policy(kind.loading_protocol(), policy)?;
    if !requires_architecture_capabilities(eredu_core::ArtifactFormat::SafeTensors, policy) {
        return Ok(());
    }
    let capabilities = eredu_architectures::preparation::safetensors_capabilities(kind, config)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    validate_preparation_capability_intersection(
        kind,
        eredu_core::ArtifactFormat::SafeTensors,
        policy,
        capabilities,
    )
}

pub(crate) fn validate_gguf_preparation(
    architecture: GgufArchitecture,
    checkpoint: &GgufCheckpoint,
    options: ModelLoadOptions,
) -> Result<(), Error> {
    let policy = options.preparation_policy()?;
    eredu_core::validate_preparation_policy(architecture.model_kind().loading_protocol(), policy)?;
    if !requires_architecture_capabilities(eredu_core::ArtifactFormat::Gguf, policy) {
        return Ok(());
    }
    let capabilities =
        eredu_architectures::preparation::gguf_capabilities(architecture, checkpoint)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    validate_preparation_capability_intersection(
        architecture.model_kind(),
        eredu_core::ArtifactFormat::Gguf,
        policy,
        capabilities,
    )
}

pub(crate) fn validate_inspected_preparation(
    inspection: &eredu_core::ArtifactInspection,
    policy: eredu_core::PreparationPolicy,
) -> Result<(), Error> {
    eredu_core::validate_preparation_policy(inspection.configuration().loading_protocol, policy)?;
    if !requires_architecture_capabilities(inspection.format(), policy) {
        return Ok(());
    }
    let configuration = inspection.configuration();
    let kind = ModelKind::resolve_family(&configuration.family)?;
    let capabilities = match inspection.format() {
        eredu_core::ArtifactFormat::SafeTensors => {
            eredu_architectures::preparation::safetensors_capabilities(
                kind,
                configuration.json.as_ref().ok_or_else(|| {
                    Error::UnsupportedArchitecture(
                        "SafeTensors inspection omitted normalized JSON configuration".into(),
                    )
                })?,
            )
        }
        eredu_core::ArtifactFormat::Gguf => eredu_architectures::preparation::gguf_capabilities(
            GgufArchitecture::resolve(&configuration.declared_model_type)?,
            inspection.gguf_checkpoint().ok_or_else(|| {
                Error::UnsupportedArchitecture(
                    "GGUF inspection omitted portable checkpoint metadata".into(),
                )
            })?,
        ),
    }
    .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    validate_preparation_capability_intersection(kind, inspection.format(), policy, capabilities)
}

fn validate_gguf_load_policy(
    architecture: GgufArchitecture,
    options: ModelLoadOptions,
) -> Result<(), Error> {
    let policy = options.preparation_policy()?;
    eredu_core::validate_preparation_policy(architecture.model_kind().loading_protocol(), policy)?;
    Ok(())
}

pub fn validate_safetensors(
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

pub fn validate_gguf(
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
            if let Err(error) = validate_gguf_load_policy(architecture, options) {
                invalid_geometry(error.to_string())
            } else {
                validate_neutral_gemma4_gguf(checkpoint, metadata)
            }
        }
        GgufArchitecture::Inkling => {
            if let Err(error) = validate_gguf_load_policy(architecture, options) {
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
            if let Err(error) = validate_gguf_load_policy(architecture, options) {
                invalid_geometry(error.to_string())
            } else {
                validate_neutral_nemotron_gguf(checkpoint, metadata)
            }
        }
        GgufArchitecture::Qwen2 | GgufArchitecture::Qwen3 | GgufArchitecture::Qwen3Moe => {
            validate_neutral_qwen_gguf(checkpoint, metadata)
        }
        architecture @ (GgufArchitecture::Qwen3Vl | GgufArchitecture::Qwen3VlMoe) => {
            if let Err(error) = validate_gguf_load_policy(architecture, options) {
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
    let is_moe = args.has_sparse_moe_layers();
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

pub fn validate_inkling_mmproj_gguf(
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
    let formats = match crate::backend::runtime::checkpoint::load::gguf_quantization_configs(
        checkpoint,
        eredu_architectures::inkling::translate_mmproj_weight_name,
    ) {
        Ok(formats) => formats,
        Err(error) => return invalid_geometry(error.to_string()),
    };
    args = match args.with_gguf_projector_metadata(model_metadata, metadata, formats) {
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

pub fn validate_gemma4_mmproj_gguf(
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

pub fn validate_muse_glimmer_projector_gguf(
    model_checkpoint: &GgufCheckpoint,
    model_metadata: &HashMap<String, GgufMetadataValue>,
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
) -> StructuralValidation {
    let formats = match crate::backend::runtime::checkpoint::load::gguf_quantization_configs(
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

pub fn validate_qwen3_vl_projector_gguf(
    architecture: GgufArchitecture,
    model_checkpoint: &GgufCheckpoint,
    model_metadata: &HashMap<String, GgufMetadataValue>,
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
) -> StructuralValidation {
    let context =
        match eredu_architectures::qwen::TextConfigContext::from_qwen3_vl_gguf_architecture(
            architecture,
        ) {
            Ok(context) => context,
            Err(error) => return invalid_geometry(error.to_string()),
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

pub fn validate_qwen35_projector_gguf(
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
            .map(|tensor| tensor.descriptor().row_major_shape())
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

    fn expert_cache_options() -> ModelLoadOptions {
        ModelLoadOptions::default().with_weight_residency(
            eredu_runtime::WeightResidency::with_expert_cache(
                eredu_runtime::NonExpertWeightResidency::FullyResident,
                eredu_runtime::ExpertCacheLoadOptions::default(),
            ),
        )
    }

    fn kimi_linear_config() -> Value {
        serde_json::json!({
            "model_type":"kimi_linear","vocab_size":16,"hidden_size":12,"num_hidden_layers":2,
            "num_attention_heads":3,"num_key_value_heads":3,"intermediate_size":17,"head_dim":4,
            "model_max_length":64,"linear_attn_config":{"kda_layers":[1],"full_attn_layers":[2],"num_heads":3,"head_dim":4,"short_conv_kernel_size":3},
            "num_experts":2,"moe_intermediate_size":9,"kv_lora_rank":6,"qk_nope_head_dim":4,"qk_rope_head_dim":2,"v_head_dim":4,
            "mla_use_nope":true,"num_experts_per_token":1,"num_shared_experts":1,"routed_scaling_factor":1.0,
            "first_k_dense_replace":1,"num_expert_group":1,"topk_group":1
        })
    }

    fn gemma4_config(routed: bool) -> Value {
        let mut config = serde_json::json!({
            "model_type":"gemma4", "tie_word_embeddings":true,
            "text_config":{
                "model_type":"gemma4_text", "hidden_size":32,
                "num_hidden_layers":1, "intermediate_size":64,
                "num_attention_heads":4, "num_key_value_heads":2, "head_dim":8,
                "rms_norm_eps":0.00001, "vocab_size":32,
                "max_position_embeddings":128, "layer_types":["full_attention"],
                "enable_moe_block":false
            }
        });
        if routed {
            config["text_config"]["enable_moe_block"] = true.into();
            config["text_config"]["num_experts"] = 2.into();
            config["text_config"]["top_k_experts"] = 1.into();
            config["text_config"]["moe_intermediate_size"] = 32.into();
        }
        config
    }

    fn muse_glimmer_config(routed: bool) -> Value {
        let mut config = serde_json::json!({
            "architectures":["MuseGlimmerForConditionalGeneration"],
            "model_type":"muse_glimmer", "image_token_id":22, "video_token_id":23,
            "out_hidden_size":32, "projector_hidden_size":16,
            "text_config":{
                "model_type":"muse_glimmer_text", "hidden_size":16,
                "num_hidden_layers":2, "intermediate_size":24,
                "num_attention_heads":4, "num_key_value_heads":2, "head_dim":4,
                "rms_norm_eps":0.00001, "post_norm_eps":0.00001,
                "vocab_size":24, "max_position_embeddings":64,
                "rope_theta":10000.0,
                "layer_types":["sliding_attention","full_attention"],
                "layer_rope_theta":[10000.0,0.0], "sliding_window":8,
                "tie_word_embeddings":false, "hidden_act":"silu",
                "attention_dropout":0.0, "qk_scale_factor":1.0,
                "output_multiplier":1.0, "final_logit_softcapping":30.0
            },
            "vision_config":{
                "model_type":"muse_glimmer_vision", "hidden_size":8,
                "intermediate_size":12, "num_attention_heads":2,
                "num_hidden_layers":1, "patch_size":2, "patch_temporal":1,
                "merge_size":2, "pos_emb_height":2, "pos_emb_width":2,
                "max_position_embeddings":4, "layer_norm_eps":0.00001,
                "hidden_act":"gelu", "layer_types":["full_attention"],
                "rope_parameters":{"rope_theta":10000.0,"rope_type":"default"}
            }
        });
        if routed {
            config["text_config"]["intermediate_size"] = 0.into();
            config["text_config"]["moe_intermediate_size"] = 12.into();
            config["text_config"]["num_experts"] = 8.into();
            config["text_config"]["num_experts_per_tok"] = 2.into();
            config["text_config"]["norm_topk_prob"] = true.into();
        }
        config
    }

    fn dense_qwen3_config() -> Value {
        serde_json::json!({
            "model_type":"qwen3", "hidden_size":16, "num_hidden_layers":3,
            "intermediate_size":32, "num_attention_heads":4,
            "num_key_value_heads":2, "head_dim":4, "rms_norm_eps":0.000001,
            "vocab_size":64, "max_position_embeddings":128,
            "rope_theta":1000000.0, "tie_word_embeddings":true
        })
    }

    #[test]
    fn expert_cache_admits_normalized_gemma4_and_muse_glimmer_moe() {
        let options = expert_cache_options();
        validate_safetensors_preparation(ModelKind::Gemma4, &gemma4_config(true), options).unwrap();
        validate_safetensors_preparation(
            ModelKind::MuseGlimmer,
            &muse_glimmer_config(true),
            options,
        )
        .unwrap();
    }

    #[test]
    fn nonresident_quantization_admits_kimi_linear_capability_intersection() {
        let options =
            ModelLoadOptions::with_quantization(eredu_checkpoint::WeightQuantization::MxFp4)
                .with_weight_residency(eredu_runtime::WeightResidency::layerwise_host(
                    eredu_runtime::LayerwiseLoadOptions::default(),
                ));
        validate_safetensors_preparation(ModelKind::KimiLinear, &kimi_linear_config(), options)
            .unwrap();
    }

    #[test]
    fn expert_cache_rejects_dense_variants_of_mixed_families() {
        let options = expert_cache_options();
        for (kind, config) in [
            (ModelKind::Gemma4, gemma4_config(false)),
            (ModelKind::MuseGlimmer, muse_glimmer_config(false)),
            (ModelKind::Qwen3, dense_qwen3_config()),
        ] {
            assert!(matches!(
                validate_safetensors_preparation(kind, &config, options),
                Err(Error::Artifact(
                    eredu_core::artifact::ArtifactError::UnsupportedResidencyPolicy(_)
                ))
            ));
        }
    }

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
