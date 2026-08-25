//! MLX architecture binding against portable checkpoint catalogs.

use std::collections::HashMap;

use safemlx::ops::{GgufCheckpoint, GgufMetadataValue};
#[cfg(test)]
use serde_json::Value;

use eredu_architectures::{GgufArchitecture, ModelKind};

use super::ModelLoadOptions;
use crate::backend::error::Error;
use crate::backend::runtime::checkpoint::load::GgufTensorNames;
use eredu_checkpoint::{recipe::RecipeCatalog, validation::SafetensorsCatalog};

pub use eredu_checkpoint::validation::{
    CheckpointIssue as StructuralIssue, CheckpointIssueKind as StructuralIssueKind,
    CheckpointValidation as StructuralValidation,
};

pub(crate) trait StructuralSafetensorsCatalog: SafetensorsCatalog + RecipeCatalog {}

impl<T> StructuralSafetensorsCatalog for T where T: SafetensorsCatalog + RecipeCatalog + ?Sized {}

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
    validate_gguf(architecture, &checkpoint, options).into_loader_result()?;
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

const fn mlx_supports_safetensors_quantization(kind: ModelKind) -> bool {
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

const fn mlx_supports_complete_gguf_quantization(kind: ModelKind) -> bool {
    matches!(
        kind,
        ModelKind::GptOss
            | ModelKind::KimiLinear
            | ModelKind::Lfm2
            | ModelKind::Llama
            | ModelKind::NemotronH
            | ModelKind::Qwen2
            | ModelKind::Qwen3
            | ModelKind::Qwen3Next
            | ModelKind::Qwen3Vl
            | ModelKind::Qwen3VlMoe
            | ModelKind::Qwen35
    )
}

pub(crate) fn validate_complete_gguf_quantization(
    kind: ModelKind,
    requested: bool,
) -> Result<(), Error> {
    if !requested || mlx_supports_complete_gguf_quantization(kind) {
        return Ok(());
    }
    Err(Error::Artifact(
        eredu_core::artifact::ArtifactError::UnsupportedQuantizationPolicy(format!(
            "load-time quantization is unavailable for complete GGUF {} materialization on MLX",
            kind.canonical_name()
        )),
    ))
}

pub(crate) fn requires_distributed_stage_loader(
    kind: ModelKind,
    topology: eredu_core::ParallelTopology,
) -> bool {
    topology.is_axis_active(eredu_core::ParallelAxis::Pipeline)
        || topology.is_axis_active(eredu_core::ParallelAxis::Expert)
        || matches!(
            kind,
            ModelKind::DeepSeekV3
                | ModelKind::DeepSeekV4
                | ModelKind::Qwen3Next
                | ModelKind::Qwen35
                | ModelKind::Qwen3Vl
                | ModelKind::Qwen3VlMoe
        )
}

fn validate_quantization_capability(
    kind: ModelKind,
    format: eredu_core::ArtifactFormat,
    policy: eredu_core::PreparationPolicy,
    capabilities: eredu_architectures::preparation::ArchitectureCapabilities,
) -> Result<(), Error> {
    if policy.quantization.is_none() {
        return Ok(());
    }
    if let Some(topology) = policy.topology.filter(|topology| !topology.is_replicated()) {
        if requires_distributed_stage_loader(kind, topology) {
            return Ok(());
        }
        return Err(Error::Artifact(
            eredu_core::artifact::ArtifactError::UnsupportedQuantizationPolicy(format!(
                "load-time quantization is unavailable for complete tensor-parallel {} materialization on MLX",
                kind.canonical_name()
            )),
        ));
    }
    if format == eredu_core::ArtifactFormat::Gguf {
        return validate_complete_gguf_quantization(kind, true);
    }
    let supported = mlx_supports_safetensors_quantization(kind)
        && (policy.residency == eredu_core::ResidencyRequest::FullyResident
            || capabilities.nonresident_safetensors_quantization());
    if supported {
        return Ok(());
    }
    Err(Error::Artifact(
        eredu_core::artifact::ArtifactError::UnsupportedQuantizationPolicy(format!(
            "load-time quantization is unavailable for the normalized {} architecture from SafeTensors with {:?} weights on MLX",
            kind.canonical_name(),
            policy.residency,
        )),
    ))
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
    if let Some(topology) = policy.topology {
        validate_parallel_capabilities(
            capabilities,
            topology,
            match format {
                eredu_core::ArtifactFormat::SafeTensors => "SafeTensors",
                eredu_core::ArtifactFormat::Gguf => "GGUF",
            },
            kind.canonical_name(),
        )?;
    }
    validate_quantization_capability(kind, format, policy, capabilities)?;
    if policy.residency == eredu_core::ResidencyRequest::ExpertCache {
        validate_expert_cache_capability(kind, capabilities)?;
    }
    Ok(())
}

fn requires_architecture_capabilities(policy: eredu_core::PreparationPolicy) -> bool {
    policy
        .topology
        .is_some_and(|topology| !topology.is_replicated())
        || policy.residency == eredu_core::ResidencyRequest::ExpertCache
        || policy.quantization.is_some()
}

pub(crate) fn validate_parallel_capabilities(
    capabilities: eredu_architectures::preparation::ArchitectureCapabilities,
    topology: eredu_core::ParallelTopology,
    artifact: &str,
    architecture: &str,
) -> Result<(), Error> {
    let plan = capabilities.parallel_plan();
    let unsupported = |capability: &str| {
        Error::Parallel(format!(
            "{artifact} architecture {architecture:?} has no architecture-owned {capability} plan; no checkpoint payload was materialized"
        ))
    };
    if topology.is_axis_active(eredu_core::ParallelAxis::Pipeline) && !plan.pipeline_parallel() {
        return Err(unsupported("pipeline-parallel"));
    }
    if topology.is_axis_active(eredu_core::ParallelAxis::Tensor) && !plan.tensor_parallel() {
        return Err(unsupported("tensor-parallel"));
    }
    if topology.is_axis_active(eredu_core::ParallelAxis::Expert) && !plan.expert_parallel() {
        return Err(unsupported("expert-parallel"));
    }
    Ok(())
}

#[cfg(test)]
fn validate_safetensors_preparation_for_test(
    kind: ModelKind,
    config: &Value,
    options: ModelLoadOptions,
) -> Result<(), Error> {
    let policy = options.preparation_policy()?;
    eredu_core::validate_preparation_policy(kind.loading_protocol(), policy)?;
    if !requires_architecture_capabilities(policy) {
        return Ok(());
    }
    let capabilities = eredu_architectures::preparation::safetensors_capabilities(kind, config)
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
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
    if !requires_architecture_capabilities(policy) {
        return Ok(());
    }
    let capabilities =
        eredu_architectures::preparation::gguf_capabilities(architecture, checkpoint)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    validate_preparation_capability_intersection(
        architecture.model_kind(),
        eredu_core::ArtifactFormat::Gguf,
        policy,
        capabilities,
    )
}

pub(crate) fn validate_inspected_preparation(
    inspection: &eredu_core::ArtifactInspection<
        eredu_architectures::processor_plan::ArtifactArchitecturePlan,
    >,
    policy: eredu_core::PreparationPolicy,
) -> Result<(), Error> {
    eredu_core::validate_preparation_policy(inspection.configuration().loading_protocol, policy)?;
    if !requires_architecture_capabilities(policy) {
        return Ok(());
    }
    let configuration = inspection.configuration();
    let architecture_plan = inspection.architecture_plan();
    let kind = architecture_plan.model_kind();
    let capabilities = match inspection.format() {
        eredu_core::ArtifactFormat::SafeTensors => {
            eredu_architectures::preparation::safetensors_capabilities(
                kind,
                configuration.json.as_ref().ok_or_else(|| {
                    Error::Artifact(eredu_core::artifact::ArtifactError::InvalidArtifact(
                        "SafeTensors inspection omitted normalized JSON configuration".into(),
                    ))
                })?,
            )
        }
        eredu_core::ArtifactFormat::Gguf => eredu_architectures::preparation::gguf_capabilities(
            architecture_plan.gguf_architecture().ok_or_else(|| {
                Error::ArchitectureModel(
                    "GGUF preparation omitted its architecture-owned GGUF identity".into(),
                )
            })?,
            inspection.gguf_checkpoint().ok_or_else(|| {
                Error::Artifact(eredu_core::artifact::ArtifactError::InvalidArtifact(
                    "GGUF inspection omitted portable checkpoint metadata".into(),
                ))
            })?,
        ),
    }
    .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
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
    plan: &eredu_architectures::configuration::SafetensorsArchitecturePlan,
    store: &(impl StructuralSafetensorsCatalog + ?Sized),
    options: ModelLoadOptions,
) -> StructuralValidation {
    let validation =
        eredu_checkpoint::validation::validate_safetensors_plan(store, plan.checkpoint());
    let validation = if validation == StructuralValidation::Exact {
        match plan.model() {
            eredu_architectures::configuration::SafetensorsModelConfig::Moshi(config) => {
                match eredu_architectures::moshi::canonical_recipes(config, store) {
                    Ok(_) => StructuralValidation::Exact,
                    Err(error) => invalid_geometry(error),
                }
            }
            _ => validation,
        }
    } else {
        validation
    };
    validation.with_strict_catalog(options.weight_residency.strict_loading())
}

pub fn validate_gguf(
    architecture: GgufArchitecture,
    checkpoint: &GgufCheckpoint,
    options: ModelLoadOptions,
) -> StructuralValidation {
    if let Err(error) = validate_gguf_load_policy(architecture, options) {
        return invalid_geometry(error.to_string());
    }
    eredu_architectures::configuration::validate_gguf_checkpoint(architecture, checkpoint.catalog())
}

struct NeutralMuseGlimmerGgufCatalog<'a>(&'a GgufCheckpoint);

impl eredu_architectures::muse_glimmer::GgufTensorCatalog for NeutralMuseGlimmerGgufCatalog<'_> {
    fn contains(&self, name: &str) -> bool {
        self.0.contains_gguf_tensor(name)
    }
}

struct NeutralQwenGgufCatalog<'a>(&'a GgufCheckpoint);

impl eredu_architectures::qwen::GgufTensorCatalog for NeutralQwenGgufCatalog<'_> {
    fn contains(&self, name: &str) -> bool {
        self.0.contains_gguf_tensor(name)
    }
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
    let primary = eredu_architectures::configuration::validate_gguf_checkpoint(
        GgufArchitecture::Inkling,
        model_checkpoint.catalog(),
    );
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
    let primary = eredu_architectures::configuration::validate_gguf_checkpoint(
        GgufArchitecture::Gemma4,
        model_checkpoint.catalog(),
    );
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
    let args = match eredu_architectures::muse_glimmer::DecoderConfig::from_gguf_catalog(
        &NeutralMuseGlimmerGgufCatalog(model_checkpoint),
        model_metadata,
    )
    .and_then(|args| args.with_gguf_projector_metadata(metadata, formats))
    {
        Ok(args) => args,
        Err(error) => return invalid_geometry(error.to_string()),
    };
    let primary = eredu_architectures::configuration::validate_gguf_checkpoint(
        GgufArchitecture::MuseGlimmer,
        model_checkpoint.catalog(),
    );
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
    let vision = match eredu_architectures::qwen::vl::vision_config_from_gguf_catalog(
        &NeutralQwenVisionGgufCatalog(checkpoint),
        metadata,
    ) {
        Ok(vision) => vision,
        Err(error) => return invalid_geometry(error.to_string()),
    };
    let composite = match eredu_architectures::qwen::vl::model_args_from_gguf_parts(
        text,
        model_metadata,
        vision,
    ) {
        Ok(composite) => composite,
        Err(error) => return invalid_geometry(error.to_string()),
    };
    let plan = match eredu_architectures::qwen::vl::projector_gguf_plan(&composite) {
        Ok(plan) => plan,
        Err(error) => return invalid_geometry(error),
    };
    validate_neutral_qwen_projector(checkpoint, &composite.vision, &plan)
}

pub fn validate_qwen35_projector_gguf(
    model_checkpoint: &GgufCheckpoint,
    model_metadata: &HashMap<String, GgufMetadataValue>,
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
) -> StructuralValidation {
    let parsed = match eredu_architectures::qwen::hybrid::model_args_from_gguf_catalog(
        &NeutralQwenGgufCatalog(model_checkpoint),
        model_metadata,
    ) {
        Ok(parsed) => parsed,
        Err(error) => return invalid_geometry(error.to_string()),
    };
    let vision = match eredu_architectures::qwen::hybrid::vision_config_from_gguf_catalog(
        &NeutralQwenVisionGgufCatalog(checkpoint),
        metadata,
    ) {
        Ok(vision) => vision,
        Err(error) => return invalid_geometry(error.to_string()),
    };
    let composite = match eredu_architectures::qwen::hybrid::with_gguf_vision_projector(
        parsed,
        model_metadata,
        vision,
    ) {
        Ok(composite) => composite,
        Err(error) => return invalid_geometry(error.to_string()),
    };
    let plan = match eredu_architectures::qwen::hybrid::conditional_projector_gguf_plan(&composite)
    {
        Ok(plan) => plan,
        Err(error) => return invalid_geometry(error),
    };
    validate_neutral_qwen_projector(
        checkpoint,
        composite
            .vision
            .as_ref()
            .expect("admitted Qwen3.5 projector has vision config"),
        &plan,
    )
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
    vision: &eredu_architectures::qwen::vision::VisionConfig,
    plan: &eredu_checkpoint::schema::GgufCheckpointPlan,
) -> StructuralValidation {
    let deepstack = vision.deepstack_layers();
    if let Err(error) = checkpoint.catalog().translated_outputs(|name| {
        eredu_architectures::qwen::vision::translate_gguf_weight_name(name, &deepstack)
    }) {
        return invalid_geometry(error.to_string());
    }
    eredu_checkpoint::validation::validate_gguf_plan(checkpoint, plan)
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
        validate_safetensors_preparation_for_test(ModelKind::Gemma4, &gemma4_config(true), options)
            .unwrap();
        validate_safetensors_preparation_for_test(
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
        validate_safetensors_preparation_for_test(
            ModelKind::KimiLinear,
            &kimi_linear_config(),
            options,
        )
        .unwrap();
    }

    #[test]
    fn complete_tensor_parallel_quantization_is_rejected_during_preflight() {
        let topology = crate::backend::MlxParallelContext::for_rank(
            0,
            2,
            1,
            1,
            crate::backend::DeviceAssignment::new(safemlx::DeviceType::Cpu, 0),
        )
        .unwrap();
        let options =
            ModelLoadOptions::with_quantization(eredu_checkpoint::WeightQuantization::MxFp4)
                .with_parallel_topology(topology);

        let error = validate_safetensors_preparation_for_test(
            ModelKind::Qwen3,
            &dense_qwen3_config(),
            options,
        )
        .unwrap_err();

        assert!(matches!(
            error,
            Error::Artifact(
                eredu_core::artifact::ArtifactError::UnsupportedQuantizationPolicy(message)
            ) if message.contains("complete tensor-parallel")
        ));
    }

    #[test]
    fn distributed_stage_quantization_is_admitted_during_preflight() {
        let topology = crate::backend::MlxParallelContext::for_rank(
            0,
            1,
            2,
            1,
            crate::backend::DeviceAssignment::new(safemlx::DeviceType::Cpu, 0),
        )
        .unwrap();
        let options =
            ModelLoadOptions::with_quantization(eredu_checkpoint::WeightQuantization::MxFp4)
                .with_parallel_topology(topology);

        validate_safetensors_preparation_for_test(ModelKind::Qwen3, &dense_qwen3_config(), options)
            .unwrap();
    }

    #[test]
    fn replicated_gguf_quantization_requires_a_bound_complete_loader() {
        let policy = eredu_core::PreparationPolicy {
            quantization: Some(eredu_core::QuantizationRequest::MxFp4),
            ..eredu_core::PreparationPolicy::default()
        };
        let capabilities = eredu_architectures::preparation::ArchitectureCapabilities::default();

        validate_quantization_capability(
            ModelKind::Qwen3,
            eredu_core::ArtifactFormat::Gguf,
            policy,
            capabilities,
        )
        .unwrap();
        let error = validate_quantization_capability(
            ModelKind::DeepSeekV3,
            eredu_core::ArtifactFormat::Gguf,
            policy,
            capabilities,
        )
        .unwrap_err();

        assert!(matches!(
            error,
            Error::Artifact(
                eredu_core::artifact::ArtifactError::UnsupportedQuantizationPolicy(message)
            ) if message.contains("deepseek_v3") && message.contains("GGUF")
        ));
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
                validate_safetensors_preparation_for_test(kind, &config, options),
                Err(Error::Artifact(
                    eredu_core::artifact::ArtifactError::UnsupportedResidencyPolicy(_)
                ))
            ));
        }
    }

    #[test]
    fn preparation_rejects_an_unsupported_exact_parallel_axis() {
        let topology = crate::backend::MlxParallelContext::for_rank(
            0,
            1,
            1,
            2,
            crate::backend::DeviceAssignment::new(safemlx::DeviceType::Cpu, 0),
        )
        .unwrap();
        let error = validate_safetensors_preparation_for_test(
            ModelKind::Qwen3,
            &dense_qwen3_config(),
            ModelLoadOptions::with_parallel(topology),
        )
        .unwrap_err();

        assert!(matches!(error, Error::Parallel(message) if message.contains("expert-parallel")));
    }

    #[test]
    fn preparation_accepts_supported_exact_parallel_axes() {
        let topology = crate::backend::MlxParallelContext::for_rank(
            0,
            2,
            3,
            1,
            crate::backend::DeviceAssignment::new(safemlx::DeviceType::Cpu, 0),
        )
        .unwrap();

        validate_safetensors_preparation_for_test(
            ModelKind::Qwen3,
            &dense_qwen3_config(),
            ModelLoadOptions::with_parallel(topology),
        )
        .unwrap();
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
