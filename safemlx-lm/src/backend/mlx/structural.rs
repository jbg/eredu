//! MLX architecture binding against portable checkpoint catalogs.

use std::collections::HashMap;
#[cfg(test)]
use std::path::Path;

use safemlx::ops::{GgufCheckpoint, GgufMetadataValue};
use serde_json::Value;

use safemlx_lm_core::{GgufArchitecture, ModelKind};

use super::ModelLoadOptions;
use crate::backend::mlx::{
    architectures::{
        deepseek_v3::checkpoint as deepseek_v3_checkpoint,
        deepseek_v4::checkpoint as deepseek_v4_checkpoint,
        gemma4::{checkpoint as gemma4_checkpoint, model as gemma4},
        gpt_oss::checkpoint as gpt_oss_checkpoint,
        inkling::{checkpoint as inkling_checkpoint, model as inkling},
        kimi_linear::checkpoint as kimi_linear_checkpoint,
        lfm2::checkpoint as lfm2_checkpoint,
        llama::checkpoint as llama_checkpoint,
        moshi::personaplex_checkpoint,
        muse_glimmer::checkpoint as muse_glimmer_checkpoint,
        nemotron_h::checkpoint as nemotron_h_checkpoint,
        qwen::{
            dense::checkpoint as dense_qwen_checkpoint,
            hybrid::checkpoint as qwen_hybrid_checkpoint, vl::checkpoint as qwen_vl_checkpoint,
        },
    },
    error::Error,
    runtime::checkpoint::store::SafetensorsWeightStore,
};

pub(crate) use crate::backend::mlx::runtime::checkpoint::contract::{
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
            safemlx_lm_core::ArtifactFormat::Gguf,
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

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
#[allow(dead_code)] // Reserved for fail-closed structural policies.
pub(crate) enum StructuralValidationPolicy {
    Exact,
    Unverified,
}

/// Exhaustive policy table for high-level SafeTensors loader families.
pub(crate) const fn safetensors_policy(kind: ModelKind) -> StructuralValidationPolicy {
    match kind {
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
        | ModelKind::PersonaPlex
        | ModelKind::Qwen2
        | ModelKind::Qwen3
        | ModelKind::Qwen3Next
        | ModelKind::Qwen3Vl
        | ModelKind::Qwen3VlMoe
        | ModelKind::Qwen35 => StructuralValidationPolicy::Exact,
    }
}

/// Exhaustive policy table for concrete GGUF loader architectures.
pub(crate) const fn gguf_policy(architecture: GgufArchitecture) -> StructuralValidationPolicy {
    match architecture {
        GgufArchitecture::Llama
        | GgufArchitecture::Mistral
        | GgufArchitecture::MuseGlimmer
        | GgufArchitecture::DeepSeek2
        | GgufArchitecture::DeepSeek4
        | GgufArchitecture::Lfm2
        | GgufArchitecture::Lfm2Moe
        | GgufArchitecture::GptOss
        | GgufArchitecture::Gemma4
        | GgufArchitecture::Inkling
        | GgufArchitecture::Qwen2
        | GgufArchitecture::Qwen3
        | GgufArchitecture::Qwen3Moe
        | GgufArchitecture::NemotronH
        | GgufArchitecture::NemotronHMoe
        | GgufArchitecture::Qwen35
        | GgufArchitecture::Qwen35Moe
        | GgufArchitecture::Qwen3Next
        | GgufArchitecture::Qwen3Vl
        | GgufArchitecture::Qwen3VlMoe
        | GgufArchitecture::KimiLinear => StructuralValidationPolicy::Exact,
    }
}

pub(crate) fn validate_safetensors(
    kind: ModelKind,
    config: &Value,
    store: &SafetensorsWeightStore,
    options: ModelLoadOptions,
) -> StructuralValidation {
    let validation = match safetensors_policy(kind) {
        StructuralValidationPolicy::Exact => match kind {
            ModelKind::DeepSeekV3 => deepseek_v3_checkpoint::validate_safetensors(
                config,
                store,
                !options.weight_residency.is_fully_resident(),
            ),
            ModelKind::DeepSeekV4 => deepseek_v4_checkpoint::validate_safetensors(config, store),
            ModelKind::Gemma4 => gemma4_checkpoint::validate_safetensors(
                config,
                store,
                !options.weight_residency.is_fully_resident(),
                options.weight_residency.expert_cache().is_some(),
            ),
            ModelKind::GptOss => gpt_oss_checkpoint::validate_safetensors(config, store),
            ModelKind::Inkling => inkling_checkpoint::validate_safetensors(config, store),
            ModelKind::KimiLinear => kimi_linear_checkpoint::validate_safetensors(config, store),
            ModelKind::Lfm2 => lfm2_checkpoint::validate_safetensors(
                config,
                store,
                !options.weight_residency.is_fully_resident(),
            ),
            ModelKind::Llama => llama_checkpoint::validate_safetensors(config, store),
            ModelKind::MuseGlimmer => muse_glimmer_checkpoint::validate_safetensors(config, store),
            ModelKind::NemotronH => nemotron_h_checkpoint::validate_safetensors(config, store),
            ModelKind::PersonaPlex => personaplex_checkpoint::validate_safetensors(config, store),
            ModelKind::Qwen2 | ModelKind::Qwen3 => {
                dense_qwen_checkpoint::validate_safetensors(config, store)
            }
            ModelKind::Qwen3Next => qwen_hybrid_checkpoint::validate_qwen3_next_safetensors(
                config,
                store,
                options.weight_residency.expert_cache().is_some(),
            ),
            ModelKind::Qwen3Vl | ModelKind::Qwen3VlMoe => qwen_vl_checkpoint::validate_safetensors(
                kind == ModelKind::Qwen3VlMoe,
                config,
                store,
                !options.weight_residency.is_fully_resident(),
            ),
            ModelKind::Qwen35 => qwen_hybrid_checkpoint::validate_qwen35_safetensors(
                config,
                store,
                options.weight_residency.expert_cache().is_some(),
            ),
        },
        StructuralValidationPolicy::Unverified => unverified(kind.model_type_name()),
    };
    validation.with_strict_catalog(options.weight_residency.strict_loading())
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
    validate_safetensors(kind, &config, &store, options).into_loader_result()
}

pub(crate) fn validate_gguf(
    architecture: GgufArchitecture,
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
    options: ModelLoadOptions,
) -> StructuralValidation {
    let validation = match gguf_policy(architecture) {
        StructuralValidationPolicy::Exact => match architecture {
            GgufArchitecture::DeepSeek2 => {
                deepseek_v3_checkpoint::validate_gguf(checkpoint, metadata)
            }
            GgufArchitecture::DeepSeek4 => {
                deepseek_v4_checkpoint::validate_gguf(checkpoint, metadata)
            }
            GgufArchitecture::GptOss => gpt_oss_checkpoint::validate_gguf(checkpoint, metadata),
            GgufArchitecture::Gemma4 => {
                if let Err(error) = architecture.validate_load_policy(options) {
                    invalid_geometry(error.to_string())
                } else {
                    gemma4_checkpoint::validate_gguf(checkpoint, metadata)
                }
            }
            GgufArchitecture::Inkling => {
                if let Err(error) = architecture.validate_load_policy(options) {
                    invalid_geometry(error.to_string())
                } else {
                    inkling_checkpoint::validate_gguf(checkpoint, metadata)
                }
            }
            GgufArchitecture::Lfm2 | GgufArchitecture::Lfm2Moe => {
                let variant = if architecture == GgufArchitecture::Lfm2Moe {
                    lfm2_checkpoint::GgufVariant::Moe
                } else {
                    lfm2_checkpoint::GgufVariant::Dense
                };
                lfm2_checkpoint::validate_gguf(variant, checkpoint, metadata)
            }
            GgufArchitecture::Llama | GgufArchitecture::Mistral => {
                llama_checkpoint::validate_gguf(checkpoint, metadata)
            }
            GgufArchitecture::MuseGlimmer => {
                muse_glimmer_checkpoint::validate_gguf(checkpoint, metadata)
            }
            GgufArchitecture::NemotronH | GgufArchitecture::NemotronHMoe => {
                if let Err(error) = architecture.validate_load_policy(options) {
                    invalid_geometry(error.to_string())
                } else {
                    let variant = if architecture == GgufArchitecture::NemotronHMoe {
                        nemotron_h_checkpoint::GgufVariant::Moe
                    } else {
                        nemotron_h_checkpoint::GgufVariant::Dense
                    };
                    nemotron_h_checkpoint::validate_gguf(variant, checkpoint, metadata)
                }
            }
            GgufArchitecture::Qwen2 | GgufArchitecture::Qwen3 | GgufArchitecture::Qwen3Moe => {
                let variant = match architecture {
                    GgufArchitecture::Qwen2 => dense_qwen_checkpoint::GgufVariant::Qwen2,
                    GgufArchitecture::Qwen3 => dense_qwen_checkpoint::GgufVariant::Qwen3,
                    GgufArchitecture::Qwen3Moe => dense_qwen_checkpoint::GgufVariant::Qwen3Moe,
                    _ => unreachable!("covered by the outer architecture match"),
                };
                dense_qwen_checkpoint::validate_gguf(variant, checkpoint, metadata)
            }
            architecture @ (GgufArchitecture::Qwen3Vl | GgufArchitecture::Qwen3VlMoe) => {
                if let Err(error) = architecture.validate_load_policy(options) {
                    invalid_geometry(error.to_string())
                } else {
                    let variant = match architecture {
                        GgufArchitecture::Qwen3Vl => qwen_vl_checkpoint::GgufVariant::Dense,
                        GgufArchitecture::Qwen3VlMoe => qwen_vl_checkpoint::GgufVariant::Moe,
                        _ => unreachable!("covered by the outer architecture match"),
                    };
                    qwen_vl_checkpoint::validate_gguf(variant, checkpoint, metadata)
                }
            }
            GgufArchitecture::KimiLinear => {
                kimi_linear_checkpoint::validate_gguf(checkpoint, metadata)
            }
            GgufArchitecture::Qwen35
            | GgufArchitecture::Qwen35Moe
            | GgufArchitecture::Qwen3Next => {
                let variant = match architecture {
                    GgufArchitecture::Qwen35 => qwen_hybrid_checkpoint::GgufVariant::Qwen35,
                    GgufArchitecture::Qwen35Moe => qwen_hybrid_checkpoint::GgufVariant::Qwen35Moe,
                    GgufArchitecture::Qwen3Next => qwen_hybrid_checkpoint::GgufVariant::Qwen3Next,
                    _ => unreachable!("covered by the outer architecture match"),
                };
                qwen_hybrid_checkpoint::validate_gguf(
                    variant,
                    checkpoint,
                    metadata,
                    options.weight_residency.expert_cache().is_some(),
                )
            }
        },
        StructuralValidationPolicy::Unverified => unverified(architecture.metadata_name()),
    };
    validation.with_strict_catalog(options.weight_residency.strict_loading())
}

fn unverified(architecture: &str) -> StructuralValidation {
    StructuralValidation::Unverified(StructuralIssue {
        kind: StructuralIssueKind::ValidationUnavailable,
        detail: format!(
            "exact header-only structural validation is not yet implemented for {architecture}"
        ),
        tensor_name: None,
        tensor_type_code: None,
        metadata_key: None,
    })
}

pub(crate) fn validate_inkling_mmproj_gguf(
    model_metadata: &HashMap<String, GgufMetadataValue>,
    mmproj: &inkling::InklingMmprojGguf,
) -> StructuralValidation {
    inkling_checkpoint::validate_mmproj_gguf(model_metadata, mmproj)
}

pub(crate) fn validate_gemma4_mmproj_gguf(
    model_checkpoint: &GgufCheckpoint,
    model_metadata: &HashMap<String, GgufMetadataValue>,
    mmproj: &gemma4::Gemma4MmprojGguf,
) -> StructuralValidation {
    gemma4_checkpoint::validate_mmproj_gguf(model_checkpoint, model_metadata, mmproj)
}

pub(crate) fn validate_muse_glimmer_projector_gguf(
    model_checkpoint: &GgufCheckpoint,
    model_metadata: &HashMap<String, GgufMetadataValue>,
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
) -> StructuralValidation {
    muse_glimmer_checkpoint::validate_projector_gguf(
        model_checkpoint,
        model_metadata,
        checkpoint,
        metadata,
    )
}

pub(crate) fn validate_qwen3_vl_projector_gguf(
    model_checkpoint: &GgufCheckpoint,
    model_metadata: &HashMap<String, GgufMetadataValue>,
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
) -> StructuralValidation {
    qwen_vl_checkpoint::validate_projector_gguf(
        model_checkpoint,
        model_metadata,
        checkpoint,
        metadata,
    )
}

pub(crate) fn validate_qwen35_projector_gguf(
    model_checkpoint: &GgufCheckpoint,
    model_metadata: &HashMap<String, GgufMetadataValue>,
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
) -> StructuralValidation {
    qwen_hybrid_checkpoint::validate_projector_gguf(
        model_checkpoint,
        model_metadata,
        checkpoint,
        metadata,
    )
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
        let malformed = crate::backend::mlx::runtime::checkpoint::contract::shape_mismatch(
            "model.weight",
            &[2, 2],
            &[1],
        );
        assert_eq!(
            StructuralValidation::Invalid(vec![unexpected.clone(), malformed.clone()])
                .with_strict_catalog(false),
            StructuralValidation::Invalid(vec![malformed])
        );

        let error = StructuralValidation::Invalid(vec![unexpected])
            .into_loader_result()
            .unwrap_err();
        assert!(matches!(
            error,
            Error::StrictLoadValidation { missing, unused }
                if missing.is_empty() && unused == ["unrelated.weight"]
        ));
    }
}

#[cfg(test)]
mod dense_qwen_tests {
    use std::collections::BTreeSet;

    use super::*;
    use crate::backend::mlx::architectures::qwen::dense;

    fn qwen2_args(tied: bool) -> dense::DecoderConfig {
        dense::config_from_hf_value(&serde_json::json!({
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
        let tied = dense_qwen_checkpoint::safetensors_plan(&qwen2_args(true)).unwrap();
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
        assert!(dense_qwen_checkpoint::is_redundant_tied_output_head_key(
            &qwen2_args(true),
            "lm_head.weight"
        ));
        assert!(!dense_qwen_checkpoint::is_redundant_tied_output_head_key(
            &qwen2_args(false),
            "lm_head.weight"
        ));

        let untied = dense_qwen_checkpoint::safetensors_plan(&qwen2_args(false)).unwrap();
        assert!(untied
            .common_tensors
            .iter()
            .any(|tensor| tensor.key == "lm_head.weight"));
    }
}
