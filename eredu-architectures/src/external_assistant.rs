//! Architecture-owned inspection and preparation of external draft assistants.

use std::{collections::HashMap, path::PathBuf};

use eredu_checkpoint::{
    schema::SafetensorsCheckpointPlan,
    validation::{resolve_gguf_plan, resolve_safetensors_plan, ResolvedCheckpointPlan},
};
use eredu_core::{
    artifact::ArtifactError, checkpoint::TensorCatalog, ArtifactFormat, LoadingProtocol,
    ModelConfiguration, ModelConfigurationResolver, ResolvedModelConfiguration,
};
use eredu_gguf::{Checkpoint, MetadataValue};
use serde_json::Value;

use crate::{gemma4, muse_glimmer};

/// Inspected checkpoint source consumed by a concrete assistant materializer.
#[derive(Debug, Clone)]
pub enum ExternalAssistantCheckpoint {
    /// Header-inspected Hugging Face SafeTensors directory.
    SafeTensors {
        /// Submitted artifact directory containing the admitted payload members.
        source: PathBuf,
        /// Exact header catalog admitted during neutral preparation.
        catalog: TensorCatalog,
        /// Strict architecture schema used to revalidate the reopened source.
        plan: SafetensorsCheckpointPlan,
        /// Exact physical layout selected during neutral admission.
        resolution: ResolvedCheckpointPlan,
    },
    /// Header-inspected and architecture-admitted GGUF checkpoint.
    Gguf {
        /// Portable checkpoint handle retained from inspection.
        checkpoint: Checkpoint,
        /// Exact architecture layout selected during admission.
        resolution: ResolvedCheckpointPlan,
        /// Canonical physical-to-logical tensor mapping resolved during admission.
        tensor_mapping: Vec<eredu_gguf::TranslatedTensorLayout>,
    },
}

/// Fully inspected Gemma 4 assistant materialization input.
#[derive(Debug, Clone)]
pub struct Gemma4AssistantPreparationPlan {
    checkpoint: ExternalAssistantCheckpoint,
    config: gemma4::AssistantConfig,
}

impl Gemma4AssistantPreparationPlan {
    /// Consumes the plan into its admitted checkpoint and normalized configuration.
    pub fn into_parts(self) -> (ExternalAssistantCheckpoint, gemma4::AssistantConfig) {
        (self.checkpoint, self.config)
    }
}

/// Fully inspected Muse-Glimmer DFlash materialization input.
#[derive(Debug, Clone)]
pub struct MuseGlimmerAssistantPreparationPlan {
    checkpoint: ExternalAssistantCheckpoint,
    config: muse_glimmer::DFlashConfig,
}

impl MuseGlimmerAssistantPreparationPlan {
    /// Consumes the plan into its admitted checkpoint and normalized configuration.
    pub fn into_parts(self) -> (ExternalAssistantCheckpoint, muse_glimmer::DFlashConfig) {
        (self.checkpoint, self.config)
    }
}

/// Architecture-dispatched external assistant materialization plan.
#[derive(Debug, Clone)]
pub enum ExternalAssistantPreparationPlan {
    /// Gemma 4 shared-KV assistant.
    Gemma4(Gemma4AssistantPreparationPlan),
    /// Muse-Glimmer anchor-plus-mask DFlash assistant.
    MuseGlimmer(MuseGlimmerAssistantPreparationPlan),
}

impl ExternalAssistantPreparationPlan {
    /// Architecture identity whose tokenizer contract the assistant shares.
    pub const fn tokenizer_model_kind(&self) -> crate::configuration::ModelKind {
        match self {
            Self::Gemma4(_) => crate::configuration::ModelKind::Gemma4,
            Self::MuseGlimmer(_) => crate::configuration::ModelKind::MuseGlimmer,
        }
    }
}

/// Inspects and admits an external draft assistant without selecting a backend.
///
/// Configuration, container format, family dispatch, GGUF metadata, and the
/// strict tensor contract and resolved layout are fixed here. Concrete
/// backends receive only this plan and may open or map the selected weight
/// payloads during materialization after revalidating that admission.
pub fn prepare_external_assistant(
    source: impl AsRef<std::path::Path>,
) -> Result<ExternalAssistantPreparationPlan, ArtifactError> {
    let inspection = eredu_core::inspect_artifact(source, &AssistantConfigurations)?;
    let family = inspection.configuration().family.as_str();
    let config_json = inspection.configuration().json.as_ref();
    let metadata = inspection.gguf_checkpoint().map(gguf_metadata);

    match family {
        "gemma4_assistant" => {
            let config = match inspection.format() {
                ArtifactFormat::SafeTensors => gemma4::AssistantConfig::from_json(
                    &serde_json::to_vec(config_json.expect("SafeTensors inspection has JSON"))?,
                )
                .map_err(invalid_assistant)?,
                ArtifactFormat::Gguf => gemma4::AssistantConfig::from_gguf_metadata(
                    inspection
                        .gguf_checkpoint()
                        .expect("GGUF inspection has checkpoint"),
                    metadata.as_ref().expect("GGUF inspection has metadata"),
                )
                .map_err(invalid_assistant)?,
            };
            let checkpoint = prepared_checkpoint(
                &inspection,
                || gemma4::assistant_safetensors_plan(&config),
                || gemma4::assistant_gguf_plan(&config),
                gemma4::translate_assistant_gguf_weight_name,
            )?;
            Ok(ExternalAssistantPreparationPlan::Gemma4(
                Gemma4AssistantPreparationPlan { checkpoint, config },
            ))
        }
        "muse_glimmer_assistant" => {
            let config = match inspection.format() {
                ArtifactFormat::SafeTensors => muse_glimmer::DFlashConfig::from_hf_json(
                    &serde_json::to_vec(config_json.expect("SafeTensors inspection has JSON"))?,
                )
                .map_err(invalid_assistant)?,
                ArtifactFormat::Gguf => muse_glimmer::DFlashConfig::from_gguf_metadata(
                    metadata.as_ref().expect("GGUF inspection has metadata"),
                )
                .map_err(invalid_assistant)?,
            };
            let checkpoint = prepared_checkpoint(
                &inspection,
                || muse_glimmer::dflash_safetensors_plan(&config),
                || muse_glimmer::dflash_gguf_plan(&config),
                muse_glimmer::translate_dflash_gguf_weight_name,
            )?;
            Ok(ExternalAssistantPreparationPlan::MuseGlimmer(
                MuseGlimmerAssistantPreparationPlan { checkpoint, config },
            ))
        }
        other => Err(ArtifactError::UnsupportedModelType(other.into())),
    }
}

fn prepared_checkpoint(
    inspection: &eredu_core::ArtifactInspection,
    safetensors_plan: impl FnOnce() -> Result<SafetensorsCheckpointPlan, String>,
    gguf_plan: impl FnOnce() -> Result<eredu_checkpoint::schema::GgufCheckpointPlan, String>,
    gguf_translate: impl FnMut(&str) -> String,
) -> Result<ExternalAssistantCheckpoint, ArtifactError> {
    match inspection.format() {
        ArtifactFormat::SafeTensors => {
            let plan = safetensors_plan().map_err(ArtifactError::InvalidArtifact)?;
            let resolution = resolve_safetensors_plan(
                &crate::configuration::PortableSafetensorsCatalog(inspection.tensors()),
                &plan,
            )
            .map_err(|validation| {
                invalid_assistant(format!(
                    "external assistant checkpoint contract did not resolve: {validation:?}"
                ))
            })?;
            Ok(ExternalAssistantCheckpoint::SafeTensors {
                source: inspection.path().to_owned(),
                catalog: inspection.tensors().clone(),
                plan,
                resolution,
            })
        }
        ArtifactFormat::Gguf => {
            let checkpoint = inspection
                .gguf_checkpoint()
                .expect("GGUF inspection has checkpoint");
            let plan = gguf_plan().map_err(ArtifactError::InvalidArtifact)?;
            let resolution = resolve_gguf_plan(checkpoint, &plan).map_err(|validation| {
                ArtifactError::InvalidArtifact(format!(
                    "external assistant checkpoint contract did not resolve: {validation:?}"
                ))
            })?;
            let tensor_mapping = checkpoint
                .translated_outputs(gguf_translate)
                .map_err(|error| ArtifactError::InvalidArtifact(error.to_string()))?;
            Ok(ExternalAssistantCheckpoint::Gguf {
                checkpoint: checkpoint.clone(),
                resolution,
                tensor_mapping,
            })
        }
    }
}

fn invalid_assistant(error: impl std::fmt::Display) -> ArtifactError {
    ArtifactError::InvalidArtifact(error.to_string())
}

fn gguf_metadata(checkpoint: &Checkpoint) -> HashMap<String, MetadataValue> {
    checkpoint
        .metadata()
        .iter()
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect()
}

struct AssistantConfigurations;

impl ModelConfigurationResolver for AssistantConfigurations {
    type ArtifactPlan = ();

    fn resolve_safetensors(
        &self,
        json: &Value,
    ) -> Result<ResolvedModelConfiguration<Self::ArtifactPlan>, ArtifactError> {
        let bytes = serde_json::to_vec(json)?;
        let model_type = json
            .get("model_type")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                ArtifactError::InvalidArtifact("assistant config is missing model_type".into())
            })?;
        let family = match model_type {
            "gemma4_assistant" => {
                gemma4::AssistantConfig::from_json(&bytes).map_err(invalid_assistant)?;
                "gemma4_assistant"
            }
            "muse_glimmer_assistant" => {
                muse_glimmer::DFlashConfig::from_hf_json(&bytes).map_err(invalid_assistant)?;
                "muse_glimmer_assistant"
            }
            other => return Err(ArtifactError::UnsupportedModelType(other.into())),
        };
        Ok(ResolvedModelConfiguration::new(
            ModelConfiguration {
                declared_model_type: model_type.into(),
                effective_model_type: model_type.into(),
                family: family.into(),
                loading_protocol: LoadingProtocol::Model,
                json: Some(json.clone()),
            },
            (),
        ))
    }

    fn resolve_gguf(
        &self,
        architecture: &str,
        checkpoint: &Checkpoint,
    ) -> Result<ResolvedModelConfiguration<Self::ArtifactPlan>, ArtifactError> {
        let metadata = gguf_metadata(checkpoint);
        let family = match architecture {
            "gemma4_assistant" | "gemma4-assistant" => {
                gemma4::AssistantConfig::from_gguf_metadata(checkpoint, &metadata)
                    .map_err(invalid_assistant)?;
                "gemma4_assistant"
            }
            "dflash" => {
                muse_glimmer::DFlashConfig::from_gguf_metadata(&metadata)
                    .map_err(invalid_assistant)?;
                "muse_glimmer_assistant"
            }
            other => return Err(ArtifactError::UnsupportedGgufArchitecture(other.into())),
        };
        Ok(ResolvedModelConfiguration::new(
            ModelConfiguration {
                declared_model_type: architecture.into(),
                effective_model_type: family.into(),
                family: family.into(),
                loading_protocol: LoadingProtocol::Model,
                json: None,
            },
            (),
        ))
    }

    fn gguf_companion_requirements(
        &self,
        _architecture: &str,
        _checkpoint: &Checkpoint,
    ) -> Result<Vec<eredu_core::GgufCompanionRequirement>, ArtifactError> {
        Ok(Vec::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eredu_checkpoint::schema::StoredDtypeConstraint;
    use safetensors::tensor::{serialize_to_file, Dtype, TensorView};

    const GEMMA_ASSISTANT: &str = r#"{
      "model_type":"gemma4_assistant","backbone_hidden_size":32,
      "use_ordered_embeddings":false,"tie_word_embeddings":false,"block_size":4,
      "text_config":{"model_type":"gemma4_text","hidden_size":32,
        "num_hidden_layers":1,"intermediate_size":64,"num_attention_heads":4,
        "num_key_value_heads":2,"head_dim":8,"rms_norm_eps":0.00001,
        "vocab_size":32,"max_position_embeddings":128,"tie_word_embeddings":false,
        "attention_k_eq_v":false,"layer_types":["full_attention"]}
    }"#;

    type TestTensor = (String, Vec<usize>, Vec<u8>);

    fn gemma_tensors() -> Vec<TestTensor> {
        let config = gemma4::AssistantConfig::from_json(GEMMA_ASSISTANT.as_bytes()).unwrap();
        let plan = gemma4::assistant_safetensors_plan(&config).unwrap();
        assert!(plan.layout_groups.is_empty());
        plan.common_tensors
            .into_iter()
            .map(|tensor| {
                assert_eq!(tensor.dtype, StoredDtypeConstraint::Floating);
                let elements = tensor.shape.iter().product::<usize>();
                (tensor.key, tensor.shape, vec![0; elements * 4])
            })
            .collect()
    }

    fn safetensors_artifact(config: &str, tensors: Vec<TestTensor>) -> tempfile::TempDir {
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(directory.path().join("config.json"), config).unwrap();
        let views = tensors
            .iter()
            .map(|(name, shape, bytes)| {
                (
                    name.as_str(),
                    TensorView::new(Dtype::F32, shape.clone(), bytes).unwrap(),
                )
            })
            .collect::<Vec<_>>();
        serialize_to_file(views, None, &directory.path().join("model.safetensors")).unwrap();
        directory
    }

    #[test]
    fn safetensors_assistant_is_dispatched_before_backend_materialization() {
        let artifact = safetensors_artifact(GEMMA_ASSISTANT, gemma_tensors());
        let preparation = prepare_external_assistant(artifact.path()).unwrap();
        let ExternalAssistantPreparationPlan::Gemma4(plan) = preparation else {
            panic!("Gemma assistant was dispatched to the wrong family");
        };
        let (checkpoint, config) = plan.into_parts();
        assert_eq!(config.model_type, "gemma4_assistant");
        assert!(matches!(
            checkpoint,
            ExternalAssistantCheckpoint::SafeTensors {
                source,
                catalog,
                resolution,
                ..
            }
                if source == artifact.path()
                    && catalog.len() == resolution.source_keys().len()
        ));
    }

    #[test]
    fn safetensors_assistant_rejects_missing_extra_and_malformed_tensors_during_preparation() {
        let mut missing = gemma_tensors();
        missing.pop();
        let missing = safetensors_artifact(GEMMA_ASSISTANT, missing);
        assert!(matches!(
            prepare_external_assistant(missing.path()),
            Err(ArtifactError::InvalidArtifact(_))
        ));

        let mut extra = gemma_tensors();
        extra.push(("undeclared.weight".into(), vec![1], vec![0; 4]));
        let extra = safetensors_artifact(GEMMA_ASSISTANT, extra);
        assert!(matches!(
            prepare_external_assistant(extra.path()),
            Err(ArtifactError::InvalidArtifact(_))
        ));

        let mut malformed = gemma_tensors();
        malformed[0].1.push(1);
        let malformed = safetensors_artifact(GEMMA_ASSISTANT, malformed);
        assert!(matches!(
            prepare_external_assistant(malformed.path()),
            Err(ArtifactError::InvalidArtifact(_))
        ));
    }

    #[test]
    fn safetensors_assistant_admission_rejects_missing_or_conflicting_identities() {
        let mut missing: Value = serde_json::from_str(GEMMA_ASSISTANT).unwrap();
        missing.as_object_mut().unwrap().remove("model_type");
        let missing = safetensors_artifact(&missing.to_string(), gemma_tensors());
        assert!(matches!(
            prepare_external_assistant(missing.path()),
            Err(ArtifactError::InvalidArtifact(_))
        ));

        let mut conflicting: Value = serde_json::from_str(GEMMA_ASSISTANT).unwrap();
        conflicting["text_config"]["model_type"] = "llama".into();
        let conflicting = safetensors_artifact(&conflicting.to_string(), gemma_tensors());
        assert!(matches!(
            prepare_external_assistant(conflicting.path()),
            Err(ArtifactError::InvalidArtifact(_))
        ));
    }

    #[test]
    fn ordinary_model_cannot_cross_the_external_assistant_boundary() {
        let artifact = safetensors_artifact(
            r#"{"model_type":"llama"}"#,
            vec![("weight".into(), vec![1], vec![0; 4])],
        );
        assert!(matches!(
            prepare_external_assistant(artifact.path()),
            Err(ArtifactError::UnsupportedModelType(model_type)) if model_type == "llama"
        ));
    }
}
