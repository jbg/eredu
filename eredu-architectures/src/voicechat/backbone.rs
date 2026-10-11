//! Retained VoiceChat weights projected onto the existing Nemotron-H parameter contract.

use super::{
    artifact::{ParameterAdmission, RetainedVoiceChatArtifacts},
    checkpoint::Component,
};
use crate::nemotron_h::{self, ModelArgs};
use eredu_checkpoint::{
    recipe::{DerivedWeightRecipe, RecipeDtype, RecipeError},
    store::{RestrictedCheckpointSource, SharedCheckpointSource, StoreError},
    validation::{
        validate_safetensors_plan, CatalogTensorMetadata, SafetensorsCatalog, StrictLoadFailure,
    },
    StoredDtype,
};
use std::{collections::BTreeMap, sync::Arc};

#[path = "backbone_construction.rs"]
mod construction;
pub use construction::{BackboneConstructionError, BackboneUnit, PreparedBackboneBindings};

/// A missing component or incompatible retained architecture configuration.
#[derive(Debug, thiserror::Error)]
pub enum BackbonePreparationError {
    /// The selected task did not retain the backbone source role.
    #[error("VoiceChat backbone component was not retained")]
    MissingComponent,
    /// The exact base configuration cannot describe this retained component.
    #[error("invalid VoiceChat backbone: {0}")]
    Configuration(String),
    /// Exact normalized recipe outputs disagree with the Nemotron catalog.
    #[error(transparent)]
    Catalog(#[from] StrictLoadFailure),
    /// Retained recipe metadata cannot be evaluated.
    #[error(transparent)]
    Recipe(#[from] RecipeError),
    /// Exact source restriction failed.
    #[error(transparent)]
    Source(#[from] StoreError),
}

/// Exact restricted sources and recipes for ordinary Nemotron-H construction.
///
/// Runtime names differ from the enclosing speech checkpoint. Recipes retain
/// original physical names, source guards, conversion bounds and provenance.
/// No artifact is reopened and no backend resource is allocated here.
#[derive(Clone)]
pub struct PreparedBackboneParameters {
    args: ModelArgs,
    source: SharedCheckpointSource,
    parameters: BTreeMap<String, ParameterAdmission>,
}
impl PreparedBackboneParameters {
    /// Projects admitted canonical parameters using the already verified base asset.
    pub fn from_retained(
        artifacts: &RetainedVoiceChatArtifacts,
    ) -> Result<Self, BackbonePreparationError> {
        let source = artifacts
            .component_source(Component::Backbone)
            .ok_or(BackbonePreparationError::MissingComponent)?
            .clone();
        let args =
            nemotron_h::model_args_from_config_value(artifacts.assets().backbone_configuration())
                .map_err(|e| BackbonePreparationError::Configuration(e.to_string()))?;
        Self::from_parameters(args, source, artifacts.plan().parameters())
    }

    /// Admits explicit canonical component recipes against ordinary Nemotron geometry.
    ///
    /// Recipe outputs, naming, and the complete target catalog are checked without
    /// payload reads. Only the selected component's exact physical inputs remain
    /// visible, retaining source guards and any original storage transformations.
    pub fn from_parameters(
        mut args: ModelArgs,
        source: SharedCheckpointSource,
        parameters: &BTreeMap<String, ParameterAdmission>,
    ) -> Result<Self, BackbonePreparationError> {
        args.validate()
            .map_err(|error| BackbonePreparationError::Configuration(error.to_string()))?;
        if args.num_nextn_predict_layers != 0 {
            return Err(BackbonePreparationError::Configuration(
                "VoiceChat backbone requires one target group".into(),
            ));
        }
        let parameters = project_parameters(&args, parameters)?
            .into_iter()
            .map(|(name, parameter)| {
                super::parameters::execution_parameter(&parameter, &source)
                    .map(|parameter| (name, parameter))
            })
            .collect::<Result<BTreeMap<_, _>, _>>()?;
        // The admitted physical matrix formats are construction authority for
        // dense and quantized inputs alike, including mixed checkpoint policies.
        args.quantized_weight_configs = Some(
            parameters
                .iter()
                .filter_map(|(name, parameter)| {
                    parameter
                        .linear_format
                        .weight_quantization()
                        .map(|format| (name.clone(), format))
                })
                .collect(),
        );
        let mut catalog = BTreeMap::new();
        for (name, parameter) in &parameters {
            let output = parameter.recipe.infer(source.as_ref())?;
            if output != parameter.output {
                return Err(BackbonePreparationError::Configuration(format!(
                    "retained recipe output disagrees for {name}"
                )));
            }
            let stored_dtype = match output.dtype {
                RecipeDtype::F16 => StoredDtype::F16,
                RecipeDtype::BF16 => StoredDtype::BF16,
                RecipeDtype::F32 => StoredDtype::F32,
                RecipeDtype::U32 => StoredDtype::U32,
                RecipeDtype::U8 => StoredDtype::U8,
                dtype => {
                    return Err(BackbonePreparationError::Configuration(format!(
                        "unsupported backbone output dtype {dtype:?} for {name}"
                    )));
                }
            };
            catalog.insert(
                name.clone(),
                CatalogTensorMetadata {
                    shape: output.shape,
                    stored_dtype,
                },
            );
        }
        let plan =
            nemotron_h::safetensors_plan(&args).map_err(BackbonePreparationError::Configuration)?;
        validate_safetensors_plan(&OutputCatalog(catalog), &plan).into_loader_result()?;
        let inputs = parameters
            .values()
            .flat_map(|parameter| {
                parameter
                    .recipe
                    .source_keys()
                    .into_iter()
                    .map(str::to_owned)
            })
            .collect();
        let source = Arc::new(RestrictedCheckpointSource::including(
            source,
            "voicechat/backbone/normalized-target",
            inputs,
        )?);
        Ok(Self {
            args,
            source,
            parameters,
        })
    }
    /// Existing Nemotron-H block, state and parallel geometry.
    pub fn args(&self) -> &ModelArgs {
        &self.args
    }
    /// The exact restricted source retained during artifact preparation.
    pub fn source(&self) -> &SharedCheckpointSource {
        &self.source
    }
    /// Runtime parameter names mapped to original admitted identities and bounds.
    pub fn parameters(&self) -> &BTreeMap<String, ParameterAdmission> {
        &self.parameters
    }
    /// Static embedding, final normalization and agent-head recipes.
    pub fn static_recipes(&self) -> BTreeMap<String, DerivedWeightRecipe> {
        self.parameters
            .iter()
            .filter(|(name, _)| !name.starts_with("model.layers."))
            .map(|(name, value)| (name.clone(), value.recipe.clone()))
            .collect()
    }
    /// Recipes for one existing independently materializable Nemotron-H block.
    pub fn layer_recipes(
        &self,
        index: usize,
    ) -> Result<BTreeMap<String, DerivedWeightRecipe>, BackbonePreparationError> {
        if index >= self.args.num_hidden_layers as usize {
            return Err(BackbonePreparationError::Configuration(
                "backbone layer is outside its schedule".into(),
            ));
        }
        let prefix = format!("model.layers.{index}.");
        Ok(self
            .parameters
            .iter()
            .filter(|(name, _)| name.starts_with(&prefix))
            .map(|(name, value)| (name.clone(), value.recipe.clone()))
            .collect())
    }
}

struct OutputCatalog(BTreeMap<String, CatalogTensorMetadata>);

impl SafetensorsCatalog for OutputCatalog {
    fn keys(&self) -> Vec<String> {
        self.0.keys().cloned().collect()
    }

    fn metadata(&self, key: &str) -> Result<CatalogTensorMetadata, String> {
        self.0.get(key).cloned().ok_or_else(|| key.into())
    }
}

pub(crate) fn project_parameters(
    args: &ModelArgs,
    parameters: &BTreeMap<String, ParameterAdmission>,
) -> Result<BTreeMap<String, ParameterAdmission>, BackbonePreparationError> {
    let mut output = BTreeMap::new();
    for (name, parameter) in parameters
        .iter()
        .filter(|(_, p)| p.component == Component::Backbone)
    {
        let conventional = if let Some(suffix) = name.strip_prefix("stt_model.embed_tokens.") {
            format!("backbone.embeddings.{suffix}")
        } else if let Some(suffix) = name.strip_prefix("stt_model.lm_head.") {
            format!("lm_head.{suffix}")
        } else {
            format!(
                "backbone.{}",
                name.strip_prefix("stt_model.llm.").ok_or_else(|| {
                    BackbonePreparationError::Configuration(format!(
                        "unknown backbone identity {name}"
                    ))
                })?
            )
        };
        let runtime = nemotron_h::checkpoint::normalized_parameter_key(&conventional, args)
            .map_err(BackbonePreparationError::Configuration)?;
        if output.insert(runtime.clone(), parameter.clone()).is_some() {
            return Err(BackbonePreparationError::Configuration(format!(
                "duplicate runtime parameter {runtime}"
            )));
        }
    }
    if output.is_empty() {
        return Err(BackbonePreparationError::MissingComponent);
    }
    // VoiceChat canonical recipes already restore A_log from split GGUF -exp(A_log).
    // Applying Nemotron's physical-storage recipe a second time would change it.
    Ok(output)
}

#[cfg(test)]
#[path = "backbone_tests.rs"]
mod tests;
