//! Retained VoiceChat weights projected onto the existing Nemotron-H parameter contract.

use super::{
    artifact::{ParameterAdmission, RetainedVoiceChatArtifacts},
    checkpoint::Component,
};
use crate::nemotron_h::{self, ModelArgs};
use eredu_checkpoint::{
    recipe::{DerivedWeightRecipe, RecipeDtype, RecipeError},
    store::{RestrictedCheckpointSource, SharedCheckpointSource, StoreError},
    validation::{validate_safetensors_plan, CatalogTensorMetadata, StrictLoadFailure},
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
    /// Shared canonical component preparation failed.
    #[error(transparent)]
    Component(#[from] eredu_runtime::ComponentPreparationError),
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
    declaration: Arc<nemotron_h::Declaration>,
    source: SharedCheckpointSource,
    parameters: BTreeMap<String, ParameterAdmission>,
}
impl PreparedBackboneParameters {
    /// Typed backbone scope shared by state admission and materialization.
    pub(crate) fn declaration(&self) -> &nemotron_h::Declaration {
        &self.declaration
    }

    /// Selects encoded recipes before parameter pool admission and module construction.
    pub fn with_quantization(
        mut self,
        quantization: eredu_checkpoint::WeightQuantization,
    ) -> Result<Self, BackbonePreparationError> {
        let args = nemotron_h::load_time_quantization(self.args(), quantization)
            .map_err(BackbonePreparationError::Configuration)?;
        let declaration = Arc::new(
            self.declaration
                .with_formats(args)
                .map_err(|e| BackbonePreparationError::Configuration(e.to_string()))?,
        );
        let plan = declaration
            .canonical_plan()
            .map_err(BackbonePreparationError::Configuration)?;
        let eligible = plan
            .common_tensors
            .iter()
            .filter_map(|tensor| {
                tensor
                    .key
                    .strip_suffix(".scales")
                    .map(|prefix| format!("{prefix}.weight"))
            })
            .filter(|name| self.parameters.contains_key(name))
            .collect();
        let (source, parameters) = super::quantization::project_parameters(
            self.source,
            &self.parameters,
            quantization,
            &eligible,
            RecipeDtype::F32,
        )?;
        self.declaration = declaration;
        self.source = source;
        self.parameters = parameters;
        Ok(self)
    }

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
        let released = released_declaration()?;
        let declaration = if &args == released.args() {
            Arc::clone(released)
        } else {
            Arc::new(
                nemotron_h::Declaration::new(args)
                    .map_err(|e| BackbonePreparationError::Configuration(e.to_string()))?,
            )
        };
        Self::from_parameters(declaration, source, artifacts.plan().parameters())
    }

    /// Admits explicit canonical component recipes against ordinary Nemotron geometry.
    ///
    /// Recipe outputs, naming, and the complete target catalog are checked without
    /// payload reads. Only the selected component's exact physical inputs remain
    /// visible, retaining source guards and any original storage transformations.
    pub fn from_parameters(
        declaration: Arc<nemotron_h::Declaration>,
        source: SharedCheckpointSource,
        parameters: &BTreeMap<String, ParameterAdmission>,
    ) -> Result<Self, BackbonePreparationError> {
        let mut args = declaration.args().clone();
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
        let declaration = Arc::new(
            declaration
                .with_formats(args)
                .map_err(|e| BackbonePreparationError::Configuration(e.to_string()))?,
        );
        let plan = declaration
            .canonical_plan()
            .map_err(BackbonePreparationError::Configuration)?;
        validate_safetensors_plan(&catalog, &plan).into_loader_result()?;
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
            declaration,
            source,
            parameters,
        })
    }

    /// Existing Nemotron-H block, state and parallel geometry.
    pub fn args(&self) -> &ModelArgs {
        self.declaration.args()
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
        if index >= self.args().num_hidden_layers as usize {
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

pub(crate) fn project_parameters(
    args: &ModelArgs,
    parameters: &BTreeMap<String, ParameterAdmission>,
) -> Result<BTreeMap<String, ParameterAdmission>, BackbonePreparationError> {
    let mut output = BTreeMap::new();
    for (name, parameter) in parameters
        .iter()
        .filter(|(_, p)| p.component == Component::Backbone)
    {
        let runtime =
            runtime_parameter_name(args, name).map_err(BackbonePreparationError::Configuration)?;
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

/// Converts the enclosing checkpoint namespace to the retained Nemotron identity.
pub(crate) fn runtime_parameter_name(args: &ModelArgs, name: &str) -> Result<String, String> {
    let conventional = if let Some(suffix) = name.strip_prefix("stt_model.embed_tokens.") {
        format!("backbone.embeddings.{suffix}")
    } else if let Some(suffix) = name.strip_prefix("stt_model.lm_head.") {
        format!("lm_head.{suffix}")
    } else {
        format!(
            "backbone.{}",
            name.strip_prefix("stt_model.llm.")
                .ok_or_else(|| format!("unknown backbone identity {name}"))?
        )
    };
    nemotron_h::checkpoint::normalized_parameter_key(&conventional, args)
}
/// Projects a declared runtime slot into the speech source namespace, checking the inverse.
pub(crate) fn checkpoint_parameter_name(args: &ModelArgs, name: &str) -> Result<String, String> {
    let source = if let Some(suffix) = name.strip_prefix("model.embeddings.") {
        format!("stt_model.embed_tokens.{suffix}")
    } else if let Some(suffix) = name.strip_prefix("lm_head.") {
        format!("stt_model.lm_head.{suffix}")
    } else if let Some(suffix) = name.strip_prefix("model.norm_f.") {
        format!("stt_model.llm.norm_f.{suffix}")
    } else if let Some(rest) = name.strip_prefix("model.layers.") {
        let (layer, suffix) = rest
            .split_once('.')
            .ok_or_else(|| format!("invalid backbone slot {name}"))?;
        let index = layer.parse::<usize>().map_err(|e| e.to_string())?;
        let policy = args
            .layer_schedule
            .get(index)
            .ok_or_else(|| format!("backbone slot {name} exceeds its schedule"))?;
        let operator = match policy {
            nemotron_h::LayerPolicy::Mamba => "mamba",
            nemotron_h::LayerPolicy::SelfAttention(_) => "attention",
            nemotron_h::LayerPolicy::DenseMlp => "mlp",
            nemotron_h::LayerPolicy::SparseMoe => "moe",
        };
        let suffix = if let Some(field) = suffix.strip_prefix(&format!("{operator}.")) {
            format!("mixer.{field}")
        } else {
            suffix.to_owned()
        };
        format!("stt_model.llm.layers.{layer}.{suffix}")
    } else {
        return Err(format!("unknown Nemotron target slot {name}"));
    };
    if runtime_parameter_name(args, &source)? != name {
        return Err(format!(
            "backbone slot {name} has an incompatible source identity"
        ));
    }
    Ok(source)
}

/// Source-free published target scope, shared by speech source extensions and construction.
pub(crate) fn released_declaration(
) -> Result<&'static Arc<nemotron_h::Declaration>, BackbonePreparationError> {
    static DECLARATION: std::sync::OnceLock<Result<Arc<nemotron_h::Declaration>, String>> =
        std::sync::OnceLock::new();
    DECLARATION
        .get_or_init(|| {
            let args = nemotron_h::model_args_from_config_value(
                &super::config::released_backbone_configuration(),
            )
            .map_err(|e| e.to_string())?;
            nemotron_h::Declaration::new(args)
                .map(Arc::new)
                .map_err(|e| e.to_string())
        })
        .as_ref()
        .map_err(|e| BackbonePreparationError::Configuration(e.clone()))
}
