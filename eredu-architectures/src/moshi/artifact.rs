//! Architecture-owned inspection and preparation of Moshi-family artifacts.

use std::{
    fs::File,
    path::{Component, Path, PathBuf},
};

use eredu_checkpoint::{
    recipe::{AtomicRecipeSet, RecipeCatalog},
    schema::SafetensorsCheckpointPlan,
    store::{SafetensorsWeightStore, StoreError},
    validation::{resolve_safetensors_plan, CheckpointValidation, SafetensorsCatalog},
};
use serde_json::Value;

use super::{canonical_recipes, safetensors_plan, CheckpointLayout, MoshiConfig, MoshiConfigError};

/// Fully inspected, backend-neutral input to realtime model materialization.
///
/// The plan fixes normalized architecture semantics, the physical checkpoint
/// source, the resolved strict checkpoint contract, and canonical binding
/// recipes before a concrete backend is selected.
#[derive(Debug, Clone)]
pub struct RealtimePreparationPlan {
    artifact_root: PathBuf,
    checkpoint_source: PathBuf,
    config: MoshiConfig,
    checkpoint_plan: SafetensorsCheckpointPlan,
    recipes: AtomicRecipeSet,
}

impl RealtimePreparationPlan {
    /// Submitted artifact directory used to name and identify source files.
    pub fn artifact_root(&self) -> &Path {
        &self.artifact_root
    }

    /// Architecture-selected SafeTensors file or indexed directory.
    pub fn checkpoint_source(&self) -> &Path {
        &self.checkpoint_source
    }

    /// Strictly normalized source architecture.
    pub fn config(&self) -> &MoshiConfig {
        &self.config
    }

    /// Strict checkpoint schema validated during inspection and revalidated at load.
    pub fn checkpoint_plan(&self) -> &SafetensorsCheckpointPlan {
        &self.checkpoint_plan
    }

    /// Architecture-declared canonical parameter recipes.
    pub fn recipes(&self) -> &AtomicRecipeSet {
        &self.recipes
    }

    /// Consumes the preparation into materialization inputs.
    pub fn into_parts(
        self,
    ) -> (
        PathBuf,
        PathBuf,
        MoshiConfig,
        SafetensorsCheckpointPlan,
        AtomicRecipeSet,
    ) {
        (
            self.artifact_root,
            self.checkpoint_source,
            self.config,
            self.checkpoint_plan,
            self.recipes,
        )
    }
}

/// Inspects and prepares one released Moshi-family artifact without a backend.
///
/// This reads configuration and SafeTensors metadata only. Tensor payloads are
/// left untouched for the selected backend to materialize.
pub fn prepare_realtime_model(
    artifact: impl AsRef<Path>,
) -> Result<RealtimePreparationPlan, RealtimePreparationError> {
    let artifact_root = artifact.as_ref();
    if !artifact_root.is_dir() {
        return Err(RealtimePreparationError::InvalidArtifact(format!(
            "Moshi artifact must be a directory, got {}",
            artifact_root.display()
        )));
    }
    let config_path = artifact_root.join("config.json");
    let config_value = if config_path.exists() {
        Some(serde_json::from_reader(File::open(&config_path)?)?)
    } else {
        None
    };
    let config = MoshiConfig::from_config_value(config_value.as_ref())?;
    let checkpoint_source = checkpoint_source(artifact_root, &config, config_value.as_ref())?;
    let store = SafetensorsWeightStore::open(&checkpoint_source)?;
    let checkpoint_plan =
        safetensors_plan(&config).map_err(RealtimePreparationError::InvalidArchitecture)?;
    let recipes =
        admit_checkpoint(&config, &checkpoint_plan, &store).map_err(|error| match error {
            CheckpointPreparationError::Schema(validation) => {
                RealtimePreparationError::InvalidCheckpoint(format!("{validation:?}"))
            }
            CheckpointPreparationError::Recipes(error) => {
                RealtimePreparationError::InvalidArchitecture(error)
            }
        })?;
    Ok(RealtimePreparationPlan {
        artifact_root: artifact_root.to_owned(),
        checkpoint_source,
        config,
        checkpoint_plan,
        recipes,
    })
}

/// Proves the strict physical schema and canonical recipe publication as one
/// architecture-owned checkpoint admission.
fn admit_checkpoint<C>(
    config: &MoshiConfig,
    checkpoint_plan: &SafetensorsCheckpointPlan,
    catalog: &C,
) -> Result<AtomicRecipeSet, CheckpointPreparationError>
where
    C: SafetensorsCatalog + RecipeCatalog + ?Sized,
{
    resolve_safetensors_plan(catalog, checkpoint_plan)
        .map_err(CheckpointPreparationError::Schema)?;
    canonical_recipes(config, catalog).map_err(CheckpointPreparationError::Recipes)
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum CheckpointPreparationError {
    #[error("checkpoint schema validation failed: {0:?}")]
    Schema(CheckpointValidation),
    #[error("canonical recipe validation failed: {0}")]
    Recipes(String),
}

fn checkpoint_source(
    artifact_root: &Path,
    config: &MoshiConfig,
    value: Option<&Value>,
) -> Result<PathBuf, RealtimePreparationError> {
    if artifact_root.join("model.safetensors.index.json").exists()
        || config.checkpoint_layout() == CheckpointLayout::PersonaPlexPytorch
    {
        return Ok(artifact_root.to_owned());
    }
    let name = value
        .and_then(Value::as_object)
        .and_then(|object| object.get("moshi_name"))
        .and_then(Value::as_str)
        .unwrap_or("model.safetensors");
    let name = Path::new(name);
    if name.components().count() != 1
        || !matches!(name.components().next(), Some(Component::Normal(_)))
    {
        return Err(RealtimePreparationError::InvalidArtifact(format!(
            "Moshi artifact filename must be a single relative component, got {name:?}"
        )));
    }
    Ok(artifact_root.join(name))
}

/// Failure while architecture code inspects a realtime artifact.
#[derive(Debug, thiserror::Error)]
pub enum RealtimePreparationError {
    /// The submitted directory or filename policy is invalid.
    #[error("invalid Moshi artifact: {0}")]
    InvalidArtifact(String),
    /// Configuration or architecture geometry is invalid.
    #[error("invalid Moshi architecture: {0}")]
    InvalidArchitecture(String),
    /// The checkpoint catalog does not satisfy the architecture contract.
    #[error("invalid Moshi checkpoint: {0}")]
    InvalidCheckpoint(String),
    /// Filesystem inspection failed.
    #[error(transparent)]
    Io(#[from] std::io::Error),
    /// Configuration JSON could not be decoded.
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    /// Moshi configuration normalization failed.
    #[error(transparent)]
    Config(#[from] MoshiConfigError),
    /// The neutral SafeTensors store rejected the source.
    #[error(transparent)]
    CheckpointStore(#[from] StoreError),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_checkpoint_filename_policy_is_architecture_owned_and_confined() {
        let root = Path::new("fixture");
        let config = MoshiConfig::native_v0_1().unwrap();
        let value = serde_json::json!({"moshi_name":"weights.safetensors"});
        assert_eq!(
            checkpoint_source(root, &config, Some(&value)).unwrap(),
            root.join("weights.safetensors")
        );
        for invalid in ["../weights.safetensors", "/weights.safetensors", ""] {
            let value = serde_json::json!({"moshi_name": invalid});
            assert!(checkpoint_source(root, &config, Some(&value)).is_err());
        }
    }

    #[test]
    fn personaplex_uses_the_indexed_artifact_directory() {
        let config =
            MoshiConfig::from_json(r#"{"model_type":"personaplex","version":"7b-v1"}"#).unwrap();
        assert_eq!(
            checkpoint_source(Path::new("fixture"), &config, None).unwrap(),
            Path::new("fixture")
        );
    }
}
