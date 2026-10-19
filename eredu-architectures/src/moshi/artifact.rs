//! Architecture-owned inspection and preparation of Moshi-family artifacts.

use std::{
    collections::BTreeMap,
    path::{Component, Path, PathBuf},
};

use eredu_checkpoint::{
    recipe::{AtomicRecipeSet, RecipeCatalog, RecipeMetadata},
    safetensors::{SafetensorsMetadataCatalog, SafetensorsShardError, SafetensorsShards},
    schema::SafetensorsCheckpointPlan,
    store::{StoreError, TensorMetadata},
    validation::{
        resolve_safetensors_plan, CheckpointValidation, ResolvedCheckpointPlan, SafetensorsCatalog,
    },
};
use serde_json::Value;
use sha2::{Digest, Sha256};

use super::{canonical_recipes, safetensors_plan, CheckpointLayout, MoshiConfig, MoshiConfigError};

/// Fully inspected, backend-neutral input to realtime model materialization.
///
/// The plan fixes normalized architecture semantics, the physical checkpoint
/// source, the resolved strict checkpoint contract, and canonical binding
/// recipes before a concrete backend is selected.
#[derive(Debug, Clone)]
pub struct MoshiPreparationPlan {
    artifact_root: PathBuf,
    checkpoint_source: PathBuf,
    config: MoshiConfig,
    checkpoint_plan: SafetensorsCheckpointPlan,
    resolved_checkpoint_plan: ResolvedCheckpointPlan,
    source_metadata: BTreeMap<String, TensorMetadata>,
    recipes: AtomicRecipeSet,
    recipe_outputs: BTreeMap<String, RecipeMetadata>,
    admitted_shards: Option<SafetensorsShards>,
    metadata_contract_identity: String,
}

impl MoshiPreparationPlan {
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

    /// Exact physical layout selected by strict checkpoint admission.
    pub fn resolved_checkpoint_plan(&self) -> &ResolvedCheckpointPlan {
        &self.resolved_checkpoint_plan
    }

    /// Catalog metadata for every exact selected physical source.
    ///
    /// Backing-shard provenance is retained when the catalog supplies it. This
    /// preparation never invents missing shard data; the checkpoint source and
    /// resolved plan retain the exact artifact and source identities instead.
    pub fn source_metadata(&self) -> &BTreeMap<String, TensorMetadata> {
        &self.source_metadata
    }

    /// Architecture-declared canonical parameter recipes.
    pub fn recipes(&self) -> &AtomicRecipeSet {
        &self.recipes
    }

    /// Inferred output metadata for every canonical recipe.
    pub fn recipe_outputs(&self) -> &BTreeMap<String, RecipeMetadata> {
        &self.recipe_outputs
    }

    /// Strictly discovered filesystem shards, when preparation used a path.
    ///
    /// Source-generic metadata catalogs intentionally return `None`; their
    /// eventual payload source is supplied by the generic constructor instead.
    pub const fn admitted_shards(&self) -> Option<&SafetensorsShards> {
        self.admitted_shards.as_ref()
    }

    /// Relocation-stable identity of admitted configuration and header metadata.
    ///
    /// This is deliberately not a payload-content digest. Payload identity is
    /// attached only after selected construction is allowed to read payloads.
    pub fn metadata_contract_identity(&self) -> &str {
        &self.metadata_contract_identity
    }

    /// Consumes the preparation into a named materialization artifact.
    pub fn into_artifact(self) -> MoshiPreparationArtifact {
        MoshiPreparationArtifact {
            artifact_root: Some(self.artifact_root),
            checkpoint_source: Some(self.checkpoint_source),
            config: Some(self.config),
            checkpoint_plan: Some(self.checkpoint_plan),
            resolved_checkpoint_plan: Some(self.resolved_checkpoint_plan),
            source_metadata: Some(self.source_metadata),
            recipes: Some(self.recipes),
            recipe_outputs: Some(self.recipe_outputs),
            admitted_shards: self.admitted_shards,
            metadata_contract_identity: Some(self.metadata_contract_identity),
        }
    }
}

/// Named consuming artifact for realtime materialization inputs.
pub struct MoshiPreparationArtifact {
    artifact_root: Option<PathBuf>,
    checkpoint_source: Option<PathBuf>,
    config: Option<MoshiConfig>,
    checkpoint_plan: Option<SafetensorsCheckpointPlan>,
    resolved_checkpoint_plan: Option<ResolvedCheckpointPlan>,
    source_metadata: Option<BTreeMap<String, TensorMetadata>>,
    recipes: Option<AtomicRecipeSet>,
    recipe_outputs: Option<BTreeMap<String, RecipeMetadata>>,
    admitted_shards: Option<SafetensorsShards>,
    metadata_contract_identity: Option<String>,
}

impl MoshiPreparationArtifact {
    /// Takes the submitted artifact root exactly once.
    pub fn take_artifact_root(&mut self) -> PathBuf {
        self.artifact_root
            .take()
            .expect("artifact root already taken")
    }
    /// Takes the physical checkpoint source exactly once.
    pub fn take_checkpoint_source(&mut self) -> PathBuf {
        self.checkpoint_source
            .take()
            .expect("checkpoint source already taken")
    }
    /// Takes normalized architecture configuration exactly once.
    pub fn take_config(&mut self) -> MoshiConfig {
        self.config.take().expect("realtime config already taken")
    }
    /// Takes the strict checkpoint plan exactly once.
    pub fn take_checkpoint_plan(&mut self) -> SafetensorsCheckpointPlan {
        self.checkpoint_plan
            .take()
            .expect("checkpoint plan already taken")
    }
    /// Takes the exact resolved physical checkpoint layout exactly once.
    pub fn take_resolved_checkpoint_plan(&mut self) -> ResolvedCheckpointPlan {
        self.resolved_checkpoint_plan
            .take()
            .expect("resolved checkpoint plan already taken")
    }
    /// Takes selected physical source metadata exactly once.
    pub fn take_source_metadata(&mut self) -> BTreeMap<String, TensorMetadata> {
        self.source_metadata
            .take()
            .expect("checkpoint source metadata already taken")
    }
    /// Takes canonical binding recipes exactly once.
    pub fn take_recipes(&mut self) -> AtomicRecipeSet {
        self.recipes.take().expect("realtime recipes already taken")
    }
    /// Takes inferred canonical recipe output metadata exactly once.
    pub fn take_recipe_outputs(&mut self) -> BTreeMap<String, RecipeMetadata> {
        self.recipe_outputs
            .take()
            .expect("realtime recipe outputs already taken")
    }
    /// Takes strictly admitted filesystem shards when path preparation supplied them.
    pub fn take_admitted_shards(&mut self) -> Option<SafetensorsShards> {
        self.admitted_shards.take()
    }
    /// Takes the admitted metadata-contract identity exactly once.
    pub fn take_metadata_contract_identity(&mut self) -> String {
        self.metadata_contract_identity
            .take()
            .expect("metadata contract identity already taken")
    }
}

/// Inspects and prepares one released Moshi-family artifact without a backend.
///
/// This reads configuration and SafeTensors metadata only. Tensor payloads are
/// left untouched for the selected backend to materialize.
pub(crate) fn prepare_artifact(
    artifact: impl AsRef<Path>,
    config_value: Option<&Value>,
) -> Result<MoshiPreparationPlan, MoshiPreparationError> {
    let artifact = artifact.as_ref();
    let artifact_root = if artifact.is_dir() {
        artifact
    } else if artifact.is_file() {
        artifact.parent().unwrap_or(Path::new("."))
    } else {
        return Err(MoshiPreparationError::InvalidArtifact(format!(
            "Moshi artifact must be an existing directory or SafeTensors file, got {}",
            artifact.display()
        )));
    };
    let config = MoshiConfig::from_config_value(config_value)?;
    let checkpoint_source = if artifact.is_file() {
        artifact.to_owned()
    } else {
        checkpoint_source(artifact_root, &config, config_value)?
    };
    let catalog = SafetensorsMetadataCatalog::discover(&checkpoint_source)?;
    let admitted_shards = catalog.admitted_shards();
    let mut preparation =
        prepare_moshi_from_catalog(artifact_root, &checkpoint_source, config, &catalog)?;
    preparation.admitted_shards = Some(admitted_shards);
    Ok(preparation)
}

/// Prepares one normalized Moshi-family artifact from an exact metadata catalog.
///
/// `artifact_root` and `checkpoint_source` are retained as the authoritative
/// artifact identities. The catalog is used only for strict SafeTensors schema
/// admission and canonical recipe publication; this function has no tensor
/// payload-reading capability.
pub fn prepare_moshi_from_catalog<C>(
    artifact_root: impl AsRef<Path>,
    checkpoint_source: impl AsRef<Path>,
    config: MoshiConfig,
    catalog: &C,
) -> Result<MoshiPreparationPlan, MoshiPreparationError>
where
    C: SafetensorsCatalog + RecipeCatalog + ?Sized,
{
    let checkpoint_plan =
        safetensors_plan(&config).map_err(MoshiPreparationError::InvalidArchitecture)?;
    let admitted =
        admit_checkpoint(&config, &checkpoint_plan, catalog).map_err(|error| match error {
            CheckpointPreparationError::Schema(validation) => {
                MoshiPreparationError::InvalidCheckpoint(format!("{validation:?}"))
            }
            CheckpointPreparationError::Metadata(error) => {
                MoshiPreparationError::InvalidCheckpoint(error)
            }
            CheckpointPreparationError::Recipes(error) => {
                MoshiPreparationError::InvalidArchitecture(error)
            }
        })?;
    let metadata_contract_identity = metadata_contract_identity(&config, &admitted);
    Ok(MoshiPreparationPlan {
        artifact_root: artifact_root.as_ref().to_owned(),
        checkpoint_source: checkpoint_source.as_ref().to_owned(),
        config,
        checkpoint_plan,
        resolved_checkpoint_plan: admitted.resolved_checkpoint_plan,
        source_metadata: admitted.source_metadata,
        recipes: admitted.recipes,
        recipe_outputs: admitted.recipe_outputs,
        admitted_shards: None,
        metadata_contract_identity,
    })
}

struct AdmittedCheckpoint {
    resolved_checkpoint_plan: ResolvedCheckpointPlan,
    source_metadata: BTreeMap<String, TensorMetadata>,
    recipes: AtomicRecipeSet,
    recipe_outputs: BTreeMap<String, RecipeMetadata>,
}

fn metadata_contract_identity(config: &MoshiConfig, admitted: &AdmittedCheckpoint) -> String {
    fn component(hasher: &mut Sha256, value: impl AsRef<[u8]>) {
        let value = value.as_ref();
        hasher.update((value.len() as u64).to_le_bytes());
        hasher.update(value);
    }

    let mut hasher = Sha256::new();
    component(&mut hasher, b"eredu-moshi-metadata-contract-v1");
    component(&mut hasher, config.architecture_fingerprint());
    component(&mut hasher, admitted.resolved_checkpoint_plan.identity());
    for (name, metadata) in &admitted.source_metadata {
        component(&mut hasher, name);
        component(&mut hasher, format!("{:?}", metadata.logical_shape));
        component(&mut hasher, format!("{:?}", metadata.physical_shape));
        component(&mut hasher, format!("{:?}", metadata.stored_dtype));
        component(&mut hasher, metadata.encoded_byte_len.to_le_bytes());
        component(
            &mut hasher,
            metadata
                .backing_shard
                .as_deref()
                .and_then(Path::file_name)
                .map(|name| name.to_string_lossy())
                .as_deref()
                .unwrap_or("metadata-source"),
        );
    }
    for (target, recipe) in admitted.recipes.iter() {
        component(&mut hasher, target);
        component(&mut hasher, format!("{recipe:?}"));
        let output = admitted
            .recipe_outputs
            .get(target)
            .expect("admitted recipe has inferred metadata");
        component(&mut hasher, format!("{:?}", output.shape));
        component(&mut hasher, format!("{:?}", output.dtype));
        component(&mut hasher, output.byte_len.to_le_bytes());
    }
    for (alias, owner) in admitted.recipes.aliases() {
        component(&mut hasher, alias);
        component(&mut hasher, owner);
    }
    let digest = hasher.finalize();
    let mut encoded = String::with_capacity(71);
    encoded.push_str("sha256:");
    const HEX: &[u8; 16] = b"0123456789abcdef";
    for byte in digest {
        encoded.push(HEX[usize::from(byte >> 4)] as char);
        encoded.push(HEX[usize::from(byte & 0x0f)] as char);
    }
    encoded
}

/// Proves the strict physical schema and canonical recipe publication as one
/// architecture-owned checkpoint admission.
fn admit_checkpoint<C>(
    config: &MoshiConfig,
    checkpoint_plan: &SafetensorsCheckpointPlan,
    catalog: &C,
) -> Result<AdmittedCheckpoint, CheckpointPreparationError>
where
    C: SafetensorsCatalog + RecipeCatalog + ?Sized,
{
    let resolved_checkpoint_plan = resolve_safetensors_plan(catalog, checkpoint_plan)
        .map_err(CheckpointPreparationError::Schema)?;
    let source_metadata = resolved_checkpoint_plan
        .source_keys()
        .iter()
        .map(|key| {
            let admitted = catalog.metadata(key).map_err(|error| {
                CheckpointPreparationError::Metadata(format!(
                    "selected checkpoint source {key:?} admission metadata is unavailable: {error}"
                ))
            })?;
            let retained = catalog
                .tensor_metadata(key)
                .map_err(|error| {
                    CheckpointPreparationError::Metadata(format!(
                        "selected checkpoint source {key:?} metadata is unavailable: {error}"
                    ))
                })?;
            if retained.name != *key
                || retained.logical_shape != admitted.shape
                || retained.stored_dtype != admitted.stored_dtype
            {
                return Err(CheckpointPreparationError::Metadata(format!(
                    "selected checkpoint source {key:?} retained metadata differs from admitted metadata"
                )));
            }
            Ok((key.clone(), retained))
        })
        .collect::<Result<_, _>>()?;
    let recipes =
        canonical_recipes(config, catalog).map_err(CheckpointPreparationError::Recipes)?;
    let recipe_outputs = recipes
        .iter()
        .map(|(target, recipe)| {
            recipe
                .infer(catalog)
                .map(|metadata| (target.to_owned(), metadata))
                .map_err(|error| {
                    CheckpointPreparationError::Recipes(format!(
                        "canonical recipe {target:?} metadata inference failed: {error}"
                    ))
                })
        })
        .collect::<Result<_, _>>()?;
    Ok(AdmittedCheckpoint {
        resolved_checkpoint_plan,
        source_metadata,
        recipes,
        recipe_outputs,
    })
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum CheckpointPreparationError {
    #[error("checkpoint schema validation failed: {0:?}")]
    Schema(CheckpointValidation),
    #[error("checkpoint metadata retention failed: {0}")]
    Metadata(String),
    #[error("canonical recipe validation failed: {0}")]
    Recipes(String),
}

fn checkpoint_source(
    artifact_root: &Path,
    config: &MoshiConfig,
    value: Option<&Value>,
) -> Result<PathBuf, MoshiPreparationError> {
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
        return Err(MoshiPreparationError::InvalidArtifact(format!(
            "Moshi artifact filename must be a single relative component, got {name:?}"
        )));
    }
    Ok(artifact_root.join(name))
}

/// Failure while architecture code inspects a realtime artifact.
#[derive(Debug, thiserror::Error)]
pub enum MoshiPreparationError {
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
    /// Strict filesystem shard discovery or header admission failed.
    #[error(transparent)]
    CheckpointShards(#[from] SafetensorsShardError),
}
