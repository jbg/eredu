//! Independently admitted BF16 image/video and waveform sources.
use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
    sync::Arc,
};

use super::{checkpoint, ConfigError, DecoderConfig, RoutedFeedForwardEquation};
use crate::{
    artifact_preparation::{ArtifactSourceRole, PhysicalArtifactPreparation},
    processor_plan::{MuseProcessorPlan, ProcessorPlanError},
};
use eredu_checkpoint::{
    artifact::{ArtifactFile, ArtifactFingerprintSource},
    recipe::{DerivedWeightRecipe, RecipeCatalog},
    safetensors::{checkpoint_member_path, SafetensorsMetadataCatalog, SafetensorsShardError},
    schema::SafetensorsCheckpointPlan,
    store::{open_prepared_safetensors_source, SharedCheckpointSource, StoreError, TensorMetadata},
    validation::{
        resolve_safetensors_plan, CatalogTensorMetadata, ResolvedCheckpointPlan, SafetensorsCatalog,
    },
};
use eredu_core::ArtifactCompanionRole;

/// A media source fails before its parameters can be bound.
#[derive(Debug, thiserror::Error)]
pub enum MediaCompanionError {
    /// The target has no independent Muse media-source contract.
    #[error("selected target does not admit independent Muse BF16 media companions")]
    UnsupportedTarget,
    /// No media role was requested.
    #[error("media companion admission requires a vision or audio role")]
    EmptySelection,
    /// The selected role does not identify an independent media tower.
    #[error("{0:?} is not an independent vision or audio companion role")]
    InvalidRole(ArtifactCompanionRole),
    /// The source omits the requested component.
    #[error("media companion {path} omits {roles:?}", path = .path.display())]
    MissingComponent {
        /// Selected source directory.
        path: PathBuf,
        /// Required media consumers.
        roles: Vec<ArtifactCompanionRole>,
    },
    /// The source does not describe the target decoder or its token namespace.
    #[error("media companion differs from its target at {field}")]
    Incompatible {
        /// First incompatible geometry or token field.
        field: &'static str,
    },
    /// Source geometry cannot be normalized.
    #[error(transparent)]
    Configuration(#[from] ConfigError),
    /// A source member or tensor header is invalid.
    #[error(transparent)]
    Checkpoint(#[from] SafetensorsShardError),
    /// Selected tensor layouts do not meet the component schema.
    #[error("invalid BF16 media companion catalog: {0}")]
    Catalog(String),
    /// A retained source could not be bound.
    #[error(transparent)]
    Source(#[from] StoreError),
    /// The processor does not match the selected tower.
    #[error(transparent)]
    Processor(#[from] ProcessorPlanError),
    /// A source asset could not be read.
    #[error("cannot read media companion asset {path}: {source}", path = .path.display())]
    Io {
        /// Asset whose read failed.
        path: PathBuf,
        /// Original host I/O failure.
        #[source]
        source: std::io::Error,
    },
    /// An admitted source member changed.
    #[error("media companion source identity changed: {0}")]
    Changed(String),
}

/// Immutable component configuration, exact sources and canonical recipes.
/// Physical catalogs retain complete members; the resolution exposes only the
/// selected media keys, including when decoder tensors share the same files.
#[derive(Debug, Clone)]
pub struct Bf16MediaCompanion {
    config: DecoderConfig,
    processor: MuseProcessorPlan,
    catalog: SafetensorsMetadataCatalog,
    schema: SafetensorsCheckpointPlan,
    resolution: ResolvedCheckpointPlan,
    recipes: BTreeMap<String, DerivedWeightRecipe>,
    roles: BTreeMap<ArtifactSourceRole, BTreeSet<String>>,
    assets: Vec<(ArtifactFile, Arc<[u8]>)>,
    files: Vec<ArtifactFile>,
    guards: Vec<Arc<ArtifactFingerprintSource>>,
}

impl Bf16MediaCompanion {
    /// Admits explicit component directories, sharing one physical source when
    /// both roles resolve to the same directory. An empty selection opens nothing.
    pub fn admit_selected(
        target: &DecoderConfig,
        sources: &eredu_core::MediaCompanionSources,
    ) -> Result<Vec<Self>, MediaCompanionError> {
        let mut grouped = BTreeMap::<PathBuf, BTreeSet<ArtifactCompanionRole>>::new();
        for (role, path) in [
            (ArtifactCompanionRole::Vision, sources.vision()),
            (ArtifactCompanionRole::Audio, sources.audio()),
        ] {
            if let Some(path) = path {
                let canonical = path
                    .canonicalize()
                    .map_err(|source| MediaCompanionError::Io {
                        path: path.into(),
                        source,
                    })?;
                grouped.entry(canonical).or_default().insert(role);
            }
        }
        grouped
            .into_iter()
            .map(|(path, roles)| Self::admit(target, path, roles))
            .collect()
    }

    /// Admits selected BF16 towers without acquiring tensor payloads. Config and
    /// processor assets use the same access roots as checkpoint payloads.
    pub fn admit(
        target: &DecoderConfig,
        path: impl AsRef<Path>,
        selected: impl IntoIterator<Item = ArtifactCompanionRole>,
    ) -> Result<Self, MediaCompanionError> {
        let mut requested = BTreeSet::new();
        for role in selected {
            requested.insert(match role {
                ArtifactCompanionRole::Vision => ArtifactSourceRole::Vision,
                ArtifactCompanionRole::Audio => ArtifactSourceRole::Audio,
                role => return Err(MediaCompanionError::InvalidRole(role)),
            });
        }
        let root = path.as_ref();
        let missing = || MediaCompanionError::MissingComponent {
            path: root.to_path_buf(),
            roles: requested.iter().map(public_role).collect(),
        };
        if requested.is_empty() {
            return Err(MediaCompanionError::EmptySelection);
        }
        let prefix = match (
            requested.contains(&ArtifactSourceRole::Vision),
            requested.contains(&ArtifactSourceRole::Audio),
        ) {
            (true, true) => "media/vision-audio",
            (true, false) => "media/vision",
            _ => "media/audio",
        };
        let config_path = checkpoint_member_path(root, Path::new("config.json"))?;
        let processor_path = checkpoint_member_path(root, Path::new("processor_config.json"))?;
        let mut asset_files = vec![
            ArtifactFile::new(format!("{prefix}/config"), &config_path),
            ArtifactFile::new(format!("{prefix}/processor"), &processor_path),
        ];
        if root.join("model.safetensors.index.json").exists() {
            asset_files.push(ArtifactFile::new(
                format!("{prefix}/index"),
                checkpoint_member_path(root, Path::new("model.safetensors.index.json"))?,
            ));
        }
        let asset_guard = Arc::new(
            ArtifactFingerprintSource::new(asset_files.clone())
                .map_err(|e| MediaCompanionError::Changed(e.to_string()))?,
        );
        let mut assets = Vec::new();
        for file in &asset_files {
            let bytes = std::fs::read(file.path()).map_err(|source| MediaCompanionError::Io {
                path: file.path().into(),
                source,
            })?;
            assets.push((file.clone(), Arc::<[u8]>::from(bytes)));
        }
        let mut config = DecoderConfig::from_hf_json(&assets[0].1)?;
        compatible_decoder(target, &config, &requested)?;
        if requested.contains(&ArtifactSourceRole::Vision) {
            let vision = config.vision_config.as_mut().ok_or_else(missing)?;
            vision.weight_quantization = None;
            vision.quantized_weight_configs.clear();
        } else {
            config.vision_config = None;
        }
        if requested.contains(&ArtifactSourceRole::Audio) {
            config
                .audio_config
                .as_mut()
                .ok_or_else(missing)?
                .linear_formats
                .clear();
        } else {
            config.audio_config = None;
        }
        let processor = MuseProcessorPlan::from_hf_json(&assets[1].1)?;
        processor.validate_components(&config)?;
        let catalog = SafetensorsMetadataCatalog::discover_members_matching(root, |name| {
            checkpoint::media_source_role(name).is_some_and(|role| requested.contains(&role))
        })
        .map_err(|e| {
            if e == SafetensorsShardError::EmptyTensorSelection {
                missing()
            } else {
                e.into()
            }
        })?;
        let keys = catalog
            .tensors()
            .keys()
            .filter(|name| {
                checkpoint::media_source_role(name).is_some_and(|role| requested.contains(&role))
            })
            .cloned()
            .collect();
        let view = MediaCatalog {
            catalog: &catalog,
            keys,
        };
        let schema = checkpoint::bf16_media_plan(&config, &requested)
            .map_err(MediaCompanionError::Catalog)?;
        let resolution = resolve_safetensors_plan(&view, &schema)
            .map_err(|e| MediaCompanionError::Catalog(format!("{e:?}")))?;
        let recipes = checkpoint::bf16_media_recipes(&config, &requested, &view)
            .map_err(MediaCompanionError::Catalog)?;
        let roles = checkpoint::media_source_roles(resolution.source_keys());
        let payload_files = catalog
            .shards()
            .logical_payload_paths()
            .iter()
            .map(|(role, path)| ArtifactFile::new(format!("{prefix}/{role}"), path))
            .collect::<Vec<_>>();
        let payload_guard = Arc::new(
            ArtifactFingerprintSource::new(payload_files.clone())
                .map_err(|e| MediaCompanionError::Changed(e.to_string()))?,
        );
        let mut files = asset_files;
        files.extend(payload_files);
        let admitted = Self {
            config,
            processor,
            catalog,
            schema,
            resolution,
            recipes,
            roles,
            assets,
            files,
            guards: vec![asset_guard, payload_guard],
        };
        admitted.verify_unchanged()?;
        Ok(admitted)
    }

    /// Source-specific component geometry and arithmetic, without unselected towers.
    pub fn configuration(&self) -> &DecoderConfig {
        &self.config
    }
    /// Processor limits and framing from this source's retained asset.
    pub fn processor(&self) -> &MuseProcessorPlan {
        &self.processor
    }
    /// Exact selected physical names for each logical media consumer.
    pub fn roles(&self) -> &BTreeMap<ArtifactSourceRole, BTreeSet<String>> {
        &self.roles
    }
    /// Component-only canonical recipes, excluding co-resident decoder weights.
    pub fn recipes(&self) -> &BTreeMap<String, DerivedWeightRecipe> {
        &self.recipes
    }
    /// Retained source members for content identity and resolution guards.
    pub fn files(&self) -> &[ArtifactFile] {
        &self.files
    }
    /// Complete component schema validated at admission.
    pub fn schema(&self) -> &SafetensorsCheckpointPlan {
        &self.schema
    }
    pub(crate) fn tensor_catalog(
        &self,
    ) -> Result<eredu_core::checkpoint::TensorCatalog, MediaCompanionError> {
        use eredu_core::checkpoint::{TensorCatalog, TensorDescriptor, TensorDtype, TensorStorage};
        TensorCatalog::new(self.resolution.source_keys().iter().map(|name| {
            let metadata = &self.catalog.tensors()[name];
            TensorDescriptor {
                name: name.clone(),
                shape: metadata.logical_shape.clone(),
                dtype: TensorDtype::Bf16,
                storage: metadata.backing_shard.as_ref().map(|path| TensorStorage {
                    member: path.to_string_lossy().into_owned(),
                    offset: self
                        .catalog
                        .tensor_offset(name)
                        .expect("admitted component offset"),
                    length: metadata.encoded_byte_len,
                }),
            }
        }))
        .map_err(|e| MediaCompanionError::Catalog(e.to_string()))
    }

    /// Exact physical authority consumed by ordinary prepared construction.
    pub fn preparation(&self) -> PhysicalArtifactPreparation {
        PhysicalArtifactPreparation::Safetensors {
            catalog: self.catalog.tensors().clone(),
            shards: self.catalog.admitted_shards(),
            resolution: self.resolution.clone(),
        }
    }
    /// Opens the retained component resolution without repeating artifact discovery.
    pub fn bind_source(
        &self,
        readers: usize,
    ) -> Result<SharedCheckpointSource, MediaCompanionError> {
        self.verify_unchanged()?;
        let source = open_prepared_safetensors_source(
            self.catalog.admitted_shards(),
            self.catalog.tensors().clone(),
            self.resolution.clone(),
            readers,
        )?;
        self.verify_unchanged()?;
        Ok(source)
    }
    /// Checks source continuity before content identity or prepared construction.
    pub fn verify_unchanged(&self) -> Result<(), MediaCompanionError> {
        for guard in &self.guards {
            guard
                .validate_unchanged()
                .map_err(|e| MediaCompanionError::Changed(e.to_string()))?;
        }
        for (file, expected) in &self.assets {
            let bytes = std::fs::read(file.path()).map_err(|source| MediaCompanionError::Io {
                path: file.path().into(),
                source,
            })?;
            if bytes.as_slice() != expected.as_ref() {
                return Err(MediaCompanionError::Changed(
                    file.path().display().to_string(),
                ));
            }
        }
        for guard in &self.guards {
            guard
                .validate_unchanged()
                .map_err(|e| MediaCompanionError::Changed(e.to_string()))?;
        }
        Ok(())
    }
}

fn public_role(role: &ArtifactSourceRole) -> ArtifactCompanionRole {
    match role {
        ArtifactSourceRole::Vision => ArtifactCompanionRole::Vision,
        ArtifactSourceRole::Audio => ArtifactCompanionRole::Audio,
        _ => unreachable!("validated media role"),
    }
}

fn compatible_decoder(
    target: &DecoderConfig,
    source: &DecoderConfig,
    roles: &BTreeSet<ArtifactSourceRole>,
) -> Result<(), MediaCompanionError> {
    macro_rules! same { ($($field:ident),+ $(,)?) => { $(if target.$field != source.$field {
        return Err(MediaCompanionError::Incompatible { field: stringify!($field) });
    })+ }; }
    same!(
        identity,
        hidden_size,
        vocab_size,
        num_hidden_layers,
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        num_experts,
        num_experts_per_tok,
        moe_intermediate_size,
        attention_schedule,
        layer_uses_rope
    );
    if target.num_experts == 0 {
        same!(intermediate_size);
    }
    let expert_width = |config: &DecoderConfig| match config.routed_feed_forward {
        RoutedFeedForwardEquation::CompressedSigmoid { expert_width, .. } => Some(expert_width),
        _ => None,
    };
    if expert_width(target) != expert_width(source) {
        return Err(MediaCompanionError::Incompatible {
            field: "expert_width",
        });
    }
    if roles.contains(&ArtifactSourceRole::Vision) {
        same!(image_token_id, video_token_id);
    }
    if roles.contains(&ArtifactSourceRole::Audio) {
        same!(audio_token_id);
    }
    Ok(())
}

struct MediaCatalog<'a> {
    catalog: &'a SafetensorsMetadataCatalog,
    keys: BTreeSet<String>,
}
impl RecipeCatalog for MediaCatalog<'_> {
    fn tensor_metadata(&self, key: &str) -> Result<TensorMetadata, StoreError> {
        if !self.keys.contains(key) {
            return Err(StoreError::UnknownTensor { key: key.into() });
        }
        RecipeCatalog::tensor_metadata(self.catalog, key)
    }
}
impl SafetensorsCatalog for MediaCatalog<'_> {
    fn keys(&self) -> Vec<String> {
        self.keys.iter().cloned().collect()
    }
    fn metadata(&self, key: &str) -> Result<CatalogTensorMetadata, String> {
        if !self.keys.contains(key) {
            return Err(format!(
                "tensor {key:?} is outside the selected media source"
            ));
        }
        SafetensorsCatalog::metadata(self.catalog, key)
    }
}
