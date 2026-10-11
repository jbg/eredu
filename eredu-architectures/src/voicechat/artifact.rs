//! Header admission and exact component source retention for VoiceChat.
//!
//! Admission reads metadata. Retention binds the admitted stores once; callers
//! consume restricted views and canonical recipes without rediscovering files.
//! This is a storage contract, not executable-model construction.

use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
    sync::Arc,
};

use super::{
    assets::PreparedVoiceChatAssets,
    checkpoint::{self, Component, GgufLayout},
    config::{ConfigError, VoiceChatConfig},
};
use crate::artifact_preparation::{
    ArtifactSourceDeclarations, ArtifactSourceRole, BoundArtifactSources, PhysicalArtifactId,
    PhysicalArtifactPreparation,
};
use eredu_checkpoint::{
    gguf_store::GgufCatalog,
    recipe::{DerivedWeightRecipe, RecipeCatalog, RecipeMetadata},
    safetensors::SafetensorsMetadataCatalog,
    store::{
        CheckpointLease, CheckpointSource, EncodedTensorLease, ReadPolicy, SharedCheckpointSource,
        StoreError, TensorReadRequest, TensorSelection,
    },
    validation::{resolve_gguf_plan, resolve_safetensors_plan},
};
use eredu_core::{artifact::ArtifactError, GgufCompanionRole};

/// A rejected source graph. No native resources are allocated during admission.
#[derive(Debug, thiserror::Error)]
pub enum AdmissionError {
    /// Configuration does not describe the released equations.
    #[error(transparent)]
    Configuration(#[from] ConfigError),
    /// Integer geometry is valid but released buffer semantics do not match.
    #[error(transparent)]
    Integer(#[from] super::controls::IntegerBufferError),
    /// Saved host feature coefficients are malformed or unreadable.
    #[error(transparent)]
    Preprocessing(#[from] super::preprocessing::PreprocessingPreparationError),
    /// Auxiliary tokenizers disagree with embedded metadata.
    #[error(transparent)]
    Assets(#[from] super::assets::VoiceChatAssetError),
    /// Header or recipe violates the architecture's storage contract.
    #[error("invalid VoiceChat artifact: {0}")]
    InvalidArtifact(String),
    /// An exact source store could not be retained.
    #[error(transparent)]
    Source(#[from] ArtifactError),
    /// SafeTensors metadata admission retains the original filesystem error.
    #[error(transparent)]
    SafeTensors(#[from] eredu_checkpoint::safetensors::SafetensorsShardError),
    /// GGUF metadata admission retains the original parser or filesystem error.
    #[error(transparent)]
    Gguf(#[from] eredu_gguf::Error),
    /// Complete typed tensor-schema findings.
    #[error("VoiceChat checkpoint schema mismatch: {0:?}")]
    Schema(eredu_checkpoint::validation::CheckpointValidation),
    /// Canonical recipe geometry or bounded conversion admission failed.
    #[error(transparent)]
    Recipe(#[from] eredu_checkpoint::recipe::RecipeError),
    /// Metadata or a bounded payload lease failed.
    #[error(transparent)]
    Store(#[from] StoreError),
    /// More than one primary source provides the same canonical parameter.
    #[error("duplicate VoiceChat parameter {0}")]
    DuplicateParameter(String),
    /// Selected roles require an explicit additional source.
    #[error("VoiceChat component {component:?} is missing parameters: {keys:?}")]
    MissingComponent {
        /// Selected component lacking complete canonical inputs.
        component: Component,
        /// Exact absent canonical parameter identities.
        keys: Vec<String>,
    },
}

/// One canonical parameter with explicit conversion accounting.
#[derive(Debug, Clone)]
pub struct ParameterAdmission {
    /// Canonical released identity.
    pub name: String,
    /// Physical projection format; companions and non-matrix parameters are dense.
    pub linear_format: eredu_checkpoint::LinearFormat,
    /// Architecture component owning the parameter.
    pub component: Component,
    /// Physical source authority used by its recipe.
    pub artifact: PhysicalArtifactId,
    /// Canonical shape and scalar representation after conversion.
    pub output: RecipeMetadata,
    /// Original encoded bytes of the unique physical inputs.
    pub stored_bytes: u64,
    /// Additional decoded F32 vector retained while the converter creates output bytes.
    pub conversion_workspace_bytes: u64,
    /// Conservative full-parameter bound including encoded input and recipe temporaries.
    /// Bounded selections are sized from their selected recipes at materialization.
    pub materialization_peak_bytes: u64,
    /// Portable transform from the restricted component source to canonical layout.
    pub recipe: DerivedWeightRecipe,
}

/// One admitted physical artifact, before selecting component roles.
#[derive(Debug, Clone)]
pub struct AdmittedVoiceChatSource {
    config: VoiceChatConfig,
    physical: PhysicalArtifactPreparation,
    parameters: BTreeMap<String, ParameterAdmission>,
    layout: Option<GgufLayout>,
}

impl AdmittedVoiceChatSource {
    /// Admits the released dense or affine SafeTensors catalog and retained shards.
    pub fn safetensors(path: &Path, config: &VoiceChatConfig) -> Result<Self, AdmissionError> {
        let catalog = SafetensorsMetadataCatalog::discover(path)?;
        let formats = checkpoint::safetensors_formats(config, &catalog)
            .map_err(AdmissionError::InvalidArtifact)?;
        for (name, declared) in &config.linear_formats {
            if formats
                .get(name)
                .copied()
                .unwrap_or(eredu_checkpoint::LinearFormat::Dense)
                != *declared
            {
                return Err(AdmissionError::InvalidArtifact(format!(
                    "declared projection format disagrees with physical storage for {name}"
                )));
            }
        }
        let plan = checkpoint::safetensors_plan(config, &formats)
            .map_err(AdmissionError::InvalidArtifact)?;
        let resolution =
            resolve_safetensors_plan(&catalog, &plan).map_err(AdmissionError::Schema)?;
        let bytes = catalog
            .tensors()
            .iter()
            .map(|(k, v)| (k.clone(), v.encoded_byte_len))
            .collect();
        let parameters = admit_recipes(config, None, &catalog, &bytes, &BTreeMap::new(), &formats)?;
        Ok(Self {
            config: config.clone(),
            physical: PhysicalArtifactPreparation::Safetensors {
                catalog: catalog.tensors().clone(),
                shards: catalog.admitted_shards(),
                resolution,
            },
            parameters,
            layout: None,
        })
    }

    /// Admits one published GGUF layout. The caller declares its component role;
    /// header metadata must agree with that role.
    pub fn gguf(
        path: &Path,
        config: &VoiceChatConfig,
        layout: GgufLayout,
    ) -> Result<Self, AdmissionError> {
        Self::from_gguf_headers(
            eredu_gguf::Checkpoint::open_with_representation(
                path,
                eredu_gguf::Limits::default(),
                eredu_gguf::QuantizedTensorRepresentation::DecodedF32,
            )?,
            config,
            layout,
        )
    }

    /// Admits retained GGUF headers. A metadata-only checkpoint remains metadata-only:
    /// it cannot later confer payload-opening authority through `retain`.
    pub fn from_gguf_headers(
        checkpoint: eredu_gguf::Checkpoint,
        config: &VoiceChatConfig,
        layout: GgufLayout,
    ) -> Result<Self, AdmissionError> {
        checkpoint::validate_gguf_metadata(config, layout, checkpoint.metadata())
            .map_err(AdmissionError::InvalidArtifact)?;
        let plan =
            checkpoint::gguf_plan(config, layout).map_err(AdmissionError::InvalidArtifact)?;
        let resolution = resolve_gguf_plan(&checkpoint, &plan).map_err(AdmissionError::Schema)?;
        // These canonical recipes consume floating values, including convolutions
        // and transformed recurrent weights. Decoding is deferred to bounded leases;
        // no packed-linear execution mechanism is implied by storage admission.
        let quantized = checkpoint
            .tensors()
            .filter(|t| {
                t.descriptor()
                    .ggml_type
                    .block_and_bytes()
                    .is_ok_and(|(n, _)| n > 1)
            })
            .map(|t| t.descriptor().name.clone())
            .collect::<Vec<_>>();
        let conversion_workspace = checkpoint
            .tensors()
            .filter(|t| quantized.contains(&t.descriptor().name))
            .map(|t| {
                let bytes = t.descriptor().element_count()?.checked_mul(4).ok_or(
                    eredu_gguf::Error::Overflow("VoiceChat decoded conversion workspace"),
                )?;
                Ok((t.descriptor().name.clone(), bytes))
            })
            .collect::<Result<BTreeMap<_, _>, eredu_gguf::Error>>()?;
        let bytes = checkpoint
            .tensors()
            .map(|t| (t.descriptor().name.clone(), t.descriptor().byte_len))
            .collect();
        let checkpoint = checkpoint.into_tensor_representation(
            quantized,
            eredu_gguf::QuantizedTensorRepresentation::DecodedF32,
        )?;
        let mapping = checkpoint
            .tensors()
            .flat_map(|t| {
                t.outputs()
                    .iter()
                    .map(move |output| eredu_gguf::TranslatedTensorLayout {
                        physical_name: t.descriptor().name.clone(),
                        original_name: output.name.clone(),
                        layout: output.clone(),
                    })
            })
            .collect::<Vec<_>>();
        let catalog = GgufCatalog::from_resolved_checkpoint(&checkpoint, &resolution, &mapping)?;
        let parameters = admit_recipes(
            config,
            Some(layout),
            &catalog,
            &bytes,
            &conversion_workspace,
            &BTreeMap::new(),
        )?;
        Ok(Self {
            config: config.clone(),
            physical: PhysicalArtifactPreparation::Gguf {
                checkpoint,
                mapping,
                resolution,
            },
            parameters,
            layout: Some(layout),
        })
    }

    /// Canonical parameter metadata; no tensor payload has been materialized.
    pub fn parameters(&self) -> &BTreeMap<String, ParameterAdmission> {
        &self.parameters
    }
}

fn admit_recipes(
    config: &VoiceChatConfig,
    layout: Option<GgufLayout>,
    catalog: &impl RecipeCatalog,
    bytes: &BTreeMap<String, u64>,
    conversion_workspace: &BTreeMap<String, u64>,
    formats: &BTreeMap<String, eredu_checkpoint::LinearFormat>,
) -> Result<BTreeMap<String, ParameterAdmission>, AdmissionError> {
    let mut recipes =
        checkpoint::canonical_recipes(config, layout).map_err(AdmissionError::InvalidArtifact)?;
    for (name, format) in formats {
        if let eredu_checkpoint::LinearFormat::Affine(_) = format {
            let prefix = name
                .strip_suffix(".weight")
                .expect("declared projection matrix");
            for suffix in ["scales", "biases"] {
                let companion = format!("{prefix}.{suffix}");
                recipes.insert(
                    companion.clone(),
                    DerivedWeightRecipe::source(companion, TensorSelection::Full),
                );
            }
        }
    }
    recipes
        .into_iter()
        .map(|(name, recipe)| {
            let output = recipe.infer(catalog)?;
            let stored_bytes = recipe
                .source_keys()
                .into_iter()
                .try_fold(0u64, |sum, key| {
                    sum.checked_add(*bytes.get(key).ok_or_else(|| {
                        AdmissionError::InvalidArtifact(format!("missing physical input {key}"))
                    })?)
                    .ok_or_else(|| {
                        AdmissionError::InvalidArtifact(
                            "parameter storage byte count overflow".into(),
                        )
                    })
                })?;
            let conversion_workspace_bytes =
                recipe
                    .source_keys()
                    .into_iter()
                    .try_fold(0u64, |sum, key| {
                        sum.checked_add(conversion_workspace.get(key).copied().unwrap_or(0))
                            .ok_or_else(|| {
                                AdmissionError::InvalidArtifact(
                                    "conversion workspace byte count overflow".into(),
                                )
                            })
                    })?;
            let peak = recipe
                .peak_materialization_bytes(catalog)?
                .checked_add(conversion_workspace_bytes)
                .ok_or_else(|| {
                    AdmissionError::InvalidArtifact("conversion workspace peak overflow".into())
                })?;
            let materialization_peak_bytes = peak.checked_add(stored_bytes).ok_or_else(|| {
                AdmissionError::InvalidArtifact("parameter conversion byte count overflow".into())
            })?;
            let component = checkpoint::component_for_key(&name).ok_or_else(|| {
                AdmissionError::InvalidArtifact(format!("unknown component for {name}"))
            })?;
            Ok((
                name.clone(),
                ParameterAdmission {
                    linear_format: formats
                        .get(&name)
                        .copied()
                        .unwrap_or(eredu_checkpoint::LinearFormat::Dense),
                    name,
                    component,
                    artifact: PhysicalArtifactId::Primary,
                    output,
                    stored_bytes,
                    conversion_workspace_bytes,
                    materialization_peak_bytes,
                    recipe,
                },
            ))
        })
        .collect()
}

/// Selected complete component graph, retaining exact physical admission decisions.
#[derive(Debug, Clone)]
pub struct VoiceChatArtifactPlan {
    config: VoiceChatConfig,
    declarations: ArtifactSourceDeclarations,
    parameters: BTreeMap<String, ParameterAdmission>,
}

impl VoiceChatArtifactPlan {
    /// Combines primary sources and fills only missing parameters from an explicit
    /// complete companion. Selecting no components is invalid.
    pub fn new(
        config: &VoiceChatConfig,
        sources: Vec<AdmittedVoiceChatSource>,
        companion: Option<AdmittedVoiceChatSource>,
        required: BTreeSet<Component>,
    ) -> Result<Self, AdmissionError> {
        config.validate()?;
        if sources.is_empty() || required.is_empty() {
            return Err(AdmissionError::InvalidArtifact(
                "at least one primary source and selected component are required".into(),
            ));
        }
        if companion
            .as_ref()
            .is_some_and(|s| s.layout.is_some_and(|l| l != GgufLayout::Complete))
        {
            return Err(AdmissionError::InvalidArtifact(
                "companion must contain the complete canonical release catalog".into(),
            ));
        }
        let mut declarations = ArtifactSourceDeclarations::default();
        let mut parameters = BTreeMap::new();
        for (index, source) in sources.into_iter().enumerate() {
            let id = if index == 0 {
                PhysicalArtifactId::Primary
            } else {
                PhysicalArtifactId::Companion(GgufCompanionRole::Named(format!(
                    "voicechat/component/{index}"
                )))
            };
            insert_source(
                config,
                &required,
                source,
                id,
                false,
                &mut declarations,
                &mut parameters,
            )?;
        }
        if let Some(source) = companion {
            insert_source(
                config,
                &required,
                source,
                PhysicalArtifactId::Companion(GgufCompanionRole::Named(
                    "voicechat/companion".into(),
                )),
                true,
                &mut declarations,
                &mut parameters,
            )?;
        }
        let expected =
            checkpoint::canonical_recipes(config, None).map_err(AdmissionError::InvalidArtifact)?;
        for component in required {
            let keys = expected
                .keys()
                .filter(|k| {
                    checkpoint::component_for_key(k) == Some(component)
                        && !parameters.contains_key(*k)
                })
                .cloned()
                .collect::<Vec<_>>();
            if !keys.is_empty() {
                return Err(AdmissionError::MissingComponent { component, keys });
            }
        }
        Ok(Self {
            config: config.clone(),
            declarations,
            parameters,
        })
    }
    /// Exact selected canonical recipes and conversion bounds.
    pub fn parameters(&self) -> &BTreeMap<String, ParameterAdmission> {
        &self.parameters
    }
    /// Physical artifacts and restricted component projections.
    pub fn sources(&self) -> &ArtifactSourceDeclarations {
        &self.declarations
    }
    /// Normalized architecture policy shared by admission and construction.
    pub fn config(&self) -> &VoiceChatConfig {
        &self.config
    }
    /// Opens admitted physical stores once and retains exact auxiliary assets.
    /// This does not construct an executable model or allocate a native tensor.
    pub fn retain(
        self,
        assets: PreparedVoiceChatAssets,
        max_cached_readers: usize,
    ) -> Result<RetainedVoiceChatArtifacts, AdmissionError> {
        let mut asset_config = VoiceChatConfig::from_value(assets.configuration())?;
        asset_config.linear_formats = self.config.linear_formats.clone();
        if asset_config != self.config {
            return Err(AdmissionError::InvalidArtifact(
                "retained assets disagree with admitted configuration".into(),
            ));
        }
        for physical in self.declarations.artifacts.values() {
            if let PhysicalArtifactPreparation::Gguf { checkpoint, .. } = physical {
                if checkpoint.metadata().get("general.architecture")
                    == Some(&eredu_gguf::MetadataValue::String("nemotron_h".into()))
                {
                    assets.validate_split_backbone_tokenizer(checkpoint.metadata())?;
                }
            }
        }
        let sources = self.declarations.bind(max_cached_readers)?;
        let controls = super::controls::VOICECHAT_INTEGER_BUFFERS
            .iter()
            .filter(|c| self.parameters.contains_key(c.key))
            .copied()
            .collect::<Vec<_>>();
        let integer_buffers = super::controls::retain_integer_buffers(&controls, |key, range| {
            let parameter = &self.parameters[key];
            let source = &sources.roles
                [&ArtifactSourceRole::Component(component_label(parameter.component))];
            // Integer recipes only cast I32 to I64 and reshape singleton pads.
            // Reading integers directly avoids any float round trip.
            let keys = parameter.recipe.source_keys();
            if keys.len() != 1 {
                return Err(StoreError::Internal(
                    "integer recipe has multiple inputs".into(),
                ));
            }
            read_integer_range(source.as_ref(), keys[0], range)
        })?;
        for (id, physical) in &self.declarations.artifacts {
            if let PhysicalArtifactPreparation::Gguf { checkpoint, .. } = physical {
                if checkpoint.metadata().get("general.architecture")
                    == Some(&eredu_gguf::MetadataValue::String("voicechat_tts".into()))
                    && self
                        .declarations
                        .role(ArtifactSourceRole::Component(component_label(
                            Component::Tts,
                        )))
                        .is_some_and(|roles| roles.contains_key(id))
                {
                    super::controls::validate_split_character_tables(|key, range| {
                        read_integer_range(sources.physical[id].as_ref(), key, range)
                    })?;
                }
            }
        }
        let preprocessing = sources
            .roles
            .get(&ArtifactSourceRole::Component(component_label(
                Component::Perception,
            )))
            .map(|source| {
                super::preprocessing::PreparedPreprocessing::retain(
                    &self.parameters,
                    source.as_ref(),
                )
            })
            .transpose()?;
        Ok(RetainedVoiceChatArtifacts {
            plan: self,
            assets: Arc::new(assets),
            sources,
            integer_buffers,
            preprocessing,
        })
    }
}

fn insert_source(
    config: &VoiceChatConfig,
    required: &BTreeSet<Component>,
    source: AdmittedVoiceChatSource,
    id: PhysicalArtifactId,
    fallback: bool,
    declarations: &mut ArtifactSourceDeclarations,
    parameters: &mut BTreeMap<String, ParameterAdmission>,
) -> Result<(), AdmissionError> {
    if source.config != *config {
        return Err(AdmissionError::InvalidArtifact(
            "source configuration mismatch".into(),
        ));
    }
    let mut used = false;
    let split_tts = source.layout == Some(GgufLayout::Tts);
    for (key, mut parameter) in source.parameters {
        if !required.contains(&parameter.component) {
            continue;
        }
        if parameters.contains_key(&key) {
            if fallback {
                continue;
            }
            return Err(AdmissionError::DuplicateParameter(key));
        }
        parameter.artifact = id.clone();
        let keys = declarations
            .roles
            .entry(ArtifactSourceRole::Component(component_label(
                parameter.component,
            )))
            .or_default()
            .entry(id.clone())
            .or_default();
        keys.extend(
            parameter
                .recipe
                .source_keys()
                .into_iter()
                .map(str::to_owned),
        );
        parameters.insert(key, parameter);
        used = true;
    }
    if used {
        if split_tts && required.contains(&Component::Tts) {
            declarations
                .roles
                .entry(ArtifactSourceRole::Component(component_label(
                    Component::Tts,
                )))
                .or_default()
                .entry(id.clone())
                .or_default()
                .extend(
                    super::controls::VOICECHAT_SPLIT_CHARACTER_BUFFERS
                        .iter()
                        .map(|c| c.key.to_owned()),
                );
        }
        declarations.artifacts.insert(id, source.physical);
    }
    Ok(())
}

/// Stores, canonical recipes and tokenizer bytes retained for later construction.
/// Access is restricted to selected component inputs; no path-based reopening is needed.
pub struct RetainedVoiceChatArtifacts {
    plan: VoiceChatArtifactPlan,
    assets: Arc<PreparedVoiceChatAssets>,
    sources: BoundArtifactSources,
    integer_buffers: BTreeMap<String, Arc<[i64]>>,
    preprocessing: Option<super::preprocessing::PreparedPreprocessing>,
}
impl RetainedVoiceChatArtifacts {
    /// Exact host feature coefficients when the perception component is selected.
    pub fn preprocessing(&self) -> Option<&super::preprocessing::PreparedPreprocessing> {
        self.preprocessing.as_ref()
    }
    /// Exact selected metadata and canonical recipes.
    pub fn plan(&self) -> &VoiceChatArtifactPlan {
        &self.plan
    }
    /// Immutable, digest-validated auxiliary assets.
    pub fn assets(&self) -> &PreparedVoiceChatAssets {
        &self.assets
    }
    /// Restricted source for the selected component, with original provenance.
    pub fn component_source(&self, component: Component) -> Option<&SharedCheckpointSource> {
        self.sources
            .roles
            .get(&ArtifactSourceRole::Component(component_label(component)))
    }
    /// Number of physical stores shared by all component views.
    pub fn physical_source_count(&self) -> usize {
        self.sources.physical.len()
    }
    /// Canonical integer payload storage retained on the host, excluding map overhead.
    /// This storage is separate from learned parameter materialization.
    pub fn retained_integer_bytes(&self) -> usize {
        self.integer_buffers
            .values()
            .map(|values| std::mem::size_of_val(values.as_ref()))
            .sum()
    }
    pub(super) fn integer_buffers(&self) -> &BTreeMap<String, Arc<[i64]>> {
        &self.integer_buffers
    }
}

fn component_label(component: Component) -> &'static str {
    match component {
        Component::Perception => "voicechat/perception",
        Component::Backbone => "voicechat/backbone",
        Component::FunctionHead => "voicechat/function",
        Component::Tts => "voicechat/tts",
        Component::CodecDecoder => "voicechat/codec_decoder",
        Component::CodecEncoder => "voicechat/codec_encoder",
        Component::CodecQuantizer => "voicechat/codec_quantizer",
        Component::Rnnt => "voicechat/rnnt",
        Component::Controls => "voicechat/controls",
    }
}

// Both stores honor a contiguous scalar selection, including rank-zero buffers.
fn read_integer_range(
    source: &dyn CheckpointSource,
    key: &str,
    range: std::ops::Range<usize>,
) -> Result<Vec<i64>, StoreError> {
    let metadata = source.source_metadata(key)?;
    if !matches!(
        metadata.stored_dtype,
        eredu_checkpoint::StoredDtype::I32 | eredu_checkpoint::StoredDtype::I64
    ) {
        return Err(StoreError::Internal(format!(
            "VoiceChat integer buffer {key} has non-integer storage"
        )));
    }
    let lease = source.acquire_lease(TensorReadRequest {
        key: key.into(),
        selection: TensorSelection::Contiguous {
            offset_elements: range.start,
            shape: vec![range.len()],
        },
        policy: ReadPolicy::RequireBounded,
    })?;
    let converted;
    let native_endian = matches!(lease, CheckpointLease::Gguf(_));
    let bytes = if let CheckpointLease::Gguf(gguf) = &lease {
        converted = gguf.materialize_portable()?.into_converted();
        match &converted {
            eredu_gguf::ConvertedTensor::Dense(tensor) => tensor.data.as_slice(),
            _ => {
                return Err(StoreError::Internal(
                    "integer source converted to non-dense storage".into(),
                ));
            }
        }
    } else {
        lease
            .encoded_bytes()
            .ok_or_else(|| StoreError::Internal("integer lease has no bytes".into()))?
    };
    let width = if metadata.stored_dtype == eredu_checkpoint::StoredDtype::I32 {
        4
    } else {
        8
    };
    if bytes.len() != range.len() * width {
        return Err(StoreError::Internal(
            "integer lease returned an incorrect byte count".into(),
        ));
    }
    Ok(bytes
        .chunks_exact(width)
        .map(|value| {
            if width == 4 {
                let bytes = value.try_into().expect("four bytes");
                i64::from(if native_endian {
                    i32::from_ne_bytes(bytes)
                } else {
                    i32::from_le_bytes(bytes)
                })
            } else {
                let bytes = value.try_into().expect("eight bytes");
                if native_endian {
                    i64::from_ne_bytes(bytes)
                } else {
                    i64::from_le_bytes(bytes)
                }
            }
        })
        .collect())
}

#[cfg(test)]
#[path = "artifact_tests.rs"]
pub(in crate::voicechat) mod tests;
