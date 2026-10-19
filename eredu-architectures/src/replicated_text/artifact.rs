//! Immutable artifact declarations retained independently of execution policy.
use super::*;
use eredu_checkpoint::recipe::{ArtifactCatalog, RecipeCatalog, RecipeInferenceCache};

/// Canonical parameters and recipes for an exact admitted language-model artifact.
/// This contract exposes metadata only; it retains no readable checkpoint source,
/// selected mechanism, load policy, or backend object.
#[derive(Debug, Clone)]
pub struct NormalizedArtifactPreparation {
    catalog: Arc<ArtifactRecipeCatalog>,
    sources: Arc<crate::artifact_preparation::ArtifactSourceDeclarations>,
    physical: BTreeMap<String, ReplicatedTextPhysicalSource>,
    pub(crate) parameters: FinalizedMaterializationParameters,
    pub(super) prediction: Option<FinalizedMaterializationParameters>,
    pub(super) shared_source_keys: BTreeSet<String>,
}
impl NormalizedArtifactPreparation {
    pub(crate) fn from_parameters(
        metadata: BTreeMap<String, eredu_checkpoint::store::TensorMetadata>,
        physical: BTreeMap<String, ReplicatedTextPhysicalSource>,
        parameters: FinalizedMaterializationParameters,
    ) -> Self {
        Self {
            catalog: Arc::new(ArtifactRecipeCatalog {
                source: Arc::new(ParameterCatalog {
                    metadata,
                    physical: physical.clone(),
                }),
                allowed_keys: None,
                recipes: Default::default(),
            }),
            physical,
            sources: Arc::default(),
            parameters,
            prediction: None,
            shared_source_keys: BTreeSet::new(),
        }
    }
    pub(crate) fn with_sources(
        mut self,
        sources: Arc<crate::artifact_preparation::ArtifactSourceDeclarations>,
    ) -> Self {
        self.sources = sources;
        self
    }
    /// Exact physical tensor provenance, independent of logical consumers.
    pub fn physical_sources(&self) -> &BTreeMap<String, ReplicatedTextPhysicalSource> {
        &self.physical
    }
    /// Exact physical artifacts and logical role assignments.
    pub fn sources(&self) -> &crate::artifact_preparation::ArtifactSourceDeclarations {
        &self.sources
    }
    pub(crate) fn source_keys(&self) -> Vec<String> {
        self.catalog.source_keys()
    }
    /// Exact metadata catalog for cold recipe inference.
    pub fn catalog(&self) -> &dyn RecipeCatalog {
        self.catalog.as_ref()
    }
    /// Canonical target declarations, including optional and shared parameters.
    pub fn parameters(&self) -> &[ReplicatedTextParameterRequirement] {
        &self.parameters.0
    }
    /// Family-authored transformations before request-specific selection.
    pub fn recipes(&self) -> &BTreeMap<String, eredu_checkpoint::recipe::DerivedWeightRecipe> {
        &self.parameters.1
    }
}

#[derive(Debug)]
pub(crate) struct ArtifactRecipeCatalog {
    source: Arc<dyn ColdCatalog>,
    allowed_keys: Option<BTreeSet<String>>,
    recipes: RecipeInferenceCache,
}
impl ArtifactRecipeCatalog {
    fn check_key(&self, key: &str) -> Result<(), eredu_checkpoint::store::StoreError> {
        if self
            .allowed_keys
            .as_ref()
            .is_some_and(|keys| !keys.contains(key))
        {
            return Err(eredu_checkpoint::store::StoreError::UnknownTensor { key: key.into() });
        }
        Ok(())
    }
}
impl RecipeCatalog for ArtifactRecipeCatalog {
    fn tensor_metadata(
        &self,
        key: &str,
    ) -> Result<eredu_checkpoint::store::TensorMetadata, eredu_checkpoint::store::StoreError> {
        self.check_key(key)?;
        self.source.source_metadata(key)
    }
    fn recipe_cache(&self) -> Option<&RecipeInferenceCache> {
        Some(&self.recipes)
    }
}

pub(crate) fn catalog(
    inspection: &ArtifactInspection<ArtifactArchitecturePlan>,
) -> Result<Arc<ArtifactRecipeCatalog>, ReplicatedTextRequirementsError> {
    inspection
        .architecture_plan()
        .validation(inspection.admission_token())
        .catalog
        .get_or_init(|| {
            Ok(Arc::new(ArtifactRecipeCatalog {
                source: derive_recipe_source(inspection)?,
                allowed_keys: None,
                recipes: Default::default(),
            }))
        })
        .clone()
}
/// Target recipes cannot name an additive predictor's admitted tensors.
pub(crate) fn target_catalog(
    inspection: &ArtifactInspection<ArtifactArchitecturePlan>,
) -> Result<Arc<ArtifactRecipeCatalog>, ReplicatedTextRequirementsError> {
    let catalog = catalog(inspection)?;
    if inspection
        .architecture_plan()
        .prediction_extension()
        .is_none()
        && inspection.architecture_plan().media_companions().is_empty()
    {
        return Ok(catalog);
    }
    let sources = crate::artifact_preparation::source_declarations(inspection)?;
    Ok(Arc::new(ArtifactRecipeCatalog {
        source: catalog.source.clone(),
        allowed_keys: sources
            .role(crate::artifact_preparation::ArtifactSourceRole::Target)
            .map(|artifacts| artifacts.values().flatten().cloned().collect()),
        recipes: Default::default(),
    }))
}

pub(super) fn normalize(
    inspection: &ArtifactInspection<ArtifactArchitecturePlan>,
    config: &EligibleConfig<'_>,
) -> Result<Arc<NormalizedArtifactPreparation>, ReplicatedTextRequirementsError> {
    inspection
        .architecture_plan()
        .validation(inspection.admission_token())
        .normalized
        .get_or_init(|| {
            let plan = inspection.architecture_plan();
            let catalog = catalog(inspection)?;
            // Physical admission includes the additive predictor. Target recipes
            // may inspect only their own role; predictor recipes use the complete
            // architecture and a separate catalog below.
            let sources = crate::artifact_preparation::source_declarations(inspection)?;
            let target_catalog = target_catalog(inspection)?;
            let recipe_source = target_catalog.as_ref();
            let mut parameters = match (
                plan.safetensors_architecture(),
                plan.gguf_plan(),
                inspection.gguf_checkpoint(),
            ) {
                (Some(architecture), None, None) => {
                    let mut schema = architecture.checkpoint().clone();
                    let keep = |tensor: &eredu_checkpoint::schema::SafetensorsTensorConstraint| {
                        !plan.replaces_media_parameter(&tensor.key)
                    };
                    schema.common_tensors.retain(keep);
                    for group in &mut schema.layout_groups {
                        for variant in &mut group.variants {
                            variant.tensors.retain(keep);
                        }
                    }
                    safetensors_schema_parameters(
                        &schema,
                        architecture
                            .checkpoint_resolution()
                            .expect("admitted target"),
                        inspection.tensors(),
                        config,
                        recipe_source,
                    )?
                }
                (None, Some(architecture), Some(checkpoint)) => {
                    let checkpoint = architecture.prepared_checkpoint(checkpoint);
                    let primary_mapping = plan
                        .gguf_media_projector()
                        .map_or(architecture.tensor_mapping(), |projector| {
                            projector.primary_tensor_mapping()
                        });
                    let primary_mapping = primary_mapping
                        .iter()
                        .filter(|mapping| !plan.replaces_media_parameter(&mapping.layout.name))
                        .cloned()
                        .collect::<Vec<_>>();
                    let metadata_sources = metadata_source_keys(architecture.checkpoint());
                    let mut parameters = gguf_parameters(
                        &primary_mapping,
                        &metadata_sources,
                        checkpoint,
                        config,
                        recipe_source,
                    )?;
                    if let Some(projector_plan) = plan.gguf_media_projector() {
                        let companion = inspection
                            .validated_gguf()
                            .and_then(|validated| {
                                validated.companion(
                                    &eredu_core::artifact::ArtifactCompanionRole::MediaProjector,
                                )
                            })
                            .ok_or_else(|| {
                                ReplicatedTextRequirementsError::InvalidArtifact(
                            "admitted GGUF media-projector plan omitted its exact companion".into(),
                        )
                            })?;
                        let mappings = projector_plan
                            .tensor_mapping()
                            .iter()
                            .filter(|mapping| !plan.replaces_media_parameter(&mapping.layout.name))
                            .cloned()
                            .collect::<Vec<_>>();
                        if !mappings.is_empty() {
                            parameters.extend(
                                gguf_parameters(
                                    &mappings,
                                    &metadata_source_keys(projector_plan.checkpoint()),
                                    companion.checkpoint(),
                                    config,
                                    recipe_source,
                                )?
                                .into_iter()
                                .filter(|parameter| parameter.presence().has_physical_source()),
                            );
                        }
                        parameters = finish_parameters(parameters)?;
                    }
                    parameters
                }
                _ => {
                    return Err(ReplicatedTextRequirementsError::InvalidArtifact(
                        "artifact container and admitted architecture plan disagree".into(),
                    ));
                }
            };

            if let Some(adapter) = plan.draft_adapter() {
                parameters.extend(safetensors_schema_parameters(
                    &adapter.schema,
                    &adapter.resolution,
                    &adapter.tensors,
                    config,
                    recipe_source,
                )?);
                parameters = finish_parameters(parameters)?;
            }
            let mut media_recipes = BTreeMap::new();
            for companion in plan.media_companions() {
                let tensors = companion
                    .tensor_catalog()
                    .map_err(|e| ReplicatedTextRequirementsError::InvalidArtifact(e.to_string()))?;
                parameters.extend(safetensors_schema_parameters(
                    companion.schema(),
                    companion.preparation().resolution(),
                    &tensors,
                    config,
                    recipe_source,
                )?);
                media_recipes.extend(companion.recipes().clone());
            }
            if !plan.media_companions().is_empty() {
                parameters = finish_parameters(parameters)?;
            }
            let parameters = finalize_materialization_parameters(
                config,
                parameters,
                recipe_source,
                media_recipes,
            )?;
            let shared_source_keys = config.shared_derived_source_keys(&parameters.1);
            let prediction = plan
                .prediction_extension()
                .map(|extension| {
                    prediction_extension_materialization_parameters(inspection, extension)
                })
                .transpose()?;
            let physical = catalog
                .source_keys()
                .into_iter()
                .map(|key| Ok((key.clone(), exact_physical_source(catalog.as_ref(), &key)?)))
                .collect::<Result<_, ReplicatedTextRequirementsError>>()?;
            Ok(Arc::new(NormalizedArtifactPreparation {
                physical,
                sources,
                catalog,
                parameters,
                prediction,
                shared_source_keys,
            }))
        })
        .clone()
}

/// Preserve admission of source-only schema values without making them neural
/// destinations. Inputs to any target or auxiliary derivation remain binding-owned.
pub(super) fn nonparameter_sources(
    inspection: &ArtifactInspection<ArtifactArchitecturePlan>,
    normalized: &NormalizedArtifactPreparation,
) -> BTreeSet<String> {
    fn safetensors(
        schema: &eredu_checkpoint::schema::SafetensorsCheckpointPlan,
    ) -> BTreeSet<String> {
        schema
            .common_tensors
            .iter()
            .chain(
                schema
                    .layout_groups
                    .iter()
                    .flat_map(|g| &g.variants)
                    .flat_map(|v| &v.tensors),
            )
            .filter(|t| {
                matches!(
                    t.role,
                    eredu_checkpoint::schema::TensorRole::ArtifactMetadata
                        | eredu_checkpoint::schema::TensorRole::RecipeInput
                )
            })
            .flat_map(|t| std::iter::once(t.key.clone()).chain(t.aliases.clone()))
            .collect()
    }
    let plan = inspection.architecture_plan();
    let mut candidates = BTreeSet::new();
    if let Some(architecture) = plan.safetensors_architecture() {
        candidates.extend(safetensors(architecture.checkpoint()));
    }
    if let Some(architecture) = plan.gguf_plan() {
        candidates.extend(
            metadata_source_keys(architecture.checkpoint())
                .into_iter()
                .map(str::to_owned),
        );
    }
    if let Some(projector) = plan.gguf_media_projector() {
        candidates.extend(
            metadata_source_keys(projector.checkpoint())
                .into_iter()
                .map(str::to_owned),
        );
    }
    if let Some(adapter) = plan.draft_adapter() {
        candidates.extend(safetensors(&adapter.schema));
    }
    if let Some(extension) = plan.prediction_extension() {
        candidates.extend(safetensors(extension.complete_architecture().checkpoint()));
    }
    for companion in plan.media_companions() {
        candidates.extend(safetensors(companion.schema()));
    }
    let declarations = std::iter::once(&normalized.parameters).chain(normalized.prediction.iter());
    let referenced = declarations
        .flat_map(|(parameters, recipes, _)| {
            parameters
                .iter()
                .flat_map(|p| p.sources().iter().map(String::as_str))
                .chain(recipes.values().flat_map(|r| r.source_keys()))
        })
        .collect::<BTreeSet<_>>();
    candidates
        .retain(|key| normalized.physical.contains_key(key) && !referenced.contains(key.as_str()));
    candidates
}
#[derive(Debug)]
pub(super) struct InspectionCheckpointSource {
    pub(super) recipes: eredu_checkpoint::recipe::RecipeInferenceCache,
    pub(super) metadata: BTreeMap<String, eredu_checkpoint::store::TensorMetadata>,
    pub(super) backend: eredu_checkpoint::store::WeightStoreBackend,
}

impl ArtifactCatalog for InspectionCheckpointSource {
    fn source_keys(&self) -> Vec<String> {
        self.metadata.keys().cloned().collect()
    }

    fn source_metadata(
        &self,
        key: &str,
    ) -> Result<eredu_checkpoint::store::TensorMetadata, eredu_checkpoint::store::StoreError> {
        self.metadata.get(key).cloned().ok_or_else(|| {
            eredu_checkpoint::store::StoreError::UnknownTensor {
                key: key.to_owned(),
            }
        })
    }

    fn source_backend(
        &self,
    ) -> Result<eredu_checkpoint::store::WeightStoreBackend, eredu_checkpoint::store::StoreError>
    {
        Ok(self.backend)
    }
    fn source_provenance(
        &self,
        key: &str,
    ) -> Result<eredu_checkpoint::store::TensorSourceProvenance, eredu_checkpoint::store::StoreError>
    {
        let m = self.source_metadata(key)?;
        Ok(eredu_checkpoint::store::TensorSourceProvenance {
            catalog_key: key.into(),
            physical_tensor: key.into(),
            output: key.into(),
            backing_shard: m.backing_shard,
            source_encoding: SourceTensorEncoding::Safetensors(m.stored_dtype),
        })
    }
}
impl RecipeCatalog for InspectionCheckpointSource {
    fn tensor_metadata(
        &self,
        key: &str,
    ) -> Result<eredu_checkpoint::store::TensorMetadata, eredu_checkpoint::store::StoreError> {
        self.source_metadata(key)
    }
    fn recipe_cache(&self) -> Option<&RecipeInferenceCache> {
        Some(&self.recipes)
    }
}

#[derive(Debug)]
struct GgufInspectionCatalog(eredu_checkpoint::gguf_store::GgufCatalog);
impl RecipeCatalog for GgufInspectionCatalog {
    fn tensor_metadata(
        &self,
        key: &str,
    ) -> Result<eredu_checkpoint::store::TensorMetadata, eredu_checkpoint::store::StoreError> {
        self.0.metadata(key)
    }
}
impl ArtifactCatalog for GgufInspectionCatalog {
    fn source_keys(&self) -> Vec<String> {
        self.0.keys()
    }
    fn source_metadata(
        &self,
        key: &str,
    ) -> Result<eredu_checkpoint::store::TensorMetadata, eredu_checkpoint::store::StoreError> {
        self.0.metadata(key)
    }
    fn source_provenance(
        &self,
        key: &str,
    ) -> Result<eredu_checkpoint::store::TensorSourceProvenance, eredu_checkpoint::store::StoreError>
    {
        self.0.source_provenance(key)
    }
    fn source_backend(
        &self,
    ) -> Result<eredu_checkpoint::store::WeightStoreBackend, eredu_checkpoint::store::StoreError>
    {
        Ok(eredu_checkpoint::store::WeightStoreBackend::Gguf)
    }
}
#[derive(Debug)]
struct CompositeInspectionCatalog(Vec<Arc<dyn ColdCatalog>>);
impl RecipeCatalog for CompositeInspectionCatalog {
    fn tensor_metadata(
        &self,
        key: &str,
    ) -> Result<eredu_checkpoint::store::TensorMetadata, eredu_checkpoint::store::StoreError> {
        self.source_metadata(key)
    }
}
impl ArtifactCatalog for CompositeInspectionCatalog {
    fn source_keys(&self) -> Vec<String> {
        self.0.iter().flat_map(|c| c.source_keys()).collect()
    }
    fn source_metadata(
        &self,
        key: &str,
    ) -> Result<eredu_checkpoint::store::TensorMetadata, eredu_checkpoint::store::StoreError> {
        self.0
            .iter()
            .find_map(|c| c.source_metadata(key).ok())
            .ok_or_else(|| eredu_checkpoint::store::StoreError::UnknownTensor { key: key.into() })
    }
    fn source_provenance(
        &self,
        key: &str,
    ) -> Result<eredu_checkpoint::store::TensorSourceProvenance, eredu_checkpoint::store::StoreError>
    {
        self.0
            .iter()
            .find_map(|c| c.source_provenance(key).ok())
            .ok_or_else(|| eredu_checkpoint::store::StoreError::UnknownTensor { key: key.into() })
    }
    fn source_backend(
        &self,
    ) -> Result<eredu_checkpoint::store::WeightStoreBackend, eredu_checkpoint::store::StoreError>
    {
        let mut kinds = self.0.iter().map(|c| c.source_backend());
        let first = kinds
            .next()
            .transpose()?
            .unwrap_or(eredu_checkpoint::store::WeightStoreBackend::Composite);
        for kind in kinds {
            if kind? != first {
                return Ok(eredu_checkpoint::store::WeightStoreBackend::Composite);
            }
        }
        Ok(first)
    }
}
pub(super) trait ColdCatalog: ArtifactCatalog + std::fmt::Debug + Send + Sync {}
impl<T: ArtifactCatalog + std::fmt::Debug + Send + Sync> ColdCatalog for T {}
impl ArtifactCatalog for ArtifactRecipeCatalog {
    fn source_keys(&self) -> Vec<String> {
        self.allowed_keys.as_ref().map_or_else(
            || self.source.source_keys(),
            |keys| keys.iter().cloned().collect(),
        )
    }
    fn source_backend(
        &self,
    ) -> Result<eredu_checkpoint::store::WeightStoreBackend, eredu_checkpoint::store::StoreError>
    {
        self.source.source_backend()
    }
    fn source_provenance(
        &self,
        key: &str,
    ) -> Result<eredu_checkpoint::store::TensorSourceProvenance, eredu_checkpoint::store::StoreError>
    {
        self.check_key(key)?;
        self.source.source_provenance(key)
    }
}
/// Derives exact replicated text requirements from an admitted artifact.
fn derive_recipe_source(
    inspection: &ArtifactInspection<ArtifactArchitecturePlan>,
) -> Result<Arc<dyn ColdCatalog>, ReplicatedTextRequirementsError> {
    use crate::artifact_preparation::PhysicalArtifactPreparation;
    let declarations = crate::artifact_preparation::source_declarations(inspection)?;
    let mut catalogs: Vec<Arc<dyn ColdCatalog>> = Vec::new();
    let mut keys = BTreeSet::new();
    for (id, artifact) in declarations.artifacts() {
        let active = declarations
            .roles
            .values()
            .filter_map(|roles| roles.get(id))
            .flatten()
            .cloned()
            .collect::<BTreeSet<_>>();
        if active.is_empty() {
            continue;
        }
        let catalog: Arc<dyn ColdCatalog> = match artifact {
            PhysicalArtifactPreparation::Safetensors {
                catalog,
                resolution,
                ..
            } => Arc::new(InspectionCheckpointSource {
                recipes: Default::default(),
                metadata: catalog
                    .iter()
                    .filter(|(key, _)| resolution.source_keys().contains(*key))
                    .map(|(key, value)| (key.clone(), value.clone()))
                    .collect(),
                backend: eredu_checkpoint::store::WeightStoreBackend::Safetensors,
            }),
            PhysicalArtifactPreparation::Gguf {
                checkpoint,
                mapping,
                resolution,
            } => Arc::new(GgufInspectionCatalog(
                eredu_checkpoint::gguf_store::GgufCatalog::from_resolved_checkpoint(
                    checkpoint, resolution, mapping,
                )
                .map_err(|e| ReplicatedTextRequirementsError::InvalidArtifact(e.to_string()))?,
            )),
            PhysicalArtifactPreparation::MetadataOnly { .. } => {
                let architecture = inspection
                    .architecture_plan()
                    .safetensors_architecture()
                    .ok_or_else(|| {
                        ReplicatedTextRequirementsError::InvalidArtifact(
                            "metadata-only source requires a SafeTensors schema".into(),
                        )
                    })?;
                safetensors_inspection_recipe_source(inspection, architecture)?
            }
        };
        let catalog: Arc<dyn ColdCatalog> = Arc::new(ArtifactRecipeCatalog {
            source: catalog,
            allowed_keys: Some(active),
            recipes: Default::default(),
        });
        for key in catalog.source_keys() {
            if !keys.insert(key.clone()) {
                return Err(ReplicatedTextRequirementsError::InvalidArtifact(format!(
                    "physical artifacts overlap at source {key:?}"
                )));
            }
        }
        catalogs.push(catalog);
    }
    if catalogs.len() == 1 {
        Ok(catalogs.remove(0))
    } else {
        Ok(Arc::new(CompositeInspectionCatalog(catalogs)))
    }
}

pub(super) fn safetensors_inspection_recipe_source(
    inspection: &ArtifactInspection<ArtifactArchitecturePlan>,
    architecture: &crate::configuration::SafetensorsArchitecturePlan,
) -> Result<Arc<dyn ColdCatalog>, ReplicatedTextRequirementsError> {
    let selected = architecture.checkpoint_resolution().ok_or_else(|| {
        ReplicatedTextRequirementsError::InvalidArtifact(
            "SafeTensors architecture omitted exact catalog admission".into(),
        )
    })?;
    let mut metadata = BTreeMap::new();
    for key in selected.source_keys() {
        let descriptor = inspection.tensors().get(key).ok_or_else(|| {
            ReplicatedTextRequirementsError::InvalidArtifact(format!(
                "admitted SafeTensors source {key:?} is absent from its catalog"
            ))
        })?;
        let backing_shard = descriptor
            .storage
            .as_ref()
            .map(|storage| std::path::PathBuf::from(&storage.member));
        metadata.insert(
            key.clone(),
            eredu_checkpoint::store::TensorMetadata {
                name: key.clone(),
                logical_shape: descriptor.shape.clone(),
                physical_shape: descriptor.shape.clone(),
                stored_dtype: stored_dtype(&descriptor.dtype)?,
                encoded_byte_len: descriptor
                    .storage
                    .as_ref()
                    .map_or(0, |storage| storage.length),
                backing_shard,
            },
        );
    }
    Ok(Arc::new(InspectionCheckpointSource {
        recipes: Default::default(),
        metadata,
        backend: eredu_checkpoint::store::WeightStoreBackend::Safetensors,
    }))
}

#[derive(Debug)]
struct ParameterCatalog {
    metadata: BTreeMap<String, eredu_checkpoint::store::TensorMetadata>,
    physical: BTreeMap<String, ReplicatedTextPhysicalSource>,
}
impl RecipeCatalog for ParameterCatalog {
    fn tensor_metadata(
        &self,
        key: &str,
    ) -> Result<eredu_checkpoint::store::TensorMetadata, eredu_checkpoint::store::StoreError> {
        self.metadata
            .get(key)
            .cloned()
            .ok_or_else(|| eredu_checkpoint::store::StoreError::UnknownTensor { key: key.into() })
    }
}
impl ArtifactCatalog for ParameterCatalog {
    fn source_keys(&self) -> Vec<String> {
        self.metadata.keys().cloned().collect()
    }
    fn source_backend(
        &self,
    ) -> Result<eredu_checkpoint::store::WeightStoreBackend, eredu_checkpoint::store::StoreError>
    {
        Ok(
            if self
                .physical
                .values()
                .any(|p| matches!(p.source_encoding(), SourceTensorEncoding::Gguf { .. }))
            {
                eredu_checkpoint::store::WeightStoreBackend::Gguf
            } else {
                eredu_checkpoint::store::WeightStoreBackend::Safetensors
            },
        )
    }
    fn source_provenance(
        &self,
        key: &str,
    ) -> Result<eredu_checkpoint::store::TensorSourceProvenance, eredu_checkpoint::store::StoreError>
    {
        let p = self.physical.get(key).ok_or_else(|| {
            eredu_checkpoint::store::StoreError::UnknownTensor { key: key.into() }
        })?;
        Ok(eredu_checkpoint::store::TensorSourceProvenance {
            catalog_key: p.catalog_key().into(),
            physical_tensor: p.tensor().into(),
            backing_shard: Some(p.shard().to_owned()),
            output: p.output().into(),
            source_encoding: p.source_encoding().clone(),
        })
    }
}
pub(super) fn safetensors_parameters(
    architecture: &crate::configuration::SafetensorsArchitecturePlan,
    catalog: &TensorCatalog,
    config: &EligibleConfig<'_>,
    source_catalog: &(impl eredu_checkpoint::recipe::ArtifactCatalog + ?Sized),
) -> Result<Vec<ReplicatedTextParameterRequirement>, ReplicatedTextRequirementsError> {
    let selected = architecture.checkpoint_resolution().ok_or_else(|| {
        ReplicatedTextRequirementsError::InvalidArtifact(
            "SafeTensors architecture omitted exact catalog admission".into(),
        )
    })?;
    safetensors_schema_parameters(
        architecture.checkpoint(),
        selected,
        catalog,
        config,
        source_catalog,
    )
}

fn safetensors_schema_parameters(
    plan: &eredu_checkpoint::schema::SafetensorsCheckpointPlan,
    selected: &eredu_checkpoint::validation::ResolvedCheckpointPlan,
    catalog: &TensorCatalog,
    config: &EligibleConfig<'_>,
    source_catalog: &(impl eredu_checkpoint::recipe::ArtifactCatalog + ?Sized),
) -> Result<Vec<ReplicatedTextParameterRequirement>, ReplicatedTextRequirementsError> {
    let source_linear_shapes = config
        .linear_parameter_shapes()
        .map_err(ReplicatedTextRequirementsError::InvalidArchitecture)?;
    let mut constraints = plan.common_tensors.iter().collect::<Vec<_>>();
    for group in &plan.layout_groups {
        for variant in &group.variants {
            if variant.tensors.iter().all(|constraint| {
                selected.source_keys().contains(&constraint.key)
                    || constraint
                        .aliases
                        .iter()
                        .any(|alias| selected.source_keys().contains(alias))
                    || constraint.requirement
                        == eredu_checkpoint::schema::TensorRequirement::Optional
            }) {
                constraints.extend(variant.tensors.iter());
                break;
            }
        }
    }
    let mut constraints_by_name = BTreeMap::new();
    for constraint in &constraints {
        constraints_by_name
            .entry(constraint.key.as_str())
            .or_insert(*constraint);
    }
    let declared_companions = constraints
        .iter()
        .filter_map(|constraint| {
            constraint.linear_companion.as_ref().map(|companion| {
                let primary = constraints_by_name
                    .get(companion.primary.as_str())
                    .map(|candidate| {
                        config.canonical_parameter_name(&candidate.key, &candidate.aliases)
                    })
                    .unwrap_or_else(|| companion.primary.clone());
                let name = config.canonical_parameter_name(&constraint.key, &constraint.aliases);
                let role = match companion.kind {
                    eredu_checkpoint::schema::LinearCompanionKind::GlobalScale => {
                        eredu_nn::LinearCompanionRole::GlobalScale
                    }
                    eredu_checkpoint::schema::LinearCompanionKind::Scale => {
                        eredu_nn::LinearCompanionRole::Scale
                    }
                    eredu_checkpoint::schema::LinearCompanionKind::AffineBias => {
                        eredu_nn::LinearCompanionRole::AffineBias
                    }
                };
                (name, (role, primary))
            })
        })
        .collect::<BTreeMap<_, _>>();
    let linear_shapes = source_linear_shapes
        .into_iter()
        .map(|(name, shape)| {
            let canonical = constraints_by_name
                .get(name.as_str())
                .map_or(name, |constraint| {
                    config.canonical_parameter_name(&constraint.key, &constraint.aliases)
                });
            (canonical, shape)
        })
        .collect::<BTreeMap<_, _>>();
    let mut parameters = Vec::new();
    for constraint in constraints {
        if matches!(
            constraint.role,
            eredu_checkpoint::schema::TensorRole::ArtifactMetadata
                | eredu_checkpoint::schema::TensorRole::RecipeInput
        ) {
            continue;
        }
        let canonical = config.canonical_parameter_name(&constraint.key, &constraint.aliases);
        let source = std::iter::once(&constraint.key)
            .chain(constraint.aliases.iter())
            .find(|name| selected.source_keys().contains(*name))
            .cloned();
        let descriptor = source
            .as_deref()
            .map(|source| {
                catalog.get(source).ok_or_else(|| {
                    ReplicatedTextRequirementsError::InvalidArtifact(format!(
                        "admitted SafeTensors source {source:?} is absent from its catalog"
                    ))
                })
            })
            .transpose()?;
        let presence = match (constraint.requirement, source.is_some()) {
            (eredu_checkpoint::schema::TensorRequirement::Required, true) => {
                ReplicatedTextParameterPresence::Required
            }
            (eredu_checkpoint::schema::TensorRequirement::Optional, true) => {
                ReplicatedTextParameterPresence::OptionalPresent
            }
            (eredu_checkpoint::schema::TensorRequirement::Optional, false) => {
                ReplicatedTextParameterPresence::OptionalAbsent
            }
            (eredu_checkpoint::schema::TensorRequirement::Required, false) => {
                return Err(ReplicatedTextRequirementsError::InvalidArtifact(format!(
                    "required admitted SafeTensors parameter {:?} has no source",
                    constraint.key
                )));
            }
        };
        let role = config.parameter_role(
            &canonical,
            constraint.role == eredu_checkpoint::schema::TensorRole::Companion,
            &linear_shapes,
        );
        let logical_shape = if role == ReplicatedTextParameterRole::Embedding {
            config
                .embedding_shape()
                .map_err(ReplicatedTextRequirementsError::InvalidArchitecture)?
        } else {
            linear_shapes
                .get(&canonical)
                .cloned()
                .unwrap_or_else(|| constraint.shape.clone())
        };
        let aliases = std::iter::once(constraint.key.as_str())
            .chain(constraint.aliases.iter().map(String::as_str))
            .filter(|name| *name != canonical)
            .map(str::to_owned)
            .collect::<Vec<_>>();
        let physical_sources = source
            .as_ref()
            .map(|source| exact_physical_source(source_catalog, source))
            .transpose()?
            .into_iter()
            .collect::<Vec<_>>();
        let source_encoding = descriptor
            .map(|descriptor| stored_dtype(&descriptor.dtype))
            .transpose()?
            .map(SourceTensorEncoding::Safetensors);
        let physical_shape = descriptor.map(|descriptor| descriptor.shape.clone());
        if let EligibleConfig::QwenHybrid(args) = config {
            if let Some(targets) = qwen_next_fused_targets(args, &canonical, &constraint.shape)? {
                for (target, shape) in targets {
                    let role = if constraint.role == eredu_checkpoint::schema::TensorRole::Companion
                    {
                        ReplicatedTextParameterRole::FormatCompanion
                    } else {
                        ReplicatedTextParameterRole::LinearWeight
                    };
                    let selects_lowering = role == ReplicatedTextParameterRole::LinearWeight;
                    parameters.push(parameter_requirement(
                        target.clone(),
                        if selects_lowering {
                            source.clone().into_iter().collect()
                        } else {
                            Vec::new()
                        },
                        physical_sources.clone(),
                        aliases.clone(),
                        source_encoding.clone(),
                        physical_shape.clone(),
                        shape,
                        config.native_format(&target),
                        false,
                        role,
                        parameter_owner(config, &target),
                        ReplicatedTextParameterPresence::Derived {
                            recipe: "qwen3_next.grouped_projection_split".into(),
                        },
                    )?);
                }
                continue;
            }
        }
        let native_executable = if matches!(config, EligibleConfig::GptOss(_))
            && canonical.contains(".mlp.experts.")
            && (canonical.ends_with("gate_up_proj") || canonical.ends_with("down_proj"))
        {
            config.native_format(&canonical)
        } else {
            match role {
                ReplicatedTextParameterRole::LinearWeight
                | ReplicatedTextParameterRole::Embedding => {
                    config.native_format_for_shape(&canonical, &logical_shape)
                }
                ReplicatedTextParameterRole::FormatCompanion
                | ReplicatedTextParameterRole::Normalization
                | ReplicatedTextParameterRole::LinearBias
                | ReplicatedTextParameterRole::Other => LinearFormat::Dense,
                _ => LinearFormat::Dense,
            }
        };
        parameters.push(parameter_requirement(
            canonical.clone(),
            source.clone().into_iter().collect(),
            physical_sources,
            aliases,
            source_encoding,
            physical_shape,
            logical_shape,
            native_executable,
            linear_shapes.contains_key(&canonical),
            role,
            parameter_owner(config, &canonical),
            presence,
        )?);
    }
    for parameter in &mut parameters {
        if let Some((role, primary)) = declared_companions.get(parameter.name()) {
            *parameter = parameter
                .clone()
                .with_linear_companion(*role, primary.clone())
                .map_err(|error| {
                    ReplicatedTextRequirementsError::InvalidArchitecture(error.to_string())
                })?;
        }
    }
    finish_parameters_with_tied_output(parameters, config)
}

/// Source-only GGUF entries are admitted and retained without neural bindings.
fn metadata_source_keys(plan: &eredu_checkpoint::schema::GgufCheckpointPlan) -> BTreeSet<&str> {
    plan.common_tensors
        .iter()
        .chain(
            plan.layout_groups
                .iter()
                .flat_map(|group| &group.variants)
                .flat_map(|variant| &variant.tensors),
        )
        .filter(|tensor| {
            matches!(
                tensor.role,
                eredu_checkpoint::schema::TensorRole::ArtifactMetadata
                    | eredu_checkpoint::schema::TensorRole::RecipeInput
            )
        })
        .flat_map(|tensor| {
            std::iter::once(tensor.key.as_str()).chain(tensor.aliases.iter().map(String::as_str))
        })
        .collect()
}

pub(super) fn gguf_parameters(
    tensor_mapping: &[eredu_gguf::TranslatedTensorLayout],
    metadata_sources: &BTreeSet<&str>,
    checkpoint: &eredu_gguf::Checkpoint,
    config: &EligibleConfig<'_>,
    source_catalog: &(impl eredu_checkpoint::recipe::ArtifactCatalog + ?Sized),
) -> Result<Vec<ReplicatedTextParameterRequirement>, ReplicatedTextRequirementsError> {
    let linear_shapes = config
        .linear_parameter_shapes()
        .map_err(ReplicatedTextRequirementsError::InvalidArchitecture)?;
    let mut physical = BTreeMap::new();
    for shard in checkpoint.shards() {
        for tensor in shard.tensors() {
            physical.insert(
                tensor.descriptor().name.as_str(),
                (
                    tensor,
                    SourceTensorEncoding::Gguf {
                        ggml_type: tensor.descriptor().ggml_type,
                        endian: shard.endian(),
                    },
                ),
            );
        }
    }
    let mut parameters = Vec::new();
    for mapping in tensor_mapping {
        if metadata_sources.contains(mapping.physical_name.as_str()) {
            continue;
        }
        let Some((tensor, source_encoding)) = physical.get(mapping.physical_name.as_str()) else {
            return Err(ReplicatedTextRequirementsError::InvalidArtifact(format!(
                "admitted GGUF mapping references absent tensor {:?}",
                mapping.physical_name
            )));
        };
        let SourceTensorEncoding::Gguf { endian, .. } = source_encoding else {
            unreachable!("GGUF catalog produces GGUF source encodings")
        };
        let native = crate::linear_format::gguf_tensor_format(tensor, *endian)
            .map_err(ReplicatedTextRequirementsError::InvalidArtifact)?;
        let companion = mapping.original_name != mapping.physical_name;
        let role = config.parameter_role(&mapping.layout.name, companion, &linear_shapes);
        let translated_shape = mapping
            .layout
            .shape
            .iter()
            .map(|dimension| {
                usize::try_from(*dimension).map_err(|_| {
                    ReplicatedTextRequirementsError::InvalidArtifact(format!(
                        "GGUF logical shape for {:?} exceeds usize",
                        mapping.layout.name
                    ))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        if let EligibleConfig::QwenHybrid(args) = config {
            if let Some(targets) =
                qwen_next_fused_targets(args, &mapping.layout.name, &translated_shape)?
            {
                let physical_shape = tensor
                    .descriptor()
                    .row_major_shape()
                    .into_iter()
                    .map(usize::try_from)
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|_| {
                        ReplicatedTextRequirementsError::InvalidArtifact(format!(
                            "GGUF physical shape for {:?} exceeds usize",
                            mapping.physical_name
                        ))
                    })?;
                let provenance = exact_physical_source(source_catalog, &mapping.layout.name)?;
                for (target, shape) in targets {
                    let role = if companion {
                        ReplicatedTextParameterRole::FormatCompanion
                    } else {
                        ReplicatedTextParameterRole::LinearWeight
                    };
                    let selects_lowering = !companion;
                    let logical_shape = if selects_lowering {
                        linear_shapes.get(&target).cloned().unwrap_or(shape)
                    } else {
                        shape
                    };
                    parameters.push(parameter_requirement(
                        target.clone(),
                        vec![mapping.layout.name.clone()],
                        vec![provenance.clone()],
                        vec![mapping.original_name.clone()],
                        Some(source_encoding.clone()),
                        Some(physical_shape.clone()),
                        logical_shape,
                        if selects_lowering {
                            native
                        } else {
                            LinearFormat::Dense
                        },
                        false,
                        role,
                        parameter_owner(config, &target),
                        ReplicatedTextParameterPresence::Derived {
                            recipe: "qwen3_next.grouped_projection_split".into(),
                        },
                    )?);
                }
                continue;
            }
        }
        let logical_shape = if role == ReplicatedTextParameterRole::Embedding {
            config
                .embedding_shape()
                .map_err(ReplicatedTextRequirementsError::InvalidArchitecture)?
        } else {
            let discovered = linear_shapes
                .get(&mapping.layout.name)
                .cloned()
                .unwrap_or(translated_shape.clone());
            config.logical_parameter_shape(&mapping.layout.name, discovered)
        };
        let derived = companion
            || (config.parameter_requires_shape_recipe(&mapping.layout.name)
                && logical_shape != translated_shape);
        let presence = if derived {
            ReplicatedTextParameterPresence::Derived {
                recipe: format!(
                    "gguf-output:{}:{}",
                    mapping.physical_name, mapping.original_name
                ),
            }
        } else {
            ReplicatedTextParameterPresence::Required
        };
        parameters.push(parameter_requirement(
            mapping.layout.name.clone(),
            vec![mapping.layout.name.clone()],
            vec![exact_physical_source(source_catalog, &mapping.layout.name)?],
            vec![mapping.original_name.clone()],
            Some(source_encoding.clone()),
            Some({
                tensor
                    .descriptor()
                    .row_major_shape()
                    .into_iter()
                    .map(usize::try_from)
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|_| {
                        ReplicatedTextRequirementsError::InvalidArtifact(format!(
                            "GGUF physical shape for {:?} exceeds usize",
                            mapping.physical_name
                        ))
                    })
            }?),
            logical_shape,
            if derived { LinearFormat::Dense } else { native },
            !derived && linear_shapes.contains_key(&mapping.layout.name),
            role,
            parameter_owner(config, &mapping.layout.name),
            presence,
        )?);
    }
    // GGUF encoded-linear companions are dynamically typed native slots. The
    // architecture explicitly admits only the exact catalog dtype selected for
    // each companion; ordinary weights retain exact destination dtype matching.
    for parameter in &mut parameters {
        if parameter.role() != ReplicatedTextParameterRole::FormatCompanion {
            continue;
        }
        let dtypes = parameter
            .sources()
            .iter()
            .map(|source| {
                source_catalog
                    .source_metadata(source)
                    .map(|metadata| metadata.stored_dtype.into())
                    .map_err(|error| {
                        ReplicatedTextRequirementsError::InvalidArtifact(error.to_string())
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        *parameter = parameter
            .clone()
            .with_permitted_native_source_dtypes(dtypes);
    }
    finish_parameters_with_tied_output(parameters, config)
}
