//! Exact physical artifact declarations, independent of logical source roles.
use eredu_checkpoint::{
    store::{CompositeCheckpointSource, RestrictedCheckpointSource, SharedCheckpointSource},
    validation::ResolvedCheckpointPlan,
};
use eredu_core::{artifact::ArtifactError, ArtifactCompanionRole};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

pub use crate::replicated_text::artifact::NormalizedArtifactPreparation;

/// A physical artifact is opened once even when several logical roles consume it.
#[derive(Debug, Clone)]
pub enum PhysicalArtifactPreparation {
    /// Exact admitted SafeTensors headers, shard identities and chosen aliases.
    Safetensors {
        /// Admitted physical tensor headers.
        catalog: BTreeMap<String, eredu_checkpoint::store::TensorMetadata>,
        /// Retained shard file identities.
        shards: eredu_checkpoint::safetensors::SafetensorsShards,
        /// Exact chosen alias and layout resolution.
        resolution: ResolvedCheckpointPlan,
    },
    /// Header-only admission supports cold forecasting but grants no file-opening authority.
    MetadataOnly {
        /// Exact chosen aliases from supplied immutable headers.
        resolution: ResolvedCheckpointPlan,
    },
    /// Exact admitted GGUF headers and canonical physical-to-logical mapping.
    Gguf {
        /// Admitted container headers and file identities.
        checkpoint: eredu_gguf::Checkpoint,
        /// Canonical outputs of each physical tensor.
        mapping: Vec<eredu_gguf::TranslatedTensorLayout>,
        /// Exact chosen layout resolution.
        resolution: ResolvedCheckpointPlan,
    },
}
impl PhysicalArtifactPreparation {
    fn contains_same_sources(&self, other: &Self, keys: &BTreeSet<String>) -> bool {
        let (Self::Safetensors { catalog, shards, resolution },
             Self::Safetensors { catalog: selected, shards: selected_shards, .. }) = (self, other) else {
            return false;
        };
        fn location<'a>(shards: &'a eredu_checkpoint::safetensors::SafetensorsShards, key: &str) -> Option<&'a std::path::Path> {
            shards.tensor_locations().and_then(|locations| locations.get(key)).map(|path| path.as_path())
                .or_else(|| (shards.payload_paths().len() == 1).then(|| shards.payload_paths()[0].as_path()))
        }
        keys.iter().all(|key| {
            resolution.source_keys().contains(key)
                && catalog.get(key).zip(selected.get(key)).is_some_and(|(left, right)| {
                    left.logical_shape == right.logical_shape && left.physical_shape == right.physical_shape
                        && left.stored_dtype == right.stored_dtype && left.encoded_byte_len == right.encoded_byte_len
                })
                && location(shards, key).is_some_and(|path| Some(path) == location(selected_shards, key))
        })
    }

    /// Resolution authorizing this physical artifact.
    pub fn resolution(&self) -> &ResolvedCheckpointPlan {
        match self {
            Self::Safetensors { resolution, .. }
            | Self::Gguf { resolution, .. }
            | Self::MetadataOnly { resolution } => resolution,
        }
    }
    pub(crate) fn open(&self, cache: usize) -> Result<SharedCheckpointSource, ArtifactError> {
        match self {
            Self::MetadataOnly { .. } => Err(ArtifactError::MetadataOnly),
            Self::Safetensors {
                catalog,
                shards,
                resolution,
            } => eredu_checkpoint::store::open_prepared_safetensors_source(
                shards.clone(),
                catalog.clone(),
                resolution.clone(),
                cache,
            )
            .map_err(Into::into),
            Self::Gguf {
                checkpoint,
                mapping,
                resolution,
            } => Ok(Arc::new(
                eredu_checkpoint::gguf_store::GgufWeightStore::builder()
                    .max_cached_readers(cache)?
                    .add_resolved_checkpoint(checkpoint.clone(), resolution, mapping)?
                    .build()?,
            )),
        }
    }
}
/// Physical identity inside one admitted artifact graph.
#[derive(Debug, Clone, Eq, PartialEq, Ord, PartialOrd)]
pub enum PhysicalArtifactId {
    /// Main checkpoint artifact.
    Primary,
    /// Independently admitted companion artifact.
    Companion(ArtifactCompanionRole),
}
/// Logical consumers independent of physical container boundaries.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Ord, PartialOrd)]
pub enum ArtifactSourceRole {
    /// Ordinary target execution, including its shared vocabulary.
    Target,
    /// Prediction-owned parameters, excluding the shared target vocabulary.
    Prediction,
    /// Image/video tower and projector parameters.
    Vision,
    /// Waveform tower and perception-projection parameters.
    Audio,
    /// Architecture-owned component independent of its physical container.
    Component(&'static str),
}

/// Immutable physical opening authority and exact logical source projections.
#[derive(Debug, Clone, Default)]
pub struct ArtifactSourceDeclarations {
    pub(crate) artifacts: BTreeMap<PhysicalArtifactId, PhysicalArtifactPreparation>,
    pub(crate) roles: BTreeMap<ArtifactSourceRole, BTreeMap<PhysicalArtifactId, BTreeSet<String>>>,
}
impl ArtifactSourceDeclarations {
    /// Exact physical artifacts; logical role sharing does not duplicate this set.
    pub fn artifacts(&self) -> &BTreeMap<PhysicalArtifactId, PhysicalArtifactPreparation> {
        &self.artifacts
    }
    /// Exact physical keys assigned to a logical role.
    pub fn role(
        &self,
        role: ArtifactSourceRole,
    ) -> Option<&BTreeMap<PhysicalArtifactId, BTreeSet<String>>> {
        self.roles.get(&role)
    }
    pub(crate) fn bind(&self, cache: usize) -> Result<BoundArtifactSources, ArtifactError> {
        let physical = self
            .artifacts
            .iter()
            .map(|(id, artifact)| Ok((id.clone(), artifact.open(cache)?)))
            .collect::<Result<BTreeMap<_, _>, ArtifactError>>()?;
        let mut roles = BTreeMap::new();
        for (role, declarations) in &self.roles {
            let views = declarations
                .iter()
                .map(|(id, keys)| {
                    let source = physical.get(id).ok_or_else(|| {
                        ArtifactError::InvalidArchitecturePlan(
                            "source role refers to an unadmitted artifact".into(),
                        )
                    })?;
                    if source.source_keys().into_iter().collect::<BTreeSet<_>>() == *keys {
                        return Ok(source.clone());
                    }
                    Ok(Arc::new(RestrictedCheckpointSource::including(
                        source.clone(),
                        format!("{role:?}"),
                        keys.clone(),
                    )?) as SharedCheckpointSource)
                })
                .collect::<Result<Vec<_>, ArtifactError>>()?;
            let view = if views.len() == 1 {
                views[0].clone()
            } else {
                Arc::new(CompositeCheckpointSource::new(views)?)
            };
            roles.insert(*role, view);
        }
        Ok(BoundArtifactSources { physical, roles })
    }
}
pub(crate) struct BoundArtifactSources {
    pub physical: BTreeMap<PhysicalArtifactId, SharedCheckpointSource>,
    pub roles: BTreeMap<ArtifactSourceRole, SharedCheckpointSource>,
}

pub(crate) fn source_declarations(
    inspection: &eredu_core::ArtifactInspection<crate::processor_plan::ArtifactArchitecturePlan>,
) -> Result<Arc<ArtifactSourceDeclarations>, crate::replicated_text::ReplicatedTextRequirementsError>
{
    use crate::replicated_text::ReplicatedTextRequirementsError as Error;
    let invalid = |e: String| Error::InvalidArtifact(e);
    inspection.architecture_plan().validation(inspection.admission_token()).sources.get_or_init(|| {
        let plan = inspection.architecture_plan();
        let mut declarations = ArtifactSourceDeclarations::default();
        match inspection.format() {
            eredu_core::ArtifactFormat::SafeTensors => {
                let target = plan.safetensors_architecture().ok_or_else(|| invalid("SafeTensors artifact omitted its architecture plan".into()))?;
                let complete = plan.prediction_extension().map(|e| e.complete_architecture()).unwrap_or(target);
                let resolution = complete.checkpoint_resolution().ok_or_else(|| invalid("SafeTensors source omitted its admitted checkpoint resolution".into()))?.clone();
                let target_resolution = target.checkpoint_resolution().ok_or_else(|| invalid("SafeTensors target omitted its admitted checkpoint resolution".into()))?;
                declarations.roles.insert(ArtifactSourceRole::Target, BTreeMap::from([(PhysicalArtifactId::Primary, target_resolution.source_keys().clone())]));
                if matches!(target.model(), crate::configuration::SafetensorsModelConfig::Muse(_)) {
                    for (role, keys) in crate::muse::checkpoint::media_source_roles(target_resolution.source_keys()) {
                        declarations.roles.insert(role, BTreeMap::from([(PhysicalArtifactId::Primary, keys)]));
                    }
                }
                if let Some(extension) = plan.prediction_extension() {
                    declarations.roles.insert(ArtifactSourceRole::Prediction, BTreeMap::from([(PhysicalArtifactId::Primary, extension.source_keys(target).map_err(|e| invalid(e.to_string()))?)]));
                }
                let physical = if inspection.metadata_provenance().is_some() {
                    PhysicalArtifactPreparation::MetadataOnly { resolution }
                } else {
                    PhysicalArtifactPreparation::Safetensors {
                        catalog: eredu_core::artifact::prepared_safetensors_metadata(inspection.tensors()).map_err(|e| invalid(e.to_string()))?,
                        shards: inspection.safetensors_shards().ok_or_else(|| invalid("SafeTensors artifact has no admitted shards".into()))?.clone(),
                        resolution,
                    }
                };
                declarations.artifacts.insert(PhysicalArtifactId::Primary, physical);
            }
            eredu_core::ArtifactFormat::Gguf => {
                let primary = plan.gguf_plan().ok_or_else(|| invalid("GGUF artifact omitted its architecture plan".into()))?;
                let validated = inspection.validated_gguf().ok_or_else(|| invalid("GGUF artifact omitted its validated headers".into()))?;
                if validated.companions().any(|(role, _)| *role != ArtifactCompanionRole::MediaProjector) { return Err(invalid("GGUF artifact retained an unsupported companion role".into())); }
                let checkpoint = primary.prepared_checkpoint(validated.checkpoint());
                let resolution = eredu_checkpoint::validation::resolve_gguf_plan(checkpoint, primary.checkpoint()).map_err(|e| invalid(format!("{e:?}")))?;
                let mapping = plan.gguf_media_projector().map_or(primary.tensor_mapping(), |p| p.primary_tensor_mapping());
                let catalog = eredu_checkpoint::gguf_store::GgufCatalog::from_resolved_checkpoint(checkpoint, &resolution, mapping).map_err(|e| invalid(e.to_string()))?;
                declarations.roles.insert(ArtifactSourceRole::Target, BTreeMap::from([(PhysicalArtifactId::Primary, catalog.keys().into_iter().collect())]));
                declarations.artifacts.insert(PhysicalArtifactId::Primary, PhysicalArtifactPreparation::Gguf { checkpoint: checkpoint.clone(), mapping: mapping.to_vec(), resolution });
                match (plan.gguf_media_projector(), validated.companion(&ArtifactCompanionRole::MediaProjector)) {
                    (Some(projector), Some(companion)) => {
                        let resolution = eredu_checkpoint::validation::resolve_gguf_plan(companion.checkpoint(), projector.checkpoint()).map_err(|e| invalid(format!("{e:?}")))?;
                        let catalog = eredu_checkpoint::gguf_store::GgufCatalog::from_resolved_checkpoint(companion.checkpoint(), &resolution, projector.tensor_mapping()).map_err(|e| invalid(e.to_string()))?;
                        let keys = catalog.keys().into_iter().collect::<BTreeSet<_>>();
                        let id = PhysicalArtifactId::Companion(ArtifactCompanionRole::MediaProjector);
                        declarations.roles.get_mut(&ArtifactSourceRole::Target).expect("target role").insert(id.clone(), keys.clone());
                        if matches!(projector.model(), crate::gguf_companion::GgufMediaProjectorConfig::Muse(_)) {
                            for (role, keys) in crate::muse::checkpoint::media_source_roles(&keys) {
                                declarations.roles.insert(role, BTreeMap::from([(id.clone(), keys)]));
                            }
                        } else {
                            declarations.roles.insert(ArtifactSourceRole::Vision, BTreeMap::from([(id.clone(), keys)]));
                        }
                        declarations.artifacts.insert(id, PhysicalArtifactPreparation::Gguf { checkpoint: companion.checkpoint().clone(), mapping: projector.tensor_mapping().to_vec(), resolution });
                    }
                    (None, None) => {}
                    _ => return Err(invalid("GGUF projector headers differ from admitted companion role".into())),
                }
            }
            _ => return Err(invalid("unsupported artifact container".into())),
        }
        if let Some(adapter) = plan.draft_adapter() {
            let id = PhysicalArtifactId::Companion(ArtifactCompanionRole::Named("draft_adapter".into()));
            let keys = adapter.resolution.source_keys().clone();
            declarations.roles.get_mut(&ArtifactSourceRole::Target).expect("target role").insert(id.clone(), keys.clone());
            declarations.roles.insert(ArtifactSourceRole::Component("draft_adapter"), BTreeMap::from([(id.clone(), keys)]));
            declarations.artifacts.insert(id, PhysicalArtifactPreparation::Safetensors {
                catalog: adapter.catalog.tensors().clone(), shards: adapter.catalog.admitted_shards(), resolution: adapter.resolution.clone(),
            });
        }
        for companion in plan.media_companions() {
            companion.verify_unchanged().map_err(|e| invalid(e.to_string()))?;
            let preparation = companion.preparation();
            let keys = preparation.resolution().source_keys();
            let existing = declarations.artifacts.iter().find_map(|(id, source)|
                source.contains_same_sources(&preparation, keys).then(|| id.clone()));
            let id = existing.clone().unwrap_or_else(|| crate::processor_plan::muse_media_sources::physical_id(companion));
            for (role, keys) in companion.roles() {
                // A selected source replaces this logical component, including
                // its previous container's aliases and encoding companions.
                if let Some(previous) = declarations.roles.remove(role) {
                    let target = declarations.roles.get_mut(&ArtifactSourceRole::Target).expect("target role");
                    for (physical, replaced) in previous {
                        if let Some(active) = target.get_mut(&physical) {
                            active.retain(|key| !replaced.contains(key));
                        }
                    }
                }
                declarations.roles.insert(*role, BTreeMap::from([(id.clone(), keys.clone())]));
                declarations.roles.get_mut(&ArtifactSourceRole::Target).expect("target role")
                    .entry(id.clone()).or_default().extend(keys.iter().cloned());
            }
            if existing.is_none() && declarations.artifacts.insert(id, preparation).is_some() {
                return Err(invalid("independent media sources overlap a physical companion role".into()));
            }
        }
        for artifacts in declarations.roles.values_mut() {
            artifacts.retain(|_, keys| !keys.is_empty());
        }
        Ok(Arc::new(declarations))
    }).clone()
}
