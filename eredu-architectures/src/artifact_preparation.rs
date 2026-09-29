//! Exact physical artifact declarations, independent of logical source roles.
use eredu_checkpoint::{
    store::{CompositeCheckpointSource, RestrictedCheckpointSource, SharedCheckpointSource},
    validation::ResolvedCheckpointPlan,
};
use eredu_core::{artifact::ArtifactError, checkpoint::TensorCatalog, GgufCompanionRole};
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
        tensors: TensorCatalog,
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
                tensors,
                shards,
                resolution,
            } => eredu_core::artifact::open_prepared_safetensors_artifact(
                tensors,
                shards.clone(),
                resolution.clone(),
                cache,
            ),
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
    Companion(GgufCompanionRole),
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
                if let Some(extension) = plan.prediction_extension() {
                    declarations.roles.insert(ArtifactSourceRole::Prediction, BTreeMap::from([(PhysicalArtifactId::Primary, extension.source_keys(target).map_err(|e| invalid(e.to_string()))?)]));
                }
                let physical = if inspection.metadata_provenance().is_some() {
                    PhysicalArtifactPreparation::MetadataOnly { resolution }
                } else {
                    PhysicalArtifactPreparation::Safetensors {
                        tensors: inspection.tensors().clone(),
                        shards: inspection.safetensors_shards().ok_or_else(|| invalid("SafeTensors artifact has no admitted shards".into()))?.clone(),
                        resolution,
                    }
                };
                declarations.artifacts.insert(PhysicalArtifactId::Primary, physical);
            }
            eredu_core::ArtifactFormat::Gguf => {
                let primary = plan.gguf_plan().ok_or_else(|| invalid("GGUF artifact omitted its architecture plan".into()))?;
                let validated = inspection.validated_gguf().ok_or_else(|| invalid("GGUF artifact omitted its validated headers".into()))?;
                if validated.companions().any(|(role, _)| *role != GgufCompanionRole::MediaProjector) { return Err(invalid("GGUF artifact retained an unsupported companion role".into())); }
                let checkpoint = primary.prepared_checkpoint(validated.checkpoint());
                let resolution = eredu_checkpoint::validation::resolve_gguf_plan(checkpoint, primary.checkpoint()).map_err(|e| invalid(format!("{e:?}")))?;
                let mapping = plan.gguf_media_projector().map_or(primary.tensor_mapping(), |p| p.primary_tensor_mapping());
                let catalog = eredu_checkpoint::gguf_store::GgufCatalog::from_resolved_checkpoint(checkpoint, &resolution, mapping).map_err(|e| invalid(e.to_string()))?;
                declarations.roles.insert(ArtifactSourceRole::Target, BTreeMap::from([(PhysicalArtifactId::Primary, catalog.keys().into_iter().collect())]));
                declarations.artifacts.insert(PhysicalArtifactId::Primary, PhysicalArtifactPreparation::Gguf { checkpoint: checkpoint.clone(), mapping: mapping.to_vec(), resolution });
                match (plan.gguf_media_projector(), validated.companion(&GgufCompanionRole::MediaProjector)) {
                    (Some(projector), Some(companion)) => {
                        let resolution = eredu_checkpoint::validation::resolve_gguf_plan(companion.checkpoint(), projector.checkpoint()).map_err(|e| invalid(format!("{e:?}")))?;
                        let catalog = eredu_checkpoint::gguf_store::GgufCatalog::from_resolved_checkpoint(companion.checkpoint(), &resolution, projector.tensor_mapping()).map_err(|e| invalid(e.to_string()))?;
                        let keys = catalog.keys().into_iter().collect::<BTreeSet<_>>();
                        let id = PhysicalArtifactId::Companion(GgufCompanionRole::MediaProjector);
                        declarations.roles.get_mut(&ArtifactSourceRole::Target).expect("target role").insert(id.clone(), keys.clone());
                        declarations.roles.insert(ArtifactSourceRole::Vision, BTreeMap::from([(id.clone(), keys)]));
                        declarations.artifacts.insert(id, PhysicalArtifactPreparation::Gguf { checkpoint: companion.checkpoint().clone(), mapping: projector.tensor_mapping().to_vec(), resolution });
                    }
                    (None, None) => {}
                    _ => return Err(invalid("GGUF projector headers differ from admitted companion role".into())),
                }
            }
            _ => return Err(invalid("unsupported artifact container".into())),
        }
        Ok(Arc::new(declarations))
    }).clone()
}
