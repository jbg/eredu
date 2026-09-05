//! Architecture-aware construction of exact backend-neutral checkpoint sources.

use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

use eredu_checkpoint::{
    gguf_store::open_prepared_gguf_source,
    store::{
        CheckpointSource, CompositeCheckpointSource, RestrictedCheckpointSource,
        SharedCheckpointSource, StoreError, TensorMetadata,
    },
    validation::{resolve_gguf_plan, ResolvedCheckpointPlan},
};
use eredu_core::{
    artifact::{
        fingerprint_filesystem_artifact, fingerprint_safetensors_artifact, ArtifactError,
        ArtifactFile, ArtifactIdentity, GgufCompanionRole,
    },
    ArtifactFormat, ModelArtifact, ModelPreparationPlan,
};

use crate::{
    configuration::PredictionExtensionPlan, processor_plan::ArtifactArchitecturePlan,
    SelectedPreparation,
};

// Internal projection of the total execution selection onto source admission.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum MediaProjectorSourcePolicy {
    Forbidden,
    Allowed,
}

/// Exact resolved contracts retained with a prepared source graph.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct PreparedSourceResolutions {
    primary: ResolvedCheckpointPlan,
    target: ResolvedCheckpointPlan,
    companions: BTreeMap<GgufCompanionRole, ResolvedCheckpointPlan>,
    target_companions: BTreeMap<GgufCompanionRole, ResolvedCheckpointPlan>,
}

impl PreparedSourceResolutions {
    /// Contract used to construct the primary physical source.
    pub const fn primary(&self) -> &ResolvedCheckpointPlan {
        &self.primary
    }

    /// Contract authorizing the ordinary prediction target.
    pub const fn target(&self) -> &ResolvedCheckpointPlan {
        &self.target
    }

    /// Contract used to construct one separately admitted companion source.
    pub fn companion(&self, role: &GgufCompanionRole) -> Option<&ResolvedCheckpointPlan> {
        self.companions.get(role)
    }

    /// All companion contracts in deterministic semantic-role order.
    pub fn companions(
        &self,
    ) -> impl Iterator<Item = (&GgufCompanionRole, &ResolvedCheckpointPlan)> {
        self.companions.iter()
    }

    /// Companion contract that is an explicit member of the target graph.
    pub fn target_companion(&self, role: &GgufCompanionRole) -> Option<&ResolvedCheckpointPlan> {
        self.target_companions.get(role)
    }

    /// All companion contracts consumed by the exact target graph.
    pub fn target_companions(
        &self,
    ) -> impl Iterator<Item = (&GgufCompanionRole, &ResolvedCheckpointPlan)> {
        self.target_companions.iter()
    }
}

/// One exact architecture-selected source graph prepared before native materialization.
///
/// Every logical view shares the same underlying physical source objects. Creating
/// target or extension views therefore cannot multiply reader caches or reopen an
/// admitted artifact.
pub struct PreparedModelSources {
    selected: SelectedPreparation,
    inspection: eredu_core::ArtifactInspection<ArtifactArchitecturePlan>,
    graph: PreparedModelSourceGraph,
}

/// Exact source roles released only by consuming their paired total selection.
pub struct PreparedModelSourceGraph {
    source_identity: ArtifactIdentity,
    execution_identity: String,
    format: ArtifactFormat,
    architecture: ArtifactArchitecturePlan,
    prediction_extension: Option<PredictionExtensionPlan>,
    primary: SharedCheckpointSource,
    companions: BTreeMap<GgufCompanionRole, SharedCheckpointSource>,
    complete: SharedCheckpointSource,
    target: SharedCheckpointSource,
    extension: Option<SharedCheckpointSource>,
    resolutions: PreparedSourceResolutions,
    source_metadata: BTreeMap<String, TensorMetadata>,
}

impl PreparedModelSources {
    /// Authoritative total selection inseparably paired with these exact sources.
    pub const fn selected(&self) -> &SelectedPreparation {
        &self.selected
    }

    /// Consumes the authoritative pairing immediately before typed execution dispatch.
    pub fn into_parts(
        self,
    ) -> (
        SelectedPreparation,
        eredu_core::ArtifactInspection<ArtifactArchitecturePlan>,
        PreparedModelSourceGraph,
    ) {
        (self.selected, self.inspection, self.graph)
    }

    /// Exact prepared source graph paired with the selection.
    pub const fn graph(&self) -> &PreparedModelSourceGraph {
        &self.graph
    }

    /// Content-exact identity of the complete admitted physical source graph.
    pub const fn source_identity(&self) -> ArtifactIdentity {
        self.graph.source_identity()
    }

    /// Architecture identity retained by the selected neutral execution requirements.
    pub fn execution_identity(&self) -> &str {
        self.graph.execution_identity()
    }

    /// Admitted physical container format.
    pub const fn format(&self) -> ArtifactFormat {
        self.graph.format()
    }

    /// Target architecture paired with these exact source roles.
    pub const fn architecture(&self) -> &ArtifactArchitecturePlan {
        self.graph.architecture()
    }

    /// Selected embedded prediction extension, when requested and admitted.
    pub const fn prediction_extension(&self) -> Option<&PredictionExtensionPlan> {
        self.graph.prediction_extension()
    }

    /// Primary artifact source, excluding separately stored companions.
    pub fn primary(&self) -> &SharedCheckpointSource {
        self.graph.primary()
    }

    /// Separately stored source for one architecture-declared semantic role.
    pub fn companion(&self, role: &GgufCompanionRole) -> Option<&SharedCheckpointSource> {
        self.graph.companion(role)
    }

    /// Separately stored companions in deterministic semantic-role order.
    pub fn companions(
        &self,
    ) -> impl Iterator<Item = (&GgufCompanionRole, &SharedCheckpointSource)> {
        self.graph.companions()
    }

    /// Complete selected source graph used while materializing auxiliary roles.
    pub fn complete(&self) -> &SharedCheckpointSource {
        self.graph.complete()
    }

    /// Explicit ordinary-target projection, which cannot expose extension-only keys.
    pub fn target(&self) -> &SharedCheckpointSource {
        self.graph.target()
    }

    /// Explicit extension-only projection, when embedded prediction was selected.
    pub fn extension(&self) -> Option<&SharedCheckpointSource> {
        self.graph.extension()
    }

    /// Exact resolved contracts used to build and project this source graph.
    pub const fn resolutions(&self) -> &PreparedSourceResolutions {
        self.graph.resolutions()
    }

    /// Metadata snapshot for every key in the complete selected source graph.
    pub const fn source_metadata(&self) -> &BTreeMap<String, TensorMetadata> {
        self.graph.source_metadata()
    }
}

impl PreparedModelSourceGraph {
    /// Content-exact identity of the complete admitted physical source graph.
    pub const fn source_identity(&self) -> ArtifactIdentity {
        self.source_identity
    }

    /// Architecture identity retained by the selected neutral execution requirements.
    pub fn execution_identity(&self) -> &str {
        &self.execution_identity
    }

    /// Admitted physical container format.
    pub const fn format(&self) -> ArtifactFormat {
        self.format
    }

    /// Target architecture paired with these exact source roles.
    pub const fn architecture(&self) -> &ArtifactArchitecturePlan {
        &self.architecture
    }

    /// Selected embedded prediction extension, when requested and admitted.
    pub const fn prediction_extension(&self) -> Option<&PredictionExtensionPlan> {
        self.prediction_extension.as_ref()
    }

    /// Primary artifact source, excluding separately stored companions.
    pub fn primary(&self) -> &SharedCheckpointSource {
        &self.primary
    }

    /// Separately stored source for one architecture-declared semantic role.
    pub fn companion(&self, role: &GgufCompanionRole) -> Option<&SharedCheckpointSource> {
        self.companions.get(role)
    }

    /// Separately stored companions in deterministic semantic-role order.
    pub fn companions(
        &self,
    ) -> impl Iterator<Item = (&GgufCompanionRole, &SharedCheckpointSource)> {
        self.companions.iter()
    }

    /// Complete selected source graph used while materializing auxiliary roles.
    pub fn complete(&self) -> &SharedCheckpointSource {
        &self.complete
    }

    /// Explicit ordinary-target projection, which cannot expose extension-only keys.
    pub fn target(&self) -> &SharedCheckpointSource {
        &self.target
    }

    /// Explicit extension-only projection, when embedded prediction was selected.
    pub fn extension(&self) -> Option<&SharedCheckpointSource> {
        self.extension.as_ref()
    }

    /// Exact resolved contracts used to build and project this source graph.
    pub const fn resolutions(&self) -> &PreparedSourceResolutions {
        &self.resolutions
    }

    /// Metadata snapshot for every key in the complete selected source graph.
    pub const fn source_metadata(&self) -> &BTreeMap<String, TensorMetadata> {
        &self.source_metadata
    }
}

/// Failure while converting one selected portable artifact into exact neutral sources.
#[derive(Debug, thiserror::Error)]
pub enum PreparedModelSourcesError {
    /// The artifact and its architecture-owned admission proof disagree.
    #[error("invalid prepared source selection: {0}")]
    InvalidSelection(String),
    /// Architecture projection or artifact identity validation failed.
    #[error(transparent)]
    Artifact(#[from] ArtifactError),
    /// An exact checkpoint source or logical view could not be constructed.
    #[error(transparent)]
    Store(#[from] StoreError),
}

/// Consumes one admitted artifact into the only architecture-aware prepared source graph.
///
/// This operation performs the SafeTensors/GGUF choice once, constructs every
/// admitted physical source once, and publishes no backend-native resource.
/// Tensor payload conversion remains lazy behind checkpoint leases.
pub fn prepare_model_sources(
    plan: ModelPreparationPlan<ArtifactArchitecturePlan>,
    selected: SelectedPreparation,
) -> Result<PreparedModelSources, PreparedModelSourcesError> {
    if !selected
        .admission_token()
        .same_admission(&plan.inspection().admission_token())
    {
        return Err(PreparedModelSourcesError::InvalidSelection(
            "selected preparation originated from a different artifact inspection".into(),
        ));
    }
    if plan.admitted_session_capabilities() != selected.session_capabilities() {
        return Err(PreparedModelSourcesError::InvalidSelection(
            "selected session facilities differ from the admitted preparation plan".into(),
        ));
    }
    let selected_admission = selected.admission();
    if plan.policy() != selected_admission.request().policy()
        || plan.route() != selected_admission.route()
    {
        return Err(PreparedModelSourcesError::InvalidSelection(
            "selected preparation policy or route differs from the admitted preparation plan"
                .into(),
        ));
    }
    let max_cached_sources = selected.text_realization().residency().max_cached_shards();
    let media_projector = if selected.allows_media_projector() {
        MediaProjectorSourcePolicy::Allowed
    } else {
        MediaProjectorSourcePolicy::Forbidden
    };
    let selected_prediction_extension = selected.prediction_extension().cloned();
    let execution_identity = selected
        .text_realization()
        .requirements()
        .architecture_identity()
        .to_owned();
    let inspection = plan.inspection().clone();
    let complete_architecture = inspection.architecture_plan().clone();
    let projection = complete_architecture.prediction_target_projection()?;
    let (architecture, admitted_extension) = projection.map_or_else(
        || (complete_architecture.clone(), None),
        |(target, extension)| (target, Some(extension)),
    );
    let prediction_extension = match (admitted_extension, selected_prediction_extension) {
        (Some(admitted), Some(selected)) if admitted.same_admission(&selected) => Some(selected),
        (Some(_), Some(_)) => {
            return Err(PreparedModelSourcesError::InvalidSelection(
                "selected prediction extension differs from artifact admission".into(),
            ))
        }
        (None, Some(_)) => {
            return Err(PreparedModelSourcesError::InvalidSelection(
                "selected prediction extension has no artifact admission".into(),
            ))
        }
        (_, None) => None,
    };
    let execution_inspection = plan
        .inspection()
        .clone()
        .map_architecture_plan(|_| architecture.clone());
    let expected_execution_identity =
        match crate::replicated_text::replicated_text_execution_class(&execution_inspection)
            .map_err(|error| PreparedModelSourcesError::InvalidSelection(error.to_string()))?
        {
            crate::replicated_text::ReplicatedTextExecutionClass::Replicated(requirements) => {
                requirements.architecture_identity().to_owned()
            }
            crate::replicated_text::ReplicatedTextExecutionClass::Routed(requirements) => {
                requirements.text().architecture_identity().to_owned()
            }
            crate::replicated_text::ReplicatedTextExecutionClass::Composite(requirements) => {
                requirements.execution().architecture_identity().to_owned()
            }
        };
    if execution_identity != expected_execution_identity {
        return Err(PreparedModelSourcesError::InvalidSelection(format!(
            "selected execution identity {execution_identity:?} does not match inspected artifact identity {expected_execution_identity:?}"
        )));
    }
    let source_identity = source_graph_identity(plan.inspection(), &execution_identity)?;

    let graph = match plan.into_artifact() {
        ModelArtifact::SafeTensors {
            tensors, shards, ..
        } => prepare_safetensors_sources(
            source_identity,
            execution_identity,
            architecture,
            prediction_extension,
            tensors,
            shards,
            max_cached_sources,
        ),
        ModelArtifact::Gguf { validated, .. } => {
            if prediction_extension.is_some() {
                return Err(PreparedModelSourcesError::InvalidSelection(
                    "GGUF artifacts do not admit embedded prediction source projections".into(),
                ));
            }
            prepare_gguf_sources(
                source_identity,
                execution_identity,
                architecture,
                validated,
                max_cached_sources,
                media_projector,
            )
        }
        _ => Err(PreparedModelSourcesError::InvalidSelection(
            "unsupported artifact format for prepared model sources".into(),
        )),
    }?;
    Ok(PreparedModelSources {
        selected,
        inspection,
        graph,
    })
}

fn source_graph_identity(
    inspection: &eredu_core::ArtifactInspection<ArtifactArchitecturePlan>,
    execution_identity: &str,
) -> Result<ArtifactIdentity, ArtifactError> {
    match inspection.format() {
        ArtifactFormat::SafeTensors => fingerprint_safetensors_artifact(
            execution_identity,
            inspection.safetensors_shards().ok_or_else(|| {
                ArtifactError::InvalidArtifact(
                    "SafeTensors inspection omitted admitted shards".into(),
                )
            })?,
        ),
        ArtifactFormat::Gguf => {
            let validated = inspection.validated_gguf().ok_or_else(|| {
                ArtifactError::InvalidArtifact("GGUF inspection omitted its admission proof".into())
            })?;
            let mut files = Vec::new();
            files.extend(validated.checkpoint().shards().iter().map(|shard| {
                ArtifactFile::new(
                    format!("primary/split/{:05}", shard.split_no()),
                    shard.path(),
                )
            }));
            for (role, companion) in validated.companions() {
                let role = match role {
                    GgufCompanionRole::MediaProjector => "media-projector".to_owned(),
                    GgufCompanionRole::Named(name) => format!("named/{name}"),
                    _ => {
                        return Err(ArtifactError::InvalidArtifact(
                            "unsupported GGUF companion role in source identity".into(),
                        ))
                    }
                };
                files.extend(companion.checkpoint().shards().iter().map(|shard| {
                    ArtifactFile::new(
                        format!("companion/{role}/split/{:05}", shard.split_no()),
                        shard.path(),
                    )
                }));
            }
            fingerprint_filesystem_artifact(execution_identity, files)
        }
        _ => Err(ArtifactError::InvalidArtifact(
            "unsupported artifact format in prepared source identity".into(),
        )),
    }
}

fn prepare_safetensors_sources(
    source_identity: ArtifactIdentity,
    execution_identity: String,
    architecture: ArtifactArchitecturePlan,
    prediction_extension: Option<PredictionExtensionPlan>,
    tensors: eredu_core::checkpoint::TensorCatalog,
    shards: eredu_checkpoint::safetensors::SafetensorsShards,
    max_cached_shards: usize,
) -> Result<PreparedModelSourceGraph, PreparedModelSourcesError> {
    let target_architecture = architecture.safetensors_architecture().ok_or_else(|| {
        PreparedModelSourcesError::InvalidSelection(
            "SafeTensors artifact omitted its architecture plan".into(),
        )
    })?;
    let target_resolution = target_architecture
        .checkpoint_resolution()
        .ok_or_else(|| {
            PreparedModelSourcesError::InvalidSelection(
                "SafeTensors target omitted its admitted checkpoint resolution".into(),
            )
        })?
        .clone();
    let source_resolution = prediction_extension
        .as_ref()
        .map(PredictionExtensionPlan::complete_architecture)
        .unwrap_or(target_architecture)
        .checkpoint_resolution()
        .ok_or_else(|| {
            PreparedModelSourcesError::InvalidSelection(
                "SafeTensors source omitted its admitted checkpoint resolution".into(),
            )
        })?
        .clone();
    let primary = eredu_core::artifact::open_prepared_safetensors_artifact(
        &tensors,
        shards,
        source_resolution.clone(),
        max_cached_shards,
    )?;
    let complete = Arc::clone(&primary);
    let (target, extension) = match prediction_extension.as_ref() {
        Some(extension) => {
            let extension_keys = extension.source_keys(target_architecture)?;
            let target_keys = target_resolution.source_keys().clone();
            projected_prediction_views(&complete, target_keys, extension_keys)?
        }
        None => (Arc::clone(&complete), None),
    };
    let source_metadata = metadata_snapshot(complete.as_ref())?;
    Ok(PreparedModelSourceGraph {
        source_identity,
        execution_identity,
        format: ArtifactFormat::SafeTensors,
        architecture,
        prediction_extension,
        primary,
        companions: BTreeMap::new(),
        complete,
        target,
        extension,
        resolutions: PreparedSourceResolutions {
            primary: source_resolution,
            target: target_resolution,
            companions: BTreeMap::new(),
            target_companions: BTreeMap::new(),
        },
        source_metadata,
    })
}

fn prepare_gguf_sources(
    source_identity: ArtifactIdentity,
    execution_identity: String,
    architecture: ArtifactArchitecturePlan,
    validated: eredu_core::ValidatedGguf,
    max_cached_readers: usize,
    media_projector: MediaProjectorSourcePolicy,
) -> Result<PreparedModelSourceGraph, PreparedModelSourcesError> {
    let primary_plan = architecture.gguf_plan().ok_or_else(|| {
        PreparedModelSourcesError::InvalidSelection(
            "GGUF artifact omitted its architecture plan".into(),
        )
    })?;
    let projector_plan = architecture.gguf_media_projector();
    let (checkpoint, mut admitted_companions) = validated.into_parts();
    let admitted_projector = admitted_companions.remove(&GgufCompanionRole::MediaProjector);
    if let Some(role) = admitted_companions.keys().next() {
        return Err(PreparedModelSourcesError::InvalidSelection(format!(
            "GGUF artifact retained an unsupported companion role {role:?}"
        )));
    }
    if media_projector == MediaProjectorSourcePolicy::Forbidden
        && (projector_plan.is_some() || admitted_projector.is_some())
    {
        return Err(PreparedModelSourcesError::InvalidSelection(
            "selected execution cannot consume the admitted GGUF media projector".into(),
        ));
    }
    let admitted_projector = match (projector_plan, admitted_projector) {
        (Some(plan), Some(checkpoint)) => Some((plan, checkpoint)),
        (None, None) => None,
        (Some(_), None) => {
            return Err(PreparedModelSourcesError::InvalidSelection(
                "GGUF projector plan omitted its admitted companion checkpoint".into(),
            ))
        }
        (None, Some(_)) => {
            return Err(PreparedModelSourcesError::InvalidSelection(
                "GGUF projector checkpoint omitted its typed architecture plan".into(),
            ))
        }
    };

    let primary_mapping = admitted_projector
        .as_ref()
        .map_or(primary_plan.tensor_mapping(), |(plan, _)| {
            plan.primary_tensor_mapping()
        });
    let primary_resolution =
        resolve_gguf_plan(&checkpoint, primary_plan.checkpoint()).map_err(|validation| {
            PreparedModelSourcesError::InvalidSelection(format!(
                "GGUF primary checkpoint contract no longer resolves: {validation:?}"
            ))
        })?;
    let primary: SharedCheckpointSource = Arc::new(open_prepared_gguf_source(
        checkpoint,
        primary_plan.checkpoint(),
        primary_mapping,
        max_cached_readers,
    )?);
    let mut companions = BTreeMap::new();
    let mut companion_resolutions = BTreeMap::new();
    if let Some((plan, admitted)) = admitted_projector {
        let resolution =
            resolve_gguf_plan(admitted.checkpoint(), plan.checkpoint()).map_err(|validation| {
                PreparedModelSourcesError::InvalidSelection(format!(
                    "GGUF media-projector contract no longer resolves: {validation:?}"
                ))
            })?;
        let source: SharedCheckpointSource = Arc::new(open_prepared_gguf_source(
            admitted.checkpoint().clone(),
            plan.checkpoint(),
            plan.tensor_mapping(),
            max_cached_readers,
        )?);
        companions.insert(GgufCompanionRole::MediaProjector, source);
        companion_resolutions.insert(GgufCompanionRole::MediaProjector, resolution);
    }
    let complete = if companions.is_empty() {
        Arc::clone(&primary)
    } else {
        Arc::new(CompositeCheckpointSource::new(
            std::iter::once(Arc::clone(&primary)).chain(companions.values().cloned()),
        )?)
    };
    let source_metadata = metadata_snapshot(complete.as_ref())?;
    let target_companion_resolutions = companion_resolutions.clone();
    Ok(PreparedModelSourceGraph {
        source_identity,
        execution_identity,
        format: ArtifactFormat::Gguf,
        architecture,
        prediction_extension: None,
        primary: Arc::clone(&primary),
        companions,
        complete: Arc::clone(&complete),
        target: complete,
        extension: None,
        resolutions: PreparedSourceResolutions {
            primary: primary_resolution.clone(),
            target: primary_resolution,
            companions: companion_resolutions,
            target_companions: target_companion_resolutions,
        },
        source_metadata,
    })
}

fn projected_prediction_views(
    complete: &SharedCheckpointSource,
    target_keys: BTreeSet<String>,
    extension_keys: BTreeSet<String>,
) -> Result<(SharedCheckpointSource, Option<SharedCheckpointSource>), StoreError> {
    if !target_keys.is_disjoint(&extension_keys) {
        return Err(StoreError::Internal(
            "prediction target and extension source projections overlap".into(),
        ));
    }
    let complete_keys = complete.source_keys().into_iter().collect::<BTreeSet<_>>();
    let projected = target_keys
        .union(&extension_keys)
        .cloned()
        .collect::<BTreeSet<_>>();
    if projected != complete_keys {
        return Err(StoreError::Internal(
            "prediction target and extension projections do not cover the selected source".into(),
        ));
    }
    let target: SharedCheckpointSource = Arc::new(RestrictedCheckpointSource::including(
        Arc::clone(complete),
        "prediction-target",
        target_keys,
    )?);
    let extension: SharedCheckpointSource = Arc::new(RestrictedCheckpointSource::including(
        Arc::clone(complete),
        "prediction-extension",
        extension_keys,
    )?);
    Ok((target, Some(extension)))
}

fn metadata_snapshot(
    source: &dyn CheckpointSource,
) -> Result<BTreeMap<String, TensorMetadata>, StoreError> {
    source
        .source_keys()
        .into_iter()
        .map(|key| source.source_metadata(&key).map(|metadata| (key, metadata)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use eredu_checkpoint::store::{
        CheckpointLease, EncodedTensorLease, MemoryWeightStore, ReadPolicy, TensorReadRequest,
        TensorSelection, WeightStoreDiagnostics,
    };
    use safetensors::tensor::Dtype;
    use std::{
        collections::{BTreeMap, HashMap, HashSet},
        fs::File,
        sync::atomic::{AtomicUsize, Ordering},
    };

    fn write_f32_gguf(
        path: &std::path::Path,
        metadata: &BTreeMap<String, eredu_gguf::MetadataValue>,
        plan: &eredu_checkpoint::schema::GgufCheckpointPlan,
    ) {
        use eredu_gguf::{GgmlType, TensorInput, Writer};

        let tensors = plan
            .common_tensors
            .iter()
            .chain(
                plan.layout_groups
                    .iter()
                    .filter(|group| group.required)
                    .filter_map(|group| group.variants.first())
                    .flat_map(|variant| variant.tensors.iter()),
            )
            .filter(|constraint| {
                constraint.requirement == eredu_checkpoint::schema::TensorRequirement::Required
            })
            .map(|constraint| {
                let dimensions = constraint
                    .shape
                    .iter()
                    .rev()
                    .map(|dimension| u64::try_from(*dimension).unwrap())
                    .collect::<Vec<_>>();
                let bytes = vec![0_u8; constraint.shape.iter().product::<usize>() * 4];
                (constraint.key.clone(), dimensions, bytes)
            })
            .collect::<Vec<_>>();
        let inputs = tensors
            .iter()
            .map(|(name, dimensions, bytes)| TensorInput {
                name,
                dimensions,
                ggml_type: GgmlType::F32,
                data: bytes,
            })
            .collect::<Vec<_>>();
        Writer::default()
            .write(File::create(path).unwrap(), metadata, &inputs)
            .unwrap();
    }

    fn gemma4_gguf_fixture() -> tempfile::TempDir {
        use eredu_gguf::{MetadataArray, MetadataValue};

        let root = tempfile::tempdir().unwrap();
        let model_metadata = BTreeMap::from([
            (
                "general.architecture".into(),
                MetadataValue::String("gemma4".into()),
            ),
            ("gemma4.block_count".into(), MetadataValue::Uint32(2)),
            ("gemma4.embedding_length".into(), MetadataValue::Uint32(4)),
            (
                "gemma4.attention.head_count".into(),
                MetadataValue::Uint32(2),
            ),
            (
                "gemma4.attention.key_length".into(),
                MetadataValue::Uint32(2),
            ),
            (
                "gemma4.attention.key_length_swa".into(),
                MetadataValue::Uint32(2),
            ),
            (
                "gemma4.attention.shared_kv_layers".into(),
                MetadataValue::Uint32(0),
            ),
            (
                "gemma4.attention.layer_norm_rms_epsilon".into(),
                MetadataValue::Float32(1e-6),
            ),
            ("gemma4.vocab_size".into(), MetadataValue::Uint32(8)),
            ("gemma4.context_length".into(), MetadataValue::Uint32(32)),
            (
                "gemma4.final_logit_softcapping".into(),
                MetadataValue::Float32(30.0),
            ),
            (
                "gemma4.feed_forward_length".into(),
                MetadataValue::Array(MetadataArray::Uint32(vec![8, 8])),
            ),
            (
                "gemma4.attention.head_count_kv".into(),
                MetadataValue::Array(MetadataArray::Uint32(vec![1, 1])),
            ),
            (
                "gemma4.attention.sliding_window_pattern".into(),
                MetadataValue::Array(MetadataArray::Bool(vec![false, false])),
            ),
            ("gemma4.image_token_id".into(), MetadataValue::Int32(2)),
        ]);
        let model_hash = model_metadata
            .clone()
            .into_iter()
            .collect::<HashMap<_, _>>();
        let catalog = HashSet::from([
            "output.weight".to_owned(),
            "blk.0.attn_k.weight".to_owned(),
            "blk.0.attn_v.weight".to_owned(),
            "blk.1.attn_k.weight".to_owned(),
            "blk.1.attn_v.weight".to_owned(),
        ]);
        let text = crate::gemma4::ModelArgs::from_gguf_metadata(&catalog, &model_hash).unwrap();
        let model_plan = crate::gemma4::gguf_plan(&text).unwrap();
        write_f32_gguf(
            &root.path().join("model.gguf"),
            &model_metadata,
            &model_plan,
        );

        let projector_metadata = BTreeMap::from([
            (
                "general.architecture".into(),
                MetadataValue::String("clip".into()),
            ),
            (
                "general.type".into(),
                MetadataValue::String("mmproj".into()),
            ),
            ("clip.has_vision_encoder".into(), MetadataValue::Bool(true)),
            ("clip.has_audio_encoder".into(), MetadataValue::Bool(false)),
            (
                "clip.vision.projector_type".into(),
                MetadataValue::String("gemma4".into()),
            ),
            (
                "clip.vision.embedding_length".into(),
                MetadataValue::Uint32(4),
            ),
            (
                "clip.vision.feed_forward_length".into(),
                MetadataValue::Uint32(8),
            ),
            ("clip.vision.block_count".into(), MetadataValue::Uint32(1)),
            (
                "clip.vision.attention.head_count".into(),
                MetadataValue::Uint32(1),
            ),
            (
                "clip.vision.attention.head_count_kv".into(),
                MetadataValue::Uint32(1),
            ),
            (
                "clip.vision.attention.key_length".into(),
                MetadataValue::Uint32(4),
            ),
            ("clip.vision.patch_size".into(), MetadataValue::Uint32(2)),
            (
                "clip.vision.pooling_kernel_size".into(),
                MetadataValue::Uint32(2),
            ),
            (
                "clip.vision.position_embedding_size".into(),
                MetadataValue::Uint32(4),
            ),
            (
                "clip.vision.attention.layer_norm_rms_epsilon".into(),
                MetadataValue::Float32(1e-6),
            ),
        ]);
        let projector_hash = projector_metadata
            .clone()
            .into_iter()
            .collect::<HashMap<_, _>>();
        let family =
            crate::gemma4::family_from_gguf_metadata(text, &model_hash, Some(&projector_hash))
                .unwrap();
        let projector_plan = crate::gemma4::mmproj_gguf_plan(&family).unwrap();
        write_f32_gguf(
            &root.path().join("mmproj.gguf"),
            &projector_metadata,
            &projector_plan,
        );
        root
    }

    struct LeaseCountingSource {
        source: MemoryWeightStore,
        acquisitions: Arc<AtomicUsize>,
    }

    impl CheckpointSource for LeaseCountingSource {
        fn source_keys(&self) -> Vec<String> {
            self.source.source_keys()
        }

        fn source_metadata(&self, key: &str) -> Result<TensorMetadata, StoreError> {
            self.source.source_metadata(key)
        }

        fn acquire_lease(&self, request: TensorReadRequest) -> Result<CheckpointLease, StoreError> {
            self.acquisitions.fetch_add(1, Ordering::Relaxed);
            self.source.acquire_lease(request)
        }

        fn source_diagnostics(&self) -> Result<WeightStoreDiagnostics, StoreError> {
            self.source.source_diagnostics()
        }
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    #[test]
    fn prediction_views_are_disjoint_exact_and_share_one_source() {
        let acquisitions = Arc::new(AtomicUsize::new(0));
        let source: SharedCheckpointSource = Arc::new(LeaseCountingSource {
            source: MemoryWeightStore::from_safetensors([
                (
                    "target.weight".into(),
                    Dtype::F32,
                    vec![1],
                    f32_bytes(&[1.0]),
                ),
                (
                    "extension.weight".into(),
                    Dtype::F32,
                    vec![1],
                    f32_bytes(&[2.0]),
                ),
            ])
            .unwrap(),
            acquisitions: Arc::clone(&acquisitions),
        });
        let (target, extension) = projected_prediction_views(
            &source,
            BTreeSet::from(["target.weight".into()]),
            BTreeSet::from(["extension.weight".into()]),
        )
        .unwrap();
        let extension = extension.unwrap();

        assert_eq!(target.source_keys(), ["target.weight"]);
        assert_eq!(extension.source_keys(), ["extension.weight"]);
        assert!(target.source_metadata("extension.weight").is_err());
        assert!(extension.source_metadata("target.weight").is_err());
        assert_eq!(
            target
                .acquire_lease(TensorReadRequest {
                    key: "target.weight".into(),
                    selection: TensorSelection::Full,
                    policy: ReadPolicy::RequireBounded,
                })
                .unwrap()
                .encoded_bytes()
                .unwrap(),
            f32_bytes(&[1.0])
        );
        assert_eq!(
            extension
                .acquire_lease(TensorReadRequest {
                    key: "extension.weight".into(),
                    selection: TensorSelection::Full,
                    policy: ReadPolicy::RequireBounded,
                })
                .unwrap()
                .encoded_bytes()
                .unwrap(),
            f32_bytes(&[2.0])
        );
        assert_eq!(acquisitions.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn prediction_views_reject_overlap_and_incomplete_partition() {
        let source: SharedCheckpointSource = Arc::new(
            MemoryWeightStore::from_safetensors([
                ("a".into(), Dtype::F32, vec![1], f32_bytes(&[1.0])),
                ("b".into(), Dtype::F32, vec![1], f32_bytes(&[2.0])),
            ])
            .unwrap(),
        );
        assert!(projected_prediction_views(
            &source,
            BTreeSet::from(["a".into()]),
            BTreeSet::from(["a".into(), "b".into()]),
        )
        .is_err());
        assert!(
            projected_prediction_views(&source, BTreeSet::from(["a".into()]), BTreeSet::new(),)
                .is_err()
        );
    }

    #[test]
    fn gguf_primary_target_and_companion_roles_are_exact_and_payload_lazy() {
        let root = gemma4_gguf_fixture();
        let inspection =
            crate::configuration::inspect_artifact(root.path().join("model.gguf")).unwrap();
        let selected = crate::select_preparation(
            &inspection,
            &eredu_runtime::NormalizedLoadRequest::default(),
            &crate::preparation_selection::tests::BoundedIndependentAdapter::default(),
        )
        .unwrap();
        let plan = eredu_core::plan_model_preparation(
            inspection,
            eredu_core::PreparationPolicy::new(None, eredu_core::ResidencyRequest::FullyResident),
            selected.session_capabilities(),
        )
        .unwrap();
        let sources = prepare_model_sources(plan, selected).unwrap();

        let role = GgufCompanionRole::MediaProjector;
        let primary = sources.primary().source_keys();
        let companion = sources.companion(&role).unwrap().source_keys();
        let complete = sources.complete().source_keys();
        assert_eq!(sources.format(), ArtifactFormat::Gguf);
        assert!(!primary.is_empty());
        assert!(!companion.is_empty());
        assert!(primary.iter().all(|key| !companion.contains(key)));
        assert_eq!(complete.len(), primary.len() + companion.len());
        assert_eq!(sources.target().source_keys(), complete);
        assert!(sources.extension().is_none());
        assert!(sources.resolutions().companion(&role).is_some());
        for source in [
            sources.primary(),
            sources.companion(&role).unwrap(),
            sources.target(),
        ] {
            let diagnostics = source.source_diagnostics().unwrap();
            assert_eq!(diagnostics.physical_reads, 0);
            assert_eq!(diagnostics.physical_read_bytes, 0);
            assert!(diagnostics.payload_shard_paths.is_empty());
        }

        let forbidden =
            crate::configuration::inspect_artifact(root.path().join("model.gguf")).unwrap();
        let (_llama_root, llama) = crate::preparation_selection::tests::inspected_llama();
        let selected = crate::select_preparation(
            &llama,
            &eredu_runtime::NormalizedLoadRequest::default(),
            &crate::preparation_selection::tests::BoundedIndependentAdapter::default(),
        )
        .unwrap();
        let forbidden = eredu_core::plan_model_preparation(
            forbidden,
            eredu_core::PreparationPolicy::new(None, eredu_core::ResidencyRequest::FullyResident),
            selected.session_capabilities(),
        )
        .unwrap();
        assert!(matches!(
            prepare_model_sources(forbidden, selected),
            Err(PreparedModelSourcesError::InvalidSelection(_))
        ));
    }
}
