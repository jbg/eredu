//! Supplied immutable metadata and transport-independent artifact analysis.

use super::*;
pub use eredu_checkpoint::safetensors::SafetensorsHeader;
use eredu_checkpoint::safetensors::SafetensorsHeaderCatalog;
pub use eredu_gguf::CheckpointHeader as GgufHeader;
use eredu_gguf::CheckpointHeader;

/// Caller-supplied provenance for one immutable artifact revision.
/// This identifies the fetched objects; it does not authenticate their contents.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct MetadataProvenance {
    /// Repository, object namespace, or application-defined source identifier.
    pub source: String,
    /// Immutable revision or content-addressed manifest identifier.
    pub revision: String,
}

/// Complete metadata inputs for planning one checkpoint variant.
#[derive(Debug, Clone)]
pub struct ArtifactMetadata {
    /// Immutable source identity retained with the analysis.
    pub provenance: MetadataProvenance,
    /// Format-specific headers and configuration.
    pub checkpoint: CheckpointMetadata,
    /// Complete applicable processor sidecars, keyed by their relative names.
    /// An absent key declares that the sidecar is absent from this revision.
    pub sidecars: BTreeMap<String, Vec<u8>>,
}

/// Format-specific supplied checkpoint metadata.
#[derive(Debug, Clone)]
pub enum CheckpointMetadata {
    /// Hugging Face configuration and complete SafeTensors shard headers.
    SafeTensors {
        /// Exact `config.json` bytes.
        config: Vec<u8>,
        /// Exact index bytes; required for multiple shards.
        index: Option<Vec<u8>>,
        /// Complete header set, including each full source object length.
        headers: Vec<SafetensorsHeader>,
    },
    /// GGUF metadata and tensor descriptors for the primary and companions.
    Gguf {
        /// Ordered shard headers, starting with shard zero.
        headers: Vec<CheckpointHeader>,
        /// Explicitly selected architecture-declared companions.
        companions: Vec<GgufCompanionMetadata>,
    },
}

/// Metadata for an explicitly selected GGUF companion.
#[derive(Debug, Clone)]
pub struct GgufCompanionMetadata {
    /// Architecture-declared semantic role.
    pub role: GgufCompanionRole,
    /// Complete companion shard set, starting with shard zero.
    pub headers: Vec<CheckpointHeader>,
}

impl ArtifactMetadata {
    /// Container format supplied by the caller.
    pub fn format(&self) -> ArtifactFormat {
        match self.checkpoint {
            CheckpointMetadata::SafeTensors { .. } => ArtifactFormat::SafeTensors,
            CheckpointMetadata::Gguf { .. } => ArtifactFormat::Gguf,
        }
    }
}

/// Validates supplied headers and resolves architecture without filesystem access.
/// The result supports cold planning and explicitly rejects source preparation.
pub fn inspect_artifact_metadata<R: ModelConfigurationResolver>(
    metadata: &ArtifactMetadata,
    resolver: &R,
) -> Result<ArtifactInspection<R::ArtifactPlan>, ArtifactError> {
    if metadata.provenance.source.trim().is_empty()
        || metadata.provenance.revision.trim().is_empty()
    {
        return Err(ArtifactError::InvalidArtifactIdentity(
            "source and immutable revision are required".into(),
        ));
    }
    let path = Path::new(&metadata.provenance.source);
    let provenance = Some(metadata.provenance.clone());
    match &metadata.checkpoint {
        CheckpointMetadata::SafeTensors {
            config,
            index,
            headers,
        } => {
            let json: Value = serde_json::from_slice(config)?;
            let (configuration, plan) = resolver.resolve_safetensors(&json)?.into_parts();
            let catalog = SafetensorsHeaderCatalog::from_headers(headers, index.as_deref())?;
            let tensors =
                safetensors_tensor_catalog(catalog.tensors(), |name| catalog.tensor_offset(name))?;
            finish_safetensors_inspection(
                path,
                configuration,
                plan,
                tensors,
                None,
                &metadata.sidecars,
                resolver,
                provenance,
            )
        }
        CheckpointMetadata::Gguf {
            headers,
            companions,
        } => {
            let checkpoint = GgufCheckpoint::from_headers(headers, eredu_gguf::Limits::default())?;
            let architecture = checkpoint
                .metadata()
                .get("general.architecture")
                .and_then(MetadataValue::as_str)
                .ok_or(ArtifactError::MissingGgufArchitecture)?;
            let (configuration, plan) = resolver
                .resolve_gguf(architecture, &checkpoint)?
                .into_parts();
            let requirements = resolver.gguf_companion_requirements(architecture, &checkpoint)?;
            let mut selected = BTreeMap::new();
            for supplied in companions {
                let requirement = requirements
                    .iter()
                    .find(|requirement| requirement.role == supplied.role)
                    .ok_or_else(|| {
                        ArtifactError::InvalidArtifact(format!(
                            "undeclared GGUF companion role {:?}",
                            supplied.role
                        ))
                    })?;
                let companion =
                    GgufCheckpoint::from_headers(&supplied.headers, eredu_gguf::Limits::default())?;
                validate_gguf_container(&companion)?;
                if requirement.encoding == GgufCompanionEncoding::DenseRequired
                    && companion.tensors().any(|tensor| {
                        !matches!(
                            tensor.descriptor().ggml_type,
                            GgmlType::F32 | GgmlType::F16 | GgmlType::Bf16
                        )
                    })
                {
                    return Err(ArtifactError::InvalidArtifact(format!(
                        "GGUF companion {:?} requires dense tensors",
                        supplied.role
                    )));
                }
                let companion_path = PathBuf::from(&supplied.headers[0].member);
                if selected
                    .insert(
                        supplied.role.clone(),
                        ValidatedGgufCompanion {
                            path: companion_path,
                            checkpoint: companion,
                        },
                    )
                    .is_some()
                {
                    return Err(ArtifactError::InvalidArtifact(
                        "duplicate GGUF companion role".into(),
                    ));
                }
            }
            for requirement in &requirements {
                if requirement.required && !selected.contains_key(&requirement.role) {
                    return Err(ArtifactError::MissingRequiredGgufCompanion {
                        role: requirement.role.clone(),
                        filename_prefix: requirement.filename_prefix.clone(),
                        searched_directories: Vec::new(),
                    });
                }
            }
            finish_gguf_inspection(
                path,
                checkpoint,
                selected,
                configuration,
                plan,
                resolver,
                provenance,
            )
        }
    }
}
