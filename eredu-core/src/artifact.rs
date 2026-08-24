//! Backend-neutral model artifact inspection and preparation planning.
//!
//! Inspection parses configuration and checkpoint headers only. It never
//! materializes tensor payloads or creates a device/runtime object.

use crate::checkpoint::{TensorCatalog, TensorDescriptor, TensorDtype, TensorStorage};
use eredu_gguf::{Checkpoint as GgufCheckpoint, MetadataValue};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    io::Read,
    path::{Component, Path, PathBuf},
};

/// Backend-neutral loader contract required by an inspected artifact.
///
/// Architecture registries select this protocol while resolving family identity.
/// Core uses it to route preparation without knowing any concrete model family.
#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LoadingProtocol {
    /// Ordinary whole-model preparation followed by a model session.
    Model,
    /// Realtime multi-stream preparation followed by a realtime session.
    Realtime,
}

/// Artifact container selected during inspection.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactFormat {
    /// Hugging Face SafeTensors directory.
    SafeTensors,
    /// Single-file or canonically sharded GGUF checkpoint.
    Gguf,
}

/// Resolved portable model configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelConfiguration {
    /// Submitted outer `model_type` or GGUF architecture.
    pub declared_model_type: String,
    /// Nested text architecture selected for dispatch where applicable.
    pub effective_model_type: String,
    /// Open canonical family name supplied by the architecture registry.
    pub family: String,
    /// Neutral loader contract selected by the architecture registry.
    pub loading_protocol: LoadingProtocol,
    /// Raw JSON configuration for SafeTensors artifacts.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub json: Option<Value>,
}

/// Architecture-owned resolver used by neutral artifact inspection.
///
/// Core owns the transport contract but deliberately does not recognize model
/// family aliases, GGUF architecture spellings, or nested configuration wrappers.
pub trait ModelConfigurationResolver {
    /// Architecture-owned state retained from artifact inspection through materialization.
    type ArtifactPlan: Clone + std::fmt::Debug + Default;

    /// Resolves one Hugging Face `config.json` value to its canonical family.
    fn resolve_safetensors(&self, json: &Value) -> Result<ModelConfiguration, ArtifactError>;

    /// Resolves and structurally admits one GGUF architecture and checkpoint.
    fn resolve_gguf(
        &self,
        architecture: &str,
        checkpoint: &GgufCheckpoint,
    ) -> Result<ModelConfiguration, ArtifactError>;

    /// Declares the sibling artifacts required by one admitted GGUF architecture.
    fn gguf_companion_requirements(
        &self,
        architecture: &str,
        checkpoint: &GgufCheckpoint,
    ) -> Result<Vec<GgufCompanionRequirement>, ArtifactError>;

    /// Derives typed architecture state from the exact artifact admitted by inspection.
    fn artifact_plan(
        &self,
        _path: &Path,
        _format: ArtifactFormat,
        _configuration: &ModelConfiguration,
        _validated_gguf: Option<&ValidatedGguf>,
    ) -> Result<Self::ArtifactPlan, ArtifactError> {
        Ok(Self::ArtifactPlan::default())
    }
}

/// Semantic identity of a separately stored GGUF companion.
#[derive(Debug, Clone, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum GgufCompanionRole {
    /// Media encoder and projection weights consumed with the decoder.
    MediaProjector,
    /// Architecture-declared role not covered by a common semantic variant.
    Named(String),
}

/// Encoding policy used to select among matching GGUF companions.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum GgufCompanionEncoding {
    /// Require a checkpoint whose tensor catalog contains no quantized weights.
    DenseRequired,
    /// Prefer a dense checkpoint, but admit one unambiguous quantized checkpoint.
    DensePreferred,
}

/// Architecture-declared filename and encoding policy for one GGUF companion.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct GgufCompanionRequirement {
    role: GgufCompanionRole,
    required: bool,
    filename_prefix: String,
    parent_search_depth: usize,
    encoding: GgufCompanionEncoding,
}

impl GgufCompanionRequirement {
    /// Creates one validated companion requirement.
    pub fn new(
        role: GgufCompanionRole,
        required: bool,
        filename_prefix: impl Into<String>,
        parent_search_depth: usize,
        encoding: GgufCompanionEncoding,
    ) -> Result<Self, ArtifactError> {
        let filename_prefix = filename_prefix.into();
        if filename_prefix.trim().is_empty()
            || matches!(&role, GgufCompanionRole::Named(name) if name.trim().is_empty())
        {
            return Err(ArtifactError::InvalidArtifact(
                "GGUF companion roles and filename prefixes must be non-empty".into(),
            ));
        }
        Ok(Self {
            role,
            required,
            filename_prefix,
            parent_search_depth,
            encoding,
        })
    }

    /// Semantic role of the resolved artifact.
    pub fn role(&self) -> &GgufCompanionRole {
        &self.role
    }
}

/// Parses an optional GGUF integer metadata value as lossless `u32` values.
pub fn gguf_u32_metadata_values(
    key: &str,
    value: Option<&MetadataValue>,
) -> Result<Vec<u32>, ArtifactError> {
    let Some(value) = value else {
        return Ok(Vec::new());
    };
    value.to_u32_vec().ok_or_else(|| {
        ArtifactError::InvalidArtifact(format!(
            "GGUF metadata key {key:?} must contain an integer or integer array whose values fit in u32"
        ))
    })
}

/// Header-only artifact inspection result.
#[derive(Debug, Clone)]
pub struct ArtifactInspection<P = ()> {
    path: PathBuf,
    format: ArtifactFormat,
    configuration: ModelConfiguration,
    tensors: TensorCatalog,
    validated_gguf: Option<ValidatedGguf>,
    architecture_plan: P,
}

/// Portable GGUF facts admitted by core inspection.
///
/// Backends may enrich this result with runtime-specific compatibility checks,
/// but do not need to repeat the portable metadata and catalog validation.
#[derive(Debug, Clone)]
pub struct ValidatedGguf {
    checkpoint: GgufCheckpoint,
    companions: BTreeMap<GgufCompanionRole, ValidatedGgufCompanion>,
}

/// One exact sibling GGUF admitted during portable inspection.
#[derive(Debug, Clone)]
pub struct ValidatedGgufCompanion {
    path: PathBuf,
    checkpoint: GgufCheckpoint,
}

impl ValidatedGgufCompanion {
    /// Resolved companion path.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Header-only checkpoint admitted by portable inspection.
    pub fn checkpoint(&self) -> &GgufCheckpoint {
        &self.checkpoint
    }
}

impl ValidatedGguf {
    /// Header-only checkpoint admitted by portable inspection.
    pub fn checkpoint(&self) -> &GgufCheckpoint {
        &self.checkpoint
    }

    /// Returns the resolved companion for one semantic role.
    pub fn companion(&self, role: &GgufCompanionRole) -> Option<&ValidatedGgufCompanion> {
        self.companions.get(role)
    }

    /// Returns every resolved companion in stable role order.
    pub fn companions(
        &self,
    ) -> impl Iterator<Item = (&GgufCompanionRole, &ValidatedGgufCompanion)> {
        self.companions.iter()
    }
}

impl<P> ArtifactInspection<P> {
    /// Submitted artifact path.
    pub fn path(&self) -> &Path {
        &self.path
    }
    /// Detected artifact format.
    pub const fn format(&self) -> ArtifactFormat {
        self.format
    }
    /// Resolved model configuration.
    pub fn configuration(&self) -> &ModelConfiguration {
        &self.configuration
    }
    /// Validated portable tensor catalog.
    pub fn tensors(&self) -> &TensorCatalog {
        &self.tensors
    }
    /// Validated portable GGUF result, when applicable.
    pub fn validated_gguf(&self) -> Option<&ValidatedGguf> {
        self.validated_gguf.as_ref()
    }
    /// Portable GGUF checkpoint handle, when applicable.
    pub fn gguf_checkpoint(&self) -> Option<&GgufCheckpoint> {
        self.validated_gguf().map(ValidatedGguf::checkpoint)
    }
    /// Architecture-owned state derived from the exact inspected artifact.
    pub fn architecture_plan(&self) -> &P {
        &self.architecture_plan
    }
}

/// Requested load-time weight transformation.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum QuantizationRequest {
    /// Per-group affine integer quantization.
    Affine {
        /// Scalars per quantization group.
        group_size: u32,
        /// Packed bits per scalar.
        bits: u8,
    },
    /// Microscaling FP4 with E2M1 values and E8M0 scales.
    MxFp4,
}

/// Coarse backend-neutral weight residency request.
#[derive(Debug, Clone, Copy, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResidencyRequest {
    /// Keep all owned weights resident.
    #[default]
    FullyResident,
    /// Keep a bounded layer window resident and stage remaining layers from host.
    LayerwiseHost,
    /// Stream bounded layer units from disk.
    DenseDiskStream,
    /// Manage routed experts independently of non-expert layers.
    ExpertCache,
}

/// Backend-neutral inputs to materialization-route selection.
#[derive(Debug, Clone, Copy, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct PreparationPolicy {
    /// Optional requested load-time transformation.
    pub quantization: Option<QuantizationRequest>,
    /// Requested residency family.
    pub residency: ResidencyRequest,
    /// Exact parallel topology selected for materialization, when explicitly configured.
    pub topology: Option<crate::topology::ParallelTopology>,
}

/// Canonical materialization recipe selected by the core planner.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MaterializationRoute {
    /// Resident materialization, optionally with a load-time transform.
    Resident,
    /// Bounded non-expert layer materialization.
    Layerwise,
    /// Independent routed-expert materialization.
    ExpertCache,
}

/// Fully inspected input supplied to one selected backend for materialization.
#[derive(Debug, Clone)]
pub struct ModelPreparationPlan<P = ()> {
    inspection: ArtifactInspection<P>,
    policy: PreparationPolicy,
    route: MaterializationRoute,
}

impl<P> ModelPreparationPlan<P> {
    /// Header-only inspection owned by the plan.
    pub fn inspection(&self) -> &ArtifactInspection<P> {
        &self.inspection
    }
    /// Validated caller policy.
    pub const fn policy(&self) -> PreparationPolicy {
        self.policy
    }
    /// Canonical materialization route.
    pub const fn route(&self) -> MaterializationRoute {
        self.route
    }
    /// Consume the plan into its portable artifact, architecture state, and policy.
    pub fn into_parts(self) -> (ModelArtifact, P, PreparationPolicy, MaterializationRoute) {
        let artifact = match self.inspection.validated_gguf {
            Some(validated) => ModelArtifact::Gguf {
                path: self.inspection.path,
                configuration: self.inspection.configuration,
                tensors: self.inspection.tensors,
                checkpoint: validated.checkpoint,
                companions: validated.companions,
            },
            None => ModelArtifact::SafeTensors {
                path: self.inspection.path,
                configuration: self.inspection.configuration,
                tensors: self.inspection.tensors,
            },
        };
        (
            artifact,
            self.inspection.architecture_plan,
            self.policy,
            self.route,
        )
    }
}

/// Portable artifact payload consumed by a backend materializer.
#[derive(Debug, Clone)]
pub enum ModelArtifact {
    /// SafeTensors directory and validated header catalog.
    SafeTensors {
        /// Model directory.
        path: PathBuf,
        /// Resolved configuration.
        configuration: ModelConfiguration,
        /// Header-only tensor catalog.
        tensors: TensorCatalog,
    },
    /// Validated GGUF checkpoint handle and metadata-derived configuration.
    Gguf {
        /// Submitted first-shard path.
        path: PathBuf,
        /// Resolved configuration.
        configuration: ModelConfiguration,
        /// Header-only tensor catalog.
        tensors: TensorCatalog,
        /// Pure-Rust checkpoint handle used by backend materialization.
        checkpoint: GgufCheckpoint,
        /// Exact sibling checkpoints selected during portable inspection.
        companions: BTreeMap<GgufCompanionRole, ValidatedGgufCompanion>,
    },
}

/// Inspect a local artifact without loading tensor payloads.
pub fn inspect_artifact<R: ModelConfigurationResolver>(
    path: impl AsRef<Path>,
    resolver: &R,
) -> Result<ArtifactInspection<R::ArtifactPlan>, ArtifactError> {
    let path = path.as_ref();
    if is_gguf(path) {
        inspect_gguf(path, resolver)
    } else if path.is_dir() {
        inspect_safetensors(path, resolver)
    } else if !path.exists() {
        Err(ArtifactError::MissingArtifact(path.to_path_buf()))
    } else {
        Err(ArtifactError::UnsupportedContainer(path.to_path_buf()))
    }
}

/// Validate policy and select one backend-independent materialization route.
pub fn plan_model_preparation<P>(
    inspection: ArtifactInspection<P>,
    policy: PreparationPolicy,
) -> Result<ModelPreparationPlan<P>, ArtifactError> {
    let route = validate_preparation_policy(inspection.configuration.loading_protocol, policy)?;
    Ok(ModelPreparationPlan {
        inspection,
        policy,
        route,
    })
}

/// Validate a preparation policy against resolved portable artifact facts.
pub fn validate_preparation_policy(
    protocol: LoadingProtocol,
    policy: PreparationPolicy,
) -> Result<MaterializationRoute, ArtifactError> {
    if protocol != LoadingProtocol::Model {
        return Err(ArtifactError::UnsupportedLoadingProtocol(protocol));
    }
    let route = match policy.residency {
        ResidencyRequest::FullyResident => MaterializationRoute::Resident,
        ResidencyRequest::LayerwiseHost | ResidencyRequest::DenseDiskStream => {
            MaterializationRoute::Layerwise
        }
        ResidencyRequest::ExpertCache => MaterializationRoute::ExpertCache,
    };
    Ok(route)
}

fn inspect_gguf<R: ModelConfigurationResolver>(
    path: &Path,
    resolver: &R,
) -> Result<ArtifactInspection<R::ArtifactPlan>, ArtifactError> {
    let checkpoint = GgufCheckpoint::open(path)?;
    let architecture_name = checkpoint
        .metadata()
        .get("general.architecture")
        .and_then(MetadataValue::as_str)
        .ok_or(ArtifactError::MissingGgufArchitecture)?;
    let configuration = resolver.resolve_gguf(architecture_name, &checkpoint)?;
    let requirements = resolver.gguf_companion_requirements(architecture_name, &checkpoint)?;
    let companions = resolve_gguf_companions(path, &requirements)?;
    validate_gguf_container(&checkpoint)?;
    let tensors = checkpoint
        .tensors()
        .map(|tensor| {
            let descriptor = tensor.descriptor();
            let shape = descriptor
                .dimensions
                .iter()
                .map(|&dimension| {
                    usize::try_from(dimension).map_err(|_| {
                        ArtifactError::InvalidArtifact(format!(
                            "GGUF tensor {:?} dimension {dimension} exceeds the host address space",
                            descriptor.name
                        ))
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(TensorDescriptor {
                name: descriptor.name.clone(),
                shape,
                dtype: TensorDtype::Encoded(format!("{:?}", descriptor.ggml_type)),
                storage: None,
            })
        })
        .collect::<Result<Vec<_>, ArtifactError>>()?;
    let tensors = TensorCatalog::new(tensors)?;
    let validated_gguf = ValidatedGguf {
        checkpoint,
        companions,
    };
    let architecture_plan = resolver.artifact_plan(
        path,
        ArtifactFormat::Gguf,
        &configuration,
        Some(&validated_gguf),
    )?;
    Ok(ArtifactInspection {
        path: path.to_path_buf(),
        format: ArtifactFormat::Gguf,
        configuration,
        tensors,
        validated_gguf: Some(validated_gguf),
        architecture_plan,
    })
}

/// Resolves architecture-declared GGUF companions without materializing payloads.
pub fn resolve_gguf_companions(
    primary: &Path,
    requirements: &[GgufCompanionRequirement],
) -> Result<BTreeMap<GgufCompanionRole, ValidatedGgufCompanion>, ArtifactError> {
    let mut resolved = BTreeMap::new();
    let mut declared_roles = BTreeSet::new();
    for requirement in requirements {
        if !declared_roles.insert(requirement.role.clone()) {
            return Err(ArtifactError::InvalidArtifact(format!(
                "GGUF companion role {:?} was declared more than once",
                requirement.role
            )));
        }
        let mut directories = Vec::new();
        let mut directory = primary.parent().unwrap_or_else(|| Path::new("."));
        directories.push(directory.to_path_buf());
        for _ in 0..requirement.parent_search_depth {
            let Some(parent) = directory.parent() else {
                break;
            };
            if parent == directory {
                break;
            }
            directories.push(parent.to_path_buf());
            directory = parent;
        }
        let mut candidates = Vec::new();
        for directory in &directories {
            let candidate_start = candidates.len();
            for entry in std::fs::read_dir(directory)? {
                let path = entry?.path();
                let name = path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or_default();
                if path != primary
                    && path.is_file()
                    && name
                        .get(..requirement.filename_prefix.len())
                        .is_some_and(|prefix| {
                            prefix.eq_ignore_ascii_case(&requirement.filename_prefix)
                        })
                    && is_gguf(&path)
                {
                    let checkpoint = GgufCheckpoint::open(&path)?;
                    if checkpoint.physical_tensor_count() == 0 {
                        return Err(ArtifactError::InvalidArtifact(format!(
                            "GGUF companion {} contains no tensors",
                            path.display()
                        )));
                    }
                    let dense = checkpoint.tensors().all(|tensor| {
                        matches!(
                            tensor.descriptor().ggml_type,
                            eredu_gguf::GgmlType::F32
                                | eredu_gguf::GgmlType::F16
                                | eredu_gguf::GgmlType::Bf16
                        )
                    });
                    candidates.push((path, checkpoint, dense));
                }
            }
            if candidates.len() != candidate_start {
                break;
            }
        }
        candidates.sort_by(|left, right| left.0.cmp(&right.0));
        candidates.dedup_by(|left, right| left.0 == right.0);
        let dense = candidates
            .iter()
            .filter(|candidate| candidate.2)
            .collect::<Vec<_>>();
        let selected = match requirement.encoding {
            GgufCompanionEncoding::DenseRequired => match dense.as_slice() {
                [candidate] => Some(*candidate),
                [] if candidates.is_empty() => None,
                [] => {
                    return Err(ArtifactError::InvalidArtifact(format!(
                        "GGUF companion {:?} requires dense F32, F16, or BF16 tensors, but all {} matching candidates are quantized",
                        requirement.role,
                        candidates.len()
                    )))
                }
                _ => return Err(ambiguous_companion(requirement, &directories, dense.len())),
            },
            GgufCompanionEncoding::DensePreferred => match dense.as_slice() {
                [candidate] => Some(*candidate),
                [] => match candidates.as_slice() {
                    [candidate] => Some(candidate),
                    [] => None,
                    _ => {
                        return Err(ambiguous_companion(
                            requirement,
                            &directories,
                            candidates.len(),
                        ))
                    }
                },
                _ => return Err(ambiguous_companion(requirement, &directories, dense.len())),
            },
        };
        match selected {
            Some((path, checkpoint, _)) => {
                resolved.insert(
                    requirement.role.clone(),
                    ValidatedGgufCompanion {
                        path: path.clone(),
                        checkpoint: checkpoint.clone(),
                    },
                );
            }
            None if requirement.required => {
                return Err(ArtifactError::InvalidArtifact(format!(
                    "required GGUF companion {:?} matching {:?} was not found in {}",
                    requirement.role,
                    requirement.filename_prefix,
                    display_directories(&directories)
                )))
            }
            None => {}
        }
    }
    Ok(resolved)
}

fn ambiguous_companion(
    requirement: &GgufCompanionRequirement,
    directories: &[PathBuf],
    candidates: usize,
) -> ArtifactError {
    ArtifactError::InvalidArtifact(format!(
        "GGUF companion {:?} is ambiguous: found {candidates} preferred candidates in {}",
        requirement.role,
        display_directories(directories)
    ))
}

fn display_directories(directories: &[PathBuf]) -> String {
    directories
        .iter()
        .map(|directory| directory.display().to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

fn validate_gguf_container(checkpoint: &GgufCheckpoint) -> Result<(), ArtifactError> {
    if checkpoint.physical_tensor_count() == 0 {
        return Err(ArtifactError::InvalidArtifact(
            "GGUF model checkpoint contains no tensors".into(),
        ));
    }
    Ok(())
}

fn inspect_safetensors<R: ModelConfigurationResolver>(
    path: &Path,
    resolver: &R,
) -> Result<ArtifactInspection<R::ArtifactPlan>, ArtifactError> {
    let config_path = path.join("config.json");
    let json: Value = serde_json::from_reader(File::open(&config_path)?)?;
    let configuration = resolver.resolve_safetensors(&json)?;
    let shards = safetensors_shards(path)?;
    let mut descriptors = Vec::new();
    let mut names = BTreeSet::new();
    for shard in shards {
        for descriptor in inspect_safetensors_header(&shard)? {
            if !names.insert(descriptor.name.clone()) {
                return Err(ArtifactError::DuplicateTensor(descriptor.name));
            }
            descriptors.push(descriptor);
        }
    }
    let tensors = TensorCatalog::new(descriptors)?;
    if tensors.is_empty() {
        return Err(ArtifactError::InvalidArtifact(
            "SafeTensors checkpoint contains no tensors".into(),
        ));
    }
    let architecture_plan =
        resolver.artifact_plan(path, ArtifactFormat::SafeTensors, &configuration, None)?;
    Ok(ArtifactInspection {
        path: path.to_path_buf(),
        format: ArtifactFormat::SafeTensors,
        configuration,
        tensors,
        validated_gguf: None,
        architecture_plan,
    })
}

#[derive(Deserialize)]
struct SafetensorsIndex {
    weight_map: BTreeMap<String, String>,
}

fn safetensors_shards(path: &Path) -> Result<Vec<PathBuf>, ArtifactError> {
    let index_path = path.join("model.safetensors.index.json");
    if index_path.exists() {
        let index: SafetensorsIndex = serde_json::from_reader(File::open(&index_path)?)?;
        if index.weight_map.is_empty() {
            return Err(ArtifactError::InvalidArtifact(
                "SafeTensors index weight_map is empty".into(),
            ));
        }
        let mut shards = BTreeSet::new();
        for relative in index.weight_map.values() {
            let relative = Path::new(relative);
            if relative.is_absolute()
                || relative
                    .components()
                    .any(|component| !matches!(component, Component::Normal(_) | Component::CurDir))
            {
                return Err(ArtifactError::UnsafeShardPath(relative.to_path_buf()));
            }
            shards.insert(path.join(relative));
        }
        return Ok(shards.into_iter().collect());
    }
    Ok(vec![path.join("model.safetensors")])
}

#[derive(Deserialize)]
struct RawSafetensorInfo {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [u64; 2],
}

fn inspect_safetensors_header(path: &Path) -> Result<Vec<TensorDescriptor>, ArtifactError> {
    const MAX_HEADER_BYTES: u64 = 100_000_000;
    let mut file = File::open(path)?;
    let file_len = file.metadata()?.len();
    let mut length = [0_u8; 8];
    file.read_exact(&mut length)?;
    let header_len = u64::from_le_bytes(length);
    if header_len > MAX_HEADER_BYTES {
        return Err(ArtifactError::InvalidArtifact(format!(
            "SafeTensors header in {} exceeds {MAX_HEADER_BYTES} bytes",
            path.display()
        )));
    }
    let mut header = vec![
        0_u8;
        usize::try_from(header_len).map_err(|_| {
            ArtifactError::InvalidArtifact("SafeTensors header length overflows usize".into())
        })?
    ];
    file.read_exact(&mut header)?;
    let raw: BTreeMap<String, Value> = serde_json::from_slice(&header)?;
    let payload_start = 8_u64
        .checked_add(header_len)
        .ok_or_else(|| ArtifactError::InvalidArtifact("SafeTensors offset overflow".into()))?;
    let mut entries = raw
        .into_iter()
        .filter(|(name, _)| name != "__metadata__")
        .map(|(name, value)| {
            serde_json::from_value::<RawSafetensorInfo>(value).map(|info| (name, info))
        })
        .collect::<Result<Vec<_>, _>>()?;
    entries.sort_by_key(|(_, info)| info.data_offsets[0]);
    let mut output = Vec::with_capacity(entries.len());
    let mut expected_offset = 0_u64;
    for (name, info) in entries {
        // SafeTensors rank-zero tensors are scalar parameters with one stored
        // element. Gemma media clipping bounds use this representation.
        if info.shape.contains(&0) {
            return Err(ArtifactError::InvalidArtifact(format!(
                "SafeTensors tensor {name:?} has an invalid shape"
            )));
        }
        let [start, end] = info.data_offsets;
        if start != expected_offset || end < start {
            return Err(ArtifactError::InvalidArtifact(format!(
                "SafeTensors tensor {name:?} has non-contiguous data offsets"
            )));
        }
        expected_offset = end;
        let absolute = payload_start
            .checked_add(start)
            .ok_or_else(|| ArtifactError::InvalidArtifact("SafeTensors offset overflow".into()))?;
        output.push(TensorDescriptor {
            name,
            shape: info.shape,
            dtype: safetensors_dtype(&info.dtype),
            storage: Some(TensorStorage {
                member: path.display().to_string(),
                offset: absolute,
                length: end - start,
            }),
        });
    }
    if payload_start
        .checked_add(expected_offset)
        .ok_or_else(|| ArtifactError::InvalidArtifact("SafeTensors length overflow".into()))?
        != file_len
    {
        return Err(ArtifactError::InvalidArtifact(format!(
            "SafeTensors payload length does not match header in {}",
            path.display()
        )));
    }
    Ok(output)
}

fn safetensors_dtype(dtype: &str) -> TensorDtype {
    match dtype {
        "F32" => TensorDtype::F32,
        "F16" => TensorDtype::F16,
        "BF16" => TensorDtype::Bf16,
        "I8" => TensorDtype::I8,
        "U8" => TensorDtype::U8,
        "I32" => TensorDtype::I32,
        other => TensorDtype::Encoded(other.into()),
    }
}

fn is_gguf(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("gguf"))
}

/// Portable artifact inspection/planning failure.
#[derive(Debug, thiserror::Error)]
pub enum ArtifactError {
    /// Artifact path does not exist.
    #[error("model artifact does not exist: {0}")]
    MissingArtifact(PathBuf),
    /// Path is not a supported artifact container.
    #[error("model artifact must be a SafeTensors directory or .gguf file: {0}")]
    UnsupportedContainer(PathBuf),
    /// Model type is not recognized.
    #[error("unsupported model type: {0}")]
    UnsupportedModelType(String),
    /// GGUF architecture is not recognized.
    #[error("unsupported GGUF architecture: {0}")]
    UnsupportedGgufArchitecture(String),
    /// GGUF architecture metadata is absent or has the wrong type.
    #[error("GGUF metadata is missing string key \"general.architecture\"")]
    MissingGgufArchitecture,
    /// Header/catalog content is contradictory.
    #[error("invalid model artifact: {0}")]
    InvalidArtifact(String),
    /// A tensor name occurred more than once.
    #[error("duplicate checkpoint tensor {0:?}")]
    DuplicateTensor(String),
    /// Indexed shard path escapes the artifact root.
    #[error("unsafe SafeTensors shard path {0}")]
    UnsafeShardPath(PathBuf),
    /// Requested quantization transformation is unavailable for the artifact.
    #[error("unsupported model quantization policy: {0}")]
    UnsupportedQuantizationPolicy(String),
    /// Requested residency mode is unavailable for the artifact.
    #[error("unsupported model residency policy: {0}")]
    UnsupportedResidencyPolicy(String),
    /// The general model planner cannot satisfy the resolved loader contract.
    #[error("model artifact requires the {0:?} loading protocol")]
    UnsupportedLoadingProtocol(LoadingProtocol),
    /// Ordinary filesystem error.
    #[error(transparent)]
    Io(#[from] std::io::Error),
    /// JSON configuration/header error.
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    /// GGUF parsing/catalog error.
    #[error(transparent)]
    Gguf(#[from] eredu_gguf::Error),
    /// Neutral tensor catalog error.
    #[error(transparent)]
    Catalog(#[from] crate::checkpoint::CatalogError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use eredu_gguf::{GgmlType, MetadataArray, TensorInput, Writer};
    use std::io::Write;

    struct FixtureResolver;

    #[derive(Debug, Clone, Default, Eq, PartialEq)]
    struct FixtureArtifactPlan {
        format: Option<ArtifactFormat>,
    }

    impl ModelConfigurationResolver for FixtureResolver {
        type ArtifactPlan = FixtureArtifactPlan;

        fn resolve_safetensors(&self, json: &Value) -> Result<ModelConfiguration, ArtifactError> {
            let model_type = json
                .get("model_type")
                .and_then(Value::as_str)
                .ok_or_else(|| ArtifactError::InvalidArtifact("missing model_type".into()))?;
            let family = match model_type {
                "llama" => "llama",
                "gemma4" => "gemma4",
                "future" => "future_family",
                other => return Err(ArtifactError::UnsupportedModelType(other.into())),
            };
            Ok(ModelConfiguration {
                declared_model_type: model_type.into(),
                effective_model_type: model_type.into(),
                family: family.into(),
                loading_protocol: LoadingProtocol::Model,
                json: Some(json.clone()),
            })
        }

        fn resolve_gguf(
            &self,
            architecture: &str,
            _checkpoint: &GgufCheckpoint,
        ) -> Result<ModelConfiguration, ArtifactError> {
            let family = match architecture {
                "llama" => "llama",
                "future" => "future_family",
                other => return Err(ArtifactError::UnsupportedGgufArchitecture(other.into())),
            };
            Ok(ModelConfiguration {
                declared_model_type: architecture.into(),
                effective_model_type: architecture.into(),
                family: family.into(),
                loading_protocol: LoadingProtocol::Model,
                json: None,
            })
        }

        fn gguf_companion_requirements(
            &self,
            architecture: &str,
            _checkpoint: &GgufCheckpoint,
        ) -> Result<Vec<GgufCompanionRequirement>, ArtifactError> {
            if architecture == "future" {
                return Ok(vec![GgufCompanionRequirement::new(
                    GgufCompanionRole::MediaProjector,
                    false,
                    "mmproj",
                    0,
                    GgufCompanionEncoding::DensePreferred,
                )?]);
            }
            Ok(Vec::new())
        }

        fn artifact_plan(
            &self,
            _path: &Path,
            format: ArtifactFormat,
            _configuration: &ModelConfiguration,
            _validated_gguf: Option<&ValidatedGguf>,
        ) -> Result<Self::ArtifactPlan, ArtifactError> {
            Ok(FixtureArtifactPlan {
                format: Some(format),
            })
        }
    }

    fn write_safetensors_fixture(root: &Path, model_type: &str) {
        std::fs::write(
            root.join("config.json"),
            format!(r#"{{"model_type":"{model_type}"}}"#),
        )
        .unwrap();
        let header =
            br#"{"token_embd.weight":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}}"#;
        let mut file = File::create(root.join("model.safetensors")).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(header).unwrap();
        file.write_all(&[0_u8; 16]).unwrap();
    }

    fn write_gguf_fixture(path: &Path, ggml_type: GgmlType) {
        let metadata = BTreeMap::from([(
            "general.architecture".into(),
            MetadataValue::String("clip".into()),
        )]);
        let (dimensions, data) = match ggml_type {
            GgmlType::F32 => (vec![1], 1.0_f32.to_le_bytes().to_vec()),
            GgmlType::Q8_0 => (vec![32], vec![0_u8; 34]),
            other => panic!("unsupported fixture encoding {other:?}"),
        };
        Writer::default()
            .write(
                File::create(path).unwrap(),
                &metadata,
                &[TensorInput {
                    name: "projector.weight",
                    dimensions: &dimensions,
                    ggml_type,
                    data: &data,
                }],
            )
            .unwrap();
    }

    #[test]
    fn companion_planning_selects_by_catalog_encoding_not_filename() {
        let root = tempfile::tempdir().unwrap();
        let primary = root.path().join("model.gguf");
        write_gguf_fixture(&primary, GgmlType::F32);
        let quantized_name = root.path().join("mmproj-f16.gguf");
        let dense_name = root.path().join("mmproj-q4_k.gguf");
        write_gguf_fixture(&quantized_name, GgmlType::Q8_0);
        write_gguf_fixture(&dense_name, GgmlType::F32);
        let requirement = GgufCompanionRequirement::new(
            GgufCompanionRole::MediaProjector,
            true,
            "mmproj",
            1,
            GgufCompanionEncoding::DensePreferred,
        )
        .unwrap();

        let companions = resolve_gguf_companions(&primary, &[requirement]).unwrap();

        assert_eq!(
            companions
                .get(&GgufCompanionRole::MediaProjector)
                .unwrap()
                .path(),
            dense_name
        );
    }

    #[test]
    fn dense_only_and_required_companion_policies_fail_closed() {
        let root = tempfile::tempdir().unwrap();
        let primary = root.path().join("model.gguf");
        write_gguf_fixture(&primary, GgmlType::F32);
        write_gguf_fixture(&root.path().join("mmproj.gguf"), GgmlType::Q8_0);
        let optional = GgufCompanionRequirement::new(
            GgufCompanionRole::MediaProjector,
            false,
            "mmproj",
            0,
            GgufCompanionEncoding::DenseRequired,
        )
        .unwrap();
        assert!(resolve_gguf_companions(&primary, &[optional]).is_err());
        let required = GgufCompanionRequirement::new(
            GgufCompanionRole::MediaProjector,
            true,
            "mmproj",
            0,
            GgufCompanionEncoding::DenseRequired,
        )
        .unwrap();
        assert!(resolve_gguf_companions(&primary, &[required]).is_err());
    }

    #[test]
    fn loading_protocol_is_family_agnostic() {
        assert!(matches!(
            validate_preparation_policy(LoadingProtocol::Realtime, PreparationPolicy::default()),
            Err(ArtifactError::UnsupportedLoadingProtocol(
                LoadingProtocol::Realtime
            ))
        ));
    }

    #[test]
    fn gguf_u32_metadata_is_lossless_and_fail_closed() {
        let values = MetadataValue::Array(MetadataArray::Uint64(vec![0, u32::MAX.into()]));
        assert_eq!(
            gguf_u32_metadata_values("tokenizer.ids", Some(&values)).unwrap(),
            vec![0, u32::MAX]
        );
        assert!(gguf_u32_metadata_values(
            "tokenizer.ids",
            Some(&MetadataValue::Uint64(u64::from(u32::MAX) + 1))
        )
        .is_err());
        assert!(
            gguf_u32_metadata_values("tokenizer.ids", Some(&MetadataValue::Int32(-1))).is_err()
        );
        assert!(gguf_u32_metadata_values(
            "tokenizer.ids",
            Some(&MetadataValue::String("1".into()))
        )
        .is_err());
        assert!(gguf_u32_metadata_values("tokenizer.ids", None)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn safetensors_inspection_and_planning_are_backend_neutral() {
        let root = tempfile::tempdir().unwrap();
        write_safetensors_fixture(root.path(), "llama");
        let inspection = inspect_artifact(root.path(), &FixtureResolver).unwrap();
        assert_eq!(inspection.configuration().family, "llama");
        assert_eq!(inspection.tensors().len(), 1);
        let plan = plan_model_preparation(inspection, PreparationPolicy::default()).unwrap();
        assert_eq!(plan.route(), MaterializationRoute::Resident);
        let (artifact, architecture_plan, _, _) = plan.into_parts();
        assert!(matches!(artifact, ModelArtifact::SafeTensors { .. }));
        assert_eq!(architecture_plan.format, Some(ArtifactFormat::SafeTensors));
    }

    #[test]
    fn core_accepts_families_defined_only_by_the_resolver() {
        let root = tempfile::tempdir().unwrap();
        write_safetensors_fixture(root.path(), "future");
        let inspection = inspect_artifact(root.path(), &FixtureResolver).unwrap();
        assert_eq!(inspection.configuration().family, "future_family");
        assert_eq!(
            inspection.configuration().loading_protocol,
            LoadingProtocol::Model
        );
        assert!(plan_model_preparation(inspection, PreparationPolicy::default()).is_ok());
    }

    #[test]
    fn safetensors_inspection_accepts_rank_zero_scalar_parameters() {
        let root = tempfile::tempdir().unwrap();
        std::fs::write(
            root.path().join("config.json"),
            r#"{"model_type":"gemma4"}"#,
        )
        .unwrap();
        let header = br#"{"clip.output_max":{"dtype":"F32","shape":[],"data_offsets":[0,4]}}"#;
        let mut file = File::create(root.path().join("model.safetensors")).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(header).unwrap();
        file.write_all(&0.0_f32.to_le_bytes()).unwrap();

        let inspection = inspect_artifact(root.path(), &FixtureResolver).unwrap();
        assert_eq!(
            inspection.tensors().get("clip.output_max").unwrap().shape,
            Vec::<usize>::new()
        );
    }

    #[test]
    fn parallel_policy_binds_the_exact_neutral_topology() {
        let root = tempfile::tempdir().unwrap();
        write_safetensors_fixture(root.path(), "llama");
        let topology = crate::topology::ParallelTopology::new(2, 3, 4, 1).unwrap();
        let policy = PreparationPolicy {
            topology: Some(topology),
            ..PreparationPolicy::default()
        };

        let plan = plan_model_preparation(
            inspect_artifact(root.path(), &FixtureResolver).unwrap(),
            policy,
        )
        .unwrap();

        assert_eq!(plan.policy(), policy);
        assert_eq!(plan.policy().topology, Some(topology));
        assert_eq!(plan.route(), MaterializationRoute::Resident);
    }

    #[test]
    fn policy_leaves_expert_cache_capability_to_architecture_and_backend() {
        let root = tempfile::tempdir().unwrap();
        write_safetensors_fixture(root.path(), "llama");
        let plan = plan_model_preparation(
            inspect_artifact(root.path(), &FixtureResolver).unwrap(),
            PreparationPolicy {
                residency: ResidencyRequest::ExpertCache,
                ..PreparationPolicy::default()
            },
        )
        .unwrap();
        assert_eq!(plan.route(), MaterializationRoute::ExpertCache);
    }

    #[test]
    fn policy_leaves_nonresident_quantization_capability_to_architecture_and_backend() {
        let root = tempfile::tempdir().unwrap();
        write_safetensors_fixture(root.path(), "llama");
        let policy = PreparationPolicy {
            quantization: Some(QuantizationRequest::MxFp4),
            residency: ResidencyRequest::LayerwiseHost,
            ..PreparationPolicy::default()
        };
        let plan = plan_model_preparation(
            inspect_artifact(root.path(), &FixtureResolver).unwrap(),
            policy,
        )
        .unwrap();
        assert_eq!(plan.policy(), policy);
        assert_eq!(plan.route(), MaterializationRoute::Layerwise);
    }

    #[test]
    fn gguf_plan_owns_the_portable_checkpoint_for_later_materialization() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("model.gguf");
        let data = 1.0_f32.to_le_bytes();
        let metadata = BTreeMap::from([
            (
                "general.architecture".into(),
                MetadataValue::String("llama".into()),
            ),
            ("llama.block_count".into(), MetadataValue::Uint32(1)),
            ("llama.embedding_length".into(), MetadataValue::Uint32(1)),
        ]);
        Writer::default()
            .write(
                File::create(&path).unwrap(),
                &metadata,
                &[TensorInput {
                    name: "token_embd.weight",
                    dimensions: &[1],
                    ggml_type: GgmlType::F32,
                    data: &data,
                }],
            )
            .unwrap();

        let inspection = inspect_artifact(&path, &FixtureResolver).unwrap();
        let validated = inspection.validated_gguf().unwrap();
        assert_eq!(validated.checkpoint().physical_tensor_count(), 1);
        assert_eq!(inspection.configuration().declared_model_type, "llama");

        let plan = plan_model_preparation(inspection, PreparationPolicy::default()).unwrap();
        let (artifact, architecture_plan, _, _) = plan.into_parts();
        let ModelArtifact::Gguf {
            configuration,
            checkpoint,
            ..
        } = artifact
        else {
            panic!("expected GGUF artifact");
        };
        assert_eq!(architecture_plan.format, Some(ArtifactFormat::Gguf));
        assert_eq!(configuration.family, "llama");
        assert_eq!(checkpoint.physical_tensor_count(), 1);
    }

    #[test]
    fn core_accepts_architecture_owned_gguf_schema() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("model.gguf");
        let data = 1.0_f32.to_le_bytes();
        let metadata = BTreeMap::from([
            (
                "general.architecture".into(),
                MetadataValue::String("future".into()),
            ),
            ("future.state_width".into(), MetadataValue::Uint32(1)),
        ]);
        Writer::default()
            .write(
                File::create(&path).unwrap(),
                &metadata,
                &[TensorInput {
                    name: "state.in_proj",
                    dimensions: &[1],
                    ggml_type: GgmlType::F32,
                    data: &data,
                }],
            )
            .unwrap();

        let inspection = inspect_artifact(&path, &FixtureResolver).unwrap();

        assert_eq!(inspection.configuration().family, "future_family");
        assert!(inspection.tensors().get("state.in_proj").is_some());
    }

    #[test]
    fn preparation_plan_carries_the_exact_inspected_companion() {
        let root = tempfile::tempdir().unwrap();
        let primary = root.path().join("model.gguf");
        let scalar = 1.0_f32.to_le_bytes();
        Writer::default()
            .write(
                File::create(&primary).unwrap(),
                &BTreeMap::from([(
                    "general.architecture".into(),
                    MetadataValue::String("future".into()),
                )]),
                &[TensorInput {
                    name: "state.in_proj",
                    dimensions: &[1],
                    ggml_type: GgmlType::F32,
                    data: &scalar,
                }],
            )
            .unwrap();
        let projector = root.path().join("mmproj.gguf");
        write_gguf_fixture(&projector, GgmlType::F32);

        let inspection = inspect_artifact(&primary, &FixtureResolver).unwrap();
        assert_eq!(
            inspection
                .validated_gguf()
                .unwrap()
                .companion(&GgufCompanionRole::MediaProjector)
                .unwrap()
                .path(),
            projector
        );
        let ModelArtifact::Gguf { companions, .. } =
            plan_model_preparation(inspection, PreparationPolicy::default())
                .unwrap()
                .into_parts()
                .0
        else {
            panic!("expected GGUF preparation artifact");
        };
        assert_eq!(
            companions
                .get(&GgufCompanionRole::MediaProjector)
                .unwrap()
                .path(),
            projector
        );
    }
}
