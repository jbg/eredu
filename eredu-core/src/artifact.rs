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
    /// Resolves one Hugging Face `config.json` value to its canonical family.
    fn resolve_safetensors(&self, json: &Value) -> Result<ModelConfiguration, ArtifactError>;

    /// Resolves and structurally admits one GGUF architecture and checkpoint.
    fn resolve_gguf(
        &self,
        architecture: &str,
        checkpoint: &GgufCheckpoint,
    ) -> Result<ModelConfiguration, ArtifactError>;
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
pub struct ArtifactInspection {
    path: PathBuf,
    format: ArtifactFormat,
    configuration: ModelConfiguration,
    tensors: TensorCatalog,
    validated_gguf: Option<ValidatedGguf>,
}

/// Portable GGUF facts admitted by core inspection.
///
/// Backends may enrich this result with runtime-specific compatibility checks,
/// but do not need to repeat the portable metadata and catalog validation.
#[derive(Debug, Clone)]
pub struct ValidatedGguf {
    checkpoint: GgufCheckpoint,
}

impl ValidatedGguf {
    /// Header-only checkpoint admitted by portable inspection.
    pub fn checkpoint(&self) -> &GgufCheckpoint {
        &self.checkpoint
    }
}

impl ArtifactInspection {
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
    /// Whether a non-replicated distributed topology was requested.
    pub distributed: bool,
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
pub struct ModelPreparationPlan {
    inspection: ArtifactInspection,
    policy: PreparationPolicy,
    route: MaterializationRoute,
}

impl ModelPreparationPlan {
    /// Header-only inspection owned by the plan.
    pub fn inspection(&self) -> &ArtifactInspection {
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
    /// Consume the plan into its portable artifact and policy.
    pub fn into_parts(self) -> (ModelArtifact, PreparationPolicy, MaterializationRoute) {
        let artifact = match self.inspection.validated_gguf {
            Some(validated) => ModelArtifact::Gguf {
                path: self.inspection.path,
                configuration: self.inspection.configuration,
                tensors: self.inspection.tensors,
                checkpoint: validated.checkpoint,
            },
            None => ModelArtifact::SafeTensors {
                path: self.inspection.path,
                configuration: self.inspection.configuration,
                tensors: self.inspection.tensors,
            },
        };
        (artifact, self.policy, self.route)
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
    },
}

/// Inspect a local artifact without loading tensor payloads.
pub fn inspect_artifact(
    path: impl AsRef<Path>,
    resolver: &impl ModelConfigurationResolver,
) -> Result<ArtifactInspection, ArtifactError> {
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
pub fn plan_model_preparation(
    inspection: ArtifactInspection,
    policy: PreparationPolicy,
) -> Result<ModelPreparationPlan, ArtifactError> {
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

fn inspect_gguf(
    path: &Path,
    resolver: &impl ModelConfigurationResolver,
) -> Result<ArtifactInspection, ArtifactError> {
    let checkpoint = GgufCheckpoint::open(path)?;
    let architecture_name = checkpoint
        .metadata()
        .get("general.architecture")
        .and_then(MetadataValue::as_str)
        .ok_or(ArtifactError::MissingGgufArchitecture)?;
    let configuration = resolver.resolve_gguf(architecture_name, &checkpoint)?;
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
    Ok(ArtifactInspection {
        path: path.to_path_buf(),
        format: ArtifactFormat::Gguf,
        configuration,
        tensors,
        validated_gguf: Some(ValidatedGguf { checkpoint }),
    })
}

fn validate_gguf_container(checkpoint: &GgufCheckpoint) -> Result<(), ArtifactError> {
    if checkpoint.physical_tensor_count() == 0 {
        return Err(ArtifactError::InvalidArtifact(
            "GGUF model checkpoint contains no tensors".into(),
        ));
    }
    Ok(())
}

fn inspect_safetensors(
    path: &Path,
    resolver: &impl ModelConfigurationResolver,
) -> Result<ArtifactInspection, ArtifactError> {
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
    Ok(ArtifactInspection {
        path: path.to_path_buf(),
        format: ArtifactFormat::SafeTensors,
        configuration,
        tensors,
        validated_gguf: None,
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

    impl ModelConfigurationResolver for FixtureResolver {
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
        assert!(matches!(
            plan.into_parts().0,
            ModelArtifact::SafeTensors { .. }
        ));
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
    fn distributed_policy_uses_the_same_neutral_preparation_plan() {
        let root = tempfile::tempdir().unwrap();
        write_safetensors_fixture(root.path(), "llama");
        let policy = PreparationPolicy {
            distributed: true,
            ..PreparationPolicy::default()
        };

        let plan = plan_model_preparation(
            inspect_artifact(root.path(), &FixtureResolver).unwrap(),
            policy,
        )
        .unwrap();

        assert_eq!(plan.policy(), policy);
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
        let ModelArtifact::Gguf {
            configuration,
            checkpoint,
            ..
        } = plan.into_parts().0
        else {
            panic!("expected GGUF artifact");
        };
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
}
