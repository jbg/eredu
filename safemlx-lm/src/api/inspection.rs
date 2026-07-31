//! Side-effect-free model artifact compatibility inspection.

use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
};

use safemlx::ops::{GgufCheckpoint, GgufMetadataValue};
use serde::Serialize;
use serde_json::{json, Map, Value};

use super::*;
use crate::runtime::checkpoint::store::{SafetensorsWeightStore, WeightStore};

/// Options applied while inspecting a model artifact.
#[derive(Debug, Clone, Default)]
pub struct ModelInspectionOptions {
    /// The exact loading policy that admission should validate.
    pub load: ModelLoadOptions,
    /// Optional concrete chat request to render and behaviorally probe.
    ///
    /// When omitted, inspection uses bounded synthetic probes to recognize a
    /// semantic protocol and native-tool envelope. Real tool schemas and
    /// request-specific template kwargs still require per-request validation.
    pub chat_request: Option<ChatTemplateRequest>,
}

/// Physical artifact selected for inspection.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactKind {
    /// Hugging Face-style directory containing SafeTensors weights.
    SafeTensorsDirectory,
    /// Single-file or canonically sharded GGUF checkpoint.
    GgufCheckpoint,
}

/// A readiness result that does not collapse distinct failure modes to a bool.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum InspectionReadiness {
    /// The inspected artifacts establish this capability.
    Ready,
    /// The artifact omits a required component.
    Missing,
    /// SafeMLX does not implement the artifact or requested combination.
    Unsupported,
    /// The relevant artifact data is malformed.
    Invalid,
    /// A concrete request must be supplied before this can be decided.
    RequestDependent,
    /// The check necessarily occurs later in loading or execution.
    Unverified,
    /// The capability does not apply to this artifact.
    NotApplicable,
}

/// Severity attached to an inspection issue.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum InspectionSeverity {
    /// Prevents the stated readiness or requested load route.
    Error,
    /// Does not prevent the selected load, but limits a capability or proof.
    Warning,
    /// Actionable context that is neither a rejection nor a warning.
    Info,
}

/// Stable machine-readable issue category.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Ord, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum InspectionIssueCode {
    /// Artifact path or container structure is invalid.
    InvalidContainer,
    /// `config.json` or architecture metadata is invalid.
    InvalidConfiguration,
    /// Architecture dispatch has no SafeMLX implementation.
    UnsupportedArchitecture,
    /// A referenced checkpoint shard is absent or contradictory.
    MissingCheckpointShard,
    /// A tensor storage encoding cannot be consumed.
    UnsupportedTensorEncoding,
    /// No usable embedded or sidecar tokenizer is available.
    MissingTokenizer,
    /// No checkpoint or sidecar chat template is available.
    MissingChatTemplate,
    /// A required multimodal projector is absent or ambiguous.
    MissingMediaProjector,
    /// A media processor or its build feature is unavailable.
    MissingProcessor,
    /// The requested on-load quantization is incompatible.
    UnsupportedQuantizationRequest,
    /// The requested weight residency route is incompatible.
    UnsupportedResidencyPolicy,
    /// The requested parallel topology cannot use this loader.
    UnsupportedParallelTopology,
    /// No fail-closed semantic streaming protocol was recognized.
    UnsupportedSemanticProtocol,
    /// No fail-closed native-tool protocol was recognized.
    UnsupportedToolProtocol,
    /// EOS metadata is absent; callers may need request-time stop criteria.
    MissingEosMetadata,
    /// Exact architecture binding still requires loader-time module validation.
    ValidationUnavailableUntilLoad,
    /// Real messages, schemas, kwargs, or runtime state still need validation.
    RequestSpecificValidation,
    /// An ordinary local I/O operation failed.
    Io,
}

/// One structured diagnostic produced by inspection.
#[derive(Debug, Clone, Eq, PartialEq, Serialize)]
pub struct InspectionIssue {
    /// Stable category for routing and UI behavior.
    pub code: InspectionIssueCode,
    /// Diagnostic severity.
    pub severity: InspectionSeverity,
    /// Human-readable actionable detail.
    pub detail: String,
    /// Relevant artifact or sidecar path.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path: Option<PathBuf>,
    /// Relevant GGUF or JSON metadata key.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata_key: Option<String>,
    /// Relevant logical or physical tensor name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tensor_name: Option<String>,
    /// Relevant numeric GGML tensor type code.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tensor_type_code: Option<u32>,
}

/// One tensor storage encoding observed in artifact headers.
#[derive(Debug, Clone, Eq, PartialEq, Ord, PartialOrd, Serialize)]
pub struct ArtifactTensorEncoding {
    /// Stable textual representation (for example `BF16` or `Q4K`).
    pub name: String,
    /// GGML type code for GGUF encodings; absent for SafeTensors dtypes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ggml_type_code: Option<u32>,
}

/// Input modality advertised by the resolved loader architecture.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Ord, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactModality {
    /// Text tokens.
    Text,
    /// Still images.
    Image,
    /// Video frame sequences.
    Video,
    /// Audio waveforms or features.
    Audio,
}

/// A sidecar or companion requirement discovered during inspection.
#[derive(Debug, Clone, Eq, PartialEq, Serialize)]
pub struct InspectionRequirement {
    /// Machine-readable issue category corresponding to this requirement.
    pub code: InspectionIssueCode,
    /// Current readiness of the requirement.
    pub readiness: InspectionReadiness,
    /// Human-readable explanation.
    pub detail: String,
    /// Expected or selected path, when applicable.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path: Option<PathBuf>,
}

/// Structured pre-load compatibility report for a local model artifact.
#[derive(Debug, Clone, Serialize)]
pub struct ModelInspectionReport {
    /// Submitted local artifact path.
    pub path: PathBuf,
    /// Detected artifact container.
    pub artifact_kind: ArtifactKind,
    /// Resolved high-level SafeMLX model family, when architecture dispatch succeeded.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_kind: Option<ModelKind>,
    /// Submitted model type or GGUF `general.architecture` value.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub architecture: Option<String>,
    /// GGUF versions observed across validated shards.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gguf_versions: Option<Vec<u32>>,
    /// Number of checkpoint payload shards, if their catalog was established.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub checkpoint_shards: Option<usize>,
    /// Number of cataloged logical tensors, if established.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tensor_count: Option<usize>,
    /// Every distinct storage encoding observed in checkpoint headers.
    pub tensor_encodings: Vec<ArtifactTensorEncoding>,
    /// Expected input modalities derived from config/architecture metadata.
    pub expected_modalities: Vec<ArtifactModality>,
    /// Container/config/header validity.
    pub container: InspectionReadiness,
    /// SafeMLX model-loader readiness independent of tokenizer/chat sidecars.
    pub model_loadability: InspectionReadiness,
    /// Compatibility with [`ModelInspectionOptions::load`].
    pub requested_load: InspectionReadiness,
    /// Combined model and tokenizer readiness for raw text generation.
    pub text_generation: InspectionReadiness,
    /// Tokenizer reconstruction/readiness.
    pub tokenizer: InspectionReadiness,
    /// Chat-template availability and parseability.
    pub chat_template: InspectionReadiness,
    /// Behavioral structured semantic-streaming readiness.
    pub semantic_streaming: InspectionReadiness,
    /// Behavioral native-tool readiness.
    pub native_tools: InspectionReadiness,
    /// Processor/projector readiness for advertised non-text modalities.
    pub multimodal: InspectionReadiness,
    /// Discovered sidecar and request requirements.
    pub requirements: Vec<InspectionRequirement>,
    /// Structured rejection reasons, warnings, and limitations.
    pub issues: Vec<InspectionIssue>,
}

impl ModelInspectionReport {
    /// Returns whether the artifact and requested load policy passed preflight.
    pub fn is_loadable(&self) -> bool {
        self.container == InspectionReadiness::Ready
            && self.model_loadability == InspectionReadiness::Ready
            && self.requested_load == InspectionReadiness::Ready
    }

    fn new(path: &Path, artifact_kind: ArtifactKind) -> Self {
        Self {
            path: path.to_path_buf(),
            artifact_kind,
            model_kind: None,
            architecture: None,
            gguf_versions: None,
            checkpoint_shards: None,
            tensor_count: None,
            tensor_encodings: Vec::new(),
            expected_modalities: Vec::new(),
            container: InspectionReadiness::Unverified,
            model_loadability: InspectionReadiness::Unverified,
            requested_load: InspectionReadiness::Unverified,
            text_generation: InspectionReadiness::Unverified,
            tokenizer: InspectionReadiness::Unverified,
            chat_template: InspectionReadiness::Unverified,
            semantic_streaming: InspectionReadiness::Unverified,
            native_tools: InspectionReadiness::Unverified,
            multimodal: InspectionReadiness::Unverified,
            requirements: Vec::new(),
            issues: Vec::new(),
        }
    }

    fn issue(
        &mut self,
        code: InspectionIssueCode,
        severity: InspectionSeverity,
        detail: impl Into<String>,
        path: Option<PathBuf>,
    ) {
        self.issues.push(InspectionIssue {
            code,
            severity,
            detail: detail.into(),
            path,
            metadata_key: None,
            tensor_name: None,
            tensor_type_code: None,
        });
    }
}

/// Inspects a local SafeTensors model directory or GGUF checkpoint without
/// instantiating a model, materializing tensor payloads, or creating an MLX
/// execution stream.
pub fn inspect_model(
    path: impl AsRef<Path>,
    options: ModelInspectionOptions,
) -> Result<ModelInspectionReport, Error> {
    let path = path.as_ref();
    if is_gguf_file(path) {
        Ok(inspect_gguf(path, options))
    } else if path.is_dir() {
        Ok(inspect_safetensors(path, options))
    } else if !path.exists() {
        Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("model artifact does not exist: {}", path.display()),
        )
        .into())
    } else {
        Err(Error::UnsupportedArchitecture(format!(
            "model artifact must be a SafeTensors directory or .gguf file: {}",
            path.display()
        )))
    }
}

fn inspect_safetensors(path: &Path, options: ModelInspectionOptions) -> ModelInspectionReport {
    let mut report = ModelInspectionReport::new(path, ArtifactKind::SafeTensorsDirectory);
    let config_path = path.join("config.json");
    let config: Option<Value> = match std::fs::read_to_string(&config_path) {
        Ok(raw) => match serde_json::from_str(&raw) {
            Ok(value) => Some(value),
            Err(error) => {
                report.container = InspectionReadiness::Invalid;
                report.model_loadability = InspectionReadiness::Invalid;
                report.requested_load = InspectionReadiness::Invalid;
                report.issue(
                    InspectionIssueCode::InvalidConfiguration,
                    InspectionSeverity::Error,
                    format!("invalid config.json: {error}"),
                    Some(config_path.clone()),
                );
                None
            }
        },
        Err(error) => {
            report.container = if error.kind() == std::io::ErrorKind::NotFound {
                InspectionReadiness::Missing
            } else {
                InspectionReadiness::Invalid
            };
            report.model_loadability = report.container;
            report.requested_load = report.container;
            report.issue(
                InspectionIssueCode::InvalidConfiguration,
                InspectionSeverity::Error,
                format!("could not read config.json: {error}"),
                Some(config_path.clone()),
            );
            None
        }
    };

    if let Some(config) = &config {
        match super::config::resolve_model_config(config) {
            Ok(supported) => {
                report.model_kind = Some(supported.kind);
                report.architecture = Some(supported.effective_model_type);
                report.expected_modalities = modalities_for_safetensors(supported.kind, config);
                report.model_loadability = InspectionReadiness::Ready;
                match validate_load_policy(
                    supported.kind,
                    ArtifactLoadKind::Safetensors,
                    options.load,
                ) {
                    Ok(()) => report.requested_load = InspectionReadiness::Ready,
                    Err(error) => reject_load_policy(&mut report, &error),
                }
            }
            Err(error) => {
                report.model_loadability = match &error {
                    super::config::ModelConfigResolutionError::Loader(
                        Error::UnsupportedModelType(_),
                    ) => InspectionReadiness::Unsupported,
                    _ => InspectionReadiness::Invalid,
                };
                report.requested_load = report.model_loadability;
                report.issue(
                    match &error {
                        super::config::ModelConfigResolutionError::Loader(
                            Error::UnsupportedModelType(_),
                        ) => InspectionIssueCode::UnsupportedArchitecture,
                        _ => InspectionIssueCode::InvalidConfiguration,
                    },
                    InspectionSeverity::Error,
                    error.to_string(),
                    Some(config_path),
                );
            }
        }
    }

    match SafetensorsWeightStore::open(path) {
        Ok(store) => {
            let keys = store.keys();
            let mut encodings = BTreeSet::new();
            let mut shards = BTreeSet::new();
            let mut catalog_error = None;
            for key in &keys {
                match store.metadata(key) {
                    Ok(metadata) => {
                        encodings.insert(format!("{:?}", metadata.stored_dtype));
                        if let Some(shard) = metadata.backing_shard {
                            shards.insert(shard);
                        }
                    }
                    Err(error) => {
                        catalog_error = Some(error);
                        break;
                    }
                }
            }
            report.tensor_count = Some(keys.len());
            report.checkpoint_shards = Some(shards.len());
            report.tensor_encodings = encodings
                .into_iter()
                .map(|name| ArtifactTensorEncoding {
                    name,
                    ggml_type_code: None,
                })
                .collect();
            if keys.is_empty() {
                report.container = InspectionReadiness::Invalid;
                report.model_loadability = InspectionReadiness::Invalid;
                report.requested_load = InspectionReadiness::Invalid;
                report.issue(
                    InspectionIssueCode::InvalidContainer,
                    InspectionSeverity::Error,
                    "SafeTensors checkpoint contains no tensors",
                    Some(path.to_path_buf()),
                );
            } else if let Some(error) = catalog_error {
                report.container = InspectionReadiness::Invalid;
                report.model_loadability = InspectionReadiness::Invalid;
                report.requested_load = InspectionReadiness::Invalid;
                let missing = matches!(
                    error,
                    crate::runtime::checkpoint::store::WeightStoreError::MissingShard { .. }
                );
                report.issue(
                    if missing {
                        InspectionIssueCode::MissingCheckpointShard
                    } else {
                        InspectionIssueCode::InvalidContainer
                    },
                    InspectionSeverity::Error,
                    error.to_string(),
                    Some(path.to_path_buf()),
                );
            } else if report.container == InspectionReadiness::Unverified {
                report.container = InspectionReadiness::Ready;
                report.issue(
                    InspectionIssueCode::ValidationUnavailableUntilLoad,
                    InspectionSeverity::Warning,
                    "all SafeTensors shard headers and index mappings are valid; exact architecture parameter binding is rechecked by strict loading because some model modules do not yet expose a pure expected-tensor catalog",
                    Some(path.to_path_buf()),
                );
            }
        }
        Err(error) => {
            report.container = InspectionReadiness::Invalid;
            report.model_loadability = InspectionReadiness::Invalid;
            report.requested_load = InspectionReadiness::Invalid;
            let missing = matches!(
                error,
                crate::runtime::checkpoint::store::WeightStoreError::MissingShard { .. }
            );
            report.issue(
                if missing {
                    InspectionIssueCode::MissingCheckpointShard
                } else {
                    InspectionIssueCode::InvalidContainer
                },
                InspectionSeverity::Error,
                error.to_string(),
                Some(path.to_path_buf()),
            );
        }
    }

    inspect_safetensors_sidecars(&mut report, path, options.chat_request);
    finalize_text_readiness(&mut report);
    report
}

fn inspect_gguf(path: &Path, options: ModelInspectionOptions) -> ModelInspectionReport {
    let mut report = ModelInspectionReport::new(path, ArtifactKind::GgufCheckpoint);
    let checkpoint = match GgufCheckpoint::open(path) {
        Ok(checkpoint) => checkpoint,
        Err(error) => {
            let detail = error.to_string();
            let type_code = parse_unsupported_type_code(&detail);
            report.container = InspectionReadiness::Invalid;
            report.model_loadability = InspectionReadiness::Invalid;
            report.requested_load = InspectionReadiness::Invalid;
            report.text_generation = InspectionReadiness::Invalid;
            report.tokenizer = InspectionReadiness::Unverified;
            report.chat_template = InspectionReadiness::Unverified;
            report.semantic_streaming = InspectionReadiness::Unverified;
            report.native_tools = InspectionReadiness::Unverified;
            report.multimodal = InspectionReadiness::Unverified;
            report.issues.push(InspectionIssue {
                code: if type_code.is_some() {
                    InspectionIssueCode::UnsupportedTensorEncoding
                } else {
                    InspectionIssueCode::InvalidContainer
                },
                severity: InspectionSeverity::Error,
                detail,
                path: Some(path.to_path_buf()),
                metadata_key: None,
                tensor_name: None,
                tensor_type_code: type_code,
            });
            return report;
        }
    };
    report.container = InspectionReadiness::Ready;
    report.checkpoint_shards = Some(checkpoint.catalog().shards().len());
    report.tensor_count = Some(checkpoint.catalog().logical_outputs().count());
    report.gguf_versions = Some(
        checkpoint
            .catalog()
            .shards()
            .iter()
            .map(|shard| shard.version())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect(),
    );
    report.tensor_encodings = checkpoint
        .catalog()
        .tensors()
        .map(|tensor| tensor.descriptor().ggml_type)
        .map(|encoding| (encoding.code(), format!("{encoding:?}")))
        .collect::<BTreeMap<_, _>>()
        .into_iter()
        .map(|(code, name)| ArtifactTensorEncoding {
            name,
            ggml_type_code: Some(code),
        })
        .collect();

    let metadata = crate::runtime::checkpoint::load::gguf_metadata(&checkpoint);
    let architecture = match metadata.get("general.architecture") {
        Some(GgufMetadataValue::String(value)) => {
            report.architecture = Some(value.clone());
            value.clone()
        }
        Some(_) => {
            report.model_loadability = InspectionReadiness::Invalid;
            report.requested_load = InspectionReadiness::Invalid;
            report.issues.push(InspectionIssue {
                code: InspectionIssueCode::InvalidConfiguration,
                severity: InspectionSeverity::Error,
                detail: "GGUF metadata key general.architecture has the wrong type".into(),
                path: Some(path.to_path_buf()),
                metadata_key: Some("general.architecture".into()),
                tensor_name: None,
                tensor_type_code: None,
            });
            inspect_gguf_sidecars(&mut report, path, &metadata, options.chat_request);
            finalize_text_readiness(&mut report);
            return report;
        }
        None => {
            report.model_loadability = InspectionReadiness::Invalid;
            report.requested_load = InspectionReadiness::Invalid;
            report.issues.push(InspectionIssue {
                code: InspectionIssueCode::InvalidConfiguration,
                severity: InspectionSeverity::Error,
                detail: "GGUF metadata is missing required key general.architecture".into(),
                path: Some(path.to_path_buf()),
                metadata_key: Some("general.architecture".into()),
                tensor_name: None,
                tensor_type_code: None,
            });
            inspect_gguf_sidecars(&mut report, path, &metadata, options.chat_request);
            finalize_text_readiness(&mut report);
            return report;
        }
    };

    match GgufArchitecture::resolve(&architecture) {
        Ok(gguf_architecture) => {
            let kind = gguf_architecture.model_kind();
            report.model_kind = Some(kind);
            report.expected_modalities = modalities_for_gguf(gguf_architecture);
            if let Err(error) = gguf_architecture.validate_catalog(&checkpoint, &metadata) {
                report.model_loadability = InspectionReadiness::Invalid;
                report.requested_load = InspectionReadiness::Invalid;
                report.issue(
                    InspectionIssueCode::InvalidConfiguration,
                    InspectionSeverity::Error,
                    error.to_string(),
                    Some(path.to_path_buf()),
                );
            } else {
                report.model_loadability = InspectionReadiness::Ready;
                report.issue(
                    InspectionIssueCode::ValidationUnavailableUntilLoad,
                    InspectionSeverity::Warning,
                    "GGUF architecture metadata, common tensor anchors, encodings, and shared loader policy passed; exact module parameter binding is rechecked by the architecture strict loader",
                    Some(path.to_path_buf()),
                );
            }
            match gguf_architecture.validate_load_policy(options.load) {
                Ok(()) if report.model_loadability == InspectionReadiness::Ready => {
                    match validate_gguf_quantization_source(
                        &checkpoint,
                        &metadata,
                        options.load.quantization,
                    ) {
                        Ok(()) => report.requested_load = InspectionReadiness::Ready,
                        Err(error) => reject_load_policy(&mut report, &error),
                    }
                }
                Ok(()) => {}
                Err(error) => reject_load_policy(&mut report, &error),
            }
            inspect_gguf_projector(&mut report, path, gguf_architecture);
        }
        Err(error) => {
            report.model_loadability = InspectionReadiness::Unsupported;
            report.requested_load = InspectionReadiness::Unsupported;
            report.issue(
                InspectionIssueCode::UnsupportedArchitecture,
                InspectionSeverity::Error,
                error.to_string(),
                Some(path.to_path_buf()),
            );
            report.multimodal = InspectionReadiness::NotApplicable;
        }
    }

    if let Err(error) = gguf_eos_token_ids(&metadata) {
        report.issue(
            InspectionIssueCode::InvalidConfiguration,
            InspectionSeverity::Error,
            error.to_string(),
            Some(path.to_path_buf()),
        );
        report.model_loadability = InspectionReadiness::Invalid;
        report.requested_load = InspectionReadiness::Invalid;
    }
    inspect_gguf_sidecars(&mut report, path, &metadata, options.chat_request);
    finalize_text_readiness(&mut report);
    report
}

fn inspect_safetensors_sidecars(
    report: &mut ModelInspectionReport,
    path: &Path,
    request: Option<ChatTemplateRequest>,
) {
    let tokenizer = match load_tokenizer(path) {
        Ok(tokenizer) => {
            report.tokenizer = InspectionReadiness::Ready;
            Some(tokenizer)
        }
        Err(error) => {
            report.tokenizer = InspectionReadiness::Missing;
            report.issue(
                InspectionIssueCode::MissingTokenizer,
                InspectionSeverity::Error,
                error.to_string(),
                Some(path.join("tokenizer.json")),
            );
            None
        }
    };
    let template = match load_chat_template(path) {
        Ok(Some(template)) => {
            report.chat_template = InspectionReadiness::Ready;
            Some(template)
        }
        Ok(None) => {
            report.chat_template = InspectionReadiness::Missing;
            report.issue(
                InspectionIssueCode::MissingChatTemplate,
                InspectionSeverity::Warning,
                "no tokenizer_config.json chat_template or chat_template.jinja is available",
                Some(path.to_path_buf()),
            );
            None
        }
        Err(error) => {
            report.chat_template = InspectionReadiness::Invalid;
            report.issue(
                InspectionIssueCode::MissingChatTemplate,
                InspectionSeverity::Warning,
                error.to_string(),
                Some(path.to_path_buf()),
            );
            None
        }
    };
    if let (Some(tokenizer), Some(template)) = (tokenizer, template) {
        let kwargs = load_tokenizer_template_kwargs(path).unwrap_or_default();
        let eos = eos_token_ids_from_sidecar_dir(path).unwrap_or_default();
        inspect_chat_behavior(report, tokenizer, template, kwargs, eos, request);
    } else {
        report.semantic_streaming = InspectionReadiness::Missing;
        report.native_tools = InspectionReadiness::Missing;
    }
    inspect_safetensors_media(report, path);
}

fn inspect_gguf_sidecars(
    report: &mut ModelInspectionReport,
    path: &Path,
    metadata: &std::collections::HashMap<String, GgufMetadataValue>,
    request: Option<ChatTemplateRequest>,
) {
    let tokenizer = match load_gguf_tokenizer_from_metadata(path, metadata) {
        Ok(tokenizer) => {
            report.tokenizer = InspectionReadiness::Ready;
            Some(tokenizer)
        }
        Err(error) => {
            report.tokenizer = InspectionReadiness::Missing;
            report.issue(
                InspectionIssueCode::MissingTokenizer,
                InspectionSeverity::Error,
                format!(
                    "GGUF tokenizer metadata is unusable and no acceptable sibling tokenizer.json was loaded: {error}"
                ),
                Some(gguf_sidecar_dir(path).join("tokenizer.json")),
            );
            None
        }
    };
    let embedded = match metadata.get("tokenizer.chat_template") {
        Some(GgufMetadataValue::String(template)) => {
            Some(ModelChatTemplate::Single(template.clone()))
        }
        Some(_) => {
            report.chat_template = InspectionReadiness::Invalid;
            report.issues.push(InspectionIssue {
                code: InspectionIssueCode::MissingChatTemplate,
                severity: InspectionSeverity::Warning,
                detail: "GGUF tokenizer.chat_template must be a string".into(),
                path: Some(path.to_path_buf()),
                metadata_key: Some("tokenizer.chat_template".into()),
                tensor_name: None,
                tensor_type_code: None,
            });
            None
        }
        None => None,
    };
    let template = embedded.or_else(|| load_chat_template(gguf_sidecar_dir(path)).ok().flatten());
    if template.is_some() {
        report.chat_template = InspectionReadiness::Ready;
    } else if report.chat_template != InspectionReadiness::Invalid {
        report.chat_template = InspectionReadiness::Missing;
        report.issue(
            InspectionIssueCode::MissingChatTemplate,
            InspectionSeverity::Warning,
            "GGUF has no embedded chat template and no acceptable sidecar template",
            Some(path.to_path_buf()),
        );
    }
    if let (Some(tokenizer), Some(template)) = (tokenizer, template) {
        let eos = merge_eos_token_id_sources([
            eos_token_ids_from_sidecar_dir(gguf_sidecar_dir(path)).unwrap_or_default(),
            gguf_eos_token_ids(metadata).unwrap_or_default(),
        ]);
        inspect_chat_behavior(
            report,
            tokenizer.tokenizer,
            template,
            tokenizer.template_kwargs,
            eos,
            request,
        );
    } else {
        report.semantic_streaming = InspectionReadiness::Missing;
        report.native_tools = InspectionReadiness::Missing;
    }
}

fn inspect_chat_behavior(
    report: &mut ModelInspectionReport,
    tokenizer: tokenizers::Tokenizer,
    template: ModelChatTemplate,
    kwargs: Map<String, Value>,
    eos_token_ids: Vec<u32>,
    request: Option<ChatTemplateRequest>,
) {
    let mut tokenizer = ChatTokenizer::from_tokenizer(tokenizer);
    tokenizer.set_template_kwargs(kwargs);
    let compiler = ConstraintCompiler::from_tokenizer(&tokenizer, &eos_token_ids);
    let model_id = report.path.display().to_string();
    if let Some(request) = request {
        match prepare_chat_from_parts(
            &mut tokenizer,
            template,
            &model_id,
            &eos_token_ids,
            Some(&compiler),
            request,
        ) {
            Ok(prepared) => apply_prepared_chat(report, &prepared),
            Err(error) => {
                report.semantic_streaming = InspectionReadiness::Unsupported;
                report.native_tools = InspectionReadiness::Unsupported;
                report.issue(
                    InspectionIssueCode::UnsupportedSemanticProtocol,
                    InspectionSeverity::Error,
                    error.to_string(),
                    Some(report.path.clone()),
                );
            }
        }
        return;
    }

    let semantic_request = ChatTemplateRequest {
        messages: vec![json!({"role": "user", "content": "__safemlx_inspection_probe__"})],
        tool_choice: ToolChoice::None,
        add_generation_prompt: true,
        ..ChatTemplateRequest::default()
    };
    match prepare_chat_from_parts(
        &mut tokenizer,
        template.clone(),
        &model_id,
        &eos_token_ids,
        Some(&compiler),
        semantic_request,
    ) {
        Ok(prepared) => {
            report.semantic_streaming = match prepared.semantic_support() {
                SemanticSupport::Supported => InspectionReadiness::Ready,
                SemanticSupport::Unsupported { reason } => {
                    report.issue(
                        InspectionIssueCode::UnsupportedSemanticProtocol,
                        InspectionSeverity::Warning,
                        reason.clone(),
                        Some(report.path.clone()),
                    );
                    InspectionReadiness::Unsupported
                }
            };
        }
        Err(error) => {
            report.semantic_streaming = InspectionReadiness::Unsupported;
            report.issue(
                InspectionIssueCode::UnsupportedSemanticProtocol,
                InspectionSeverity::Warning,
                error.to_string(),
                Some(report.path.clone()),
            );
        }
    }

    let tool_request = ChatTemplateRequest {
        messages: vec![json!({"role": "user", "content": "__safemlx_tool_probe__"})],
        tools: vec![json!({
            "type": "function",
            "function": {
                "name": "safemlx_probe",
                "description": "inspection probe",
                "parameters": {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ["value"]
                }
            }
        })],
        tool_choice: ToolChoice::Required,
        add_generation_prompt: true,
        ..ChatTemplateRequest::default()
    };
    match prepare_chat_from_parts(
        &mut tokenizer,
        template,
        &model_id,
        &eos_token_ids,
        Some(&compiler),
        tool_request,
    ) {
        Ok(prepared) => {
            report.native_tools = match prepared.native_tool_support() {
                NativeToolSupport::Supported => InspectionReadiness::Ready,
                NativeToolSupport::Unsupported { reason } => {
                    report.issue(
                        InspectionIssueCode::UnsupportedToolProtocol,
                        InspectionSeverity::Warning,
                        reason.clone(),
                        Some(report.path.clone()),
                    );
                    InspectionReadiness::Unsupported
                }
            };
        }
        Err(error) => {
            report.native_tools = InspectionReadiness::Unsupported;
            report.issue(
                InspectionIssueCode::UnsupportedToolProtocol,
                InspectionSeverity::Warning,
                error.to_string(),
                Some(report.path.clone()),
            );
        }
    }
    report.issue(
        InspectionIssueCode::RequestSpecificValidation,
        InspectionSeverity::Info,
        "native-tool readiness used a bounded behavioral probe; validate real messages, tool schemas, choices, parallel-call policy, and template kwargs with chat_request",
        Some(report.path.clone()),
    );
}

fn apply_prepared_chat(report: &mut ModelInspectionReport, prepared: &PreparedChat) {
    report.semantic_streaming = match prepared.semantic_support() {
        SemanticSupport::Supported => InspectionReadiness::Ready,
        SemanticSupport::Unsupported { reason } => {
            report.issue(
                InspectionIssueCode::UnsupportedSemanticProtocol,
                InspectionSeverity::Warning,
                reason.clone(),
                Some(report.path.clone()),
            );
            InspectionReadiness::Unsupported
        }
    };
    report.native_tools = match prepared.native_tool_support() {
        NativeToolSupport::Supported => InspectionReadiness::Ready,
        NativeToolSupport::Unsupported { reason } => {
            report.issue(
                InspectionIssueCode::UnsupportedToolProtocol,
                InspectionSeverity::Warning,
                reason.clone(),
                Some(report.path.clone()),
            );
            InspectionReadiness::Unsupported
        }
    };
}

fn inspect_gguf_projector(
    report: &mut ModelInspectionReport,
    path: &Path,
    architecture: GgufArchitecture,
) {
    match architecture {
        GgufArchitecture::Qwen3Vl => match qwen3_vl::find_qwen3_vl_mmproj(path) {
            Ok(projector) => match GgufCheckpoint::open(&projector) {
                Ok(checkpoint) => {
                    let metadata = crate::runtime::checkpoint::load::gguf_metadata(&checkpoint);
                    match qwen3_vl::validate_qwen3_vl_mmproj(&metadata) {
                        Ok(()) => {
                            report.multimodal = if cfg!(feature = "image-processing") {
                                InspectionReadiness::Ready
                            } else {
                                InspectionReadiness::Unsupported
                            };
                            report.requirements.push(InspectionRequirement {
                                code: InspectionIssueCode::MissingMediaProjector,
                                readiness: InspectionReadiness::Ready,
                                detail: "validated qwen3vl vision projector".into(),
                                path: Some(projector),
                            });
                        }
                        Err(error) => reject_projector(report, projector, error.to_string(), true),
                    }
                }
                Err(error) => reject_projector(report, projector, error.to_string(), true),
            },
            Err(error) => reject_projector(report, path.to_path_buf(), error.to_string(), true),
        },
        GgufArchitecture::Inkling => match inkling::open_sibling_mmproj(path) {
            Ok(Some(_)) => {
                report.multimodal = if cfg!(all(
                    feature = "image-processing",
                    feature = "audio-processing"
                )) {
                    InspectionReadiness::Ready
                } else {
                    InspectionReadiness::Unsupported
                };
            }
            Ok(None) => {
                report.multimodal = InspectionReadiness::Missing;
                report.requirements.push(InspectionRequirement {
                    code: InspectionIssueCode::MissingMediaProjector,
                    readiness: InspectionReadiness::Missing,
                    detail: "Inkling text loading is available, but image/audio input requires a sibling mmproj GGUF".into(),
                    path: None,
                });
                report.issue(
                    InspectionIssueCode::MissingMediaProjector,
                    InspectionSeverity::Warning,
                    "Inkling has no sibling multimodal projector; text loading remains available",
                    Some(path.to_path_buf()),
                );
            }
            Err(error) => reject_projector(report, path.to_path_buf(), error.to_string(), false),
        },
        _ => report.multimodal = InspectionReadiness::NotApplicable,
    }
}

fn reject_projector(
    report: &mut ModelInspectionReport,
    path: PathBuf,
    detail: String,
    required_for_model: bool,
) {
    report.multimodal = InspectionReadiness::Missing;
    if required_for_model {
        report.model_loadability = InspectionReadiness::Missing;
        report.requested_load = InspectionReadiness::Missing;
    }
    report.requirements.push(InspectionRequirement {
        code: InspectionIssueCode::MissingMediaProjector,
        readiness: InspectionReadiness::Missing,
        detail: detail.clone(),
        path: Some(path.clone()),
    });
    report.issue(
        InspectionIssueCode::MissingMediaProjector,
        if required_for_model {
            InspectionSeverity::Error
        } else {
            InspectionSeverity::Warning
        },
        detail,
        Some(path),
    );
}

fn inspect_safetensors_media(report: &mut ModelInspectionReport, path: &Path) {
    if report.expected_modalities == [ArtifactModality::Text] {
        report.multimodal = InspectionReadiness::NotApplicable;
        return;
    }
    let sidecar = ["processor_config.json", "preprocessor_config.json"]
        .into_iter()
        .map(|name| path.join(name))
        .find(|candidate| candidate.exists());
    if let Some(sidecar) = sidecar {
        let features_available = report
            .expected_modalities
            .iter()
            .all(|modality| match modality {
                ArtifactModality::Text => true,
                ArtifactModality::Image | ArtifactModality::Video => {
                    cfg!(feature = "image-processing")
                }
                ArtifactModality::Audio => cfg!(feature = "audio-processing"),
            });
        report.multimodal = if features_available {
            InspectionReadiness::Ready
        } else {
            InspectionReadiness::Unsupported
        };
        report.requirements.push(InspectionRequirement {
            code: InspectionIssueCode::MissingProcessor,
            readiness: report.multimodal,
            detail: if features_available {
                "processor sidecar and required media build features are available".into()
            } else {
                "processor sidecar exists, but required image/audio processing features are not enabled".into()
            },
            path: Some(sidecar),
        });
    } else {
        report.multimodal = InspectionReadiness::Missing;
        report.issue(
            InspectionIssueCode::MissingProcessor,
            InspectionSeverity::Warning,
            "the resolved multimodal architecture has no processor_config.json or preprocessor_config.json sidecar",
            Some(path.to_path_buf()),
        );
    }
}

fn modalities_for_safetensors(kind: ModelKind, config: &Value) -> Vec<ArtifactModality> {
    let mut modalities = BTreeSet::from([ArtifactModality::Text]);
    match kind {
        ModelKind::Gemma4 | ModelKind::Inkling => {
            modalities.insert(ArtifactModality::Image);
            modalities.insert(ArtifactModality::Audio);
        }
        ModelKind::Qwen3Vl | ModelKind::Qwen3VlMoe => {
            modalities.insert(ArtifactModality::Image);
            modalities.insert(ArtifactModality::Video);
        }
        ModelKind::Qwen35Moe
            if config.get("vision_config").is_some()
                || config
                    .get("text_config")
                    .and_then(|text| text.get("vision_config"))
                    .is_some() =>
        {
            modalities.insert(ArtifactModality::Image);
            modalities.insert(ArtifactModality::Video);
        }
        ModelKind::DeepSeekV3
        | ModelKind::GptOss
        | ModelKind::KimiLinear
        | ModelKind::Llama
        | ModelKind::Lfm2
        | ModelKind::NemotronH
        | ModelKind::PersonaPlex
        | ModelKind::Qwen3
        | ModelKind::Qwen3Next
        | ModelKind::Qwen35Moe => {}
    }
    modalities.into_iter().collect()
}

fn modalities_for_gguf(architecture: GgufArchitecture) -> Vec<ArtifactModality> {
    match architecture {
        GgufArchitecture::Inkling => vec![
            ArtifactModality::Text,
            ArtifactModality::Image,
            ArtifactModality::Audio,
        ],
        GgufArchitecture::Qwen3Vl => vec![
            ArtifactModality::Text,
            ArtifactModality::Image,
            ArtifactModality::Video,
        ],
        _ => vec![ArtifactModality::Text],
    }
}

fn reject_load_policy(report: &mut ModelInspectionReport, error: &Error) {
    report.requested_load = InspectionReadiness::Unsupported;
    let (code, detail) = match error {
        Error::Quantization(detail) => (
            InspectionIssueCode::UnsupportedQuantizationRequest,
            detail.clone(),
        ),
        Error::Parallel(detail) => (
            InspectionIssueCode::UnsupportedParallelTopology,
            detail.clone(),
        ),
        Error::UnsupportedArchitecture(detail)
            if detail.contains("residency")
                || detail.contains("stream")
                || detail.contains("expert cach") =>
        {
            (
                InspectionIssueCode::UnsupportedResidencyPolicy,
                detail.clone(),
            )
        }
        _ => (
            InspectionIssueCode::UnsupportedArchitecture,
            error.to_string(),
        ),
    };
    report.issue(
        code,
        InspectionSeverity::Error,
        detail,
        Some(report.path.clone()),
    );
}

fn finalize_text_readiness(report: &mut ModelInspectionReport) {
    report.text_generation = if report.model_loadability == InspectionReadiness::Ready
        && report.requested_load == InspectionReadiness::Ready
        && report.tokenizer == InspectionReadiness::Ready
    {
        InspectionReadiness::Ready
    } else if report.model_loadability == InspectionReadiness::Invalid
        || report.container == InspectionReadiness::Invalid
    {
        InspectionReadiness::Invalid
    } else if report.model_loadability == InspectionReadiness::Unsupported
        || report.requested_load == InspectionReadiness::Unsupported
    {
        InspectionReadiness::Unsupported
    } else if report.tokenizer == InspectionReadiness::Missing
        || report.model_loadability == InspectionReadiness::Missing
    {
        InspectionReadiness::Missing
    } else {
        InspectionReadiness::Unverified
    };
}

fn parse_unsupported_type_code(detail: &str) -> Option<u32> {
    let marker = "unsupported GGML type ";
    let start = detail.find(marker)? + marker.len();
    detail[start..]
        .split(|character: char| !character.is_ascii_digit())
        .next()?
        .parse()
        .ok()
}

#[cfg(test)]
mod tests {
    use std::io::{Seek, Write};

    use safemlx_gguf::{GgmlType, MetadataValue, TensorInput, Writer};
    use safetensors::tensor::{serialize_to_file, Dtype, TensorView};
    use tokenizers::{
        decoders::byte_level::ByteLevel, models::wordlevel::WordLevel, AddedToken, Tokenizer,
    };

    use super::*;

    fn llama_config() -> Value {
        json!({
            "model_type": "llama",
            "hidden_size": 8,
            "num_hidden_layers": 1,
            "intermediate_size": 16,
            "num_attention_heads": 2,
            "rms_norm_eps": 0.00001,
            "vocab_size": 32,
            "num_key_value_heads": 2,
            "max_position_embeddings": 128,
            "head_dim": 4
        })
    }

    fn write_safetensors_dir(config: &Value) -> tempfile::TempDir {
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(
            directory.path().join("config.json"),
            serde_json::to_vec(config).unwrap(),
        )
        .unwrap();
        let bytes = [0xff_u8; 4];
        let view = TensorView::new(Dtype::F32, vec![1], &bytes).unwrap();
        serialize_to_file(
            [("poison.weight", view)],
            None,
            &directory.path().join("model.safetensors"),
        )
        .unwrap();
        directory
    }

    fn save_wordlevel_tokenizer(path: &Path) {
        std::fs::write(
            path,
            br#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"<unk>":0,"hello":1,"__safemlx_inspection_probe__":2,"__safemlx_tool_probe__":3},"unk_token":"<unk>"}}"#,
        )
        .unwrap();
    }

    fn write_gguf(
        path: &Path,
        architecture: &str,
        encoding: GgmlType,
        extra: impl IntoIterator<Item = (String, MetadataValue)>,
    ) {
        let mut metadata = BTreeMap::from([
            (
                "general.architecture".into(),
                MetadataValue::String(architecture.into()),
            ),
            ("general.file_type".into(), MetadataValue::Uint32(0)),
            (
                format!("{architecture}.block_count"),
                MetadataValue::Uint32(1),
            ),
            (
                format!("{architecture}.embedding_length"),
                MetadataValue::Uint32(8),
            ),
        ]);
        metadata.extend(extra);
        let (block, bytes) = encoding.block_and_bytes().unwrap();
        let payload = vec![0xff; bytes as usize];
        Writer::default()
            .write(
                std::fs::File::create(path).unwrap(),
                &metadata,
                &[TensorInput {
                    name: "token_embd.weight",
                    dimensions: &[block],
                    ggml_type: encoding,
                    data: &payload,
                }],
            )
            .unwrap();
    }

    fn write_unknown_gguf_type(path: &Path, code: u32) {
        let mut file = std::fs::File::create(path).unwrap();
        file.write_all(b"GGUF").unwrap();
        file.write_all(&3_u32.to_le_bytes()).unwrap();
        file.write_all(&1_u64.to_le_bytes()).unwrap();
        file.write_all(&0_u64.to_le_bytes()).unwrap();
        file.write_all(&1_u64.to_le_bytes()).unwrap();
        file.write_all(b"x").unwrap();
        file.write_all(&1_u32.to_le_bytes()).unwrap();
        file.write_all(&1_u64.to_le_bytes()).unwrap();
        file.write_all(&code.to_le_bytes()).unwrap();
        file.write_all(&0_u64.to_le_bytes()).unwrap();
        let position = file.stream_position().unwrap();
        file.write_all(&vec![0; (64 - position) as usize]).unwrap();
    }

    #[test]
    fn loader_and_inspection_model_kind_dispatch_are_exhaustive() {
        for kind in ModelKind::ALL {
            let result = validate_load_policy(
                kind,
                ArtifactLoadKind::Safetensors,
                ModelLoadOptions::default(),
            );
            assert_eq!(result.is_ok(), kind != ModelKind::PersonaPlex, "{kind:?}");
        }
    }

    #[test]
    fn gguf_architecture_resolution_covers_documented_dispatch() {
        for name in GgufArchitecture::SUPPORTED_NAMES
            .replace(", and ", ", ")
            .split(", ")
        {
            assert!(GgufArchitecture::resolve(name).is_ok(), "{name}");
        }
    }

    #[test]
    fn supported_safetensors_directory_is_cataloged_without_reading_payload_values() {
        let directory = write_safetensors_dir(&llama_config());
        let report = inspect_model(directory.path(), ModelInspectionOptions::default()).unwrap();
        assert_eq!(report.container, InspectionReadiness::Ready);
        assert_eq!(report.model_kind, Some(ModelKind::Llama));
        assert_eq!(report.model_loadability, InspectionReadiness::Ready);
        assert_eq!(report.requested_load, InspectionReadiness::Ready);
        assert_eq!(report.tensor_count, Some(1));
        assert_eq!(report.tensor_encodings[0].name, "F32");
    }

    #[test]
    fn safetensors_config_rejections_are_structured_and_legacy_check_remains_compatible() {
        let unsupported = write_safetensors_dir(&json!({"model_type": "falcon"}));
        let report = inspect_model(unsupported.path(), ModelInspectionOptions::default()).unwrap();
        assert_eq!(report.model_loadability, InspectionReadiness::Unsupported);
        assert!(report
            .issues
            .iter()
            .any(|issue| issue.code == InspectionIssueCode::UnsupportedArchitecture));
        assert!(!check_model_dir(unsupported.path()).is_supported());

        let invalid = write_safetensors_dir(&json!({"model_type": "llama", "hidden_size": "bad"}));
        let report = inspect_model(invalid.path(), ModelInspectionOptions::default()).unwrap();
        assert_eq!(report.model_loadability, InspectionReadiness::Invalid);
        assert!(report
            .issues
            .iter()
            .any(|issue| issue.code == InspectionIssueCode::InvalidConfiguration));
        assert!(
            !check_model_config(&json!({"model_type":"llama","hidden_size":"bad"})).is_supported()
        );
        assert!(check_model_config(&json!({}))
            .unsupported_reason()
            .is_some_and(|reason| reason.starts_with("invalid model config metadata:")));
    }

    #[test]
    fn missing_safetensors_index_shard_is_rejected_before_load() {
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(
            directory.path().join("config.json"),
            serde_json::to_vec(&llama_config()).unwrap(),
        )
        .unwrap();
        std::fs::write(
            directory.path().join("model.safetensors.index.json"),
            br#"{"weight_map":{"model.embed_tokens.weight":"missing-00001-of-00002.safetensors"}}"#,
        )
        .unwrap();
        let report = inspect_model(directory.path(), ModelInspectionOptions::default()).unwrap();
        assert_eq!(report.container, InspectionReadiness::Invalid);
        assert!(report
            .issues
            .iter()
            .any(|issue| issue.code == InspectionIssueCode::MissingCheckpointShard));
    }

    #[test]
    fn supported_and_unsupported_gguf_architectures_are_distinguished() {
        let directory = tempfile::tempdir().unwrap();
        let supported = directory.path().join("supported.gguf");
        write_gguf(
            &supported,
            "llama",
            GgmlType::Q4K,
            std::iter::empty::<(String, MetadataValue)>(),
        );
        let report = inspect_model(&supported, ModelInspectionOptions::default()).unwrap();
        assert_eq!(report.container, InspectionReadiness::Ready);
        assert_eq!(report.model_loadability, InspectionReadiness::Ready);
        assert_eq!(report.tensor_encodings[0].ggml_type_code, Some(12));

        let unsupported = directory.path().join("unsupported.gguf");
        write_gguf(
            &unsupported,
            "falcon",
            GgmlType::F16,
            std::iter::empty::<(String, MetadataValue)>(),
        );
        let report = inspect_model(&unsupported, ModelInspectionOptions::default()).unwrap();
        assert_eq!(report.model_loadability, InspectionReadiness::Unsupported);
        assert!(report
            .issues
            .iter()
            .any(|issue| issue.code == InspectionIssueCode::UnsupportedArchitecture));
    }

    #[test]
    fn unknown_gguf_tensor_type_has_a_numeric_structured_diagnostic() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("unknown.gguf");
        write_unknown_gguf_type(&path, 1234);
        let report = inspect_model(&path, ModelInspectionOptions::default()).unwrap();
        let issue = report
            .issues
            .iter()
            .find(|issue| issue.code == InspectionIssueCode::UnsupportedTensorEncoding)
            .unwrap();
        assert_eq!(issue.tensor_type_code, Some(1234));
    }

    #[test]
    fn gguf_tokenizer_sidecar_is_an_explicit_fallback() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("model.gguf");
        write_gguf(
            &path,
            "llama",
            GgmlType::F16,
            std::iter::empty::<(String, MetadataValue)>(),
        );
        let missing = inspect_model(&path, ModelInspectionOptions::default()).unwrap();
        assert_eq!(missing.tokenizer, InspectionReadiness::Missing);

        save_wordlevel_tokenizer(&directory.path().join("tokenizer.json"));
        let available = inspect_model(&path, ModelInspectionOptions::default()).unwrap();
        assert_eq!(available.tokenizer, InspectionReadiness::Ready);
        assert_eq!(available.text_generation, InspectionReadiness::Ready);
    }

    #[test]
    fn missing_required_qwen_projector_blocks_model_admission() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("qwen.gguf");
        write_gguf(
            &path,
            "qwen3vl",
            GgmlType::F16,
            std::iter::empty::<(String, MetadataValue)>(),
        );
        let report = inspect_model(&path, ModelInspectionOptions::default()).unwrap();
        assert_eq!(report.model_loadability, InspectionReadiness::Missing);
        assert!(report
            .issues
            .iter()
            .any(|issue| issue.code == InspectionIssueCode::MissingMediaProjector));
    }

    #[test]
    fn requested_quantization_and_residency_use_shared_loader_policy() {
        use crate::runtime::{
            checkpoint::quantization::WeightQuantization, execution::layerwise::WeightResidency,
            residency::expert_cache::ExpertCacheLoadOptions,
        };

        let directory = tempfile::tempdir().unwrap();
        let gguf = directory.path().join("packed.gguf");
        write_gguf(
            &gguf,
            "llama",
            GgmlType::Q4_0,
            [("general.file_type".into(), MetadataValue::Uint32(2))],
        );
        let report = inspect_model(
            &gguf,
            ModelInspectionOptions {
                load: ModelLoadOptions::with_quantization(WeightQuantization::MxFp4),
                chat_request: None,
            },
        )
        .unwrap();
        assert_eq!(report.requested_load, InspectionReadiness::Unsupported);
        assert!(report
            .issues
            .iter()
            .any(|issue| issue.code == InspectionIssueCode::UnsupportedQuantizationRequest));

        let safetensors = write_safetensors_dir(&llama_config());
        let report = inspect_model(
            safetensors.path(),
            ModelInspectionOptions {
                load: ModelLoadOptions::default().with_weight_residency(
                    WeightResidency::SparseExpertCache(ExpertCacheLoadOptions::default()),
                ),
                chat_request: None,
            },
        )
        .unwrap();
        assert_eq!(report.requested_load, InspectionReadiness::Unsupported);
        assert!(report
            .issues
            .iter()
            .any(|issue| issue.code == InspectionIssueCode::UnsupportedResidencyPolicy));

        let report = inspect_model(
            &gguf,
            ModelInspectionOptions {
                load: ModelLoadOptions::default().with_weight_residency(
                    WeightResidency::SparseExpertCache(ExpertCacheLoadOptions::default()),
                ),
                chat_request: None,
            },
        )
        .unwrap();
        assert_eq!(report.requested_load, InspectionReadiness::Unsupported);
        assert!(report
            .issues
            .iter()
            .any(|issue| issue.code == InspectionIssueCode::UnsupportedResidencyPolicy));
    }

    #[test]
    fn template_presence_does_not_imply_semantic_or_tool_support() {
        let directory = write_safetensors_dir(&llama_config());
        save_wordlevel_tokenizer(&directory.path().join("tokenizer.json"));
        std::fs::write(
            directory.path().join("tokenizer_config.json"),
            r#"{"chat_template":"{% for message in messages %}{{ message['content'] }}{% endfor %}"}"#,
        )
        .unwrap();
        let report = inspect_model(directory.path(), ModelInspectionOptions::default()).unwrap();
        assert_eq!(report.chat_template, InspectionReadiness::Ready);
        assert_eq!(report.semantic_streaming, InspectionReadiness::Unsupported);
        assert_eq!(report.native_tools, InspectionReadiness::Unsupported);
    }

    #[test]
    fn chat_protocol_is_recognized_from_rendered_behavior_after_source_refactor() {
        const TEMPLATE: &str =
            include_str!("../../tests/fixtures/chat_templates/gemma-4-e2b-it-3e22461f.jinja");
        let mut tokenizer = Tokenizer::new(WordLevel::default());
        tokenizer
            .add_tokens(
                (0..50)
                    .map(|index| AddedToken::from(format!("ordinary_{index}"), false))
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        tokenizer
            .add_special_tokens(
                [
                    "<|channel>",
                    "<channel|>",
                    "<|tool_call>",
                    "<tool_call|>",
                    "<|\"|>",
                    "<|tool_response>",
                    "<turn|>",
                ]
                .map(|token| AddedToken::from(token, true).normalized(false)),
            )
            .unwrap();
        tokenizer.with_decoder(Some(ByteLevel::default()));
        let directory = tempfile::tempdir().unwrap();
        let mut report =
            ModelInspectionReport::new(directory.path(), ArtifactKind::SafeTensorsDirectory);
        inspect_chat_behavior(
            &mut report,
            tokenizer,
            ModelChatTemplate::Single(format!(
                "{TEMPLATE}\n{{# source-only inspection refactor #}}"
            )),
            Map::from_iter([
                ("bos_token".into(), json!("<bos>")),
                ("eos_token".into(), json!("<eos>")),
            ]),
            Vec::new(),
            None,
        );
        assert_eq!(
            report.semantic_streaming,
            InspectionReadiness::Ready,
            "{:#?}",
            report.issues
        );
        assert_eq!(
            report.native_tools,
            InspectionReadiness::Ready,
            "{:#?}",
            report.issues
        );
    }
}
