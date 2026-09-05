//! Portable model-artifact inspection results.

use crate::{
    artifact::ArtifactError, ArtifactFormat, ArtifactInspection, GgufCompanionRole,
    InputModalities, ModelResourceProfile, Observed, PreparationAdmission,
    PreparationAdmissionError, SessionCapabilities,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
};

/// A readiness result that preserves distinct failure modes.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InspectionReadiness {
    /// The inspected artifacts establish this capability.
    Ready,
    /// The artifact omits a required component.
    Missing,
    /// The selected backend does not implement the combination.
    Unsupported,
    /// Relevant artifact data is malformed.
    Invalid,
    /// A concrete request is needed before deciding.
    RequestDependent,
    /// The check necessarily occurs during preparation or execution.
    Unverified,
    /// The capability does not apply.
    NotApplicable,
}

/// Mechanism observations required to finalize an admitted inspection report.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct RealizedInspectionOutcomes {
    /// The admitted artifact remained authoritative through construction.
    pub artifact_authority: bool,
    /// Canonical bindings were published into the constructed module.
    pub binding: bool,
    /// Native-independent model construction completed.
    pub construction: bool,
    /// The selected communication realization was exercised.
    pub communication: bool,
    /// At least one session operation completed and was observed.
    pub execution: bool,
    /// Facilities actually observed on the constructed session.
    pub session: SessionCapabilities,
}

/// Finalizes readiness from actual construction and execution observations.
///
/// Artifact identity and descriptive fields remain those admitted by the
/// portable report. Readiness is recomputed from realized mechanisms, so a
/// backend adapter cannot prove realization by replaying artifact inspection.
pub fn finalize_realized_model_inspection(
    admitted: &ModelInspectionReport,
    outcomes: RealizedInspectionOutcomes,
) -> ModelInspectionReport {
    let mut report = admitted.clone();
    let admission = admitted.preparation_admission;
    let session_matches =
        admission.is_some_and(|value| value.session_capabilities() == outcomes.session);
    report.container = if outcomes.artifact_authority {
        InspectionReadiness::Ready
    } else {
        InspectionReadiness::Invalid
    };
    report.architecture_support = if outcomes.construction {
        InspectionReadiness::Ready
    } else {
        InspectionReadiness::Invalid
    };
    report.structural_binding = if outcomes.binding {
        InspectionReadiness::Ready
    } else {
        InspectionReadiness::Invalid
    };
    report.model_loadability = if outcomes.artifact_authority
        && outcomes.binding
        && outcomes.construction
        && outcomes.communication
        && outcomes.execution
    {
        InspectionReadiness::Ready
    } else {
        InspectionReadiness::Invalid
    };
    report.requested_load =
        if report.model_loadability == InspectionReadiness::Ready && session_matches {
            InspectionReadiness::Ready
        } else {
            InspectionReadiness::Unsupported
        };
    report
}

/// Assembles the portable portion of a model inspection report from one
/// authoritative artifact admission and its exact preparation admission.
///
/// This is the canonical adapter surface for backend-independent callers: all
/// readiness derived from artifact headers, architecture capabilities, load
/// policy, and portable media facilities is recorded here rather than rebuilt
/// by individual backends or integration adapters.
pub fn assemble_portable_model_inspection<P>(
    inspection: &ArtifactInspection<P>,
    admission: PreparationAdmission,
    modalities: InputModalities,
    embedded_draft_layers: Option<usize>,
    processor: Option<(bool, MediaFeatureAvailability)>,
) -> ModelInspectionReport {
    let mut report = ModelInspectionReport::unverified(inspection.path(), inspection.format());
    report.record_artifact_inspection(inspection);
    report.record_architecture_capabilities(modalities, embedded_draft_layers);
    if let Some((has_processor, availability)) = processor {
        record_processor_inspection(&mut report, has_processor, availability);
    }
    report.record_preparation_admission(admission);
    report
}

/// Records one failed portable artifact inspection without backend policy.
pub fn reject_portable_artifact_inspection(
    report: &mut ModelInspectionReport,
    path: &Path,
    error: &ArtifactError,
) {
    let detail = error.to_string();
    let missing_media_projector = matches!(
        error,
        ArtifactError::MissingRequiredGgufCompanion {
            role: GgufCompanionRole::MediaProjector,
            ..
        }
    );
    let (code, container, architecture, structural, type_code) = match error {
        ArtifactError::UnsupportedGgufArchitecture(name) => {
            report.architecture = Some(name.clone());
            (
                InspectionIssueCode::UnsupportedArchitecture,
                InspectionReadiness::Ready,
                InspectionReadiness::Unsupported,
                InspectionReadiness::Unsupported,
                None,
            )
        }
        ArtifactError::MissingGgufArchitecture => (
            InspectionIssueCode::InvalidConfiguration,
            InspectionReadiness::Ready,
            InspectionReadiness::Invalid,
            InspectionReadiness::Invalid,
            None,
        ),
        ArtifactError::MissingRequiredGgufCompanion {
            role: GgufCompanionRole::MediaProjector,
            ..
        } => (
            InspectionIssueCode::MissingMediaProjector,
            InspectionReadiness::Ready,
            InspectionReadiness::Ready,
            InspectionReadiness::Unverified,
            None,
        ),
        ArtifactError::InvalidArtifact(_)
        | ArtifactError::InvalidArchitecturePlan(_)
        | ArtifactError::DuplicateTensor(_)
        | ArtifactError::Catalog(_) => (
            InspectionIssueCode::InvalidConfiguration,
            InspectionReadiness::Ready,
            InspectionReadiness::Unverified,
            InspectionReadiness::Invalid,
            None,
        ),
        ArtifactError::Gguf(error) => {
            let type_code = error.unsupported_tensor_type_code();
            (
                if type_code.is_some() {
                    InspectionIssueCode::UnsupportedTensorEncoding
                } else {
                    InspectionIssueCode::InvalidContainer
                },
                InspectionReadiness::Invalid,
                InspectionReadiness::Unverified,
                InspectionReadiness::Unverified,
                type_code,
            )
        }
        _ => (
            InspectionIssueCode::InvalidContainer,
            InspectionReadiness::Invalid,
            InspectionReadiness::Unverified,
            InspectionReadiness::Unverified,
            None,
        ),
    };
    report.container = container;
    report.architecture_support = architecture;
    report.structural_binding = structural;
    report.model_loadability = if missing_media_projector {
        InspectionReadiness::Missing
    } else if architecture == InspectionReadiness::Unsupported {
        InspectionReadiness::Unsupported
    } else {
        InspectionReadiness::Invalid
    };
    report.requested_load = report.model_loadability;
    if missing_media_projector {
        report.multimodal = InspectionReadiness::Missing;
        report.requirements.push(InspectionRequirement {
            code: InspectionIssueCode::MissingMediaProjector,
            readiness: InspectionReadiness::Missing,
            detail: detail.clone(),
            path: None,
        });
    }
    report.issues.push(InspectionIssue {
        code,
        severity: InspectionSeverity::Error,
        detail,
        path: Some(path.to_path_buf()),
        metadata_key: None,
        tensor_name: None,
        tensor_type_code: type_code,
    });
}

/// Severity attached to an inspection issue.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InspectionSeverity {
    /// Prevents the stated readiness or requested preparation route.
    Error,
    /// Limits a capability without preventing selected preparation.
    Warning,
    /// Actionable context that is neither rejection nor warning.
    Info,
}

/// Stable machine-readable inspection issue category.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum InspectionIssueCode {
    /// Artifact path or container structure is invalid.
    InvalidContainer,
    /// Configuration or architecture metadata is invalid.
    InvalidConfiguration,
    /// Architecture dispatch has no backend implementation.
    UnsupportedArchitecture,
    /// A referenced checkpoint shard is absent or contradictory.
    MissingCheckpointShard,
    /// A tensor storage encoding cannot be consumed.
    UnsupportedTensorEncoding,
    /// An architecture-required tensor is absent.
    MissingRequiredTensor,
    /// Tensor aliases or layouts conflict after translation.
    ConflictingTensorLayout,
    /// A catalog tensor has the wrong rank or dimensions.
    TensorShapeMismatch,
    /// Packed quantization metadata and companion tensors disagree.
    QuantizationCompanionMismatch,
    /// Configured layer, attention, or expert geometry is invalid.
    InvalidLayerOrExpertCount,
    /// No usable tokenizer is available.
    MissingTokenizer,
    /// No checkpoint or sidecar chat template is available.
    MissingChatTemplate,
    /// A required multimodal projector is absent or ambiguous.
    MissingMediaProjector,
    /// A media processor or its build feature is unavailable.
    MissingProcessor,
    /// Requested on-load quantization is incompatible.
    UnsupportedQuantizationRequest,
    /// Requested weight-residency route is incompatible.
    UnsupportedResidencyPolicy,
    /// Requested parallel topology cannot use this loader.
    UnsupportedParallelTopology,
    /// No fail-closed semantic streaming protocol was recognized.
    UnsupportedSemanticProtocol,
    /// No fail-closed native-tool protocol was recognized.
    UnsupportedToolProtocol,
    /// EOS metadata is absent.
    MissingEosMetadata,
    /// Exact binding requires preparation-time module validation.
    ValidationUnavailableUntilLoad,
    /// Request data or runtime state still needs validation.
    RequestSpecificValidation,
    /// An ordinary local I/O operation failed.
    Io,
}

/// One structured diagnostic produced by inspection.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct InspectionIssue {
    /// Stable category for routing and UI behavior.
    pub code: InspectionIssueCode,
    /// Diagnostic severity.
    pub severity: InspectionSeverity,
    /// Human-readable actionable detail.
    pub detail: String,
    /// Relevant artifact or sidecar path.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<PathBuf>,
    /// Relevant metadata key.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata_key: Option<String>,
    /// Relevant logical or physical tensor name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tensor_name: Option<String>,
    /// Relevant numeric tensor type code.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tensor_type_code: Option<u32>,
}

/// One tensor storage encoding observed in artifact headers.
#[derive(Debug, Clone, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
pub struct ArtifactTensorEncoding {
    /// Stable textual representation.
    pub name: String,
    /// GGML type code for GGUF encodings.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ggml_type_code: Option<u32>,
}

/// Input modality advertised by the resolved architecture.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
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
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct InspectionRequirement {
    /// Machine-readable issue category.
    pub code: InspectionIssueCode,
    /// Current readiness of the requirement.
    pub readiness: InspectionReadiness,
    /// Human-readable explanation.
    pub detail: String,
    /// Expected or selected path.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path: Option<PathBuf>,
}

/// Structured pre-preparation compatibility report for a local artifact.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct ModelInspectionReport {
    /// Submitted local artifact path.
    pub path: PathBuf,
    /// Detected artifact container.
    pub artifact_format: ArtifactFormat,
    /// Resolved high-level model family.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_family: Option<String>,
    /// Submitted model type or architecture value.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub architecture: Option<String>,
    /// GGUF versions observed across validated shards.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gguf_versions: Option<Vec<u32>>,
    /// Number of checkpoint payload shards.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub checkpoint_shards: Option<usize>,
    /// Number of cataloged logical tensors.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tensor_count: Option<usize>,
    /// Header-only resource accounting.
    pub resources: ModelResourceProfile,
    /// Distinct storage encodings observed in headers.
    pub tensor_encodings: Vec<ArtifactTensorEncoding>,
    /// Expected input modalities.
    pub expected_modalities: Vec<ArtifactModality>,
    /// Container/config/header validity.
    pub container: InspectionReadiness,
    /// Whether architecture dispatch selects a backend implementation.
    pub architecture_support: InspectionReadiness,
    /// Exact header/catalog binding to that implementation.
    pub structural_binding: InspectionReadiness,
    /// Model preparation readiness independent of text sidecars.
    pub model_loadability: InspectionReadiness,
    /// Compatibility with backend preparation options.
    pub requested_load: InspectionReadiness,
    /// Exact portable preparation admission selected from immutable cold facts.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub preparation_admission: Option<PreparationAdmission>,
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
    /// Processor/projector readiness for non-text modalities.
    pub multimodal: InspectionReadiness,
    /// Discovered sidecar and request requirements.
    pub requirements: Vec<InspectionRequirement>,
    /// Structured rejection reasons, warnings, and limitations.
    pub issues: Vec<InspectionIssue>,
}

impl ModelInspectionReport {
    /// Creates an initially unverified report for backend-specific enrichment.
    pub fn unverified(path: &Path, artifact_format: ArtifactFormat) -> Self {
        Self {
            path: path.to_path_buf(),
            artifact_format,
            model_family: None,
            architecture: None,
            gguf_versions: None,
            checkpoint_shards: None,
            tensor_count: None,
            resources: ModelResourceProfile::unmeasured(path.to_path_buf(), artifact_format),
            tensor_encodings: Vec::new(),
            expected_modalities: Vec::new(),
            container: InspectionReadiness::Unverified,
            architecture_support: InspectionReadiness::Unverified,
            structural_binding: InspectionReadiness::Unverified,
            model_loadability: InspectionReadiness::Unverified,
            requested_load: InspectionReadiness::Unverified,
            preparation_admission: None,
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

    /// Returns whether artifact and requested backend policy passed preflight.
    pub fn is_loadable(&self) -> bool {
        self.container == InspectionReadiness::Ready
            && self.architecture_support == InspectionReadiness::Ready
            && self.structural_binding == InspectionReadiness::Ready
            && self.model_loadability == InspectionReadiness::Ready
            && self.requested_load == InspectionReadiness::Ready
            && !self
                .issues
                .iter()
                .any(|issue| issue.code == InspectionIssueCode::ValidationUnavailableUntilLoad)
    }

    /// Applies reusable header/catalog facts from one admitted portable inspection.
    pub fn record_artifact_inspection<P>(&mut self, inspection: &ArtifactInspection<P>) {
        self.path = inspection.path().to_owned();
        self.artifact_format = inspection.format();
        self.model_family = Some(inspection.configuration().family().to_owned());
        self.architecture = Some(inspection.configuration().effective_model_type().to_owned());
        self.container = InspectionReadiness::Ready;
        self.architecture_support = InspectionReadiness::Ready;
        self.structural_binding = InspectionReadiness::Ready;
        self.model_loadability = InspectionReadiness::Ready;
        self.tensor_count = Some(inspection.tensors().len());

        let (shards, encodings, stored_bytes, largest_bytes, gguf_versions) =
            if let Some(validated) = inspection.validated_gguf() {
                let checkpoint = validated.checkpoint();
                let encodings = checkpoint
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
                let mut total = Some(0_u64);
                let mut largest = 0_u64;
                for tensor in checkpoint.tensors() {
                    let bytes = tensor.descriptor().byte_len;
                    total = total.and_then(|value| value.checked_add(bytes));
                    largest = largest.max(bytes);
                }
                let versions = checkpoint
                    .shards()
                    .iter()
                    .map(|shard| shard.version())
                    .collect::<BTreeSet<_>>()
                    .into_iter()
                    .collect();
                (
                    checkpoint.shards().len(),
                    encodings,
                    total,
                    largest,
                    Some(versions),
                )
            } else {
                let mut shards = BTreeSet::new();
                let mut encodings = BTreeSet::new();
                let mut total = Some(0_u64);
                let mut largest = 0_u64;
                for tensor in inspection.tensors().descriptors() {
                    encodings.insert(safetensors_encoding_name(&tensor.dtype));
                    if let Some(storage) = &tensor.storage {
                        shards.insert(storage.member.clone());
                        total = total.and_then(|value| value.checked_add(storage.length));
                        largest = largest.max(storage.length);
                    } else {
                        total = None;
                    }
                }
                (
                    shards.len(),
                    encodings
                        .into_iter()
                        .map(|name| ArtifactTensorEncoding {
                            name,
                            ggml_type_code: None,
                        })
                        .collect(),
                    total,
                    largest,
                    None,
                )
            };
        self.checkpoint_shards = Some(shards);
        self.tensor_encodings = encodings;
        self.gguf_versions = gguf_versions;
        self.resources.model_family = self.model_family.clone();
        self.resources.architecture = self.architecture.clone();
        self.resources.tensor_count = self.tensor_count;
        self.resources.checkpoint_shards = self.checkpoint_shards;
        match stored_bytes {
            Some(total) => {
                self.resources.stored_tensor_bytes =
                    Observed::exact(total, "authoritative portable tensor catalog");
                self.resources.largest_stored_tensor_bytes =
                    Observed::exact(largest_bytes, "authoritative portable tensor catalog");
            }
            None => {
                self.resources.stored_tensor_bytes =
                    Observed::unavailable("portable payload-byte catalog was incomplete");
                self.resources.largest_stored_tensor_bytes =
                    Observed::unavailable("portable payload-byte catalog was incomplete");
            }
        }
    }

    /// Records the exact successful cold admission used by later realization.
    pub fn record_preparation_admission(&mut self, admission: PreparationAdmission) {
        self.preparation_admission = Some(admission);
        self.requested_load = InspectionReadiness::Ready;
    }

    /// Records architecture-owned modality and embedded-drafting facts.
    pub fn record_architecture_capabilities(
        &mut self,
        modalities: InputModalities,
        embedded_draft_layers: Option<usize>,
    ) {
        self.expected_modalities = artifact_modalities(modalities);
        self.resources.embedded_draft_layers = embedded_draft_layers.map_or_else(
            || Observed::unsupported("artifact convention does not expose embedded drafting"),
            |layers| Observed::exact(layers, "normalized architecture configuration"),
        );
    }

    /// Records one stable portable admission rejection.
    pub fn reject_preparation_admission(&mut self, error: PreparationAdmissionError) {
        use PreparationAdmissionError as Admission;
        self.preparation_admission = None;
        self.requested_load = InspectionReadiness::Unsupported;
        let code = match error {
            Admission::UnsupportedQuantization(_) => {
                InspectionIssueCode::UnsupportedQuantizationRequest
            }
            Admission::UnsupportedResidency(_) | Admission::ArchitectureParameterBanks => {
                InspectionIssueCode::UnsupportedResidencyPolicy
            }
            Admission::ArchitectureParallelAxis(_) | Admission::BackendParallelAxis(_) => {
                InspectionIssueCode::UnsupportedParallelTopology
            }
            Admission::ArchitectureInputModality(_) | Admission::BackendInputModality(_) => {
                InspectionIssueCode::MissingProcessor
            }
            _ => InspectionIssueCode::UnsupportedArchitecture,
        };
        self.issue(
            code,
            InspectionSeverity::Error,
            error.to_string(),
            Some(self.path.clone()),
        );
    }

    /// Adds a structured issue with an optional artifact path.
    pub fn issue(
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

fn safetensors_encoding_name(dtype: &crate::checkpoint::TensorDtype) -> String {
    use crate::checkpoint::TensorDtype;
    match dtype {
        TensorDtype::Bf16 => "BF16".into(),
        TensorDtype::Complex64 => "C64".into(),
        TensorDtype::Encoded(name) if name == "F8_E4M3" => "F8E4M3".into(),
        TensorDtype::Encoded(name) if name == "F4" => "F4".into(),
        TensorDtype::Encoded(name) if name == "F8_E8M0" => "F8E8M0".into(),
        TensorDtype::Encoded(name) if name == "F8_E5M2" => "F8E5M2".into(),
        TensorDtype::Encoded(name) => format!("Other({name:?})"),
        dtype => format!("{dtype:?}"),
    }
}

/// Converts portable modality flags into stable inspection report values.
pub fn artifact_modalities(modalities: InputModalities) -> Vec<ArtifactModality> {
    [
        (modalities.text, ArtifactModality::Text),
        (modalities.image, ArtifactModality::Image),
        (modalities.video, ArtifactModality::Video),
        (modalities.audio, ArtifactModality::Audio),
    ]
    .into_iter()
    .filter_map(|(enabled, modality)| enabled.then_some(modality))
    .collect()
}

/// Portable availability of optional host-media processors.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct MediaFeatureAvailability {
    /// Image and video host processing is linked.
    pub image: bool,
    /// Audio host processing is linked.
    pub audio: bool,
}

/// Architecture-owned requirement for a GGUF media projector.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum MediaProjectorRequirement {
    /// No projector applies.
    NotApplicable,
    /// Text remains available when the projector is absent.
    Optional,
    /// Model loading requires a projector.
    Required,
}

/// Computes portable host-feature readiness for declared modalities.
pub fn media_feature_readiness(
    expected: &[ArtifactModality],
    availability: MediaFeatureAvailability,
) -> InspectionReadiness {
    if expected == [ArtifactModality::Text] {
        return InspectionReadiness::NotApplicable;
    }
    if expected.iter().all(|modality| match modality {
        ArtifactModality::Text => true,
        ArtifactModality::Image | ArtifactModality::Video => availability.image,
        ArtifactModality::Audio => availability.audio,
    }) {
        InspectionReadiness::Ready
    } else {
        InspectionReadiness::Unsupported
    }
}

/// Records portable processor and host-feature readiness in one report.
pub fn record_processor_inspection(
    report: &mut ModelInspectionReport,
    has_processor: bool,
    availability: MediaFeatureAvailability,
) {
    let feature_readiness = media_feature_readiness(&report.expected_modalities, availability);
    if feature_readiness == InspectionReadiness::NotApplicable {
        report.multimodal = feature_readiness;
        return;
    }
    if has_processor {
        report.multimodal = feature_readiness;
        report.requirements.push(InspectionRequirement {
            code: InspectionIssueCode::MissingProcessor,
            readiness: feature_readiness,
            detail: if feature_readiness == InspectionReadiness::Ready {
                "authoritative processor plan and required host-media features are available".into()
            } else {
                "authoritative processor plan is available, but required host-media features are not enabled".into()
            },
            path: None,
        });
    } else {
        report.multimodal = InspectionReadiness::Missing;
        report.issue(
            InspectionIssueCode::MissingProcessor,
            InspectionSeverity::Warning,
            "authoritative architecture inspection admitted no media processor",
            Some(report.path.clone()),
        );
    }
}

/// Records a validated or absent GGUF projector and returns whether it was selected.
pub fn record_media_projector_inspection(
    report: &mut ModelInspectionReport,
    artifact_path: &Path,
    requirement: MediaProjectorRequirement,
    projector: Option<PathBuf>,
) -> bool {
    match (requirement, projector) {
        (MediaProjectorRequirement::NotApplicable, _) => {
            report.multimodal = InspectionReadiness::NotApplicable;
            false
        }
        (_, Some(path)) => {
            report.requirements.push(InspectionRequirement {
                code: InspectionIssueCode::MissingMediaProjector,
                readiness: InspectionReadiness::Ready,
                detail: "portable admission validated the architecture-declared media projector"
                    .into(),
                path: Some(path),
            });
            true
        }
        (MediaProjectorRequirement::Optional, None) => {
            report.multimodal = InspectionReadiness::Missing;
            report.requirements.push(InspectionRequirement {
                code: InspectionIssueCode::MissingMediaProjector,
                readiness: InspectionReadiness::Missing,
                detail: "text loading is available, but media input requires an architecture-declared sibling projector GGUF".into(),
                path: None,
            });
            report.issue(
                InspectionIssueCode::MissingMediaProjector,
                InspectionSeverity::Warning,
                "no sibling media projector was admitted; text loading remains available",
                Some(artifact_path.to_path_buf()),
            );
            false
        }
        (MediaProjectorRequirement::Required, None) => {
            report.multimodal = InspectionReadiness::Missing;
            report.model_loadability = InspectionReadiness::Invalid;
            report.requested_load = InspectionReadiness::Invalid;
            report.requirements.push(InspectionRequirement {
                code: InspectionIssueCode::MissingMediaProjector,
                readiness: InspectionReadiness::Missing,
                detail:
                    "architecture preparation requires a validated sibling media projector GGUF"
                        .into(),
                path: None,
            });
            report.issue(
                InspectionIssueCode::MissingMediaProjector,
                InspectionSeverity::Error,
                "portable admission omitted an architecture-required media projector",
                Some(artifact_path.to_path_buf()),
            );
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn report_schema_round_trips_without_a_backend() {
        let report =
            ModelInspectionReport::unverified(Path::new("model.gguf"), ArtifactFormat::Gguf);
        let json = serde_json::to_string(&report).unwrap();
        let decoded: ModelInspectionReport = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, report);
    }

    #[test]
    fn neutral_admission_rejection_has_a_stable_report_code() {
        let mut report =
            ModelInspectionReport::unverified(Path::new("model"), ArtifactFormat::SafeTensors);
        report.reject_preparation_admission(PreparationAdmissionError::BackendParallelAxis(
            crate::ParallelAxis::Pipeline,
        ));
        assert_eq!(report.requested_load, InspectionReadiness::Unsupported);
        assert!(report.preparation_admission.is_none());
        assert_eq!(
            report.issues[0].code,
            InspectionIssueCode::UnsupportedParallelTopology
        );
    }
}
