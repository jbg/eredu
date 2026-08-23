//! Portable model-artifact inspection results.

use crate::{ArtifactFormat, ModelResourceProfile};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

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
    #[serde(skip_serializing_if = "Option::is_none")]
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
}
