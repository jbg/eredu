//! Explicit backend selection for metadata-only application inspection.

pub use eredu_core::residency::{
    CacheEvictionPolicy, OffloadConfig, OffloadError, ParameterConversionRetentionPolicy,
};
pub use eredu_core::{
    BackendId, CapabilityError, CompletionCancellationMode, DevicePlan, ExecutionPlanError,
    HardwareBackendProfile, HardwareDeviceProfile, HardwareMemorySemantics, HardwareProfile,
    InputTokenCount, InspectionIssue, InspectionIssueCode, InspectionReadiness, InspectionSeverity,
    ModelInspectionReport, ObservationKind, Observed, ParallelRankTopology, ParallelTopology,
    QuantizationRequest, SessionCapabilities,
};
pub use eredu_runtime::{
    CacheResidencyPolicy, CommunicationCompletionPolicy, DenseDiskStreamLoadOptions,
    DraftingLoadRequest, LayerWeightResidency, LayerwiseLoadOptions, NormalizedLoadRequest,
    NormalizedLoadRequestError, OrdinaryWeightResidency, PagedCacheOptions, ParallelLoadRequest,
    ParameterBankLoadOptions, PipelineActivationDtype, PipelineWireContract, WeightResidency,
};

use super::{ArtifactMetadata, ModelInspectionOutcome};

/// Explicit backend and portable policy for supplied-metadata inspection.
/// There is no default backend or implicit hardware discovery.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct MetadataInspectionOptions {
    backend: BackendId,
    load_request: NormalizedLoadRequest,
}

impl MetadataInspectionOptions {
    /// Selects a backend explicitly, with the default portable loading policy.
    pub fn new(backend: BackendId) -> Self {
        Self {
            backend,
            load_request: NormalizedLoadRequest::default(),
        }
    }

    /// Sets quantization, residency, speculation, topology and session policy.
    pub fn with_load_request(mut self, request: NormalizedLoadRequest) -> Self {
        self.load_request = request;
        self
    }

    /// Backend whose compiled capabilities inspection evaluates.
    pub fn backend(&self) -> &BackendId {
        &self.backend
    }

    /// Portable loading policy evaluated against checkpoint and backend facts.
    pub fn load_request(&self) -> &NormalizedLoadRequest {
        &self.load_request
    }
}

/// Failure to resolve an explicitly requested inspection backend.
/// Artifact and policy incompatibilities are reported in `ModelInspectionOutcome`.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
pub enum MetadataInspectionError {
    /// No adapter with this identifier is known to the facade.
    #[error("unknown inspection backend {backend}")]
    UnknownBackend {
        /// Requested backend identifier.
        backend: BackendId,
    },
    /// The named adapter is known but its Cargo feature is disabled.
    #[error("inspection backend {backend} requires the eredu Cargo feature {feature:?}")]
    BackendNotCompiled {
        /// Requested backend identifier.
        backend: BackendId,
        /// Cargo feature enabling this adapter.
        feature: &'static str,
    },
}

/// Inspects supplied headers using the explicitly requested backend's cold facts.
///
/// No checkpoint files, native devices or hardware discovery are involved. The
/// facade resolves the adapter, and the shared architecture driver validates and
/// selects execution. Successful inspection supports forecasting but grants no
/// payload-loading authority. Target hardware budgets are supplied separately
/// through [`super::GenerationMemoryOptions`].
pub fn inspect_model_metadata(
    metadata: &ArtifactMetadata,
    options: &MetadataInspectionOptions,
) -> Result<ModelInspectionOutcome, MetadataInspectionError> {
    match options.backend.as_str() {
        "mlx" => {
            #[cfg(feature = "mlx")]
            {
                Ok(eredu_architectures::inspect_model_metadata(
                    metadata,
                    &options.load_request,
                    &eredu_backend_mlx::inspection_mechanisms(),
                    eredu_core::MediaFeatureAvailability {
                        image: cfg!(feature = "image"),
                        audio: cfg!(feature = "audio"),
                    },
                ))
            }
            #[cfg(not(feature = "mlx"))]
            {
                let _ = metadata;
                Err(MetadataInspectionError::BackendNotCompiled {
                    backend: options.backend.clone(),
                    feature: "mlx",
                })
            }
        }
        _ => Err(MetadataInspectionError::UnknownBackend {
            backend: options.backend.clone(),
        }),
    }
}
