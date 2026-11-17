//! Explicit backend selection for metadata-only application inspection.

pub use eredu_core::residency::{
    CacheEvictionPolicy, OffloadConfig, OffloadError, ParameterConversionRetentionPolicy,
};
pub use eredu_core::{
    BackendId, CapabilityError, CompletionCancellationMode, DevicePlan, ExecutionPlanError,
    HardwareBackendProfile, HardwareDeviceKind, HardwareDeviceProfile, HardwareMemorySemantics,
    HardwareProfile, InputTokenCount, InspectionIssue, InspectionIssueCode, InspectionReadiness,
    InspectionSeverity, ModelInspectionReport, ObservationKind, Observed, ParallelRankTopology,
    ParallelTopology, QuantizationRequest, SessionCapabilities,
};
pub use eredu_runtime::{
    CacheResidencyPolicy, CommunicationCompletionPolicy, DenseDiskStreamLoadOptions,
    DraftingLoadRequest, LayerWeightResidency, LayerwiseLoadOptions, NormalizedLoadRequest,
    NormalizedLoadRequestError, OrdinaryWeightResidency, PagedCacheOptions, ParallelLoadRequest,
    ParameterBankLoadOptions, PipelineActivationDtype, PipelineWireContract, WeightResidency,
};

pub use eredu_architectures::inspect_model_metadata;
