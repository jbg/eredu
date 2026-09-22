//! Request-bound reservations for shared inference working memory.
mod dependency_memory;
pub use dependency_memory::DependencyMemoryPolicy;
mod dependency_estimate;
pub use dependency_estimate::{DependencyEstimateAdmissionError, DependencyEstimateFunding};
mod numerical_source;
mod realtime_frame;
pub use numerical_source::{
    NumericalSourceAdmissionError, NumericalSourceRequirements, OriginalNumericalBudgetCustody,
    OriginalNumericalLifetime, OriginalNumericalNative, OriginalNumericalSource,
};
pub use realtime_frame::{
    OriginalRealtimeBudgetCustody, OriginalRealtimeFrame, OriginalRealtimeNative,
    RealtimeFrameAdmissionError, RealtimeFrameRequirements, RealtimeNativeRequirements,
};
mod control_mutex;
mod operation_custody;
pub use operation_custody::OriginalOperationMetadataCustody;
mod fixed_baseline;
mod original_prepared_input;
#[cfg(test)]
mod original_request;
mod qualified_storage;
mod reservation_metadata;
mod speculative;
mod storage_metadata;
mod thread_startup;
mod transaction_buffers;
mod workspace_planning;
use control_mutex::ControlMutex;
pub use original_prepared_input::{OriginalPreparedHostInput, OriginalPreparedHostInputError};
pub(crate) use qualified_storage::shared_bytes as qualified_shared_bytes;
pub use reservation_metadata::WorkspaceReservationMetadataError;
pub use speculative::{
    OriginalEmbeddedCaptureLineage, OriginalEmbeddedSpeculativeRole,
    OriginalEmbeddedSpeculativeSource, OriginalEmbeddedSpeculativeStartup,
    OriginalExternalSpeculativeRole, OriginalExternalSpeculativeSource,
    OriginalExternalSpeculativeStartup, OriginalModelCaptureLineage,
    OriginalSpeculativeBudgetCustody, OriginalSpeculativeNumericalBudgetCustody,
    OriginalSpeculativeNumericalPhase, OriginalSpeculativePrefillSpan,
    OriginalSpeculativeRegisteredSource, OriginalSpeculativeRequest, OriginalSpeculativeRole,
    OriginalSpeculativeSourceIdentity, OriginalSpeculativeStartup, SpeculativeContinuationError,
    SpeculativeHostSourceSpans, SpeculativeInvocationRequirements,
    SpeculativeNumericalAdmissionError, SpeculativeNumericalRequirements,
    SpeculativeNumericalSource, SpeculativePrefillScheduleAuthority, SpeculativeRequestError,
};
pub use storage::PreparedStorageHostSlots;
pub use storage_metadata::StorageMetadataFunding;
pub use thread_startup::{HostThreadStartup, HostThreadStartupPlan};
pub use workspace_planning::{PreparedConstructionMetadata, SessionResetPreparationFunding};
mod original_prepared_native_input;
pub(crate) use original_prepared_native_input::PreparedInputHostCustody;
pub use original_prepared_native_input::{
    OriginalPreparedInputCustody, OriginalPreparedInputMaterialization,
    OriginalPreparedInputMaterializationError, PreparedNativeInputCompiler,
    RetiredPreparedInputMaterializationError,
};

pub use funding::{
    CaptureSourceSegment, PreparedPrefillChunkRetention, SettledCaptureSourceParcel,
    SettledPrefillChunkRetention,
};

mod original_stop_source;
pub use original_stop_source::{
    OriginalStopSource, OriginalStopSourceBackend, OriginalStopSourceError,
};

mod gguf_catalog;
mod gguf_composite;
pub use gguf_composite::OriginalGgufCompositeError;
mod source_erasure;
pub use source_erasure::OriginalRetainedSourceError;
mod gguf_source;
mod memory_tensor_source;
pub use memory_tensor_source::{MemoryTensorBufferQuote, OriginalMemoryTensorError};
mod safetensors_header_source;
pub use safetensors_header_source::{SafetensorsHeaderPolicyError, SafetensorsHeaderQuote};
mod safetensors_source;
pub use safetensors_source::{OriginalArtifactInspectionError, OriginalSafetensorsSourceError};
mod prepared_safetensors_source;
pub use gguf_catalog::{OriginalGgufCatalog, OriginalGgufCatalogError};
pub use gguf_source::{GgufSourceStorageKey, OriginalGgufSourceError};
pub use prepared_safetensors_source::OriginalPreparedSafetensorsError;

mod loaded_decode_source;
pub use loaded_decode_source::{
    LoadedDecodeSource, LoadedDecodeSourceBackend, LoadedDecodeSourceError,
};

mod capture_run;
pub(crate) use capture_run::InterventionEvidenceFrame;
pub use capture_run::{
    AutoregressiveCaptureFrameHostPlan, AutoregressiveCaptureHostPlan, CaptureCandidateClaim,
    CaptureCandidateFailure, CaptureCandidateHostPlan, CaptureInterventionClaim,
    CaptureInterventionEvidenceClaim, CaptureInterventionEvidenceKind, CapturePrefillFragmentClaim,
    CapturePrefillFragmentTransfer, CapturePrefillFragmentWriter, CapturePrefillHostError,
    CapturePrefillSourceBootstrap, CaptureRunHostError, CaptureRunHostPlan, CaptureStepClaim,
    CaptureTensorClaim, CaptureTokenScoreClaim, CaptureTokenScoreFailure,
    CaptureTokenScoreHostPlan, ClaimedCaptureCandidates, ClaimedCaptureTensor,
    ClaimedCaptureTokenScores, ClaimedIntervention, ClaimedInterventionEvidence,
    EmbeddedCaptureHostPlan, EmbeddedCapturePreparationError, FundedAutoregressiveCaptureBank,
    InterventionEvidenceReceipt, InterventionPrefillCursor, InterventionPrefillFragment,
    ModelCapturePreparationCause, ModelCapturePreparationError, PartitionCaptureTensorDecodeError,
    PartitionCaptureTensorDeliveryError, PartitionCaptureTensorReceipt,
    PreparedAutoregressiveCapture, PreparedCaptureRun, PreparedEmbeddedCapture,
    PreparedPartitionTensorDelivery, RoutedInterventionBatch, RoutedInterventionCursor,
    ScheduledCaptureCandidates, ScheduledCaptureCandidatesTransfer, ScheduledCaptureStep,
    ScheduledCaptureStepFinishError, ScheduledCaptureTensor, ScheduledCaptureTensorFailure,
    ScheduledCaptureTensorFinishError, ScheduledCaptureTensorTransfer,
    ScheduledCaptureTensorTransferFinishError, ScheduledCaptureTokenScores,
    ScheduledCaptureTokenScoresTransfer, SpeculativeCaptureHostPlan,
    SpeculativeCapturePreparationError,
};
pub(crate) use capture_run::{CaptureRunLedger, CaptureRunLedgerGuard};

mod text_preparation;
pub use text_preparation::{
    InferencePreparationStage, InferencePromptCompletion, InferenceSamplerCompletion,
    InferenceTextPreparation, InferenceTextStep, InferenceTextStepReceipt,
    PendingSamplingExtension, PendingTextBranchExchange,
};
mod capture_step;
pub use capture_step::{
    CaptureStepError, CaptureStepFinishError, CaptureStepHostPlan, PreparedCaptureDelivery,
    PreparedCaptureStep,
};
pub use capture_step::{PendingCaptureDelivery, PendingCaptureDeliveryError};
mod capture_tensor;
pub use capture_tensor::{
    CaptureTensorConstructionError, CaptureTensorFinishError, CaptureTensorHostPlan,
    CaptureTensorLimits, CaptureTensorTransferFinishError, PreparedCaptureTensor,
    PreparedCaptureTensorTransfer,
};
mod pending_input;
pub use pending_input::{
    FundedPendingTokenInput, InferencePendingPromptCompletion, PendingTokenInputConstructionError,
    PendingTokenInputHostPlan, PreparedPendingTokenInputHost,
};
mod storage;
pub use storage::bounded_pin::{
    BoundedPinAttempt, BoundedPinError, BoundedRegisteredStorage, FailedBoundedPinAttempt,
    OriginalPrefillStoragePinSlots, PreparedPrefillStoragePinPlan,
};
pub use storage::bounded_publication::{
    BoundedPublicationAttempt, BoundedPublicationError, BoundedPublishedAllocation,
    OriginalPrefillStoragePublicationSlots, PreparedPrefillStoragePublicationPlan,
};
pub use storage::{
    ExistingStoragePinLayout, OriginalStorageSourcesError, OriginalStorageSourcesLayout,
    PendingStorageAllocation, PreparedStoragePublication, RetainedOriginalStorageSources,
    StorageAllocation, StoragePublicationLayout, StorageRegistrationValues, StorageRegistrations,
    WorkingMemoryStorage,
};
pub use storage::{
    NumericalStoragePublicationPlan, NumericalStorageRegistration,
    PreparedNumericalStoragePublication, PreparedWorkspaceCopyPublication,
    WorkspaceCopyPublicationPlan,
};
mod funding;
pub use funding::{
    PreparedWorkingMemoryFundingScope, WorkingMemoryAllocationFunding,
    WorkingMemoryCapacityHandoff, WorkingMemoryFundingRun, WorkingMemoryFundingScope,
    WorkingMemorySamplerScope,
};
mod sampler_copy;
pub use sampler_copy::{
    BorrowedFundedSampler, FundedSamplerCopy, PreparedRunSample, RunOwnedTextSampler,
    SamplerCopyAdmissionError, SamplerCopyLimits, SamplerResumePlan,
};
mod workspace_copy;
pub use workspace_copy::{
    AdmittedWorkspaceCopy, CompletedWorkspaceSourceAccount, CompletedWorkspaceSourceLayout,
    CompletedWorkspaceStorage, CompletedWorkspaceStorageLayout, OriginalCompletedWorkspaceCopy,
    OriginalCompletedWorkspaceSource, OriginalNumericalWorkspaceCopy,
    RegisteredPreparedWorkspaceCopy, RegisteredWorkspaceCopy, WorkspaceCopyAccountLayout,
    WorkspaceCopyAdmissionError, WorkspaceCopyCustody, WorkspaceCopyLimits, WorkspaceCopyRetention,
};
mod saved_source;
pub use saved_source::RegisteredSavedSamplingSource;
mod sampling_copy;
pub use sampling_copy::{
    RegisteredSamplingCopy, RegisteredSamplingCopyWithSource, SamplingCopyAdmissionError,
};
mod decoder_copy;
pub use decoder_copy::{
    DecoderCopyAdmissionError, DecoderHostPreparationError, DecoderSlotsFinishError,
    DenseDecoderFinishError, DenseDecoderHandoffError, FundedDecoderSlots, FundedDenseHostSlots,
    HostSlotStorageKey, InitializedDecoderSlots, InitializedDenseDecoderSlots,
    RegisteredDecoderHostCopy, RegisteredDenseDecoderInitialization, RegisteredTextComponentsCopy,
};
pub use decoder_copy::{
    DecoderGroupFinishError, DecoderGroupHandoffError, FundedDenseDecoderGroupChild,
    InitializedDecoderGroupChild, InitializedDecoderTableGroup, InitializedDenseDecoderGroupChild,
    InitializedDenseDecoderTableGroup, RegisteredDecoderTableGroup,
    RegisteredDenseDecoderTableGroup, RegisteredTextComponentsGroupCopy,
};
mod controller;
pub use controller::{
    ControllerStorageContract, ControllerStorageError, ControllerWorkspaceContribution,
    ControllerWorkspaceEstimate, ControllerWorkspaceMetadataError, PreparedControllerBinding,
    PreparedControllerBindingError, RegisteredControllerStorage,
};
mod preparation;
pub use preparation::{
    TextPromptWorkspaceReport, quote_original_token_prompt_workspace, quote_text_prompt_workspace,
};
mod parameter_sources;
pub use parameter_sources::{
    CountedWorkspaceParameterSource, WorkspaceParameterCounts, WorkspaceParameterDestinations,
    WorkspaceParameterLifetime, WorkspaceParameterOwner, WorkspaceParameterProjection,
    WorkspaceParameterRecord, WorkspaceParameterRequestRecord, WorkspaceParameterRootRecord,
    WorkspaceParameterRow, WorkspaceParameterRows, WorkspaceParameterSourceError,
    WorkspaceParameterSourceLoan, WorkspaceParameterTable, WorkspaceParameterTypeLayout,
    WorkspaceParameterUnit, WorkspaceParameterUnitRecord, workspace_parameter_control_layouts,
};
mod layerwise_parameters;
pub use layerwise_parameters::{
    PreparedWorkspaceBindingCause, bind_prepared_workspace_parameters, bind_workspace_parameters,
};
mod layerwise_window;
pub use layerwise_window::{
    LayerwiseWindowError, completed_layerwise_window_bytes, quote_completed_layerwise_window,
};
mod sampling;
pub use sampling::{
    SamplingWorkspaceInputPlan, SamplingWorkspaceObserver, SamplingWorkspacePhase,
    SamplingWorkspaceReport, WorkspaceSamplingBackend, WorkspaceSamplingInput,
    WorkspaceSamplingRandomState, WorkspaceSamplingSource, quote_sampling_workspace,
    quote_sampling_workspace_with_observer,
};
mod trace;
pub use trace::with_equation_workspace;
mod inference;
pub use inference::{
    InferenceObservationError, InferenceResidualWorkspace, InferenceSpanWorkspacePlan,
    InferenceSpanWorkspaceRecord, InferenceWorkspaceError, InferenceWorkspaceObserver,
    InferenceWorkspaceReport, InferenceWorkspaceSpan, SamplingWorkspacePlanCollector,
    quote_inference_workspace, quote_inference_workspace_with_context,
    quote_inference_workspace_with_report_owner,
};
mod residual;
pub use residual::{
    AdmittedCaptureContinuation, AdmittedPrefillCapture, AggregateGenerationDecoderInput,
    CopyPreparationInferenceQuote, FailedCapturePlanPublication, GraphMetadataFacts,
    HostDestinationCause, HostDestinationFacts, HostSourceConstructionFacts,
    HostSourceConstructionProgram, IncompleteWorkspace, IncrementalInferenceQuote,
    InferenceSpanWorkspace, LoadedGenerationDecoderInput, NativeStorageError,
    NativeStorageObservation, NativeStorageRegistration, NativeStorageSelection,
    OriginalGenerationDecoderInput, OriginalGenerationDecoderSource,
    OriginalGenerationSequenceBank, OriginalGraphMetadata, OriginalHostDestinationBank,
    OriginalHostMetadataCustody, OriginalHostMetadataVec, OriginalHostSourceBank,
    OriginalHostSourceConstruction, OriginalHostSourceError, OriginalHostSourceFailure,
    OriginalHostSourceFailureCause, OriginalHostSourceProgramBanks, OriginalHostSourceProgramError,
    OriginalHostSourceReceipt, OriginalHostSourceRefusal, OriginalHostVec, OriginalHostVecError,
    OriginalNativeBudgetCustody, OriginalNativePublication, OriginalNativeStorageBank,
    OriginalNativeStorageMechanism, OriginalPredictionNativeCustody,
    OriginalPredictionRecoveryCustody, OriginalPredictionScopeRole, OriginalPrefillNativeCustody,
    OriginalPrefillRecoveryCustody, OriginalPrefillRootCustody,
    OriginalPrefillRootProjectionCustody, OriginalPrefillScopeRole,
    OriginalPreparationScopeCustody, OriginalSubmissionTracking, OriginalTextControlGuard,
    OriginalTextMetadataCustody, OriginalTextPredictionScopeSet, OriginalTextPredictionScopes,
    OriginalTextPrefillScopeSet, OriginalTextPrefillScopes, OriginalTextPreparationScopes,
    OriginalTextSamplingExtension, OriginalTokenInputBank, OriginalTokenInputFailure,
    OriginalTokenInputLayout, OwnedInferenceSpanWorkspace, OwnedPromptTokenIds,
    OwnedTextSpanWorkspace, PendingCapturePlanPublication, PreparedNativeStoragePlan,
    PreparedTextControlWorkspace, RegisteredInferenceSourceWitness,
    RegisteredPreparedWorkspaceStorage, RegisteredWorkspaceStorage,
    RegisteredWorkspaceStorageLayout, RegisteredWorkspaceStorageRow,
    ReservedInferenceSpanWorkspace, ReservedTextSpanWorkspace, ResidualInferenceQuote,
    ResidualQuoteError, SamplingExtensionQuote, SpanWorkspaceOwnerError, SubmissionTrackingFacts,
    TextHostControlFacts, TextPredictionScopeFacts, TextPrefillScopeFacts,
    TextPreparationScopeFacts, plan_prefill_incremental_with_capacity,
    plan_prefill_incremental_with_capacity_handoff, plan_prefill_residual_with_capacity,
};
pub use residual::{
    HostSourcePeakSelection, OriginalHostSourcePeakCapacity, OriginalHostSourcePending,
};
mod retention;
pub(crate) use retention::exchange_inference_state;
pub use retention::{
    InferenceRetention, InferenceStateAdmission, InferenceStateRetention, InferenceStateRevision,
};
mod communication;
mod submission;
pub use communication::{
    WorkspaceCommunicationGroup, WorkspaceCommunicationMetadata, WorkspaceCommunicationRoute,
    workspace_partition_communication,
};
pub use submission::WorkspaceCompletion;
mod paged_append;
pub use paged_append::{
    WorkspacePagedAppendState, WorkspacePagedBlock, WorkspacePagedGeometry,
    WorkspacePagedHostEntry, WorkspacePagedHostLoad, WorkspacePagedHostTrace,
};

mod state;
pub use state::{
    WorkspaceConcatLayerState, WorkspaceConcatStateFactory, WorkspacePagedLayerState,
    WorkspacePagedValues,
};
mod compressed;
pub use compressed::WorkspaceCompressedCache;
mod resident;
pub use resident::{
    WorkspaceResidentLayerState, WorkspaceResidentStateFactory, WorkspaceResidentValues,
    validate_workspace_state_realization,
};
mod pooling;
pub use pooling::{
    WorkspacePoolingLayerState, WorkspacePoolingStateFactory, WorkspacePoolingValues,
};

use eredu_core::{
    Admission, DomainMemoryCharge, DomainMemoryRequirements, EstimationCompleteness,
    InferenceGeometry, MemoryDomainError, MemoryDomainId, MemoryLimit, MemoryLimits,
    MemoryTopology,
};
use std::sync::{Arc, Mutex};
mod domains;
pub use domains::{MemoryDomainSnapshot, MemoryLedgerSnapshot, PlacementAllowanceBasis};
#[cfg(test)]
pub(crate) mod memory_fixture;

/// Failure while pricing and reserving bounded prompt execution.
#[derive(Debug, thiserror::Error)]
pub enum PrefillPlanningError {
    /// A structurally validated candidate lacks a numerical workspace bound.
    /// This explicit quote-stage gap permits trying a smaller chunk; the
    /// separate typed tracking-capacity refusal also permits that search.
    #[error("{0}")]
    IncompleteWorkspace(#[from] IncompleteWorkspace),
    /// Invalid geometry or overflowing architecture/native accounting.
    #[error("{0}")]
    Estimate(#[from] eredu_core::CapabilityError),
    /// Context or working-memory policy rejected the request.
    #[error("inference admission rejected: {0:?}")]
    Admission(eredu_core::AdmissionRejection),
    /// Shared managed capacity rejected reservation, or a selected constructor
    /// exceeds the configured tracking arena during cold quotation.
    #[error("{0}")]
    Reservation(#[from] WorkingMemoryError),
}

/// Prices the requested chunk, then each smaller chunk through one
/// position. No monotonicity of native kernel workspace is assumed. The quote
/// callback is cold: it must not allocate tensors, encode media or submit work.
/// A successful result already owns capacity against concurrent admissions.
pub fn plan_prefill(
    execution: &InferenceExecutionIdentity,
    pool: &MemoryLedger,
    capabilities: &eredu_core::ModelCapabilities,
    request: eredu_core::AdmissionRequest,
    geometry: InferenceGeometry,
    quote: impl FnMut(
        InferenceGeometry,
    ) -> Result<eredu_core::RuntimeStateEstimate, eredu_core::CapabilityError>,
) -> Result<(Admission, WorkingMemoryReservation), PrefillPlanningError> {
    plan_prefill_limited(
        execution,
        pool,
        capabilities,
        request,
        geometry,
        None,
        quote,
    )
}

/// Selects a chunk while enforcing the request's limits in every physical
/// domain. Existing storage and concurrent reservations count against each
/// limit. Completion and retained-state owners keep these constraints active;
/// later admissions cannot raise them without authenticated succession.
pub fn plan_prefill_with_capacity(
    execution: &InferenceExecutionIdentity,
    pool: &MemoryLedger,
    capabilities: &eredu_core::ModelCapabilities,
    request: eredu_core::AdmissionRequest,
    geometry: InferenceGeometry,
    capacity: MemoryLimits,
    quote: impl FnMut(
        InferenceGeometry,
    ) -> Result<eredu_core::RuntimeStateEstimate, eredu_core::CapabilityError>,
) -> Result<(Admission, WorkingMemoryReservation), PrefillPlanningError> {
    plan_prefill_limited(
        execution,
        pool,
        capabilities,
        request,
        geometry,
        Some(capacity),
        quote,
    )
}

fn plan_prefill_limited(
    execution: &InferenceExecutionIdentity,
    pool: &MemoryLedger,
    capabilities: &eredu_core::ModelCapabilities,
    request: eredu_core::AdmissionRequest,
    geometry: InferenceGeometry,
    capacity: Option<eredu_core::MemoryLimits>,
    mut quote: impl FnMut(
        InferenceGeometry,
    ) -> Result<eredu_core::RuntimeStateEstimate, eredu_core::CapabilityError>,
) -> Result<(Admission, WorkingMemoryReservation), PrefillPlanningError> {
    plan_prefill_candidates(
        capabilities,
        request.clone(),
        geometry,
        |geometry| quote(geometry).map_err(Into::into),
        |state, geometry| {
            if state
                .execution_workspace
                .as_ref()
                .map(|workspace| workspace.geometry)
                != Some(geometry)
            {
                return Err(WorkingMemoryError::IdentityMismatch.into());
            }
            match eredu_core::apply_admission_policy(capabilities, request.clone(), state)? {
                eredu_core::AdmissionResult::Admitted(admission) => {
                    let reservation =
                        pool.reserve_limited(execution, &admission, capacity.clone())?;
                    Ok((admission, reservation))
                }
                eredu_core::AdmissionResult::Rejected(rejection) => {
                    Err(PrefillPlanningError::Admission(rejection))
                }
            }
        },
    )
}

// Full and registered-root quotes share candidate enumeration and retry policy.
// Only the proof-specific admission closure determines the required byte bound.
fn plan_prefill_candidates<Q, T>(
    capabilities: &eredu_core::ModelCapabilities,
    request: eredu_core::AdmissionRequest,
    geometry: InferenceGeometry,
    mut quote: impl FnMut(InferenceGeometry) -> Result<Q, PrefillPlanningError>,
    mut admit: impl FnMut(Q, InferenceGeometry) -> Result<T, PrefillPlanningError>,
) -> Result<T, PrefillPlanningError> {
    geometry.validate()?;
    if geometry.batch_size != request.batch_size
        || geometry.cached_positions + geometry.input_positions != request.input.model_positions
        || geometry.max_output_tokens != request.max_output_tokens
    {
        return Err(WorkingMemoryError::IdentityMismatch.into());
    }
    if let Some(rejection) = eredu_core::check_admission_context(capabilities, request)? {
        return Err(PrefillPlanningError::Admission(rejection));
    }
    select_prefill_candidate(geometry, |geometry| {
        let result = match quote(geometry) {
            Ok(candidate) => admit(candidate, geometry),
            Err(PrefillPlanningError::IncompleteWorkspace(incomplete)) => {
                if incomplete.geometry() != geometry {
                    return Err(CandidateFailure::Terminal(
                        WorkingMemoryError::IdentityMismatch.into(),
                    ));
                }
                Err(PrefillPlanningError::IncompleteWorkspace(incomplete))
            }
            Err(
                error @ PrefillPlanningError::Reservation(
                    WorkingMemoryError::SubmissionTrackingCapacity { .. }
                    | WorkingMemoryError::GraphMetadataCapacity { .. },
                ),
            ) => Err(error),
            // Identity, geometry, native and arbitrary unknown-bound failures
            // are not smaller-chunk rejections; preserve their original cause.
            Err(error) => return Err(CandidateFailure::Terminal(error)),
        };
        result.map_err(|error| match candidate_can_shrink(&error) {
            true => CandidateFailure::SmallerChunk(error),
            false => CandidateFailure::Terminal(error),
        })
    })
}

fn candidate_can_shrink(error: &PrefillPlanningError) -> bool {
    match error {
        PrefillPlanningError::IncompleteWorkspace(_)
        | PrefillPlanningError::Reservation(
            WorkingMemoryError::Domain(MemoryDomainError::BudgetExceeded { .. })
            | WorkingMemoryError::DomainAllowanceExceeded { .. }
            | WorkingMemoryError::SubmissionTrackingCapacity { .. }
            | WorkingMemoryError::GraphMetadataCapacity { .. },
        )
        | PrefillPlanningError::Admission(
            eredu_core::AdmissionRejection::EstimationUnsupported { .. },
        ) => true,
        PrefillPlanningError::Reservation(WorkingMemoryError::ReservationMetadata(error)) => {
            error.planning_error().is_some_and(candidate_can_shrink)
        }
        _ => false,
    }
}

// One traversal for legacy owned diagnostics and original borrowed requirements.
// Each closed producer classifies its exact rejection without converting a fixed
// pre-admission cause into an allocating legacy error. Neither this traversal nor
// its callback grants work: only the existing accepted account commit does.
enum CandidateFailure<E> {
    SmallerChunk(E),
    Terminal(E),
}
fn select_prefill_candidate<T, E>(
    mut geometry: InferenceGeometry,
    mut candidate: impl FnMut(InferenceGeometry) -> Result<T, CandidateFailure<E>>,
) -> Result<T, E> {
    loop {
        match candidate(geometry) {
            Ok(output) => return Ok(output),
            Err(CandidateFailure::Terminal(error)) => return Err(error),
            Err(CandidateFailure::SmallerChunk(error)) => {
                // A terminal saved-state placement has no prefill at all.
                // Its refusal cannot be repaired by choosing a smaller chunk.
                if geometry.prefill_chunk_positions <= 1 {
                    return Err(error);
                }
                geometry.prefill_chunk_positions -= 1;
            }
        }
    }
}

/// Identity of a retained selected executable. Clones share identity; creating
/// a new owner is required when selection or parameter geometry changes.
#[derive(Debug, Clone, Default)]
pub struct InferenceExecutionIdentity(
    Arc<()>,
    // Prepared-copy identity storage belongs to its enclosing host account.
    // Clones carry that lifetime; the Arc retires before this final token.
    Option<eredu_core::HostPreparationAuthority>,
);
impl InferenceExecutionIdentity {
    /// Compare the actual retained selected executable identity. This read-only
    /// evidence creates no reservation, source alias or execution permission.
    pub fn same_execution(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

/// The concrete fixed collector whose accepted slots were exhausted.
/// These scalar facts report a constructor population mismatch; they confer
/// no new capacity and do not imply an unresolved native operation.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum CollectorCapacityKind {
    /// Retained roots in one work owner.
    WorkRoots,
    /// Preallocated root clone shells in one work owner.
    WorkCloneShells,
    /// Retained metadata in one work owner.
    WorkMetadata,
    /// Prepared inventory destinations in one work owner.
    WorkInventories,
    /// Retained publication destinations in one work owner.
    WorkPublications,
    /// Preallocated clone shells in one inventory.
    InventoryCloneShells,
    /// Distinct retained rows in one inventory.
    InventoryRows,
    /// Flattened storage entries prepared for publication.
    PublicationEntries,
}

/// Rejection before native preparation or submission.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
pub enum WorkingMemoryError {
    /// One authenticated backing was described with conflicting physical placement.
    #[error("managed storage aliases disagree about physical placement")]
    StoragePlacementMismatch,
    /// An allocation exceeds its admitted domain allowance, independently of spare capacity elsewhere.
    #[error(
        "allocation in {domain:?} needs {required_bytes} bytes; its account has {available_bytes} bytes"
    )]
    DomainAllowanceExceeded {
        domain: MemoryDomainId,
        required_bytes: u64,
        available_bytes: u64,
    },
    /// Invalid physical attribution, arithmetic, or physical-domain capacity.
    #[error(transparent)]
    Domain(#[from] MemoryDomainError),
    /// Fixed refusal by a participating host-metadata producer.
    #[error(transparent)]
    MetadataConstruction(eredu_nn::workspace::WorkspaceMetadataError),
    /// An owning metadata failure retaining its original cause and planning account.
    #[error(transparent)]
    ReservationMetadata(WorkspaceReservationMetadataError),
    /// Another reset has accepted its fixed entry but not yet published it.
    #[error("original reset admission is publishing its fixed account entry")]
    ResetAdmissionBusy,

    /// Required text or state workspace has no proved upper bound.
    #[error("inference working-memory bound is incomplete")]
    UnknownBound,
    /// Unquoted managed resources cannot coexist with an admitted request.
    #[error("unquoted managed work conflicts with a live inference reservation")]
    ReservedWorkActive,
    /// The supplied report did not price this exact request.
    #[error("inference working-memory admission does not match the selected request")]
    IdentityMismatch,
    /// The attempted decoder dimensions were not covered by this span's quote.
    #[error("inference span shape {actual:?} differs from admitted {expected:?}")]
    SpanShapeMismatch {
        /// Exact batch and position dimensions inspected by the quote.
        expected: [u64; 2],
        /// Actual decoder dimensions of the architecture input.
        actual: [u64; 2],
    },
    /// Native work would advance past the admitted output allowance.
    #[error("inference at position {position} exceeds admitted frontier {limit}")]
    OutputAllowanceExceeded {
        /// Position before the attempted invocation.
        position: u64,
        /// First position outside this request's admitted allowance.
        limit: u64,
    },
    /// Prefill and decode cannot substitute for each other's quoted graphs.
    #[error("inference invocation does not match the admitted prefill/decode phase")]
    InvocationPhaseMismatch,
    /// The admitted span would start at a different retained decoder position.
    #[error("prefill starts at admitted position {expected}, but retained state is at {actual}")]
    StateFrontierMismatch {
        /// Position covered by this span's admission.
        expected: u64,
        /// Actual retained decoder position.
        actual: u64,
    },
    /// An active observer requires positions absent from this admission.
    #[error(
        "inference admission covers {admitted:?} readout, but the observer requires {required:?}"
    )]
    OutputDemandMismatch {
        /// Readout geometry already priced.
        admitted: eredu_core::OutputDemand,
        /// Readout geometry required by the consumer.
        required: eredu_core::OutputDemand,
    },
    /// An explicitly admitted request cannot switch to an unquoted ingress path.
    #[error("the admitted request has no matching prepared prefill source")]
    PreparedSourceUnavailable,
    /// Chunk boundaries require exact completion for state and transient work.
    #[error("bounded prefill requires exact state completion")]
    CompletionUnavailable,
    /// A fixed collector exhausted its accepted slots. The actual cause is
    /// distinct from native/session fencing and carries no allocated message.
    #[error("original collector {kind:?} capacity exhausted: used {used}, capacity {capacity}")]
    CollectorCapacity {
        /// Concrete collector constructor.
        kind: CollectorCapacityKind,
        /// Slots already occupied or consumed before the refused attempt.
        used: usize,
        /// Slots accepted and prepared by that constructor.
        capacity: usize,
    },
    /// A prior unresolved native failure prevents further work on this session.
    #[error("inference session is fenced after an unresolved failure")]
    ExecutionFenced,
    /// This admitted request already started prefill. Cloning native retention
    /// cannot create another independently runnable request.
    #[error("inference request prefill authority was already consumed")]
    AlreadyStarted,
    /// This request already claimed preparation or a preparation stage.
    #[error("inference request preparation authority was already consumed")]
    PreparationAlreadyStarted,
    /// Claimed preparation has not published both prompt and sampler readiness.
    #[error("inference request preparation has not completed")]
    PreparationNotReady,
    /// A sampler configuration differs from the one bound at preparation admission.
    #[error("sampling configuration differs from the admitted preparation")]
    PreparationConfigurationMismatch,
    /// No original core-issued run and policy were bound before preparation.
    #[error("text preparation has no bound run authority")]
    TextRunUnbound,
    /// The preceding prediction has not finalized its logical permission.
    #[error("text run already has an active prediction permit")]
    TextStepActive,
    /// Evidence was replayed or skipped an issued attempt.
    #[error("text prediction attempt {actual} differs from expected {expected}")]
    TextStepOrdinalMismatch {
        /// Next ordinal retained by the non-restorable run authority.
        expected: u64,
        /// Ordinal supplied by the shared text machine.
        actual: u64,
    },
    /// Issued predictions consumed the run's complete output allowance.
    #[error("text run issued {issued} predictions against its {limit}-token allowance")]
    TextOutputAllowanceExceeded {
        /// Issued permits, including failed or cancelled attempts.
        issued: u64,
        /// Total outputs priced by the original request.
        limit: u64,
    },
    /// Two retained aliases disagree about one physical allocation's capacity.
    #[error("managed storage aliases report capacities {expected_bytes} and {actual_bytes}")]
    StorageCapacityMismatch {
        /// Capacity already registered for this identity.
        expected_bytes: u64,
        /// Conflicting capacity in the proposed registration.
        actual_bytes: u64,
    },
    /// A configured tracking arena cannot hold one selected constructor's
    /// simultaneously live blocks. This necessary bound is not a complete fit.
    #[error(
        "submission tracking needs at least {required_bytes} bytes; configured capacity is {configured_bytes}"
    )]
    SubmissionTrackingCapacity {
        /// Exact known constructor minimum from the selected backend mechanism.
        required_bytes: u64,
        /// User-configured tracking arena capacity.
        configured_bytes: u64,
    },
    /// The selected Graph constructor population exceeds its configured arena.
    #[error(
        "graph metadata needs {required_bytes} bytes; configured capacity is {configured_bytes}"
    )]
    GraphMetadataCapacity {
        /// Physical capacity required by the selected producer's Graph requests.
        required_bytes: u64,
        /// User-configured Graph arena ceiling.
        configured_bytes: u64,
    },
    /// The qualified managed control producer failed its actual reservation.
    #[error("managed control storage reservation failed")]
    ControlStorageReserve(#[source] std::collections::TryReserveError),
    /// Byte arithmetic overflowed.
    #[error("inference working-memory accounting overflow")]
    Overflow,
    /// Another accepted original account is publishing its prepaid fixed node.
    #[error("original account construction is in progress")]
    AccountConstructionBusy,
    /// An accounting operation unwound. Further admissions fail closed.
    #[error("inference working-memory owner is poisoned")]
    Poisoned,
}

struct Usage {
    transaction_buffers: Option<transaction_buffers::TransactionBuffers>,
    reset_entries: Option<Box<resident_reset::Entry>>,
    reset_pending: Option<resident_reset::Pending>,
    reset_retiring_count: usize,
    domains: Vec<domains::DomainUsage>,
    domain_ids: Vec<MemoryDomainId>,
    host_slot: usize,
    reservations: usize,
    unquoted_owners: usize,
    storage: storage::directory::Directory,
    funding: funding::AccountLedger,
    account_retiring: Option<funding::RetiringAccount>,
    pending_original: Option<funding::PendingOriginal>,
    next_funding: u64,
    next_reset: u64,
}

impl std::fmt::Debug for Usage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Usage")
            .field("reserved", &self.reserved)
            .field("reservations", &self.reservations)
            .field("unquoted_owners", &self.unquoted_owners)
            .field("registered", &self.registered)
            .field("storage_namespaces", &self.storage.len())
            .field("peak", &self.peak)
            .field("domain_count", &self.domains.len())
            .field("funding_accounts", &self.funding.len())
            .finish()
    }
}

#[derive(Debug)]
struct Pool {
    construction_execution: InferenceExecutionIdentity,
    construction_ready: std::sync::Condvar,
    topology: Arc<MemoryTopology>,
    limits: MemoryLimits,
    baseline: DomainMemoryRequirements,
    domains: Vec<domains::FixedDomain>,
    host_slot: usize,
    storage_accounting_id: eredu_core::SharedStorageAccountingId,
    host_placement: Arc<eredu_core::MemoryPlacement>,
    usage: Mutex<Usage>,
}

impl Pool {
    fn check_host_increment(&self, usage: &Usage, bytes: u64) -> Result<u64, WorkingMemoryError> {
        let used = self
            .existing
            .checked_add(usage.registered)
            .and_then(|n| n.checked_add(usage.reserved))
            .ok_or(WorkingMemoryError::Overflow)?;
        Ok(self
            .capacity(usage, None)?
            .check(self.topology.host_domain(), used, bytes)?)
    }
    fn capacity(
        &self,
        usage: &Usage,
        requested: Option<&MemoryLimits>,
    ) -> Result<MemoryLimit, WorkingMemoryError> {
        self.domain_capacity(usage, self.topology.host_domain(), requested)
    }
    #[cfg(test)]
    fn available(
        &self,
        usage: &Usage,
        requested: Option<&MemoryLimits>,
    ) -> Result<u64, WorkingMemoryError> {
        let MemoryLimit::Finite(limit) = self.capacity(usage, requested)? else {
            return Err(WorkingMemoryError::IdentityMismatch);
        };
        let used = self.check_host_increment(usage, 0)?;
        limit.checked_sub(used).ok_or(WorkingMemoryError::Overflow)
    }
}

// Shared numeric admission only. Each caller first proves its own concrete
// operation; this private plan cannot turn an arbitrary byte count into a public
// inference or copy permission.
enum ReservationRequirement<'a> {
    Host(u64),
    Domains(&'a DomainMemoryRequirements),
    Charges(&'a [DomainMemoryCharge]),
}
impl ReservationRequirement<'_> {
    fn charge(
        &self,
        pool: &MemoryLedger,
        domain: MemoryDomainId,
    ) -> Result<DomainMemoryCharge, WorkingMemoryError> {
        Ok(match self {
            Self::Host(bytes) if domain == pool.topology().host_domain() => DomainMemoryCharge {
                accounted_bytes: *bytes,
                ..Default::default()
            },
            Self::Host(_) => DomainMemoryCharge::default(),
            Self::Domains(requirements) => requirements.get(domain)?,
            Self::Charges(charges) => *charges
                .get(pool.topology().slot(domain)?)
                .ok_or(WorkingMemoryError::IdentityMismatch)?,
        })
    }
}
struct PreparedAccountCommit<'h> {
    ceilings: Option<funding::CapacityHandoffPlan<'h>>,
    reservations: usize,
    required: ReservationRequirement<'h>,
}
impl<'h> PreparedAccountCommit<'h> {
    fn prepare(
        pool: &MemoryLedger,
        execution: &InferenceExecutionIdentity,
        usage: &Usage,
        required: u64,
        capacity: Option<&'h MemoryLimits>,
        handoffs: &'h [WorkingMemoryCapacityHandoff],
    ) -> Result<Self, WorkingMemoryError> {
        Self::prepare_requirement(
            pool,
            execution,
            usage,
            ReservationRequirement::Host(required),
            capacity,
            handoffs,
        )
    }
    fn prepare_domains(
        pool: &MemoryLedger,
        execution: &InferenceExecutionIdentity,
        usage: &Usage,
        required: &'h DomainMemoryRequirements,
        capacity: Option<&'h MemoryLimits>,
        handoffs: &'h [WorkingMemoryCapacityHandoff],
    ) -> Result<Self, WorkingMemoryError> {
        required.validate(pool.topology())?;
        Self::prepare_requirement(
            pool,
            execution,
            usage,
            ReservationRequirement::Domains(required),
            capacity,
            handoffs,
        )
    }
    fn prepare_requirement(
        pool: &MemoryLedger,
        execution: &InferenceExecutionIdentity,
        usage: &Usage,
        required: ReservationRequirement<'h>,
        capacity: Option<&'h MemoryLimits>,
        handoffs: &'h [WorkingMemoryCapacityHandoff],
    ) -> Result<Self, WorkingMemoryError> {
        if usage.unquoted_owners != 0 {
            return Err(WorkingMemoryError::UnknownBound);
        }
        if usage.pending_original.is_some() {
            return Err(WorkingMemoryError::AccountConstructionBusy);
        }
        let ceilings =
            funding::CapacityHandoffPlan::prepare(pool, execution, usage, capacity, handoffs)?;
        Self::prepare_checked(pool, usage, required, capacity, Some(ceilings))
    }
    fn prepare_reset(
        pool: &MemoryLedger,
        usage: &Usage,
        required: &'h DomainMemoryRequirements,
        capacity: &'h MemoryLimits,
    ) -> Result<Self, WorkingMemoryError> {
        required.validate(pool.topology())?;
        capacity.validate(pool.topology())?;
        if usage.unquoted_owners != 0 {
            return Err(WorkingMemoryError::UnknownBound);
        }
        if usage.pending_original.is_some() {
            return Err(WorkingMemoryError::AccountConstructionBusy);
        }
        Self::prepare_checked(
            pool,
            usage,
            ReservationRequirement::Domains(required),
            Some(capacity),
            None,
        )
    }
    fn prepare_checked(
        pool: &MemoryLedger,
        usage: &Usage,
        required: ReservationRequirement<'h>,
        capacity: Option<&MemoryLimits>,
        ceilings: Option<funding::CapacityHandoffPlan<'h>>,
    ) -> Result<Self, WorkingMemoryError> {
        let reservations = usage
            .reservations
            .checked_add(1)
            .ok_or(WorkingMemoryError::Overflow)?;
        for (slot, (domain, _)) in pool.topology().domains().enumerate() {
            let current = &usage.domains[slot];
            let charge = required.charge(pool, domain)?;
            let increment = charge.total()?;
            let used = pool.0.domains[slot]
                .existing
                .checked_add(current.registered)
                .and_then(|n| n.checked_add(current.reserved))
                .ok_or(WorkingMemoryError::Overflow)?;
            let limit = match &ceilings {
                Some(ceilings) => ceilings.effective_capacity(pool, usage, domain)?,
                None => pool.0.domain_capacity(usage, domain, capacity)?,
            };
            limit.check(domain, used, increment)?;
            current
                .reserved
                .checked_add(increment)
                .ok_or(WorkingMemoryError::Overflow)?;
            current
                .placement_allowances
                .checked_add(charge.placement_allowance_bytes)
                .ok_or(WorkingMemoryError::Overflow)?;
            current
                .estimates
                .checked_add(charge.estimated_overhead_bytes)
                .ok_or(WorkingMemoryError::Overflow)?;
            current
                .headroom
                .checked_add(charge.headroom_bytes)
                .ok_or(WorkingMemoryError::Overflow)?;
        }
        Ok(Self {
            ceilings,
            reservations,
            required,
        })
    }
    fn commit(self, pool: &MemoryLedger, usage: &mut Usage) {
        if let Some(ceilings) = self.ceilings {
            ceilings.commit(usage);
        }
        usage.reservations = self.reservations;
        for (slot, (domain, _)) in pool.topology().domains().enumerate() {
            let charge = self
                .required
                .charge(pool, domain)
                .expect("validated requirements");
            let current = &mut usage.domains[slot];
            current.reserved += charge.total().expect("validated charge");
            current.placement_allowances += charge.placement_allowance_bytes;
            current.estimates += charge.estimated_overhead_bytes;
            current.headroom += charge.headroom_bytes;
            current.peak = current
                .peak
                .max(pool.0.domains[slot].existing + current.registered + current.reserved);
        }
    }
}

/// One process-local coordinator for the backend's immutable physical topology.
/// All sessions using those domains share storage identities, funding accounts,
/// source pins, and completion-safe retirement under this single owner.
/// Locations mapped to one physical domain share its capacity. Authenticated
/// aliases of one backing allocation share its charge. Storage registrations
/// cover changing retained inventories; the fixed baseline is disjoint from
/// those registrations and remains live for the ledger lifetime.
/// Owners without a complete bound must hold an unquoted lease, excluding
/// request reservations until their work settles and storage is accounted for.
/// This bounds Eredu-managed allocations, not other process/system allocations.
#[derive(Debug, Clone)]
pub struct MemoryLedger(Arc<Pool>);

impl MemoryLedger {
    /// Private accounting identity for construction before a funded operation
    /// creates its own execution identity. It grants no execution authority.
    pub(in crate::working_memory) fn construction_identity(&self) -> &InferenceExecutionIdentity {
        &self.0.construction_execution
    }
    /// Identity used to attach immutable storage to this accounting owner.
    /// It describes no physical placement and grants no allocation permission.
    pub fn shared_storage_accounting_id(&self) -> &eredu_core::SharedStorageAccountingId {
        &self.0.storage_accounting_id
    }

    /// Whether two handles refer to the same accounting ledger.
    /// This compares identity only and performs no accounting or native work.
    pub fn same_ledger(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }

    /// Creates one coordinator for the backend's immutable physical topology.
    /// Baseline allocations remain charged for this ledger's entire lifetime.
    pub fn new(
        topology: Arc<MemoryTopology>,
        limits: MemoryLimits,
        mut baseline: DomainMemoryRequirements,
    ) -> Result<Self, WorkingMemoryError> {
        limits.validate(&topology)?;
        baseline.validate(&topology)?;
        let fixed = Self::fixed_owner_bytes(&topology, &limits, &baseline)?;
        baseline.add_allocation(
            fixed,
            &eredu_core::MemoryPlacement::fixed(&topology, topology.host_domain())?,
        )?;
        let host_slot = topology.slot(topology.host_domain())?;
        let domain_ids = topology.domains().map(|(id, _)| id).collect();
        let host_placement = Arc::new(eredu_core::MemoryPlacement::fixed(
            &topology,
            topology.host_domain(),
        )?);
        let mut domains = Vec::with_capacity(topology.len());
        let mut current = Vec::with_capacity(topology.len());
        for (domain, charge) in baseline.iter() {
            let existing = charge.total()?;
            let capacity = limits.get(domain)?;
            capacity.check(domain, 0, existing)?;
            domains.push(domains::FixedDomain { existing, capacity });
            current.push(domains::DomainUsage {
                peak: existing,
                ..Default::default()
            });
        }
        let transaction_buffers = Some(transaction_buffers::TransactionBuffers::new(&topology));
        let ledger = Self(Arc::new(Pool {
            construction_execution: InferenceExecutionIdentity::default(),
            construction_ready: std::sync::Condvar::new(),
            topology,
            limits,
            baseline,
            domains,
            host_slot,
            host_placement,
            storage_accounting_id: eredu_core::SharedStorageAccountingId::default(),
            usage: Mutex::new(Usage {
                transaction_buffers,
                reset_entries: None,
                reset_pending: None,
                reset_retiring_count: 0,
                domains: current,
                host_slot,
                domain_ids,
                reservations: 0,
                unquoted_owners: 0,
                storage: Default::default(),
                account_retiring: None,
                pending_original: None,
                funding: Default::default(),
                next_funding: 0,
                next_reset: 0,
            }),
        }));
        drop(
            ledger
                .0
                .usage
                .lock()
                .map_err(|_| WorkingMemoryError::Poisoned)?,
        );
        // Initialize Darwin's lazy condition-variable backing before this
        // coordinator can escape to competing constructors.
        ledger.0.construction_ready.notify_all();
        Ok(ledger)
    }

    /// Atomically reserves a complete admission against all other live requests.
    /// Safety reserve and state/media are already included by core admission.
    /// Any live unquoted lease rejects admission with `UnknownBound`.
    pub fn reserve(
        &self,
        execution: &InferenceExecutionIdentity,
        admission: &Admission,
    ) -> Result<WorkingMemoryReservation, WorkingMemoryError> {
        self.reserve_limited(execution, admission, None)
    }

    /// Atomically reserves complete request work under per-domain ceilings,
    /// including baselines, registered storage and concurrent reservations.
    /// Clones retain these constraints until the last charge owner retires.
    pub fn reserve_with_capacity(
        &self,
        execution: &InferenceExecutionIdentity,
        admission: &Admission,
        capacity: MemoryLimits,
    ) -> Result<WorkingMemoryReservation, WorkingMemoryError> {
        self.reserve_limited(execution, admission, Some(capacity))
    }

    /// Retains managed storage or work whose complete bound is unavailable.
    /// Acquire before starting unquoted work. Any live request reservation,
    /// including a zero-byte reservation, rejects acquisition. Conversely, this
    /// lease prevents both reservation methods from admitting requests.
    ///
    /// This records incomplete domain accounting; it grants no byte capacity.
    /// Retain it through native completion, failures and surviving descendants.
    /// Release it only after the work settles and all surviving managed storage
    /// has a complete registration or has retired. Storage registration may
    /// coexist with this lease and does not remove it automatically.
    pub fn acquire_unquoted(&self) -> Result<WorkingMemoryUnquotedLease, WorkingMemoryError> {
        let bytes = Self::unquoted_owner_control_bytes()?;
        let mut usage = self
            .0
            .usage
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)?;
        if usage.reservations != 0 {
            return Err(WorkingMemoryError::ReservedWorkActive);
        }
        let owners = usage
            .unquoted_owners
            .checked_add(1)
            .ok_or(WorkingMemoryError::Overflow)?;
        let charge = self.0.check_host_increment(&usage, bytes)?;
        let slot = self.0.host_slot;
        let registered = usage.domains[slot]
            .registered
            .checked_add(bytes)
            .ok_or(WorkingMemoryError::Overflow)?;
        let metadata = usage.domains[slot]
            .registry_metadata
            .checked_add(bytes)
            .ok_or(WorkingMemoryError::Overflow)?;
        usage.unquoted_owners = owners;
        usage.domains[slot].registered = registered;
        usage.domains[slot].registry_metadata = metadata;
        usage.domains[slot].peak = usage.domains[slot].peak.max(charge);
        let owner = UnquotedOwner {
            pool: self.clone(),
            bytes,
        };
        drop(usage);
        Ok(WorkingMemoryUnquotedLease(Some(Arc::new(owner))))
    }

    /// Host allowance for one admission-excluding owner, retained by all aliases.
    /// This pays its ownership controls; it supplies no bound for native payloads.
    pub fn unquoted_owner_control_bytes() -> Result<u64, WorkingMemoryError> {
        use std::{alloc::Layout, mem::size_of};
        let allocation = Layout::new::<[usize; 2]>()
            .extend(Layout::new::<UnquotedOwner>())
            .map_err(|_| WorkingMemoryError::Overflow)?
            .0
            .pad_to_align()
            .size();
        let frames = [
            allocation,
            size_of::<UnquotedOwner>(),
            size_of::<WorkingMemoryUnquotedLease>(),
            size_of::<Result<WorkingMemoryUnquotedLease, WorkingMemoryError>>(),
            size_of::<Option<UnquotedOwner>>(),
            size_of::<&MemoryLedger>(),
            size_of::<u64>() * 4,
            size_of::<usize>() * 2,
        ];
        let bytes = frames
            .into_iter()
            .try_fold(std::mem::size_of_val(&frames), usize::checked_add)
            .ok_or(WorkingMemoryError::Overflow)?;
        u64::try_from(bytes).map_err(|_| WorkingMemoryError::Overflow)
    }

    /// Number of independent live unquoted owners; cloned leases count once.
    /// A nonzero result means byte reports exclude unbounded managed resources.
    /// Zero is meaningful only when all domain owners follow this accounting
    /// protocol. This is cold evidence; admission rechecks under the same lock.
    pub fn unquoted_owner_count(&self) -> Result<usize, WorkingMemoryError> {
        Ok(self
            .0
            .usage
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)?
            .unquoted_owners)
    }

    fn reserve_limited(
        &self,
        execution: &InferenceExecutionIdentity,
        admission: &Admission,
        capacity: Option<eredu_core::MemoryLimits>,
    ) -> Result<WorkingMemoryReservation, WorkingMemoryError> {
        self.reserve_limited_with_minimum(execution, admission, capacity, None)
    }

    fn reserve_limited_with_minimum(
        &self,
        execution: &InferenceExecutionIdentity,
        admission: &Admission,
        capacity: Option<eredu_core::MemoryLimits>,
        residual: Option<(&DomainMemoryRequirements, residual::RegisteredStoragePin)>,
    ) -> Result<WorkingMemoryReservation, WorkingMemoryError> {
        self.reserve_limited_with_minimum_and_handoffs(
            execution,
            admission,
            capacity,
            residual,
            &[],
        )
    }

    fn reserve_limited_with_minimum_and_handoffs(
        &self,
        execution: &InferenceExecutionIdentity,
        admission: &Admission,
        capacity: Option<eredu_core::MemoryLimits>,
        residual: Option<(&DomainMemoryRequirements, residual::RegisteredStoragePin)>,
        handoffs: &[WorkingMemoryCapacityHandoff],
    ) -> Result<WorkingMemoryReservation, WorkingMemoryError> {
        self.reserve_limited_with_source(execution, admission, capacity, residual, handoffs, None)
    }

    fn reserve_limited_with_source(
        &self,
        execution: &InferenceExecutionIdentity,
        admission: &Admission,
        capacity: Option<eredu_core::MemoryLimits>,
        residual: Option<(&DomainMemoryRequirements, residual::RegisteredStoragePin)>,
        handoffs: &[WorkingMemoryCapacityHandoff],
        source: Option<&dyn saved_source::SavedSourceValidation>,
    ) -> Result<WorkingMemoryReservation, WorkingMemoryError> {
        self.reserve_limited_with_source_metadata(
            execution, admission, capacity, residual, handoffs, source, None,
        )
    }

    fn reserve_limited_with_source_metadata(
        &self,
        execution: &InferenceExecutionIdentity,
        admission: &Admission,
        capacity: Option<eredu_core::MemoryLimits>,
        residual: Option<(&DomainMemoryRequirements, residual::RegisteredStoragePin)>,
        handoffs: &[WorkingMemoryCapacityHandoff],
        source: Option<&dyn saved_source::SavedSourceValidation>,
        planning_metadata: Option<eredu_nn::workspace::HostMetadataFunding>,
    ) -> Result<WorkingMemoryReservation, WorkingMemoryError> {
        // This owner is declared before all staged metadata and the usage loan.
        // Fixed failures release their unpublished prefixes before it retires.
        let planning_metadata = planning_metadata;
        let state = &admission.state;
        let workspace = state
            .execution_workspace
            .as_ref()
            .ok_or(WorkingMemoryError::UnknownBound)?;
        let state_domains = state
            .physical_domains
            .as_ref()
            .ok_or(WorkingMemoryError::UnknownBound)?;
        let workspace_domains = workspace
            .physical_domains
            .as_ref()
            .ok_or(WorkingMemoryError::UnknownBound)?;
        if !eredu_core::AdmissionStateRequirements::from(state).physical_domains {
            return Err(WorkingMemoryError::UnknownBound);
        }
        if state.completeness == EstimationCompleteness::PersistentStateOnly
            || state.persistent_state_completeness == EstimationCompleteness::PersistentStateOnly
        {
            return Err(WorkingMemoryError::UnknownBound);
        }
        let geometry = workspace.geometry;
        geometry.validate_fixed().map_err(|cause| match cause {
            eredu_core::AdmissionPolicyError::ArithmeticOverflow { .. } => {
                WorkingMemoryError::Overflow
            }
            _ => WorkingMemoryError::IdentityMismatch,
        })?;
        // Aggregate values are diagnostics, but a supplied value must still
        // describe its report. Authenticated residual quotations validate their
        // own incremental diagnostics before entering this shared worker.
        if residual.is_none() {
            if let Some(reported) = admission.incremental_required_bytes {
                let full = workspace
                    .peak_bytes_fixed()
                    .ok()
                    .flatten()
                    .and_then(|bytes| state.requested_state_bytes.checked_add(bytes));
                if full != Some(reported) {
                    return Err(WorkingMemoryError::IdentityMismatch);
                }
            }
        }
        if state_domains.geometry != geometry
            || workspace_domains.geometry != geometry
            || state
                .selected_state_backing
                .as_ref()
                .is_some_and(|backing| backing.geometry != geometry)
        {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        if planning_metadata.is_none() {
            return self
                .reserve_standalone(execution, admission, capacity, residual, handoffs, source);
        }
        let requested_positions = geometry
            .cached_positions
            .checked_add(geometry.input_positions)
            .and_then(|positions| positions.checked_add(geometry.max_output_tokens))
            .ok_or(WorkingMemoryError::Overflow)?;
        if geometry.batch_size != state.assumptions.batch_size
            || requested_positions != admission.requested_positions
            || admission.requested_positions != state.assumptions.requested_positions
        {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let funding = planning_metadata.as_ref().expect("funded preparation path");
        let full = [
            &state_domains.decoder_state,
            &state_domains.media_embeddings,
            &state_domains.media_workspace,
            &workspace_domains.activations,
            &workspace_domains.attention,
            &workspace_domains.vocabulary,
            &workspace_domains.state_update,
            &workspace_domains.materialization,
            &workspace_domains.retained,
        ];
        for part in full {
            part.validate(self.topology())?;
        }
        for (domain, _) in self.topology().domains() {
            full.iter()
                .try_fold(DomainMemoryCharge::default(), |sum, part| {
                    sum.checked_add(part.get(domain)?)
                })?;
        }
        let residual_parts;
        let parts = match &residual {
            Some((requirements, _)) => {
                residual_parts = [*requirements];
                &residual_parts[..]
            }
            None => &full[..],
        };
        let projection = transaction_buffers::RequirementProjection {
            parts,
            headroom: &admission.additional_headroom,
            host_bytes: 0,
        };
        let limits_backing = u64::try_from(
            self.topology()
                .len()
                .checked_mul(std::mem::size_of::<MemoryLimit>())
                .ok_or(WorkingMemoryError::Overflow)?,
        )
        .map_err(|_| WorkingMemoryError::Overflow)?;
        funding
            .reserve_metadata(reservation_metadata::constructor_bytes_from_backing(
                self.topology(),
                projection.backing_bytes(self.topology())?,
                limits_backing,
            )?)
            .map_err(reservation_metadata::funding_error)?;
        let requirements = projection.materialize(self.topology())?;
        let mut resolved = MemoryLimits::unlimited(self.topology());
        resolved.resolve_named_in_place(
            self.topology(),
            &admission.memory_limits,
            capacity.as_ref(),
        )?;
        let capacity = Some(resolved);
        let metadata = WorkspaceReportMetadata::with_funding(funding);
        let retained_admission = metadata
            .clone_admission(admission)
            .map_err(|cause| reservation_metadata::neural_error(metadata.error(cause), funding))?;
        if let Some(source) = source {
            funding
                .reserve_metadata(source.pin_control_bytes()?)
                .map_err(reservation_metadata::funding_error)?;
        }
        let borrowed_storage = match (residual.map(|(_, pin)| pin), source) {
            (Some(pin), Some(source)) => Some(
                residual::RegisteredStoragePin::pair_metadata(pin, source.pin(), metadata)
                    .map_err(|cause| reservation_metadata::neural_error(cause, funding))?,
            ),
            (Some(pin), None) => Some(pin),
            (None, Some(source)) => Some(source.pin()),
            (None, None) => None,
        };
        let node = funding::AccountNode::empty_with_planning(planning_metadata.clone());
        let prepared_state = funding::FundingState::new(
            self.topology(),
            &requirements,
            capacity.clone(),
            execution,
            true,
            0,
            0,
        )?;
        let mut usage = self
            .0
            .usage
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)?;
        if let Some(source) = source {
            source.validate(self, &usage)?;
        }
        let commit = PreparedAccountCommit::prepare_domains(
            self,
            execution,
            &usage,
            &requirements,
            capacity.as_ref(),
            handoffs,
        )?;
        let account_id = usage.next_funding;
        let next = account_id
            .checked_add(1)
            .ok_or(WorkingMemoryError::Overflow)?;
        commit.commit(self, &mut usage);
        usage
            .funding
            .publish(node, account_id, prepared_state, false);
        usage.next_funding = next;
        drop(usage);
        Ok(WorkingMemoryReservation(ReservationOwner::new(
            Reservation {
                account_id,
                pool: self.clone(),
                execution: execution.clone(),
                admission: retained_admission,
                geometry,
                requirements,
                capacity,
                funding: None,
                borrowed_storage,
                span_workspace: None,
                start: ControlMutex::new(text_preparation::RequestStart::Fresh),
                planning_metadata,
            },
        )))
    }
}

#[derive(Debug)]
struct UnquotedOwner {
    pool: MemoryLedger,
    bytes: u64,
}

impl Drop for UnquotedOwner {
    fn drop(&mut self) {
        // Poison remains sticky for new work; existing owners still retire.
        let mut usage = self.pool.0.usage.lock().unwrap_or_else(|p| p.into_inner());
        usage.unquoted_owners = usage
            .unquoted_owners
            .checked_sub(1)
            .expect("retained unquoted owner");
        let slot = self.pool.0.host_slot;
        usage.domains[slot].registered = usage.domains[slot]
            .registered
            .checked_sub(self.bytes)
            .expect("retained unquoted host controls");
        usage.domains[slot].registry_metadata = usage.domains[slot]
            .registry_metadata
            .checked_sub(self.bytes)
            .expect("retained unquoted host controls");
    }
}

/// Shared ownership of one unquoted ledger participant. Clones preserve the
/// same admission blocker and paid host controls until the final owner retires.
/// Its payload has no byte bound; this is not a request reservation.
#[derive(Debug, Clone)]
#[must_use = "retain the lease until native work settles and its storage is accounted for or retired"]
pub struct WorkingMemoryUnquotedLease(Option<Arc<UnquotedOwner>>);

impl WorkingMemoryUnquotedLease {
    fn inner(&self) -> &Arc<UnquotedOwner> {
        self.0.as_ref().expect("live unquoted lease")
    }
}

impl Drop for WorkingMemoryUnquotedLease {
    fn drop(&mut self) {
        // Retire the Arc allocation before its final value refunds the charge.
        if let Some(owner) = self.0.take().and_then(Arc::into_inner) {
            drop(owner);
        }
    }
}

#[cfg(test)]
mod unquoted_tests {
    use super::*;

    #[test]
    fn host_controls_fit_exactly_and_rejection_is_atomic() {
        let bytes = MemoryLedger::unquoted_owner_control_bytes().unwrap();
        for available in [0, bytes - 1] {
            let pool = memory_fixture::host_ledger(available, 0).unwrap();
            let before = pool.snapshot().unwrap();
            assert!(
                matches!(pool.acquire_unquoted(), Err(WorkingMemoryError::Domain(
                MemoryDomainError::BudgetExceeded { requested_bytes, .. })) if requested_bytes == bytes)
            );
            assert_eq!(pool.snapshot().unwrap(), before);
        }
        let pool = memory_fixture::host_ledger(bytes, 0).unwrap();
        let before = pool.snapshot().unwrap();
        let owner = pool.acquire_unquoted().unwrap();
        let alias = owner.clone();
        let charged = pool.snapshot().unwrap();
        assert_eq!(charged.unquoted_owners, 1);
        assert_eq!(
            charged.domains[0].current_charge_bytes,
            before.domains[0].current_charge_bytes + bytes
        );
        assert_eq!(charged.domains[0].registry_metadata_bytes, bytes);
        assert!(pool.acquire_unquoted().is_err());
        assert_eq!(pool.snapshot().unwrap(), charged);
        drop(owner);
        assert_eq!(pool.snapshot().unwrap(), charged);
        drop(alias);
        let retired = pool.snapshot().unwrap();
        assert_eq!(retired.unquoted_owners, 0);
        assert_eq!(
            retired.domains[0].current_charge_bytes,
            before.domains[0].current_charge_bytes
        );
        assert_eq!(
            retired.domains[0].historical_peak_bytes,
            charged.domains[0].current_charge_bytes
        );
    }

    #[test]
    fn unlimited_owners_still_charge_host_controls_and_check_overflow() {
        let topology = memory_fixture::host_topology();
        let pool = MemoryLedger::new(
            topology.clone(),
            MemoryLimits::unlimited(&topology),
            DomainMemoryRequirements::zero(&topology),
        )
        .unwrap();
        let before = pool.snapshot().unwrap();
        let owner = pool.acquire_unquoted().unwrap();
        let charged = pool.snapshot().unwrap();
        assert_eq!(
            charged.domains[0].current_charge_bytes,
            before.domains[0].current_charge_bytes
                + MemoryLedger::unquoted_owner_control_bytes().unwrap()
        );
        drop(owner);
        {
            let mut usage = pool.0.usage.lock().unwrap();
            usage.registered = u64::MAX - pool.0.existing;
            usage.peak = u64::MAX;
        }
        let before = pool.snapshot().unwrap();
        assert!(matches!(
            pool.acquire_unquoted(),
            Err(WorkingMemoryError::Domain(MemoryDomainError::Overflow))
                | Err(WorkingMemoryError::Overflow)
        ));
        assert_eq!(pool.snapshot().unwrap(), before);
    }

    #[test]
    fn unquoted_owner_overflow_preserves_accounting() {
        let pool = crate::working_memory::memory_fixture::host_ledger(100, 40).unwrap();
        pool.0.usage.lock().unwrap().unquoted_owners = usize::MAX;
        assert!(matches!(
            pool.acquire_unquoted(),
            Err(WorkingMemoryError::Overflow)
        ));
        assert_eq!(pool.unquoted_owner_count().unwrap(), usize::MAX);
        assert_eq!(pool.payload_used_bytes().unwrap(), 40);
        assert_eq!(pool.payload_peak_bytes().unwrap(), 40);
        assert_eq!(pool.payload_effective_capacity().unwrap(), 100);
        let usage = pool.0.usage.lock().unwrap();
        assert_eq!(usage.reservations, 0);
        assert_eq!(
            usage
                .funding
                .capacity(pool.topology().host_domain())
                .unwrap(),
            MemoryLimit::Unlimited
        );
    }

    #[test]
    fn poison_rejects_new_unquoted_work_but_retirement_still_releases_owners() {
        let bytes = MemoryLedger::unquoted_owner_control_bytes().unwrap();
        let pool = crate::working_memory::memory_fixture::host_ledger(100 + bytes, 40).unwrap();
        let owner = pool.acquire_unquoted().unwrap();
        let completion = owner.clone();
        let poisoned = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _usage = pool.0.usage.lock().unwrap();
            panic!("accounting operation failed");
        }));
        assert!(poisoned.is_err());
        assert!(matches!(
            pool.acquire_unquoted(),
            Err(WorkingMemoryError::Poisoned)
        ));
        assert_eq!(
            pool.unquoted_owner_count(),
            Err(WorkingMemoryError::Poisoned)
        );
        assert_eq!(pool.payload_used_bytes(), Err(WorkingMemoryError::Poisoned));
        assert_eq!(pool.payload_peak_bytes(), Err(WorkingMemoryError::Poisoned));
        drop(owner);
        {
            let usage = pool.0.usage.lock().unwrap_err().into_inner();
            assert_eq!(usage.unquoted_owners, 1);
        }
        drop(completion);
        let usage = pool.0.usage.lock().unwrap_err().into_inner();
        assert_eq!(usage.unquoted_owners, 0);
        assert_eq!(usage.reserved, 0);
        assert_eq!(usage.peak, pool.0.existing + bytes);
        assert_eq!(usage.registry_metadata, 0);
    }
}

#[derive(Debug)]
struct Reservation {
    account_id: u64,
    pool: MemoryLedger,
    execution: InferenceExecutionIdentity,
    admission: Admission,
    geometry: InferenceGeometry,
    requirements: DomainMemoryRequirements,
    capacity: Option<eredu_core::MemoryLimits>,
    funding: Option<u64>,
    borrowed_storage: Option<residual::RegisteredStoragePin>,
    // Identity metadata only: historical requests never retain the span Vec.
    span_workspace: Option<residual::SpanWorkspaceIdentity>,
    start: ControlMutex<text_preparation::RequestStart>,
    // Independent initial metadata admission, last through the closed owner.
    planning_metadata: Option<eredu_nn::workspace::HostMetadataFunding>,
}

#[derive(Debug)]
struct ReservationOwner(Option<Arc<Reservation>>);
impl ReservationOwner {
    fn new(value: Reservation) -> Self {
        Self(Some(Arc::new(value)))
    }
    fn inner(&self) -> &Arc<Reservation> {
        self.0.as_ref().expect("live reservation owner")
    }
    fn get_mut(&mut self) -> Option<&mut Reservation> {
        Arc::get_mut(self.0.as_mut()?)
    }
    fn same(&self, other: &Self) -> bool {
        Arc::ptr_eq(self.inner(), other.inner())
    }
}
impl std::ops::Deref for ReservationOwner {
    type Target = Reservation;
    fn deref(&self) -> &Reservation {
        self.inner()
    }
}
impl Clone for ReservationOwner {
    fn clone(&self) -> Self {
        Self(Some(Arc::clone(self.inner())))
    }
}
impl Drop for ReservationOwner {
    fn drop(&mut self) {
        let Some(value) = self.0.take().and_then(Arc::into_inner) else {
            return;
        };
        // The Arc allocation has retired before this value can release metadata.
        let Reservation {
            account_id,
            pool,
            execution,
            admission,
            geometry: _,
            requirements,
            capacity,
            funding,
            borrowed_storage,
            span_workspace,
            start,
            planning_metadata,
        } = value;
        // Retire Q/account cleanup before releasing this final metadata alias,
        // including when a payload destructor unwinds below.
        let _planning_metadata = planning_metadata;
        struct Retire {
            pool: MemoryLedger,
            id: u64,
            funded: bool,
        }
        impl Drop for Retire {
            fn drop(&mut self) {
                let mut usage = funding::lock_for_retirement(&self.pool);
                if !self.funded {
                    usage.funding.retire_unfunded(self.id);
                }
                funding::retire_metadata(&mut usage, self.id);
            }
        }
        let _retire = Retire {
            pool,
            id: account_id,
            funded: funding.is_some(),
        };
        // Payload destruction (including poisoned preparation mutex contents)
        // finishes before Retire, including when an ordinary destructor unwinds.
        drop((
            execution,
            admission,
            requirements,
            capacity,
            borrowed_storage,
            span_workspace,
            start,
        ));
    }
}

/// Shared ownership of one admitted request. Ordinary reservations retain their
/// complete charge until the final clone retires. Explicit conversion through
/// [`Self::into_funding`] separates metadata lifetime from workspace funding;
/// the funding run and certified work scopes then control that balance.
/// Metadata and funded storage preserve their capacity/exclusion barrier. Only
/// explicit succession delegated by the unique funding owner can raise the live
/// account's ceiling; original admission diagnostics remain unchanged. Copies
/// never create another allocation permission or refund.
#[derive(Debug, Clone)]
#[must_use = "retain the reservation through native completion and retained state lifetime"]
pub struct WorkingMemoryReservation(ReservationOwner);

impl WorkingMemoryReservation {
    /// Original admission diagnostics and incremental requirement. Full selected
    /// state and workspace estimates remain available even when registered
    /// storage reduces the reservation. Funding conversion preserves this
    /// metadata; it conveys no new submission or allocation authority.
    pub fn admission(&self) -> &Admission {
        &self.0.admission
    }

    /// Whether allocation requires separate funding and a private execution proof.
    /// This includes both converted evidence and a residual reservation whose
    /// borrowed storage must transfer into funding before native preparation.
    /// Converted metadata alone no longer retains unused workspace.
    /// The result remains true after run closure, so historical identity and
    /// geometry checks can still validate predecessors without reopening work.
    pub fn requires_funding_scope(&self) -> bool {
        self.0.funding.is_some() || self.0.borrowed_storage.is_some()
    }

    /// Checks ledger identity independently of executable and geometry.
    /// A retained reservation grants no allocation permission in another ledger.
    pub fn validate_ledger(&self, pool: &MemoryLedger) -> Result<(), WorkingMemoryError> {
        if self.0.pool.same_ledger(pool) {
            Ok(())
        } else {
            Err(WorkingMemoryError::IdentityMismatch)
        }
    }

    /// Validates the exact executable and geometry before expensive work.
    pub fn validate(
        &self,
        execution: &InferenceExecutionIdentity,
        geometry: InferenceGeometry,
    ) -> Result<(), WorkingMemoryError> {
        if Arc::ptr_eq(&self.0.execution.0, &execution.0) && self.0.geometry == geometry {
            Ok(())
        } else {
            Err(WorkingMemoryError::IdentityMismatch)
        }
    }

    /// Request geometry whose entire lifetime this reservation covers.
    pub fn geometry(&self) -> InferenceGeometry {
        self.0.geometry
    }

    /// Original incremental quote, including new state, transients and reserve.
    /// A residual quote excludes separately pinned existing storage by identity.
    /// For a converted funding reservation this remains its admission bound,
    /// not its changing unassigned workspace balance or total domain usage.
    pub fn requirements(&self) -> &DomainMemoryRequirements {
        &self.0.requirements
    }
}

/// Request authority backed by an ordinary ledger reservation under finite or
/// unlimited limits. Geometry, preparation and completion custody are shared.
#[derive(Debug, Clone)]
pub struct InferenceRequest {
    // Retire the outer preparation alias before the final request/reservation.
    preparation: Option<Arc<text_preparation::TextPreparationAuthority>>,
    authority: RequestOwner,
}

#[derive(Debug)]
struct RequestOwner(Option<Arc<WorkingMemoryReservation>>);
impl RequestOwner {
    fn new(value: WorkingMemoryReservation) -> Self {
        Self(Some(Arc::new(value)))
    }
    fn same(&self, other: &Self) -> bool {
        Arc::ptr_eq(
            self.0.as_ref().expect("live request"),
            other.0.as_ref().expect("live request"),
        )
    }
}
impl std::ops::Deref for RequestOwner {
    type Target = WorkingMemoryReservation;
    fn deref(&self) -> &WorkingMemoryReservation {
        self.0.as_deref().expect("live request owner")
    }
}
impl Clone for RequestOwner {
    fn clone(&self) -> Self {
        Self(Some(Arc::clone(self.0.as_ref().expect("live request"))))
    }
}
impl Drop for RequestOwner {
    fn drop(&mut self) {
        if let Some(value) = self.0.take().and_then(Arc::into_inner) {
            drop(value);
        }
    }
}

impl From<WorkingMemoryReservation> for InferenceRequest {
    fn from(reservation: WorkingMemoryReservation) -> Self {
        Self {
            authority: RequestOwner::new(reservation),
            preparation: None,
        }
    }
}
impl From<&WorkingMemoryReservation> for InferenceRequest {
    fn from(reservation: &WorkingMemoryReservation) -> Self {
        reservation.clone().into()
    }
}
impl From<&InferenceRequest> for InferenceRequest {
    fn from(request: &InferenceRequest) -> Self {
        request.clone()
    }
}

impl InferenceRequest {
    /// Exact selected standard-library control layout for one ordinary request.
    /// This is a logical construction fact for ordinary copy planning, not an
    /// original reservation, allocator overhead estimate, or native-work grant.
    pub fn control_bytes() -> Option<u64> {
        control_mutex::require_known_layout().ok()?;
        let (layout, _) = std::alloc::Layout::new::<[std::sync::atomic::AtomicUsize; 2]>()
            .extend(std::alloc::Layout::new::<WorkingMemoryReservation>())
            .ok()?;
        u64::try_from(layout.pad_to_align().size()).ok()
    }

    /// Exact inference geometry retained by this authority.
    pub fn geometry(&self) -> InferenceGeometry {
        self.authority.geometry()
    }

    /// Reserved admission evidence retained through execution and completion.
    /// Converted evidence requires separate live funding for native allocation;
    /// presence alone does not prove that workspace remains charged.
    pub fn memory_reservation(&self) -> &WorkingMemoryReservation {
        &self.authority
    }

    /// Whether this request's reservation requires separate live execution
    /// funding. The account remains subject to the ledger in every limit mode.
    pub fn requires_funding_scope(&self) -> bool {
        self.memory_reservation().requires_funding_scope()
    }

    /// Checks the retained executable and geometry before any native work.
    pub fn validate(
        &self,
        execution: &InferenceExecutionIdentity,
        geometry: InferenceGeometry,
    ) -> Result<(), WorkingMemoryError> {
        self.authority.validate(execution, geometry)
    }

    /// Requires the exact request, including its charge owner. Equal geometry
    /// cannot replace a request with differently charged work.
    pub fn validate_same_request(&self, other: &Self) -> Result<(), WorkingMemoryError> {
        let same = self
            .memory_reservation()
            .0
            .same(&other.memory_reservation().0);
        if same {
            Ok(())
        } else {
            Err(WorkingMemoryError::IdentityMismatch)
        }
    }

    fn start_state(&self) -> &ControlMutex<text_preparation::RequestStart> {
        &self.memory_reservation().0.start
    }
}

pub use storage::{
    CapturePlanPublicationCause, CapturePlanStorageKey, PreparedCapturePlanPublication,
};

pub(crate) mod resident_reset;
pub use resident_reset::{
    HostSlotSource, OriginalResidentResetSource, PreparedParameterStateReset,
    PreparedResidentEmptyState, PreparedResidentKvReset, ResidentEmptyStateError,
    ResidentResetDisplaced, ResidentResetError, ResidentResetInstallation, ResidentResetLayer,
    ResidentResetProjection, ResidentResetPublicationCustody, ResidentResetPublicationProfile,
    ResidentResetSession, ResidentResetSource, ResidentTableResetState,
    UnquotedOriginalSlotSources,
};

mod original_chat;
mod original_file;
pub use original_chat::{
    ControllerCompilationOutput, ControllerCompilationSources, OriginalChatBackend,
    OriginalChatConsumer, OriginalChatConsumerError, OriginalChatFileError,
    OriginalChatProfileError, OriginalChatProfilePreparation, OriginalChatRenderError,
    OriginalChatRenderOperationError, OriginalChatSourceError, OriginalChatTemplate,
    OriginalChatTemplateError, OriginalControllerCompilation, OriginalControllerCompilationError,
    OriginalControllerCompiler, OriginalRenderedChat,
};

mod original_tokenizer;
pub use original_tokenizer::{
    OriginalEncodedTokenIds, OriginalTextSourceBudget, OriginalTextSourceBudgetError,
    OriginalTextSourceError, OriginalTokenizer, OriginalTokenizerBackend,
    OriginalTokenizerEncodeError, OriginalTokenizerError, OriginalTokenizerInput,
    OriginalTokenizerInputError, OriginalTokenizerPrefixError, OriginalTokenizerSourceError,
};

mod original_composite_semantics;
pub use original_composite_semantics::{
    BoundCompositeSemanticStorage, CompositeChatProjection, CompositeGeneratedText,
    CompositeSemanticCoordinates, CompositeSemanticDiagnostic, CompositeSemanticPartRecord,
    CompositeSemanticRole, CopiedMediaStateBinding, MediaSessionBinding,
    OriginalCompositeSemanticStorage, OriginalCompositeSemanticStorageError,
    PreparedCompositeSemanticBuilder, PreparedCompositeSemanticLayout,
    PreparedCompositeSemanticRecipe,
};

mod shared_native_initialization;
pub use shared_native_initialization::*;

mod report_metadata;
pub use report_metadata::{WorkspaceReportError, WorkspaceReportMetadata};
pub use residual::WorkspaceCopyCompositionError;

pub use residual::{
    NativeEquationStorage, NativePrefillEnvelope, NativePrefillEnvelopeBuilder,
    NativeStorageCarryoverReport,
};

mod prepared_semantic;
pub use prepared_semantic::{
    OriginalSpeculativeHostError, PreparedSemanticSource, PreparedSemanticState,
};
mod decoder_transition;

mod host_source_custody;
pub use host_source_custody::OriginalHostSourceCustody;

mod original_declaration_source;
mod original_intervention_source;
pub use original_intervention_source::{
    OriginalInterventionSource, OriginalInterventionSourceError,
};
// Closed fixed summary destinations share the existing original capture account.
pub use capture_run::{
    CaptureHistogramClaim, CaptureHistogramFailure, CaptureHistogramHostPlan, CaptureHostF32,
    CaptureSummaryClaim, CaptureSummaryFailure, CaptureSummaryHostPlan, ClaimedCaptureHistogram,
    ClaimedCaptureSummary, ScheduledCaptureHistogram, ScheduledCaptureHistogramTransfer,
    ScheduledCaptureSummaryTransfer,
};
mod original_forbidden_source;
pub use original_forbidden_source::{OriginalForbiddenSource, OriginalForbiddenSourceError};
mod original_capture_source;
pub use original_capture_source::{OriginalCaptureSource, OriginalCaptureSourceError};

mod communication_preparation;
pub use communication_preparation::{
    CommunicationPreparationCustody, CommunicationPreparationError,
    CommunicationPreparationProducer, PreparedCommunicationPreparation,
    RetiredCommunicationPreparationError,
};

mod original_semantic_channels;
pub use original_semantic_channels::{
    OriginalSemanticChannelSource, OriginalSemanticChannelSourceError,
    OriginalSemanticControllerSource, OriginalToolValidation,
};

mod semantic_channel_parser;
pub use semantic_channel_parser::{
    OriginalSemanticChannelParser, OriginalSemanticChannelParserError,
};

mod original_token_trie;
pub use original_token_trie::{OriginalTokenTrieSource, OriginalTokenTrieSourceError};

pub use capture_run::{
    CapturePartitionRoutedClaim, CapturePartitionRoutedHostPlan, CapturePartitionRoutedTransfer,
    CapturePartitionRoutedWriter, CaptureRoutedBatchTransfer, CaptureRoutedBatchWriter,
    CaptureRoutedClaim, CaptureRoutedFailure, CaptureRoutedHostError, CaptureRoutedHostPlan,
    CaptureRoutedModelTransfer, CaptureRoutedPrefillTransfer, CaptureRoutedPrefillWriter,
    ClaimedCaptureRoutedUnits, ClaimedPartitionRoutedUnits, PartitionRoutedCaptureFailure,
    PreparedPartitionRoutedCapture, ScheduledCaptureRoutedTransfer, ScheduledCaptureRoutedUnits,
};

pub use capture_run::{
    NativePartitionFragmentDestination, OwnedPartitionFragmentHostPlan,
    PartitionFragmentDestination, PartitionFragmentDestinationError,
    PartitionFragmentHostBindingError, PartitionFragmentHostPlan,
    PartitionFragmentHostPreparationError, PartitionFragmentValue,
    PreparedPartitionFragmentDestinations, PreparedPartitionFragmentHostFunding,
};

pub(crate) use capture_run::{
    PartitionCaptureHookContinuation, PartitionCaptureHookReturnError, PartitionLocalCaptureHook,
};

pub(crate) use capture_run::{PartitionCaptureRankSource, PreparedPartitionFragmentDelivery};

mod original_json_tree;
pub use original_json_tree::{
    OriginalJsonChildren, OriginalJsonNode, OriginalJsonNumber, OriginalJsonTree,
    OriginalJsonTreeError,
};
mod original_json_value;
pub use original_json_value::{OriginalJsonValue, OriginalJsonValueError, OriginalJsonValueKind};

mod original_json_object;
pub use original_json_object::{OriginalJsonField, OriginalJsonObject, OriginalJsonObjectError};
