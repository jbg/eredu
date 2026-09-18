//! Request-bound reservations for shared inference working memory.
mod realtime_frame;
pub use realtime_frame::{RealtimeFrameRequirements,RealtimeNativeRequirements,RealtimeFrameAdmissionError,
    OriginalRealtimeFrame,OriginalRealtimeNative,OriginalRealtimeBudgetCustody};
mod control_mutex;
mod operation_custody;
pub use operation_custody::OriginalOperationMetadataCustody;
mod fixed_baseline;
mod original_prepared_input;
mod original_request;
mod qualified_storage;
mod reservation_metadata;
mod speculative;
mod thread_startup;
mod workspace_planning;
pub use workspace_planning::SessionResetPreparationFunding;
use control_mutex::ControlMutex;
pub use original_prepared_input::{OriginalPreparedHostInput, OriginalPreparedHostInputError};
pub(crate) use qualified_storage::shared_bytes as qualified_shared_bytes;
pub use reservation_metadata::WorkspaceReservationMetadataError;
pub use speculative::{
    OriginalEmbeddedCaptureLineage, OriginalEmbeddedSpeculativeRole,
    OriginalEmbeddedSpeculativeSource, OriginalEmbeddedSpeculativeStartup,
    OriginalExternalSpeculativeRole, OriginalExternalSpeculativeSource,
    OriginalExternalSpeculativeStartup, OriginalSpeculativeBudgetCustody,
    OriginalSpeculativeNumericalBudgetCustody, OriginalSpeculativeNumericalPhase,
    OriginalSpeculativePrefillSpan, OriginalSpeculativeRegisteredSource,
    OriginalSpeculativeRequest, OriginalSpeculativeRole, OriginalSpeculativeSourceIdentity,
    OriginalSpeculativeStartup, SpeculativeContinuationError, SpeculativeInvocationRequirements,
    SpeculativeNumericalAdmissionError, SpeculativeNumericalRequirements,
    SpeculativeNumericalSource, SpeculativeRequestError,
};
pub use thread_startup::{HostThreadStartup, HostThreadStartupPlan};
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

mod original_bpe_model;
mod original_stop_source;
pub use original_bpe_model::{OriginalBpeModel, OriginalBpeModelError};
pub use original_stop_source::{
    OriginalStopSource, OriginalStopSourceBackend, OriginalStopSourceError,
};

mod gguf_catalog;
mod gguf_composite;
pub use gguf_composite::OriginalGgufCompositeError;
mod source_erasure;
pub use source_erasure::OriginalRetainedSourceError;
mod gguf_source;
pub use gguf_catalog::{OriginalGgufCatalog, OriginalGgufCatalogError};
pub use gguf_source::{GgufSourceStorageKey, OriginalGgufSourceError};

mod loaded_decode_source;
pub use loaded_decode_source::{
    LoadedDecodeSource, LoadedDecodeSourceBackend, LoadedDecodeSourceError,
};

mod capture_run;
pub(crate) use capture_run::PartitionInterventionEvidenceFrame;
pub use capture_run::{
    CaptureCandidateClaim, CaptureCandidateFailure, CaptureCandidateHostPlan,
    CaptureInterventionClaim, RoutedInterventionCursor, RoutedInterventionBatch, InterventionPrefillCursor, InterventionPrefillFragment, CaptureInterventionEvidenceClaim, CaptureInterventionEvidenceKind,
    CapturePrefillFragmentClaim, CapturePrefillFragmentTransfer, CapturePrefillFragmentWriter,
    CapturePrefillHostError, CapturePrefillSourceBootstrap, CaptureRunHostError,
    CaptureRunHostPlan, CaptureStepClaim, CaptureTensorClaim, CaptureTokenScoreClaim,
    CaptureTokenScoreFailure, CaptureTokenScoreHostPlan, ClaimedCaptureCandidates,
    ClaimedCaptureTensor, ClaimedCaptureTokenScores, ClaimedIntervention,
    ClaimedInterventionEvidence, EmbeddedCaptureHostPlan, EmbeddedCapturePreparationError,
    InterventionEvidenceReceipt, PartitionCaptureTensorDecodeError,
    PartitionCaptureTensorDeliveryError, PartitionCaptureTensorReceipt, PreparedCaptureRun,
    PreparedEmbeddedCapture, PreparedPartitionTensorDelivery, ScheduledCaptureCandidates,
    ScheduledCaptureCandidatesTransfer, ScheduledCaptureStep, ScheduledCaptureStepFinishError,
    ScheduledCaptureTensor, ScheduledCaptureTensorFailure, ScheduledCaptureTensorFinishError,
    ScheduledCaptureTensorTransfer, ScheduledCaptureTensorTransferFinishError,
    ScheduledCaptureTokenScores, ScheduledCaptureTokenScoresTransfer, SpeculativeCaptureHostPlan,
    SpeculativeCapturePreparationError,
};
pub(crate) use capture_run::{CaptureRunLedger, CaptureRunLedgerGuard};

mod text_preparation;
pub use text_preparation::{
    PendingSamplingExtension, PendingTextBranchExchange,
    InferencePreparationStage, InferencePromptCompletion, InferenceSamplerCompletion,
    InferenceTextPreparation, InferenceTextStep, InferenceTextStepReceipt,
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
    RetainedOriginalStorageSources, WorkingMemoryStorage,
};
pub use storage::{PreparedWorkspaceCopyPublication, WorkspaceCopyPublicationPlan};
mod funding;
pub use funding::{
    PreparedWorkingMemoryFundingScope, WorkingMemoryCapacityHandoff, WorkingMemoryFundingRun,
    WorkingMemoryFundingScope, WorkingMemorySamplerScope,
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
    PreparedControllerBinding, PreparedControllerBindingError,
    ControllerWorkspaceEstimate, ControllerWorkspaceMetadataError, RegisteredControllerStorage,
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
    SamplingWorkspaceObserver, SamplingWorkspacePhase, SamplingWorkspaceReport, SamplingWorkspaceInputPlan,
    WorkspaceSamplingBackend, WorkspaceSamplingInput, WorkspaceSamplingRandomState,
    WorkspaceSamplingSource, quote_sampling_workspace, quote_sampling_workspace_with_observer,
};
mod trace;
pub use trace::with_equation_workspace;
mod inference;
pub use inference::{
    InferenceObservationError, InferenceResidualWorkspace, InferenceSpanWorkspacePlan, SamplingWorkspacePlanCollector,
    InferenceSpanWorkspaceRecord, InferenceWorkspaceError, InferenceWorkspaceObserver,
    InferenceWorkspaceReport, InferenceWorkspaceSpan, quote_inference_workspace,
    quote_inference_workspace_with_context, quote_inference_workspace_with_report_owner,
};
mod residual;
pub use residual::{
    AdmittedCaptureContinuation, AdmittedPrefillCapture, AggregateGenerationDecoderInput,
    SamplingExtensionQuote, OriginalTextSamplingExtension,
    CopyPreparationInferenceQuote, FailedCapturePlanPublication, GraphMetadataFacts,
    HostDestinationCause, HostDestinationFacts, HostSourceConstructionFacts, HostSourceConstructionProgram, OriginalHostSourceProgramBanks, OriginalHostSourceProgramError, IncompleteWorkspace,
    IncrementalInferenceQuote, InferenceSpanWorkspace, LoadedGenerationDecoderInput,
    NativeStorageError, NativeStorageObservation, NativeStorageRegistration,
    NativeStorageSelection, OriginalGenerationDecoderInput, OriginalGenerationDecoderSource,
    OriginalGenerationSequenceBank, OriginalGraphMetadata, OriginalHostDestinationBank,
    OriginalHostMetadataCustody, OriginalHostMetadataVec, OriginalHostSourceBank,
    OriginalHostSourceConstruction, OriginalHostSourceError, OriginalHostSourceFailure,
    OriginalHostSourceFailureCause, OriginalHostSourceReceipt, OriginalHostSourceRefusal,
    OriginalHostVec, OriginalHostVecError, OriginalNativeBudgetCustody, OriginalNativePublication,
    OriginalNativeStorageBank, OriginalNativeStorageMechanism, OriginalPredictionNativeCustody,
    OriginalPredictionRecoveryCustody, OriginalPredictionScopeRole, OriginalPrefillNativeCustody,
    OriginalPrefillRecoveryCustody, OriginalPrefillRootCustody,
    OriginalPrefillRootProjectionCustody, OriginalPrefillScopeRole,
    OriginalPreparationScopeCustody, OriginalSubmissionTracking, OriginalTextControlGuard,
    OriginalTextMetadataCustody, OriginalTextPredictionScopeSet, OriginalTextPredictionScopes,
    OriginalTextPrefillScopeSet, OriginalTextPrefillScopes, OriginalTextPreparationScopes,
    OriginalTokenInputBank, OriginalTokenInputFailure, OriginalTokenInputLayout,
    OwnedInferenceSpanWorkspace, OwnedPromptTokenIds, OwnedTextSpanWorkspace,
    PendingCapturePlanPublication, PreparedNativeStoragePlan, PreparedTextControlWorkspace,
    RegisteredInferenceSourceWitness, RegisteredPreparedWorkspaceStorage,
    RegisteredWorkspaceStorage, RegisteredWorkspaceStorageLayout, ReservedInferenceSpanWorkspace,
    ReservedTextSpanWorkspace, ResidualInferenceQuote, ResidualQuoteError, SpanWorkspaceOwnerError,
    SubmissionTrackingFacts, TextHostControlFacts, TextPredictionScopeFacts, TextPrefillScopeFacts,
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

use eredu_core::{Admission, EstimationCompleteness, InferenceGeometry};
use std::sync::{Arc, Mutex};

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
    pool: &WorkingMemoryPool,
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

/// Selects a chunk while also enforcing this request's ceiling on the complete
/// shared managed domain. Existing residency and concurrent reservations count
/// against it. The successful reservation keeps this ceiling in force until all
/// completion and retained-state owners release it; later admissions cannot
/// weaken an earlier live ceiling by requesting a larger budget.
pub fn plan_prefill_with_capacity(
    execution: &InferenceExecutionIdentity,
    pool: &WorkingMemoryPool,
    capabilities: &eredu_core::ModelCapabilities,
    request: eredu_core::AdmissionRequest,
    geometry: InferenceGeometry,
    capacity: u64,
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
    pool: &WorkingMemoryPool,
    capabilities: &eredu_core::ModelCapabilities,
    request: eredu_core::AdmissionRequest,
    geometry: InferenceGeometry,
    capacity: Option<u64>,
    mut quote: impl FnMut(
        InferenceGeometry,
    ) -> Result<eredu_core::RuntimeStateEstimate, eredu_core::CapabilityError>,
) -> Result<(Admission, WorkingMemoryReservation), PrefillPlanningError> {
    plan_prefill_candidates(
        capabilities,
        request,
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
            match eredu_core::apply_admission_policy(capabilities, request, state, None)? {
                eredu_core::AdmissionResult::Admitted(admission) => {
                    let reservation = pool.reserve_limited(execution, &admission, capacity)?;
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
            WorkingMemoryError::BudgetExceeded { .. }
            | WorkingMemoryError::SubmissionTrackingCapacity { .. }
            | WorkingMemoryError::GraphMetadataCapacity { .. },
        )
        | PrefillPlanningError::Admission(
            eredu_core::AdmissionRejection::MemoryBudgetExceeded { .. }
            | eredu_core::AdmissionRejection::EstimationUnsupported { .. },
        ) => true,
        PrefillPlanningError::Reservation(WorkingMemoryError::ReservationMetadata(error)) =>
            error.planning_error().is_some_and(candidate_can_shrink),
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
    /// Shared reservations would exceed the managed capacity.
    #[error("inference needs {required_bytes} bytes; {available_bytes} managed bytes remain")]
    BudgetExceeded {
        /// Incremental request reservation.
        required_bytes: u64,
        /// Capacity after existing residency and other live reservations.
        available_bytes: u64,
    },
    /// A new domain ceiling is already below existing live managed storage.
    #[error(
        "managed memory already uses {used_bytes} bytes, above the requested {capacity_bytes}-byte domain capacity"
    )]
    CapacityBelowUsage {
        /// Complete requested domain ceiling, including existing residency.
        capacity_bytes: u64,
        /// Registered residency and reservations already alive in the domain.
        used_bytes: u64,
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
    reset_entries: Option<Box<resident_reset::Entry>>,
    reset_pending: Option<resident_reset::Pending>,
    reset_retiring_capacity: u64,
    reset_retiring_count: usize,
    reserved: u64,
    reservations: usize,
    unquoted_owners: usize,
    registered: u64,
    storage: storage::directory::Directory,
    peak: u64,
    funding: funding::AccountLedger,
    account_retiring: Option<funding::RetiringAccount>,
    pending_original: Option<funding::PendingOriginal>,
    next_funding: u64,
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
            .field("active_capacity_limit", &self.funding.capacity())
            .field("funding_accounts", &self.funding.len())
            .finish()
    }
}

#[derive(Debug)]
struct Pool {
    capacity: u64,
    existing: u64,
    storage_domain: eredu_core::SharedStorageDomain,
    usage: Mutex<Usage>,
}

impl Pool {
    fn capacity(&self, usage: &Usage, requested: Option<u64>) -> u64 {
        self.capacity
            .min(resident_reset::capacity(usage))
            .min(
                usage.funding.capacity().min(
                    usage
                        .account_retiring
                        .as_ref()
                        .map_or(u64::MAX, funding::RetiringAccount::capacity),
                ),
            )
            .min(
                usage
                    .pending_original
                    .as_ref()
                    .map_or(u64::MAX, funding::PendingOriginal::capacity),
            )
            .min(requested.unwrap_or(u64::MAX))
    }

    fn available(&self, usage: &Usage, requested: Option<u64>) -> Result<u64, WorkingMemoryError> {
        let capacity_bytes = self.capacity(usage, requested);
        let used_bytes = self
            .existing
            .checked_add(usage.registered)
            .and_then(|bytes| bytes.checked_add(usage.reserved))
            .ok_or(WorkingMemoryError::Overflow)?;
        capacity_bytes
            .checked_sub(used_bytes)
            .ok_or(WorkingMemoryError::CapacityBelowUsage {
                capacity_bytes,
                used_bytes,
            })
    }
}

// Shared numeric admission only. Each caller first proves its own concrete
// operation; this private plan cannot turn an arbitrary byte count into a public
// inference or copy permission.
struct PreparedAccountCommit<'h> {
    ceilings: funding::CapacityHandoffPlan<'h>,
    reservations: usize,
    reserved: u64,
    peak: u64,
}

impl<'h> PreparedAccountCommit<'h> {
    fn prepare(
        pool: &WorkingMemoryPool,
        execution: &InferenceExecutionIdentity,
        usage: &Usage,
        required: u64,
        capacity: Option<u64>,
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
        let used = pool
            .0
            .existing
            .checked_add(usage.registered)
            .and_then(|bytes| bytes.checked_add(usage.reserved))
            .ok_or(WorkingMemoryError::Overflow)?;
        let capacity_bytes = ceilings.effective_capacity(pool.0.capacity, capacity);
        let available =
            capacity_bytes
                .checked_sub(used)
                .ok_or(WorkingMemoryError::CapacityBelowUsage {
                    capacity_bytes,
                    used_bytes: used,
                })?;
        if required > available {
            return Err(WorkingMemoryError::BudgetExceeded {
                required_bytes: required,
                available_bytes: available,
            });
        }
        Ok(Self {
            ceilings,
            reservations: usage
                .reservations
                .checked_add(1)
                .ok_or(WorkingMemoryError::Overflow)?,
            reserved: usage
                .reserved
                .checked_add(required)
                .ok_or(WorkingMemoryError::Overflow)?,
            peak: usage.peak.max(
                used.checked_add(required)
                    .ok_or(WorkingMemoryError::Overflow)?,
            ),
        })
    }

    fn commit(self, usage: &mut Usage) {
        self.ceilings.commit(usage);
        usage.reservations = self.reservations;
        usage.reserved = self.reserved;
        usage.peak = self.peak;
    }
}

/// One physical managed-memory domain. All sessions sharing that domain must
/// share this owner. Unified host/device aliases count once. Use storage
/// registrations for changing retained inventories; `existing` is a fixed,
/// disjoint baseline whose owners live for the entire pool lifetime.
/// Owners without a complete bound must hold an unquoted lease, excluding
/// request reservations until their work settles and storage is accounted for.
/// This bounds Eredu-managed allocations, not other process/system allocations.
#[derive(Debug, Clone)]
pub struct WorkingMemoryPool(Arc<Pool>);

impl WorkingMemoryPool {
    /// Payload-free identity used to attach existing immutable storage to this
    /// domain. It grants no allocation, registration or execution permission.
    pub fn shared_storage_domain(&self) -> &eredu_core::SharedStorageDomain {
        &self.0.storage_domain
    }

    /// Whether two handles account for the same managed resource domain.
    /// This compares identity only and performs no accounting or native work.
    pub fn same_domain(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }

    /// Reserves already resident managed allocations before admitting requests.
    pub fn new(capacity: u64, existing: u64) -> Result<Self, WorkingMemoryError> {
        if existing > capacity {
            return Err(WorkingMemoryError::BudgetExceeded {
                required_bytes: existing,
                available_bytes: capacity,
            });
        }
        Ok(Self(Arc::new(Pool {
            capacity,
            existing,
            storage_domain: eredu_core::SharedStorageDomain::default(),
            usage: Mutex::new(Usage {
                reset_entries: None,
                reset_pending: None,
                reset_retiring_capacity: u64::MAX,
                reset_retiring_count: 0,
                reserved: 0,
                reservations: 0,
                unquoted_owners: 0,
                registered: 0,
                storage: Default::default(),
                peak: existing,
                account_retiring: None,
                pending_original: None,
                funding: Default::default(),
                next_funding: 0,
            }),
        })))
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

    /// Atomically reserves complete request work under a ceiling on all managed
    /// bytes in this domain, including the baseline, registered storage and other
    /// requests. Clones retain one ceiling; the last charge owner releases it.
    /// `AdmissionRequest::application_memory_budget_bytes` remains the separate
    /// incremental request limit applied by core admission.
    pub fn reserve_with_capacity(
        &self,
        execution: &InferenceExecutionIdentity,
        admission: &Admission,
        capacity: u64,
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
        let mut usage = self
            .0
            .usage
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)?;
        if usage.reservations != 0 {
            return Err(WorkingMemoryError::ReservedWorkActive);
        }
        usage.unquoted_owners = usage
            .unquoted_owners
            .checked_add(1)
            .ok_or(WorkingMemoryError::Overflow)?;
        Ok(WorkingMemoryUnquotedLease(Arc::new(UnquotedOwner {
            pool: self.clone(),
        })))
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
        capacity: Option<u64>,
    ) -> Result<WorkingMemoryReservation, WorkingMemoryError> {
        self.reserve_limited_with_minimum(execution, admission, capacity, None)
    }

    fn reserve_limited_with_minimum(
        &self,
        execution: &InferenceExecutionIdentity,
        admission: &Admission,
        capacity: Option<u64>,
        residual: Option<(u64, residual::RegisteredStoragePin)>,
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
        capacity: Option<u64>,
        residual: Option<(u64, residual::RegisteredStoragePin)>,
        handoffs: &[WorkingMemoryCapacityHandoff],
    ) -> Result<WorkingMemoryReservation, WorkingMemoryError> {
        self.reserve_limited_with_source(execution, admission, capacity, residual, handoffs, None)
    }

    fn reserve_limited_with_source(
        &self,
        execution: &InferenceExecutionIdentity,
        admission: &Admission,
        capacity: Option<u64>,
        residual: Option<(u64, residual::RegisteredStoragePin)>,
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
        capacity: Option<u64>,
        residual: Option<(u64, residual::RegisteredStoragePin)>,
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
        let peak = workspace
            .peak_bytes()
            .map_err(|_| WorkingMemoryError::Overflow)?
            .ok_or(WorkingMemoryError::UnknownBound)?;
        if state.completeness == EstimationCompleteness::PersistentStateOnly
            || state.persistent_state_completeness == EstimationCompleteness::PersistentStateOnly
            || state
                .selected_state_backing
                .as_ref()
                .is_some_and(|backing| backing.bytes().is_none())
        {
            return Err(WorkingMemoryError::UnknownBound);
        }
        let minimum = state
            .requested_state_bytes
            .checked_add(peak)
            .ok_or(WorkingMemoryError::Overflow)?;
        let geometry = workspace.geometry;
        if state
            .selected_state_backing
            .as_ref()
            .is_some_and(|backing| backing.geometry != geometry)
        {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let minimum = residual.as_ref().map_or(minimum, |(bytes, _)| *bytes);
        let requested_positions = geometry
            .cached_positions
            .checked_add(geometry.input_positions)
            .and_then(|positions| positions.checked_add(geometry.max_output_tokens))
            .ok_or(WorkingMemoryError::Overflow)?;
        if admission.incremental_required_bytes < minimum
            || geometry.batch_size != state.assumptions.batch_size
            || requested_positions != admission.requested_positions
            || admission.requested_positions != state.assumptions.requested_positions
        {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        // Stage descriptive metadata before locking or publishing accounting.
        // It contains no numerical owners and cannot keep credited storage live.
        let metadata = planning_metadata.as_ref().map_or_else(
            WorkspaceReportMetadata::ordinary,
            WorkspaceReportMetadata::with_funding,
        );
        if let Some(funding) = &planning_metadata {
            funding
                .reserve_metadata(reservation_metadata::constructor_bytes()?)
                .map_err(reservation_metadata::funding_error)?;
        }
        let retained_admission = match &planning_metadata {
            Some(funding) => metadata.clone_admission(admission).map_err(|cause| {
                reservation_metadata::neural_error(metadata.error(cause), funding)
            })?,
            None => admission.clone(),
        };
        // These pins do not reduce the full bound. Stage ownership before the
        // usage lock so attachment/registration destructors cannot reenter it.
        let borrowed_storage = match (residual.map(|(_, pin)| pin), source) {
            (Some(pin), Some(source)) => Some(match &planning_metadata {
                Some(funding) => {
                    residual::RegisteredStoragePin::pair_metadata(pin, source.pin(), metadata)
                        .map_err(|cause| reservation_metadata::neural_error(cause, funding))?
                }
                None => residual::RegisteredStoragePin::aggregate([pin, source.pin()]),
            }),
            (Some(pin), None) => Some(pin),
            (None, Some(source)) => Some(source.pin()),
            (None, None) => None,
        };
        let node = funding::AccountNode::empty_with_planning(planning_metadata.clone());
        let mut usage = self
            .0
            .usage
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)?;
        if let Some(source) = source {
            source.validate(self, &usage)?;
        }
        let commit = PreparedAccountCommit::prepare(
            self,
            execution,
            &usage,
            admission.incremental_required_bytes,
            capacity,
            handoffs,
        )?;
        let account_id = usage.next_funding;
        let next = account_id
            .checked_add(1)
            .ok_or(WorkingMemoryError::Overflow)?;
        commit.commit(&mut usage);
        usage.funding.publish(
            node,
            account_id,
            funding::FundingState::new(
                admission.incremental_required_bytes,
                capacity,
                execution,
                true,
                0,
                0,
            ),
            false,
        );
        usage.next_funding = next;
        drop(usage);
        Ok(WorkingMemoryReservation(ReservationOwner::new(
            Reservation {
                account_id,
                pool: self.clone(),
                execution: execution.clone(),
                admission: retained_admission,
                geometry,
                bytes: admission.incremental_required_bytes,
                capacity,
                funding: None,
                borrowed_storage,
                span_workspace: None,
                start: ControlMutex::new(text_preparation::RequestStart::Fresh),
                planning_metadata,
            },
        )))
    }

    /// Tightest configured or live request ceiling. This is policy, not a native
    /// allocator observation. Admission rechecks it atomically with reservation.
    pub fn effective_capacity(&self) -> Result<u64, WorkingMemoryError> {
        let usage = self
            .0
            .usage
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)?;
        Ok(self.0.capacity(&usage, None))
    }

    /// Current known managed charge, including unique registered storage,
    /// existing residency and reserved request bounds. Unquoted resources are
    /// excluded; consult `unquoted_owner_count` before treating this as complete.
    pub fn used_bytes(&self) -> Result<u64, WorkingMemoryError> {
        let usage = self
            .0
            .usage
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)?;
        Ok(self.0.existing + usage.registered + usage.reserved)
    }

    /// Maximum known managed charge since pool creation; never rewound.
    /// Unquoted resources are excluded, including earlier unquoted work whose
    /// lease has since retired. This is not an observed allocation peak.
    pub fn peak_bytes(&self) -> Result<u64, WorkingMemoryError> {
        Ok(self
            .0
            .usage
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)?
            .peak)
    }
}

#[derive(Debug)]
struct UnquotedOwner {
    pool: WorkingMemoryPool,
}

impl Drop for UnquotedOwner {
    fn drop(&mut self) {
        // Poison remains sticky for new work; existing owners still retire.
        let mut usage = self.pool.0.usage.lock().unwrap_or_else(|p| p.into_inner());
        usage.unquoted_owners -= 1;
    }
}

/// Shared ownership of one unquoted domain participant. Clones preserve the
/// same admission blocker until the final owner retires. This lease carries no
/// byte bound and must not substitute for a strict request reservation.
#[derive(Debug, Clone)]
#[must_use = "retain the lease until native work settles and its storage is accounted for or retired"]
pub struct WorkingMemoryUnquotedLease(#[allow(dead_code)] Arc<UnquotedOwner>);

#[cfg(test)]
mod unquoted_tests {
    use super::*;

    #[test]
    fn unquoted_owner_overflow_preserves_accounting() {
        let pool = WorkingMemoryPool::new(100, 40).unwrap();
        pool.0.usage.lock().unwrap().unquoted_owners = usize::MAX;
        assert!(matches!(
            pool.acquire_unquoted(),
            Err(WorkingMemoryError::Overflow)
        ));
        assert_eq!(pool.unquoted_owner_count().unwrap(), usize::MAX);
        assert_eq!(pool.used_bytes().unwrap(), 40);
        assert_eq!(pool.peak_bytes().unwrap(), 40);
        assert_eq!(pool.effective_capacity().unwrap(), 100);
        let usage = pool.0.usage.lock().unwrap();
        assert_eq!(usage.reservations, 0);
        assert_eq!(usage.funding.capacity(), u64::MAX);
    }

    #[test]
    fn poison_rejects_new_unquoted_work_but_retirement_still_releases_owners() {
        let pool = WorkingMemoryPool::new(100, 40).unwrap();
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
        assert_eq!(pool.used_bytes(), Err(WorkingMemoryError::Poisoned));
        assert_eq!(pool.peak_bytes(), Err(WorkingMemoryError::Poisoned));
        drop(owner);
        {
            let usage = pool.0.usage.lock().unwrap_err().into_inner();
            assert_eq!(usage.unquoted_owners, 1);
        }
        drop(completion);
        let usage = pool.0.usage.lock().unwrap_err().into_inner();
        assert_eq!(usage.unquoted_owners, 0);
        assert_eq!(usage.reserved, 0);
        assert_eq!(usage.peak, 40);
    }
}

#[derive(Debug)]
struct Reservation {
    account_id: u64,
    pool: WorkingMemoryPool,
    execution: InferenceExecutionIdentity,
    admission: Admission,
    geometry: InferenceGeometry,
    bytes: u64,
    capacity: Option<u64>,
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
            bytes: _,
            capacity: _,
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
            pool: WorkingMemoryPool,
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

    /// Checks the accounting domain independently of executable and geometry.
    /// A retained reservation is not permission to allocate in another pool.
    pub fn validate_domain(&self, pool: &WorkingMemoryPool) -> Result<(), WorkingMemoryError> {
        if self.0.pool.same_domain(pool) {
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
    pub fn bytes(&self) -> u64 {
        self.0.bytes
    }
}

/// Request authority shared by budgeted and explicitly unbudgeted execution.
/// Both modes bind geometry and consume one prefill start. Only `Reserved`
/// storage carries an enforceable memory charge; missing bounds must never be
/// converted to this unbudgeted mode to satisfy a strict/budgeted admission.
#[derive(Debug, Clone)]
pub struct InferenceRequest {
    // Retire the outer preparation alias before the final request/reservation.
    preparation: Option<Arc<text_preparation::TextPreparationAuthority>>,
    authority: RequestOwner,
}

#[derive(Debug)]
enum RequestAuthority {
    Reserved(WorkingMemoryReservation),
    Unbudgeted {
        execution: InferenceExecutionIdentity,
        geometry: InferenceGeometry,
        start: ControlMutex<text_preparation::RequestStart>,
    },
}

#[derive(Debug)]
struct RequestOwner(Option<Arc<RequestAuthority>>);
impl RequestOwner {
    fn new(value: RequestAuthority) -> Self {
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
    type Target = RequestAuthority;
    fn deref(&self) -> &RequestAuthority {
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
            authority: RequestOwner::new(RequestAuthority::Reserved(reservation)),
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
    pub fn unbudgeted_control_bytes() -> Option<u64> {
        control_mutex::require_known_layout().ok()?;
        let (layout, _) = std::alloc::Layout::new::<[std::sync::atomic::AtomicUsize; 2]>()
            .extend(std::alloc::Layout::new::<RequestAuthority>())
            .ok()?;
        u64::try_from(layout.pad_to_align().size()).ok()
    }

    /// Creates explicit no-budget request authority. This still enables bounded
    /// chunk scheduling/completion, but makes no allocation-bound claim. Callers
    /// requesting strict coverage or any application budget must use a complete
    /// `WorkingMemoryPool::reserve` result instead; there is no automatic fallback.
    pub fn without_memory_budget(
        execution: &InferenceExecutionIdentity,
        geometry: InferenceGeometry,
    ) -> Result<Self, eredu_core::CapabilityError> {
        geometry.validate()?;
        Ok(Self {
            authority: RequestOwner::new(RequestAuthority::Unbudgeted {
                execution: execution.clone(),
                geometry,
                start: ControlMutex::new(text_preparation::RequestStart::Fresh),
            }),
            preparation: None,
        })
    }

    /// Exact inference geometry retained by this authority.
    pub fn geometry(&self) -> InferenceGeometry {
        match &*self.authority {
            RequestAuthority::Reserved(reservation) => reservation.geometry(),
            RequestAuthority::Unbudgeted { geometry, .. } => *geometry,
        }
    }

    /// Reserved admission evidence, absent for explicitly unbudgeted execution.
    /// Converted evidence requires separate live funding for native allocation;
    /// presence alone does not prove that workspace remains charged.
    pub fn memory_reservation(&self) -> Option<&WorkingMemoryReservation> {
        match &*self.authority {
            RequestAuthority::Reserved(reservation) => Some(reservation),
            RequestAuthority::Unbudgeted { .. } => None,
        }
    }

    /// Whether this request's reservation requires separate live execution
    /// funding. False also includes explicitly unbudgeted requests, which still
    /// require their provider's ordinary unquoted-work admission.
    pub fn requires_funding_scope(&self) -> bool {
        self.memory_reservation()
            .is_some_and(WorkingMemoryReservation::requires_funding_scope)
    }

    /// Checks the retained executable and geometry before any native work.
    pub fn validate(
        &self,
        execution: &InferenceExecutionIdentity,
        geometry: InferenceGeometry,
    ) -> Result<(), WorkingMemoryError> {
        match &*self.authority {
            RequestAuthority::Reserved(reservation) => reservation.validate(execution, geometry),
            RequestAuthority::Unbudgeted {
                execution: expected,
                geometry: retained,
                ..
            } if Arc::ptr_eq(&execution.0, &expected.0) && geometry == *retained => Ok(()),
            _ => Err(WorkingMemoryError::IdentityMismatch),
        }
    }

    /// Requires the exact request, including its charge owner. Equal geometry
    /// cannot replace a budgeted request with unbudgeted or differently charged work.
    pub fn validate_same_request(&self, other: &Self) -> Result<(), WorkingMemoryError> {
        let same = match (&*self.authority, &*other.authority) {
            (RequestAuthority::Reserved(a), RequestAuthority::Reserved(b)) => a.0.same(&b.0),
            _ => self.authority.same(&other.authority),
        };
        if same {
            Ok(())
        } else {
            Err(WorkingMemoryError::IdentityMismatch)
        }
    }

    fn start_state(&self) -> &ControlMutex<text_preparation::RequestStart> {
        match &*self.authority {
            RequestAuthority::Reserved(reservation) => &reservation.0.start,
            RequestAuthority::Unbudgeted { start, .. } => start,
        }
    }
}

pub use storage::{
    CapturePlanPublicationCause, CapturePlanStorageKey, PreparedCapturePlanPublication,
};

pub(crate) mod resident_reset;
pub use resident_reset::{
    HostSlotSource, OriginalResidentResetSource, PreparedResidentEmptyState,
    PreparedResidentKvReset, ResidentEmptyStateError, ResidentKvResetLayer,
    ResidentResetDisplaced, ResidentResetError, ResidentResetInstallation, ResidentResetProjection,
    ResidentResetPublicationCustody, ResidentResetPublicationProfile, ResidentResetSession,
    ResidentResetSource, ResidentTableResetState, UnquotedOriginalSlotSources,
};

mod original_chat;
mod original_file;
pub use original_chat::{
    ControllerCompilationOutput, ControllerCompilationSources, OriginalControllerCompiler,
    OriginalControllerCompilation, OriginalControllerCompilationError,
    OriginalChatBackend, OriginalChatFileError, OriginalChatSourceError, OriginalChatRenderOperationError,
    OriginalChatProfileError, OriginalChatProfilePreparation, OriginalChatRenderError,
    OriginalChatTemplate, OriginalChatTemplateError, OriginalRenderedChat, OriginalChatConsumer, OriginalChatConsumerError,
};

mod original_tokenizer;
pub use original_tokenizer::{
    OriginalEncodedTokenIds, OriginalTextSourceBudget, OriginalTextSourceBudgetError,
    OriginalTextSourceError, OriginalTokenizerSourceError, OriginalTokenizer, OriginalTokenizerBackend,
    OriginalTokenizerEncodeError, OriginalTokenizerError, OriginalTokenizerPrefixError, OriginalTokenizerInput, OriginalTokenizerInputError,
};

mod original_composite_semantics;
pub use original_composite_semantics::{
    BoundCompositeSemanticStorage, CompositeSemanticCoordinates, CompositeSemanticDiagnostic,
    CompositeChatProjection, CompositeGeneratedText, CompositeSemanticPartRecord, CompositeSemanticRole, CopiedMediaStateBinding,
    MediaSessionBinding, OriginalCompositeSemanticStorage, OriginalCompositeSemanticStorageError,
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
    OriginalSpeculativeHostError, PreparedSemanticState,
    PreparedSemanticSource,
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
    CaptureHistogramClaim, CaptureHistogramFailure, CaptureHistogramHostPlan, CaptureSummaryClaim,
    CaptureSummaryFailure, CaptureSummaryHostPlan, CaptureHostF32, ClaimedCaptureHistogram, ClaimedCaptureSummary,
    ScheduledCaptureHistogram, ScheduledCaptureHistogramTransfer, ScheduledCaptureSummaryTransfer,
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
    OriginalSemanticChannelSource, OriginalSemanticChannelSourceError, OriginalSemanticControllerSource, OriginalToolValidation,
};

mod semantic_channel_parser;
pub use semantic_channel_parser::{
    OriginalSemanticChannelParser, OriginalSemanticChannelParserError,
};

mod original_token_trie;
pub use original_token_trie::{OriginalTokenTrieSource, OriginalTokenTrieSourceError};

pub use capture_run::{
    CaptureRoutedHostError, CaptureRoutedHostPlan, CaptureRoutedClaim,
    ScheduledCaptureRoutedUnits, ScheduledCaptureRoutedTransfer,
    ClaimedCaptureRoutedUnits, CaptureRoutedFailure, CaptureRoutedPrefillWriter, CaptureRoutedPrefillTransfer,
    CaptureRoutedBatchWriter, CaptureRoutedBatchTransfer, CaptureRoutedModelTransfer,
    CapturePartitionRoutedHostPlan, CapturePartitionRoutedClaim, PreparedPartitionRoutedCapture,
    CapturePartitionRoutedWriter, CapturePartitionRoutedTransfer, ClaimedPartitionRoutedUnits,
    PartitionRoutedCaptureFailure,
};

pub use capture_run::{PartitionFragmentHostPlan, PreparedPartitionFragmentDestinations, PartitionFragmentDestination, NativePartitionFragmentDestination, PartitionFragmentValue, PartitionFragmentDestinationError, PreparedPartitionFragmentHostFunding, PartitionFragmentHostBindingError, PartitionFragmentHostPreparationError};

pub(crate) use capture_run::{PartitionLocalCaptureHook,PartitionCaptureHookContinuation,PartitionCaptureHookReturnError};

pub(crate) use capture_run::{PreparedPartitionFragmentDelivery,PartitionCaptureRankSource};

mod original_json_allocation;
mod original_json_tree;
pub use original_json_tree::{
    OriginalJsonChildren, OriginalJsonNode, OriginalJsonNumber, OriginalJsonTree, OriginalJsonTreeError,
};
mod original_json_value;
pub use original_json_value::{OriginalJsonValue, OriginalJsonValueError, OriginalJsonValueKind};

mod original_json_object;
pub use original_json_object::{OriginalJsonField, OriginalJsonObject, OriginalJsonObjectError};
