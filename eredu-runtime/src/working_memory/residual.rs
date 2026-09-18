//! Residual admission bound to already registered physical storage.

mod copy_preparation;
mod diagnostics;
use diagnostics::QuoteDiagnostics;
mod span_workspace;
pub(in crate::working_memory) use span_workspace::OriginalTokenDomainBinding;
pub(super) use span_workspace::SpanWorkspaceIdentity;
use span_workspace::SpanWorkspaceSeal;
pub(in crate::working_memory) use span_workspace::TextControlBinding;
pub use span_workspace::{
    AdmittedCaptureContinuation, AdmittedPrefillCapture, AggregateGenerationDecoderInput,
    SamplingExtensionQuote, OriginalTextSamplingExtension,
    FailedCapturePlanPublication, GraphMetadataFacts, HostDestinationCause, HostDestinationFacts,
    HostSourceConstructionFacts, HostSourceConstructionProgram, OriginalHostSourceProgramBanks, OriginalHostSourceProgramError, InferenceSpanWorkspace, LoadedGenerationDecoderInput,
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
    ReservedInferenceSpanWorkspace, ReservedTextSpanWorkspace, SpanWorkspaceOwnerError,
    SubmissionTrackingFacts, TextHostControlFacts, TextPredictionScopeFacts, TextPrefillScopeFacts,
    TextPreparationScopeFacts,
};
pub use span_workspace::{
    HostSourcePeakSelection, OriginalHostSourcePeakCapacity, OriginalHostSourcePending,
};
mod registered_sources;
pub use copy_preparation::{CopyPreparationInferenceQuote, WorkspaceCopyCompositionError};
pub use registered_sources::RegisteredInferenceSourceWitness;
use registered_sources::RegisteredInferenceSources;

use super::{
    ControllerWorkspaceEstimate, InferenceExecutionIdentity, InferenceWorkspaceReport,
    PrefillPlanningError, WorkingMemoryCapacityHandoff, WorkingMemoryError, WorkingMemoryPool,
    WorkingMemoryReservation, WorkingMemoryStorage,
};
use eredu_core::{
    Admission, AdmissionRequest, CapabilityError, ExecutionWorkspaceEstimate, InferenceGeometry,
    ModelCapabilities, RuntimeStateEstimate, TextControllerContract, WorkspaceBound,
};
use eredu_nn::workspace::{WorkspaceBorrowedStorage, WorkspaceContext, WorkspaceExistingStorage};
use std::sync::Arc;

mod pin_layout;
mod registered_storage;
pub use registered_storage::{
    RegisteredPreparedWorkspaceStorage, RegisteredWorkspaceStorage,
    RegisteredWorkspaceStorageLayout,
};

/// Erased accounting-only custody, transferable from reservation to run/scopes.
#[derive(Clone)]
pub(super) enum RegisteredStoragePin {
    // Existing closed source bundle, copied by handle without a new container.
    Sources(registered_sources::RegisteredInferenceSources),
    #[allow(dead_code)]
    Shared(Arc<dyn Send + Sync>),
    // Preserve the concrete group's owned retirement through quarantine without
    // allocating an outer Arc or exposing its erased strong handle.
    #[allow(dead_code)]
    Opening(crate::working_memory::storage::bounded_pin::OpeningPinOwner),
    Prepared(PreparedStoragePins),
    Planned(PlanningStoragePins),
    // The completed prepared-input B account contains no native/source backedge.
    PreparedInput(crate::working_memory::PreparedInputHostCustody),
    // Numerical-account custody only; no array, native budget or request-list backedge.
    Completed(crate::working_memory::workspace_copy::completed::CompletedSourceCustody),
}

// These closed aliases can escape independently through a reservation or run.
// Their shared header and child pin retire before the current planning account.
pub(super) struct PlanningStoragePins(Option<Arc<PlanningPinGroup>>);
struct PlanningPinGroup {
    pin: RegisteredStoragePin,
    _funding: eredu_nn::workspace::HostMetadataFunding,
}
impl Clone for PlanningStoragePins {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}
impl Drop for PlanningStoragePins {
    fn drop(&mut self) {
        if let Some(owner) = self.0.take() {
            drop(Arc::into_inner(owner));
        }
    }
}

// The accepted preparation account remains after every child pin and group
// allocation. No raw strong/weak owner escapes this closed retirement path.
pub(super) struct PreparedStoragePins(Option<Arc<PreparedPinGroup>>);
struct PreparedPinGroup {
    _pins: PreparedPinContents,
    authority: eredu_core::HostPreparationAuthority,
}
#[allow(dead_code)]
enum PreparedPinContents {
    Single(Arc<dyn Send + Sync>),
    Pair([RegisteredStoragePin; 2]),
    Group(Vec<RegisteredStoragePin>),
}
impl Clone for PreparedStoragePins {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}
impl Drop for PreparedStoragePins {
    fn drop(&mut self) {
        if let Some(owner) = self.0.take() {
            drop(Arc::into_inner(owner));
        }
    }
}

impl std::fmt::Debug for RegisteredStoragePin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("RegisteredStoragePin")
    }
}

impl RegisteredStoragePin {
    fn prepared(
        pins: PreparedPinContents,
        authority: eredu_core::HostPreparationAuthority,
    ) -> Self {
        Self::Prepared(PreparedStoragePins(Some(Arc::new(PreparedPinGroup {
            _pins: pins,
            authority,
        }))))
    }
    // Lifetime only: propagating this alias neither prices nor authorizes a new
    // wrapper. Its owning producer must include that wrapper in its host bound.
    pub(super) fn preparation(&self) -> Option<&eredu_core::HostPreparationAuthority> {
        match self {
            Self::Prepared(owner) => {
                Some(&owner.0.as_deref().expect("live prepared pins").authority)
            }
            Self::Completed(source) => Some(source.host()),
            Self::Planned(owner) => owner
                .0
                .as_deref()
                .expect("live planning pins")
                .pin
                .preparation(),
            _ => None,
        }
    }
    // Move the already closed opening owner; no raw Arc or new allocation.
    pub(super) fn from_opening(
        owner: crate::working_memory::storage::bounded_pin::OpeningPinOwner,
    ) -> Self {
        Self::Opening(owner)
    }

    // Fixed publication-source join: both child pins precede their Arc owner.
    pub(super) fn pair(first: Self, second: Self) -> Self {
        let preparation = first
            .preparation()
            .or_else(|| second.preparation())
            .cloned();
        match preparation {
            Some(authority) => {
                Self::prepared(PreparedPinContents::Pair([first, second]), authority)
            }
            None => Self::Shared(Arc::new([first, second])),
        }
    }

    pub(super) fn new<K: Ord + Send + Sync + 'static>(storage: WorkingMemoryStorage<K>) -> Self {
        let preparation = storage.source_preparation().cloned();
        match preparation {
            Some(authority) => {
                Self::prepared(PreparedPinContents::Single(Arc::new(storage)), authority)
            }
            None => Self::Shared(Arc::new(storage)),
        }
    }

    // Only accounting handles enter this aggregate. Construction and destruction
    // occur outside the pool lock; no numerical owner can form a reference cycle.
    pub(super) fn aggregate(pins: impl IntoIterator<Item = Self>) -> Self {
        let pins: Vec<_> = pins.into_iter().collect();
        let preparation = pins.iter().find_map(Self::preparation).cloned();
        match preparation {
            Some(authority) => Self::prepared(PreparedPinContents::Group(pins), authority),
            None => Self::Shared(Arc::new(pins)),
        }
    }
}

/// Failure to associate a complete numerical quote with charged storage.
#[derive(Debug, thiserror::Error)]
pub enum ResidualQuoteError {
    /// Association is valid, but required numerical coverage is unavailable.
    #[error("{0}")]
    IncompleteWorkspace(#[from] IncompleteWorkspace),
    /// Invalid or incomplete equation/geometry composition.
    #[error(transparent)]
    Estimate(#[from] CapabilityError),
    /// Missing, mismatched or unregistered borrowed storage.
    #[error(transparent)]
    Storage(#[from] WorkingMemoryError),
}

/// Missing numerical coverage for one structurally validated cold candidate.
/// Construction is private: an arbitrary missing storage proof, geometry error
/// or native failure cannot be relabeled as retryable. This carries domain
/// identity only, with no registered storage pin or numerical payload.
#[derive(Debug, thiserror::Error)]
#[error("incomplete {component} for chunk {chunk}: {source}", chunk = .geometry.prefill_chunk_positions)]
pub struct IncompleteWorkspace {
    geometry: InferenceGeometry,
    pool: WorkingMemoryPool,
    component: &'static str,
    #[source]
    source: WorkingMemoryError,
}

impl IncompleteWorkspace {
    fn new(geometry: InferenceGeometry, pool: &WorkingMemoryPool, component: &'static str) -> Self {
        Self {
            geometry,
            pool: pool.clone(),
            component,
            source: WorkingMemoryError::UnknownBound,
        }
    }

    /// Exact geometry whose validated composition lacks a numerical bound.
    pub fn geometry(&self) -> InferenceGeometry {
        self.geometry
    }

    /// Required numerical component that could not be bounded.
    pub fn component(&self) -> &'static str {
        self.component
    }

    fn validate(
        &self,
        pool: &WorkingMemoryPool,
        geometry: InferenceGeometry,
    ) -> Result<(), WorkingMemoryError> {
        if self.geometry != geometry || !self.pool.same_domain(pool) {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        Ok(())
    }
}

/// Complete diagnostics and a sealed proof of incremental demand in one pool.
/// Only equation traces and checked controller/registered-root contributions
/// construct this value. It cannot authorize a scalar discount of a peak.
#[derive(Debug, Clone)]
pub struct IncrementalInferenceQuote {
    state: QuoteDiagnostics,
    geometry: InferenceGeometry,
    incremental_bytes: u64,
    // Exact new equation peak composed above the registered opening sources.
    // Other quotation paths cannot claim this residual replacement credit.
    equation_incremental_bytes: Option<u64>,
    pool: WorkingMemoryPool,
    pin: Option<RegisteredStoragePin>,
    controller: Option<TextControllerContract>,
    sources: Option<RegisteredInferenceSources>,
    span_workspace: InferenceSpanWorkspace,
    span_seal: Option<SpanWorkspaceSeal>,
}

impl IncrementalInferenceQuote {
    /// Composes full equations with the controller's separately checked full and
    /// incremental enclosing contributions. Only the declared fixed shared host
    /// sources receive credit; all state and equation peaks remain fully priced.
    pub fn compose_controller(
        equations: &InferenceWorkspaceReport,
        state: RuntimeStateEstimate,
        contribution: ControllerWorkspaceEstimate,
    ) -> Result<Self, ResidualQuoteError> {
        Self::compose_controller_metadata(
            equations,
            state,
            contribution,
            super::WorkspaceReportMetadata::ordinary(),
        )
        .map_err(WorkspaceCopyCompositionError::into_legacy)
    }
    /// The same controller-only source credit with counted report destinations.
    pub fn compose_controller_metadata(
        equations: &InferenceWorkspaceReport,
        state: RuntimeStateEstimate,
        contribution: ControllerWorkspaceEstimate,
        metadata: super::WorkspaceReportMetadata<'_>,
    ) -> Result<Self, WorkspaceCopyCompositionError> {
        metadata.admit::<Self>()?;
        metadata.admit::<WorkspaceCopyCompositionError>()?;
        metadata.admit::<InferenceSpanWorkspace>()?;
        if contribution.geometry() != equations.geometry() {
            return Err(WorkingMemoryError::IdentityMismatch.into());
        }
        validate_workspace_fixed(contribution.full(), equations.geometry())
            .map_err(super::WorkspaceReportError::from)?;
        validate_workspace_fixed(contribution.incremental(), equations.geometry())
            .map_err(super::WorkspaceReportError::from)?;
        let state = equations.refine_state_backing_metadata(state, metadata)?;
        if !equations.has_complete_state_spans() {
            return Err(WorkingMemoryError::UnknownBound.into());
        }
        let span_workspace =
            InferenceSpanWorkspace::new_fixed(equations.span_workspace_plan(), contribution.full())
                .map_err(super::WorkspaceReportError::from)?;
        let incremental = equations.compose_metadata(
            metadata.clone_state(&state)?,
            metadata.clone_execution(contribution.incremental())?,
            metadata,
        )?;
        let state = equations.compose_metadata(
            state,
            metadata.clone_execution(contribution.full())?,
            metadata,
        )?;
        // Validate full diagnostics before classifying an incremental gap.
        validate_requirement_with(&state, equations.geometry())?;
        let incremental_bytes =
            full_requirement_with(&incremental, equations.geometry(), contribution.pool())?;
        Ok(Self {
            state: QuoteDiagnostics::new(state, metadata)?,
            geometry: equations.geometry(),
            incremental_bytes,
            equation_incremental_bytes: None,
            pool: contribution.pool().clone(),
            pin: contribution.pin(),
            controller: Some(*contribution.controller_contract()),
            sources: None,
            span_workspace,
            span_seal: None,
        })
    }

    /// Existing host planning account for counted reservation metadata only.
    /// This closed clone creates neither source credit nor native permission.
    pub(crate) fn metadata_funding(&self) -> Option<eredu_nn::workspace::HostMetadataFunding> {
        self.span_workspace.plan().metadata_funding()
    }

    /// Original state and workspace diagnostics, retaining all credited sources.
    pub fn state(&self) -> &RuntimeStateEstimate {
        &self.state
    }

    /// Exact request and chunk geometry used to obtain this proof.
    pub fn geometry(&self) -> InferenceGeometry {
        self.geometry
    }

    /// Complete incremental requirement before the caller's safety reserve.
    pub fn incremental_bytes(&self) -> u64 {
        self.incremental_bytes
    }

    /// Accounting domain in which all credited sources remain pinned.
    pub fn pool(&self) -> &WorkingMemoryPool {
        &self.pool
    }

    /// Bound controller decisions, when this quote used a controller contribution.
    /// Legacy decoder-only quotes carry no controller contract.
    pub fn controller_contract(&self) -> Option<&TextControllerContract> {
        self.controller.as_ref()
    }

    fn reserve(
        &self,
        pool: &WorkingMemoryPool,
        execution: &InferenceExecutionIdentity,
        admission: &Admission,
        capacity: u64,
        handoffs: &[WorkingMemoryCapacityHandoff],
    ) -> Result<WorkingMemoryReservation, WorkingMemoryError> {
        if !pool.same_domain(&self.pool) || admission.state != *self.state {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        // An empty accounting aggregate still marks this sealed route as needing
        // funding conversion. No storage or permission is fabricated by the pin.
        let planning_metadata = self.metadata_funding();
        let pin = match (self.pin.clone(), planning_metadata.as_ref()) {
            (Some(pin), _) => pin,
            (None, Some(funding)) => RegisteredStoragePin::empty_metadata(
                super::WorkspaceReportMetadata::with_funding(funding),
            )
            .map_err(|cause| super::reservation_metadata::neural_error(cause, funding))?,
            (None, None) => RegisteredStoragePin::aggregate([]),
        };
        let reservation = pool.reserve_limited_with_source_metadata(
            execution,
            admission,
            Some(capacity),
            Some((self.incremental_bytes, pin)),
            handoffs,
            self.sources
                .as_ref()
                .map(|source| source as &dyn super::saved_source::SavedSourceValidation),
            planning_metadata,
        )?;
        Ok(self.attach_span_identity(reservation))
    }
}

fn validate_workspace(
    workspace: &ExecutionWorkspaceEstimate,
    geometry: InferenceGeometry,
) -> Result<u64, CapabilityError> {
    validate_workspace_fixed(workspace, geometry).map_err(Into::into)
}
fn validate_workspace_fixed(
    workspace: &ExecutionWorkspaceEstimate,
    geometry: InferenceGeometry,
) -> Result<u64, eredu_core::AdmissionPolicyError> {
    workspace.geometry.validate_fixed()?;
    if workspace.geometry != geometry {
        return Err(eredu_core::AdmissionPolicyError::InvalidConfiguration {
            field: "equation_workspace",
            detail: "outside workspace does not match inspected request geometry",
        });
    }
    // Unknown coverage must not hide overflow in the other known components.
    [
        &workspace.activations,
        &workspace.attention,
        &workspace.vocabulary,
        &workspace.state_update,
        &workspace.materialization,
        &workspace.retained,
    ]
    .into_iter()
    .filter_map(WorkspaceBound::bytes)
    .try_fold(0_u64, |sum, bytes| {
        sum.checked_add(bytes)
            .ok_or(eredu_core::AdmissionPolicyError::ArithmeticOverflow {
                operation: "simultaneous execution workspace",
            })
    })
}

fn validate_requirement(
    state: &RuntimeStateEstimate,
    geometry: InferenceGeometry,
) -> Result<(), ResidualQuoteError> {
    validate_requirement_with(state, geometry).map_err(WorkspaceCopyCompositionError::into_legacy)
}
fn full_requirement(
    state: &RuntimeStateEstimate,
    geometry: InferenceGeometry,
    pool: &WorkingMemoryPool,
) -> Result<u64, ResidualQuoteError> {
    full_requirement_with(state, geometry, pool).map_err(WorkspaceCopyCompositionError::into_legacy)
}
fn validate_requirement_with(
    state: &RuntimeStateEstimate,
    geometry: InferenceGeometry,
) -> Result<(), WorkspaceCopyCompositionError> {
    let workspace = state
        .execution_workspace
        .as_ref()
        .ok_or(WorkingMemoryError::UnknownBound)?;
    let known =
        validate_workspace_fixed(workspace, geometry).map_err(super::WorkspaceReportError::from)?;
    state
        .requested_state_bytes
        .checked_add(known)
        .ok_or(WorkingMemoryError::Overflow)?;
    Ok(())
}

fn full_requirement_with(
    state: &RuntimeStateEstimate,
    geometry: InferenceGeometry,
    pool: &WorkingMemoryPool,
) -> Result<u64, WorkspaceCopyCompositionError> {
    validate_requirement_with(state, geometry)?;
    let workspace = state
        .execution_workspace
        .as_ref()
        .ok_or(WorkingMemoryError::UnknownBound)?;
    state
        .requested_state_bytes
        .checked_add(
            workspace
                .peak_bytes_fixed()
                .map_err(super::WorkspaceReportError::from)?
                .ok_or_else(|| {
                    IncompleteWorkspace::new(geometry, pool, "full request workspace")
                })?,
        )
        .ok_or_else(|| WorkingMemoryError::Overflow.into())
}

/// Incremental proof retaining its typed registered decoder-root association.
/// Existing callers can inspect that association or erase it into the common
/// admission proof without dropping the independent accounting pin.
#[derive(Debug, Clone)]
pub struct ResidualInferenceQuote<K: Ord + Send + 'static> {
    proof: IncrementalInferenceQuote,
    storage: RegisteredWorkspaceStorage<K>,
}

impl<K: Clone + Ord + Send + Sync + 'static> ResidualInferenceQuote<K> {
    /// Composes exact residual equation demand with all enclosing costs.
    /// `outside` has the same full-coverage obligations as ordinary composition:
    /// preparation, sampling, controller, transfer and other disjoint resources.
    /// Logical/full selected-state diagnostics remain available without replacing
    /// them with a smaller synthetic state estimate.
    pub fn compose(
        equations: &InferenceWorkspaceReport,
        state: RuntimeStateEstimate,
        outside: ExecutionWorkspaceEstimate,
        storage: &RegisteredWorkspaceStorage<K>,
    ) -> Result<Self, ResidualQuoteError> {
        Self::compose_metadata(
            equations,
            state,
            outside,
            storage,
            super::WorkspaceReportMetadata::ordinary(),
        )
        .map_err(WorkspaceCopyCompositionError::into_legacy)
    }
    /// Ordinary combined source-credit adapter over the same fixed worker.
    pub fn compose_with_controller(
        equations: &InferenceWorkspaceReport,
        state: RuntimeStateEstimate,
        contribution: ControllerWorkspaceEstimate,
        storage: &RegisteredWorkspaceStorage<K>,
    ) -> Result<Self, ResidualQuoteError> {
        Self::compose_with_controller_metadata(
            equations,
            state,
            contribution,
            storage,
            super::WorkspaceReportMetadata::ordinary(),
        )
        .map_err(WorkspaceCopyCompositionError::into_legacy)
    }
    /// The same decoder source credit using counted diagnostic destinations.
    pub fn compose_metadata(
        equations: &InferenceWorkspaceReport,
        state: RuntimeStateEstimate,
        outside: ExecutionWorkspaceEstimate,
        storage: &RegisteredWorkspaceStorage<K>,
        metadata: super::WorkspaceReportMetadata<'_>,
    ) -> Result<Self, WorkspaceCopyCompositionError> {
        Self::compose_parts_metadata(
            equations,
            state,
            metadata.clone_execution(&outside)?,
            outside,
            storage,
            None,
            None,
            None,
            metadata,
        )
    }

    /// Combines exact decoder-root credit with the checked fixed shared-controller
    /// contribution. Both independent pins survive conversion into one reservation.
    pub fn compose_with_controller_metadata(
        equations: &InferenceWorkspaceReport,
        state: RuntimeStateEstimate,
        contribution: ControllerWorkspaceEstimate,
        storage: &RegisteredWorkspaceStorage<K>,
        metadata: super::WorkspaceReportMetadata<'_>,
    ) -> Result<Self, WorkspaceCopyCompositionError> {
        Self::compose_with_controller_source_metadata(
            equations,
            state,
            contribution,
            storage,
            None,
            metadata,
        )
    }
    fn compose_with_controller_source_metadata(
        equations: &InferenceWorkspaceReport,
        state: RuntimeStateEstimate,
        contribution: ControllerWorkspaceEstimate,
        storage: &RegisteredWorkspaceStorage<K>,
        prepared_source: Option<&crate::input::OriginalPreparedWorkspaceSource>,
        metadata: super::WorkspaceReportMetadata<'_>,
    ) -> Result<Self, WorkspaceCopyCompositionError> {
        if contribution.geometry() != equations.geometry()
            || !storage.pool.same_domain(contribution.pool())
        {
            return Err(WorkingMemoryError::IdentityMismatch.into());
        }
        Self::compose_parts_metadata(
            equations,
            state,
            metadata.clone_execution(contribution.full())?,
            metadata.clone_execution(contribution.incremental())?,
            storage,
            contribution.pin(),
            Some(*contribution.controller_contract()),
            prepared_source,
            metadata,
        )
    }

    fn compose_parts_metadata(
        equations: &InferenceWorkspaceReport,
        state: RuntimeStateEstimate,
        full_outside: ExecutionWorkspaceEstimate,
        incremental_outside: ExecutionWorkspaceEstimate,
        storage: &RegisteredWorkspaceStorage<K>,
        controller_pin: Option<RegisteredStoragePin>,
        controller: Option<TextControllerContract>,
        prepared_source: Option<&crate::input::OriginalPreparedWorkspaceSource>,
        metadata: super::WorkspaceReportMetadata<'_>,
    ) -> Result<Self, WorkspaceCopyCompositionError> {
        metadata.admit::<Self>()?;
        metadata.admit::<IncrementalInferenceQuote>()?;
        metadata.admit::<InferenceSpanWorkspace>()?;
        metadata.admit::<WorkspaceCopyCompositionError>()?;
        validate_workspace_fixed(&full_outside, equations.geometry())
            .map_err(super::WorkspaceReportError::from)?;
        validate_workspace_fixed(&incremental_outside, equations.geometry())
            .map_err(super::WorkspaceReportError::from)?;
        let state = equations.refine_state_backing_metadata(state, metadata)?;
        let residual = equations
            .residual_workspace()
            .ok_or(WorkingMemoryError::UnknownBound)?;
        if !residual.borrowed_storage().same_identity(&storage.borrowed) {
            return Err(WorkingMemoryError::IdentityMismatch.into());
        }
        if !equations.has_complete_state_spans() || !residual.has_complete_association() {
            return Err(WorkingMemoryError::UnknownBound.into());
        }
        // Preserve an already provable residual overflow even when another
        // component lacks coverage. This is a fatal accounting error.
        if let (Some(peak), Some(outside)) = (
            residual.peak_bytes(),
            incremental_outside
                .peak_bytes_fixed()
                .map_err(super::WorkspaceReportError::from)?,
        ) {
            peak.checked_add(outside)
                .ok_or(WorkingMemoryError::Overflow)?;
        }
        let span_workspace =
            InferenceSpanWorkspace::new_fixed(equations.span_workspace_plan(), &full_outside)
                .map_err(super::WorkspaceReportError::from)?;
        let state = equations.compose_metadata(state, full_outside, metadata)?;
        validate_requirement_with(&state, equations.geometry())?;
        let peak = residual.peak_bytes().ok_or_else(|| {
            IncompleteWorkspace::new(
                equations.geometry(),
                &storage.pool,
                "residual equation workspace",
            )
        })?;
        let outside_bytes = incremental_outside
            .peak_bytes_fixed()
            .map_err(super::WorkspaceReportError::from)?
            .ok_or_else(|| {
                IncompleteWorkspace::new(equations.geometry(), &storage.pool, "enclosing workspace")
            })?;
        let incremental_bytes = peak
            .checked_add(outside_bytes)
            .ok_or(WorkingMemoryError::Overflow)?;
        let decoder_pin =
            RegisteredStoragePin::new_metadata(storage._registration.clone(), metadata)
                .map_err(super::WorkspaceReportError::from)?;
        let decoder_pin = match prepared_source {
            Some(source) => RegisteredStoragePin::pair_metadata(
                decoder_pin,
                RegisteredStoragePin::PreparedInput(source.account_pin()),
                metadata,
            )
            .map_err(super::WorkspaceReportError::from)?,
            None => decoder_pin,
        };
        let pin = match controller_pin {
            Some(host) => RegisteredStoragePin::pair_metadata(decoder_pin, host, metadata)
                .map_err(super::WorkspaceReportError::from)?,
            None => decoder_pin,
        };
        Ok(Self {
            proof: IncrementalInferenceQuote {
                state: QuoteDiagnostics::new(state, metadata)?,
                geometry: equations.geometry(),
                incremental_bytes,
                equation_incremental_bytes: Some(peak),
                pool: storage.pool.clone(),
                pin: Some(pin),
                controller,
                sources: None,
                span_workspace,
                span_seal: None,
            },
            storage: storage.clone(),
        })
    }

    /// Original complete state and workspace report, with no residency credit.
    pub fn state(&self) -> &RuntimeStateEstimate {
        self.proof.state()
    }
    /// Exact inspected request and chunk geometry.
    pub fn geometry(&self) -> InferenceGeometry {
        self.proof.geometry()
    }
    /// Proved new storage/workspace, before caller safety reserve.
    pub fn incremental_bytes(&self) -> u64 {
        self.proof.incremental_bytes()
    }
    /// Registered roots used to prove this quote. A successful reservation also
    /// retains independent pin custody through its funding run and work scopes.
    pub fn registered_storage(&self) -> &RegisteredWorkspaceStorage<K> {
        &self.storage
    }
    /// Erases provider key/metadata roots while preserving accounting custody.
    pub fn into_incremental(self) -> IncrementalInferenceQuote {
        self.proof
    }
}

/// Prices exact registered-root credit with the ordinary shared candidate and
/// context/application-memory policies. The accepted quote retains typed root
/// access; diagnostics are borrowed from the returned reservation so their
/// metadata funding cannot retire before the report.
pub fn plan_prefill_residual_with_capacity<K: Clone + Ord + Send + Sync + 'static>(
    execution: &InferenceExecutionIdentity,
    pool: &WorkingMemoryPool,
    capabilities: &ModelCapabilities,
    request: AdmissionRequest,
    geometry: InferenceGeometry,
    capacity: u64,
    quote: impl FnMut(InferenceGeometry) -> Result<ResidualInferenceQuote<K>, PrefillPlanningError>,
) -> Result<
    (
        WorkingMemoryReservation,
        ResidualInferenceQuote<K>,
    ),
    PrefillPlanningError,
> {
    plan_incremental(
        execution,
        pool,
        capabilities,
        request,
        geometry,
        capacity,
        &[],
        quote,
        |quote| &quote.proof,
    )
}

/// Admits either full equations with fixed-controller credit or combined
/// controller/decoder credit. The reservation independently retains all pins
/// and requires funding conversion before native work. Historical diagnostic
/// metadata need not survive for those source charges to remain protected.
/// Borrow admission diagnostics through the reservation; no independently owned
/// copy escapes its funding lifetime.
pub fn plan_prefill_incremental_with_capacity(
    execution: &InferenceExecutionIdentity,
    pool: &WorkingMemoryPool,
    capabilities: &ModelCapabilities,
    request: AdmissionRequest,
    geometry: InferenceGeometry,
    capacity: u64,
    quote: impl FnMut(InferenceGeometry) -> Result<IncrementalInferenceQuote, PrefillPlanningError>,
) -> Result<
    (
        WorkingMemoryReservation,
        IncrementalInferenceQuote,
    ),
    PrefillPlanningError,
> {
    plan_prefill_incremental_with_capacity_handoff(
        execution,
        pool,
        capabilities,
        request,
        geometry,
        capacity,
        &[],
        quote,
    )
}

/// Admits a successor with explicitly delegated completed-account ceilings.
///
/// Handoffs come only from unique funding owners, never request metadata. This
/// uses the ordinary candidate order and complete quote validation. Rejected
/// quotes and candidates change no ceiling, reservation or peak. A fitting
/// candidate atomically adopts eligible predecessor ceilings and reserves its
/// own demand; unrelated ceilings and every physical storage charge remain.
/// Once admission succeeds, later preparation failure does not roll back the
/// authorized ceiling change. The capabilities grant no execution permission.
#[allow(clippy::too_many_arguments)]
pub fn plan_prefill_incremental_with_capacity_handoff(
    execution: &InferenceExecutionIdentity,
    pool: &WorkingMemoryPool,
    capabilities: &ModelCapabilities,
    request: AdmissionRequest,
    geometry: InferenceGeometry,
    capacity: u64,
    handoffs: &[WorkingMemoryCapacityHandoff],
    quote: impl FnMut(InferenceGeometry) -> Result<IncrementalInferenceQuote, PrefillPlanningError>,
) -> Result<
    (
        WorkingMemoryReservation,
        IncrementalInferenceQuote,
    ),
    PrefillPlanningError,
> {
    plan_incremental(
        execution,
        pool,
        capabilities,
        request,
        geometry,
        capacity,
        handoffs,
        quote,
        |quote| quote,
    )
}

fn plan_incremental<Q>(
    execution: &InferenceExecutionIdentity,
    pool: &WorkingMemoryPool,
    capabilities: &ModelCapabilities,
    request: AdmissionRequest,
    geometry: InferenceGeometry,
    capacity: u64,
    handoffs: &[WorkingMemoryCapacityHandoff],
    mut quote: impl FnMut(InferenceGeometry) -> Result<Q, PrefillPlanningError>,
    proof: fn(&Q) -> &IncrementalInferenceQuote,
) -> Result<(WorkingMemoryReservation, Q), PrefillPlanningError> {
    // Identity failures are fatal even when every numerical candidate is
    // incomplete or exceeds application policy. Eligibility and ceiling changes
    // are still checked atomically with the selected reservation below.
    pool.validate_capacity_handoff_identities(execution, handoffs)?;
    super::plan_prefill_candidates(
        capabilities,
        request,
        geometry,
        |geometry| {
            let result = quote(geometry);
            if let Err(PrefillPlanningError::IncompleteWorkspace(incomplete)) = &result {
                incomplete.validate(pool, geometry)?;
            }
            result
        },
        |quote, geometry| {
            let bound = proof(&quote);
            if bound.geometry != geometry || !bound.pool.same_domain(pool) {
                return Err(WorkingMemoryError::IdentityMismatch.into());
            }
            let funding = bound.metadata_funding();
            let metadata = funding.as_ref().map_or_else(
                super::WorkspaceReportMetadata::ordinary,
                super::WorkspaceReportMetadata::with_funding,
            );
            let failure = |error: super::WorkspaceReportError| match funding.as_ref() {
                Some(funding) => PrefillPlanningError::Reservation(
                    super::reservation_metadata::neural_error(metadata.error(error), funding),
                ),
                None => PrefillPlanningError::Estimate(error.into_capability()),
            };
            let incremental = metadata.bounded(
                bound.incremental_bytes,
                format_args!("complete equation demand and checked enclosing contributions; only exact pinned decoder roots and fixed shared controller sources receive credit"),
            ).map_err(failure)?;
            match metadata.apply_admission_with_incremental(
                capabilities,
                request,
                &bound.state,
                &incremental,
                None,
            ).map_err(failure)? {
                eredu_core::AdmissionResult::Admitted(admission) => {
                    let reservation =
                        bound.reserve(pool, execution, &admission, capacity, handoffs)?;
                    Ok((reservation, quote))
                }
                eredu_core::AdmissionResult::Rejected(rejection) => {
                    let owns_text = matches!(&rejection,
                        eredu_core::AdmissionRejection::EstimationUnsupported { .. }
                        | eredu_core::AdmissionRejection::AvailableMemoryUnavailable { .. });
                    let error = PrefillPlanningError::Admission(rejection);
                    // Scalar refusals need no escaped allocation owner. An owned
                    // diagnostic keeps the same producer account through its
                    // existing metadata-error enclosure.
                    match funding.as_ref().filter(|_| owns_text) {
                        Some(funding) => Err(super::reservation_metadata::neural_error(
                            metadata.source(error), funding,
                        ).into()),
                        None => Err(error),
                    }
                }
            }
        },
    )
}

#[cfg(test)]
mod tests;

pub use span_workspace::{
    NativeEquationStorage, NativePrefillEnvelope, NativePrefillEnvelopeBuilder,
    NativeStorageCarryoverReport,
};
