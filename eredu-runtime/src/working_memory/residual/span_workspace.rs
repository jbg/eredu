//! Opt-in retained span diagnostics bound to the actual successful reservation.
use super::*;
use crate::working_memory::{
    InferenceSpanWorkspacePlan, InferenceWorkspaceSpan, Usage, WorkingMemoryFundingRun,
    funding::{SpanHostCustody, SpanHostOwner},
    saved_source::SavedSourceValidation,
};
use std::mem::size_of;
use eredu_nn::workspace::WorkspaceMetadataAllocation;
mod text_controls;
pub(in crate::working_memory) use text_controls::OriginalTokenDomainBinding;
#[cfg(test)]
pub(in crate::working_memory) use text_controls::SequenceExtractionError;
pub(in crate::working_memory) use text_controls::TextControlBinding;
pub use text_controls::{
    AdmittedCaptureContinuation, AdmittedPrefillCapture, AggregateGenerationDecoderInput,
    SamplingExtensionQuote, OriginalTextSamplingExtension,
    FailedCapturePlanPublication, GraphMetadataFacts, HostDestinationCause, HostDestinationFacts,
    HostSourceConstructionFacts, HostSourceConstructionProgram, OriginalHostSourceProgramBanks, OriginalHostSourceProgramError, LoadedGenerationDecoderInput, NativeStorageError,
    NativeStorageObservation, NativeStorageRegistration, NativeStorageSelection,
    OriginalGenerationDecoderInput, OriginalGenerationDecoderSource,
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
    OwnedPromptTokenIds, OwnedTextSpanWorkspace, PendingCapturePlanPublication,
    PreparedNativeStoragePlan, PreparedTextControlWorkspace, ReservedTextSpanWorkspace,
    SubmissionTrackingFacts, TextHostControlFacts, TextPredictionScopeFacts, TextPrefillScopeFacts,
    TextPreparationScopeFacts,
};
pub use text_controls::{
    HostSourcePeakSelection, OriginalHostSourcePeakCapacity, OriginalHostSourcePending,
};

/// Full original candidate mechanism terms paired with its actual equation plan.
/// Values are diagnostic upper bounds, not a remaining-capacity check or grant.
/// In particular this does not establish current opening-root registration,
/// source health, exclusive spending, or native settlement.
#[derive(Debug, Clone)]
pub struct InferenceSpanWorkspace {
    plan: InferenceSpanWorkspacePlan,
    preparation: Option<u64>,
    attention: Option<u64>,
    materialization: Option<u64>,
    // A sampling-only program includes its original host/history and tensor
    // peaks, rather than invoking model preparation/attention mechanisms.
    sampling: Option<u64>,
    text_controls: Option<PreparedTextControlWorkspace>,
}
impl InferenceSpanWorkspace {
    pub(super) fn new(
        plan: &InferenceSpanWorkspacePlan,
        full: &ExecutionWorkspaceEstimate,
    ) -> Result<Self, CapabilityError> {
        Self::new_fixed(plan, full).map_err(Into::into)
    }
    pub(super) fn new_fixed(
        plan: &InferenceSpanWorkspacePlan,
        full: &ExecutionWorkspaceEstimate,
    ) -> Result<Self, eredu_core::AdmissionPolicyError> {
        if plan.geometry() != full.geometry {
            return Err(eredu_core::AdmissionPolicyError::InvalidConfiguration {
                field: "span_workspace",
                detail: "original mechanism terms differ from equation geometry",
            });
        }
        let preparation = add_fixed(full.activations.bytes(), full.state_update.bytes())?;
        let attention = full.attention.bytes();
        let materialization = full.materialization.bytes();
        // Unknown terms must not conceal overflow among independently known terms.
        let known = [
            full.activations.bytes(),
            full.state_update.bytes(),
            attention,
            materialization,
        ]
        .into_iter()
        .flatten()
        .try_fold(0u64, |a, b| checked_fixed(a, b))?;
        for record in plan.records() {
            if let Some(n) = record.new_allocation_bytes() {
                checked_fixed(known, n)?;
            }
        }
        Ok(Self {
            plan: plan.clone(),
            preparation,
            attention,
            materialization,
            sampling: None,
            text_controls: None,
        })
    }
    /// Exact original traversal, shared with the completed equation report.
    pub fn plan(&self) -> &InferenceSpanWorkspacePlan {
        &self.plan
    }
    /// Full original activation/source preparation and state-update terms before
    /// equation composition. Native ordinary text includes its complete prompt
    /// source/initialization/identity peak; copy preparation includes the FULL
    /// source-inclusive copy term. This conservative diagnostic does not subtract
    /// already constructed prompt storage or registered sources.
    pub fn source_preparation_bytes(&self) -> Option<u64> {
        self.preparation
    }
    /// Original untraced attention contribution, separate from traced equations.
    pub fn outside_attention_bytes(&self) -> Option<u64> {
        self.attention
    }
    /// Same candidate's full selected host/disk materialization bound; resident
    /// contributes its original proven zero. No route is reselected here.
    pub fn materialization_bytes(&self) -> Option<u64> {
        self.materialization
    }
    /// Conservative new equation plus full original preparation/materialization
    /// terms. Sampling, controller and cumulative retained H remain separate.
    pub fn span_bytes(&self, index: usize) -> Option<u64> {
        if matches!(self.plan.records().get(index)?.span(), crate::working_memory::InferenceWorkspaceSpan::Sampling(_)) {
            return self.sampling;
        }
        self.plan
            .records()
            .get(index)?
            .new_allocation_bytes()?
            .checked_add(self.preparation?)?
            .checked_add(self.attention?)?
            .checked_add(self.materialization?)
    }
    /// Unreserved record payload, its fixed attachment slot, candidate, seal,
    /// association, compact owner, typed owner-preserving error and witness
    /// controls, including three explicit by-value construction moves and the
    /// named plan/full/raw owned-retirement extraction overlap.
    /// The original record allocation is excluded only while its actual
    /// planning account remains attached. New attachment and execution controls
    /// are added once to full AND incremental admission on opt-in.
    pub fn retention_peak_bytes(&self) -> Option<u64> {
        let controls = size_of::<Self>()
            .checked_add(size_of::<SpanWorkspaceSeal>())?
            .checked_add(size_of::<SpanHostCustody>())?
            .checked_add(crate::working_memory::funding::span_workspace_raw_payload_bytes())?
            .checked_add(size_of::<OwnedInferenceSpanWorkspace>())?
            .checked_add(size_of::<SpanWorkspaceOwnerError>())?
            .checked_add(size_of::<Option<RegisteredInferenceSourceWitness>>())?
            .checked_add(size_of::<std::sync::MutexGuard<'static, ()>>())?
            .checked_add(size_of::<Option<SpanWorkspaceIdentity>>())?
            .checked_add(size_of::<ReservedInferenceSpanWorkspace<'static>>())?
            .checked_add(size_of::<
                crate::working_memory::inference::InferenceSpanWorkspaceRecord,
            >())?
            .checked_mul(3)?;
        // Both the seal identity and newly shared custody have actual Arc
        // headers. Historical reservations still retain no record payload.
        let controls = controls
            .checked_add(6 * size_of::<usize>())?
            .checked_add(crate::working_memory::funding::span_retirement_control_bytes()?)?
            .checked_add(InferenceSpanWorkspacePlan::retirement_control_bytes()?)?;
        let controls = if self.text_controls.is_some() {
            controls.checked_add(text_controls::control_metadata_bytes()?)?
        } else {
            controls
        };
        self.plan
            .unreserved_capacity_bytes()?
            .checked_add(u64::try_from(controls).ok()?)
    }
}
fn checked(a: u64, b: u64) -> Result<u64, CapabilityError> {
    checked_fixed(a, b).map_err(Into::into)
}
fn add(a: Option<u64>, b: Option<u64>) -> Result<Option<u64>, CapabilityError> {
    add_fixed(a, b).map_err(Into::into)
}
fn checked_fixed(a: u64, b: u64) -> Result<u64, eredu_core::AdmissionPolicyError> {
    a.checked_add(b)
        .ok_or(eredu_core::AdmissionPolicyError::ArithmeticOverflow {
            operation: "original span mechanism workspace",
        })
}
fn add_fixed(
    a: Option<u64>,
    b: Option<u64>,
) -> Result<Option<u64>, eredu_core::AdmissionPolicyError> {
    a.zip(b).map(|(a, b)| checked_fixed(a, b)).transpose()
}

#[derive(Debug, Clone)]
pub(in crate::working_memory) struct SpanWorkspaceIdentity {
    allocation: Arc<()>,
    // Every identity alias holds the payer until after its shared shell retires.
    _funding: Option<eredu_core::HostMetadataFunding>,
}
impl SpanWorkspaceIdentity {
    fn same(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.allocation, &other.allocation)
    }
}
#[derive(Debug, Clone)]
pub(super) struct SpanWorkspaceSeal {
    identity: SpanWorkspaceIdentity,
}

/// Borrowed association with one successfully reserved original candidate.
/// This contains no owner, capacity authorization, native scope or completion.
/// The borrowed reservation's identity remains authoritative for future binding.
#[derive(Debug)]
pub struct ReservedInferenceSpanWorkspace<'a> {
    workspace: &'a InferenceSpanWorkspace,
    reservation: &'a WorkingMemoryReservation,
    sources: Option<&'a RegisteredInferenceSources>,
}
impl ReservedInferenceSpanWorkspace<'_> {
    /// Same retained diagnostic plan and original full mechanism contributions.
    pub fn workspace(&self) -> &InferenceSpanWorkspace {
        self.workspace
    }
    /// Original actual reservation, never one reconstructed from its geometry.
    pub fn reservation(&self) -> &WorkingMemoryReservation {
        self.reservation
    }
    pub(in crate::working_memory) fn validate_request(
        &self,
        request: &crate::working_memory::InferenceRequest,
    ) -> Result<(), WorkingMemoryError> {
        let reservation = request
            .memory_reservation()
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        if !reservation.0.same(&self.reservation.0) {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        Ok(())
    }
    // The exact accepted candidate's original registered-source validators.
    // Borrowed, not cloned: no new pin/owner or independently supplied witness.
    pub(in crate::working_memory) fn validate_sources(
        &self,
        pool: &WorkingMemoryPool,
        usage: &Usage,
    ) -> Result<(), WorkingMemoryError> {
        if !self.reservation.0.pool.same_domain(pool) {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        if let Some(sources) = self.sources {
            sources.validate(pool, usage)?;
        }
        Ok(())
    }
    pub(in crate::working_memory) fn source_witness(
        &self,
    ) -> Option<RegisteredInferenceSourceWitness> {
        self.sources
            .cloned()
            .map(|sources| sources.into_witness(self.reservation.0.pool.clone()))
    }
    /// Finds an exact originally scheduled span, without accepting caller bytes.
    pub fn span_bytes(&self, span: &InferenceWorkspaceSpan) -> Option<u64> {
        let index = self
            .workspace
            .plan
            .records()
            .iter()
            .position(|record| record.span() == span)?;
        self.workspace.span_bytes(index)
    }
}
impl IncrementalInferenceQuote {
    /// Original full mechanism terms and exact trace plan. Unsealed diagnostics
    /// cannot be associated with a successful reservation through this API.
    pub fn span_workspace(&self) -> &InferenceSpanWorkspace {
        &self.span_workspace
    }
    /// Opt in BEFORE admission to retaining the original immutable span plan.
    /// No report, bytes, source credit or extension callback can replace it.
    /// This adds measured host retention to unchanged full and residual bounds;
    /// it still supplies no pre-evaluation check or native permission.
    pub fn with_span_workspace(self) -> Result<Self, ResidualQuoteError> {
        self.seal_span_workspace()
    }
    fn seal_span_workspace(mut self) -> Result<Self, ResidualQuoteError> {
        if self.span_seal.is_some() {
            return Err(WorkingMemoryError::IdentityMismatch.into());
        }
        // A terminal saved-state placement has no equation roles. Its copied
        // destination, source pins and enclosing controls remain fully priced;
        // sealing the empty schedule must not manufacture a forward span.
        let geometry = self.span_workspace.plan.geometry();
        let terminal = geometry == self.geometry
            && geometry.input_positions == 0
            && geometry.max_output_tokens == 0
            && geometry.prefill_chunk_positions == 0
            && geometry.output == eredu_core::OutputDemand::StateOnly
            && geometry.validate_fixed().is_ok();
        if (self.span_workspace.plan.records().is_empty() && !terminal)
            || (0..self.span_workspace.plan.records().len())
                .any(|i| self.span_workspace.span_bytes(i).is_none())
        {
            return Err(WorkingMemoryError::UnknownBound.into());
        }
        let funding = self.span_workspace.plan.metadata_funding();
        cold_controls::<(
            Self, SpanWorkspaceSeal, SpanWorkspaceIdentity, Result<Self, ResidualQuoteError>,
            InferenceGeometry, bool, Result<(), eredu_core::AdmissionPolicyError>,
        )>(funding.as_ref())?;
        if let Some(funding) = &funding {
            funding.reserve_metadata(shared_shell::<()>()?).map_err(crate::working_memory::reservation_metadata::funding_error)?;
        }
        let host = self.span_workspace.protected_peak_bytes()?;
        let source = self
            .span_workspace
            .text_controls
            .as_ref()
            .map_or(0, |c| c.capture_publication_source_bytes());
        let retained = host
            .checked_add(source)
            .ok_or(WorkingMemoryError::Overflow)?;
        let incremental = self
            .incremental_bytes
            .checked_add(retained)
            .ok_or(WorkingMemoryError::Overflow)?;
        let has_text_controls = self.span_workspace.text_controls.is_some();
        let workspace = self
            .state_mut()?
            .execution_workspace
            .as_mut()
            .ok_or(WorkingMemoryError::UnknownBound)?;
        let WorkspaceBound::Bounded { bytes, assumptions } = &mut workspace.retained else {
            return Err(WorkingMemoryError::UnknownBound.into());
        };
        *bytes = bytes
            .checked_add(retained)
            .ok_or(WorkingMemoryError::Overflow)?;
        append_assumptions(assumptions,
            "; original immutable span schedule actual capacity and measured retained controls/moves",
            if has_text_controls {
                "; original named Q, optional finite pin and single-C publication controls protected with P; exact new C reserved separately, without native capacity credit"
            } else { "" }, funding.as_ref())?;
        validate_requirement_with(&self.state, self.geometry).map_err(|error| match funding.as_ref() {
            Some(funding) => ResidualQuoteError::Storage(crate::working_memory::reservation_metadata::neural_error(
                error.into_workspace(crate::working_memory::WorkspaceReportMetadata::with_funding(funding)), funding)),
            None => error.into_legacy(),
        })?;
        self.incremental_bytes = incremental;
        self.span_seal = Some(SpanWorkspaceSeal {
            identity: SpanWorkspaceIdentity { allocation: Arc::new(()), _funding: funding },
        });
        Ok(self)
    }
    /// Borrows the plan only if this very sealed candidate created the supplied
    /// successful reservation. Equal geometry/bytes, an unsealed quote, or an
    /// independently sealed quote is insufficient. Copies of the exact quote
    /// and reservation preserve the association; no payload is stored on history.
    pub fn reserved_span_workspace<'a>(
        &'a self,
        reservation: &'a WorkingMemoryReservation,
    ) -> Result<ReservedInferenceSpanWorkspace<'a>, WorkingMemoryError> {
        let seal = self
            .span_seal
            .as_ref()
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        if !self.pool.same_domain(&reservation.0.pool)
            || self.geometry != reservation.0.geometry
            || !reservation
                .0
                .span_workspace
                .as_ref()
                .is_some_and(|id| seal.identity.same(id))
        {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        Ok(ReservedInferenceSpanWorkspace {
            workspace: &self.span_workspace,
            reservation,
            sources: self.sources.as_ref(),
        })
    }
    pub(super) fn span_identity(&self) -> Option<SpanWorkspaceIdentity> {
        self.span_seal.as_ref().map(|seal| seal.identity.clone())
    }
    pub(super) fn attach_span_identity(
        &self,
        mut reservation: WorkingMemoryReservation,
    ) -> WorkingMemoryReservation {
        // This is the fresh private reserve result, before any clone or escape.
        reservation
            .0
            .get_mut()
            .expect("fresh private reservation")
            .span_workspace = self.span_identity();
        reservation
    }
}

/// Compact, move-only custody for one successfully accepted span workspace.
/// The complete original host peak is attached to the actual shared plan, so
/// diagnostic aliases created before promotion retain it as well. This owner
/// supplies neither a native scope nor a remaining-span authorization.
#[derive(Debug)]
#[must_use = "retain the accepted workspace through its original execution"]
pub struct OwnedInferenceSpanWorkspace {
    sources: Option<RegisteredInferenceSources>,
    seal: SpanWorkspaceSeal,
    reservation: WorkingMemoryReservation,
    // Last: every earlier field retires while the shared plan still holds P.
    workspace: InferenceSpanWorkspace,
}
impl OwnedInferenceSpanWorkspace {
    /// Actual original candidate, with full unreduced mechanism contributions.
    pub fn workspace(&self) -> &InferenceSpanWorkspace {
        &self.workspace
    }
    /// Original successful reservation metadata. Cloning this metadata grants
    /// neither another promotion nor another native scope.
    pub fn reservation(&self) -> &WorkingMemoryReservation {
        &self.reservation
    }
    /// Borrows the same immutable association and exact explicit source joins.
    /// Current source/account health is still rechecked by the closed consumer;
    /// this conversion does not certify opening inventory or native completion.
    pub fn as_reserved_span_workspace(&self) -> ReservedInferenceSpanWorkspace<'_> {
        debug_assert!(self
            .reservation
            .0
            .span_workspace
            .as_ref()
            .is_some_and(|id| self.seal.identity.same(id)));
        ReservedInferenceSpanWorkspace {
            workspace: &self.workspace,
            reservation: &self.reservation,
            sources: self.sources.as_ref(),
        }
    }
}

/// Failed promotion retains the exact original quote, including an attachment
/// already committed before a later health failure. No source or plan is lost.
#[derive(Debug)]
pub struct SpanWorkspaceOwnerError {
    quote: IncrementalInferenceQuote,
    cause: WorkingMemoryError,
}
impl SpanWorkspaceOwnerError {
    /// Original typed accounting failure.
    pub fn cause(&self) -> &WorkingMemoryError {
        &self.cause
    }
    /// Recover the original quote and typed cause without cloning diagnostics.
    pub fn into_parts(self) -> (IncrementalInferenceQuote, WorkingMemoryError) {
        (self.quote, self.cause)
    }
}
impl std::fmt::Display for SpanWorkspaceOwnerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("could not attach original span workspace custody")
    }
}
impl std::error::Error for SpanWorkspaceOwnerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}
impl IncrementalInferenceQuote {
    /// Promote this exact sealed, accepted candidate under its ORIGINAL funding
    /// account. Full P is protected once, before attaching to the actual shared
    /// record owner. Independently sealed, unrelated and duplicate promotions
    /// reject; no second reservation or public byte allocator is introduced.
    ///
    /// Success retires full diagnostic/controller/residual fields outside Usage.
    /// The optional witness covers only explicitly joined registered sources,
    /// not opaque residual/controller/copy pins or a native opening inventory.
    pub fn into_funded_span_workspace(
        self,
        run: &WorkingMemoryFundingRun,
        reservation: &WorkingMemoryReservation,
    ) -> Result<
        (
            OwnedInferenceSpanWorkspace,
            Option<RegisteredInferenceSourceWitness>,
        ),
        SpanWorkspaceOwnerError,
    > {
        if self.span_workspace.text_controls.is_some() {
            return Err(SpanWorkspaceOwnerError {
                quote: self,
                cause: WorkingMemoryError::IdentityMismatch,
            });
        }
        self.promote_span_workspace(run, reservation)
    }
    fn promote_span_workspace(
        self,
        run: &WorkingMemoryFundingRun,
        reservation: &WorkingMemoryReservation,
    ) -> Result<
        (
            OwnedInferenceSpanWorkspace,
            Option<RegisteredInferenceSourceWitness>,
        ),
        SpanWorkspaceOwnerError,
    > {
        let result = self
            .reserved_span_workspace(reservation)
            .and_then(|receipt| self.span_workspace.plan.attach_original_host(run, &receipt));
        if let Err(cause) = result {
            return Err(SpanWorkspaceOwnerError { quote: self, cause });
        }
        Ok(self.finish_span_workspace(reservation))
    }
    fn finish_span_workspace(
        self,
        reservation: &WorkingMemoryReservation,
    ) -> (
        OwnedInferenceSpanWorkspace,
        Option<RegisteredInferenceSourceWitness>,
    ) {
        let Self {
            state,
            geometry: _,
            incremental_bytes: _,
            equation_incremental_bytes: _,
            pool,
            pin,
            controller,
            sources,
            span_workspace,
            span_seal,
        } = self;
        let owned = OwnedInferenceSpanWorkspace {
            sources: sources.clone(),
            seal: span_seal.expect("validated accepted span seal"),
            reservation: reservation.clone(),
            workspace: span_workspace,
        };
        let witness = sources.map(|sources| sources.into_witness(pool));
        // Original run/scopes already retain their accepted accounting pins.
        // These old full diagnostics cannot escape through the compact owner.
        drop((state, pin, controller));
        (owned, witness)
    }
}

#[cfg(test)]
pub(in crate::working_memory) use text_controls::token_input_fault;

pub use text_controls::{
    NativeEquationStorage, NativePrefillEnvelope, NativePrefillEnvelopeBuilder,
    NativeStorageCarryoverReport,
};

// Same cold producer in ordinary and source-paid plans. The existing account is
// retained by the plan and every detached identity; these helpers mint no credit.
fn cold_controls<T>(funding: Option<&eredu_core::HostMetadataFunding>) -> Result<(), WorkingMemoryError> {
    if let Some(funding) = funding {
        let bytes = [size_of::<T>(), size_of::<Result<(), WorkingMemoryError>>(),
            size_of::<Option<&eredu_core::HostMetadataFunding>>(),
            eredu_core::HostMetadataFunding::reservation_control_bytes()]
            .into_iter().try_fold(0usize, usize::checked_add).ok_or(WorkingMemoryError::Overflow)?;
        funding.reserve_metadata(bytes).map_err(crate::working_memory::reservation_metadata::funding_error)?;
    }
    Ok(())
}
fn shared_shell<T>() -> Result<usize, WorkingMemoryError> {
    std::alloc::Layout::new::<[std::sync::atomic::AtomicUsize; 2]>()
        .extend(std::alloc::Layout::new::<T>()).map(|layout| layout.0.pad_to_align().size())
        .map_err(|_| WorkingMemoryError::Overflow)
}
fn append_assumptions(target: &mut String, first: &str, second: &str,
    funding: Option<&eredu_core::HostMetadataFunding>) -> Result<(), WorkingMemoryError> {
    cold_controls::<(&mut String, &str, &str, std::fmt::Arguments<'_>, String)>(funding)?;
    let value = match funding {
        Some(funding) => funding.metadata_string(format_args!("{target}{first}{second}"))
            .map_err(|error| crate::working_memory::reservation_metadata::neural_error(error, funding))?,
        None => format!("{target}{first}{second}"),
    };
    *target = value;
    Ok(())
}
