//! Original-account installation only. Execution dispatch and shared delivery
//! remain in the shared text driver; this module issues no native operation.
use super::*;
use eredu_core::{capture::SharedCapturePlan, TextStepContext};
use eredu_runtime::{
    capture::FundedCaptureSession,
    layered::{BoundCaptureSelection, PreparedCaptureSelection},
    working_memory::{
        MemoryLedger, OwnedTextSpanWorkspace, PreparedCaptureRun, RegisteredInferenceSourceWitness,
        WorkingMemoryError,
    },
};

/// The original collector, causal selection, source witness and accepted plan.
/// It owns neither a session nor its historical quote, so there is no cycle.
/// Collector/source aliases retire before the final span-plan owner. Escaped
/// host receipts independently retain the original cumulative capture H.
pub(in crate::composition::mlx::session) struct InstalledCapture {
    collector: FundedCaptureSession,
    first_prediction: u64,
    continuation: bool,
    selection: PreparedCaptureSelection,
    witness: RegisteredInferenceSourceWitness,
    rows: Option<crate::composition::mlx::replicated_text::NativeOpeningRowsOwner>,
    span: OwnedTextSpanWorkspace,
}

/// Borrowed accepted physical opening. Prompt fragments and the saved one-token
/// continuation use the same native prefill driver with distinct source checks.
#[derive(Debug)]
pub(super) enum InstalledPrefillCapture<'a> {
    Prompt(eredu_runtime::working_memory::AdmittedPrefillCapture<'a>),
    Continuation(eredu_runtime::working_memory::AdmittedCaptureContinuation<'a>),
}

/// Temporary conversion payload has the same strict destruction order as the
/// installed owner, including every validation/conversion error path.
pub(super) struct InstallationPayload {
    bank: Option<PreparedCaptureRun>,
    selection: PreparedCaptureSelection,
    witness: RegisteredInferenceSourceWitness,
    rows: Option<crate::composition::mlx::replicated_text::NativeOpeningRowsOwner>,
    span: OwnedTextSpanWorkspace,
}

impl InstallationPayload {
    pub(super) fn original_controls(
        &self,
    ) -> eredu_runtime::working_memory::OriginalTextControlGuard {
        self.span.control_guard()
    }
}

// Native forwarding controls only. The existing selection fact covers bound
// construction/assembly, P covers the accepted view constructor, and the original
// bank covers its observer's bound plus accepted-view reference.
// Sum the returned/caller pair, two reference-only call pairs, the submission
// Option and the shared helper's optional reference before original Q sealing.
pub(super) fn prefill_borrow_control_bytes() -> Option<u64> {
    type Bound = InstalledPrefillCapture<'static>;
    type Parts = (&'static mut FundedCaptureSession, Bound);
    type Call = (&'static mut FundedCaptureSession, &'static Bound);
    type TokenError = eredu_runtime::ReplicatedTextSessionError<eredu_nn::Error, Error, Error>;
    type CheckedPaths =
        Result<Option<&'static eredu_runtime::PreparedLayeredObservationPaths>, TokenError>;
    let bytes = std::mem::size_of::<Result<Parts, Error>>()
        .checked_add(std::mem::size_of::<Parts>())?
        .checked_add(std::mem::size_of::<Call>().checked_mul(2)?)?
        .checked_add(std::mem::size_of::<Option<Call>>())?
        .checked_add(std::mem::size_of::<Option<&Bound>>())?
        .checked_add(std::mem::size_of::<
            Option<(
                &eredu_runtime::capture::FundedCaptureCheckpoint,
                eredu_core::OriginalTextResumeKind,
            )>,
        >())?
        .checked_add(std::mem::size_of::<
            Option<&eredu_runtime::working_memory::AdmittedCaptureContinuation<'static>>,
        >())?
        .checked_add(std::mem::size_of::<bool>())?
        // Current-token validation return and its existing BeforeStateMutation
        // envelope. Dynamic native error sources/retirement remain separate.
        .checked_add(std::mem::size_of::<CheckedPaths>())?
        .checked_add(std::mem::size_of::<TokenError>())?
        .checked_add(std::mem::size_of::<Box<TokenError>>())?;
    u64::try_from(bytes).ok()
}

impl InstalledCapture {
    // The only production caller moves the original pending owners together.
    pub(super) fn new(
        bank: PreparedCaptureRun,
        selection: PreparedCaptureSelection,
        witness: RegisteredInferenceSourceWitness,
        span: OwnedTextSpanWorkspace,
    ) -> Result<Self, Error> {
        Self::new_with_opening_rows(bank, selection, witness, span, None)
    }

    pub(super) fn new_with_opening_rows(
        bank: PreparedCaptureRun,
        selection: PreparedCaptureSelection,
        witness: RegisteredInferenceSourceWitness,
        span: OwnedTextSpanWorkspace,
        rows: Option<crate::composition::mlx::replicated_text::NativeOpeningRowsOwner>,
    ) -> Result<Self, Error> {
        Self::new_with_opening_rows_and_error_allowance(bank, selection, witness, span, rows)
            .map(|(installed, _unused_allowance)| installed)
    }

    pub(super) fn new_with_opening_rows_and_error_allowance(
        bank: PreparedCaptureRun,
        selection: PreparedCaptureSelection,
        witness: RegisteredInferenceSourceWitness,
        span: OwnedTextSpanWorkspace,
        rows: Option<crate::composition::mlx::replicated_text::NativeOpeningRowsOwner>,
    ) -> Result<(Self, super::text_error::OriginalErrorAllowance), Error> {
        Self::construct(bank, selection, witness, span, rows, None)
    }

    pub(super) fn from_checkpoint(
        bank: PreparedCaptureRun,
        selection: PreparedCaptureSelection,
        witness: RegisteredInferenceSourceWitness,
        span: OwnedTextSpanWorkspace,
        checkpoint: &eredu_runtime::capture::FundedCaptureCheckpoint,
        kind: eredu_core::OriginalTextResumeKind,
    ) -> Result<Self, Error> {
        Self::construct(
            bank,
            selection,
            witness,
            span,
            None,
            Some((checkpoint, kind)),
        )
        .map(|(installed, _unused_allowance)| installed)
    }

    fn construct(
        bank: PreparedCaptureRun,
        selection: PreparedCaptureSelection,
        witness: RegisteredInferenceSourceWitness,
        span: OwnedTextSpanWorkspace,
        rows: Option<crate::composition::mlx::replicated_text::NativeOpeningRowsOwner>,
        checkpoint: Option<(
            &eredu_runtime::capture::FundedCaptureCheckpoint,
            eredu_core::OriginalTextResumeKind,
        )>,
    ) -> Result<(Self, super::text_error::OriginalErrorAllowance), Error> {
        let mut pending = InstallationPayload {
            bank: Some(bank),
            rows,
            selection,
            witness,
            span,
        };
        let allowance = super::text_error::OriginalErrorAllowance::installation_payload(&pending);
        let result = (|| {
            pending
                .selection
                .validate_sources(
                    pending.bank.as_ref().expect("owned bank").source(),
                    pending.selection.paths(),
                )
                .map_err(|error| Error::Other(Box::new(error)))?;
            let geometry = pending.span.workspace().plan().geometry();
            if let Some((checkpoint, _)) = checkpoint {
                checkpoint
                    .validate_continuation_geometry(geometry)
                    .map_err(|cause| Error::Other(Box::new(cause)))?;
                if !checkpoint
                    .source()
                    .same_storage(pending.bank.as_ref().expect("owned bank").source())
                {
                    return Err(mismatch());
                }
            } else {
                pending
                    .selection
                    .bind_geometry(geometry)
                    .map_err(|cause| Error::Other(Box::new(cause)))?;
            }
            let bank = pending.bank.take().expect("owned bank");
            let collector = match checkpoint {
                Some((checkpoint, eredu_core::OriginalTextResumeKind::Restore)) => checkpoint
                    .into_restoration(bank)
                    .map_err(|cause| Error::Other(Box::new(cause)))?,
                Some((checkpoint, eredu_core::OriginalTextResumeKind::Branch)) => checkpoint
                    .into_continuation(bank)
                    .map_err(|cause| Error::Other(Box::new(cause)))?,
                None => bank
                    .into_capture_session()
                    .map_err(|cause| Error::Other(Box::new(cause)))?,
            };
            let InstallationPayload {
                rows,
                selection,
                witness,
                span,
                ..
            } = pending;
            Ok(Self {
                rows,
                continuation: checkpoint.is_some(),
                collector,
                first_prediction: checkpoint
                    .map_or(0, |(checkpoint, _)| checkpoint.next_prediction()),
                selection,
                witness,
                span,
            })
        })();
        match result {
            Ok(installed) => Ok((installed, allowance)),
            Err(cause) => allowance.finish(Err(cause)),
        }
    }

    pub(in crate::composition::mlx::session) fn source(&self) -> &SharedCapturePlan {
        self.collector.source()
    }

    pub(in crate::composition::mlx::session) fn source_witness(
        &self,
    ) -> &RegisteredInferenceSourceWitness {
        &self.witness
    }

    pub(in crate::composition::mlx::session) fn saved_selection(
        &self,
    ) -> &PreparedCaptureSelection {
        &self.selection
    }

    pub(in crate::composition::mlx::session) fn collector(&self) -> &FundedCaptureSession {
        &self.collector
    }

    pub(in crate::composition::mlx::session) fn collector_mut(
        &mut self,
    ) -> &mut FundedCaptureSession {
        &mut self.collector
    }

    /// The same immutable companion and candidate retained before admission.
    /// No native permission or opening-inventory proof is created by this view.
    pub(in crate::composition::mlx::session) fn prefill_selection(
        &self,
    ) -> Result<BoundCaptureSelection<'_>, Error> {
        self.selection
            .validate_sources(self.source(), self.selection.paths())
            .map_err(|error| Error::Other(Box::new(error)))?;
        let geometry = self.span.workspace().plan().geometry();
        let bound = if self.continuation && self.first_prediction == 0 {
            self.selection.bind_prompt_prefix(geometry)
        } else {
            self.selection.bind_geometry(geometry)
        };
        bound.map_err(|error| Error::Other(Box::new(error)))
    }

    /// Borrow the actual installed companion and collector as disjoint fields.
    /// Validate before the caller acquires any native text work.
    pub(in crate::composition::mlx::session) fn prefill_parts(
        &mut self,
    ) -> Result<(&mut FundedCaptureSession, InstalledPrefillCapture<'_>), Error> {
        self.selection
            .validate_sources(self.collector.source(), self.selection.paths())
            .map_err(|error| Error::Other(Box::new(error)))?;
        let admitted = if self.first_prediction == 0 {
            let geometry = self.span.workspace().plan().geometry();
            let bound = if self.continuation {
                self.selection.bind_prompt_prefix(geometry)
            } else {
                self.selection.bind_geometry(geometry)
            }
            .map_err(|cause| Error::Other(Box::new(cause)))?;
            InstalledPrefillCapture::Prompt(self.span.prefill_capture(bound).map_err(memory)?)
        } else {
            InstalledPrefillCapture::Continuation(
                self.span
                    .capture_continuation(&self.selection, self.first_prediction)
                    .map_err(memory)?,
            )
        };
        Ok((&mut self.collector, admitted))
    }

    /// Borrow the already protected original plan, without another reservation.
    pub(in crate::composition::mlx::session) fn span_workspace(&self) -> &OwnedTextSpanWorkspace {
        &self.span
    }

    /// Read-only check before each original step, including sources registered
    /// after the reservation. This does not retrofit pins into existing scopes.
    pub(in crate::composition::mlx::session) fn validate_sources(
        &self,
        pool: &MemoryLedger,
    ) -> Result<(), Error> {
        self.witness.validate(pool).map_err(memory)
    }
}

/// Authenticate the complete original association before consuming its bank.
/// Core owns the subsequent single Instrumentation readiness vote. This private
/// entry neither enables the public options gate nor creates/finishes a step.
pub(super) fn install(
    runtime: &ModelRuntime<MlxBackend<'_>>,
    state: &mut MlxTextGenerationState,
    preparation: &MlxTextPreparation,
    source: &SharedCapturePlan,
    context: &TextStepContext,
) -> Result<(), BackendFailure> {
    let (installed, allowance) = take_installation(runtime, state, preparation, source, context)
        .map_err(BackendFailure::from_error)?;
    // Same allowance follows the first consumed pending bank through conversion
    // and row installation. No preflight/replay path mints another owner.
    let result = (|| {
        debug_assert_eq!(installed.collector().spent_steps(), 0);
        debug_assert!(!installed.collector().has_pending_step());
        if let Some(rows) = &installed.rows {
            #[cfg(test)]
            super::text_quote::intercept_consumed_row_installation(
                runtime,
                preparation,
                source,
                rows,
            )?;
            runtime
                .session()
                .payload
                .model
                .erased()
                .install_opening_rows(rows)?;
        }
        state.funded_capture = Some(installed);
        Ok(())
    })();
    allowance.finish_backend(result)
}

fn take_installation(
    runtime: &ModelRuntime<MlxBackend<'_>>,
    state: &MlxTextGenerationState,
    preparation: &MlxTextPreparation,
    source: &SharedCapturePlan,
    context: &TextStepContext,
) -> Result<(InstalledCapture, super::text_error::OriginalErrorAllowance), Error> {
    runtime.validate_session_admission()?;
    let session = runtime.session();
    session.validate_backend(runtime.backend())?;
    // Installation is a cold identity/health check, not a recovery/reaping site.
    session
        .authority
        .borrow()
        .require_idle()
        .map_err(|error| Error::Other(Box::new(error)))?;
    require_empty(state)?;
    let request = preparation.request.as_ref().ok_or_else(mismatch)?;
    let quote = preparation.quote.as_ref().ok_or_else(mismatch)?;
    if !state
        .sampling
        .quote
        .as_ref()
        .is_some_and(|actual| actual.same_owner(quote))
    {
        return Err(mismatch());
    }
    request
        .validate_initial_ready_context(context)
        .map_err(memory)?;
    quote.validate(runtime, request.request())?;
    let reservation = request.request().memory_reservation();
    reservation
        .validate_ledger(&session.payload.memory_ledger)
        .map_err(memory)?;
    reservation
        .validate_ledger(runtime.backend().memory_ledger())
        .map_err(memory)?;
    state
        .funding
        .as_ref()
        .ok_or_else(mismatch)?
        .validate_reservation(reservation)
        .map_err(memory)?;
    if state.sampling.next_prediction != 0 || quote.local_prediction(0)? != 0 {
        return Err(mismatch());
    }
    if !state
        .sampling
        .inference_retention
        .requests()
        .any(|retained| retained.validate_same_request(quote.request()).is_ok())
    {
        return Err(mismatch());
    }
    let mut parameter_epoch = state.sampling.parameter_epoch;
    session.validate_parameter_epoch(&mut parameter_epoch)?;
    quote.validate_frontier(runtime, 0)?;
    let retained = session
        .payload
        .model
        .erased()
        .retained_inference_authority()?;
    quote.validate_opening(&retained)?;
    quote.validate_capture_source(session, source)?;
    // Source health is rechecked by take_capture_installation before taking it.
    // No RefCell/Usage guard survives bank conversion or either owner's drop.
    quote.take_capture_installation_with_error_allowance(session, source)
}

/// Central mutual exclusion for this new slot and the unchanged legacy slot.
/// Legacy managed configuration already rejects quoted state; it must continue
/// to do so until the enclosing original options route is connected.
fn require_empty(state: &MlxTextGenerationState) -> Result<(), Error> {
    if state.capture.is_some() || state.funded_capture.is_some() {
        return Err(memory(WorkingMemoryError::PreparationAlreadyStarted));
    }
    Ok(())
}
fn memory(error: WorkingMemoryError) -> Error {
    Error::Other(Box::new(error))
}
fn mismatch() -> Error {
    memory(WorkingMemoryError::IdentityMismatch)
}

#[cfg(all(
    test,
    target_vendor = "apple",
    feature = "metal",
    not(feature = "cuda")
))]
mod tests;
