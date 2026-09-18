//! The original native quote's one consuming host sequence bank.
use super::*;
use crate::backend::runtime::residency::storage::StorageIdentity;
use eredu_core::{
    GenerationSequenceBankRejection as Rejection, GenerationSequencePreparation,
    RetainedGenerationSequence,
};
use eredu_runtime::working_memory::{
    OriginalGenerationDecoderSource, OriginalGenerationSequenceBank, OriginalTextControlGuard,
    OriginalTextPreparationScopes, OwnedTextSpanWorkspace, PreparedTextControlWorkspace,
    RegisteredInferenceSourceWitness, SpanWorkspaceOwnerError, TextHostControlFacts,
    WorkingMemoryFundingRun,
};
use std::mem::size_of;

/// Existing native failures keep their original conversion. The new owning
/// promotion error enters core's concrete source owner without a native Box.
pub(super) enum AdmissionFailure {
    Native(Error),
    Sequence(BackendFailure),
}
impl From<Error> for AdmissionFailure {
    fn from(cause: Error) -> Self {
        Self::Native(cause)
    }
}
impl AdmissionFailure {
    pub(super) fn retained<E: std::error::Error + Send + Sync + 'static>(
        cause: E,
        sequence: bool,
    ) -> Self {
        if sequence {
            Self::Sequence(BackendFailure::new(
                BackendFailureKind::InvalidSession,
                cause,
            ))
        } else {
            Self::Native(Error::Other(Box::new(cause)))
        }
    }

    pub(super) fn into_backend(self) -> BackendFailure {
        match self {
            Self::Native(cause) => cause.into_backend_failure(),
            Self::Sequence(cause) => cause,
        }
    }
}

/// Lexical destruction only, not custody. The source's actual reservation/run
/// remains owned outside this loan. Nested guards are installed immediately
/// after each accepted owner transition, so source retirement precedes it on
/// error/unwind. No allocation, source clone, account hold or late quote.
pub(super) struct DecoderStaging<'a> {
    source: &'a mut Option<OriginalGenerationDecoderSource>,
}
impl<'a> DecoderStaging<'a> {
    pub(super) fn new(source: &'a mut Option<OriginalGenerationDecoderSource>) -> Self {
        Self { source }
    }
    pub(super) fn reborrow(&mut self) -> DecoderStaging<'_> {
        DecoderStaging::new(self.source)
    }
    pub(super) fn has_source(&self) -> bool {
        self.source.is_some()
    }
    pub(super) fn pending(&mut self) -> &mut Option<OriginalGenerationDecoderSource> {
        self.source
    }
}
impl Drop for DecoderStaging<'_> {
    fn drop(&mut self) {
        #[cfg(test)]
        let had_source = self.source.is_some();
        drop(self.source.take());
        #[cfg(test)]
        if had_source {
            fixture::decoder_retired(std::ptr::from_ref(self.source) as usize);
        }
    }
}

/// Common cold admission staging, including no-decoder branches. The shared
/// quote-control fact prices this once for capture/sequence or legacy diagnostics.
/// Source/destination allocations themselves enter only original decoder R.
pub(super) fn decoder_staging_control_bytes() -> Option<u64> {
    [
        // Pre-candidate source, two nested lexical cleanups, and consuming bank
        // attachment/return controls. Source/buffers themselves are original R.
        size_of::<Option<OriginalGenerationDecoderSource>>(),
        size_of::<DecoderStaging<'static>>(),
        size_of::<DecoderStaging<'static>>(),
        size_of::<&mut eredu_runtime::working_memory::WorkingMemoryReservation>(),
        size_of::<&mut Option<OriginalGenerationDecoderSource>>(),
        size_of::<Result<WorkingMemoryFundingRun, WorkingMemoryError>>(),
        size_of::<Result<Option<OriginalGenerationDecoderSource>, BackendFailure>>(),
        size_of::<Result<(), BackendFailure>>(),
        size_of::<Result<OriginalGenerationSequenceBank, BackendFailure>>(),
    ]
    .into_iter()
    .try_fold(0usize, usize::checked_add)
    .and_then(|n| u64::try_from(n).ok())
}

/// Capture keeps its span in the existing installation. Sequence-only requests
/// keep that same original span here. Neither path constructs a second hold.
#[derive(Debug)]
pub(super) struct SequenceQuotation {
    pub(super) input: Option<super::token_input::InputQuotation>,
    pending: RefCell<Option<OriginalGenerationSequenceBank>>,
    span: Option<OwnedTextSpanWorkspace>,
    // Historical quote aliases remain protected after the bank has been taken.
    controls: OriginalTextControlGuard,
}
impl SequenceQuotation {
    pub(super) fn validate_saved_host_handoff(&self) -> Result<(), WorkingMemoryError> {
        let pending = self
            .pending
            .try_borrow()
            .map_err(|_| WorkingMemoryError::UnknownBound)?;
        if pending.is_some() {
            return Err(WorkingMemoryError::UnknownBound);
        }
        Ok(())
    }
    pub(super) fn take_native_storage(
        &mut self,
        run: &eredu_runtime::working_memory::WorkingMemoryFundingRun,
        selection: &eredu_runtime::working_memory::NativeStorageSelection,
    ) -> Result<Option<crate::backend::runtime::residency::storage::native_storage::Bank>, Error>
    {
        self.span.as_mut().ok_or_else(|| memory(WorkingMemoryError::IdentityMismatch))?
            .take_native_storage_bank::<crate::backend::runtime::residency::storage::native_storage::MlxNativeStorage>(run, selection).map_err(memory)
    }
    pub(super) fn take_output_source_constructions(
        &mut self,
    ) -> Result<Option<eredu_runtime::working_memory::OriginalHostSourceBank>, Error> {
        self.span
            .as_mut()
            .ok_or_else(|| memory(WorkingMemoryError::IdentityMismatch))?
            .take_output_source_constructions()
            .map_err(memory)
    }

    pub(super) fn take_host_destinations(
        &mut self,
    ) -> Result<Option<eredu_runtime::working_memory::OriginalHostDestinationBank>, Error> {
        self.span
            .as_mut()
            .ok_or_else(|| memory(WorkingMemoryError::IdentityMismatch))?
            .take_host_destinations()
            .map_err(memory)
    }

    pub(super) fn take_submission_tracking(
        &mut self,
        preparation: &InferenceTextPreparation,
    ) -> Result<Option<eredu_runtime::working_memory::OriginalSubmissionTracking>, Error> {
        self.span
            .as_mut()
            .ok_or_else(|| memory(WorkingMemoryError::IdentityMismatch))?
            .take_submission_tracking(preparation)
            .map_err(memory)
    }

    pub(super) fn take_graph_metadata(
        &mut self,
        preparation: &InferenceTextPreparation,
    ) -> Result<Option<eredu_runtime::working_memory::OriginalGraphMetadata>, Error> {
        self.span
            .as_mut()
            .ok_or_else(|| memory(WorkingMemoryError::IdentityMismatch))?
            .take_graph_metadata(preparation)
            .map_err(memory)
    }

    pub(super) fn take_prefill_scopes(
        &mut self,
    ) -> Result<Option<eredu_runtime::working_memory::OriginalTextPrefillScopes>, Error> {
        self.span
            .as_mut()
            .ok_or_else(|| memory(WorkingMemoryError::IdentityMismatch))?
            .take_prefill_scopes()
            .map_err(memory)
    }

    pub(super) fn take_prediction_scopes(
        &mut self,
    ) -> Result<Option<eredu_runtime::working_memory::OriginalTextPredictionScopes>, Error> {
        self.span
            .as_mut()
            .ok_or_else(|| memory(WorkingMemoryError::IdentityMismatch))?
            .take_prediction_scopes()
            .map_err(memory)
    }

    pub(super) fn install_decoder(
        &mut self,
        source: &mut Option<OriginalGenerationDecoderSource>,
    ) -> Result<(), BackendFailure> {
        // Unique not-yet-exposed quote: no RefCell loan or callback survives the
        // consuming bank attachment. A rejected source stays under its caller's
        // lexical cleanup; it cannot become charged to a foreign bank.
        let bank = self
            .pending
            .get_mut()
            .take()
            .expect("original unextracted decoder bank");
        *self.pending.get_mut() = Some(bank.with_decoder_source(source)?);
        Ok(())
    }
    pub(super) fn take_preparation_scopes(
        &mut self,
    ) -> Result<Option<OriginalTextPreparationScopes>, Error> {
        self.span
            .as_mut()
            .ok_or_else(|| memory(WorkingMemoryError::IdentityMismatch))?
            .take_preparation_scopes()
            .map_err(memory)
    }

    pub(super) fn from_span(span: &mut OwnedTextSpanWorkspace) -> Option<Self> {
        let bank = span.take_generation_sequence_bank()?;
        Some(Self {
            input: super::token_input::InputQuotation::from_span(span),
            pending: RefCell::new(Some(bank)),
            span: None,
            controls: span.control_guard(),
        })
    }

    pub(super) fn promote(
        accepted: IncrementalInferenceQuote,
        funding: &WorkingMemoryFundingRun,
        request: &InferenceRequest,
    ) -> Result<Self, AdmissionFailure> {
        let (span, witness) = accepted
            .into_funded_text_span_workspace(
                funding,
                request
                    .memory_reservation()
                    .expect("accepted original reservation"),
            )
            .map_err(|cause| {
                AdmissionFailure::Sequence(BackendFailure::new(
                    BackendFailureKind::InvalidSession,
                    cause,
                ))
            })?;
        Self::from_promoted(span, witness)
    }

    pub(super) fn promote_saved(
        accepted: eredu_runtime::working_memory::CopyPreparationInferenceQuote<StorageIdentity>,
        funding: &WorkingMemoryFundingRun,
        request: &InferenceRequest,
    ) -> Result<Self, AdmissionFailure> {
        let (span, witness) = accepted
            .into_funded_text_span_workspace(
                funding,
                request
                    .memory_reservation()
                    .expect("accepted original reservation"),
            )
            .map_err(|cause| {
                AdmissionFailure::Sequence(BackendFailure::new(
                    BackendFailureKind::InvalidSession,
                    cause,
                ))
            })?;
        Self::from_promoted(span, witness)
    }

    fn from_promoted(
        mut span: OwnedTextSpanWorkspace,
        witness: Option<RegisteredInferenceSourceWitness>,
    ) -> Result<Self, AdmissionFailure> {
        // The full original guard keeps declared source health/custody; this
        // temporary witness is not an independent bank or result owner.
        drop(witness);
        let sequence = Self {
            input: super::token_input::InputQuotation::from_span(&mut span),
            pending: RefCell::new(span.take_generation_sequence_bank()),
            controls: span.control_guard(),
            span: Some(span),
        };
        Ok(sequence)
    }

    pub(super) fn control_guard(&self) -> OriginalTextControlGuard {
        self.controls.clone()
    }

    fn take(&self) -> Result<OriginalGenerationSequenceBank, BackendFailure> {
        let bank = {
            let mut pending = self
                .pending
                .try_borrow_mut()
                .map_err(|_| Rejection::Busy.into_backend_failure())?;
            pending.take()
        };
        bank.ok_or_else(|| Rejection::Unavailable.into_backend_failure())
    }
}

/// New concrete wrapper/error controls. The enclosing quote allocation is
/// priced separately from its final actual type by owner_control_bytes().
pub(super) fn control_bytes() -> Result<u64, Error> {
    let promotion = BackendFailure::source_retention_peak_bytes::<SpanWorkspaceOwnerError>()
        .ok_or_else(|| memory(WorkingMemoryError::Overflow))?;
    let bytes = [
        size_of::<SequenceQuotation>(),
        size_of::<Option<SequenceQuotation>>(),
        size_of::<Result<SequenceQuotation, AdmissionFailure>>(),
        size_of::<AdmissionFailure>(),
        // Native promotion arguments/result/destructuring and bank handoff.
        size_of::<IncrementalInferenceQuote>().max(size_of::<
            eredu_runtime::working_memory::CopyPreparationInferenceQuote<StorageIdentity>,
        >()),
        // The saved-source wrapper moves this sealed copy owner beside its
        // incremental proof until shared promotion has preserved the source pin.
        size_of::<eredu_runtime::working_memory::RegisteredWorkspaceCopy<StorageIdentity>>(),
        size_of::<OwnedTextSpanWorkspace>(),
        size_of::<
            Result<
                (
                    OwnedTextSpanWorkspace,
                    Option<RegisteredInferenceSourceWitness>,
                ),
                SpanWorkspaceOwnerError,
            >,
        >(),
        size_of::<OriginalGenerationSequenceBank>(),
        size_of::<Option<OriginalGenerationSequenceBank>>(),
        size_of::<std::cell::RefMut<'_, Option<OriginalGenerationSequenceBank>>>(),
        size_of::<&mut OwnedTextSpanWorkspace>(),
        size_of::<OriginalTextControlGuard>(),
        size_of::<Option<&GenerationSequencePreparation<'_, '_>>>(),
        size_of::<Result<OriginalGenerationSequenceBank, BackendFailure>>(),
        size_of::<Result<RetainedGenerationSequence, BackendFailure>>(),
        promotion,
    ]
    .into_iter()
    .try_fold(0usize, usize::checked_add)
    .and_then(|n| u64::try_from(n).ok())
    .ok_or_else(|| memory(WorkingMemoryError::Overflow))?;
    Ok(bytes)
}

pub(super) fn capture_control_bytes() -> Result<u64, Error> {
    use crate::backend::runtime::residency::storage::StorageIdentity;
    use eredu_runtime::working_memory::FailedCapturePlanPublication;
    let publication = BackendFailure::source_retention_peak_bytes::<
        FailedCapturePlanPublication<StorageIdentity>,
    >()
    .and_then(|n| u64::try_from(n).ok())
    .ok_or_else(|| memory(WorkingMemoryError::Overflow))?;
    control_bytes()?
        .checked_add(publication)
        .ok_or_else(|| memory(WorkingMemoryError::Overflow))
}

pub(super) fn seal(
    quote: IncrementalInferenceQuote,
    claim: Option<&GenerationSequencePreparation<'_, '_>>,
    tracking: Option<eredu_runtime::working_memory::SubmissionTrackingFacts>,
    graph: Option<eredu_runtime::working_memory::GraphMetadataFacts>,
    prefill: Option<eredu_runtime::working_memory::TextPrefillScopeFacts>,
    native: Option<
        eredu_runtime::working_memory::PreparedNativeStoragePlan<
            crate::backend::runtime::residency::storage::native_storage::MlxNativeStorage,
        >,
    >,
) -> Result<IncrementalInferenceQuote, Error> {
    let controls = prepare_controls(
        quote.geometry(),
        quote.span_workspace().plan(),
        claim,
        tracking,
        graph,
        prefill,
        native,
    )?;
    quote
        .with_span_workspace_and_text_controls(controls)
        .map_err(|cause| Error::Other(Box::new(cause)))
}

pub(super) fn seal_saved(
    quote: eredu_runtime::working_memory::CopyPreparationInferenceQuote<StorageIdentity>,
    claim: Option<&GenerationSequencePreparation<'_, '_>>,
    tracking: Option<eredu_runtime::working_memory::SubmissionTrackingFacts>,
    graph: Option<eredu_runtime::working_memory::GraphMetadataFacts>,
    prefill: Option<eredu_runtime::working_memory::TextPrefillScopeFacts>,
    native: Option<
        eredu_runtime::working_memory::PreparedNativeStoragePlan<
            crate::backend::runtime::residency::storage::native_storage::MlxNativeStorage,
        >,
    >,
) -> Result<eredu_runtime::working_memory::CopyPreparationInferenceQuote<StorageIdentity>, Error> {
    let controls = prepare_controls(
        quote.geometry(),
        quote.span_workspace().plan(),
        claim,
        tracking,
        graph,
        prefill,
        native,
    )?;
    quote
        .with_span_workspace_and_text_controls(controls)
        .map_err(|cause| Error::Other(Box::new(cause)))
}

fn prepare_controls(
    geometry: eredu_core::InferenceGeometry,
    plan: &eredu_runtime::working_memory::InferenceSpanWorkspacePlan,
    claim: Option<&GenerationSequencePreparation<'_, '_>>,
    tracking: Option<eredu_runtime::working_memory::SubmissionTrackingFacts>,
    graph: Option<eredu_runtime::working_memory::GraphMetadataFacts>,
    prefill: Option<eredu_runtime::working_memory::TextPrefillScopeFacts>,
    native: Option<
        eredu_runtime::working_memory::PreparedNativeStoragePlan<
            crate::backend::runtime::residency::storage::native_storage::MlxNativeStorage,
        >,
    >,
) -> Result<PreparedTextControlWorkspace, Error> {
    let admission = super::owner_control_bytes()?
        .checked_add(control_bytes()?)
        .ok_or_else(|| memory(WorkingMemoryError::Overflow))?;
    let controls = PreparedTextControlWorkspace::prepare_controls(
        geometry,
        plan,
        TextHostControlFacts::new(
            Some(admission),
            Some(0),
            Some(super::super::text_funding::text_work_control_bytes(
                geometry.max_output_tokens,
            )?),
        ),
    )
    .map_err(memory)?
    .with_preparation_scopes(super::preparation::facts()?)
    .map_err(memory)?
    .with_prediction_scopes(super::prediction::facts()?)
    .map_err(memory)?;
    let controls = match claim {
        Some(claim) => controls.with_generation_sequence(claim).map_err(memory)?,
        None => controls,
    };
    let controls = match tracking {
        Some(facts) => controls.with_submission_tracking(facts).map_err(memory)?,
        None => controls,
    };
    let controls = match graph {
        Some(facts) => controls.with_graph_metadata(facts).map_err(memory)?,
        None => controls,
    };
    let controls = match prefill {
        Some(facts) => {
            #[cfg(all(test, target_vendor = "apple", feature = "metal", not(feature = "cuda")))]
            let facts = super::prefill_tests::pointwise_prefill_controls(facts)?;
            controls.with_prefill_scopes(facts).map_err(memory)?
        }
        None => controls,
    };
    let controls = match native {
        Some(plan) => controls.with_native_storage(plan).map_err(memory)?,
        None => controls,
    };
    #[cfg(all(test, target_vendor = "apple", feature = "metal", not(feature = "cuda")))]
    let controls = super::prefill_tests::host_destination_controls(controls)?;
    Ok(controls)
}

/// Preflight neither invokes allocating native validators nor grants native
/// readiness. Prompt/Sampling still validate their actual target and opening.
fn preflight(
    runtime: &ModelRuntime<MlxBackend<'_>>,
    preparation: &InferenceTextPreparation,
    quote: &TextExecutionQuote,
) -> Result<(), Rejection> {
    let session = runtime.session();
    if session.poison.get()
        || !Rc::ptr_eq(&quote.session, &session.poison)
        || !quote.model_pool.same_domain(&session.payload.memory_pool)
        || !quote
            .context_pool
            .same_domain(runtime.backend().memory_pool())
        || !session.parameter_epoch_matches(quote.parameter_epoch)
        || quote.opening.require_sealed().is_err()
        || quote
            .request
            .validate_same_request(preparation.request())
            .is_err()
        || preparation
            .request()
            .validate(
                session
                    .payload
                    .model
                    .erased()
                    .inference_execution_identity(),
                quote.request.geometry(),
            )
            .is_err()
    {
        return Err(Rejection::IdentityMismatch);
    }
    let idle = {
        let authority = session
            .authority
            .try_borrow()
            .map_err(|_| Rejection::Busy)?;
        authority.require_idle().is_ok()
    };
    if !idle {
        return Err(Rejection::Busy);
    }
    Ok(())
}

pub(super) fn prepare(
    runtime: &ModelRuntime<MlxBackend<'_>>,
    preparation: &MlxTextPreparation,
    claim: GenerationSequencePreparation<'_, '_>,
) -> Result<RetainedGenerationSequence, BackendFailure> {
    let request = preparation
        .request
        .as_ref()
        .ok_or_else(|| Rejection::Unavailable.into_backend_failure())?;
    let quote = preparation
        .quote
        .as_ref()
        .ok_or_else(|| Rejection::Unavailable.into_backend_failure())?;
    let sequence = quote
        .sequence
        .as_ref()
        .ok_or_else(|| Rejection::Unavailable.into_backend_failure())?;
    preflight(runtime, request, quote).map_err(Rejection::into_backend_failure)?;
    // Read-only exact input-claim validation precedes I and R consumption.
    // The input slot loan ends before construction, owning errors or Drop;
    // construction repeats validation and a consumed failure cannot restore it.
    if let Some(input) = &sequence.input {
        input.prepare(request, &claim)?;
    }
    let bank = sequence.take()?;
    bank.prepare(request, claim)
}

#[cfg(test)]
pub(super) mod fixture;
