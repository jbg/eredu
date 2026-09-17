//! Named original native control diagnostics; custody comes only from acceptance.
use super::*;
use crate::working_memory::storage::bounded_pin::{
    OriginalPrefillStoragePinSlots, PinLayout, PreparedPrefillStoragePinPlan,
};
use crate::working_memory::storage::bounded_publication::{
    BatchPublicationLayout, OriginalPrefillStoragePublicationSlots,
    PreparedPrefillStoragePublicationPlan,
};
use crate::working_memory::storage::capture_publication::{
    CapturePlanStorageKey, PreparedCapturePlanPublication, PublicationLayout,
};
use crate::working_memory::{WorkingMemoryFundingScope, WorkingMemoryPool};
use eredu_core::capture::SharedCapturePlan;
mod capture_publication;
mod graph_metadata;
mod host_destinations;
mod native_storage;
pub use host_destinations::{
    HostDestinationCause, HostDestinationFacts, HostSourceConstructionFacts, HostSourceConstructionProgram, OriginalHostSourceProgramBanks, OriginalHostSourceProgramError,
    OriginalHostDestinationBank, OriginalHostMetadataCustody, OriginalHostMetadataVec,
    OriginalHostSourceBank, OriginalHostSourceConstruction, OriginalHostSourceError,
    OriginalHostSourceFailure, OriginalHostSourceFailureCause, OriginalHostSourceReceipt,
    OriginalHostSourceRefusal, OriginalHostVec, OriginalHostVecError,
};
pub use host_destinations::{
    HostSourcePeakSelection, OriginalHostSourcePeakCapacity, OriginalHostSourcePending,
};
pub use native_storage::{
    NativeEquationStorage, NativePrefillEnvelope, NativePrefillEnvelopeBuilder,
    NativeStorageCarryoverReport, NativeStorageError, NativeStorageObservation,
    NativeStorageRegistration, NativeStorageSelection, OriginalNativeBudgetCustody,
    OriginalNativePublication, OriginalNativeStorageBank, OriginalNativeStorageMechanism,
    PreparedNativeStoragePlan,
};
mod prediction;
mod prefill;
pub use prefill::{
    OriginalPrefillNativeCustody, OriginalPrefillRecoveryCustody, OriginalPrefillRootCustody,
    OriginalPrefillRootProjectionCustody, OriginalPrefillScopeRole, OriginalTextPrefillScopeSet,
    OriginalTextPrefillScopes, TextPrefillScopeFacts,
};
mod tracking;
pub use graph_metadata::{GraphMetadataFacts, OriginalGraphMetadata};
pub use tracking::{OriginalSubmissionTracking, SubmissionTrackingFacts};
mod prefill_capture;
mod preparation;
pub use prediction::{
    OriginalPredictionNativeCustody, OriginalPredictionRecoveryCustody,
    OriginalPredictionScopeRole, OriginalTextPredictionScopeSet, OriginalTextPredictionScopes,
    TextPredictionScopeFacts,
};
mod sequence;
pub(in crate::working_memory) use sequence::OriginalTokenDomainBinding;
mod token_input;
pub use capture_publication::{FailedCapturePlanPublication, PendingCapturePlanPublication};
pub use prefill_capture::{AdmittedCaptureContinuation, AdmittedPrefillCapture};
pub use preparation::{
    OriginalPreparationScopeCustody, OriginalTextPreparationScopes, TextPreparationScopeFacts,
};
use sequence::SequenceBinding;
#[cfg(test)]
pub(in crate::working_memory) use sequence::SequenceExtractionError;
pub use sequence::{
    AggregateGenerationDecoderInput, LoadedGenerationDecoderInput, OriginalGenerationDecoderInput,
    OriginalGenerationDecoderSource, OriginalGenerationSequenceBank,
};
pub use token_input::{
    OriginalTokenInputBank, OriginalTokenInputFailure, OriginalTokenInputLayout,
    OwnedPromptTokenIds,
};

fn checkpoint_geometry_error(
    cause: crate::capture::FundedCaptureCheckpointError,
) -> WorkingMemoryError {
    match cause {
        crate::capture::FundedCaptureCheckpointError::Host(
            crate::working_memory::CaptureRunHostError::Memory(cause),
        ) => cause,
        _ => WorkingMemoryError::IdentityMismatch,
    }
}

/// Backend mechanism facts for the actual finite original text program. Unknown
/// remains unknown. These values grant no host allocation or native operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TextHostControlFacts {
    admission: Option<u64>,
    carrier: Option<u64>,
    work: Option<u64>,
}
impl TextHostControlFacts {
    /// The native provider measures its actual admission/install controls, one
    /// active carrier and all potentially coexisting original work owners.
    pub const fn new(admission: Option<u64>, carrier: Option<u64>, work: Option<u64>) -> Self {
        Self {
            admission,
            carrier,
            work,
        }
    }
    /// Admission, pending and installed native owner controls.
    pub const fn admission_bytes(self) -> Option<u64> {
        self.admission
    }
    /// Fixed native carrier payload and constructor overlap.
    pub const fn carrier_bytes(self) -> Option<u64> {
        self.carrier
    }
    /// All original Prompt, Sampling and finite inference work controls.
    pub const fn work_bytes(self) -> Option<u64> {
        self.work
    }
    /// Check known overflow even when another required term is unknown.
    pub fn total_bytes(self) -> Result<Option<u64>, WorkingMemoryError> {
        let values = [self.admission, self.carrier, self.work];
        let sum = values.into_iter().flatten().try_fold(0u64, |a, b| {
            a.checked_add(b).ok_or(WorkingMemoryError::Overflow)
        })?;
        Ok(values.iter().all(Option::is_some).then_some(sum))
    }
}
#[derive(Debug, Clone)]
pub(in crate::working_memory) struct TextControlBinding {
    identity: Arc<()>,
    source: Option<eredu_core::SharedStorageIdentity>,
    sequence: Option<SequenceBinding>,
    prefill_paths: Option<crate::host_metadata::HostMetadataIdentity>,
    capture_first: Option<u64>,
    geometry: InferenceGeometry,
    facts: TextHostControlFacts,
    preparation_scopes: Option<TextPreparationScopeFacts>,
    prediction_scopes: Option<TextPredictionScopeFacts>,
    prefill_scopes: Option<TextPrefillScopeFacts>,
    tracking: Option<SubmissionTrackingFacts>,
    graph_metadata: Option<GraphMetadataFacts>,
    host_destinations: Option<HostDestinationFacts>,
    output_sources: Option<HostSourceConstructionFacts>,
    native_storage: Option<Arc<native_storage::plan::NativeStorageLayout>>,
    pins: Option<Arc<PinLayout>>,
    publication: Option<Arc<PublicationLayout>>,
    batches: Option<Arc<BatchPublicationLayout>>,
}
impl TextControlBinding {
    pub(in crate::working_memory) fn native_span_remainder(
        &self,
        workspace: &InferenceSpanWorkspace,
        span: &InferenceWorkspaceSpan,
        held: Option<u64>,
    ) -> Result<Option<u64>, WorkingMemoryError> {
        self.native_storage
            .as_ref()
            .map(|layout| layout.remainder(workspace, span, held))
            .transpose()
    }
    pub(in crate::working_memory) fn batch_layout(&self) -> Option<&Arc<BatchPublicationLayout>> {
        self.batches.as_ref()
    }
    pub(in crate::working_memory) fn publication_layout(&self) -> Option<&Arc<PublicationLayout>> {
        self.publication.as_ref()
    }
    pub(in crate::working_memory) fn pin_layout(&self) -> Option<&Arc<PinLayout>> {
        self.pins.as_ref()
    }
    pub(in crate::working_memory) fn source_identity(
        &self,
    ) -> Option<&eredu_core::SharedStorageIdentity> {
        self.source.as_ref()
    }
    pub(in crate::working_memory) fn same(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.identity, &other.identity)
            && self.source == other.source
            && self.sequence == other.sequence
            && self.prefill_paths == other.prefill_paths
            && self.capture_first == other.capture_first
            && self.geometry == other.geometry
            && self.facts == other.facts
            && self.preparation_scopes == other.preparation_scopes
            && self.prediction_scopes == other.prediction_scopes
            && self.prefill_scopes == other.prefill_scopes
            && self.tracking == other.tracking
            && self.graph_metadata == other.graph_metadata
            && self.host_destinations == other.host_destinations
            && match (&self.native_storage, &other.native_storage) {
                (None, None) => true,
                (Some(a), Some(b)) => Arc::ptr_eq(a, b),
                _ => false,
            }
            && match (&self.publication, &other.publication) {
                (None, None) => true,
                (Some(a), Some(b)) => Arc::ptr_eq(a, b),
                _ => false,
            }
            && match (&self.batches, &other.batches) {
                (None, None) => true,
                (Some(a), Some(b)) => Arc::ptr_eq(a, b),
                _ => false,
            }
            && match (&self.pins, &other.pins) {
                (None, None) => true,
                (Some(a), Some(b)) => Arc::ptr_eq(a, b),
                _ => false,
            }
    }
}
/// Immutable diagnostic tied to the actual plan/source/candidate. Clones retain
/// the same span plan, so earlier aliases receive its later aggregate custody.
/// Source payload remains independently owned/published; identity is not a pin.
#[derive(Clone, Debug)]
pub struct PreparedTextControlWorkspace {
    binding: TextControlBinding,
    plan: InferenceSpanWorkspacePlan,
}
impl PreparedTextControlWorkspace {
    /// No bytes can be substituted after sealing. This diagnostic constructor
    /// does not verify the native provider's implementation or fund that source.
    pub fn prepare(
        source: &SharedCapturePlan,
        geometry: InferenceGeometry,
        plan: &InferenceSpanWorkspacePlan,
        facts: TextHostControlFacts,
    ) -> Result<Self, WorkingMemoryError> {
        geometry
            .validate()
            .map_err(|_| WorkingMemoryError::IdentityMismatch)?;
        let admission = source.admission();
        let request = admission.request();
        if plan.geometry() != geometry
            || request.batch != geometry.batch_size
            || request.prompt_tokens != geometry.input_positions
            || request.max_predictions != geometry.max_output_tokens
            || admission.text_origin().map(|o| o.cached_positions)
                != Some(geometry.cached_positions)
        {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        Self::prepare_capture_source(source, geometry, plan, facts)
    }

    /// Same finite control constructor for a closed saved capture frontier.
    /// Its exact origin validates the fresh geometry before any binding birth.
    pub fn prepare_capture_continuation(
        checkpoint: &crate::capture::FundedCaptureCheckpoint,
        geometry: InferenceGeometry,
        plan: &InferenceSpanWorkspacePlan,
        facts: TextHostControlFacts,
    ) -> Result<Self, WorkingMemoryError> {
        checkpoint
            .validate_continuation_geometry(geometry)
            .map_err(checkpoint_geometry_error)?;
        if plan.geometry() != geometry {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        Self::prepare_capture_source(checkpoint.source(), geometry, plan, facts)
    }

    fn prepare_capture_source(
        source: &SharedCapturePlan,
        geometry: InferenceGeometry,
        plan: &InferenceSpanWorkspacePlan,
        facts: TextHostControlFacts,
    ) -> Result<Self, WorkingMemoryError> {
        geometry
            .cached_positions
            .checked_add(geometry.input_positions)
            .and_then(|n| n.checked_add(geometry.max_output_tokens))
            .ok_or(WorkingMemoryError::Overflow)?;
        facts.total_bytes()?;
        Ok(Self {
            binding: TextControlBinding {
                identity: Arc::new(()),
                source: Some(source.storage_identity().clone()),
                sequence: None,
                prefill_paths: None,
                capture_first: None,
                geometry,
                facts,
                preparation_scopes: None,
                prediction_scopes: None,
                prefill_scopes: None,
                tracking: None,
                graph_metadata: None,
                host_destinations: None,
                output_sources: None,
                native_storage: None,
                pins: None,
                publication: None,
                batches: None,
            },
            plan: plan.clone(),
        })
    }
    /// Bind the typed finite rows BEFORE original acceptance. Rebinding or an
    /// independently equal equation plan rejects; S is added once beside P+Q.
    pub fn with_prefill_storage_pins<K: Ord + Send + 'static>(
        mut self,
        pins: PreparedPrefillStoragePinPlan<K>,
    ) -> Result<Self, WorkingMemoryError> {
        if self.binding.pins.is_some() || self.binding.source.is_none() {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        self.binding.pins = Some(pins.into_layout(&self.plan)?);
        Ok(self)
    }
    /// Bind one exact physical C layout before the original seal. Its measured
    /// controls join the protected host term; newly required C is separate.
    pub fn with_capture_plan_publication<K: CapturePlanStorageKey>(
        mut self,
        publication: PreparedCapturePlanPublication<K>,
    ) -> Result<Self, WorkingMemoryError> {
        if self.binding.publication.is_some() {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        self.binding.publication = Some(
            publication.into_layout(
                &self.plan,
                self.binding
                    .source
                    .as_ref()
                    .ok_or(WorkingMemoryError::IdentityMismatch)?,
            )?,
        );
        Ok(self)
    }
    /// Bind finite new-key rows to the SAME typed original C layout. Original
    /// C publication must complete before the bank can issue a row. S is held
    /// once with P+Q; this builder adds no physical bytes or later allowance.
    pub fn with_prefill_storage_publications<K: CapturePlanStorageKey>(
        mut self,
        plan: PreparedPrefillStoragePublicationPlan<K>,
    ) -> Result<Self, WorkingMemoryError> {
        if self.binding.batches.is_some() {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let original = self
            .binding
            .publication
            .as_ref()
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        self.binding.batches = Some(plan.into_layout(&self.plan, original)?);
        Ok(self)
    }
    /// Cumulative controls for the finite new-key publication schedule.
    pub fn storage_publication_control_bytes(&self) -> u64 {
        self.binding.batches.as_ref().map_or(0, |p| p.bytes())
    }
    /// Exact immutable binding, including the original plan and any sealed
    /// typed pin/single-source/batch publication layout. Equal scalar facts alone do not match.
    pub fn same_binding(&self, other: &Self) -> bool {
        self.binding.same(&other.binding) && self.plan.same_plan(&other.plan)
    }
    /// Measured single-source publication controls; zero when absent.
    pub fn capture_publication_control_bytes(&self) -> u64 {
        self.binding
            .publication
            .as_ref()
            .map_or(0, |p| p.controls())
    }
    /// Exact new physical C reserved separately from protected host controls.
    pub fn capture_publication_source_bytes(&self) -> u64 {
        self.binding.publication.as_ref().map_or(0, |p| p.new_bytes)
    }
    /// Bounded-pin controls only; absent for the unchanged ordinary path.
    pub fn storage_pin_control_bytes(&self) -> u64 {
        self.binding.pins.as_ref().map_or(0, |p| p.bytes())
    }
    /// Original named facts; this does not grant their use.
    pub fn facts(&self) -> TextHostControlFacts {
        self.binding.facts
    }
    /// Exact originally supplied physical candidate.
    pub fn geometry(&self) -> InferenceGeometry {
        self.binding.geometry
    }
    /// Same actual shared equation schedule, including later custody.
    pub fn plan(&self) -> &InferenceSpanWorkspacePlan {
        &self.plan
    }
    /// Original capture source identity; absent for sequence-only controls.
    pub fn source_identity(&self) -> Option<&eredu_core::SharedStorageIdentity> {
        self.binding.source.as_ref()
    }
}
pub(super) fn control_metadata_bytes() -> Option<usize> {
    size_of::<PreparedTextControlWorkspace>()
        .checked_add(size_of::<TextControlBinding>())?
        .checked_add(size_of::<TextHostControlFacts>())?
        .checked_add(size_of::<OwnedTextSpanWorkspace>())?
        .checked_add(size_of::<OriginalTextControlGuard>())?
        .checked_add(size_of::<ReservedTextSpanWorkspace<'static>>())?
        .checked_mul(3)?
        .checked_add(2 * size_of::<usize>())?
        .checked_add(prefill_capture::control_peak_bytes()?)
}
impl InferenceSpanWorkspace {
    /// Original named control contribution, distinct from record/neutral P.
    pub fn text_controls(&self) -> Option<&PreparedTextControlWorkspace> {
        self.text_controls.as_ref()
    }
    pub(in crate::working_memory) fn control_binding(&self) -> Option<&TextControlBinding> {
        self.text_controls.as_ref().map(|c| &c.binding)
    }
    pub(in crate::working_memory) fn protected_peak_bytes(
        &self,
    ) -> Result<u64, WorkingMemoryError> {
        let p = self
            .retention_peak_bytes()
            .ok_or(WorkingMemoryError::Overflow)?;
        let q = match &self.text_controls {
            Some(c) => c
                .facts()
                .total_bytes()?
                .ok_or(WorkingMemoryError::UnknownBound)?,
            None => 0,
        };
        let s = self
            .text_controls
            .as_ref()
            .map_or(0, |c| c.storage_pin_control_bytes());
        let publication = self
            .text_controls
            .as_ref()
            .map_or(0, |c| c.capture_publication_control_bytes());
        let batches = self
            .text_controls
            .as_ref()
            .map_or(0, |c| c.storage_publication_control_bytes());
        let r = self
            .text_controls
            .as_ref()
            .map_or(0, |c| c.sequence_storage_bytes());
        let input = self
            .text_controls
            .as_ref()
            .and_then(|c| c.binding.sequence.as_ref())
            .and_then(|s| s.input.as_ref())
            .map_or(0, token_input::TokenInputBinding::bytes);
        #[cfg(debug_assertions)]
        if std::env::var_os("EREDU_ORIGINAL_QUOTE_TRACE").is_some() {
            eprintln!("ORIGINAL_QUOTE_PROTECTED chunk={} records={} retention={} input={} controls={} sequence={} pins={} publication={} batches={}",
                self.plan.geometry().prefill_chunk_positions, self.plan.records().len(),
                p, input, q, r, s, publication, batches);
            if let Some(controls) = self.text_controls.as_ref() {
                eprintln!("ORIGINAL_QUOTE_CONTROLS admission={:?} carrier={:?} work={:?} graph_capacity={:?} graph_provider={:?} prefill_scopes={:?} prefill_spans={:?} prefill_total={:?} prefill_operations={:?}",
                    controls.facts().admission_bytes(), controls.facts().carrier_bytes(), controls.facts().work_bytes(),
                    controls.binding.graph_metadata.map(|graph| graph.capacity().get()),
                    controls.binding.graph_metadata.map(|graph| graph.provider_bytes()),
                    controls.binding.prefill_scopes.map(|prefill| prefill.plan().scope_count()),
                    controls.binding.prefill_scopes.map(|prefill| prefill.plan().span_count()),
                    controls.binding.prefill_scopes.map(|prefill| prefill.total_bytes()),
                    controls.binding.prefill_scopes.map(|prefill| prefill.operation_control_bytes()));
            }
        }
        p.checked_add(input)
            .and_then(|n| n.checked_add(q))
            .and_then(|n| n.checked_add(r))
            .and_then(|n| n.checked_add(s))
            .and_then(|n| n.checked_add(publication))
            .and_then(|n| n.checked_add(batches))
            .ok_or(WorkingMemoryError::Overflow)
    }
}
impl IncrementalInferenceQuote {
    /// Seal P, named Q, optional input I/output R and pin/publication controls once BEFORE reservation. Existing enclosing
    /// estimates must not already include these terms or derived new C. Exact C stays outside the host hold. No diagnostic can upgrade a seal.
    pub fn with_span_workspace_and_text_controls(
        mut self,
        controls: PreparedTextControlWorkspace,
    ) -> Result<Self, ResidualQuoteError> {
        if self.span_seal.is_some()
            || self.span_workspace.text_controls.is_some()
            || controls.geometry() != self.geometry
            || !controls.plan.same_plan(self.span_workspace.plan())
        {
            return Err(WorkingMemoryError::IdentityMismatch.into());
        }
        #[cfg(debug_assertions)]
        if std::env::var_os("EREDU_TRACE_HOST_PARALLEL_STORAGE").is_some() {
            eprintln!("HOST_PARALLEL_STORAGE_SEAL controls={:?} prefill={:?} native_generations={:?} equation_incremental={:?} host_records_complete={}",
                controls.facts(), controls.binding.prefill_scopes,
                controls.binding.native_storage.as_ref().and_then(|native| native.equation_generations),
                self.equation_incremental_bytes,
                self.span_workspace.plan.records().iter().all(|record| record.host_workspace_bytes().is_some()));
        }
        controls
            .facts()
            .total_bytes()?
            .ok_or(WorkingMemoryError::UnknownBound)?;
        if let Some(existing) = controls
            .binding
            .publication
            .as_ref()
            .and_then(|p| p.existing.clone())
        {
            self = self.with_source_validation(existing)?;
        }
        if let Some(native) = controls.binding.native_storage.as_ref() {
            if let Some(generations) = native.equation_generations {
                let old = self
                    .equation_incremental_bytes
                    .ok_or(WorkingMemoryError::UnknownBound)?;
                let host = self
                    .span_workspace
                    .plan
                    .records()
                    .iter()
                    .try_fold(0u64, |peak, record| {
                        record.host_workspace_bytes().map(|bytes| peak.max(bytes))
                    })
                    .ok_or(WorkingMemoryError::UnknownBound)?;
                let replacement = generations
                    .checked_add(host)
                    .ok_or(WorkingMemoryError::Overflow)?;
                // Keep an independently larger old bound. This is a replacement
                // of the same named owners, not an extra P charge or a discount
                // inferred from current free capacity.
                let extra = replacement.saturating_sub(old);
                self.incremental_bytes = self
                    .incremental_bytes
                    .checked_add(extra)
                    .ok_or(WorkingMemoryError::Overflow)?;
                let workspace = self
                    .state
                    .execution_workspace
                    .as_mut()
                    .ok_or(WorkingMemoryError::UnknownBound)?;
                let WorkspaceBound::Bounded { bytes, assumptions } = &mut workspace.activations
                else {
                    return Err(WorkingMemoryError::UnknownBound.into());
                };
                *bytes = bytes
                    .checked_add(extra)
                    .ok_or(WorkingMemoryError::Overflow)?;
                assumptions.push_str("; registered equation peak replaced once by all selected native generations plus the original disjoint host peak");
            }
        }
        self.span_workspace.text_controls = Some(controls);
        self.seal_span_workspace()
    }
    /// Consume the accepted original text quote. One plan attachment protects
    /// P+Q+optional R+finite storage controls, while the compact guard retains custody without the record payload. Publication-bound plans require their consuming pending transition.
    pub fn into_funded_text_span_workspace(
        self,
        run: &WorkingMemoryFundingRun,
        reservation: &WorkingMemoryReservation,
    ) -> Result<
        (
            OwnedTextSpanWorkspace,
            Option<RegisteredInferenceSourceWitness>,
        ),
        SpanWorkspaceOwnerError,
    > {
        if self.span_workspace.text_controls.is_none()
            || self
                .span_workspace
                .control_binding()
                .is_some_and(|b| b.publication_layout().is_some())
        {
            return Err(SpanWorkspaceOwnerError {
                quote: self,
                cause: WorkingMemoryError::IdentityMismatch,
            });
        }
        let (span, witness) = self.promote_span_workspace(run, reservation)?;
        Ok((OwnedTextSpanWorkspace::from_promoted(span), witness))
    }
}
impl OwnedTextSpanWorkspace {
    fn from_promoted(span: OwnedInferenceSpanWorkspace) -> Self {
        let controls = OriginalTextControlGuard {
            custody: span
                .workspace
                .plan()
                .original_host()
                .expect("successful original attachment")
                .clone(),
        };
        Self {
            span,
            controls,
            sequence_taken: false,
            token_input_taken: false,
            preparation_scopes_taken: false,
            prediction_scopes_taken: false,
            prefill_scopes_taken: false,
            tracking_taken: false,
            graph_taken: false,
            host_destinations_taken: false,
            output_sources_taken: false,
            native_storage_taken: false,
            pins_taken: false,
            publications_taken: false,
        }
    }
}
/// Host custody only: no plan records, source/native payload, scope, constructor
/// or certification. Closed native owners must enforce their finite population.
#[derive(Clone, Debug)]
pub struct OriginalTextControlGuard {
    pub(in crate::working_memory) custody: SpanHostOwner,
}
/// Retains only the accepted raw host-control account, with no source bank,
/// native partition, record payload or authority to construct/reissue storage.
#[derive(Clone, Debug)]
pub struct OriginalTextMetadataCustody {
    _raw: crate::working_memory::funding::RawSpanHostOwner,
}
impl OriginalTextMetadataCustody {
    pub(in crate::working_memory) fn same_account(&self, other: &Self) -> bool { self._raw.same(&other._raw) }
    /// Whether this is the only remaining alias of this exact request's raw
    /// host custody. This test observation neither grants exclusive access nor
    /// proves native completion; callers must first settle the real work.
    /// No Arc or Weak escapes, so final control storage still retires before
    /// the accounted payload. Unrelated pool owners do not affect the result.
    #[cfg(any(test, feature = "original-custody-test-support"))]
    pub fn is_sole_owner(&self) -> bool {
        self._raw.is_sole_owner()
    }
    /// Compare only the existing shared accounting domain.
    pub fn matches_domain(&self, domain: &eredu_core::SharedStorageDomain) -> bool {
        self._raw
            .pool()
            .shared_storage_domain()
            .same_identity(domain)
    }
    pub(in crate::working_memory) fn validate_initialization(
        &self, source: &crate::working_memory::SharedNativeInitializationCustody,
    ) -> Result<(), WorkingMemoryError> {
        source.validate_pool(self._raw.pool())
    }
    /// Validate a closed metadata attachment without creating one.
    pub fn validate_metadata(
        &self,
        metadata: &crate::SharedHostMetadata,
    ) -> Result<(), WorkingMemoryError> {
        metadata.validate_original_attachment(self._raw.pool().shared_storage_domain())
    }
    /// Validate a fixed metadata token without growing attachment storage.
    pub fn validate_slot_metadata(
        &self,
        metadata: &crate::HostSlotMetadata,
    ) -> Result<(), WorkingMemoryError> {
        metadata.validate_original_attachment(self._raw.pool().shared_storage_domain())
    }
    /// Validate already registered source payloads in the same raw account's
    /// pool without converting their original or ordinary accounting origin.
    pub fn validate_retained_source_inventory<K>(
        &self, key: &K, bytes: u64,
    ) -> Result<(), WorkingMemoryError>
    where K: Ord + Send + Sync + 'static + crate::working_memory::GgufSourceStorageKey {
        self._raw.pool().validate_retained_source_inventory(key, bytes)
    }

    /// Read-only validation against the original raw account's pool. The
    /// returned result carries no identity, amount, credit or funding authority.
    pub fn validate_source_inventory(
        &self,
        identity: &eredu_checkpoint::store::SourceStorageIdentity,
        bytes: u64,
    ) -> Result<(), WorkingMemoryError> {
        self._raw
            .pool()
            .validate_original_source_inventory(identity, bytes)
    }
}
impl OriginalTextControlGuard {
    /// Lifetime projection for a closed, already priced metadata owner.
    pub fn metadata_custody(&self) -> OriginalTextMetadataCustody {
        OriginalTextMetadataCustody {
            _raw: self.custody.raw().clone(),
        }
    }
    /// Read-only original reservation, account and retained-source health check.
    pub fn validate_reservation(
        &self,
        reservation: &WorkingMemoryReservation,
    ) -> Result<(), WorkingMemoryError> {
        self.custody.validate_control_reservation(reservation)
    }
    /// Read-only check of the same original unexposed scope. This does not
    /// expose its native funding interface, activate it, or certify work.
    pub fn validate_prepared_scope(
        &self,
        prepared: &crate::working_memory::PreparedWorkingMemoryFundingScope,
    ) -> Result<(), WorkingMemoryError> {
        self.custody.validate_control_native(prepared.native())
    }
    /// Read-only check of the existing original scope; creates no permission.
    pub fn validate_native_scope(
        &self,
        native: &WorkingMemoryFundingScope,
    ) -> Result<(), WorkingMemoryError> {
        self.custody.validate_control_native(native)
    }
}
/// Compact original span association plus matching control custody. Native
/// owners can keep the guard alone after record payloads retire.
#[derive(Debug)]
#[must_use]
pub struct OwnedTextSpanWorkspace {
    span: OwnedInferenceSpanWorkspace,
    controls: OriginalTextControlGuard,
    sequence_taken: bool,
    token_input_taken: bool,
    preparation_scopes_taken: bool,
    prediction_scopes_taken: bool,
    prefill_scopes_taken: bool,
    tracking_taken: bool,
    graph_taken: bool,
    host_destinations_taken: bool,
    output_sources_taken: bool,
    native_storage_taken: bool,
    pins_taken: bool,
    publications_taken: bool,
}
impl OwnedTextSpanWorkspace {
    /// Extract the one bank sealed before acceptance. Equal diagnostics and
    /// cloned guards cannot construct a replacement or change its ceilings.
    pub fn take_prefill_storage_publications<K: CapturePlanStorageKey>(
        &mut self,
    ) -> Result<OriginalPrefillStoragePublicationSlots<K>, WorkingMemoryError> {
        if self.publications_taken {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let layout = self
            .workspace()
            .control_binding()
            .and_then(|b| b.batch_layout())
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        if !layout.matches::<K>() {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        self.controls.validate_reservation(self.reservation())?;
        let bank =
            OriginalPrefillStoragePublicationSlots::new(layout.clone(), self.controls.clone());
        self.publications_taken = true;
        Ok(bank)
    }
    /// Extract exactly one typed bank from the consuming accepted owner. No
    /// clone of a diagnostic or custody guard can construct a replacement.
    pub fn take_prefill_storage_pins<K: Ord + Send + 'static>(
        &mut self,
    ) -> Result<OriginalPrefillStoragePinSlots<K>, WorkingMemoryError> {
        if self.pins_taken {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let layout = self
            .workspace()
            .control_binding()
            .and_then(|b| b.pin_layout())
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        if !layout.matches::<K>() {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        self.controls.validate_reservation(self.reservation())?;
        let bank = OriginalPrefillStoragePinSlots::new(layout.clone(), self.controls.clone());
        self.pins_taken = true;
        Ok(bank)
    }
    /// Original full span workspace with separately reported P, Q, optional R and S.
    pub fn workspace(&self) -> &InferenceSpanWorkspace {
        self.span.workspace()
    }
    /// Exact original reservation metadata.
    pub fn reservation(&self) -> &WorkingMemoryReservation {
        self.span.reservation()
    }
    /// Aggregate original P+Q+optional R+finite storage controls; C payload is separate.
    /// Raw source-side custody may outlive the final plan/control alias.
    pub fn protected_host_bytes(&self) -> u64 {
        self.workspace()
            .protected_peak_bytes()
            .expect("promoted complete original controls")
    }
    /// Shared host custody without a record/source/native payload owner.
    pub fn control_guard(&self) -> OriginalTextControlGuard {
        self.controls.clone()
    }
    /// Borrow the matching accepted control and span association.
    pub fn as_reserved_text_span_workspace(&self) -> ReservedTextSpanWorkspace<'_> {
        ReservedTextSpanWorkspace {
            span: self.span.as_reserved_span_workspace(),
            controls: &self.controls,
        }
    }
}
/// Accepted complete host-control view, not interchangeable with a P-only or
/// pending-publication receipt. The closed
/// native join must also establish complete current opening publication.
#[derive(Debug)]
pub struct ReservedTextSpanWorkspace<'a> {
    pub(in crate::working_memory) span: ReservedInferenceSpanWorkspace<'a>,
    pub(in crate::working_memory) controls: &'a OriginalTextControlGuard,
}
impl ReservedTextSpanWorkspace<'_> {
    /// Original full span workspace with separately reported P, Q, optional R and S.
    pub fn workspace(&self) -> &InferenceSpanWorkspace {
        self.span.workspace()
    }
    /// Exact original reservation metadata.
    pub fn reservation(&self) -> &WorkingMemoryReservation {
        self.span.reservation()
    }
    /// Full original numerical requirement for the exact scheduled span.
    pub fn span_bytes(&self, span: &InferenceWorkspaceSpan) -> Option<u64> {
        self.span.span_bytes(span)
    }
    pub(in crate::working_memory) fn validate_locked(
        &self,
        pool: &WorkingMemoryPool,
        usage: &Usage,
    ) -> Result<(), WorkingMemoryError> {
        let binding = self
            .workspace()
            .control_binding()
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        if !self
            .workspace()
            .plan()
            .original_host()?
            .same(&self.controls.custody)
        {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        self.controls
            .custody
            .validate_control_locked(pool, usage, self.reservation(), binding)?;
        self.span.validate_sources(pool, usage)
    }
}

#[cfg(test)]
pub(in crate::working_memory) use token_input::tests::fault as token_input_fault;
