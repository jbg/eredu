//! Capture sources and cumulative host storage in the original text admission.
#[cfg(test)]
pub(super) mod funding_probe;
use super::*;
mod saved;
mod interventions;
mod partition;
pub(in crate::composition::mlx::session::model_session) use partition::OriginalPartitionCaptureFrame;
use crate::backend::runtime::residency::storage::StorageIdentity;
use eredu_core::capture::SharedCapturePlan;
use eredu_runtime::{
    HostMetadataIdentity, SharedLayeredObservationPaths,
    layered::{BoundCaptureSelection, PreparedCaptureSelection},
    working_memory::{
        CaptureRunHostPlan, OriginalTextControlGuard, OwnedTextSpanWorkspace,
        PreparedCapturePlanPublication, PreparedCaptureRun, PreparedTextControlWorkspace,
        RegisteredInferenceSourceWitness, TextHostControlFacts, WorkingMemoryFundingRun,
        WorkingMemoryFundingScope, WorkingMemoryStorage,
    },
};

use crate::composition::mlx::replicated_text::{NativeOpeningRowsOwner, SealedOpeningRows};

/// Borrow only the actual immutable sources while evaluating cold candidates.
pub(in crate::composition::mlx::session::model_session) struct CaptureAdmission<'a> {
    session: &'a MlxModelSession,
    checkpoint: Option<&'a eredu_runtime::capture::FundedCaptureCheckpoint>,
    saved_witness: Option<&'a RegisteredInferenceSourceWitness>,
    opening_rows: Option<RefCell<Option<SealedOpeningRows>>>,
    source: &'a SharedCapturePlan,
    paths: &'a SharedLayeredObservationPaths,
    selection: PreparedCaptureSelection,
    host: CaptureRunHostPlan<'a>,
    text_interventions: RefCell<Option<crate::composition::mlx::session::intervention::PreparedTextInterventions>>,
    partition_hosts: RefCell<Option<partition::host::HostAdmission>>,
    partition_evidence: RefCell<Option<partition::host::evidence::Admission>>,
    paths_pin: Option<WorkingMemoryStorage<StorageIdentity>>,
    plan_pin: Option<WorkingMemoryStorage<StorageIdentity>>,
    new_source_bytes: u64,
}

/// Promotion may attach before later construction fails. Keep the consumed
/// admission and its pins before custody even while individual fields move.
struct PromotedCaptureAdmission<'a> {
    admission: CaptureAdmission<'a>,
    witness: Option<RegisteredInferenceSourceWitness>,
    span: OwnedTextSpanWorkspace,
}

impl<'a> CaptureAdmission<'a> {
    pub(super) fn new(
        session: &'a MlxModelSession,
        geometry: InferenceGeometry,
        source: &'a SharedCapturePlan,
    ) -> Result<Self, Error> {
        Self::new_input(session, geometry, source, false)
    }

    pub(super) fn new_media(
        session: &'a MlxModelSession,
        geometry: InferenceGeometry,
        source: &'a SharedCapturePlan,
    ) -> Result<Self, Error> {
        Self::new_input(session, geometry, source, true)
    }

    fn new_input(
        session: &'a MlxModelSession,
        geometry: InferenceGeometry,
        source: &'a SharedCapturePlan,
        media: bool,
    ) -> Result<Self, Error> {
        validate_semantics(session, source, geometry)?;
        let paths = session
            .payload
            .model
            .erased()
            .shared_observation_paths()
            .ok_or_else(unknown)?;
        session
            .payload
            .model
            .erased()
            .validate_prepared_observation_paths(paths)?;
        // `validate_semantics` checked the exact current collector report,
        // including retained partition placement. Bind that same immutable plan
        // to the ordinary causal path source without re-reading a base report.
        let selection = if media {
            paths.prepare_media_capture_selection(source)
        } else {
            paths.prepare_capture_selection(source)
        }.map_err(|error| Error::Other(Box::new(error)))?;
        let host =
            CaptureRunHostPlan::prepare(source).map_err(|error| Error::Other(Box::new(error)))?;
        Self::new_selected(session, source, paths, selection, host, None)
    }

    fn new_selected(
        session: &'a MlxModelSession,
        source: &'a SharedCapturePlan,
        paths: &'a SharedLayeredObservationPaths,
        selection: PreparedCaptureSelection,
        host: CaptureRunHostPlan<'a>,
        checkpoint: Option<&'a eredu_runtime::capture::FundedCaptureCheckpoint>,
    ) -> Result<Self, Error> {
        let pool = &session.payload.memory_pool;
        let paths_pin = pool
            .pin_registered_storage([(
                StorageIdentity::HostMetadata(paths.identity().registry_key().clone()),
                paths.capacity_bytes().ok_or_else(unknown)?,
            )])
            .map_err(memory)?;
        let capacity = source.capacity_bytes().ok_or_else(unknown)?;
        let plan_pin = match pool.pin_registered_storage([(
            StorageIdentity::CapturePlan(source.storage_identity().clone()),
            capacity,
        )]) {
            Ok(pin) => Some(pin),
            Err(WorkingMemoryError::IdentityMismatch) => None,
            Err(error) => return Err(memory(error)),
        };
        let new_source_bytes = if plan_pin.is_some() { 0 } else { capacity };
        Ok(Self {
            session,
            checkpoint,
            saved_witness: None,
            opening_rows: None,
            source,
            paths,
            selection,
            host,
            text_interventions: RefCell::new(None),
            partition_hosts: RefCell::new(None),
            partition_evidence: RefCell::new(None),
            paths_pin: Some(paths_pin),
            plan_pin,
            new_source_bytes,
        })
    }

    pub(super) fn with_opening_rows(mut self) -> Self {
        self.opening_rows = Some(RefCell::new(None));
        self
    }

    pub(super) fn prepared_selection(&self) -> &PreparedCaptureSelection {
        &self.selection
    }

    /// Completed input quotation must use the same media selection that emitted
    /// its equation/native trace, including both physical immutable owners.
    pub(in crate::composition::mlx::session::model_session) fn validate_media_trace(
        &self,
        traced: &PreparedCaptureSelection,
    ) -> Result<(), Error> {
        if !traced.is_prepared_media() || !self.selection.is_prepared_media() {
            return Err(memory(WorkingMemoryError::IdentityMismatch));
        }
        traced.validate_sources(self.source, self.paths)
            .map_err(|_| memory(WorkingMemoryError::IdentityMismatch))?;
        Ok(())
    }

    pub(super) fn physical_output(&self, minimum: OutputDemand) -> OutputDemand {
        let output = self.selection.physical_output(minimum);
        if self.host.intervention_source().is_some_and(|source|
            source.plan().admission().requires_sequence_scores(eredu_core::capture::CapturePhase::Prefill, 0)) {
            OutputDemand::Sequence
        } else {
            output
        }
    }

    pub(super) fn bind_geometry(
        &self,
        geometry: InferenceGeometry,
    ) -> Result<BoundCaptureSelection<'_>, Error> {
        let bound = if self.checkpoint.is_some() {
            self.selection.bind_prompt_prefix(geometry)
        } else {
            self.selection.bind_geometry(geometry)
        };
        bound.map_err(|error| Error::Other(Box::new(error)))
    }

    fn control_peak_bytes() -> Result<u64, Error> {
        // Concrete pending/installed fields, including padding and construction
        // moves. Source payload and span record capacity are charged separately.
        let fixed = std::mem::size_of::<Self>()
            .checked_add(std::mem::size_of::<PendingCaptureBank>())
            .and_then(|n| n.checked_add(std::mem::size_of::<CaptureQuotation>()))
            .and_then(|n| {
                n.checked_add(std::mem::size_of::<
                    super::super::text_capture::InstalledCapture,
                >())
            })
            .and_then(|n| {
                n.checked_add(std::mem::size_of::<
                    super::super::text_capture::InstallationPayload,
                >())
            })
            .and_then(|n| n.checked_add(std::mem::size_of::<PromotedCaptureAdmission<'_>>()))
            .and_then(|n| n.checked_mul(3))
            .and_then(|n| n.checked_add(std::mem::size_of::<(
                [&MlxModelSession; 2], [InferenceGeometry; 2],
                [&SharedCapturePlan; 2], bool, Result<Self, Error>,
            )>()))
            .and_then(|n| u64::try_from(n).ok())
            .ok_or_else(|| memory(WorkingMemoryError::Overflow))?;
        fixed
            .checked_add(super::owner_control_bytes()?)
            .ok_or_else(|| memory(WorkingMemoryError::Overflow))?
            .checked_add(
                PreparedCaptureSelection::control_peak_bytes()
                    .ok_or_else(|| memory(WorkingMemoryError::Overflow))?,
            )
            .and_then(|bytes| {
                bytes.checked_add(super::super::text_capture::prefill_borrow_control_bytes()?)
            })
            .and_then(|bytes| bytes.checked_add(super::super::text_error::control_peak_bytes()?))
            .ok_or_else(|| memory(WorkingMemoryError::Overflow))
    }

    pub(super) fn intervention_quote(
        &self,
    ) -> Option<crate::composition::mlx::session::intervention::TextInterventionQuote<'_>> {
        self.host.intervention_source().map(|source|
            crate::composition::mlx::session::intervention::TextInterventionQuote {
                source, rows: &self.text_interventions,
            })
    }

    pub(in crate::composition::mlx::session::model_session) fn validate_intervention_source(
        &self, actual: Option<&eredu_runtime::working_memory::OriginalInterventionSource>,
    ) -> Result<(), Error> {
        match (actual, self.host.intervention_source()) {
            (None, None) => Ok(()),
            (Some(actual), Some(expected)) if actual.same_source(expected) => Ok(()),
            _ => Err(memory(WorkingMemoryError::IdentityMismatch)),
        }
    }

    /// The model diagnostic must have used this exact cumulative host program.
    pub(super) fn validate_host(&self, host: &CaptureRunHostPlan<'_>) -> Result<(), Error> {
        if !host.source().same_storage(self.source)
            || host.initialization_peak_bytes() != self.host.initialization_peak_bytes()
            || match (host.intervention_source(), self.host.intervention_source()) {
                (None, None) => false,
                (Some(actual), Some(expected)) => !actual.same_source(expected),
                _ => true,
            }
        {
            return Err(memory(WorkingMemoryError::IdentityMismatch));
        }
        Ok(())
    }

    pub(super) fn add_enclosing(
        &self,
        workspace: &mut ExecutionWorkspaceEstimate,
    ) -> Result<(), Error> {
        self.add_enclosing_metadata(
            workspace,
            eredu_runtime::working_memory::WorkspaceReportMetadata::ordinary(),
        )
    }

    pub(super) fn add_enclosing_metadata(
        &self,
        workspace: &mut ExecutionWorkspaceEstimate,
        metadata: eredu_runtime::working_memory::WorkspaceReportMetadata<'_>,
    ) -> Result<(), Error> {
        let partition_hosts = match metadata.funding() {
            Some(funding) => partition::host::HostAdmission::prepare(self.session,self.source,
                workspace.geometry,self.checkpoint.map_or(0,|checkpoint|checkpoint.next_prediction()),&funding)?,
            None => None,
        };
        let partition_evidence=match metadata.funding() {
            Some(funding)=>partition::host::evidence::Admission::prepare(self.session,self.source,
                self.host.intervention_source(),&self.selection,workspace.geometry,
                self.checkpoint.map_or(0,|checkpoint|checkpoint.next_prediction()),&funding)?,
            None=>None,
        };
        let extra = self.host.initialization_peak_bytes()
            .checked_add(partition_hosts.as_ref().map_or(0,|hosts|hosts.bytes()))
            .and_then(|bytes|bytes.checked_add(partition_evidence.as_ref().map_or(0,|hosts|hosts.bytes())))
            .ok_or_else(||memory(WorkingMemoryError::Overflow))?;
        self.partition_hosts.replace(partition_hosts);
        self.partition_evidence.replace(partition_evidence);
        if let WorkspaceBound::Bounded { bytes, assumptions } = &mut workspace.retained {
            *bytes = bytes
                .checked_add(extra)
                .ok_or_else(|| memory(WorkingMemoryError::Overflow))?;
            metadata.append(assumptions, "; cumulative capture host bank once; the accepted span seal separately adds exact new C and original P+Q+publication-slot custody for measured controls; native capture transforms remain in the equation spans")
                .map_err(|cause| Error::Neural(metadata.error(cause)))?;
        }
        Ok(())
    }

    /// Named native facts are measured from the actual selected source and
    /// concrete owner representations. They are diagnostics until consumed by
    /// this candidate's original quote seal; no later byte grant exists.
    fn control_facts(&self, geometry: InferenceGeometry) -> Result<TextHostControlFacts, Error> {
        self.validate_geometry(geometry)?;
        Ok(TextHostControlFacts::new(
            Some(Self::control_peak_bytes()?),
            Some(super::super::text_funding::capture_carrier_control_bytes(
                self.source, self.host.intervention_source(),
            )?),
            Some(super::super::text_funding::text_work_control_bytes(
                geometry.max_output_tokens,
            )?),
        ))
    }

    fn control_facts_with_sequence(
        &self,
        geometry: InferenceGeometry,
        sequence: bool,
    ) -> Result<TextHostControlFacts, Error> {
        let facts = self.control_facts(geometry)?;
        if !sequence {
            return Ok(facts);
        }
        let mut extra = super::sequence::capture_control_bytes()?
            .checked_add(
                u64::try_from(std::mem::size_of::<
                    std::cell::RefMut<'_, Option<PendingCaptureBank>>,
                >())
                .map_err(|_| memory(WorkingMemoryError::Overflow))?,
            )
            .ok_or_else(|| memory(WorkingMemoryError::Overflow))?;
        if self.opening_rows.is_some() {
            extra = extra
                .checked_add(
                    SealedOpeningRows::sequence_error_control_bytes()
                        .ok_or_else(|| memory(WorkingMemoryError::Overflow))?,
                )
                .ok_or_else(|| memory(WorkingMemoryError::Overflow))?;
        }
        let admission = facts
            .admission_bytes()
            .map(|n| {
                n.checked_add(extra)
                    .ok_or_else(|| memory(WorkingMemoryError::Overflow))
            })
            .transpose()?;
        Ok(TextHostControlFacts::new(
            admission,
            facts.carrier_bytes(),
            facts.work_bytes(),
        ))
    }

    pub(super) fn seal_controls(
        &self,
        quote: IncrementalInferenceQuote,
    ) -> Result<IncrementalInferenceQuote, Error> {
        self.seal_controls_with_sequence(quote, None, None, None, None, None)
    }

    pub(super) fn seal_controls_with_sequence(
        &self,
        quote: IncrementalInferenceQuote,
        claim: Option<&eredu_core::GenerationSequencePreparation<'_, '_>>,
        tracking: Option<eredu_runtime::working_memory::SubmissionTrackingFacts>,
        graph: Option<eredu_runtime::working_memory::GraphMetadataFacts>,
        prefill: Option<eredu_runtime::working_memory::TextPrefillScopeFacts>,
        native: Option<
            eredu_runtime::working_memory::PreparedNativeStoragePlan<
                crate::backend::runtime::residency::storage::native_storage::MlxNativeStorage,
            >,
        >,
    ) -> Result<IncrementalInferenceQuote, Error> {
        let publication = PreparedCapturePlanPublication::prepare(
            quote.pool(),
            quote.span_workspace().plan(),
            self.source,
            StorageIdentity::CapturePlan(self.source.storage_identity().clone()),
            self.plan_pin.as_ref(),
        )
        .map_err(memory)?;
        if publication.new_source_bytes() != self.new_source_bytes {
            return Err(memory(WorkingMemoryError::IdentityMismatch));
        }
        let active = eredu_runtime::capture::CaptureObservationStep::new(
            self.source.admission(),
            eredu_core::capture::CapturePhase::Prefill,
            0,
        )
        .map_err(|e| Error::Other(Box::new(e)))?
        .has_selected_prefill_hook();
        if let Some(slot) = self.opening_rows.as_ref().filter(|_| active) {
            let proposal = self
                .session
                .payload
                .model
                .erased()
                .prepare_opening_rows(self.bind_geometry(quote.geometry())?)?;
            let facts = self.control_facts_with_sequence(quote.geometry(), claim.is_some())?;
            let (sealed, quote) = proposal.seal_with_sequence(
                self.join_sources(quote)?,
                facts,
                publication,
                claim,
                tracking,
                graph,
                prefill,
                native,
            )?;
            // Candidate replacement is cold, before any original reservation.
            let old = slot.replace(Some(sealed));
            drop(old);
            return Ok(quote);
        }
        let prepared = PreparedTextControlWorkspace::prepare(
            self.source,
            quote.geometry(),
            quote.span_workspace().plan(),
            self.control_facts_with_sequence(quote.geometry(), claim.is_some())?,
        )
        .map_err(memory)?
        .with_prefill_capture_selection(self.bind_geometry(quote.geometry())?)
        .map_err(memory)?
        .with_capture_plan_publication(publication)
        .map_err(memory)?;
        let prepared = prepared
            .with_preparation_scopes(super::preparation::facts()?)
            .map_err(memory)?;
        let prepared = prepared
            .with_prediction_scopes(super::prediction::facts()?)
            .map_err(memory)?;
        let prepared = match claim {
            Some(claim) => prepared.with_generation_sequence(claim).map_err(memory)?,
            None => prepared,
        };
        let prepared = match tracking {
            Some(facts) => prepared.with_submission_tracking(facts).map_err(memory)?,
            None => prepared,
        };
        let prepared = match graph {
            Some(facts) => prepared.with_graph_metadata(facts).map_err(memory)?,
            None => prepared,
        };
        let prepared = match prefill {
            Some(facts) => prepared.with_prefill_scopes(facts).map_err(memory)?,
            None => prepared,
        };
        let prepared = match native {
            Some(plan) => prepared.with_native_storage(plan).map_err(memory)?,
            None => prepared,
        };
        self.join_sources(quote)?
            .with_span_workspace_and_text_controls(prepared)
            .map_err(|error| Error::Other(Box::new(error)))
    }

    pub(super) fn join_sources(
        &self,
        mut quote: IncrementalInferenceQuote,
    ) -> Result<IncrementalInferenceQuote, Error> {
        quote = quote
            .with_registered_sources(self.paths_pin.as_ref().ok_or_else(unknown)?.clone())
            .map_err(memory)?;
        if let Some(plan) = &self.plan_pin {
            quote = quote
                .with_registered_sources(plan.clone())
                .map_err(memory)?;
        }
        Ok(quote)
    }

    /// Uses the original controller/source publication scope and original run.
    /// Returned bank is protected before Prompt; publication is host-only.
    pub(super) fn prepare(
        self,
        session: &MlxModelSession,
        request: &InferenceRequest,
        funding: &WorkingMemoryFundingRun,
        scope: &WorkingMemoryFundingScope,
        accepted: IncrementalInferenceQuote,
        sequence: bool,
    ) -> Result<CaptureQuotation, AdmissionFailure> {
        let geometry = accepted.geometry();
        self.bind_geometry(geometry)?;
        self.selection
            .validate_sources(self.source, self.paths)
            .map_err(|error| Error::Other(Box::new(error)))?;
        validate_semantics(session, self.source, geometry)?;
        session
            .payload
            .model
            .erased()
            .validate_prepared_observation_paths(self.paths)?;
        if !accepted.pool().same_domain(&session.payload.memory_pool)
            || geometry != request.geometry()
        {
            return Err(memory(WorkingMemoryError::IdentityMismatch).into());
        }
        // The exact original seal protects pending construction before any C
        // publication. New C uses reserved coverage without reducing P+Q+slots;
        // same-pool races retain the originally accepted amount.
        let pending = accepted
            .begin_capture_plan_publication::<StorageIdentity>(
                funding,
                request.memory_reservation().ok_or_else(unknown)?,
                self.source,
            )
            .map_err(|error| AdmissionFailure::retained(error, sequence))?;
        let (span, witness) = pending
            .publish_and_finish(scope)
            .map_err(|error| AdmissionFailure::retained(error, sequence))?;
        self.finish_prepared(geometry, request, funding, span, witness, sequence)
    }

    fn finish_prepared(
        self,
        geometry: InferenceGeometry,
        request: &InferenceRequest,
        funding: &WorkingMemoryFundingRun,
        span: OwnedTextSpanWorkspace,
        witness: RegisteredInferenceSourceWitness,
        sequence: bool,
    ) -> Result<CaptureQuotation, AdmissionFailure> {
        let mut promoted = PromotedCaptureAdmission {
            admission: self,
            witness: Some(witness),
            span,
        };
        let witness = promoted
            .witness
            .take()
            .ok_or_else(|| memory(WorkingMemoryError::IdentityMismatch))?;
        // Publication and the final S+C health join now precede the ordinary
        // guard. No dynamic source map or late full-witness rebuild occurs.
        let sealed_rows = promoted
            .admission
            .opening_rows
            .as_ref()
            .and_then(|slot| slot.borrow_mut().take());
        // The slot loan is gone before allocation, direct error construction,
        // or destruction of an owner-preserving rejected row plan.
        let rows = sealed_rows
            .map(|sealed| {
                if sequence {
                    sealed
                        .allocate_with_sequence(&mut promoted.span)
                        .map_err(AdmissionFailure::Sequence)
                } else {
                    sealed
                        .allocate(&mut promoted.span)
                        .map_err(AdmissionFailure::Native)
                }
            })
            .transpose()?;
        let controls = promoted.span.control_guard();
        let partition_hosts=promoted.admission.partition_hosts.borrow_mut().take()
            .map(|hosts|hosts.protect(funding,request.memory_reservation().ok_or_else(unknown)?))
            .transpose()?;
        let partition_evidence=promoted.admission.partition_evidence.borrow_mut().take()
            .map(|hosts|hosts.protect(funding,request.memory_reservation().ok_or_else(unknown)?))
            .transpose()?;
        let bank = funding
            .prepare_capture_run(
                request.memory_reservation().ok_or_else(unknown)?,
                promoted.admission.host,
            )
            .map_err(|error| Error::Other(Box::new(error)))?;
        #[cfg(test)]
        funding_probe::record(promoted.admission.source, &promoted.span, &bank);
        Ok(CaptureQuotation {
            source: promoted.admission.source.storage_identity().clone(),
            capture_source: promoted.admission.source.clone(),
            paths: promoted.admission.paths.identity().clone(),
            geometry,
            first_prediction: promoted
                .admission
                .checkpoint
                .map_or(0, |checkpoint| checkpoint.next_prediction()),
            continuation: promoted.admission.checkpoint.is_some(),
            pending: RefCell::new(Some(PendingCaptureBank {
                bank,
                selection: promoted.admission.selection,
                witness,
                span: promoted.span,
            })),
            rows,
            partition_hosts: RefCell::new(partition_hosts),
            partition_evidence: RefCell::new(partition_evidence),
            text_interventions: promoted.admission.text_interventions.into_inner()
                .map(crate::composition::mlx::session::intervention::PreparedTextInterventions::into_owner)
                .transpose().map_err(memory)?,
            controls,
        })
    }
}

#[derive(Debug)]
struct PendingCaptureBank {
    // Payload/source aliases retire before the final original span-plan owner.
    bank: PreparedCaptureRun,
    selection: PreparedCaptureSelection,
    witness: RegisteredInferenceSourceWitness,
    span: OwnedTextSpanWorkspace,
}

/// Historical quote keeps identities after the installer moves its one bank.
#[derive(Debug)]
pub(super) struct CaptureQuotation {
    source: eredu_core::SharedStorageIdentity,
    capture_source: SharedCapturePlan,
    paths: HostMetadataIdentity,
    geometry: InferenceGeometry,
    first_prediction: u64,
    continuation: bool,
    pending: RefCell<Option<PendingCaptureBank>>,
    partition_hosts: RefCell<Option<partition::host::HostRows>>,
    partition_evidence: RefCell<Option<partition::host::evidence::Hosts>>,
    // Last, including after the installer removes pending: the historical
    // fixed quote object is still an actual Q-owned control payload.
    rows: Option<NativeOpeningRowsOwner>,
    text_interventions: Option<crate::composition::mlx::session::intervention::PreparedTextInterventionsOwner>,
    controls: OriginalTextControlGuard,
}

impl CaptureQuotation {
    pub(super) fn text_interventions(&self)
        -> Option<crate::composition::mlx::session::intervention::PreparedTextInterventionsOwner> {
        self.text_interventions.clone()
    }

    pub(super) fn validate_checkpoint_source(
        &self,
        source: &SharedCapturePlan,
        geometry: InferenceGeometry,
    ) -> Result<(), WorkingMemoryError> {
        if source.storage_identity() != &self.source || geometry != self.geometry {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        Ok(())
    }

    pub(super) fn take_output_source_constructions(
        &self,
    ) -> Result<Option<eredu_runtime::working_memory::OriginalHostSourceBank>, Error> {
        let result = {
            let mut pending = self
                .pending
                .try_borrow_mut()
                .map_err(|_| memory(WorkingMemoryError::AlreadyStarted))?;
            pending
                .as_mut()
                .ok_or(WorkingMemoryError::IdentityMismatch)
                .and_then(|pending| pending.span.take_output_source_constructions())
        };
        result.map_err(memory)
    }

    pub(super) fn take_host_destinations(
        &self,
    ) -> Result<Option<eredu_runtime::working_memory::OriginalHostDestinationBank>, Error> {
        let result = {
            let mut pending = self
                .pending
                .try_borrow_mut()
                .map_err(|_| memory(WorkingMemoryError::AlreadyStarted))?;
            pending
                .as_mut()
                .ok_or(WorkingMemoryError::IdentityMismatch)
                .and_then(|pending| pending.span.take_host_destinations())
        };
        result.map_err(memory)
    }

    pub(super) fn take_native_storage(
        &self,
        run: &eredu_runtime::working_memory::WorkingMemoryFundingRun,
        selection: &eredu_runtime::working_memory::NativeStorageSelection,
    ) -> Result<Option<crate::backend::runtime::residency::storage::native_storage::Bank>, Error>
    {
        let result = {
            let mut pending = self
                .pending
                .try_borrow_mut()
                .map_err(|_| memory(WorkingMemoryError::AlreadyStarted))?;
            pending.as_mut().ok_or(WorkingMemoryError::IdentityMismatch)
                .and_then(|pending| pending.span.take_native_storage_bank::<crate::backend::runtime::residency::storage::native_storage::MlxNativeStorage>(run, selection))
        };
        result.map_err(memory)
    }
    pub(super) fn take_submission_tracking(
        &self,
        preparation: &InferenceTextPreparation,
    ) -> Result<Option<eredu_runtime::working_memory::OriginalSubmissionTracking>, Error> {
        let result = {
            let mut pending = self.pending.borrow_mut();
            match pending.as_mut() {
                Some(pending) => pending.span.take_submission_tracking(preparation),
                None => Err(WorkingMemoryError::IdentityMismatch),
            }
        };
        result.map_err(memory)
    }

    pub(super) fn take_graph_metadata(
        &self,
        preparation: &InferenceTextPreparation,
    ) -> Result<Option<eredu_runtime::working_memory::OriginalGraphMetadata>, Error> {
        let result = {
            let mut pending = self.pending.borrow_mut();
            match pending.as_mut() {
                Some(pending) => pending.span.take_graph_metadata(preparation),
                None => Err(WorkingMemoryError::IdentityMismatch),
            }
        };
        result.map_err(memory)
    }

    pub(super) fn prediction_bank_transfer_bytes() -> Option<u64> {
        u64::try_from(std::mem::size_of::<
            std::cell::RefMut<'_, Option<PendingCaptureBank>>,
        >())
        .ok()
    }

    pub(super) fn take_prefill_scopes(
        &self,
    ) -> Result<Option<eredu_runtime::working_memory::OriginalTextPrefillScopes>, Error> {
        let result = {
            let mut pending = self.pending.borrow_mut();
            match pending.as_mut() {
                Some(pending) => pending.span.take_prefill_scopes(),
                None => Err(WorkingMemoryError::IdentityMismatch),
            }
        };
        result.map_err(memory)
    }

    pub(super) fn take_prediction_scopes(
        &self,
    ) -> Result<Option<eredu_runtime::working_memory::OriginalTextPredictionScopes>, Error> {
        let result = {
            let mut pending = self.pending.borrow_mut();
            match pending.as_mut() {
                Some(pending) => pending.span.take_prediction_scopes(),
                None => Err(WorkingMemoryError::IdentityMismatch),
            }
        };
        result.map_err(memory)
    }

    pub(super) fn take_preparation_scopes(
        &self,
    ) -> Result<Option<eredu_runtime::working_memory::OriginalTextPreparationScopes>, Error> {
        let result = {
            let mut pending = self.pending.borrow_mut();
            match pending.as_mut() {
                Some(pending) => pending.span.take_preparation_scopes(),
                None => Err(WorkingMemoryError::IdentityMismatch),
            }
        };
        result.map_err(memory)
    }

    pub(super) fn take_sequence_quotation(&self) -> Option<SequenceQuotation> {
        let mut pending = self.pending.borrow_mut();
        SequenceQuotation::from_span(&mut pending.as_mut()?.span)
    }

    #[cfg(test)]
    pub(super) fn sequence_facts_for_test(&self) -> (u64, u64, u64, u64, bool) {
        let pending = self.pending.borrow();
        let span = &pending.as_ref().expect("before installation").span;
        let controls = span.workspace().text_controls().unwrap();
        (
            span.protected_host_bytes(),
            controls.sequence_storage_bytes(),
            controls.facts().admission_bytes().unwrap(),
            controls.facts().work_bytes().unwrap(),
            controls.source_identity().is_some(),
        )
    }

    pub(super) fn opening_rows(&self) -> Option<NativeOpeningRowsOwner> {
        self.rows.clone()
    }

    pub(super) fn control_guard(&self) -> OriginalTextControlGuard {
        self.controls.clone()
    }

    pub(super) fn validate(
        &self,
        session: &MlxModelSession,
        source: &SharedCapturePlan,
        geometry: InferenceGeometry,
    ) -> Result<(), Error> {
        if source.storage_identity() != &self.source || geometry != self.geometry {
            return Err(memory(WorkingMemoryError::IdentityMismatch));
        }
        if !self.continuation {
            validate_semantics(session, source, geometry)?;
        } else {
            validate_current_admission(session, source.admission())?;
        };
        let paths = session
            .payload
            .model
            .erased()
            .shared_observation_paths()
            .ok_or_else(unknown)?;
        if paths.identity() != &self.paths {
            return Err(memory(WorkingMemoryError::IdentityMismatch));
        }
        session
            .payload
            .model
            .erased()
            .validate_prepared_observation_paths(paths)
    }

    /// Move only after the closed installer has checked the original run,
    /// account, stages and empty collector. Conversion occurs outside RefCell.
    pub(super) fn take_installed(
        &self,
        session: &MlxModelSession,
        source: &SharedCapturePlan,
        geometry: InferenceGeometry,
    ) -> Result<super::super::text_capture::InstalledCapture, Error> {
        self.take_installed_with_error_allowance(session, source, geometry)
            .map(|(installed, _unused_allowance)| installed)
    }

    pub(super) fn take_installed_with_error_allowance(
        &self,
        session: &MlxModelSession,
        source: &SharedCapturePlan,
        geometry: InferenceGeometry,
    ) -> Result<
        (
            super::super::text_capture::InstalledCapture,
            super::super::text_error::OriginalErrorAllowance,
        ),
        Error,
    > {
        let PendingCaptureBank {
            bank,
            selection,
            witness,
            span,
        } = self.take_pending(session, source, geometry)?;
        super::super::text_capture::InstalledCapture::new_with_opening_rows_and_error_allowance(
            bank,
            selection,
            witness,
            span,
            self.rows.clone(),
        )
    }

    fn take_pending(
        &self,
        session: &MlxModelSession,
        source: &SharedCapturePlan,
        geometry: InferenceGeometry,
    ) -> Result<PendingCaptureBank, Error> {
        self.validate(session, source, geometry)?;
        self.take_pending_validated(session, source, geometry, None)
    }

    fn take_pending_validated(
        &self,
        session: &MlxModelSession,
        source: &SharedCapturePlan,
        geometry: InferenceGeometry,
        checkpoint: Option<&eredu_runtime::capture::FundedCaptureCheckpoint>,
    ) -> Result<PendingCaptureBank, Error> {
        let mut pending = self.pending.borrow_mut();
        let current = pending
            .as_ref()
            .ok_or_else(|| memory(WorkingMemoryError::PreparationAlreadyStarted))?;
        current
            .witness
            .validate(&session.payload.memory_pool)
            .map_err(memory)?;
        if !current.bank.source().same_storage(source)
            || current.span.workspace().plan().geometry() != geometry
        {
            return Err(memory(WorkingMemoryError::IdentityMismatch));
        }
        current
            .selection
            .validate_sources(source, current.selection.paths())
            .map_err(|error| Error::Other(Box::new(error)))?;
        if let Some(checkpoint) = checkpoint {
            checkpoint
                .validate_continuation_geometry(geometry)
                .map_err(|cause| Error::Other(Box::new(cause)))?;
        } else {
            current
                .selection
                .bind_geometry(geometry)
                .map_err(|cause| Error::Other(Box::new(cause)))?;
        }
        Ok(pending.take().expect("validated pending bank"))
    }
}

fn validate_semantics(
    session: &MlxModelSession,
    source: &SharedCapturePlan,
    geometry: InferenceGeometry,
) -> Result<(), Error> {
    let admission = source.admission();
    let request = admission.request();
    let origin = admission.text_origin().ok_or_else(unknown)?;
    if request.batch != geometry.batch_size
        || request.prompt_tokens != geometry.input_positions
        || request.max_predictions != geometry.max_output_tokens
        || origin.cached_positions != geometry.cached_positions
    {
        return Err(memory(WorkingMemoryError::IdentityMismatch));
    }
    validate_current_admission(session, admission)
}

// The loaded private report was built by the same ordinary partition collector
// from this session's retained architecture/hook/native source. Borrow it; never
// fabricate support or rebuild a discovery DTO during managed revalidation.
fn validate_current_admission(
    session: &MlxModelSession,
    admission: &eredu_core::capture::AdmittedCapturePlan,
) -> Result<(), Error> {
    if let Some(transport) = &session.payload.distributed {
        let loaded = session.partition_capture.get().and_then(|loaded| loaded.as_ref().ok())
            .ok_or_else(unknown)?;
        let (_, execution, setup) = loaded.source_labels();
        let blueprint = session.payload.model.inference_blueprint().ok_or_else(unknown)?;
        if setup != transport.session_identity() || execution != blueprint.execution_identity() {
            return Err(memory(WorkingMemoryError::IdentityMismatch));
        }
        admission.revalidate(&loaded.discovery).map_err(|cause| Error::Other(Box::new(cause)))
    } else {
        session.capture_discovery.as_ref().ok_or_else(unknown)?
            .validate_capture_admission(admission)
            .map_err(|cause| Error::Other(Box::new(cause)))
    }
}

#[cfg(all(
    test,
    feature = "metal",
    target_vendor = "apple",
    not(feature = "cuda")
))]
mod tests;
