//! Actual saved capture source on the same fresh copy/admission program.
use super::*;
use eredu_runtime::capture::FundedCaptureCheckpoint;
use eredu_runtime::working_memory::CopyPreparationInferenceQuote;

type SavedQuote = CopyPreparationInferenceQuote<StorageIdentity>;

impl<'a> CaptureAdmission<'a> {
    pub(in crate::composition::mlx::session::model_session) fn validate_saved_intervention_source(
        session: &MlxModelSession,
        source: Option<&eredu_runtime::working_memory::OriginalInterventionSource>,
        metadata: eredu_runtime::working_memory::WorkspaceReportMetadata<'_>,
    ) -> Result<(), Error> {
        interventions::validate_saved_source(session, source, metadata)
    }

    pub(in crate::composition::mlx::session::model_session::text_quote) fn with_saved_interventions(
        self, rows: Option<crate::composition::mlx::session::intervention::PreparedTextInterventions>,
        geometry: InferenceGeometry,
    ) -> Result<Self, Error> {
        let checkpoint=self.checkpoint.ok_or_else(unknown)?;
        match (checkpoint.intervention_source(), rows.as_ref()) {
            (Some(source),Some(rows)) => rows.validate_range(source,checkpoint.next_prediction(),geometry.max_output_tokens)
                .map_err(|_|memory(WorkingMemoryError::IdentityMismatch))?,
            (None,None)=>{},
            _=>return Err(memory(WorkingMemoryError::IdentityMismatch)),
        }
        if self.text_interventions.borrow().is_some(){return Err(memory(WorkingMemoryError::AlreadyStarted));}
        *self.text_interventions.borrow_mut()=rows;
        Ok(self)
    }

    pub(in crate::composition::mlx::session::model_session) fn from_checkpoint(
        session: &'a MlxModelSession,
        geometry: InferenceGeometry,
        checkpoint: &'a FundedCaptureCheckpoint,
        selection: &'a PreparedCaptureSelection,
        witness: &'a RegisteredInferenceSourceWitness,
        metadata: eredu_runtime::working_memory::WorkspaceReportMetadata<'_>,
    ) -> Result<Self, Error> {
        metadata
            .admit::<Self>()
            .map_err(|cause| Error::Neural(metadata.error(cause)))?;
        metadata
            .admit::<Result<Self, Error>>()
            .map_err(|cause| Error::Neural(metadata.error(cause)))?;
        metadata
            .admit::<eredu_runtime::capture::FundedCaptureCheckpointError>()
            .map_err(|cause| Error::Neural(metadata.error(cause)))?;
        witness
            .validate(&session.payload.memory_pool)
            .map_err(memory)?;
        checkpoint
            .validate_continuation_geometry(geometry)
            .map_err(|cause| Error::Neural(validation::source(metadata, cause)))?;
        selection
            .validate_sources(checkpoint.source(), selection.paths())
            .map_err(|cause| Error::Neural(validation::source(metadata, cause)))?;
        session
            .payload
            .model
            .erased()
            .validate_prepared_observation_paths(selection.paths(), metadata)?;
        validate_current_admission(session, checkpoint.source().admission(), metadata)?;
        interventions::validate_saved_source(
            session, checkpoint.intervention_source(), metadata,
        )?;
        let host = checkpoint
            .continuation_host_plan_for(geometry)
            .map_err(|cause| Error::Neural(validation::source(metadata, cause)))?;
        // Both the saved cold quote and its later accepted reconstruction use
        // this constructor. Retain the same real projected Host descriptor on
        // each; a before-prefill checkpoint still needs those destinations.
        // The shared worker returns None only for a non-applicable source or a
        // continuation after the initial prefill observation has been spent.
        let partition_hosts = match metadata.funding() {
            Some(funding) if geometry.max_output_tokens > 0 => partition::host::HostAdmission::prepare(
                session, checkpoint.source(), geometry, checkpoint.next_prediction(), &funding,
            )?,
            _ => None,
        };
        let partition_evidence=match metadata.funding() {
            Some(funding) if geometry.max_output_tokens > 0 =>partition::host::evidence::Admission::prepare(session,checkpoint.source(),
                checkpoint.intervention_source(),selection,geometry,checkpoint.next_prediction(),&funding)?,
            _=>None,
        };
        Ok(Self {
            session,
            checkpoint: Some(checkpoint),
            saved_witness: Some(witness),
            opening_rows: None,
            source: checkpoint.source(),
            paths: selection.paths(),
            selection: selection.clone(),
            host,
            text_interventions: RefCell::new(None),
            partition_hosts: RefCell::new(partition_hosts),
            partition_evidence: RefCell::new(partition_evidence),
            // Existing pins move by closed witness alias in seal_saved. The
            // shared C publisher conservatively reserves its exact capacity;
            // no ordinary inventory is reconstructed before admission.
            paths_pin: None,
            plan_pin: None,
            new_source_bytes: checkpoint.source().capacity_bytes().ok_or_else(unknown)?,
            metadata_funding: metadata.funding(),
        })
    }

    pub(super) fn validate_geometry(&self, geometry: InferenceGeometry) -> Result<(), Error> {
        match self.checkpoint {
            Some(checkpoint) => checkpoint
                .validate_continuation_geometry(geometry)
                .map_err(|cause| Error::Neural(validation::source(self.metadata(), cause))),
            None => self.bind_geometry(geometry).map(|_| ()),
        }
    }

    /// The copied source stays credited only by the existing closed copy proof.
    /// New capture H, controls and C publication remain new request obligations.
    pub(in crate::composition::mlx::session::model_session::text_quote) fn seal_saved(
        &self,
        mut quote: SavedQuote,
        tracking: eredu_runtime::working_memory::SubmissionTrackingFacts,
        graph: eredu_runtime::working_memory::GraphMetadataFacts,
        prefill: eredu_runtime::working_memory::TextPrefillScopeFacts,
        native: eredu_runtime::working_memory::PreparedNativeStoragePlan<
            crate::backend::runtime::residency::storage::native_storage::MlxNativeStorage,
        >,
    ) -> Result<SavedQuote, Error> {
        let checkpoint = self.checkpoint.ok_or_else(unknown)?;
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
        let controls = PreparedTextControlWorkspace::prepare_capture_continuation(
            checkpoint,
            quote.geometry(),
            quote.span_workspace().plan(),
            // Saved admission uses typed source-preserving sequence failures even
            // when the facade already owns its copied decoder/cursor bank.
            self.control_facts_with_sequence(quote.geometry(), true)?,
        )
        .map_err(memory)?;
        let controls = if checkpoint.next_prediction() == 0 && quote.geometry().max_output_tokens > 0 {
            controls
                .with_prefill_capture_selection(self.bind_geometry(quote.geometry())?)
                .map_err(memory)?
        } else {
            controls
                .with_capture_continuation_selection(checkpoint, &self.selection)
                .map_err(memory)?
        };
        let controls = controls
            .with_capture_plan_publication(publication)
            .map_err(memory)?
            .with_preparation_scopes(super::super::preparation::facts()?)
            .map_err(memory)?
            .with_prediction_scopes(super::super::prediction::facts()?)
            .map_err(memory)?
            .with_submission_tracking(tracking)
            .map_err(memory)?
            .with_graph_metadata(graph)
            .map_err(memory)?
            .with_prefill_scopes(prefill)
            .map_err(memory)?
            .with_native_storage(native)
            .map_err(memory)?;
        quote = quote
            .with_saved_capture_sources(self.saved_witness.ok_or_else(unknown)?, self.source)
            .map_err(memory)?;
        quote
            .with_span_workspace_and_text_controls(controls)
            .map_err(|cause| Error::Neural(validation::source(self.metadata(), cause)))
    }

    pub(in crate::composition::mlx::session::model_session::text_quote) fn prepare_saved(
        self,
        request: &InferenceRequest,
        funding: &WorkingMemoryFundingRun,
        scope: &WorkingMemoryFundingScope,
        accepted: SavedQuote,
    ) -> Result<CaptureQuotation, AdmissionFailure> {
        let geometry = accepted.geometry();
        self.validate_geometry(geometry)?;
        self.selection
            .validate_sources(self.source, self.paths)
            .map_err(|cause| Error::Neural(validation::source(self.metadata(), cause)))?;
        self.session
            .payload
            .model
            .erased()
            .validate_prepared_observation_paths(self.paths, self.metadata())?;
        if !accepted
            .pool()
            .same_domain(&self.session.payload.memory_pool)
            || geometry != request.geometry()
        {
            return Err(memory(WorkingMemoryError::IdentityMismatch).into());
        }
        let pending = accepted
            .begin_capture_plan_publication(
                funding,
                request.memory_reservation().ok_or_else(unknown)?,
                self.source,
            )
            .map_err(|cause| AdmissionFailure::retained(cause, true))?;
        let (span, witness) = pending
            .publish_and_finish(scope)
            .map_err(|cause| AdmissionFailure::retained(cause, true))?;
        self.finish_prepared(geometry, request, funding, span, witness, true)
    }
}

impl CaptureQuotation {
    pub(in crate::composition::mlx::session::model_session::text_quote) fn take_saved_installed(
        &self,
        session: &MlxModelSession,
        checkpoint: &FundedCaptureCheckpoint,
        kind: eredu_core::OriginalTextResumeKind,
    ) -> Result<super::super::super::text_capture::InstalledCapture, Error> {
        self.validate_checkpoint_source(checkpoint.source(), self.geometry)
            .map_err(memory)?;
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
            .validate_prepared_observation_paths(paths, self.metadata())?;
        let PendingCaptureBank {
            bank,
            selection,
            witness,
            span,
        } = self.take_pending_validated(
            session,
            checkpoint.source(),
            self.geometry,
            Some(checkpoint),
        )?;
        super::super::super::text_capture::InstalledCapture::from_checkpoint(
            bank, selection, witness, span, checkpoint, kind,
        )
    }
}
