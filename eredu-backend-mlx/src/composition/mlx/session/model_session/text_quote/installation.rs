//! Shared consuming installation of one actually accepted text candidate.
//!
//! Input-specific quotation and source authentication stay with their producers.
//! This worker only promotes the accepted proof, using the same native banks and
//! failure cleanup for every source that reaches it.
use super::*;

pub(super) struct AcceptedInstallation<'a> {
    pub(super) config: TextGenerationConfig,
    pub(super) coordinates: PredictionCoordinates,
    pub(super) controller_contract: TextControllerContract,
    pub(super) storage_contract: ControllerStorageContract,
    pub(super) opening: eredu_runtime::working_memory::InferenceRetention,
    pub(super) predecessor: Option<InferenceRequest>,
    pub(super) parameter_epoch: u64,
    pub(super) capacity_bytes: u64,
    pub(super) original_native: bool,
    pub(super) has_sequence: bool,
    pub(super) capture: Option<CaptureAdmission<'a>>,
    pub(super) layerwise_workspace:
        Option<crate::backend::runtime::execution::generic::LayerwiseWorkspace>,
    pub(super) original_table: Option<original_table::Plan<'a>>,
    pub(super) accepted: Option<IncrementalInferenceQuote>,
    pub(super) native_recipe: Option<crate::backend::nn::workspace::ResidentNativeRecipe>,
    pub(super) prepared_source: Option<original_prepared::PreparedMediaQuoteSource>,
    pub(super) paged_sources: Option<crate::backend::nn::workspace::ProjectedPagedSources>,
    // Source and diagnostic fields retire before the accepted accounting owner.
    pub(super) reservation: eredu_runtime::working_memory::WorkingMemoryReservation,
    // Last: the actual planning account retains every moved cold destination.
    pub(super) planning_metadata: Option<eredu_nn::workspace::WorkspaceMetadataFunding>,
}
impl<'a> AcceptedInstallation<'a> {
    pub(super) fn install<C: TokenFilterController>(
        self,
        runtime: &ModelRuntime<MlxBackend<'_>>,
        controller: &C,
        mut decoder_staging: sequence::DecoderStaging<'_>,
    ) -> Result<(InferenceTextPreparation, TextExecutionQuoteOwner), AdmissionFailure> {
        let Self {
            config,
            coordinates,
            controller_contract,
            storage_contract,
            opening,
            predecessor,
            parameter_epoch,
            capacity_bytes,
            original_native,
            has_sequence,
            capture,
            layerwise_workspace,
            original_table,
            mut accepted,
            native_recipe,
            prepared_source,
            paged_sources,
            mut reservation,
            planning_metadata,
        } = self;
        // The moved reservation is already live. Install the lexical source
        // cleanup before the first fallible pin or conversion in this worker.
        let layerwise_workspace = layerwise_workspace;
        let mut decoder_staging = decoder_staging.reborrow();
        let session = runtime.session();
        let model = &session.payload.model;
        let policy = config.inference_policy();
        // This ready-only pin prevents a later eviction from changing the selected
        // host copy into unquoted disk materialization. It mutates residency only
        // after successful budget admission and performs no native work.
        let original_source_pin = original_native
            || policy.submission_tracking_capacity_bytes.is_some()
            || policy.graph_metadata_capacity_bytes.is_some();
        let host_sources = if original_source_pin {
            // The actual accepted original guard is available after neutral
            // capture/sequence promotion below. No producer can run before this pin.
            None
        } else {
            layerwise_workspace
                .as_ref()
                .map(|workspace| workspace.pin_sources())
                .transpose()?
                .flatten()
        };
        // A disk receipt retains only the admitted read metadata and capacity
        // envelope. Scheduling restrictions begin with each native operation.
        let disk = layerwise_workspace
            .as_ref()
            .and_then(|workspace| workspace.disk_receipt());
        let mut funding = reservation.start_funding().map_err(memory)?;
        // Funding now owns the accepted charge. Preserve the same failure ordering
        // across that transfer, before any further fallible preparation.
        let layerwise_workspace = layerwise_workspace;
        // Immediate reborrow after conversion: this source cleanup now precedes the
        // actual run too. Outer cleanup remains empty after successful bank handoff.
        let mut decoder_staging = decoder_staging.reborrow();
        #[cfg(test)]
        sequence::fixture::decoder_checkpoint(runtime, true).map_err(AdmissionFailure::Sequence)?;
        // Successful reservation commits the explicit successor ceiling. Keep its
        // delegation even if later preparation fails; dropping a token neither
        // releases old physical charges nor rolls back an adopted domain policy.
        session
            .capacity_handoffs
            .borrow_mut()
            .push(funding.take_capacity_handoff().map_err(memory)?);
        let request = InferenceRequest::from(reservation);
        let preparation = request
            .prepare_text(
                model.erased().inference_execution_identity(),
                request.geometry(),
                config,
            )
            .map_err(memory)?;
        // Existing immutable masks can outlive this run through aliases created
        // before admission. Install their exact charges on those shared owners,
        // after every request check and before any prompt or sampler work.
        let scope = funding.scope().map_err(memory)?;
        let capture = capture
            .map(|capture| {
                capture.prepare(
                    session,
                    preparation.request(),
                    &funding,
                    &scope,
                    accepted
                        .take()
                        .expect("captured admission retains its accepted source proof"),
                    has_sequence,
                )
            })
            .transpose()?;
        if storage_contract.has_original_domain() {
            storage_contract
                .prepare_original_source(
                    controller,
                    &funding,
                    preparation
                        .request()
                        .memory_reservation()
                        .ok_or_else(unknown)?,
                )
                .map_err(|error| Error::Other(Box::new(error)))?;
        } else {
            storage_contract
                .adopt(controller, &scope)
                .map_err(|error| Error::Other(Box::new(error)))?;
        }
        scope.certify().map_err(memory)?;
        let mut sequence = if has_sequence
            || (capture.is_none()
                && (policy.submission_tracking_capacity_bytes.is_some()
                    || policy.graph_metadata_capacity_bytes.is_some()
                    || original_table.is_some()))
        {
            Some(match &capture {
                Some(capture) => capture
                    .take_sequence_quotation()
                    .expect("same accepted capture span includes its original sequence bank"),
                None => SequenceQuotation::promote(
                    accepted
                        .take()
                        .expect("sequence admission retains its accepted proof"),
                    &funding,
                    preparation.request(),
                )?,
            })
        } else {
            None
        };
        let host_sources = if original_source_pin {
            let controls = capture
                .as_ref()
                .map(CaptureQuotation::control_guard)
                .or_else(|| sequence.as_ref().map(SequenceQuotation::control_guard))
                .ok_or_else(|| memory(WorkingMemoryError::IdentityMismatch))?;
            layerwise_workspace
                .as_ref()
                .map(|workspace| workspace.pin_initial_original_sources(&controls))
                .transpose()?
                .flatten()
        } else {
            host_sources
        };
        if decoder_staging.has_source() {
            // A source exists only with the genuine original sequence claim.
            sequence
                .as_mut()
                .expect("original decoder has a sequence bank")
                .install_decoder(decoder_staging.pending())
                .map_err(AdmissionFailure::Sequence)?;
        }
        let tracking_capsule = if let Some(capture) = &capture {
            capture.take_submission_tracking(&preparation)?
        } else if let Some(controls) = &mut sequence {
            controls.take_submission_tracking(&preparation)?
        } else {
            None
        };
        let record_quota = tracking_capsule.map(tracking::allocate).transpose()?;
        if (original_native || policy.submission_tracking_capacity_bytes.is_some())
            != record_quota.is_some()
        {
            return Err(memory(WorkingMemoryError::IdentityMismatch).into());
        }
        let graph_capsule = if let Some(capture) = &capture {
            capture.take_graph_metadata(&preparation)?
        } else if let Some(controls) = &mut sequence {
            controls.take_graph_metadata(&preparation)?
        } else {
            None
        };
        let graph_quota = graph_capsule.map(graph::allocate).transpose()?;
        if (original_native || policy.graph_metadata_capacity_bytes.is_some())
            != graph_quota.is_some()
        {
            return Err(memory(WorkingMemoryError::IdentityMismatch).into());
        }
        if let Some(recipe) = &native_recipe {
            let controls = capture
                .as_ref()
                .map(CaptureQuotation::control_guard)
                .or_else(|| sequence.as_ref().map(SequenceQuotation::control_guard))
                .ok_or_else(unknown)?;
            pipeline_cache::install(recipe, &controls, graph_quota.as_ref().ok_or_else(unknown)?)?;
        }
        // Extract from the same accepted source before any original role begins.
        // Native creation runs after the capture/span interior borrow has ended.
        let native_storage = if let Some(mechanism) = model.erased().native_storage_mechanism()? {
            use eredu_runtime::working_memory::OriginalNativeStorageMechanism as _;
            let bank = if let Some(capture) = &capture {
                capture.take_native_storage(&funding, mechanism.selection())?
            } else if let Some(sequence) = &mut sequence {
                sequence.take_native_storage(&funding, mechanism.selection())?
            } else {
                None
            };
            bank.map(|mut bank| {
                let controls = capture
                    .as_ref()
                    .map(CaptureQuotation::control_guard)
                    .or_else(|| sequence.as_ref().map(SequenceQuotation::control_guard))
                    .expect("accepted native bank has exact original controls");
                bank.install(mechanism).map_err(|cause| {
                    crate::backend::runtime::residency::storage::native_storage::retained_failure(
                        cause, controls, true,
                    )
                })?;
                Ok::<_, Error>(
                    crate::backend::runtime::residency::storage::native_storage::BankOwner::new(
                        bank,
                    ),
                )
            })
            .transpose()?
        } else {
            None
        };
        let preparation_scopes = if let Some(capture) = &capture {
            capture
                .take_preparation_scopes()?
                .map(preparation::PreparationScopes::new)
        } else if let Some(sequence) = &mut sequence {
            sequence
                .take_preparation_scopes()?
                .map(preparation::PreparationScopes::new)
        } else {
            None
        };
        if (capture.is_some() || sequence.is_some()) && preparation_scopes.is_none() {
            return Err(memory(WorkingMemoryError::IdentityMismatch).into());
        }
        let prediction_scopes = if let Some(capture) = &capture {
            capture.take_prediction_scopes()?.map(|bank| {
                prediction::PredictionScopes::new(
                    bank,
                    record_quota.clone(),
                    graph_quota.clone(),
                    capture.control_guard(),
                )
                .with_native_storage(native_storage.clone())
            })
        } else if let Some(sequence) = &mut sequence {
            sequence.take_prediction_scopes()?.map(|bank| {
                prediction::PredictionScopes::new(
                    bank,
                    record_quota.clone(),
                    graph_quota.clone(),
                    sequence.control_guard(),
                )
                .with_native_storage(native_storage.clone())
            })
        } else {
            None
        };
        if (capture.is_some() || sequence.is_some()) && prediction_scopes.is_none() {
            return Err(memory(WorkingMemoryError::IdentityMismatch).into());
        }
        let prefill_scopes = if let Some(capture) = &capture {
            capture.take_prefill_scopes()?.map(RefCell::new)
        } else if let Some(sequence) = &mut sequence {
            sequence.take_prefill_scopes()?.map(RefCell::new)
        } else {
            None
        };
        let original_table = original_table
            .map(|source| {
                let controls = capture
                    .as_ref()
                    .map(CaptureQuotation::control_guard)
                    .or_else(|| sequence.as_ref().map(SequenceQuotation::control_guard))
                    .ok_or_else(|| memory(WorkingMemoryError::IdentityMismatch))?;
                source.finish(session, preparation.request(), controls)
            })
            .transpose()?;
        let operation_registration = capture
            .as_ref()
            .map(CaptureQuotation::control_guard)
            .or_else(|| sequence.as_ref().map(SequenceQuotation::control_guard))
            .map(crate::backend::runtime::execution::generic::OriginalOperationRegistration::new);
        #[cfg(test)]
        prefill_tests::capture_host_destinations(sequence.as_mut())?;
        let needs_host_bank = match &prefill_scopes {
            Some(scopes) => scopes
                .try_borrow()
                .map_err(|_| Error::PrefillScopeReentrant)?
                .facts()
                .and_then(|facts| facts.host_destination_facts())
                .is_some(),
            None => false,
        };
        let operation_host_destinations = if needs_host_bank {
            if let Some(capture) = &capture {
                capture.take_host_destinations()?
            } else if let Some(sequence) = &mut sequence {
                sequence.take_host_destinations()?
            } else {
                return Err(memory(WorkingMemoryError::IdentityMismatch).into());
            }
        } else {
            None
        };
        let layerwise = layerwise_workspace.map(|workspace| {
            if record_quota.is_some() || graph_quota.is_some() {
                LayerwiseQuoteSources::Retained(workspace)
            } else {
                LayerwiseQuoteSources::Identity(workspace.into_identity())
            }
        });
        let parallel_control=match native_recipe.as_ref().map(|recipe|recipe.parallel_control_source()).transpose()?.flatten() {
            Some(source)=>{
                let controls=capture.as_ref().map(CaptureQuotation::control_guard)
                    .or_else(||sequence.as_ref().map(SequenceQuotation::control_guard))
                    .ok_or(Error::PrefillScopeUnavailable)?;
                let bank=native_storage.as_ref().ok_or(Error::PrefillScopeUnavailable)?.clone();
                Some(crate::backend::runtime::distributed::topology::original_source::control::OriginalParallelControlOwner::new(
                    source.clone(),bank,preparation.request(),controls)?.with_token_sources(
                        if let Some(capture) = &capture { capture.take_output_source_constructions()? }
                        else { sequence.as_mut().ok_or(Error::PrefillScopeUnavailable)?.take_output_source_constructions()? }
                    )?)
            }
            None=>None,
        };
        let proof = TextExecutionQuoteOwner::new(TextExecutionQuote {
            config,
            coordinates,
            controller: controller_contract,
            storage_contract,
            session: Rc::clone(&session.poison),
            model_pool: session.payload.memory_pool.clone(),
            context_pool: runtime.backend().memory_pool().clone(),
            parameter_epoch,
            layerwise,
            _host_sources: host_sources,
            disk,
            source_capacity_bytes: capacity_bytes,
            request: preparation.request().clone(),
            opening: OpeningSeal::ordinary(opening.revision().clone(), predecessor),
            capture,
            funding: RefCell::new(Some(funding)),
            sequence,
            preparation_scopes,
            prediction_scopes,
            operation_host_destinations: RefCell::new(operation_host_destinations),
            operation_banks: RefCell::new(None),
            operation_registration,
            prefill_scopes,
            record_quota,
            graph_quota,
            native_recipe,
            native_storage,
            parallel_control,
            addressable: RefCell::new(None),
        paged_sources: RefCell::new(paged_sources),
            prepared_source,
            saved_cache_source: None,
            continuation_metadata: None,
            original_table,
            planning_metadata: planning_metadata.clone(),
        });
        Ok((preparation, proof))
    }
}

/// Concrete handoff representation and its additional lexical reborrow. These
/// controls are reserved by the same quote before its candidate is accepted.
pub(super) fn control_bytes() -> Option<u64> {
    [
        std::mem::size_of::<AcceptedInstallation<'static>>(),
        std::mem::size_of::<sequence::DecoderStaging<'static>>(),
        std::mem::size_of::<&ModelRuntime<MlxBackend<'static>>>(),
        std::mem::size_of::<&()>(), // The actual sized controller loan.
        std::mem::size_of::<
            Result<(InferenceTextPreparation, TextExecutionQuoteOwner), AdmissionFailure>,
        >(),
    ]
    .into_iter()
    .try_fold(0usize, usize::checked_add)
    .and_then(|bytes| u64::try_from(bytes).ok())
}
