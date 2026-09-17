//! Construct the accepted owner only after its recursive cold quote returns.
use super::*;
use eredu_nn::workspace::WorkspaceMetadataFunding;

pub(super) fn control_bytes<C: TokenFilterController>() -> Option<usize> {
    [
        std::mem::size_of::<PreparedSavedTextResumeQuote<'_>>(),
        std::mem::size_of::<Result<PendingSavedTextAdmission, Error>>(),
        std::mem::size_of::<(
            &ModelRuntime<MlxBackend<'_>>,
            &CopiedTextComponentsOwner,
            TextGenerationConfig,
            &C,
            &TextStepContext,
            Option<&eredu_core::HostPreparationAuthority>,
            eredu_core::OriginalTextResumeKind,
            TextControllerWorkspace<'_>,
            ControllerStorageContract,
            Option<super::super::super::control_slot::PreparedControlExchange>,
            u64,
            u64,
            &Option<WorkspaceMetadataFunding>,
        )>(),
    ]
    .into_iter()
    .try_fold(0usize, usize::checked_add)
}

#[inline(never)]
pub(super) fn admit<C: TokenFilterController>(
    runtime: &ModelRuntime<MlxBackend<'_>>,
    source: &CopiedTextComponentsOwner,
    config: TextGenerationConfig,
    controller: &C,
    context: &TextStepContext,
    host: Option<&eredu_core::HostPreparationAuthority>,
    kind: eredu_core::OriginalTextResumeKind,
    mut diagnostic: PreparedSavedTextResumeQuote<'_>,
    workspace: TextControllerWorkspace<'_>,
    storage_contract: ControllerStorageContract,
    original_exchange: Option<super::super::super::control_slot::PreparedControlExchange>,
    capacity: u64,
    outputs: u64,
    planning_metadata: &Option<WorkspaceMetadataFunding>,
) -> Result<PendingSavedTextAdmission, Error> {
    let original = host.is_some();
    let session = runtime.session();
    let geometry = diagnostic.geometry();
    let capture = match (
        source.capture_checkpoint(),
        source.capture_selection(),
        source.capture_witness(),
    ) {
        (Some(checkpoint), Some(selection), Some(witness)) => {
            Some(CaptureAdmission::from_checkpoint(
                session,
                geometry,
                checkpoint,
                selection,
                witness,
                eredu_runtime::working_memory::WorkspaceReportMetadata::with_funding(
                    planning_metadata.as_ref().ok_or_else(|| unknown())?,
                ),
            )?.with_saved_interventions(diagnostic.take_text_interventions(),geometry)?)
        }
        (None, None, None) => {
            if diagnostic.take_text_interventions().is_some(){return Err(memory(WorkingMemoryError::IdentityMismatch));}
            None
        },
        _ => return Err(memory(WorkingMemoryError::IdentityMismatch)),
    };
    // The immutable pending kind and complete source matrix fix input geometry.
    // Apply the shared incremental-admission policy to that same copy program.
    geometry
        .validate()
        .map_err(|cause| Error::Other(Box::new(cause)))?;
    let contract = TextControllerContract::from_workspace(workspace, diagnostic.output_width())
        .map_err(|error| Error::Other(Box::new(error)))?;
    if diagnostic.controller_contract() != &contract {
        return Err(memory(WorkingMemoryError::IdentityMismatch));
    }
    let source_account = diagnostic.registered_source()?;
    let source_registration = diagnostic.retain_source_registration();
    let capabilities = session.payload.model.erased().capability_estimate();
    let request_policy = AdmissionRequest {
        input: diagnostic.state_input(),
        max_output_tokens: outputs,
        batch_size: 1,
        safety_reserve_bytes: 0,
        application_memory_budget_bytes: None,
        require_complete_estimate: true,
    };
    let full = if let Some(funding) = planning_metadata.as_ref() {
        let metadata =
            eredu_runtime::working_memory::WorkspaceReportMetadata::with_funding(funding);
        let incremental = metadata.bounded(
            diagnostic.incremental().incremental_bytes(),
            format_args!("sealed decoder/key copy source credit with full pending, host and future execution costs"),
        ).map_err(|cause| Error::Neural(metadata.error(cause)))?;
        match metadata
            .apply_admission_with_incremental(
                capabilities.capabilities(),
                request_policy,
                diagnostic.full(),
                &incremental,
                None,
            )
            .map_err(|cause| Error::Neural(metadata.error(cause)))?
        {
            AdmissionResult::Admitted(admission) => admission,
            AdmissionResult::Rejected(rejection) => {
                return Err(Error::Neural(
                    metadata.source(PrefillPlanningError::Admission(rejection)),
                ));
            }
        }
    } else {
        match eredu_core::apply_admission_policy_with_incremental(
            capabilities.capabilities(),
            request_policy,
            diagnostic.full().clone(),
            &eredu_core::WorkspaceBound::bounded(
                diagnostic.incremental().incremental_bytes(),
                "sealed decoder/key copy source credit with full pending, host and future execution costs",
            ),
            None,
        ).map_err(capability)? {
            AdmissionResult::Admitted(admission) => admission,
            AdmissionResult::Rejected(rejection) => {
                return Err(Error::Other(Box::new(PrefillPlanningError::Admission(rejection))));
            }
        }
    };
    let coordinates = PredictionCoordinates::new(source.sampling().next_prediction(), geometry)
        .map_err(memory)?;
    let opening = session
        .payload
        .model
        .erased()
        .retained_inference_authority()?;
    let initial_revision = opening.revision().clone();
    let opening = OpeningSeal::pending(&opening);
    let mut epoch = None;
    session.validate_parameter_epoch(&mut epoch)?;
    let parameter_epoch = epoch.ok_or_else(|| memory(WorkingMemoryError::IdentityMismatch))?;
    let execution = session
        .payload
        .model
        .erased()
        .inference_execution_identity();
    // Pruning is metadata-only. Live delegations are revalidated together with
    // both saved owners and every registered source origin under the pool lock.
    {
        let mut handoffs = session.capacity_handoffs.borrow_mut();
        let mut index = 0;
        while index < handoffs.len() {
            if handoffs[index].is_retired().map_err(memory)? {
                handoffs.swap_remove(index);
            } else {
                index += 1;
            }
        }
    }
    let mut reservation = diagnostic
        .incremental()
        .reserve_saved_source_with_capacity_handoff(
            runtime.backend().memory_pool(),
            execution,
            &full,
            capacity,
            &session.capacity_handoffs.borrow(),
            &source_account,
        )
        .map_err(memory)?;
    // Move the paid owner after its accepted reservation before any fallible
    // pin. The rest of the diagnostic remains cold temporary storage.
    let layerwise_workspace = diagnostic.take_layerwise_workspace();
    drop(source_account);
    let host_sources = if original {
        None
    } else {
        layerwise_workspace
            .as_ref()
            .map(|workspace| workspace.pin_sources())
            .transpose()?
            .flatten()
    };
    let disk = layerwise_workspace
        .as_ref()
        .and_then(|workspace| workspace.disk_receipt());
    let native_recipe = diagnostic.take_native_recipe();
    let source_native = diagnostic.take_source_native();
    let prepared_source = diagnostic.prepared_media_source()?;
    // Publication retains the exact captured cache's B account, separately
    // from equation source credit and a genuine pending media input.
    let saved_cache_source = if original
        && prepared_source.is_none()
        && source.input_identity().is_some()
    {
        let metadata = planning_metadata.as_ref().ok_or_else(|| unknown())?;
        let controls = eredu_runtime::SharedPreparedInputCacheIdentity::original_publication_source_control_bytes()
            .ok_or(Error::PrefillControl(WorkingMemoryError::Overflow))?;
        metadata
            .reserve_metadata(controls)
            .map_err(Error::WorkspacePlanning)?;
        source
            .input_identity()
            .and_then(|cache| cache.original_publication_source(runtime.backend().memory_pool()))
            .transpose()
            .map_err(Error::PrefillControl)?
    } else {
        None
    };
    let continuation_metadata = diagnostic.take_continuation_metadata();
    let paged_source_metadata = diagnostic.take_paged_source_metadata();
    let paged_host_facts = diagnostic.paged_host_facts();
    let mut accepted = original.then(|| diagnostic.into_incremental());
    let mut funding = reservation.start_funding().map_err(memory)?;
    // A successful admission commits an explicitly selected successor ceiling.
    // Later construction failure never rolls that policy change back.
    session
        .capacity_handoffs
        .borrow_mut()
        .push(funding.take_capacity_handoff().map_err(memory)?);
    let request = InferenceRequest::from(reservation);
    let preparation = request
        .prepare_text(execution, geometry, config)
        .map_err(memory)?;
    preparation.bind_run(context).map_err(memory)?;
    let scope = funding.scope().map_err(memory)?;
    let capture = capture
        .map(|capture| {
            capture
                .prepare_saved(
                    preparation.request(),
                    &funding,
                    &scope,
                    accepted.take().ok_or_else(|| unknown())?,
                )
                .map_err(|cause| Error::StorageSource(cause.into_backend()))
        })
        .transpose()?;
    if original {
        storage_contract
            .prepare_original_source(
                controller,
                &funding,
                preparation
                    .request()
                    .memory_reservation()
                    .ok_or_else(|| unknown())?,
            )
            .map_err(|error| Error::Other(Box::new(error)))?;
    } else {
        storage_contract
            .adopt(controller, &scope)
            .map_err(|error| Error::Other(Box::new(error)))?;
    }
    scope.certify().map_err(memory)?;
    let mut sequence = accepted
        .map(|accepted| {
            SequenceQuotation::promote_saved(accepted, &funding, preparation.request())
                .map_err(|cause| Error::StorageSource(cause.into_backend()))
        })
        .transpose()?;
    let controls = capture
        .as_ref()
        .map(CaptureQuotation::control_guard)
        .or_else(|| sequence.as_ref().map(SequenceQuotation::control_guard));
    let host_sources = if let Some(controls) = &controls {
        layerwise_workspace
            .as_ref()
            .map(|workspace| workspace.pin_initial_original_sources(controls))
            .transpose()?
            .flatten()
    } else {
        host_sources
    };
    let record_quota = if let Some(capture) = &capture {
        capture.take_submission_tracking(&preparation)?
    } else if let Some(sequence) = &mut sequence {
        sequence.take_submission_tracking(&preparation)?
    } else {
        None
    }
    .map(tracking::allocate)
    .transpose()?;
    let graph_quota = if let Some(capture) = &capture {
        capture.take_graph_metadata(&preparation)?
    } else if let Some(sequence) = &mut sequence {
        sequence.take_graph_metadata(&preparation)?
    } else {
        None
    }
    .map(graph::allocate)
    .transpose()?;
    if original && (record_quota.is_none() || graph_quota.is_none()) {
        return Err(unknown());
    }
    if let Some(recipe) = &native_recipe {
        pipeline_cache::install(
            recipe,
            controls.as_ref().ok_or_else(|| unknown())?,
            graph_quota.as_ref().ok_or_else(|| unknown())?,
        )?;
    }
    let native_storage = if let Some(controls) = &controls {
        use eredu_runtime::working_memory::OriginalNativeStorageMechanism as _;
        let mechanism = session
            .payload
            .model
            .erased()
            .native_storage_mechanism()?
            .ok_or_else(|| unknown())?;
        let mut bank = if let Some(capture) = &capture {
            capture.take_native_storage(&funding, mechanism.selection())?
        } else if let Some(sequence) = &mut sequence {
            sequence.take_native_storage(&funding, mechanism.selection())?
        } else {
            None
        }
        .ok_or_else(|| unknown())?;
        bank.install(mechanism).map_err(|cause| {
            crate::backend::runtime::residency::storage::native_storage::retained_failure(
                cause,
                controls.clone(),
                true,
            )
        })?;
        Some(crate::backend::runtime::residency::storage::native_storage::BankOwner::new(bank))
    } else {
        None
    };
    let preparation_scopes = if let Some(capture) = &capture {
        capture.take_preparation_scopes()?
    } else if let Some(sequence) = &mut sequence {
        sequence.take_preparation_scopes()?
    } else {
        None
    }
    .map(preparation::PreparationScopes::new);
    let prediction_scopes = if let Some(capture) = &capture {
        capture.take_prediction_scopes()?
    } else if let Some(sequence) = &mut sequence {
        sequence.take_prediction_scopes()?
    } else {
        None
    }
    .map(|bank| {
        prediction::PredictionScopes::new(
            bank,
            record_quota.clone(),
            graph_quota.clone(),
            controls
                .as_ref()
                .expect("original prediction controls")
                .clone(),
        )
        .with_native_storage(native_storage.clone())
    });
    let prefill_scopes = if let Some(capture) = &capture {
        capture.take_prefill_scopes()?
    } else if let Some(sequence) = &mut sequence {
        sequence.take_prefill_scopes()?
    } else {
        None
    }
    .map(RefCell::new);
    if original
        && (preparation_scopes.is_none() || prediction_scopes.is_none() || prefill_scopes.is_none())
    {
        return Err(unknown());
    }
    let needs_host_bank = prefill_scopes
        .as_ref()
        .map(|scopes| {
            scopes
                .borrow()
                .facts()
                .and_then(|facts| facts.host_destination_facts())
                .is_some()
        })
        .unwrap_or(false);
    let operation_host_destinations = if needs_host_bank {
        if let Some(capture) = &capture {
            capture.take_host_destinations()?
        } else {
            sequence
                .as_mut()
                .ok_or_else(|| unknown())?
                .take_host_destinations()?
        }
    } else {
        None
    };
    let operation_registration = controls
        .map(crate::backend::runtime::execution::generic::OriginalOperationRegistration::new);
    let layerwise = layerwise_workspace.map(|workspace| {
        if original {
            LayerwiseQuoteSources::Retained(workspace)
        } else {
            LayerwiseQuoteSources::Identity(workspace.into_identity())
        }
    });
    let parallel_control = match native_recipe
        .as_ref()
        .map(|recipe| recipe.parallel_control_source())
        .transpose()?
        .flatten()
    {
        Some(source) => {
            let controls = capture
                .as_ref()
                .map(CaptureQuotation::control_guard)
                .or_else(|| sequence.as_ref().map(SequenceQuotation::control_guard))
                .ok_or(Error::PrefillScopeUnavailable)?;
            let bank = native_storage
                .as_ref()
                .ok_or(Error::PrefillScopeUnavailable)?
                .clone();
            Some(crate::backend::runtime::distributed::topology::original_source::control::OriginalParallelControlOwner::new(
                source.clone(),bank,preparation.request(),controls)?.with_token_sources(
                    if let Some(capture) = &capture { capture.take_output_source_constructions()? }
                    else { sequence.as_mut().ok_or(Error::PrefillScopeUnavailable)?.take_output_source_constructions()? }
                )?)
        }
        None => None,
    };
    let quote = TextExecutionQuoteOwner::new(TextExecutionQuote {
        // The completed copied manager supplies this at seal_installed;
        // saved-source manager authority is never installed for its copy.
        addressable: RefCell::new(None),
        paged_sources: RefCell::new(None),
        native_recipe,
        native_storage,
        parallel_control,
        config,
        coordinates,
        controller: contract,
        storage_contract,
        session: Rc::clone(&session.poison),
        model_pool: session.payload.memory_pool.clone(),
        context_pool: runtime.backend().memory_pool().clone(),
        parameter_epoch,
        layerwise,
        _host_sources: host_sources,
        disk,
        // Pending native input uses the separate closed one-part host plan;
        // there is no admitted host token vector on this route.
        source_capacity_bytes: 0,
        request: preparation.request().clone(),
        opening,
        capture,
        sequence,
        preparation_scopes,
        prediction_scopes,
        operation_host_destinations: RefCell::new(operation_host_destinations),
        operation_banks: RefCell::new(None),
        operation_registration,
        prefill_scopes,
        record_quota,
        graph_quota,
        prepared_source,
        saved_cache_source,
        continuation_metadata,
        original_table: None,
        planning_metadata: (*planning_metadata).clone(),
        funding: RefCell::new(Some(funding)),
    });
    Ok(PendingSavedTextAdmission {
        paged_source_metadata,
        paged_host_facts,
        source: source.clone(),
        source_native,
        source_registration,
        ordinary: Some(MlxTextPreparation {
            request: Some(preparation),
            quote: Some(quote),
            chunk: std::num::NonZeroU64::new(1),
        }),
        context: context.clone(),
        initial_revision,
        installed: Cell::new(false),
        kind,
        original_exchange,
        _host_preparation: host.cloned(),
    })
}
