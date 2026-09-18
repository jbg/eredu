//! Shared post-trace quote assembly from the actual equation and sampling reports.
use super::*;
use crate::backend::nn::workspace::ResidentNativeRecipe;
use crate::backend::runtime::residency::storage::StorageIdentity;
use eredu_nn::workspace::WorkspaceContext;
use eredu_runtime::working_memory::{
    IncompleteWorkspace, InferenceWorkspaceReport, RegisteredPreparedWorkspaceStorage,
    RegisteredWorkspaceStorage, SamplingWorkspaceReport,
};

/// Quote-stage diagnostics preserve the actual native cause, classification,
/// and planning-account custody for both token and completed-media sources.
#[derive(Debug, thiserror::Error)]
#[error("workspace quote {stage}: {cause}; missing equation {equation:?}")]
struct WorkspaceQuoteFailure {
    stage: &'static str,
    #[source]
    cause: Error,
    equation: Option<(usize, usize, Option<String>)>,
}
fn workspace_quote_failure(
    context: Option<&WorkspaceContext>,
    stage: &'static str,
    cause: Error,
    equation: Option<(usize, usize, Option<String>)>,
) -> Error {
    if matches!(
        cause,
        Error::PrefillControl(
            WorkingMemoryError::SubmissionTrackingCapacity { .. }
                | WorkingMemoryError::GraphMetadataCapacity { .. }
        )
    )
    {
        return cause;
    }
    let Some(funding) = context.and_then(WorkspaceContext::metadata_funding) else {
        return cause;
    };
    let kind = cause
        .retained_backend_failure_kind()
        .unwrap_or(BackendFailureKind::Other);
    let preserved = cause.model_state_preserved();
    crate::composition::mlx::model::retain_planning_error_with_kind(
        WorkspaceQuoteFailure { stage, cause, equation },
        funding,
        kind,
        preserved,
    )
}

/// A loan of one completed observation. Owning callers retain the same source,
/// report metadata and funding through result publication or error retirement.
pub(super) struct ObservedWorkspace<'a> {
    pub(super) equations: &'a InferenceWorkspaceReport,
    pub(super) sampling: &'a SamplingWorkspaceReport,
    pub(super) storage: ObservedStorage<'a>,
    pub(super) state_input: InputTokenCount,
}
#[derive(Clone, Copy)]
pub(super) enum ObservedStorage<'a> {
    Unregistered,
    Registered(&'a RegisteredWorkspaceStorage<StorageIdentity>),
    CompletedInput(&'a RegisteredPreparedWorkspaceStorage<StorageIdentity>),
}
impl<'a> ObservedStorage<'a> {
    fn completed_source(
        &self,
    ) -> Option<&'a eredu_runtime::input::OriginalPreparedWorkspaceSource> {
        match self {
            Self::CompletedInput(storage) => Some(storage.prepared_source()),
            _ => None,
        }
    }
}

pub(super) fn quote_observed_with_sequence(
    session: &MlxModelSession,
    geometry: InferenceGeometry,
    source_capacity: u64,
    config: TextGenerationConfig,
    controller: TextControllerWorkspace<'_>,
    storage_contract: &ControllerStorageContract,
    registered_controller: Option<&RegisteredControllerStorage>,
    observed: ObservedWorkspace<'_>,
    native_recipe: &mut Option<ResidentNativeRecipe>,
    capture: Option<&CaptureAdmission<'_>>,
    sequence_claim: Option<&eredu_core::GenerationSequencePreparation<'_, '_>>,
    original_table: bool,
    original_native: bool,
    retained_sources: Option<&crate::backend::runtime::execution::generic::LayerwiseWorkspace>,
    context: Option<&WorkspaceContext>,
    paged_sources: Option<&crate::backend::nn::workspace::ProjectedPagedSources>,
) -> Result<Result<IncrementalInferenceQuote, IncompleteWorkspace>, Error> {
    let ObservedWorkspace {
        equations,
        sampling,
        storage,
        state_input,
    } = observed;
    let completed_source = storage.completed_source();
    let at = |stage, cause| workspace_quote_failure(context, stage, cause, None);
    if state_input.model_positions != geometry.cached_positions + geometry.input_positions {
        return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
    }
    let report_metadata = context.map_or_else(
        eredu_runtime::working_memory::WorkspaceReportMetadata::ordinary,
        eredu_runtime::working_memory::WorkspaceReportMetadata::new,
    );
    let report_error = |cause| Error::Neural(report_metadata.error(cause));
    report_metadata
        .admit::<ObservedWorkspace<'_>>()
        .map_err(report_error)?;
    report_metadata
        .admit::<(
            Option<&crate::backend::nn::workspace::ProjectedPagedSources>,
            eredu_runtime::working_memory::HostSourceConstructionFacts,
            eredu_runtime::working_memory::HostDestinationFacts,
            Result<eredu_runtime::working_memory::HostDestinationFacts, WorkingMemoryError>,
            Option<eredu_runtime::working_memory::TextPrefillScopeFacts>,
        )>()
        .map_err(report_error)?;
    report_metadata.admit::<(
        crate::backend::error::WorkspaceQuoteComponents,
        Option<crate::backend::error::WorkspaceQuoteComponents>,
        crate::backend::error::WorkspaceCandidateRefusal,
        Option<(InferenceGeometry, crate::backend::error::WorkspaceQuoteComponents)>,
    )>().map_err(report_error)?;
    let mut components = crate::backend::error::WorkspaceQuoteComponents::default();
    if let Some(recipe) = native_recipe.as_mut() {
        session
            .payload
            .model
            .erased()
            .bind_layerwise_neural_recipe(
                &session.payload.memory_pool,
                recipe,
                context
                    .and_then(WorkspaceContext::metadata_funding)
                    .as_ref(),
            )
            .map_err(|cause| at("neural boundary recipe", cause))?;
        if let Some(sources) = retained_sources {
            sources
                .bind_native_host_copies(recipe)
                .map_err(|cause| at("host copy recipe", cause))?;
        }
    }
    let requested_tracking = config.inference_policy().submission_tracking_capacity_bytes;
    let selected_tracking = match native_recipe.as_mut() {
        Some(recipe) => {
            let requirement = recipe.record_storage_requirement().map_err(|cause| {
                match recipe.take_first_missing_equation() {
                    Ok(equation) => workspace_quote_failure(context, "submission records", cause, equation),
                    Err(refusal) => refusal,
                }
            })?;
            Some(requirement.select_for_policy(requested_tracking).map_err(|cause| {
                // A configured-capacity refusal keeps its direct public form.
                if !matches!(cause, WorkingMemoryError::UnknownBound) {
                    return at("submission record capacity", Error::PrefillControl(cause));
                }
                match recipe.take_first_missing_equation() {
                    Ok(equation) => workspace_quote_failure(context, "submission record capacity", Error::PrefillControl(cause), equation),
                    Err(refusal) => refusal,
                }
            })?)
        }
        None => requested_tracking,
    };
    components.tracking = selected_tracking.map(|bytes| bytes.get());
    let input_layout = sequence_claim
        .and_then(|c| c.request().token_input())
        .map(eredu_runtime::working_memory::OriginalTokenInputLayout::prepare)
        .transpose()
        .map_err(memory)?;
    // Completed B replaces token ingress. Its recipe has already authenticated
    // the exact observed plan/path owners; a token-only selection still cannot
    // enter this source branch. The common seal retains all capture/sequence
    // publication, geometry, original funding and native-source checks below.
    if completed_source.is_some()
        && (input_layout.is_some()
            || !original_native
            || capture.is_some_and(|capture| !capture.prepared_selection().is_prepared_media())
            || sequence_claim.is_none())
    {
        return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
    }
    let (state, prompt, mut outside) = enclosing_observed_components(
        session,
        geometry,
        source_capacity,
        capture.is_none()
            && sequence_claim.is_none()
            && !original_table
            && config
                .inference_policy()
                .submission_tracking_capacity_bytes
                .is_none()
            && config
                .inference_policy()
                .graph_metadata_capacity_bytes
                .is_none(),
        input_layout.as_ref(),
        retained_sources,
        original_native
            || config
                .inference_policy()
                .submission_tracking_capacity_bytes
                .is_some()
            || config
                .inference_policy()
                .graph_metadata_capacity_bytes
                .is_some(),
        context,
        completed_source,
        state_input,
    )
    .map_err(|cause| at("enclosing state and source controls", cause))?;
    if let Some(capture) = capture {
        capture.add_enclosing_metadata(&mut outside, report_metadata)?;
    }
    if let Some(recipe) = &native_recipe {
        if let Some(reason) = recipe.unqualified_host_owner() {
            outside.retained = report_metadata
                .unknown(format_args!("{reason}"))
                .map_err(report_error)?;
        } else {
            let controls = recipe
                .control_bytes()
                .map_err(|cause| at("native recipe controls", Error::Neural(cause)))?;
            components.recipe = Some(controls);
            match crate::backend::nn::tensor::token_validation_control_bytes(recipe)
                .map_err(|cause| at("token validation controls", cause))?
            {
                Some(validation_controls) => {
                    components.validation = Some(validation_controls);
                    let controls = controls
                        .checked_add(validation_controls)
                        .ok_or_else(|| memory(WorkingMemoryError::Overflow))?;
                    if let WorkspaceBound::Bounded { bytes, assumptions } = &mut outside.retained {
                        *bytes = bytes
                            .checked_add(controls)
                            .ok_or_else(|| memory(WorkingMemoryError::Overflow))?;
                        report_metadata.append(assumptions, "; exact resident native recipe, query controls and retained validation batches").map_err(report_error)?;
                    }
                }
                None => {
                    outside.retained = report_metadata
                        .unknown(format_args!(
                            "selected token-validation batch population or host layout is unknown"
                        ))
                        .map_err(report_error)?
                }
            }
        }
    }

    let native_mechanism = if original_native
        || (config
            .inference_policy()
            .submission_tracking_capacity_bytes
            .is_some()
            && config
                .inference_policy()
                .graph_metadata_capacity_bytes
                .is_some())
    {
        session
            .payload
            .model
            .native_storage_mechanism()
            .map_err(|cause| at("selected native storage mechanism", cause))?
    } else {
        None
    };
    if let (Some(recipe), Some(_)) = (&native_recipe, &native_mechanism) {
        match pipeline_cache::control_bytes(recipe).map_err(|cause| at("pipeline cache controls", cause))? {
        Some(controls) => {
            components.pipeline = Some(controls);
            if let WorkspaceBound::Bounded { bytes, assumptions } = &mut outside.retained {
                *bytes = bytes.checked_add(controls)
                    .ok_or_else(|| memory(WorkingMemoryError::Overflow))?;
                report_metadata.append(assumptions, "; exact request-owned precompiled pipeline destinations and selector controls").map_err(report_error)?;
            }
        }
        None => outside.retained = report_metadata.unknown(format_args!("selected precompiled kernel coverage, lookup population or selector controls are unknown")).map_err(report_error)?,
    }
    }
    let requested_graph = config.inference_policy().graph_metadata_capacity_bytes;
    let (selected_graph, graph_fit_controls) =
        match (&native_recipe, &native_mechanism, requested_graph) {
            (Some(recipe), Some(mechanism), ceiling) => {
                let selected = session
                    .payload
                    .model
                    .inference_blueprint()
                    .ok_or_else(unknown)?
                    .selected();
                if (selected.text_realization().residency()
                    != eredu_runtime::LayerWeightResidency::FullyResident
                    && !(retained_sources.is_some() && recipe.has_host_copy_recipe()))
                    || (input_layout.is_none() && completed_source.is_none())
                {
                    // Materialization requires the retained exact source-copy recipe.
                    // An equation-only recipe cannot certify that request arena.
                    return Err(memory(WorkingMemoryError::UnknownBound));
                }
                let fit = match completed_source {
                    Some(source) => recipe
                        .completed_input_graph_storage_requirement(source)
                        .map_err(|cause| at("completed input graph requirement", cause))?,
                    None => {
                        let prompt_facts = mechanism
                            .prompt_input_facts(geometry)
                            .map_err(|cause| Error::Other(Box::new(cause)))?
                            .ok_or_else(unknown)?;
                        recipe.graph_storage_requirement(prompt_facts)
                            .map_err(|cause| at("prompt graph requirement", cause))?
                    }
                };
                (
                    Some(
                        fit.select_for_policy(ceiling)
                            .map_err(|cause| at("graph capacity", Error::PrefillControl(cause)))?,
                    ),
                    fit.control_bytes()
                        .ok_or_else(|| memory(WorkingMemoryError::Overflow))?,
                )
            }
            (Some(_), _, _) => {
                return Err(at(
                    "native recipe requires selected storage mechanism",
                    memory(WorkingMemoryError::UnknownBound),
                ));
            }
            // Legacy component-only callers retain their requested arena without
            // claiming complete Graph fit. Original resident producers require both.
            _ => (requested_graph, 0),
        };
    let native_program = match (&native_mechanism, &native_recipe) {
        (Some(mechanism), Some(recipe))
            if !matches!(storage, ObservedStorage::Unregistered)
                && (input_layout.is_some() || completed_source.is_some()) =>
        {
            let pool = &session.payload.memory_pool;
            let mut nonstate =
                crate::backend::runtime::residency::storage::RetainedStorage::original_census(pool);
            let mut decoder =
                crate::backend::runtime::residency::storage::RetainedStorage::original_census(pool);
            session
                .payload
                .collect_retained_idle_storage(&mut nonstate, &mut decoder)?;
            let opening_rows = match (
                nonstate.original_publication_rows(pool)?,
                decoder.original_publication_rows(pool)?,
            ) {
                (Some(nonstate), Some(decoder)) => Some(
                    nonstate
                        .checked_add(decoder)
                        .ok_or_else(|| memory(WorkingMemoryError::Overflow))?,
                ),
                _ => None,
            };
            match completed_source {
                Some(source) => mechanism
                    .completed_input_program(recipe, source, sampling, opening_rows, paged_sources.is_some())
                    .map_err(|cause| at("completed input native population", cause))?,
                None => mechanism.resident_program(
                    recipe,
                    prompt.as_ref().ok_or_else(unknown)?,
                    sampling,
                    opening_rows,
                    paged_sources.is_some(),
                ).map_err(|cause| at("token input native population", cause))?,
            }
        }
        _ => None,
    };
    components.graph = selected_graph.map(|bytes| bytes.get());
    if let Some(program) = &native_program {
        components.native_capacity = Some(program.capacity_bytes());
        components.publication_attempts = Some(program.attempts());
        components.publication_rows = Some(program.rows());
    }
    let mut outside = sampling
        .enclosing_workspace_metadata(
            match &prompt {
                Some(prompt) => prompt
                    .compose_metadata(outside, report_metadata)
                    .map_err(report_error)?,
                None => outside,
            },
            report_metadata,
        )
        .map_err(report_error)?;
    if let Some(program) = &native_program {
        program.replace_enclosing_metadata(&mut outside, report_metadata)
            .map_err(|cause| at("native enclosing storage replacement", cause))?;
    }
    let tracking = tracking::selected_facts(requested_tracking, selected_tracking)?;
    let graph = graph::selected_facts(requested_graph, selected_graph, graph_fit_controls)?;
    // Every candidate reaches this calculation with its exact geometry.
    // Smaller chunks may increase concurrently retained collector blocks.
    let prefill = graph
        .as_ref()
        .map(|graph| {
            session
                .payload
                .model
                .erased()
                .prefill_control_facts(
                    &session.payload.memory_pool,
                    geometry,
                    graph.capacity(),
                    native_recipe
                        .as_ref()
                        .map(|recipe| recipe.maximum_roots().map_err(|cause| at("native root population", cause)))
                        .transpose()?,
                    retained_sources,
                    native_recipe.as_ref(),
                    context
                        .and_then(WorkspaceContext::metadata_funding)
                        .as_ref(),
                )
                .map_err(|cause| at("prefill control facts", cause))
        })
        .transpose()?
        .flatten();
    let paged = paged_sources.map(|paged|paged.host_source_facts())
        .transpose().map_err(|cause|Error::Neural(report_metadata.source(cause)))?.flatten();
    let prefill = match prefill {
        Some(prefill)=>{
            let target=prefill.source_construction_facts();
            let compound=match native_recipe.as_ref(){
                Some(recipe)=>match report_metadata.funding(){Some(funding)=>recipe.prepare_addressable_source_program(target,paged,&funding)
                    .map_err(|cause| at("addressable source program", cause))?,
                    None=>{if recipe.records().iter().any(|row|row.addressable().is_some()){return Err(memory(WorkingMemoryError::IdentityMismatch));}None}},
                None=>None,
            };
            if compound.is_none() && target.is_some() && paged.is_some() {
                return Err(memory(WorkingMemoryError::UnknownBound));
            }
            let sources=compound.or(paged);
            if let Some(sources)=sources {
                let host=match prefill.host_destination_facts(){
                    Some(host)=>host,
                    None=>eredu_runtime::working_memory::HostDestinationFacts::new(0,0).map_err(memory)?,
                }.replace_source_constructions(target,sources).map_err(memory)?;
                Some(prefill.with_source_constructions(Some(sources)).with_host_destinations(Some(host)))
            }else{Some(prefill)}
        },None=>None,
    };
    if let Some(recipe)=native_recipe.as_ref(){
        let controls=recipe.addressable_request_control_bytes()
            .map_err(|cause| at("addressable request controls", cause))?;
        if let WorkspaceBound::Bounded{bytes,assumptions}=&mut outside.retained{
            *bytes=bytes.checked_add(controls).ok_or_else(||memory(WorkingMemoryError::Overflow))?;
            if controls!=0{report_metadata.append(assumptions,"; exact addressable request source program and independent native graph/record roles").map_err(report_error)?;}
        }
    }
    let prefill = match (prefill, native_recipe.as_ref().map(|recipe|
        recipe.initialized_input_source_facts(geometry.max_output_tokens)
            .map_err(|cause| at("initialized parallel input sources", cause))).transpose()?.flatten()) {
        (Some(prefill), Some(facts)) => Some(prefill.with_output_source_constructions(facts)),
        (prefill, _) => prefill,
    };
    let contribution = ControllerWorkspaceContribution::new_metadata(
        geometry,
        controller,
        sampling.output_width,
        storage_contract,
        &session.payload.memory_pool,
        registered_controller,
        report_metadata,
    )
    .and_then(|contribution| contribution.compose_metadata(outside, report_metadata))
    .map_err(|error| Error::Neural(error.into_workspace(report_metadata)))?;
    let quote = match storage {
        ObservedStorage::CompletedInput(storage) => storage.compose_with_controller_metadata(
            equations,
            state,
            contribution,
            report_metadata,
        ),
        ObservedStorage::Registered(storage) => {
            eredu_runtime::working_memory::ResidualInferenceQuote::compose_with_controller_metadata(
                equations,
                state,
                contribution,
                storage,
                report_metadata,
            )
            .map(|quote| quote.into_incremental())
        }
        ObservedStorage::Unregistered => IncrementalInferenceQuote::compose_controller_metadata(
            equations,
            state,
            contribution,
            report_metadata,
        ),
    };
    let quote = match quote {
        Ok(quote) => {
            components.before_seal = Some(quote.incremental_bytes());
            // This is request quotation after native model selection, not
            // portable artifact inspection. Only payload-free selected facts
            // enter the neutral plan. Missing provider decomposition stays
            // UnknownBound; Graph/Record capacities supply no payload amount.
            let native = if graph.is_some() && tracking.is_some() {
                let collector = native_program
                    .as_ref()
                    .map(|program| {
                        super::super::text_funding::native_collector_control_bytes(
                            program.attempts(),
                            program.rows(),
                            program.works(),
                        )
                    })
                    .transpose()?
                    .flatten();
                components.collector_controls = collector;
                native_mechanism
                    .as_ref()
                    .map(|mechanism| {
                        components.direct_controls = native_program.as_ref()
                            .map(|program| program.direct_control_bytes(mechanism, report_metadata))
                            .transpose()?.flatten();
                        mechanism
                            .selected_plan(
                                quote.span_workspace(),
                                native_recipe.as_ref(),
                                native_program.as_ref(),
                                collector,
                            )
                            .map_err(|cause| at("selected native plan", cause))
                    })
                    .transpose()?
            } else {
                None
            };
            components.native_controls = native.as_ref().and_then(|plan| plan.control_bytes());
            Ok(match capture {
                Some(capture) => capture.seal_controls_with_sequence(
                    quote,
                    sequence_claim,
                    tracking,
                    graph,
                    prefill,
                    native,
                )?,
                None => {
                    if sequence_claim.is_some()
                        || tracking.is_some()
                        || graph.is_some()
                        || original_table
                    {
                        sequence::seal(quote, sequence_claim, tracking, graph, prefill, native)
                            .map_err(|cause| at("sequence control seal", cause))?
                    } else {
                        quote
                    }
                }
            })
        }
        Err(eredu_runtime::working_memory::WorkspaceCopyCompositionError::Quote(
            eredu_runtime::working_memory::ResidualQuoteError::IncompleteWorkspace(incomplete),
        )) => Err(incomplete),
        Err(error) => return Err(Error::Neural(error.into_workspace(report_metadata))),
    };
    if let Ok(quote) = &quote {
        components.after_seal = Some(quote.incremental_bytes());
        components.state = Some(quote.state().requested_state_bytes);
        if let Some(workspace) = &quote.state().execution_workspace {
            components.activations = workspace.activations.bytes();
            components.vocabulary = workspace.vocabulary.bytes();
            components.retained = workspace.retained.bytes();
        }
        if let Some(recipe) = native_recipe {
            recipe.record_quote_components(components)?;
        }
    }
    Ok(quote)
}

/// Completed B enters the same native quote/seal assembler as token input.
/// The actual source and registered decoder binding remain borrowed together.
pub(in crate::composition::mlx::session::model_session) fn quote_completed_input(
    session: &MlxModelSession,
    geometry: InferenceGeometry,
    config: TextGenerationConfig,
    controller: TextControllerWorkspace<'_>,
    storage_contract: &ControllerStorageContract,
    equations: &InferenceWorkspaceReport,
    sampling: &SamplingWorkspaceReport,
    storage: &RegisteredPreparedWorkspaceStorage<StorageIdentity>,
    state_input: InputTokenCount,
    native_recipe: &mut Option<ResidentNativeRecipe>,
    capture: Option<&CaptureAdmission<'_>>,
    claim: &eredu_core::GenerationSequencePreparation<'_, '_>,
    original_table: bool,
    retained_sources: Option<&crate::backend::runtime::execution::generic::LayerwiseWorkspace>,
    context: &WorkspaceContext,
    paged_sources: Option<&crate::backend::nn::workspace::ProjectedPagedSources>,
) -> Result<Result<IncrementalInferenceQuote, IncompleteWorkspace>, Error> {
    quote_observed_with_sequence(
        session,
        geometry,
        0,
        config,
        controller,
        storage_contract,
        None,
        ObservedWorkspace {
            equations,
            sampling,
            storage: ObservedStorage::CompletedInput(storage),
            state_input,
        },
        native_recipe,
        capture,
        Some(claim),
        original_table,
        true,
        retained_sources,
        Some(context),
        paged_sources,
    )
}
