//! Fresh native populations on the source-bound saved-copy quote.
use super::*;
use crate::backend::nn::workspace::ResidentNativeRecipe;
use crate::backend::runtime::residency::storage::StorageIdentity;
use eredu_architectures::prepared_execution::PreparedTextGenerationWorkspace;
use eredu_runtime::working_memory::{CopyPreparationInferenceQuote, RegisteredWorkspaceCopy};

pub(in crate::composition::mlx::session) fn seal_saved_native_quote(
    runtime: &ModelRuntime<MlxBackend<'_>>,
    config: TextGenerationConfig,
    recipe: &ResidentNativeRecipe,
    generation: &PreparedTextGenerationWorkspace,
    layerwise: Option<&crate::backend::runtime::execution::generic::LayerwiseWorkspace>,
    source: RegisteredWorkspaceCopy<StorageIdentity>,
    pending_copy: &eredu_nn::workspace::WorkspaceTraceReport,
    prepared_input: Option<
        &eredu_runtime::working_memory::RegisteredPreparedWorkspaceStorage<StorageIdentity>,
    >,
    state: eredu_core::RuntimeStateEstimate,
    mut outside: eredu_core::ExecutionWorkspaceEstimate,
    opening_rows: Option<usize>,
    paged_sources: bool,
    paged_host_facts: Option<eredu_runtime::working_memory::HostSourceConstructionFacts>,
    capture: Option<(
        &eredu_runtime::capture::FundedCaptureCheckpoint,
        &eredu_runtime::layered::PreparedCaptureSelection,
        &eredu_runtime::working_memory::RegisteredInferenceSourceWitness,
    )>,
    metadata: eredu_runtime::working_memory::WorkspaceReportMetadata<'_>,
) -> Result<CopyPreparationInferenceQuote<StorageIdentity>, Error> {
    let session = runtime.session();
    let geometry = generation.equations.geometry();
    // Authenticate the already-initialized source/stream before selecting any
    // native preparation. No allocator, device or ordinary stream is created.
    runtime
        .backend()
        .original_copy_environment()
        .map_err(|cause| Error::Other(Box::new(cause)))?;
    let mechanism = session
        .payload
        .model
        .native_storage_mechanism()?
        .ok_or_else(|| unknown())?;
    if recipe.resume_copy().is_none() || (layerwise.is_some() && !recipe.has_host_copy_recipe()) {
        return Err(unknown());
    }
    let capture = capture
        .map(|(checkpoint, selection, witness)| {
            capture::CaptureAdmission::from_checkpoint(
                session, geometry, checkpoint, selection, witness, metadata,
            )
        })
        .transpose()?;
    if let Some(capture) = &capture {
        capture.add_enclosing_metadata(&mut outside, metadata)?;
    }
    let policy = config.inference_policy();
    let records = recipe
        .record_storage_requirement()?
        .select_for_policy(policy.submission_tracking_capacity_bytes)
        .map_err(memory)?;
    let graph_fit = recipe.resume_graph_storage_requirement()?;
    let graphs = graph_fit
        .select_for_policy(policy.graph_metadata_capacity_bytes)
        .map_err(memory)?;
    let graph_controls = graph_fit.control_bytes().ok_or_else(|| unknown())?;
    let pipeline_controls = pipeline_cache::control_bytes(recipe)?.ok_or_else(|| unknown())?;
    let copy_controls = u64::try_from(
        crate::backend::array_copy::OriginalCopyLayoutBuilder::resume_control_bytes()
            .ok_or_else(|| unknown())?,
    )
    .map_err(|_| memory(WorkingMemoryError::Overflow))?;
    let prefill = session
        .payload
        .model
        .erased()
        .prefill_control_facts(
            &session.payload.memory_pool,
            geometry,
            graphs,
            Some(recipe.maximum_roots()?),
            layerwise,
            Some(recipe),
            metadata.funding().as_ref(),
        )?
        .ok_or_else(|| unknown())?;
    let target=prefill.source_construction_facts();
    let compound=recipe.prepare_addressable_source_program(target,paged_host_facts,&metadata.funding().ok_or_else(||memory(WorkingMemoryError::IdentityMismatch))?)?;
    let prefill=if let Some(sources)=compound.or(paged_host_facts){
        let destinations=match prefill.host_destination_facts(){
            Some(host)=>host,
            None=>eredu_runtime::working_memory::HostDestinationFacts::new(0,0).map_err(memory)?,
        }.replace_source_constructions(target,sources).map_err(memory)?;
        prefill.with_source_constructions(Some(sources)).with_host_destinations(Some(destinations))
    }else{prefill};
    let prefill = match recipe.initialized_input_source_facts(geometry.max_output_tokens)? {
        Some(facts) => prefill.with_output_source_constructions(facts),
        None => prefill,
    };
    let addressable_controls=recipe.addressable_request_control_bytes()?;
    let recipe_controls = recipe
        .control_bytes()?
        .checked_add(
            crate::backend::nn::tensor::token_validation_control_bytes(recipe)?
                .ok_or_else(|| unknown())?,
        )
        .and_then(|n| n.checked_add(pipeline_controls))
        .and_then(|n| n.checked_add(addressable_controls))
        .and_then(|n| n.checked_add(copy_controls))
        .ok_or_else(|| memory(WorkingMemoryError::Overflow))?;
    let eredu_core::WorkspaceBound::Bounded { bytes, assumptions } = &mut outside.retained else {
        return Err(unknown());
    };
    *bytes = bytes
        .checked_add(recipe_controls)
        .ok_or_else(|| memory(WorkingMemoryError::Overflow))?;
    metadata
        .append(
            assumptions,
            "; actual resumed native recipe, token-validation and finite pipeline producers",
        )
        .map_err(|cause| Error::Neural(metadata.error(cause)))?;
    let program = mechanism
        .resume_program(
            recipe,
            &source,
            pending_copy,
            &generation.sampling,
            opening_rows,
            paged_sources,
        )?
        .ok_or_else(|| unknown())?;
    program.replace_enclosing_metadata(&mut outside, metadata)?;
    let quote = match prepared_input {
        Some(prepared) => source.compose_inference_with_prepared_source_metadata(
            &generation.equations,
            state,
            outside,
            prepared,
            metadata,
        ),
        None => source.compose_inference_metadata(&generation.equations, state, outside, metadata),
    }
    .map_err(|cause| Error::Neural(cause.into_workspace(metadata)))?;
    let tracking =
        tracking::selected_facts(policy.submission_tracking_capacity_bytes, Some(records))?;
    let graph = graph::selected_facts(
        policy.graph_metadata_capacity_bytes,
        Some(graphs),
        graph_controls,
    )?;
    let collector = super::super::super::text_funding::native_collector_control_bytes(
        program.attempts(),
        program.rows(),
        program.works(),
    )?
    .ok_or_else(|| unknown())?;
    let native = mechanism.selected_plan(
        quote.span_workspace(),
        Some(recipe),
        Some(&program),
        Some(collector),
    )?;
    match capture {
        Some(capture) => capture.seal_saved(
            quote,
            tracking.ok_or_else(|| unknown())?,
            graph.ok_or_else(|| unknown())?,
            prefill,
            native,
        ),
        None => sequence::seal_saved(quote, None, tracking, graph, Some(prefill), Some(native)),
    }
}
