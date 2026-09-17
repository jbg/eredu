//! The captured media variant through the same equation and saved-sampling worker.
use super::*;
use crate::composition::mlx::session::intervention::TextInterventionQuote;
use crate::backend::nn::workspace::{ResidentNativeRecipe};
use crate::composition::mlx::model::retain_planning_error;
use eredu_runtime::working_memory::{
    RegisteredPreparedWorkspaceStorage, RegisteredWorkspaceStorageLayout,
};

pub(super) struct SavedMediaQuote {
    pub(super) generation: PreparedTextGenerationWorkspace,
    pub(super) recipe: ResidentNativeRecipe,
    pub(super) layerwise: Option<crate::backend::runtime::execution::generic::LayerwiseWorkspace>,
    pub(super) storage: RegisteredPreparedWorkspaceStorage<StorageIdentity>,
}

pub(super) struct PreparedSavedMedia {
    input: eredu_architectures::prepared_execution::OriginalMediaWorkspaceInput,
    storage: RegisteredPreparedWorkspaceStorage<StorageIdentity>,
}

/// One source association, installed before the first span on this exact ledger.
pub(super) fn prepare(
    runtime: &ModelRuntime<MlxBackend<'_>>,
    source: &CopiedTextComponents,
    geometry: InferenceGeometry,
    context: &WorkspaceContext,
) -> Result<PreparedSavedMedia, Error> {
    let funding = context
        .metadata_funding()
        .ok_or(Error::PrefillControl(WorkingMemoryError::UnknownBound))?;
    let result = (|| {
        context
            .charge_metadata(std::mem::size_of::<(
                PreparedSavedMedia,
                Option<PreparedSavedMedia>,
                Result<PreparedSavedMedia, Error>,
                Option<&super::super::super::pending_input::SavedPendingMedia>,
                RegisteredWorkspaceStorageLayout<StorageIdentity>,
                std::iter::Empty<(
                    StorageIdentity,
                    eredu_nn::workspace::WorkspaceExistingStorage,
                )>,
            )>())
            .map_err(|cause| retain_planning_error(cause, funding.clone()))?;
        let media = source
            .sampling
            .pending_media()
            .ok_or(Error::PrefillControl(WorkingMemoryError::IdentityMismatch))?;
        let packet = media.packet();
        let captured = &source.sampling.arrays.source;
        if !media.semantics().binding().matches_snapshot(
            &captured.execution,
            &captured.revision,
            captured.frontier,
        ) || captured.frontier != geometry.cached_positions
            || packet.shape() != [geometry.batch_size, geometry.input_positions]
            || source.sampling.arrays.pending.is_some()
        {
            return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
        }
        // The fixed origin/epoch check binds this immutable paired source to
        // this exact executable. It does not require the live frontier to equal
        // the saved one and never projects the current decoder.
        source
            .validate_resume_origin_fixed(runtime)
            .map_err(|cause| retain_planning_error(cause, funding.clone()))?;
        let pool = runtime.backend().memory_pool();
        let input = packet
            .project_workspace_with_semantics(media.semantics(), context, pool)
            .map_err(|cause| retain_planning_error(cause, funding.clone()))?;
        // Only completed B is an existing equation source. Copied decoder/key
        // destinations belong to the new request and receive no B root credit.
        let layout = RegisteredWorkspaceStorageLayout::<StorageIdentity>::new_with_prepared_source(
            0,
            input.source_storage(),
        )
        .map_err(|cause| retain_planning_error(cause, funding.clone()))?;
        context
            .charge_metadata(layout.requested_bytes())
            .map_err(|cause| retain_planning_error(cause, funding.clone()))?;
        let storage = layout
            .construct_with_prepared_source(
                pool,
                context,
                std::iter::empty(),
                input.source_storage().clone(),
            )
            .map_err(|cause| retain_planning_error(cause, funding.clone()))?;
        Ok(PreparedSavedMedia { input, storage })
    })();
    result.map_err(|cause| retain_planning_error(cause, funding))
}

pub(super) fn quote(
    runtime: &ModelRuntime<MlxBackend<'_>>,
    source: &CopiedTextComponents,
    geometry: InferenceGeometry,
    state: &DeviceState<WorkspaceBackend, WorkspaceResidentLayerState>,
    context: &WorkspaceContext,
    sampling: BorrowedTextSamplingWorkspace<'_>,
    prepared: PreparedSavedMedia,
    addressable:Option<&crate::backend::nn::workspace::AddressableSources>,
    interventions: Option<TextInterventionQuote<'_>>,
) -> Result<SavedMediaQuote, Error> {
    let funding = context
        .metadata_funding()
        .ok_or(Error::PrefillControl(WorkingMemoryError::UnknownBound))?;
    let result: Result<SavedMediaQuote, Error> = (|| {
        context
            .charge_metadata(std::mem::size_of::<(
                SavedMediaQuote,
                Result<SavedMediaQuote, Error>, Option<TextInterventionQuote<'_>>,
                Option<&super::super::super::pending_input::SavedPendingMedia>,
                PreparedSavedMedia,
                Option<(&eredu_runtime::capture::FundedCaptureCheckpoint,
                    &eredu_runtime::layered::PreparedCaptureSelection)>,
            )>())
            .map_err(|cause| retain_planning_error(cause, funding.clone()))?;
        let media = source
            .sampling
            .pending_media()
            .ok_or(Error::PrefillControl(WorkingMemoryError::IdentityMismatch))?;
        source
            .validate_resume_origin_fixed(runtime)
            .map_err(|cause| retain_planning_error(cause, funding.clone()))?;
        let PreparedSavedMedia { input, storage } = prepared;
        let model = &runtime.session().payload.model;
        let pool = runtime.backend().memory_pool();
        let blueprint = model
            .inference_blueprint()
            .ok_or(Error::PrefillControl(WorkingMemoryError::UnknownBound))?;
        let mechanisms = model
            .resident_workspace_mechanisms()
            .ok_or(Error::PrefillControl(WorkingMemoryError::UnknownBound))?;
        model
            .erased()
            .install_workspace_parameter_representations(context)?;
        let layerwise = model.layerwise_workspace()?;
        // Native numerical tracing is independent of the selected source-copy
        // mechanism. Its authoritative constructor/copy recipe is bound below.
        let mut recorder = mechanisms.recorder(geometry, context)
            .map_err(|cause| retain_planning_error(cause, funding.clone()))?;
        if let Some(source)=addressable{recorder.bind_addressable_sources(source.clone()).map_err(|cause|retain_planning_error(cause,funding.clone()))?;}
        let capture = match (source.capture_checkpoint(), source.capture_selection()) {
            (Some(checkpoint), Some(selection)) => Some((checkpoint, selection)),
            (None, None) => None,
            _ => return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch)),
        };
        if capture.is_none() && interventions.is_some(){return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));}
        let report = if let Some((checkpoint, selection)) = capture {
            model.quote_saved_original_media_capture(
                input, media.semantics().binding(), geometry, state, context,
                sampling, checkpoint, selection, interventions, &mut recorder,
            ).map_err(|cause| retain_planning_error(cause, funding.clone()))?
        } else {
            blueprint.quote_original_media_with_existing_sampling_and_trace(
                input,
                media.semantics().binding(),
                geometry,
                state,
                context,
                None,
                sampling,
                &mut recorder,
            )
            .map_err(|cause| retain_planning_error(cause.into_failure(), funding.clone()))?
        };
        let mut recipe = recorder
            .finish(report.equations().span_workspace_plan())
            .map_err(|cause| retain_planning_error(cause, funding.clone()))?;
        model
            .erased()
            .bind_layerwise_neural_recipe(pool, &mut recipe, Some(&funding))?;
        if let Some(sources) = &layerwise {
            sources.bind_native_host_copies(&mut recipe)?;
        }
        let generation = report.into_generation(context).map_err(Error::Neural)?;
        source
            .validate_resume_origin_fixed(runtime)
            .map_err(|cause| retain_planning_error(cause, funding.clone()))?;
        Ok(SavedMediaQuote {
            generation,
            recipe,
            layerwise,
            storage,
        })
    })();
    result.map_err(|cause| retain_planning_error(cause, funding))
}
