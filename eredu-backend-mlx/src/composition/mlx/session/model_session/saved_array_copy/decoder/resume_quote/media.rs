//! The captured media variant through the same equation and saved-sampling worker.
use super::*;
use crate::backend::nn::workspace::ResidentNativeRecipe;
use crate::composition::mlx::model::retain_planning_error;
use crate::composition::mlx::session::intervention::TextInterventionQuote;
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
    parameters: &crate::backend::nn::workspace::ParameterWorkspaceBackings,
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
        let pool = runtime.backend().memory_ledger();
        let input = packet
            .project_workspace_with_semantics(media.semantics(), context, pool)
            .map_err(|cause| retain_planning_error(cause, funding.clone()))?;
        // Only completed B is an existing equation source. Copied decoder/key
        // destinations belong to the new request and receive no B root credit.
        let completed = parameters
            .completed_source(context)
            .map_err(Error::Neural)?;
        let layout = match &completed {
            Some(source) => RegisteredWorkspaceStorageLayout::<StorageIdentity>::new_with_prepared_and_completed_sources(parameters.len(), input.source_storage(), source),
            None => RegisteredWorkspaceStorageLayout::<StorageIdentity>::new_with_prepared_source(parameters.len(), input.source_storage()),
        }
        .map_err(|cause| retain_planning_error(cause, funding.clone()))?;
        context
            .charge_metadata(layout.requested_bytes())
            .map_err(|cause| retain_planning_error(cause, funding.clone()))?;
        let storage = match completed {
            Some(source) => layout.construct_with_prepared_and_completed_sources(
                pool,
                context,
                parameters.roots(),
                input.source_storage().clone(),
                source,
            ),
            None => layout.construct_with_prepared_source(
                pool,
                context,
                parameters.roots(),
                input.source_storage().clone(),
            ),
        }
        .map_err(|cause| retain_planning_error(cause, funding.clone()))?;
        Ok(PreparedSavedMedia { input, storage })
    })();
    result.map_err(|cause| retain_planning_error(cause, funding))
}

pub(super) fn quote(
    runtime: &ModelRuntime<MlxBackend<'_>>,
    source: &child::Source<'_, '_>,
    geometry: InferenceGeometry,
    state: &DeviceState<WorkspaceBackend, WorkspaceResidentLayerState>,
    context: &WorkspaceContext,
    sampling: BorrowedTextSamplingWorkspace<'_>,
    prepared: PreparedSavedMedia,
    addressable: Option<&crate::backend::nn::workspace::AddressableSources>,
    interventions: Option<TextInterventionQuote<'_>>,
    parallel: Option<&crate::backend::nn::workspace::MlxParallelWorkspace>,
    parameters: crate::composition::mlx::model::PreparedSavedParameterSource,
) -> Result<SavedMediaQuote, Error> {
    let funding = context
        .metadata_funding()
        .ok_or(Error::PrefillControl(WorkingMemoryError::UnknownBound))?;
    let result: Result<SavedMediaQuote, Error> = (|| {
        context
            .charge_metadata(std::mem::size_of::<(
                SavedMediaQuote,
                Result<SavedMediaQuote, Error>,
                Option<TextInterventionQuote<'_>>,
                Option<&super::super::super::pending_input::SavedPendingMedia>,
                PreparedSavedMedia,
                Option<(
                    &eredu_runtime::capture::FundedCaptureCheckpoint,
                    &eredu_runtime::layered::PreparedCaptureSelection,
                )>,
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
        let pool = runtime.backend().memory_ledger();
        let blueprint = model
            .inference_blueprint()
            .ok_or(Error::PrefillControl(WorkingMemoryError::UnknownBound))?;
        let mechanisms = model
            .resident_workspace_mechanisms()
            .ok_or(Error::PrefillControl(WorkingMemoryError::UnknownBound))?;
        let layerwise = parameters.layerwise;
        // Native numerical tracing is independent of the selected source-copy
        // mechanism. Its authoritative constructor/copy recipe is bound below.
        let capture = match (source.capture_checkpoint(), source.capture_selection()) {
            (Some(checkpoint), Some(selection)) => Some((checkpoint, selection)),
            (None, None) => None,
            _ => return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch)),
        };
        if capture.is_none() && interventions.is_some() {
            return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
        }
        let loaded_capture = if capture.is_some() && parallel.is_some() {
            Some(
                runtime
                    .session()
                    .partition_capture_source()
                    .ok_or_else(unknown)?,
            )
        } else {
            None
        };
        let parameter_source = layerwise
            .as_ref()
            .map(crate::composition::mlx::model::NativeLayerwiseParameters);
        let quote =
            |recorder: &mut dyn crate::composition::mlx::model::CaptureRecorder,
             communication: Option<&eredu_runtime::RetainedCommunicationSource>| {
                if let Some((checkpoint, selection)) = capture {
                    let parallel = communication.map(|communication| {
                        let loaded = loaded_capture
                            .as_ref()
                            .expect("prepared partition capture source");
                        (
                            communication,
                            (loaded.layouts(), communication.manifest().rank()),
                            None,
                        )
                    });
                    model
                        .quote_saved_original_media_capture(
                            input,
                            media.semantics().binding(),
                            geometry,
                            state,
                            context,
                            sampling,
                            checkpoint,
                            selection,
                            interventions,
                            recorder,
                            parallel,
                            parameter_source.as_ref().map(|source| {
                                source as
                        &dyn eredu_architectures::prepared_execution::WorkspaceLayerwiseParameters
                            }),
                        )
                        .map_err(|cause| retain_planning_error(cause, funding.clone()))
                } else {
                    blueprint
                        .quote_original_media_with_existing_sampling_and_trace(
                            input,
                            media.semantics().binding(),
                            geometry,
                            state,
                            context,
                            parameter_source.as_ref().map(|source| {
                                source as
                        &dyn eredu_architectures::prepared_execution::WorkspaceLayerwiseParameters
                            }),
                            sampling,
                            recorder,
                            communication,
                        )
                        .map_err(|cause| {
                            retain_planning_error(cause.into_failure(), funding.clone())
                        })
                }
            };
        context
            .charge_metadata(std::mem::size_of_val(&quote))
            .map_err(|cause| retain_planning_error(cause, funding.clone()))?;
        let (report, mut recipe) = if let Some(parallel) = parallel {
            let mut recorder = parallel
                .recorder(geometry)
                .map_err(|cause| retain_planning_error(cause, funding.clone()))?;
            if let Some(source) = layerwise.as_ref() {
                recorder
                    .bind_layerwise_span_constructor_source(source)
                    .map_err(|cause| retain_planning_error(cause, funding.clone()))?;
            }
            let report = quote(&mut recorder, Some(parallel.declaration_source()))?;
            let recipe = recorder
                .finish(report.equations().span_workspace_plan())
                .map_err(|cause| retain_planning_error(cause, funding.clone()))?;
            (report, recipe)
        } else {
            let mut recorder = mechanisms
                .recorder(geometry, context)
                .map_err(|cause| retain_planning_error(cause, funding.clone()))?;
            if let Some(source) = layerwise.as_ref() {
                recorder
                    .bind_layerwise_span_constructor_source(source)
                    .map_err(|cause| retain_planning_error(cause, funding.clone()))?;
            }
            if let Some(source) = addressable {
                recorder
                    .bind_addressable_sources(source.clone())
                    .map_err(|cause| retain_planning_error(cause, funding.clone()))?;
            }
            let report = quote(&mut recorder, None)?;
            let recipe = recorder
                .finish(report.equations().span_workspace_plan())
                .map_err(|cause| retain_planning_error(cause, funding.clone()))?;
            (report, recipe)
        };
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
