//! Sequential saved-source preparation, shared model trace, then quote assembly.
//! Large dormant return/error frames are never live below model reconstruction.
use super::*;
use crate::backend::nn::workspace::{ProjectedNativeStorage,ResidentNativeRecipe};
use crate::backend::runtime::execution::generic::LayerwiseWorkspace;
use eredu_runtime::working_memory::{RegisteredWorkspaceCopy,RegisteredPreparedWorkspaceStorage};
use std::mem::{size_of,size_of_val};
use crate::composition::mlx::session::intervention::{PreparedTextInterventions, TextInterventionQuote};
use std::cell::RefCell;

pub(super) struct Inputs {
    original:bool, geometry:InferenceGeometry, source_layout:eredu_runtime::SharedStateLayout,
    decoder_host_peak:u64, sampler_host_peak:u64, pending_host_bytes:Option<u64>,
    prepared_media:Option<media::PreparedSavedMedia>, resume_copy:Option<crate::backend::array_copy::OriginalResumeCopyPopulation>,
    state:DeviceState<WorkspaceBackend,WorkspaceResidentLayerState>, source_native:ProjectedNativeStorage,
    decoder_copy:WorkspaceTraceReport, sampling_copy:WorkspaceTraceReport,
    copied_key:Option<WorkspaceTensor>, copied_input:Option<WorkspaceTensor>, random:Option<WorkspaceSamplingRandomState>,
    copy_preparation:RegisteredWorkspaceCopy<StorageIdentity>, registered_source:WorkingMemoryStorage<StorageIdentity>,
    workspace:parallel::SavedWorkspace,
    prepared_parameters:Option<crate::composition::mlx::model::PreparedSavedParameterSource>,
}
pub(super) struct Traced {
    generation:PreparedTextGenerationWorkspace, layerwise:Option<LayerwiseWorkspace>,
    native_recipe:Option<ResidentNativeRecipe>, media_storage:Option<RegisteredPreparedWorkspaceStorage<StorageIdentity>>,
    inputs:Inputs,
    text_interventions:Option<PreparedTextInterventions>,
}
#[inline(never)]
pub(super) fn prepare(runtime:&ModelRuntime<MlxBackend<'_>>,source:&child::Source<'_, '_>,
    config:TextGenerationConfig,controller:TextControllerWorkspace<'_>,host:Option<&eredu_core::HostPreparationAuthority>,
    planning_metadata:Option<&HostMetadataFunding>)->Result<Inputs,Error> {
        let original = host.is_some();
        let memory = |cause: WorkingMemoryError| planned_error(cause, planning_metadata);
        let unknown = || memory(WorkingMemoryError::UnknownBound);
        if original {
            source
                .validate_resume_origin_fixed(runtime)
                .map_err(|cause| planned_error(cause, planning_metadata))?;
        } else {
            source.validate_resume_origin(runtime)?;
        }
        let session = runtime.session();
        let model = &session.payload.model;
        let pool = runtime.backend().memory_pool();
        let blueprint = model.inference_blueprint().ok_or_else(unknown)?;
        let selected = blueprint.selected();
        if !session.payload.memory_pool.same_domain(pool)
            || !model.has_published_idle_storage()
            || session.payload._memory_owner.is_some()
            || session.payload.parameter_state.active.is_some()
            || selected.prediction_extension().is_some()
            || model.erased().has_embedded_prediction()
            || !matches!(
                selected.text_realization().residency(),
                LayerWeightResidency::FullyResident
                    | LayerWeightResidency::LayerwiseHost(_)
                    | LayerWeightResidency::DenseDiskStream(_)
            )
            || !crate::backend::runtime::cache::kv::PagedKeyValueCache::original_residency_supported(
                selected.text_realization().state().policy(),
                model
                    .erased()
                    .capability_estimate()
                    .state_layout()
                    .layer_layout(),
                session.floating_state_dtype_bytes,
            )
        {
            return Err(unknown());
        }
        match (&session.payload.distributed, session.payload.target.has_retained_world(),
            selected.communication_manifest()) {
            (None, false, None) => {},
            (Some(_), true, Some(_)) if original => {},
            _ => return Err(unknown()),
        }
        if !selected.text_realization().exact_completion_available() {
            return Err(memory(WorkingMemoryError::CompletionUnavailable));
        }
        // Current branch storage may differ from the saved branch. It only
        // establishes complete existing occupancy, never future cache geometry.
        if original {
            let mut nonstate = RetainedStorage::original_census(pool);
            let mut decoder = RetainedStorage::original_census(pool);
            session
                .payload
                .collect_retained_idle_storage(&mut nonstate, &mut decoder)?;
            if nonstate.original_publication_rows(pool)?.is_none()
                || decoder.original_publication_rows(pool)?.is_none()
            {
                return Err(unknown());
            }
        } else {
            let idle = session.payload.retained_idle_storage()?;
            if idle.nonstate_bytes()?.is_none() || idle.decoder_state_bytes()?.is_none() {
                return Err(unknown());
            }
        }
        let geometry = resume_geometry_impl(source, config, planning_metadata)?;
        let binding = TextArrayBinding::Saved(&source.sampling);
        let host_sampler = binding
            .borrow_funded()?
            .prepare_resume(config)
            .map_err(memory)?;
        if source.sampling.temperature != config.sampling().temperature
            || (source.sampling.temperature > 0.0 && binding.key().is_none())
        {
            return Err(memory(WorkingMemoryError::PreparationConfigurationMismatch));
        }
        let pending_plan = if geometry.max_output_tokens == 0 { None } else { source
            .sampling.prepare_pending_tokens().map_err(|cause| planned_error(cause, planning_metadata))? };
        if pending_plan.is_none() && !original {
            return Err(unknown());
        }
        let dense = match host {
            Some(authority) => source
                .decoder
                .native
                .prepare_copy_fixed()
                .and_then(|native| native.into_dense_fixed())
                .map_err(|cause| cold_source::retain_failure(cause.into(), authority))?,
            None => source.decoder.native.prepare_copy()?.into_dense()?,
        };
        let decoder_host_peak = match host {
            Some(authority) => dense
                .initialization_peak_bytes_fixed()
                .map_err(|cause| cold_source::retain_failure(cause.into(), authority))?,
            None => dense.initialization_peak_bytes(pool)?,
        };
        let sampler_host_peak = host_sampler.required_host_bytes();
        let (copy_preparation, registered_source) = match host {
            Some(authority) => {
                let owner = DecoderCopyOwner::Saved(source.decoder.clone());
                let prepared = source
                    .decoder
                    .native
                    .prepare_copy_fixed()
                    .map_err(|cause| cold_source::retain_failure(cause.into(), authority))?;
                let plan = preparation::ResumeSourcePreparation::inspect(
                    &owner,
                    &prepared,
                    binding.key(),
                    binding.pending(),
                    pool,
                    model.resident_workspace_mechanisms().ok_or_else(unknown)?,
                )
                .map_err(|cause| preparation::retain_failure(cause, authority))?;
                let prepared = plan.construct(authority)?;
                (prepared.copy_preparation, prepared.registered_source)
            }
            None => {
                let mut complete_source =
                    DecoderCopyOwner::Saved(source.decoder.clone()).complete_storage()?;
                for array in binding.key().into_iter().chain(binding.pending().filter(|_| geometry.max_output_tokens > 0)) {
                    complete_source.include_array(array)?;
                }
                let registered_source = complete_source.pin_registered(pool)?;
                // A distinct sealed copy ledger binds old decoder/key roots before
                // any trace. It consumes exactly those native operands, retaining every
                // duplicate as an independent destination. Pending reshape/cast remains
                // fully priced below until its complete program has a sealed proof.
                let copy_context =
                    WorkspaceContext::new(model.resident_workspace_mechanisms().ok_or_else(unknown)?);
                let mut input_count = Some(usize::from(binding.key().is_some()));
                dense
                    .visit_operands(&mut |_| {
                        input_count = input_count.and_then(|n| n.checked_add(1))
                    })
                    .map_err(
                        crate::backend::runtime::cache::state::SnapshotProjectionCause::into_error,
                    )?;
                let input_count =
                    input_count.ok_or_else(|| memory(WorkingMemoryError::Overflow))?;
                let mut native = None;
                let mut copy_projection =
                    crate::backend::nn::workspace::OwnedArrayProjection::prepare(
                        &mut native,
                        &copy_context,
                        input_count,
                    )
                    .map_err(other)?;
                let mut copy_inputs = Vec::new();
                copy_inputs.try_reserve_exact(input_count).map_err(other)?;
                let mut failure = None;
                let mut import = |array: &Array| {
                    if failure.is_none() {
                        match copy_projection.project_prepared(array) {
                            Ok(input) => copy_inputs.push(input),
                            Err(error) => failure = Some(error),
                        }
                    }
                };
                dense.visit_operands(&mut import).map_err(
                    crate::backend::runtime::cache::state::SnapshotProjectionCause::into_error,
                )?;
                for array in binding.key() {
                    import(array);
                }
                if let Some(error) = failure {
                    return Err(planned_error(error, planning_metadata));
                }
                drop(copy_projection);
                let native = native.expect("installed copy-source inventory");
                if !native.is_complete() {
                    return Err(unknown());
                }
                let copy_sources = RegisteredWorkspaceStorage::bind(
                    pool,
                    &copy_context,
                    native
                        .iter()
                        .map(|(id, _, root)| (StorageIdentity::Native(id), root.clone())),
                )
                .map_err(memory)?;
                let copy_program = WorkspaceIsolatedCopyPlan::prepare(
                    &copy_context,
                    copy_sources.borrowed_storage(),
                    &copy_inputs,
                )
                .map_err(|cause| planned_error(cause, planning_metadata))?;
                let copy_preparation = RegisteredWorkspaceCopy::bind(copy_program, copy_sources)
                    .map_err(|cause| planned_error(cause, planning_metadata))?;
                (copy_preparation, registered_source)
            }
        };
        let mechanisms = model.resident_workspace_mechanisms().ok_or_else(unknown)?;
        let workspace = parallel::SavedWorkspace::prepare(runtime, mechanisms, planning_metadata)?;
        let context = workspace.context();
        // Unobserved serial resume traces resident-shaped parameter equations.
        // Bind its exact unit source while this Context is still untouched,
        // before any saved-state copy or begin_state_span makes extension late.
        let prepared_parameters = if original && geometry.max_output_tokens > 0 && pending_plan.is_some()
            && workspace.parallel().is_none() && source.capture_checkpoint().is_none()
        {
            Some(model.prepare_saved_parameter_source(context)?)
        } else {
            None
        };
        // Install the exact immutable B root selection before the first copy
        // or equation span on this Context. It never credits copied state/key.
        let prepared_media = if pending_plan.is_none() && geometry.max_output_tokens > 0 {
            Some(media::prepare(runtime, source, geometry, &context)?)
        } else {
            if original {
                // Copied token/state/key destinations are fully new equation
                // storage. An explicit empty selection proves that no root is
                // excluded; absence would mean the prepared source is missing.
                // Install before the first copy span, just as completed media
                // installs its actual immutable B roots above.
                use eredu_nn::workspace::{WorkspaceBorrowedStorage, WorkspaceExistingStorage};
                let bytes = WorkspaceBorrowedStorage::construction_bytes(0)
                    .and_then(|bytes| bytes.checked_add(std::mem::size_of::<
                        std::iter::Empty<&WorkspaceExistingStorage>,
                    >()))
                    .ok_or_else(|| memory(WorkingMemoryError::Overflow))?;
                context.charge_metadata(bytes)
                    .map_err(|cause| planned_error(cause, planning_metadata))?;
                let selection = WorkspaceBorrowedStorage::new_finite(
                    context, std::iter::empty::<&WorkspaceExistingStorage>(), 0,
                ).map_err(|cause| planned_error(cause, planning_metadata))?;
                context.set_borrowed_storage_checked(selection)
                    .map_err(|cause| planned_error(cause, planning_metadata))?;
            }
            None
        };
        let projected = dense
            .project_dense_workspace(NonZeroU32::new(1).unwrap(), &context)
            .map_err(|cause| planned_error(cause, planning_metadata))?;
        if !projected.source_storage.is_complete() {
            return Err(unknown());
        }
        // These metadata destinations feed actual future equations. The key's
        // allocation demand belongs to the sealed copy contribution above;
        // the pending token's complete isolation/reshape/cast is still full.
        let mut projection = if original {
            ExistingArrayProjection::with_source_count(
                &context,
                usize::from(pending_plan.is_some()) + usize::from(binding.key().is_some()),
            )
            .map_err(|cause| planned_error(cause, planning_metadata))?
        } else {
            ExistingArrayProjection::new(&context)
        };
        let key_root = binding
            .key()
            .map(|array| projection.project(array))
            .transpose()
            .map_err(|cause| planned_error(cause, planning_metadata))?;
        context
            .begin_state_span(key_root.as_slice())
            .map_err(|cause| planned_error(cause, planning_metadata))?;
        let copied_key = binding
            .key()
            .map(|array| IsolatedArrayCopy::new(array).trace(&mut projection))
            .transpose()
            .map_err(|cause| planned_error(cause, planning_metadata))?;
        let (copied_input, sampling_copy) = match &pending_plan {
            Some(pending) => {
                let pending_root = projection
                    .project(pending.numerical().source())
                    .map_err(|cause| planned_error(cause, planning_metadata))?;
                context
                    .begin_state_span(std::slice::from_ref(&pending_root))
                    .map_err(|cause| planned_error(cause, planning_metadata))?;
                let copied = pending.trace(&mut projection)?;
                let report = context
                    .finish_report(&[pending_root, copied.clone()])
                    .map_err(|cause| planned_error(cause, planning_metadata))?;
                (Some(copied), report)
            }
            None => {
                // No token isolation, reshape or cast exists for completed B.
                // The separate sealed copy still owns the actual key/decoder.
                context
                    .begin_state_span(&[])
                    .map_err(|cause| planned_error(cause, planning_metadata))?;
                let report = context
                    .finish_report(&[])
                    .map_err(|cause| planned_error(cause, planning_metadata))?;
                (None, report)
            }
        };
        if !projection.is_complete() {
            return Err(unknown());
        }
        let random = copied_key
            .clone()
            .map(WorkspaceSamplingRandomState::from_key)
            .transpose()
            .map_err(|cause| planned_error(cause, planning_metadata))?;
        let resume_copy = if original {
            let mut census = crate::backend::array_copy::OriginalCopyLayoutBuilder::new();
            let mut failure = None;
            dense
                .visit_snapshot_operands(&mut |source| {
                    if let crate::backend::runtime::cache::state::SnapshotOperand::Array(array) =
                        source
                    {
                        if failure.is_none() {
                            failure = census.push_retained_source(array).err();
                        }
                    }
                    Ok(())
                })
                .map_err(|cause| planned_error(cause, planning_metadata))?;
            for array in binding.key().into_iter().chain(binding.pending().filter(|_| geometry.max_output_tokens > 0)) {
                if failure.is_none() {
                    failure = census.push_retained_source(array).err();
                }
            }
            if dense.is_paged() {
                dense
                    .visit_snapshot_operands(&mut |source| {
                        if let crate::backend::runtime::cache::state::SnapshotOperand::Array(
                            array,
                        ) = source
                        {
                            if failure.is_none() {
                                failure = census.push_operand(array).err();
                            }
                        }
                        Ok(())
                    })
                    .map_err(|cause| planned_error(cause, planning_metadata))?;
            } else {
                dense
                    .visit_operands(&mut |array| {
                        if failure.is_none() {
                            failure = census.push_operand(array).err();
                        }
                    })
                    .map_err(|cause| planned_error(cause, planning_metadata))?;
            }
            for array in binding.key() {
                if failure.is_none() {
                    failure = census.push_operand(array).err();
                }
            }
            if let Some(cause) = failure {
                return Err(planned_error(cause, planning_metadata));
            }
            Some(
                match &pending_plan {
                    Some(pending) => census.finish_resume_input_on(pending.numerical(),runtime.backend().stream()),
                    None => census.finish_resume_completed_input_on(
                        geometry.input_positions,
                        runtime.backend().stream(),
                    ),
                }
                .map_err(|cause| planned_error(cause, planning_metadata))?,
            )
        } else {
            None
        };
        let source_layout=dense.shared_layout().clone();
        let pending_host_bytes=pending_plan.as_ref().map(|pending|pending.host_plan().initialization_peak_bytes());
        drop(projection);
        Ok(Inputs { original,geometry,source_layout,decoder_host_peak,sampler_host_peak,pending_host_bytes,
            prepared_media,resume_copy,state:projected.state,source_native:projected.source_storage,decoder_copy:projected.copy,
            sampling_copy,copied_key,copied_input,random,copy_preparation,registered_source,workspace,prepared_parameters })
}
#[inline(never)]
pub(super) fn trace(runtime:&ModelRuntime<MlxBackend<'_>>,source:&child::Source<'_, '_>,
    controller:TextControllerWorkspace<'_>,mut inputs:Inputs,planning_metadata:Option<&HostMetadataFunding>)
    ->Result<Traced,Error> {
        let memory=|cause:WorkingMemoryError|planned_error(cause,planning_metadata);
        let unknown=||memory(WorkingMemoryError::UnknownBound);
        let model=&runtime.session().payload.model;let pool=runtime.backend().memory_pool();
        let binding=TextArrayBinding::Saved(&source.sampling);
        let original=inputs.original;let geometry=inputs.geometry;let context=inputs.workspace.context();
        context.charge_metadata(size_of::<(RefCell<Option<PreparedTextInterventions>>,
            Option<TextInterventionQuote<'_>>, Option<PreparedTextInterventions>)>())
            .map_err(|cause|planned_error(cause,planning_metadata))?;
        let intervention_rows=RefCell::new(None);
        let interventions=source.capture_checkpoint().and_then(|checkpoint|checkpoint.intervention_source())
            .map(|source|TextInterventionQuote{source,rows:&intervention_rows});
        if !original && interventions.is_some(){return Err(memory(WorkingMemoryError::IdentityMismatch));}
        crate::composition::mlx::session::model_session::text_quote::CaptureAdmission::validate_saved_intervention_source(
            runtime.session(), interventions.map(|quote|quote.source), WorkspaceReportMetadata::new(context))?;
        let sampling = BorrowedTextSamplingWorkspace::new(
            binding.sampler(),
            source.sampling.temperature,
            inputs.random.as_ref(),
            controller.filter,
        );
        // The completed copy reports remain owned above. Reconstructing the
        // model emits parameter placeholders before the shared quote driver
        // opens its exact inference state spans, so it needs its own open span.
        // This keeps the existing roots and cumulative metadata funding intact.
        context.begin_span();
        let (generation, layerwise, native_recipe, media_storage) = if original {
            let (generation, layerwise, mut recipe, media_storage) = if geometry.max_output_tokens == 0 {
                if let Some(edits) = interventions {
                    edits.prepare_range(source.sampling.next_prediction, 0, context)?;
                }
                let equations = eredu_runtime::working_memory::quote_inference_workspace_with_context(
                    geometry, context, |_| -> Result<WorkspaceTraceReport, Error> { Err(unknown()) })
                    .map_err(|cause| planned_error(cause, planning_metadata))?;
                let input = source.sampling.arrays.source.sampling_input.ok_or_else(unknown)?;
                let layout = input.layout(context)?;
                let quote_sampling = |recorder: &mut dyn eredu_runtime::working_memory::SamplingWorkspaceObserver| {
                    eredu_runtime::working_memory::quote_sampling_workspace_with_observer(
                        binding.sampler(), source.sampling.temperature, inputs.random.as_ref(), input.source(&layout)?,
                        controller.filter, 0, context, Some(recorder))
                };
                let (sampling, recipe) = if let Some(parallel) = inputs.workspace.parallel() {
                    let mut recorder = parallel.recorder(geometry)?;
                    let sampling = quote_sampling(&mut recorder)?;
                    (sampling, recorder.finish(equations.span_workspace_plan())?)
                } else {
                    let mechanisms = model.resident_workspace_mechanisms().ok_or_else(unknown)?;
                    let mut recorder = mechanisms.recorder(geometry, context)?;
                    let sampling = quote_sampling(&mut recorder)?;
                    (sampling, recorder.finish(equations.span_workspace_plan())?)
                };
                (PreparedTextGenerationWorkspace { equations, sampling }, None, recipe, None)
            } else if inputs.pending_host_bytes.is_none() {
                let media = media::quote(
                    runtime,
                    source,
                    geometry,
                    &inputs.state,
                    &context,
                    sampling,
                    inputs.prepared_media.take().ok_or_else(unknown)?,
                    inputs.workspace.addressable(),
                    interventions,
                    inputs.workspace.parallel(),
                )?;
                (
                    media.generation,
                    media.layerwise,
                    media.recipe,
                    Some(media.storage),
                )
            } else if let Some(parallel) = inputs.workspace.parallel() {
                let capture = match (source.capture_checkpoint(), source.capture_selection()) {
                    (Some(checkpoint), Some(selection)) => Some((checkpoint, selection)),
                    (None, None) => None,
                    _ => return Err(memory(WorkingMemoryError::IdentityMismatch)),
                };
                let (generation, layerwise, recipe) = parallel::quote(runtime, geometry, &inputs.state,
                    sampling, parallel, capture, interventions)?;
                (generation, layerwise, recipe, None)
            } else {
                let (generation, layerwise, recipe) = model
                    .quote_replicated_text_with_existing_sampling_recipe(
                        geometry,
                        &inputs.state,
                        &context,
                        sampling,
                        pool,
                        source.capture_checkpoint().zip(source.capture_selection()),
                        inputs.workspace.addressable(),
                        interventions,
                        inputs.prepared_parameters.take(),
                    )?;
                (generation, layerwise, recipe, None)
            };
            recipe.bind_resume_copy(inputs.resume_copy.take().ok_or_else(unknown)?)?;
            (generation, layerwise, Some(recipe), media_storage)
        } else {
            let (generation, layerwise) = model.quote_replicated_text_with_existing_sampling(
                geometry,
                &inputs.state,
                &context,
                sampling,
            )?;
            (generation, layerwise, None, None)
        };
        Ok(Traced { generation,layerwise,native_recipe,media_storage,inputs,
            text_interventions:intervention_rows.into_inner() })
}
#[inline(never)]
pub(super) fn finish<'a>(runtime:&ModelRuntime<MlxBackend<'_>>,source:&child::Source<'a, '_>,
    config:TextGenerationConfig,controller:TextControllerWorkspace<'_>,traced:Traced,
    planning_metadata:Option<&HostMetadataFunding>)->Result<PreparedSavedTextResumeQuote<'a>,Error> {
        let memory=|cause:WorkingMemoryError|planned_error(cause,planning_metadata);
        let unknown=||memory(WorkingMemoryError::UnknownBound);
        let session=runtime.session();let model=&session.payload.model;let pool=runtime.backend().memory_pool();
        let binding=TextArrayBinding::Saved(&source.sampling);
        let Traced {generation,layerwise,native_recipe,media_storage,inputs,text_interventions}=traced;
        let Inputs { original,geometry,source_layout,decoder_host_peak,sampler_host_peak,pending_host_bytes,
            prepared_media:_,resume_copy:_,state:copied_state,source_native,decoder_copy,sampling_copy,
            copied_key,copied_input,random:_,copy_preparation,registered_source,workspace,prepared_parameters:_ }=inputs;
        let context=workspace.context();
        let report_metadata=WorkspaceReportMetadata::new(context);
        let report_error=|cause:eredu_runtime::working_memory::WorkspaceReportError| {
            if report_metadata.is_checked(){Error::Neural(report_metadata.error(cause))}else{other(cause.into_capability())}
        };
        if generation
            .sampling
            .host_peak_bytes
            .is_some_and(|bytes| bytes < sampler_host_peak)
        {
            // Both mechanisms describe the same retained history and growth.
            // A disagreement must never silently underfund the fixed host hold.
            return Err(memory(WorkingMemoryError::IdentityMismatch));
        }
        let contract =
            TextControllerContract::from_workspace(controller, generation.sampling.output_width)
                .map_err(|cause| planned_error(cause, planning_metadata))?;
        let capabilities = session.payload.model.erased().capability_estimate();
        // These are immutable semantic capabilities, not current branch arrays.
        // Compare their exact schedule with the saved source before refinement.
        let layout = source_layout.layout();
        if capabilities.state_layout().layer_layout() != layout.layers()
            || !capabilities
                .state_layout()
                .layer_prefix_offsets()
                .iter()
                .copied()
                .eq(layout.iter_layer_prefix_offsets())
        {
            return Err(memory(WorkingMemoryError::IdentityMismatch));
        }
        let state_input = match source.sampling.pending_media().filter(|_| geometry.max_output_tokens > 0) {
            Some(media) => media
                .packet()
                .borrowed_semantics()
                .request_position_facts()
                .and_then(|positions| {
                    positions.legacy_input_accounting(std::num::NonZeroU8::new(4).unwrap())
                })
                .map_err(|cause| planned_error(cause, planning_metadata))?,
            None => InputTokenCount::text(
                geometry
                    .cached_positions
                    .checked_add(geometry.input_positions)
                    .ok_or_else(|| memory(WorkingMemoryError::Overflow))?,
            ),
        };
        let state = eredu_core::estimate_runtime_state_facts(
            capabilities.state_layout(),
            state_input,
            geometry.max_output_tokens,
            1,
            session.floating_state_dtype_bytes,
        )
        .map_err(|cause| report_error(cause.into()))?;
        let state = report_metadata
            .state_from_facts(state)
            .map_err(report_error)?;
        let state = if geometry.max_output_tokens == 0 { state } else { generation
            .equations.refine_state_backing_metadata(state, report_metadata).map_err(report_error)? };
        let native_copy = full_span(
            &sampling_copy,
            "complete saved pending-token isolation/reshape/cast; no source credit",
            report_metadata,
        )?;
        let native_controls =
            super::super::super::text_funding::text_work_control_bytes(geometry.max_output_tokens)?
                .checked_add(super::super::super::text_quote::owner_control_bytes()?)
                .ok_or_else(|| memory(WorkingMemoryError::Overflow))?;
        let retained_source_controls = layerwise
            .as_ref()
            .map(|workspace| {
                workspace
                    .known_retained_control_bytes(false)
                    .ok_or_else(unknown)
            })
            .transpose()?
            .unwrap_or(0);
        let retained = decoder_host_peak
            .checked_add(controller.additional_host_bytes)
            .and_then(|n| n.checked_add(native_controls))
            .and_then(|n| n.checked_add(retained_source_controls))
            .ok_or_else(|| memory(WorkingMemoryError::Overflow))?;
        let pending_host = match pending_host_bytes {
            Some(bytes) => report_metadata.bounded(
                bytes,
                format_args!("closed one-part saved-token input host construction including explicit move overlap; original shared prompt identity remains source-pinned"),
            ),
            None => report_metadata.bounded(0, format_args!("completed media parts/cache/B are retained immutable source owners; new prompt controls use the admitted resume host constructor")),
        }.map_err(report_error)?;
        report_metadata
            .admit::<ExecutionWorkspaceEstimate>()
            .map_err(report_error)?;
        let outside = ExecutionWorkspaceEstimate {
            geometry,
            activations: report_metadata.clone_bound(&pending_host).map_err(report_error)?,
            attention: report_metadata.bounded(0, format_args!("attention belongs to the shared equation trace")).map_err(report_error)?,
            vocabulary: report_metadata.bounded(0, format_args!("actual populated sampling is composed once by the shared generation report")).map_err(report_error)?,
            state_update: native_copy,
            materialization: match layerwise.as_ref() {
                Some(workspace) => report_metadata.clone_bound(workspace.materialization()).map_err(report_error)?,
                None => report_metadata.bounded(0, format_args!("published fully resident model; no transfer, collective or layerwise materialization")).map_err(report_error)?,
            },
            retained: report_metadata.bounded(retained, format_args!("known retained source snapshot/identity payload (cold scratch and direct metadata remain separate); dense table initialization including moved values, concrete original quote and finite work Rc allocation/retirement controls (legacy diagnostic, no escaped-alias host hold), plus declared controller host payload; sampler history is already in the populated sampling peak")).map_err(report_error)?,
        };
        let outside = generation
            .enclosing_sampling_workspace_metadata(outside, report_metadata)
            .map_err(report_error)?;
        let paged_host_facts = if original {
            source_native
                .paged_host_source_facts(generation.equations.span_workspace_plan(), pool, &context)
                .map_err(|cause| planned_error(cause, planning_metadata))?
        } else {
            None
        };
        let incremental = if let Some(recipe) = native_recipe.as_ref() {
            let mut census = RetainedStorage::original_census(pool);
            let mut decoder_census = RetainedStorage::original_census(pool);
            session
                .payload
                .collect_retained_idle_storage(&mut census, &mut decoder_census)?;
            DecoderCopyOwner::Saved(source.decoder.clone()).collect_storage(&mut census)?;
            for array in binding.key().into_iter().chain(binding.pending()) {
                census.include_array(array)?;
            }
            let opening_rows = census
                .original_publication_rows(pool)?
                .zip(decoder_census.original_publication_rows(pool)?)
                .and_then(|(a, b)| a.checked_add(b));
            super::super::super::text_quote::seal_saved_native_quote(
                runtime,
                config,
                recipe,
                &generation,
                layerwise.as_ref(),
                copy_preparation,
                &sampling_copy,
                media_storage.as_ref(),
                state,
                outside,
                opening_rows,
                !source_native.paged_sources().is_empty(),
                paged_host_facts,
                source
                    .capture_checkpoint()
                    .zip(source.capture_selection())
                    .zip(source.capture_witness())
                    .map(|((checkpoint, selection), witness)| (checkpoint, selection, witness)),
                report_metadata,
            )?
        } else {
            copy_preparation
                .compose_inference_metadata(&generation.equations, state, outside, report_metadata)
                .map_err(|cause| {
                    if report_metadata.is_checked() {
                        Error::Neural(cause.into_workspace(report_metadata))
                    } else {
                        other(cause.into_legacy())
                    }
                })?
        };
        let full = report_metadata
            .clone_state(incremental.state())
            .map_err(report_error)?;
        context
            .charge_metadata(std::mem::size_of::<(
                Option<WorkspaceContext>,
                Option<WorkspaceContext>,
                Option<eredu_runtime::working_memory::HostSourceConstructionFacts>,
            )>())
            .map_err(|cause| planned_error(cause, planning_metadata))?;
        let continuation_metadata = original
            .then(|| model.erased().retain_continuation_metadata(&context))
            .flatten();
        let paged_source_metadata = (original
            && !source_native.paged_sources().is_empty())
        .then(|| context.clone());
        Ok(PreparedSavedTextResumeQuote {
            source: source.saved,
            child_capture: None,
            geometry,
            state_input,
            config,
            controller: contract,
            full,
            incremental,
            generation,
            native_recipe,
            layerwise,
            decoder_copy: decoder_copy,
            sampling_copy,
            decoder_host_peak,
            sampler_host_peak,
            pending_host,
            copied_state: copied_state,
            source_native: Some(source_native),
            copied_key,
            copied_input,
            media_storage,
            _registered_source: registered_source,
            continuation_metadata,
            paged_source_metadata,
            paged_host_facts,
            text_interventions,
            planning_metadata: planning_metadata.cloned(),
        })
}
pub(super) fn control_bytes()->Option<usize> {
    let parts=[size_of::<Inputs>()*2,size_of::<Traced>()*2,
        size_of::<Result<Inputs,Error>>(),size_of::<Result<Traced,Error>>(),
        size_of::<eredu_runtime::SharedStateLayout>(),size_of::<Option<u64>>(),
        size_of::<(&ModelRuntime<MlxBackend<'_>>,&CopiedTextComponents,TextGenerationConfig,TextControllerWorkspace<'_>,
            Option<&eredu_core::HostPreparationAuthority>,Option<&HostMetadataFunding>)>(),
        size_of::<(&ModelRuntime<MlxBackend<'_>>,&CopiedTextComponents,TextControllerWorkspace<'_>,Inputs,Option<&HostMetadataFunding>)>(),
        size_of::<(&ModelRuntime<MlxBackend<'_>>,&CopiedTextComponents,TextGenerationConfig,TextControllerWorkspace<'_>,Traced,Option<&HostMetadataFunding>)>()];
    parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
}
