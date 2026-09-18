//! Atomic placement of the two actual completed machines supplied by core.
use super::*;
use eredu_core::{PendingTextInput, TextBranchSource, TextStepContext};
use eredu_runtime::replicated_session::{ControlBranchSource, ControlBranchPlacement};
use eredu_nn::workspace::HostMetadataFunding;
use std::mem::{size_of, size_of_val};

#[track_caller]
fn mismatch() -> Error { memory(WorkingMemoryError::IdentityMismatch) }

pub(in crate::composition::mlx::session::model_session) fn exchange<'d>(runtime: &mut ModelRuntime<MlxBackend<'d>>,
    installed: TextBranchSource<'_, MlxBackend<'d>>, incoming: TextBranchSource<'_, MlxBackend<'d>>,
    slot: &mut super::super::MlxNativeTextState) -> Result<(), Error> {
    if installed.state().sampling.quote.is_none() && incoming.state().sampling.quote.is_none() {
        // Explicit ordinary custody uses the same typed state exchange. Missing
        // or stripped original admission must never enter this accounting policy.
        for source in [&installed, &incoming] {
            if source.state().sampling.sampler.is_funded() || source.state().funding.is_some()
                || matches!(source.pending(), Some(PendingTextInput::Prefill(prompt)) if prompt.has_original_input_custody()) {
                return Err(mismatch());
            }
        }
        return <MlxBackend<'_> as eredu_core::execution_control::NativeTextStateBackend>::exchange_native_text_state(runtime, slot);
    }
    let installed_quote = installed.state().sampling.quote.as_ref().ok_or_else(mismatch)?.clone();
    let incoming_quote = incoming.state().sampling.quote.as_ref().ok_or_else(mismatch)?.clone();
    installed_quote.request.validate_branch_exchange(installed.context(), &incoming_quote.request, incoming.context())
        .map_err(memory)?;
    runtime.validate_session_admission()?;
    runtime.session().ensure_no_submission_in_flight()?;
    installed_quote.validate(runtime, &installed_quote.request)?;
    incoming_quote.validate(runtime, &incoming_quote.request)?;
    if slot.host_preparation.is_none() || installed_quote.same_owner(&incoming_quote) { return Err(mismatch()); }
    let capacity = installed_quote.config.inference_policy().managed_memory_capacity_bytes.ok_or_else(unknown)?;
    let funding = installed_quote.model_pool.prepare_workspace_metadata(
        runtime.session().payload.model.erased().inference_execution_identity(), capacity)
        .map_err(Error::WorkspacePlanning)?;
    let result = exchange_funded(runtime, installed, incoming, slot, &installed_quote, &incoming_quote, &funding);
    result.map_err(|cause| crate::composition::mlx::model::retain_planning_error(cause, funding))
}

fn validate_source(quote: &TextExecutionQuoteOwner, source: &ControlBranchSource,
    state: &generation::MlxTextGenerationState, context: &TextStepContext,
    pending: Option<PendingTextInput<&MlxModelInput, &MlxTextToken>>) -> Result<(), Error> {
    if quote.local_prediction_fixed(state.sampling.next_prediction).map_err(memory)? != context.attempt()
        || state.sampling.temperature != quote.sampling_temperature()?
        || state.sampling.parameter_epoch != Some(quote.parameter_epoch)
        || !state.sampling.sampler.is_funded() { return Err(mismatch()); }
    source.validate_frontier(quote.prediction_frontier(state.sampling.next_prediction)?).map_err(memory)?;
    match pending {
        Some(PendingTextInput::Prefill(prompt)) => {
            if context.attempt() != 0 { return Err(mismatch()); }
            super::super::text_step::validate_prompt_binding_fixed(prompt, quote).map_err(memory)?;
            quote.opening.validate_source(source.revision(), source.admission(), quote.request.geometry().cached_positions)
                .map_err(memory)?;
            if let Some(input::OriginalMediaPacket::Original(packet)) = prompt.original_media.as_ref() {
                quote.validate_completed_pending_input(prompt).map_err(memory)?;
                let semantics = prompt.placement_semantics.as_ref().or_else(|| quote.copied_media_semantics())
                    .unwrap_or_else(|| packet.borrowed_semantics());
                source.validate_media(semantics.binding()).map_err(memory)?;
            }
        }
        Some(PendingTextInput::Decode(token)) => {
            if token.state_revision() != Some(source.revision()) { return Err(mismatch()); }
            token.step_receipt().ok_or_else(mismatch)?
                .validate_completed_source(&quote.request, context.attempt()).map_err(memory)?;
            source.admission().ok_or_else(mismatch)?.request().validate_same_request(&quote.request).map_err(memory)?;
        }
        None => {
            if context.attempt() == 0 {
                quote.opening.validate_source(source.revision(), source.admission(), quote.request.geometry().cached_positions)
                    .map_err(memory)?;
            } else { source.admission().ok_or_else(mismatch)?.request().validate_same_request(&quote.request).map_err(memory)?; }
        }
    }
    Ok(())
}

fn place_pending(pending: Option<PendingTextInput<&mut MlxModelInput, &mut MlxTextToken>>,
    quote: &TextExecutionQuoteOwner, placement: &ControlBranchPlacement) {
    match pending {
        Some(PendingTextInput::Decode(token)) => token.apply_branch_placement(placement),
        Some(PendingTextInput::Prefill(prompt)) => {
            if let Some(input::OriginalMediaPacket::Original(packet)) = prompt.original_media.as_ref() {
                let semantics = prompt.placement_semantics.as_ref().or_else(|| quote.copied_media_semantics())
                    .unwrap_or_else(|| packet.borrowed_semantics());
                let transition = placement.media_transition(semantics.binding()).expect("validated exact media placement");
                let semantics = semantics.for_copied_state(&transition).expect("actual source-to-placement transition");
                prompt.placement_semantics = Some(semantics);
            }
        }
        None => {}
    }
}

#[inline(never)]
fn exchange_funded<'d>(runtime: &mut ModelRuntime<MlxBackend<'d>>,
    installed: TextBranchSource<'_, MlxBackend<'d>>, incoming: TextBranchSource<'_, MlxBackend<'d>>,
    slot: &mut super::super::MlxNativeTextState, installed_quote: &TextExecutionQuoteOwner,
    incoming_quote: &TextExecutionQuoteOwner, funding: &HostMetadataFunding) -> Result<(), Error> {
    let parts = [size_of::<[ControlBranchSource; 2]>(), size_of::<[ControlBranchPlacement; 2]>(),
        size_of::<eredu_runtime::replicated_session::ControlExchangeResult>(),
        size_of::<Result<eredu_runtime::replicated_session::ControlExchangeResult, Error>>(),
        size_of::<eredu_runtime::working_memory::PendingTextBranchExchange>(),
        size_of::<super::super::control_slot::PreparedControlSlotError>(),
        size_of::<(TextBranchSource<'_, MlxBackend<'_>>, TextBranchSource<'_, MlxBackend<'_>>)>() ,
        size_of::<[Option<std::cell::RefMut<'_, Option<eredu_runtime::working_memory::InferenceStateRevision>>>; 2]>(),
        size_of::<[Option<eredu_architectures::media_plan::BoundPreparedMediaSemantics>; 2]>(),
        size_of::<[eredu_runtime::working_memory::CopiedMediaStateBinding; 2]>(),
        size_of::<Result<(), Error>>(), size_of::<HostMetadataFunding>()];
    funding.reserve_metadata(parts.into_iter().try_fold(size_of_val(&parts), usize::checked_add)
        .ok_or(Error::WorkspacePlanning(eredu_core::HostMetadataFundingError::Overflow))?)
        .map_err(Error::WorkspacePlanning)?;
    let sources = runtime.session().payload.model.erased().original_control_branch_sources(slot.state.as_ref())?;
    validate_source(installed_quote, &sources[0], installed.state(), installed.context(), installed.pending())?;
    if let Some(previous) = slot.displaced_placement.as_ref() { previous.validate_destination(&sources[1]).map_err(memory)?; }
    let incoming_source = slot.displaced_placement.as_ref().map(|previous| previous.source()).unwrap_or(&sources[1]);
    validate_source(incoming_quote, incoming_source, incoming.state(), incoming.context(), incoming.pending())?;
    let pending = installed_quote.request.begin_branch_exchange(installed.context(), &incoming_quote.request,
        incoming.context(), funding).map_err(memory)?;
    // Every destination loan is acquired before the shared vote/state move.
    let mut installed_opening = (installed.context().attempt() == 0)
        .then(|| installed_quote.opening.placement_slot()).transpose().map_err(memory)?;
    let mut incoming_opening = (incoming.context().attempt() == 0)
        .then(|| incoming_quote.opening.placement_slot()).transpose().map_err(memory)?;
    let (_, _, installed_pending) = installed.into_parts();
    let (_, _, incoming_pending) = incoming.into_parts();
    let moved = Cell::new(false);
    let result = installed_quote.with_parallel_control(runtime, |runtime| {
        let payload = runtime.session_mut().payload.get_mut().ok_or_else(mismatch)?;
        // These are the paid directories already attached to the two states.
        // No ordinary retain/extend operation or fresh owner-vector is allowed.
        let result = payload.model.erased_mut().exchange_original_control_state(
            slot.state.as_mut(), funding, None, Some((&pending, &sources)))?;
        let [installed, mut incoming] = result.placements.expect("typed original branch exchange returns both placements");
        if let Some(previous) = slot.displaced_placement.take() {
            incoming = incoming.through(&previous).expect("validated exact deferred placement");
        }
        std::mem::swap(&mut payload.state_memory, &mut slot.memory_retention);
        if let Some(opening) = installed_opening.as_mut() { **opening = Some(installed.revision().clone()); }
        if let Some(opening) = incoming_opening.as_mut() { **opening = Some(incoming.revision().clone()); }
        place_pending(installed_pending, installed_quote, &installed);
        place_pending(incoming_pending, incoming_quote, &incoming);
        moved.set(true);
        Ok(())
    });
    if moved.get() && result.is_err() { runtime.session().poison.set(true); }
    result
}
