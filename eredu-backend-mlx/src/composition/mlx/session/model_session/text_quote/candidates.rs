//! Native error/decision binding around the shared cold chunk planner.

use super::*;
use eredu_runtime::working_memory::{
    IncompleteWorkspace, InferenceExecutionIdentity, PrefillPlanningError,
    WorkingMemoryCapacityHandoff, WorkingMemoryReservation,
    plan_prefill_incremental_with_capacity_handoff,
};

pub(in crate::composition::mlx::session::model_session) struct TextWorkspaceCandidate {
    pub(in crate::composition::mlx::session::model_session) quote:
        Result<IncrementalInferenceQuote, IncompleteWorkspace>,
    pub(in crate::composition::mlx::session::model_session) output_width: usize,
    pub(in crate::composition::mlx::session::model_session) native_recipe:
        Option<crate::backend::nn::workspace::ResidentNativeRecipe>,
    pub(in crate::composition::mlx::session::model_session) prepared_source:
        Option<original_prepared::PreparedMediaQuoteSource>,
    pub(in crate::composition::mlx::session::model_session) paged_sources:
        Option<crate::backend::nn::workspace::ProjectedPagedSources>,
    // Last: all candidate diagnostics retire before their planning account.
    pub(in crate::composition::mlx::session::model_session) planning_metadata:
        Option<eredu_nn::workspace::WorkspaceMetadataFunding>,
}

/// The first diagnostic candidate binds decisions even when its numerical
/// coverage is incomplete. Only the shared runtime planner chooses another
/// chunk. A typed tracking-capacity refusal enters that shared search;
/// source/readiness, geometry, identity and native errors stay fatal.
#[cfg(test)]
pub(super) fn plan_candidates(
    execution: &InferenceExecutionIdentity,
    pool: &WorkingMemoryPool,
    capabilities: &eredu_core::ModelCapabilities,
    request: AdmissionRequest,
    geometry: InferenceGeometry,
    capacity: u64,
    controller: TextControllerWorkspace<'_>,
    quote: impl FnMut(InferenceGeometry) -> Result<TextWorkspaceCandidate, Error>,
) -> Result<(WorkingMemoryReservation, TextControllerContract), Error> {
    plan_candidates_with_handoff(
        execution,
        pool,
        capabilities,
        request,
        geometry,
        capacity,
        controller,
        &[],
        quote,
    )
}

pub(super) fn plan_candidates_with_handoff(
    execution: &InferenceExecutionIdentity,
    pool: &WorkingMemoryPool,
    capabilities: &eredu_core::ModelCapabilities,
    request: AdmissionRequest,
    geometry: InferenceGeometry,
    capacity: u64,
    controller: TextControllerWorkspace<'_>,
    handoffs: &[WorkingMemoryCapacityHandoff],
    quote: impl FnMut(InferenceGeometry) -> Result<TextWorkspaceCandidate, Error>,
) -> Result<(WorkingMemoryReservation, TextControllerContract), Error> {
    let (reservation, controller, accepted, native_recipe, prepared_source, paged_sources, planning_metadata) =
        plan_candidates_with_handoff_retained(
            execution,
            pool,
            capabilities,
            request,
            geometry,
            capacity,
            controller,
            handoffs,
            quote,
        )?;
    // Preserve ordinary ownership: historical request metadata keeps no cold
    // diagnostic source pins beyond the already-created reservation.
    drop(native_recipe);
    drop(prepared_source);
    drop(paged_sources);
    drop(accepted);
    drop(planning_metadata);
    Ok((reservation, controller))
}

/// Original captured admission keeps the accepted proof only through its
/// post-publication source check, then extracts accounting-only validation.
pub(super) fn plan_candidates_with_handoff_retained(
    execution: &InferenceExecutionIdentity,
    pool: &WorkingMemoryPool,
    capabilities: &eredu_core::ModelCapabilities,
    request: AdmissionRequest,
    geometry: InferenceGeometry,
    capacity: u64,
    controller: TextControllerWorkspace<'_>,
    handoffs: &[WorkingMemoryCapacityHandoff],
    mut quote: impl FnMut(InferenceGeometry) -> Result<TextWorkspaceCandidate, Error>,
) -> Result<
    (
        WorkingMemoryReservation,
        TextControllerContract,
        IncrementalInferenceQuote,
        Option<crate::backend::nn::workspace::ResidentNativeRecipe>,
        Option<original_prepared::PreparedMediaQuoteSource>,
        Option<crate::backend::nn::workspace::ProjectedPagedSources>,
        Option<eredu_nn::workspace::WorkspaceMetadataFunding>,
    ),
    Error,
> {
    let mut planning_metadata = None;
    let result: Result<_, Error> = (|| {
        let mut original: Option<(usize, TextControllerContract)> = None;
        let mut native_failure = None;
        let mut native_recipe = None;
        let mut prepared_source = None;
        let mut paged_sources = None;
        let planned = plan_prefill_incremental_with_capacity_handoff(
            execution,
            pool,
            capabilities,
            request,
            geometry,
            capacity,
            handoffs,
            |candidate| {
                // The shared planner has retired the rejected candidate before
                // requesting the next size. Release its retained recipe first.
                drop(native_recipe.take());
                drop(prepared_source.take());
                drop(paged_sources.take());
                drop(planning_metadata.take());
                let inspected = (|| {
                    let candidate = quote(candidate)?;
                    planning_metadata = candidate.planning_metadata;
                    let contract =
                        TextControllerContract::from_workspace(controller, candidate.output_width)
                            .map_err(|error| Error::Other(Box::new(error)))?;
                    match &original {
                        Some((width, expected))
                            if *width != candidate.output_width || expected != &contract =>
                        {
                            return Err(memory(WorkingMemoryError::IdentityMismatch));
                        }
                        None => original = Some((candidate.output_width, contract)),
                        Some(_) => {}
                    }
                    // Reject a different decision proof before the planner reserves
                    // capacity, including a legacy quote with no controller binding.
                    if candidate
                        .quote
                        .as_ref()
                        .is_ok_and(|quote| quote.controller_contract() != Some(&contract))
                    {
                        return Err(memory(WorkingMemoryError::IdentityMismatch));
                    }
                    native_recipe = candidate.native_recipe;
                    prepared_source = candidate.prepared_source;
                    paged_sources = candidate.paged_sources;
                    Ok(candidate.quote)
                })();
                match inspected {
                    Ok(Ok(quote)) => Ok(quote),
                    Ok(Err(incomplete)) => {
                        Err(PrefillPlanningError::IncompleteWorkspace(incomplete))
                    }
                    Err(Error::PrefillControl(
                        cause @ (WorkingMemoryError::SubmissionTrackingCapacity { .. }
                        | WorkingMemoryError::GraphMetadataCapacity { .. }),
                    )) => {
                        // This cold constructor minimum may decrease with the
                        // chunk. Only the portable planner chooses another size;
                        // keep its exact neutral cause out of the fatal-error slot.
                        Err(PrefillPlanningError::Reservation(cause))
                    }
                    Err(error) => {
                        // Native errors retain their original source outside the
                        // portable callback type. Retryable coverage never enters
                        // this slot and cannot override a later accepted candidate.
                        native_failure = Some(error);
                        Err(PrefillPlanningError::Estimate(
                            CapabilityError::InvalidConfiguration {
                                field: "native_text_workspace",
                                detail: "selected native incremental quote failed".into(),
                            },
                        ))
                    }
                }
            },
        );
        if let Some(error) = native_failure {
            return Err(error);
        }
        let (_, reservation, accepted) = planned.map_err(|error| match error {
            PrefillPlanningError::Reservation(
                cause @ (WorkingMemoryError::SubmissionTrackingCapacity { .. }
                | WorkingMemoryError::GraphMetadataCapacity { .. }),
            ) => Error::PrefillControl(cause),
            error => Error::Other(Box::new(error)),
        })?;
        let (_, controller) =
            original.ok_or_else(|| memory(WorkingMemoryError::IdentityMismatch))?;
        if accepted.controller_contract() != Some(&controller) {
            return Err(memory(WorkingMemoryError::IdentityMismatch));
        }
        if native_recipe
            .as_ref()
            .is_some_and(|recipe| !recipe.plan().same_plan(accepted.span_workspace().plan()))
        {
            return Err(memory(WorkingMemoryError::IdentityMismatch));
        }
        Ok((
            reservation,
            controller,
            accepted,
            native_recipe,
            prepared_source,
            paged_sources,
            planning_metadata.clone(),
        ))
    })();
    result.map_err(|cause| match planning_metadata {
        Some(funding) => crate::composition::mlx::model::retain_planning_error(cause, funding),
        None => cause,
    })
}

#[cfg(test)]
mod tests;
