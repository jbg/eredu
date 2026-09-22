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
    pub(in crate::composition::mlx::session::model_session) ordinary_publication:
        Option<super::super::text_funding::OrdinaryPublicationPlan>,
    // The accepted constructor context carries paid host metadata only.
    pub(in crate::composition::mlx::session::model_session) execution_metadata:
        Option<eredu_nn::workspace::WorkspaceContext>,
    // Last: all candidate diagnostics retire before their planning account.
    pub(in crate::composition::mlx::session::model_session) planning_metadata:
        Option<eredu_nn::workspace::HostMetadataFunding>,
}

/// The first diagnostic candidate binds decisions even when its numerical
/// coverage is incomplete. Only the shared runtime planner chooses another
/// chunk. A typed tracking-capacity refusal enters that shared search;
/// source/readiness, geometry, identity and native errors stay fatal.
#[cfg(test)]
pub(super) fn plan_candidates(
    execution: &InferenceExecutionIdentity,
    pool: &MemoryLedger,
    capabilities: &eredu_core::ModelCapabilities,
    request: AdmissionRequest,
    geometry: InferenceGeometry,
    capacity: eredu_core::MemoryLimits,
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
    pool: &MemoryLedger,
    capabilities: &eredu_core::ModelCapabilities,
    request: AdmissionRequest,
    geometry: InferenceGeometry,
    capacity: eredu_core::MemoryLimits,
    controller: TextControllerWorkspace<'_>,
    handoffs: &[WorkingMemoryCapacityHandoff],
    quote: impl FnMut(InferenceGeometry) -> Result<TextWorkspaceCandidate, Error>,
) -> Result<(WorkingMemoryReservation, TextControllerContract), Error> {
    let (
        reservation,
        controller,
        accepted,
        native_recipe,
        prepared_source,
        paged_sources,
        execution_metadata,
        ordinary_publication,
        planning_metadata,
    ) = plan_candidates_with_handoff_retained(
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
    drop(execution_metadata);
    let _ = ordinary_publication;
    drop(accepted);
    drop(planning_metadata);
    Ok((reservation, controller))
}

/// Original captured admission keeps the accepted proof only through its
/// post-publication source check, then extracts accounting-only validation.
pub(super) fn plan_candidates_with_handoff_retained(
    execution: &InferenceExecutionIdentity,
    pool: &MemoryLedger,
    capabilities: &eredu_core::ModelCapabilities,
    request: AdmissionRequest,
    geometry: InferenceGeometry,
    capacity: eredu_core::MemoryLimits,
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
        Option<eredu_nn::workspace::WorkspaceContext>,
        Option<super::super::text_funding::OrdinaryPublicationPlan>,
        Option<eredu_nn::workspace::HostMetadataFunding>,
    ),
    Error,
> {
    let mut planning_metadata = None;
    let mut diagnostic_metadata = None;
    let result: Result<_, Error> = (|| {
        let mut original: Option<(usize, TextControllerContract)> = None;
        let mut last_components = None;
        let mut initial_components = None;
        let mut minimum_components = None;
        let mut native_failure = None;
        let mut native_recipe = None;
        let mut prepared_source = None;
        let mut paged_sources = None;
        let mut execution_metadata = None;
        let mut ordinary_publication = None;
        let planned = plan_prefill_incremental_with_capacity_handoff(
            execution,
            pool,
            capabilities,
            request,
            geometry,
            capacity.clone(),
            handoffs,
            |candidate| {
                last_components = None;
                // The shared planner has retired the rejected candidate before
                // requesting the next size. Release its retained recipe first.
                drop(native_recipe.take());
                drop(prepared_source.take());
                drop(paged_sources.take());
                drop(execution_metadata.take());
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
                    last_components = candidate
                        .native_recipe
                        .as_ref()
                        .map(|recipe| recipe.quote_components)
                        .or_else(|| {
                            candidate
                                .ordinary_publication
                                .as_ref()
                                .map(|plan| plan.quote_components())
                        })
                        // A complete per-domain quote intentionally has no
                        // aggregate byte amount. Retain its source diagnostics
                        // using the successful quote, not an optional scalar.
                        .filter(|_| candidate.quote.is_ok())
                        .map(|components| {
                            (
                                candidate
                                    .quote
                                    .as_ref()
                                    .expect("sealed candidate")
                                    .geometry(),
                                components,
                            )
                        });
                    if let Some(value) = last_components {
                        if initial_components.is_none() {
                            if planning_metadata.is_some() {
                                // These scalar copies survive candidate retirement.
                                // Give them their own small account instead of
                                // keeping an entire rejected recipe account alive.
                                let funding = pool
                                    .prepare_workspace_metadata(execution, capacity.clone())
                                    .map_err(Error::WorkspacePlanning)?;
                                funding
                                    .reserve_metadata(std::mem::size_of::<(
                                        Option<(
                                            InferenceGeometry,
                                            crate::backend::error::WorkspaceQuoteComponents,
                                        )>,
                                        Option<(
                                            InferenceGeometry,
                                            crate::backend::error::WorkspaceQuoteComponents,
                                        )>,
                                        Option<(
                                            InferenceGeometry,
                                            crate::backend::error::WorkspaceQuoteComponents,
                                        )>,
                                        Option<eredu_core::HostMetadataFunding>,
                                        Result<(), eredu_nn::workspace::HostMetadataFundingError>,
                                    )>())
                                    .map_err(Error::WorkspacePlanning)?;
                                diagnostic_metadata = Some(funding);
                            }
                            initial_components = Some(value);
                        }
                        // Only aggregate amounts provide this report's total
                        // ordering. A Host-domain diagnostic cannot order
                        // candidates whose separate device charges differ.
                        if let Some(amount) = value.1.after_seal {
                            if minimum_components.as_ref().is_none_or(
                                |(_, components): &(
                                    InferenceGeometry,
                                    crate::backend::error::WorkspaceQuoteComponents,
                                )| {
                                    components.after_seal.is_none_or(|prior| amount < prior)
                                },
                            ) {
                                minimum_components = Some(value);
                            }
                        }
                    }
                    native_recipe = candidate.native_recipe;
                    prepared_source = candidate.prepared_source;
                    paged_sources = candidate.paged_sources;
                    execution_metadata = candidate.execution_metadata;
                    ordinary_publication = candidate.ordinary_publication;
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
        let (reservation, accepted) = planned.map_err(|error| {
            if matches!(
                &error,
                PrefillPlanningError::Reservation(WorkingMemoryError::Domain(
                    eredu_core::MemoryDomainError::BudgetExceeded { .. }
                ))
            ) {
                if let Some((geometry, components)) = last_components {
                    if let Some(funding) = &planning_metadata {
                        // The enclosing closed planning failure retains this
                        // account until after the diagnostic Box is destroyed.
                        if let Err(cause) = funding.reserve_metadata(std::mem::size_of::<(
                            crate::backend::error::WorkspaceCandidateRefusal,
                            Box<crate::backend::error::WorkspaceCandidateRefusal>,
                            Result<(), eredu_nn::workspace::HostMetadataFundingError>,
                        )>()) {
                            return Error::WorkspacePlanning(cause);
                        }
                    }
                    return Error::Other(Box::new(
                        crate::backend::error::WorkspaceCandidateRefusal {
                            geometry,
                            components,
                            initial: initial_components,
                            minimum: minimum_components,
                            cause: error,
                            _funding: diagnostic_metadata.take(),
                        },
                    ));
                }
            }
            match error {
                PrefillPlanningError::Reservation(
                    cause @ (WorkingMemoryError::SubmissionTrackingCapacity { .. }
                    | WorkingMemoryError::GraphMetadataCapacity { .. }),
                ) => Error::PrefillControl(cause),
                error => Error::Other(Box::new(error)),
            }
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
            execution_metadata,
            ordinary_publication,
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
