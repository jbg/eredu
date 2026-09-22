//! One actual resident assistant equation through the shared model scope worker.
use super::{MlxExternalAssistant, retain_external_evidence_for_placement, workspace};
use crate::backend::runtime::residency::manager::OriginalMaterializedLoan;
use crate::{
    MlxTensor,
    backend::{
        error::Error,
        nn::{shared::MlxNeuralBackend, workspace::ProjectedNativeStorage},
        runtime::cache::{
            kv::ConcatKeyValueCache,
            state::{
                CompletedResidentSource, OriginalResidentSourceBinding,
                bind_completed_resident_source_priors,
            },
        },
    },
    composition::mlx::speculative::{
        SpeculativeExecutionStreams,
        embedded_native::{
            ActiveEmbeddedNativeInvocation, EmbeddedNativeLayout, ExternalEquationRecipe,
        },
        tensor_sources::{
            completed_tensor_source, registered_tensor_sources, validate_input_evidence,
        },
    },
};
use eredu_architectures::{
    ExternalAssistantArchitecture,
    external_assistant::{ExternalOperationResult, invocation::ExternalAssistantOperation},
    speculative_execution::PreparedEmbeddedEvidence,
};
use eredu_core::{HostPreparationAuthority, speculative::SamplingPlacement};
use eredu_nn::workspace::{HostMetadataFunding, HostMetadataFundingError, WorkspaceContext};
use eredu_runtime::working_memory::{
    InferenceWorkspaceReport, OriginalExternalSpeculativeRole, OriginalSpeculativeBudgetCustody,
    SpeculativeInvocationRequirements, WorkingMemoryError,
};
use safemlx::{OriginalBufferBudget, SubmissionScope};
use std::{
    cell::Cell,
    mem::{size_of, size_of_val},
};

#[derive(Debug, thiserror::Error)]
#[error("assistant equation {stage}: {cause}")]
struct AssistantEquationFailure {
    stage: &'static str,
    #[source]
    cause: Error,
}

struct Payload {
    report: InferenceWorkspaceReport,
    storage: ProjectedNativeStorage,
    binding: OriginalResidentSourceBinding,
    prior: Vec<PreparedEmbeddedEvidence>,
    closing_roots: usize,
    context: WorkspaceContext,
    host: HostPreparationAuthority,
}
struct Work<'work, 'args, A: ExternalAssistantArchitecture, I: ExternalAssistantOperation<A>> {
    assistant: &'work mut MlxExternalAssistant<A>,
    arguments: &'work mut I::Arguments<'args, MlxTensor>,
    prior: &'work [PreparedEmbeddedEvidence],
    completed: Option<PreparedEmbeddedEvidence>,
}
fn charge(funding: &HostMetadataFunding, parts: &[usize]) -> Result<(), Error> {
    funding
        .reserve_metadata(
            parts
                .iter()
                .copied()
                .try_fold(size_of_val(parts), usize::checked_add)
                .ok_or(Error::WorkspacePlanning(HostMetadataFundingError::Overflow))?,
        )
        .map_err(Error::WorkspacePlanning)
}
fn host(funding: &HostMetadataFunding) -> Result<HostPreparationAuthority, Error> {
    charge(
        funding,
        &[
            HostPreparationAuthority::retention_bytes::<HostMetadataFunding>()
                .ok_or(Error::WorkspacePlanning(HostMetadataFundingError::Overflow))?,
            size_of::<HostPreparationAuthority>(),
            size_of::<Result<HostPreparationAuthority, Error>>(),
        ],
    )?;
    Ok(HostPreparationAuthority::retain(funding.clone()))
}
fn prior_sources<'a>(
    prior: &'a [PreparedEmbeddedEvidence],
    context: SpeculativeExecutionStreams<'_>,
    funding: &HostMetadataFunding,
) -> Result<Vec<&'a CompletedResidentSource>, Error> {
    charge(
        funding,
        &[
            size_of::<Vec<&CompletedResidentSource>>(),
            size_of::<std::slice::Iter<'_, PreparedEmbeddedEvidence>>(),
            size_of::<Result<Vec<&CompletedResidentSource>, Error>>(),
            size_of::<(
                &[PreparedEmbeddedEvidence],
                SpeculativeExecutionStreams<'_>,
                &HostMetadataFunding,
            )>(),
        ],
    )?;
    let mut result = funding.metadata_vec(prior.len()).map_err(Error::Neural)?;
    for evidence in prior {
        validate_input_evidence(evidence, context, SamplingPlacement::Draft, funding)?;
        if let Some(source) = completed_tensor_source(evidence) {
            result.push(source);
        }
    }
    Ok(result)
}
fn validate_registered(
    prior: &[PreparedEmbeddedEvidence],
    storage: &ProjectedNativeStorage,
    funding: &HostMetadataFunding,
) -> Result<(), Error> {
    charge(
        funding,
        &[
            size_of::<super::super::tensor_sources::RegisteredTensorSources<'_>>(),
            size_of::<std::slice::Iter<'_, PreparedEmbeddedEvidence>>(),
            size_of::<Result<(), Error>>(),
            size_of::<(
                &[PreparedEmbeddedEvidence],
                &ProjectedNativeStorage,
                &HostMetadataFunding,
            )>(),
        ],
    )?;
    for evidence in prior {
        for source in registered_tensor_sources(evidence) {
            // prior_sources authenticated every declaration before this exact
            // selection check; unused retained witnesses grant no extra inputs.
            source.validate_selected_projection(storage, funding)?;
        }
    }
    Ok(())
}
fn handler_controls<W, T, B, F, G, H>(
    layout: &EmbeddedNativeLayout,
    _: &W,
    handlers: &(F, G, H),
) -> Option<u64>
where
    F: FnOnce(&mut W, &Payload, &SubmissionScope, OriginalMaterializedLoan<'_>) -> Result<B, Error>,
    G: FnOnce(
        &mut W,
        &Payload,
        &ActiveEmbeddedNativeInvocation<'_, OriginalExternalSpeculativeRole>,
    ) -> Result<T, Error>,
    H: FnOnce(
        &mut W,
        &T,
        &OriginalBufferBudget,
        &OriginalSpeculativeBudgetCustody,
    ) -> Result<(), Error>,
{
    layout.control_bytes_for_role::<OriginalExternalSpeculativeRole, Payload, W, T, B, F, G, H>(
        handlers,
    )
}

pub(crate) fn execute<A, I>(
    assistant: &mut MlxExternalAssistant<A>,
    mut arguments: I::Arguments<'_, MlxTensor>,
    context: SpeculativeExecutionStreams<'_>,
) -> Result<ExternalOperationResult<I::Output<MlxTensor>>, Error>
where
    A: ExternalAssistantArchitecture,
    I: ExternalAssistantOperation<A>,
{
    let source = context
        .original_external()
        .ok_or(Error::PrefillControl(WorkingMemoryError::IdentityMismatch))?;
    let sources = source.numerical_sources();
    let (context_sources, environment) =
        context
            .original_numerical_for(SamplingPlacement::Draft)
            .ok_or_else(|| sources.retain_startup_error(WorkingMemoryError::IdentityMismatch))?;
    let phase_funding = sources.prepare_phase_metadata()?;
    let funding = &phase_funding;
    // This operation uses its actual admitted Draft environment. Same-device
    // inputs keep their completed origin; cross-device inputs must already own
    // completed destination copies. The native layout still validates its own
    // device, stream, scopes and retained mechanism source.
    if !std::ptr::eq(sources, context_sources) {
        return Err(sources.retain_startup_error(WorkingMemoryError::IdentityMismatch));
    }
    if context.draft() != environment.stream() {
        return Err(sources.retain_startup_error(WorkingMemoryError::UnknownBound));
    }
    let phase = Cell::new("environment validation");
    let result = (|| {
        sources.validate_environment(environment)?;
        charge(
            funding,
            &[
                size_of::<Payload>(),
                size_of::<Work<'_, '_, A, I>>(),
                size_of::<Cell<&'static str>>(),
                size_of::<AssistantEquationFailure>(),
                size_of::<I::Arguments<'_, MlxTensor>>(),
                size_of::<I::Output<MlxTensor>>(),
                size_of::<ExternalOperationResult<I::Output<MlxTensor>>>(),
                size_of::<Result<ExternalOperationResult<I::Output<MlxTensor>>, Error>>(),
                size_of::<WorkspaceContext>(),
                size_of::<Result<WorkspaceContext, eredu_nn::workspace::WorkspaceMetadataError>>(),
                size_of::<Vec<PreparedEmbeddedEvidence>>(),
                size_of::<Vec<PreparedEmbeddedEvidence>>(),
                size_of::<Option<usize>>(),
                size_of::<ExternalEquationRecipe>(),
                size_of::<OriginalExternalSpeculativeRole>(),
                size_of::<SpeculativeInvocationRequirements>(),
                size_of::<Result<SpeculativeInvocationRequirements, WorkingMemoryError>>(),
                size_of::<
                    Result<
                        OriginalExternalSpeculativeRole,
                        eredu_runtime::working_memory::SpeculativeRequestError,
                    >,
                >(),
                size_of::<(
                    EmbeddedNativeLayout,
                    Result<
                        EmbeddedNativeLayout,
                        super::super::embedded_native::EmbeddedNativeCause,
                    >,
                )>(),
                size_of::<Result<(), Error>>(),
                size_of::<OriginalResidentSourceBinding>(),
            ],
        )?;
        phase.set("prior evidence preparation");
        let origin = context
            .external_origin()
            .ok_or(Error::PrefillControl(WorkingMemoryError::IdentityMismatch))?;
        let mut count = Some(context.embedded_tensor_sources().len());
        I::visit_evidence(&arguments, &mut |_| {
            count = count.and_then(|n| n.checked_add(1));
        });
        let count = count.ok_or(Error::PrefillControl(WorkingMemoryError::Overflow))?;
        let mut prior = funding.metadata_vec(count).map_err(Error::Neural)?;
        for evidence in context.embedded_tensor_sources() {
            prior.push((*evidence).clone());
        }
        let mut extra = false;
        I::visit_evidence(&arguments, &mut |evidence| {
            if prior.len() < count {
                prior.push(evidence.clone());
            } else {
                extra = true;
            }
        });
        if extra || prior.len() != count {
            return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
        }
        let (roots, ordinary) = sources.numerical_prerequisites();
        let mechanism =
            workspace::AssistantMechanisms::from_stream(ordinary, environment.stream(), funding)?;
        let quote_context = mechanism
            .context(funding.clone())
            .map_err(|cause| sources.retain_startup_error(cause))?;
        let host = host(funding)?;
        phase.set("source quotation");
        let parts = workspace::quote_with_invocation::<A, I, _, _>(
            assistant,
            &arguments,
            mechanism,
            &quote_context,
            |geometry| {
                phase.set("source invocation claim");
                source
                    .claim(I::invocation_kind(), geometry, None, origin)
                    .map_err(|cause| quote_context.metadata_source(cause))
            },
            |storage, quote_context| {
                phase.set("completed predecessor admission");
                let completed = prior_sources(&prior, context, funding)
                    .map_err(|cause| quote_context.metadata_source(cause))?;
                phase.set("registered predecessor projection");
                validate_registered(&prior, storage, funding)
                    .map_err(|cause| quote_context.metadata_source(cause))?;
                phase.set("source account binding");
                let binding = bind_completed_resident_source_priors(
                    quote_context,
                    storage,
                    &completed,
                    environment,
                    funding,
                    &host,
                )
                .map_err(|cause| quote_context.metadata_source(cause))?;
                phase.set("workspace equation quotation");
                Ok(binding)
            },
        )?
        .into_parts();
        let claim = parts.invocation;
        phase.set("native layout inspection");
        let recipe = ExternalEquationRecipe::new(claim.invocation(), parts.recipe)?;
        let layout = EmbeddedNativeLayout::inspect(&recipe, environment, 0, 0)
            .map_err(|cause| sources.retain_startup_error(cause))?;
        // Recovery retains source evidence even when an unfinished native error
        // escapes and the caller's mutable argument frame has already returned.
        let mut retained_prior = funding.metadata_vec(prior.len()).map_err(Error::Neural)?;
        retained_prior.extend(prior.iter().cloned());
        let payload = Payload {
            report: parts.report,
            storage: parts.storage,
            binding: parts.bindings,
            closing_roots: parts.closing_roots,
            prior: retained_prior,
            context: parts.context,
            host,
        };
        let mut work = Work::<A, I> {
            assistant,
            arguments: &mut arguments,
            prior: &prior,
            completed: None,
        };
        let handlers = (
            |_work: &mut Work<'_, '_, A,I>, payload: &Payload, _scope: &SubmissionScope,
             _materialized: OriginalMaterializedLoan<'_>| {
                // The quote's one combined source pin owner remains in Q.
                let _ = (&payload.report, &payload.storage, &payload.binding, &payload.prior,
                    &payload.context, &payload.host);
                Ok(())
            },
            |work: &mut Work<'_, '_, A,I>, payload: &Payload,
                active: &ActiveEmbeddedNativeInvocation<'_, OriginalExternalSpeculativeRole>| {
                phase.set("native equation entry");
                active.begin_equation()?;
                phase.set("native scope validation");
                active.validate_equation_scope(context.draft())?;
                phase.set("native equation execution");
                let output = I::execute_with_metadata::<MlxNeuralBackend, ConcatKeyValueCache>(
                    &mut work.assistant.module.inner, I::reborrow(work.arguments), context.draft(), &payload.context)?;
                phase.set("native equation completion");
                active.complete(|visit| {
                    let mut count = Some(0usize);
                    let mut value = |value: &MlxTensor| {
                        count = count.and_then(|n| n.checked_add(1));
                        visit(value.as_array());
                    };
                    I::visit_arguments(work.arguments, &mut value);
                    I::visit_output(&output, &mut value);
                    if count != Some(payload.closing_roots) {
                        return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch).at_speculative_stage("assistant closing root population"));
                    }
                    Ok(())
                })?;
                Ok(output)
            },
            |work: &mut Work<'_, '_, A,I>, output: &I::Output<MlxTensor>,
                budget: &OriginalBufferBudget, custody: &OriginalSpeculativeBudgetCustody| {
                phase.set("completed predecessor validation");
                let completed = prior_sources(work.prior, context, funding)
                    .map_err(|cause| cause.at_speculative_stage("assistant predecessor revalidation"))?;
                phase.set("completed source capture");
                let source = CompletedResidentSource::capture_array_sources_with_priors(|visit| {
                    I::visit_arguments(work.arguments, &mut |value| visit(value.as_array()));
                    I::visit_output(output, &mut |value| visit(value.as_array()));
                }, &completed, budget, custody, funding)
                    .map_err(|cause| cause.at_speculative_stage("assistant completed root capture"))?
                    .with_completed_stream(environment.stream())
                    .map_err(|cause| cause.at_speculative_stage("assistant completed stream publication"))?;
                drop(completed);
                charge(funding, &[size_of::<Vec<&PreparedEmbeddedEvidence>>(),
                    size_of::<std::slice::Iter<'_, PreparedEmbeddedEvidence>>()])?;
                let mut priors = funding.metadata_vec(work.prior.len()).map_err(Error::Neural)?;
                priors.extend(work.prior.iter());
                phase.set("completed source retention");
                let evidence = retain_external_evidence_for_placement(source, |visit|{
                    I::visit_arguments(work.arguments,&mut |value|visit(value.as_array()));
                    I::visit_output(output,&mut |value|visit(value.as_array()));
                }, &priors, context, SamplingPlacement::Draft, funding)
                    .map_err(|cause| cause.at_speculative_stage("assistant completed evidence retention"))?;
                I::retain_evidence(work.arguments, evidence.clone());
                work.completed = Some(evidence);
                Ok(())
            },
        );
        let requirements = SpeculativeInvocationRequirements::new(
            recipe.plan(),
            layout.physical_bytes(),
            Some(layout.graph_bytes()),
            Some(layout.record_bytes()),
            handler_controls(&layout, &work, &handlers),
            environment
                .buffer_placement()
                .map_err(|cause| sources.retain_startup_error(cause))?,
        )
        .map_err(|cause| sources.retain_startup_error(cause))?;
        phase.set("role reservation");
        let role = sources
            .request()
            .reserve_external_role(claim, requirements)
            .map_err(|cause| sources.retain_startup_error(cause))?;
        phase.set("neural bank admission");
        role.claim_neural_bank(0)
            .map_err(|cause| sources.retain_startup_error(cause))?;
        phase.set("native role preparation");
        let output = layout.run(
            &recipe,
            environment,
            roots,
            sources.request(),
            role,
            payload,
            funding.clone(),
            &mut work,
            handlers,
        )?;
        phase.set("completed evidence publication");
        let evidence = work.completed.take().ok_or(
            Error::PrefillControl(WorkingMemoryError::IdentityMismatch)
                .at_speculative_stage("assistant completed evidence publication"),
        )?;
        Ok(ExternalOperationResult {
            output,
            evidence: Some(evidence),
        })
    })();
    result.map_err(|cause| {
        sources.retain_startup_error(AssistantEquationFailure {
            stage: phase.get(),
            cause,
        })
    })
}

use eredu_nn::workspace::WorkspaceMetadataAllocation;
