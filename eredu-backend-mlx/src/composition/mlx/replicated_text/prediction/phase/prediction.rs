//! One source-bound prediction equation through the shared native recovery.
use super::super::workspace::source_bindings::SourceBindings;
use super::super::parameters::{
    clear_prediction_module_bank, inspect_prediction_module_plan, install_prediction_module_bank,
};
use super::super::workspace::quote::{PreparedPredictionEquationQuote, PreparedPredictionIo};
use super::binding::{Binding, StateCompletion};
use super::*;
use crate::backend::{
    OriginalCopyEnvironment,
    nn::workspace::{ProjectedNativeStorage, ProjectedResidentState},
    runtime::{
        cache::state::{
            CompletedResidentSource, OriginalResidentSourceBinding,
            bind_completed_resident_source_priors,
        },
        execution::generic::{
            LayerwiseWorkspace, PredictionModuleProjection, PreparedPredictionModuleBank,
        },
    },
};
use crate::composition::mlx::speculative::{
    OriginalSpeculativeNumericalSources, PendingModelLogits, completed_tensor_source,
    embedded_native::EmbeddedNativeLayout,
};
use eredu_architectures::speculative_execution::PreparedEmbeddedEvidence;
use eredu_core::{HostPreparationAuthority, OutputDemand};
use eredu_nn::workspace::{WorkspaceContext, WorkspaceMetadataFunding};
use eredu_runtime::{
    speculative::embedded_occurrence::EmbeddedInvocationWorkspace,
    working_memory::{
        InferenceWorkspaceReport, SpeculativeInvocationRequirements, WorkingMemoryError,
    },
};
use safemlx::Array;
use std::{
    cell::RefCell,
    convert::Infallible,
    marker::PhantomData,
    mem::{size_of, size_of_val},
};

struct Payload {
    capture: Option<super::capture::Capture>,
    io: RefCell<PreparedPredictionIo>,
    logits: Option<PendingModelLogits>,
    completion: Option<RefCell<StateCompletion>>,
    modules: Option<PreparedPredictionModuleBank>,
    target: ProjectedResidentState,
    prediction: ProjectedNativeStorage,
    inputs: ProjectedNativeStorage,
    parameters: PredictionParameterStorage,
    layerwise: Option<LayerwiseWorkspace>,
    report: InferenceWorkspaceReport,
    bindings: SourceBindings,
    context: WorkspaceContext,
    funding: WorkspaceMetadataFunding,
}

// Ownership moves across the cold quote/native bank boundary. Keeping the
// bank/recovery constructor out of `run` prevents its temporaries from occupying
// the stack while the family quote is still constructing its workspace modules.
struct QuotedPreparation<'a> {
    parts: super::super::workspace::quote::PredictionEquationQuoteParts,
    claim: eredu_runtime::speculative::embedded_occurrence::EmbeddedOccurrenceClaim<'a>,
    workspace: EmbeddedInvocationWorkspace,
    capture: Option<eredu_runtime::capture::OriginalSpeculativeCaptureInvocation<'a>>,
    capture_host: Option<eredu_runtime::working_memory::EmbeddedCaptureHostPlan<'a>>,
    capture_edits: Option<crate::composition::mlx::session::intervention::PreparedModelInterventions>,
}

type QuotedOutcome<R> = (Result<R, Error>, Option<super::capture::Owner>, Option<String>);

struct Work<'a, 'state, 'context, 'observer, A, S, D, P, F>
where
    S: MlxStateMechanisms,
    A: LayeredArchitecture<MlxNeuralBackend, S, Error = eredu_nn::Error>,
    A::Unit: 'static,
    D: ReplicatedTextExecutionStrategy<
            A,
            MlxNeuralBackend,
            S,
            MlxArchitectureLayerwisePolicy<A, S>,
            MlxArchitectureLayerwisePolicy<A, S>,
        >,
    P: MaterializedPredictionExecutor<A, MlxNeuralBackend, MlxEmbeddedPredictionMaterializer>,
{
    session:
        &'a mut ReplicatedTextSession<A, MlxNeuralBackend, MlxReplicatedTextMechanisms<A, S>, D>,
    extension: &'a mut P,
    lane: PredictionPhaseState<'state, P::LaneState>,
    observer: Option<&'observer mut dyn ActivationObserver<MlxTensor, eredu_nn::Error>>,
    execute: Option<F>,
    context: SpeculativeExecutionStreams<'context>,
    projection: Option<PredictionModuleProjection>,
}

/// Clear the exact installed weak projections on success, refusal and unwind.
struct Installed<'a, A, P>
where
    P: MaterializedPredictionExecutor<A, MlxNeuralBackend, MlxEmbeddedPredictionMaterializer>,
{
    extension: &'a mut P,
    projection: Option<PredictionModuleProjection>,
    architecture: PhantomData<fn() -> A>,
}
impl<A, P> Drop for Installed<'_, A, P>
where
    P: MaterializedPredictionExecutor<A, MlxNeuralBackend, MlxEmbeddedPredictionMaterializer>,
{
    fn drop(&mut self) {
        if let Some(projection) = &self.projection {
            clear_prediction_module_bank::<A, P>(self.extension, projection);
        }
    }
}

pub(super) fn host(funding: &WorkspaceMetadataFunding) -> Result<HostPreparationAuthority, Error> {
    let parts = [
        HostPreparationAuthority::retention_bytes::<WorkspaceMetadataFunding>()
            .ok_or(Error::PrefillControl(WorkingMemoryError::Overflow))?,
        size_of::<HostPreparationAuthority>(),
        size_of::<Result<HostPreparationAuthority, Error>>(),
    ];
    funding
        .reserve_metadata(
            parts
                .into_iter()
                .try_fold(size_of_val(&parts), usize::checked_add)
                .ok_or(Error::PrefillControl(WorkingMemoryError::Overflow))?,
        )
        .map_err(Error::WorkspacePlanning)?;
    Ok(HostPreparationAuthority::retain(funding.clone()))
}

pub(super) fn priors<'a>(
    branch: Option<&'a PreparedEmbeddedEvidence>,
    context: SpeculativeExecutionStreams<'a>,
    funding: &WorkspaceMetadataFunding,
) -> Result<Vec<&'a CompletedResidentSource>, Error> {
    let (sources, environment) = context
        .original_numerical()
        .ok_or(Error::PrefillControl(WorkingMemoryError::IdentityMismatch))?;
    let input = context.embedded_tensor_sources();
    let mut prior = funding
        .metadata_vec(
            input
                .len()
                .checked_add(usize::from(branch.is_some()))
                .ok_or(Error::PrefillControl(WorkingMemoryError::Overflow))?,
        )
        .map_err(Error::from)?;
    for evidence in branch.into_iter().chain(input.iter().copied()) {
        if let Some(source) = completed_tensor_source(evidence) {
            source.validate_request_source(sources.request(), environment.stream(), funding)?;
            prior.push(source);
        } else if let Some(source) =
            crate::composition::mlx::speculative::registered_tensor_source(evidence)
        {
            source.validate(sources.request(), environment.stream(), funding)?;
            // Its backing stays on the canonical registered-copy path.
        } else {
            return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
        }
    }
    Ok(prior)
}

fn width(value: &MlxTensor, sources: &OriginalSpeculativeNumericalSources) -> Result<usize, Error> {
    sources
        .metadata_funding()
        .reserve_metadata(
            Array::descriptor_control_bytes()
                .ok_or(Error::PrefillControl(WorkingMemoryError::Overflow))?,
        )
        .map_err(Error::WorkspacePlanning)?;
    let descriptor = value
        .as_array()
        .try_descriptor()
        .map_err(|cause| sources.retain_startup_error(cause))?;
    let shape = descriptor.shape();
    let [1, positions, ..] = shape else {
        return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
    };
    usize::try_from(*positions)
        .ok()
        .filter(|n| *n != 0)
        .ok_or(Error::PrefillControl(WorkingMemoryError::IdentityMismatch))
}

#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub(super) fn run<'context, 'observer, A, S, D, P, R, F>(
    origin: Option<SpeculativeActivationOrigin>,
    session: &mut ReplicatedTextSession<A, MlxNeuralBackend, MlxReplicatedTextMechanisms<A, S>, D>,
    extension: &mut P,
    mut lane: PredictionPhaseState<'_, P::LaneState>,
    pass: ExpertPass,
    equation: &PredictionEquation<&MlxTensor>,
    completion: Option<PredictionCompletionPoint>,
    span: Option<SpeculativePrefillSpan>,
    context: SpeculativeExecutionStreams<'context>,
    mut observer: Option<&'observer mut dyn ActivationObserver<MlxTensor, eredu_nn::Error>>,
    execute: F,
) -> Result<R, Error>
where
    S: MlxStateMechanisms,
    A: LayeredArchitecture<MlxNeuralBackend, S, Error = eredu_nn::Error> + 'static,
    A::StaticModules: Parameterized<MlxTensor>,
    A::Unit: Parameterized<MlxTensor> + 'static,
    D: ReplicatedTextExecutionStrategy<
            A,
            MlxNeuralBackend,
            S,
            MlxArchitectureLayerwisePolicy<A, S>,
            MlxArchitectureLayerwisePolicy<A, S>,
        >,
    P: MaterializedPredictionExecutor<A, MlxNeuralBackend, MlxEmbeddedPredictionMaterializer>
        + 'static,
    R: PredictionPhaseRoots<MlxTensor, IndependentLogits>,
    F: for<'execution, 'observation> FnOnce(
        &mut ReplicatedTextSession<A, MlxNeuralBackend, MlxReplicatedTextMechanisms<A, S>, D>,
        &mut P,
        &mut P::LaneState,
        Option<&'observation mut dyn ActivationObserver<MlxTensor, eredu_nn::Error>>,
        SpeculativeExecutionStreams<'execution>,
    ) -> Result<R, Error>,
{
    let Some(source) = context.original_embedded() else {
        if context.original_numerical().is_some() {
            return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
        }
        return execute(session, extension, lane.state_mut(), observer, context);
    };
    let _ = pass; // Semantic pass selection stays in the same execute closure.
    let (sources, environment) = context
        .original_numerical()
        .ok_or(Error::PrefillControl(WorkingMemoryError::IdentityMismatch))?;
    let phase_funding = sources.prepare_phase_metadata()?;
    let funding = &phase_funding;
    // Preparation can fail only with a fixed funding refusal. Keep that small
    // result on the cold quote stack; the prepaid native error carrier is born
    // only if a later operation actually fails.
    let phase_source = match super::controls::PhaseSource::new(sources, funding) {
        Ok(source) => source,
        Err(cause) => return Err(Error::WorkspacePlanning(cause)),
    };
    let sources = &phase_source;
    let capture_invocation = observer
        .as_ref()
        .map(|observer| {
            observer
                .original_speculative_capture()
                .ok_or_else(|| sources.retain_startup_error(WorkingMemoryError::UnknownBound))
        })
        .transpose()?;
    let frames = [
        size_of::<Work<'_, '_, '_, '_, A, S, D, P, F>>(),
        size_of::<Payload>(),
        size_of::<QuotedPreparation<'_>>(),
        size_of::<QuotedOutcome<R>>(),
        size_of::<Result<QuotedOutcome<R>, Error>>(),
        size_of::<(&mut Work<'_, '_, '_, '_, A, S, D, P, F>, &PredictionEquation<&MlxTensor>)>(),
        size_of::<Installed<'_, A, P>>(),
        size_of::<Result<R, Error>>(),
        size_of::<R>(),
        size_of::<PredictionEquation<&MlxTensor>>(),
        size_of::<Option<PredictionCompletionPoint>>(),
    ];
    funding
        .reserve_metadata(
            frames
                .into_iter()
                .try_fold(size_of_val(&frames), usize::checked_add)
                .ok_or(Error::PrefillControl(WorkingMemoryError::Overflow))?,
        )
        .map_err(Error::WorkspacePlanning)?;
    let origin = origin.ok_or(Error::PrefillControl(WorkingMemoryError::IdentityMismatch))?;
    let (phase, positions, demand) = match equation {
        PredictionEquation::Prefill { tokens, .. } => (
            SpeculativeActivationPhase::PredictionPrefill,
            width(tokens, sources)?,
            OutputDemand::StateOnly,
        ),
        PredictionEquation::Sequential { depth, .. } => (
            SpeculativeActivationPhase::Proposal { depth: *depth },
            1,
            OutputDemand::LastPosition,
        ),
        PredictionEquation::Fused { capacity, .. } => (
            SpeculativeActivationPhase::FusedProposal,
            *capacity,
            OutputDemand::Sequence,
        ),
        PredictionEquation::Replay { tokens, .. } => (
            SpeculativeActivationPhase::PredictionReplay,
            width(tokens, sources)?,
            OutputDemand::StateOnly,
        ),
    };
    let claim = source.claim(phase, positions, span, origin)?;
    let frontier = extension
        .equation_frontier(lane.state_mut(), equation)
        .map_err(|cause| sources.retain_startup_error(cause))?;
    let workspace = EmbeddedInvocationWorkspace::prediction(claim.invocation(), frontier, demand)
        .map_err(|cause| sources.retain_startup_error(cause))?;
    let mut capture_host = None;
    let capture_edits = std::cell::RefCell::new(None);
    let host = host(funding)?;
    let quote = PreparedPredictionEquationQuote::inspect(
        session,
        extension,
        lane.state(),
        *equation,
        workspace,
        sources,
        environment,
        funding,
        capture_invocation.map(|invocation| {
            super::super::workspace::capture::CaptureWorkspaceInput {
                invocation,
                host: &mut capture_host,
                edits: &capture_edits,
            }
        }),
        |quote_context, storage| {
            crate::composition::mlx::speculative::validate_registered_tensor_inputs(context, storage[2]).map_err(|cause| cause.at_speculative_stage("prediction registered input sources"))?;
            let prior = priors(lane.evidence(), context, funding).map_err(|cause| cause.at_speculative_stage("prediction prior sources"))?;
            SourceBindings::prepare(quote_context, storage, &prior, environment, funding, host)
                .map_err(|cause| sources.retain_error(cause.at_speculative_stage("prediction source binding")))
        },
    ).map_err(|cause| sources.retain_error(cause))?;
    let parts = quote.into_parts();
    let mut work = Work {
        session,
        extension,
        lane,
        observer: None,
        execute: Some(execute),
        context,
        projection: None,
    };
    let (result, capture_owner, capture_identity) = run_quoted::<A, S, D, P, R, F>(
        &mut work,
        sources,
        equation,
        completion,
        QuotedPreparation {
            parts,
            claim,
            workspace,
            capture: capture_invocation,
            capture_host,
            capture_edits: capture_edits.into_inner(),
        },
    ).map_err(|cause| sources.retain_error(cause))?;
    let delivery = match (capture_owner.as_ref(), capture_invocation, capture_identity) {
        (Some(owner), Some(invocation), Some(identity)) => {
            owner.finish(invocation, identity, result.is_ok())
        }
        (None, None, None) => Ok(None),
        _ => Err(sources.retain_startup_error(WorkingMemoryError::IdentityMismatch)),
    };
    let delivery = delivery.and_then(|envelope| match envelope {
        Some(envelope) => capture_owner
            .as_ref()
            .ok_or(Error::PrefillControl(WorkingMemoryError::IdentityMismatch))?
            .receive(
                observer
                    .as_deref_mut()
                    .ok_or(Error::PrefillControl(WorkingMemoryError::IdentityMismatch))?,
                envelope,
            ),
        None => Ok(()),
    });
    match result {
        Ok(output) => {
            delivery.map_err(|cause| sources.retain_error(cause))?;
            Ok(output)
        }
        Err(cause) => {
            drop(delivery);
            Err(sources.retain_error(cause))
        }
    }
}

#[inline(never)]
fn run_quoted<'observer, A, S, D, P, R, F>(
    work: &mut Work<'_, '_, '_, 'observer, A, S, D, P, F>,
    sources: &super::controls::PhaseSource<'_>,
    equation: &PredictionEquation<&MlxTensor>,
    completion: Option<PredictionCompletionPoint>,
    prepared: QuotedPreparation<'_>,
) -> Result<QuotedOutcome<R>, Error>
where
    S: MlxStateMechanisms,
    A: LayeredArchitecture<MlxNeuralBackend, S, Error = eredu_nn::Error> + 'static,
    A::StaticModules: Parameterized<MlxTensor>,
    A::Unit: Parameterized<MlxTensor> + 'static,
    D: ReplicatedTextExecutionStrategy<
            A,
            MlxNeuralBackend,
            S,
            MlxArchitectureLayerwisePolicy<A, S>,
            MlxArchitectureLayerwisePolicy<A, S>,
        >,
    P: MaterializedPredictionExecutor<A, MlxNeuralBackend, MlxEmbeddedPredictionMaterializer>
        + 'static,
    R: PredictionPhaseRoots<MlxTensor, IndependentLogits>,
    F: for<'execution, 'observation> FnOnce(
        &mut ReplicatedTextSession<A, MlxNeuralBackend, MlxReplicatedTextMechanisms<A, S>, D>,
        &mut P,
        &mut P::LaneState,
        Option<&'observation mut dyn ActivationObserver<MlxTensor, eredu_nn::Error>>,
        SpeculativeExecutionStreams<'execution>,
    ) -> Result<R, Error>,
{
    let QuotedPreparation {parts, claim, workspace, capture: capture_invocation, capture_host, capture_edits} = prepared;
    let (_, environment) = work.context.original_numerical()
        .ok_or(Error::PrefillControl(WorkingMemoryError::IdentityMismatch))?;
    // Continue under the account which funded this exact quote.
    let phase_funding = parts.funding.clone();
    let funding = &phase_funding;
    let super::super::workspace::quote::PredictionEquationQuoteParts {
        report,
        capture: capture_population,
        mut recipe,
        target,
        prediction,
        inputs,
        parameters,
        layerwise,
        io,
        context: quote_context,
        bindings,
        funding: quote_funding,
    } = parts;
    let validations = recipe
        .record()
        .validation_roots()
        .ok_or_else(|| Error::PrefillControl(WorkingMemoryError::UnknownBound).at_speculative_stage("prediction validation roots"))?;
    let point_roots = completion
        .map(|point| {
            let outputs = match point {
                PredictionCompletionPoint::CapturedSeed => Some(1),
                PredictionCompletionPoint::ObservedEquation => io.completed_output_root_count(),
                PredictionCompletionPoint::CapturedCarry => None,
            }
            .ok_or(Error::PrefillControl(WorkingMemoryError::IdentityMismatch))?;
            io.completed_state_root_count()
                .and_then(|n| n.checked_add(outputs))
                .ok_or(Error::PrefillControl(WorkingMemoryError::Overflow))
        })
        .transpose()?;
    let completion_capacity = point_roots
        .map(|n| {
            n.checked_add(validations)
                .ok_or(Error::PrefillControl(WorkingMemoryError::Overflow))
        })
        .transpose()?;
    let boundary_completion = completion_capacity.filter(|n| *n != 0);
    let mut module_plan = inspect_prediction_module_plan::<A, P>(
        work.extension,
        &parameters,
        validations,
        environment.stream(),
        sources.pool(),
        &quote_context,
    ).map_err(|cause| cause.at_speculative_stage("prediction module plan"))?;
    let mut roots = if let Some(plan) = &module_plan {
        plan.boundary_roots(boundary_completion).map_err(|cause| cause.at_speculative_stage("prediction boundary roots"))?
    } else {
        let mut roots = quote_context
            .metadata_vec(usize::from(boundary_completion.is_some()))
            .map_err(Error::from)?;
        if let Some(count) = boundary_completion {
            roots.push(count);
        }
        roots
    };
    if !roots.is_empty() {
        recipe.bind_prediction_boundaries(std::mem::take(&mut roots), 0)?;
    }
    let (bank_controls, source_facts) = if let Some(plan) = &mut module_plan {
        plan.bind_source_recipe(&mut recipe, boundary_completion).map_err(|cause| cause.at_speculative_stage("prediction module recipe"))?;
        (
            plan.control_bytes()
                .ok_or_else(|| Error::PrefillControl(WorkingMemoryError::UnknownBound).at_speculative_stage("prediction module control bound"))?,
            plan.source_facts()?,
        )
    } else {
        (0, None)
    };
    let facts = io.input_facts();
    let layout = EmbeddedNativeLayout::inspect(
        &recipe,
        environment,
        facts.map_or(0, |f| f.graph_bytes()),
        facts.map_or(0, |f| f.mutable_bytes()),
    )
    .map_err(|cause| sources.retain_startup_error(cause))?;
    let handlers=(
        |work:&mut Work<'_, '_, '_, 'observer, A,S,D,P,F>,q:&Payload,scope:&safemlx::SubmissionScope| {
            let active=q.modules.as_ref().map(|bank|bank.activate(scope)).transpose()?;
            if let Some(active)=&active {
                let projection=active.projection();
                if let Err(cause)=install_prediction_module_bank::<A,P>(work.extension,&projection) {
                    clear_prediction_module_bank::<A,P>(work.extension,&projection); return Err(cause);
                }
                work.projection=Some(projection);
            }
            Ok(active)
        },
        |work:&mut Work<'_, '_, '_, 'observer, A,S,D,P,F>,q:&Payload,active:&crate::composition::mlx::speculative::embedded_native::ActiveEmbeddedNativeInvocation<'_>| {
            let mut installed=Installed::<A,P>{extension:&mut *work.extension,projection:work.projection.take(),architecture:PhantomData};
            active.begin_equation()?;
            let binding=Binding {active,sources,io:Some(&q.io),logits:q.logits.as_ref(),completion:q.completion.as_ref()};
            let context=work.context.with_embedded_invocation(&binding)?;
            let execute = work.execute.take().ok_or(Error::PrefillControl(WorkingMemoryError::IdentityMismatch))?;
            let output = match &q.capture {
                Some(capture) => capture.run(active, environment.stream(), |observer|
                    execute(work.session, &mut *installed.extension, work.lane.state_mut(), Some(observer), context)),
                None => execute(work.session, &mut *installed.extension, work.lane.state_mut(), work.observer.take(), context),
            }?;
            if let Some(bank)=&q.modules {bank.validate_complete().map_err(|cause|cause.at_speculative_stage("prediction module completion"))?;}
            if let Some(point)=&q.completion {point.try_borrow().map_err(|_|Error::PrefillScopeReentrant)?.validate_complete()?;}
            active.complete(|visit| {
                visit_target(work.session,visit)?;
                installed.extension.with_state_values(work.lane.state_mut(),|values| { for value in values {visit(value.as_array());} });
                output.visit_tensor_roots(&mut |value|visit(value.as_array()));
                output.visit_logits_roots(&mut |value|value.visit_native_roots(visit));
                if let Some(capture) = &q.capture {capture.visit_retained(visit)?;}
                q.io.try_borrow().map_err(|_|Error::PrefillScopeReentrant)?.visit_retained(visit);
                Ok(())
            })?;
            Ok(output)
        },
        |work:&mut Work<'_, '_, '_, 'observer, A,S,D,P,F>,output:&R,budget:&safemlx::OriginalBufferBudget,custody:&eredu_runtime::working_memory::OriginalSpeculativeBudgetCustody| {
            let (state,evidence)=work.lane.completion_sources();
            let prior=priors(evidence,work.context,funding)?;
            let target=work.session.inspect_runtime_execution_fixed(|_,state,_|Ok::<_,Infallible>(state))
                .map_err(|cause|sources.retain_startup_error(cause))?
                .unwrap_or_else(|never|match never {});
            let mut target_complete=true;
            // The state/evidence split permits visiting the actual mutable lane
            // while the previous immutable inventory still supplies old roots.
            let source=CompletedResidentSource::capture_array_sources_with_priors(|visit| {
                target_complete &= target.visit_snapshot_arrays(visit).is_some();
                work.extension.with_state_values(state,|values| {for value in values {visit(value.as_array());}});
                output.visit_tensor_roots(&mut |value|visit(value.as_array()));
                output.visit_logits_roots(&mut |value|value.visit_native_roots(visit));
            },&prior,budget,custody,funding)?.with_completed_stream(environment.stream())?;
            if !target_complete {return Err(Error::PrefillControl(WorkingMemoryError::UnknownBound));}
            drop(prior);
            let mut failure=None;
            output.visit_logits_roots(&mut |value| {if failure.is_none() {failure=value.seal_completed(&source).err();}});
            if let Some(cause)=failure {return Err(cause);}
            if !work.lane.retain_evidence(source).map_err(Error::StorageSource)? {
                return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
            }
            Ok(())
        },
    );
    let capture_controls = capture_invocation
        .map(|invocation| {
            super::capture::Capture::control_bytes(capture_population.retained_roots)
                .and_then(|n| n.checked_add(invocation.envelope_control_bytes()?))
                .and_then(|n| {
                    n.checked_add(size_of::<(
                        F,
                        SpeculativeExecutionStreams<'_>,
                        &mut ReplicatedTextSession<A, MlxNeuralBackend, MlxReplicatedTextMechanisms<A, S>, D>,
                        &mut P,
                        &mut P::LaneState,
                        &mut dyn ActivationObserver<MlxTensor, eredu_nn::Error>,
                    )>())
                })
                .and_then(|n| u64::try_from(n).ok())
                .ok_or_else(|| sources.retain_startup_error(WorkingMemoryError::Overflow))
        })
        .transpose()?
        .unwrap_or(0);
    let controls =
        super::controls::handler_controls::<Payload, _, R, _, _, _, _>(&layout, work, &handlers)
            .and_then(|n| n.checked_add(bank_controls))
            .and_then(|n| n.checked_add(capture_controls));
    let requirements = SpeculativeInvocationRequirements::new(
        recipe.plan(),
        layout.physical_bytes(),
        Some(layout.graph_bytes()),
        Some(layout.record_bytes()),
        controls,
    )
    .map_err(|cause| sources.retain_startup_error(Error::PrefillControl(cause).at_speculative_stage("prediction invocation requirements")))?;
    let requirements = match source_facts {
        Some(facts) => requirements
            .with_host_source_constructions(facts)
            .map_err(|cause| sources.retain_startup_error(cause))?,
        None => requirements,
    };
    let (role, capture_owner, capture_identity) = match (capture_invocation, capture_host) {
        (Some(invocation), Some(host)) => {
            let (role, pending) = sources
                .request()
                .reserve_embedded_role_with_capture(claim, workspace, requirements, host)
                .map_err(|cause| sources.retain_admission_error(cause))?;
            let mut funded = pending
                .begin()
                .map_err(|cause| sources.retain_startup_error(cause))?;
            funded
                .prepare_envelope(invocation)
                .map_err(|cause| sources.retain_startup_error(cause))?;
            let identity = invocation.admission_identity().to_owned();
            (
                role,
                Some(super::capture::Owner::prepare(funded)),
                Some(identity),
            )
        }
        (None, None) => (
            sources
                .request()
                .reserve_embedded_role(claim, workspace, requirements)
                .map_err(|cause| sources.retain_admission_error(cause))?,
            None,
            None,
        ),
        _ => return Err(sources.retain_startup_error(WorkingMemoryError::IdentityMismatch)),
    };
    let capture = capture_owner
        .as_ref()
        .map(|owner| {
            super::capture::Capture::prepare(
                owner.clone(),
                capture_population.retained_roots,
                &role,
                capture_edits,
            )
        })
        .transpose()?;
    let modules = module_plan
        .map(|plan| plan.prepare(role.clone(), environment.stream()).map_err(|cause| cause.at_speculative_stage("prediction module bank construction")))
        .transpose()?;
    if modules.is_none() {
        role.claim_neural_bank(0)
            .map_err(|cause| sources.retain_startup_error(cause))?;
    }
    let point = completion
        .zip(point_roots)
        .map(|(point, roots)| {
            StateCompletion::prepare(roots, validations, point, &role, sources).map(RefCell::new)
        })
        .transpose()?;
    let logits = if matches!(equation, PredictionEquation::Sequential { .. }) {
        Some(PendingModelLogits::prepare(
            &role,
            sources,
            environment.stream(),
        )?)
    } else {
        None
    };
    let io = io.prepare(role.clone(), recipe.plan(), sources).map_err(|cause| cause.at_speculative_stage("prediction input bank construction"))?;
    let payload = Payload {
        capture,
        io: RefCell::new(io),
        logits,
        completion: point,
        modules,
        target,
        prediction,
        inputs,
        parameters,
        layerwise,
        report,
        bindings,
        context: quote_context,
        funding: quote_funding,
    };
    let result = layout.run(
        &recipe,
        environment,
        sources.numerical_prerequisites().0,
        role,
        payload,
        funding.clone(),
        work,
        handlers,
    );
    Ok((result, capture_owner, capture_identity))
}

pub(super) fn visit_target<A, S, D>(
    session: &ReplicatedTextSession<A, MlxNeuralBackend, MlxReplicatedTextMechanisms<A, S>, D>,
    visit: &mut dyn FnMut(&Array),
) -> Result<(), Error>
where
    S: MlxStateMechanisms,
    A: LayeredArchitecture<MlxNeuralBackend, S, Error = eredu_nn::Error>,
    A::Unit: 'static,
    D: ReplicatedTextExecutionStrategy<
            A,
            MlxNeuralBackend,
            S,
            MlxArchitectureLayerwisePolicy<A, S>,
            MlxArchitectureLayerwisePolicy<A, S>,
        >,
{
    session
        .inspect_runtime_execution_fixed(|_, state, _| {
            Ok::<_, Infallible>(state.visit_snapshot_arrays(visit))
        })
        .map_err(|_| Error::PrefillControl(WorkingMemoryError::IdentityMismatch))?
        .unwrap_or_else(|never| match never {})
        .ok_or(Error::PrefillControl(WorkingMemoryError::UnknownBound))
}
