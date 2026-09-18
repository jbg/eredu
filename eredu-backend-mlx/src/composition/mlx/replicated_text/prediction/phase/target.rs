//! Target equations run through the existing shared session and native banks.
#[derive(Debug, thiserror::Error)]
#[error("embedded target operation {operation} has no native workspace bound: {detail}")]
struct MissingNativeEquation {
    operation: usize,
    detail: String,
    #[source]
    cause: eredu_runtime::working_memory::WorkingMemoryError,
}

use super::super::workspace::source_bindings::SourceBindings;
use super::super::workspace::target_quote::{
    PreparedTargetEquationQuote, TargetEquationQuoteParts,
};
use super::*;
use crate::backend::submission_recovery::prefill::nested::NestedCompletionOwner;
use crate::backend::{
    OriginalCopyEnvironment,
    nn::workspace::{ProjectedNativeStorage, ProjectedResidentState},
    runtime::{
        cache::state::{
            CompletedResidentSource,
        },
        execution::generic::{LayerwiseWorkspace, SpeculativeNeuralOwner},
    },
};
use crate::composition::mlx::speculative::embedded_native::{
    ActiveEmbeddedNativeInvocation, EmbeddedNativeLayout,
};
use eredu_core::{HostPreparationAuthority, OutputDemand};
use eredu_nn::workspace::{WorkspaceContext, HostMetadataFunding};
use eredu_runtime::{
    speculative::embedded_occurrence::EmbeddedInvocationWorkspace,
    working_memory::{
        InferenceWorkspaceReport, OriginalEmbeddedSpeculativeRole,
        OriginalSpeculativeBudgetCustody, SpeculativeInvocationRequirements, WorkingMemoryError,
    },
};
use safemlx::{Array, OriginalBufferBudget, SubmissionScope};
use std::mem::{size_of, size_of_val};
use crate::composition::mlx::speculative::embedded_native::AddressableModelSource;

type Session<A, S, D> =
    ReplicatedTextSession<A, MlxNeuralBackend, MlxReplicatedTextMechanisms<A, S>, D>;
type Work<'s, 'e, 'o, A, S, D, F> = (
    &'s mut Session<A, S, D>,
    PredictionPhaseEvidence<'e, S>,
    Option<&'o mut dyn ActivationObserver<MlxTensor, eredu_nn::Error>>,
    Option<F>,
    Option<OriginalEmbeddedSpeculativeRole>,
);

/// Both actual native projections and validated source custody survive recovery.
/// The recipe is lent independently to the same native layout and bank compiler.
struct Payload {
    addressable:Option<AddressableModelSource>,
    completion: Option<NestedCompletionOwner>,
    capture: Option<super::capture::Capture>,
    _report: InferenceWorkspaceReport,
    _state: ProjectedResidentState,
    _native_inputs: ProjectedNativeStorage,
    layerwise: Option<LayerwiseWorkspace>,
    _context: WorkspaceContext,
    _bindings: SourceBindings,
    _funding: HostMetadataFunding,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn run<'context, 'observer, A, S, D, R, F>(
    origin: Option<SpeculativeActivationOrigin>,
    session: &mut Session<A, S, D>,
    evidence: PredictionPhaseEvidence<'_, S>,
    tokens: Option<&MlxTensor>,
    phase: SpeculativeActivationPhase,
    demand: OutputDemand,
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
    R: PredictionPhaseRoots<MlxTensor, IndependentLogits>,
    F: for<'execution, 'observation> FnOnce(
        &mut Session<A, S, D>,
        Option<&'observation mut dyn ActivationObserver<MlxTensor, eredu_nn::Error>>,
        SpeculativeExecutionStreams<'execution>,
    ) -> Result<R, Error>,
{
    let Some(selected) = context.original_embedded() else {
        if context.original_numerical().is_some() {
            return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
        }
        return execute(session, observer, context);
    };
    let sources = selected.numerical_sources();
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
    let (_, environment) = context
        .original_numerical()
        .ok_or_else(|| sources.retain_startup_error(WorkingMemoryError::IdentityMismatch))?;
    let origin =
        origin.ok_or_else(|| sources.retain_startup_error(WorkingMemoryError::IdentityMismatch))?;
    let tokens =
        tokens.ok_or_else(|| sources.retain_startup_error(WorkingMemoryError::IdentityMismatch))?;
    let capture_invocation = observer
        .as_ref()
        .map(|observer| {
            observer
                .original_speculative_capture()
                .ok_or_else(|| sources.retain_startup_error(WorkingMemoryError::UnknownBound))
        })
        .transpose()?;
    funding
        .reserve_metadata(
            Array::descriptor_control_bytes()
                .ok_or_else(|| sources.retain_startup_error(WorkingMemoryError::Overflow))?,
        )
        .map_err(Error::WorkspacePlanning)?;
    let positions = {
        let descriptor = tokens
            .as_array()
            .try_descriptor()
            .map_err(|cause| sources.retain_startup_error(cause))?;
        match descriptor.shape() {
            [1, n] if *n > 0 => {
                usize::try_from(*n).map_err(|cause| sources.retain_startup_error(cause))?
            }
            _ => return Err(sources.retain_startup_error(WorkingMemoryError::IdentityMismatch)),
        }
    };
    let claim = selected.claim(phase, positions, span, origin)?;
    let workspace = EmbeddedInvocationWorkspace::target_with_readout(claim.invocation(), demand)
        .map_err(|cause| sources.retain_startup_error(cause))?;
    let mut capture_host = None;
    let capture_edits = std::cell::RefCell::new(None);
    let host = super::prediction::host(funding)?;
    let mut parts = PreparedTargetEquationQuote::inspect(
        session,
        tokens,
        context.original_prefill_input(),
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
        |quote_context, storage, prepared| {
            crate::composition::mlx::speculative::validate_registered_tensor_inputs(context, storage[1])?;
            let prior = super::prediction::priors(evidence.evidence(), context, funding)?;
            SourceBindings::prepare(quote_context, storage, prepared, &prior, environment, funding, host)
                .map_err(|cause| sources.retain_error(cause))
        },
    ).map_err(|cause| sources.retain_error(cause))?
    .into_parts();
    let frames = [
        size_of::<Payload>(),
        size_of::<Work<'_, '_, 'observer, A, S, D, F>>(),
        size_of::<Result<R, Error>>(),
        size_of::<Option<R>>(),
        size_of::<EmbeddedNativeLayout>(),
        size_of::<EmbeddedInvocationWorkspace>(),
        size_of::<Option<OriginalEmbeddedSpeculativeRole>>(),
        size_of::<binding::Binding<'_>>(),
        size_of::<safemlx::StreamCopyPlan<HostMetadataFunding>>(),
        size_of::<Result<safemlx::StreamCopyPlan<HostMetadataFunding>,safemlx::StreamCopyCause>>(),
    ];
    funding
        .reserve_metadata(
            frames
                .into_iter()
                .try_fold(size_of_val(&frames), usize::checked_add)
                .ok_or_else(|| sources.retain_startup_error(WorkingMemoryError::Overflow))?,
        )
        .map_err(Error::WorkspacePlanning)?;
    if let Some((operation, detail)) = parts.recipe.take_missing_operation_detail() {
        return Err(sources.retain_startup_error(MissingNativeEquation {
            operation,
            detail,
            cause: WorkingMemoryError::UnknownBound,
        }));
    }
    let completion_roots=parts.completion_roots;
    let validation_roots=parts.recipe.record().validation_roots()
        .ok_or_else(||sources.retain_startup_error(WorkingMemoryError::UnknownBound))?;
    let completion_capacity=completion_roots.checked_add(validation_roots)
        .filter(|n|*n!=0).ok_or_else(||sources.retain_startup_error(WorkingMemoryError::UnknownBound))?;
    let mut completion_distribution=funding.metadata_vec(1).map_err(Error::from)?;
    completion_distribution.push(completion_capacity);
    parts.recipe.bind_target_completion(completion_distribution).map_err(|cause|sources.retain_error(cause))?;
    let completion_stream=safemlx::StreamCopyPlan::<HostMetadataFunding>::capture(environment.stream())
        .map_err(|cause|sources.retain_startup_error(cause))?;
    let completion_controls=NestedCompletionOwner::control_bytes(completion_roots,validation_roots,&completion_stream)
        .and_then(|n|n.checked_add(size_of::<(Option<&MlxTensor>,&S,
            Option<&eredu_runtime::media_prefill::RetainedMediaRoots<'_,MlxTensor>>,
            &Stream,&mut dyn FnMut(&MlxTensor))>()))
        .and_then(|n|u64::try_from(n).ok())
        .ok_or_else(||sources.retain_startup_error(WorkingMemoryError::Overflow))?;
    let (bank_controls, source_facts) =
        MlxReplicatedTextMechanisms::<A, S>::bind_session_embedded_neural_recipe(
            session,
            parts.layerwise.as_ref(),
            environment.pool(),
            funding,
            &mut parts.recipe,
        ).map_err(|cause|sources.retain_error(cause))?;
    // Tokens were produced and admitted by their exact input/numerical source.
    // This target invocation creates no input upload or auxiliary input graph.
    let (addressable,source_facts,addressable_controls)=AddressableModelSource::prepare(
        parts.recipe.native_recipe().records(),source_facts,funding)?;
    let layout = EmbeddedNativeLayout::inspect(&parts.recipe, environment, 0, 0)
        .map_err(|cause| sources.retain_startup_error(cause))?;
    let TargetEquationQuoteParts {
        report,
        capture: capture_population,
        completion_roots: _,
        recipe,
        target: state,
        inputs: native_inputs,
        layerwise,
        context: workspace_context,
        bindings,
        funding: quote_funding,
    } = parts;
    let mut payload = Payload {
        addressable,
        completion: None,
        capture: None,
        _report: report,
        _state: state,
        _native_inputs: native_inputs,
        layerwise,
        _context: workspace_context,
        _bindings: bindings,
        _funding: quote_funding,
    };
    let mut work = (session, evidence, None, Some(execute), None);
    let bind = |work: &mut Work<'_, '_, 'observer, A, S, D, F>,
                payload: &Payload,
                scope: &SubmissionScope| {
        let mut partition=|root|payload.addressable.as_ref().ok_or(Error::PrefillScopeUnavailable)?.accept(root);
        let bank=MlxReplicatedTextMechanisms::<A, S>::prepare_session_embedded_neural_bank(
            work.0,
            payload.layerwise.as_ref(),
            &recipe,
            work.4
                .as_ref()
                .ok_or(Error::PrefillScopeUnavailable)?
                .clone(),
            scope,payload.addressable.is_some().then_some(&mut partition),
        )?;
        if let Some(source)=payload.addressable.as_ref(){source.bind(bank.as_ref().ok_or(Error::PrefillScopeUnavailable)?)?;}
        Ok(bank)
    };
    let execute_handler = |work: &mut Work<'_, '_, 'observer, A, S, D, F>,
               payload: &Payload,
               active: &ActiveEmbeddedNativeInvocation<'_>| {
        active.begin_equation()?;
        let completion=payload.completion.as_ref().ok_or(Error::PrefillScopeUnavailable)?;
        let (projection, activation)=completion.activate(active.role(),active.observer(),environment.stream())?;
        MlxReplicatedTextMechanisms::<A,S>::install_session_nested_completion(work.0,projection)?;
        let binding = binding::Binding {
            active,
            sources,
            io: None,
            logits: None,
            completion: None,
        };
        let execution = context.with_embedded_invocation(&binding)?;
        let execute = work.3.take().ok_or(Error::PrefillScopeReentrant)?;
        let output = match &payload.capture {
            Some(capture) => capture.run(active, environment.stream(), |observer| {
                execute(work.0, Some(observer), execution)
            }),
            None => execute(work.0, work.2.take(), execution),
        };
        drop(activation);
        let output=output?;
        completion.validate_complete()?;
        active.complete(|visitor| {
            visit_roots(work.0, &output, visitor)?;
            if let Some(capture) = &payload.capture {
                capture.visit_retained(visitor)?;
            }
            Ok(())
        })?;
        Ok(output)
    };
    let addressable_callbacks=AddressableModelSource::callback_controls::<_,Payload,_,R,_>(&execute_handler)
        .ok_or_else(||sources.retain_startup_error(WorkingMemoryError::Overflow))?;
    let run=|work:&mut Work<'_, '_, 'observer,A,S,D,F>,payload:&Payload,active:&ActiveEmbeddedNativeInvocation<'_>| {
        match payload.addressable.as_ref(){
            Some(source)=>source.run(work,payload,active,execute_handler),
            None=>execute_handler(work,payload,active),
        }
    };
    let publish = |work: &mut Work<'_, '_, 'observer, A, S, D, F>,
                   output: &R,
                   budget: &OriginalBufferBudget,
                   custody: &OriginalSpeculativeBudgetCustody| {
        let prior = super::prediction::priors(work.1.evidence(), context, funding)?;
        let mut visit_error = None;
        let completed = CompletedResidentSource::capture_array_sources_with_priors(
            |visitor| {
                if visit_error.is_none() {
                    visit_error = visit_roots(work.0, output, visitor).err();
                }
            },
            &prior,
            budget,
            custody,
            funding,
        )?
        .with_completed_stream(environment.stream())?;
        drop(prior);
        if let Some(cause) = visit_error {
            return Err(cause);
        }
        let mut seal_error = None;
        output.visit_logits_roots(&mut |value| {
            if seal_error.is_none() {
                seal_error = value.seal_completed(&completed).err();
            }
        });
        if let Some(cause) = seal_error {
            return Err(cause);
        }
        if !work
            .1
            .retain_evidence(completed)
            .map_err(Error::StorageSource)?
        {
            return Err(sources.retain_startup_error(WorkingMemoryError::IdentityMismatch));
        }
        Ok(())
    };
    let handlers = (bind, run, publish);
    let capture_controls = capture_invocation
        .map(|invocation| {
            super::capture::Capture::control_bytes(capture_population.retained_roots)
                .and_then(|n| n.checked_add(invocation.envelope_control_bytes()?))
                .and_then(|n| {
                    n.checked_add(size_of::<(
                        F,
                        SpeculativeExecutionStreams<'_>,
                        &mut ReplicatedTextSession<A, MlxNeuralBackend, MlxReplicatedTextMechanisms<A, S>, D>,
                        &mut dyn ActivationObserver<MlxTensor, eredu_nn::Error>,
                    )>())
                })
                .and_then(|n| u64::try_from(n).ok())
                .ok_or_else(|| sources.retain_startup_error(WorkingMemoryError::Overflow))
        })
        .transpose()?
        .unwrap_or(0);
    let controls = super::controls::handler_controls::<
        Payload,
        _,
        R,
        Option<SpeculativeNeuralOwner>,
        _,
        _,
        _,
    >(&layout, &work, &handlers)
    .and_then(|n| n.checked_add(bank_controls))
    .and_then(|n| n.checked_add(completion_controls))
    .and_then(|n| n.checked_add(capture_controls))
    .and_then(|n|n.checked_add(addressable_controls))
    .and_then(|n|n.checked_add(addressable_callbacks));
    let requirements = SpeculativeInvocationRequirements::new(
        recipe.plan(),
        layout.physical_bytes(),
        Some(layout.graph_bytes()),
        Some(layout.record_bytes()),
        controls,
    )
    .map_err(|cause| sources.retain_startup_error(cause))?;
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
    payload.completion=Some(NestedCompletionOwner::prepare(completion_roots,validation_roots,
        &role,funding,completion_stream).map_err(|cause|sources.retain_error(cause))?);
    payload.capture = capture_owner
        .as_ref()
        .map(|owner| {
            super::capture::Capture::prepare(
                owner.clone(),
                capture_population.retained_roots,
                &role,
                capture_edits.into_inner(),
            )
        })
        .transpose().map_err(|cause|sources.retain_error(cause))?;
    work.4 = Some(role.clone());
    let (roots, _) = sources.numerical_prerequisites();
    let result = layout.run(
        &recipe,
        environment,
        roots,
        role,
        payload,
        funding.clone(),
        &mut work,
        handlers,
    );
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

fn visit_roots<A, S, D, R>(
    session: &Session<A, S, D>,
    output: &R,
    visitor: &mut dyn FnMut(&Array),
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
    R: PredictionPhaseRoots<MlxTensor, IndependentLogits>,
{
    super::prediction::visit_target(session, visitor)?;
    output.visit_tensor_roots(&mut |value| visitor(value.as_array()));
    output.visit_logits_roots(&mut |value| value.visit_native_roots(visitor));
    Ok(())
}

use eredu_nn::workspace::WorkspaceMetadataAllocation;
