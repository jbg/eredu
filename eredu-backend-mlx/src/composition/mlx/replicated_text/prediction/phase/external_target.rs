//! External target equations use the same model scope and session banks.
use super::super::workspace::{
    external_target::ExternalTargetEquationQuote, source_bindings::SourceBindings,
};
use super::*;
use crate::backend::{
    nn::workspace::{ProjectedNativeStorage, ProjectedResidentState},
    runtime::{
        cache::state::CompletedResidentSource,
        execution::generic::{LayerwiseWorkspace, SpeculativeNeuralOwner},
    },
    submission_recovery::prefill::nested::NestedCompletionOwner,
};
use crate::composition::mlx::speculative::{
    embedded_native::{ActiveEmbeddedNativeInvocation, EmbeddedNativeLayout, ModelEquation},
    retain_external_evidence_for_placement, validate_input_evidence, completed_tensor_source,
};
use eredu_architectures::{
    composite_execution::PreparedCompositeArchitecture,
    composite_execution::{
        CompositeArchitecture, ExternalPredictionCaptureRequest, ExternalPredictionTargetCapture,
    },
    speculative_execution::PreparedEmbeddedEvidence,
};
use eredu_core::{InferenceGeometry, OutputDemand, SamplingPlacement};
use eredu_nn::{
    workspace::{WorkspaceContext, WorkspaceMetadataFunding},
    Tensor,
};
use eredu_runtime::{
    speculative::external_occurrence::ExternalInvocationKind,
    working_memory::{
        InferenceWorkspaceReport, OriginalExternalSpeculativeRole,
        OriginalSpeculativeBudgetCustody, SpeculativeInvocationRequirements, WorkingMemoryError,
    },
    ReplicatedTextSessionMechanisms,
};
use safemlx::{Array, OriginalBufferBudget, SubmissionScope};
use std::mem::{size_of, size_of_val};
use crate::composition::mlx::speculative::embedded_native::AddressableModelSource;
mod capture;
pub(crate) use capture::Capture;

type Architecture<A> = PreparedCompositeArchitecture<A>;
type Mechanisms<A> = MlxReplicatedTextMechanisms<Architecture<A>, MlxHybridState>;
type Session<A, D> = ReplicatedTextSession<Architecture<A>, MlxNeuralBackend, Mechanisms<A>, D>;

/// Exact output of the ordinary whole-input or physical-span target worker.
pub(crate) struct Output {
    pub scores: Option<MlxTensor>,
    pub capture: Option<ExternalPredictionTargetCapture<MlxTensor>>,
    pub frontier: u64,
    pub commit: Option<eredu_core::DistributedCommitOutcome>,
    pub evidence: Option<PreparedEmbeddedEvidence>,
}
impl Output {
    pub(crate) fn into_tensor(self,context:SpeculativeExecutionStreams<'_>)
        ->Result<eredu_architectures::speculative_execution::EmbeddedPredictionTensor<MlxTensor>,Error>{
        use eredu_architectures::speculative_execution::EmbeddedPredictionTensor as Packet;
        let value=self.scores.ok_or(Error::PrefillScopeUnavailable)?;
        match context.original_numerical(){
            Some((sources,_))=>{
                let funding=sources.metadata_funding();
                let bytes=Packet::<MlxTensor>::retained_control_bytes()
                    .and_then(|n|n.checked_add(size_of::<Result<Packet<MlxTensor>,Error>>()))
                    .ok_or_else(||sources.retain_startup_error(WorkingMemoryError::Overflow))?;
                funding.reserve_metadata(bytes).map_err(Error::WorkspacePlanning)?;
                let host=super::prediction::host(funding)?;
                let evidence=self.evidence.ok_or(Error::PrefillScopeUnavailable)?;
                Ok(Packet::from_prepared(value,evidence,host))
            }
            None=>Ok(Packet::ordinary(value)),
        }
    }
    fn visit(&self, visit: &mut dyn FnMut(&Array)) {
        if let Some(scores) = &self.scores {
            visit(scores.as_array());
        }
        if let Some(capture)=&self.capture { capture.visit_values(&mut |v| visit(v.as_array())); }
    }
}
struct Payload {
    addressable:Option<AddressableModelSource>,
    completion: Option<NestedCompletionOwner>,
    capture: Option<Capture>,
    _report: InferenceWorkspaceReport,
    _state: ProjectedResidentState,
    _inputs: ProjectedNativeStorage,
    layerwise: Option<LayerwiseWorkspace>,
    context: WorkspaceContext,
    _bindings: SourceBindings,
    _funding: WorkspaceMetadataFunding,
    _prior: Vec<PreparedEmbeddedEvidence>,
    _state_prior: Option<CompletedResidentSource>,
}
type Work<'a, A, D, F> = (
    &'a mut Session<A, D>,
    Option<F>,
    Option<OriginalExternalSpeculativeRole>,
    &'a mut Option<CompletedResidentSource>,
    Option<PreparedEmbeddedEvidence>,
);

fn controls<W, B, F, G, H>(
    layout: &EmbeddedNativeLayout,
    _: &W,
    handlers: &(F, G, H),
) -> Option<u64>
where
    F: FnOnce(&mut W, &Payload, &SubmissionScope) -> Result<B, Error>,
    G: FnOnce(
        &mut W,
        &Payload,
        &ActiveEmbeddedNativeInvocation<'_, OriginalExternalSpeculativeRole>,
    ) -> Result<Output, Error>,
    H: FnOnce(
        &mut W,
        &Output,
        &OriginalBufferBudget,
        &OriginalSpeculativeBudgetCustody,
    ) -> Result<(), Error>,
{
    layout
        .control_bytes_for_role::<OriginalExternalSpeculativeRole, Payload, W, Output, B, F, G, H>(
            handlers,
        )
}

fn prior_sources<'a>(
    state: Option<&'a CompletedResidentSource>,
    context: SpeculativeExecutionStreams<'a>,
    funding: &WorkspaceMetadataFunding,
) -> Result<Vec<&'a CompletedResidentSource>, Error> {
    let (sources, environment) = context
        .original_numerical_for(SamplingPlacement::Target)
        .ok_or(Error::PrefillControl(WorkingMemoryError::IdentityMismatch))?;
    let frames=[size_of::<(Option<&CompletedResidentSource>,SpeculativeExecutionStreams<'_>,&WorkspaceMetadataFunding)>(),
        size_of::<Vec<&CompletedResidentSource>>(),size_of::<Result<Vec<&CompletedResidentSource>,Error>>(),
        size_of::<std::slice::Iter<'_,&PreparedEmbeddedEvidence>>(),size_of::<Option<&CompletedResidentSource>>(),
        size_of::<Result<(),Error>>()];
    funding.reserve_metadata(frames.into_iter().try_fold(size_of_val(&frames),usize::checked_add)
        .ok_or(Error::PrefillControl(WorkingMemoryError::Overflow))?).map_err(Error::WorkspacePlanning)?;
    let inputs=context.embedded_tensor_sources();
    let mut prior=funding.metadata_vec(inputs.len().checked_add(usize::from(state.is_some()))
        .ok_or(Error::PrefillControl(WorkingMemoryError::Overflow))?)?;
    for evidence in inputs {
        validate_input_evidence(evidence,context,SamplingPlacement::Target,funding)?;
        if let Some(source)=completed_tensor_source(evidence){prior.push(source);}
    }
    // Mutable target state remains qualified by this target's own completed
    // marker. Only immutable input evidence may retain an opposite-side origin.
    if let Some(state)=state {
        state.validate_request_source(sources.request(),environment.stream(),funding)?;
        prior.push(state);
    }
    Ok(prior)
}

#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub(crate) fn run<A, D, F>(
    session: &mut Session<A, D>,
    tokens: &MlxTensor,
    request: Option<&ExternalPredictionCaptureRequest>,
    kind: ExternalInvocationKind,
    span: Option<SpeculativePrefillSpan>,
    demand: OutputDemand,
    context: SpeculativeExecutionStreams<'_>,
    state_prior: &mut Option<CompletedResidentSource>,
    execute: F,
) -> Result<Output, Error>
where
    A: CompositeArchitecture<MlxNeuralBackend, MlxHybridState, Error = eredu_nn::Error> + 'static,
    A::InputPartPlan: 'static,
    A::StaticModules: Parameterized<MlxTensor>,
    A::Unit: Parameterized<MlxTensor> + 'static,
    D: ReplicatedTextExecutionStrategy<
        Architecture<A>,
        MlxNeuralBackend,
        MlxHybridState,
        MlxArchitectureLayerwisePolicy<Architecture<A>, MlxHybridState>,
        MlxArchitectureLayerwisePolicy<Architecture<A>, MlxHybridState>,
    >,
    F: FnOnce(&mut Session<A, D>, Option<&Capture>, Option<&WorkspaceContext>) -> Result<Output, Error>,
{
    let Some(selected) = context.original_external() else {
        if context.original_numerical().is_some() {
            return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
        }
        let capture = request.map(|r|Capture::prepare::<A>(r, None)).transpose()?;
        return execute(session, capture.as_ref(), None);
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
    let origin = context
        .external_origin()
        .ok_or_else(|| sources.retain_startup_error(WorkingMemoryError::IdentityMismatch))?;
    let frames = [
        // The cold quote returns before the native continuation creates its
        // bank/recovery temporaries. Price that exact additional call loan.
        size_of::<(&mut Session<A, D>, Option<&ExternalPredictionCaptureRequest>,
            SpeculativeExecutionStreams<'_>, &mut Option<CompletedResidentSource>, F,
            &super::controls::PhaseSource<'_>,
            eredu_runtime::speculative::external_occurrence::ExternalOccurrenceClaim<'_>,
            ExternalTargetEquationQuote)>(),
        size_of::<Result<Output, Error>>(),
        size_of::<WorkspaceMetadataFunding>(),
        size_of::<&crate::backend::OriginalCopyEnvironment<'_>>(),
        size_of::<Payload>(),
        size_of::<Work<'_, A, D, F>>(),
        size_of::<Output>(),
        size_of::<Result<Output, Error>>(),
        size_of::<InferenceGeometry>(),
        size_of::<ExternalTargetEquationQuote>(),
        size_of::<Result<ExternalTargetEquationQuote, Error>>(),
        size_of::<Option<CompletedResidentSource>>(),
        size_of::<Vec<&CompletedResidentSource>>(),
        size_of::<Vec<PreparedEmbeddedEvidence>>(),
        size_of::<EmbeddedNativeLayout>(),
        size_of::<safemlx::StreamCopyPlan<WorkspaceMetadataFunding>>(),
        size_of::<
            Result<safemlx::StreamCopyPlan<WorkspaceMetadataFunding>, safemlx::StreamCopyCause>,
        >(),
        Array::descriptor_control_bytes()
            .ok_or_else(|| sources.retain_startup_error(WorkingMemoryError::Overflow))?,
    ];
    funding
        .reserve_metadata(
            frames
                .into_iter()
                .try_fold(size_of_val(&frames), usize::checked_add)
                .ok_or_else(|| sources.retain_startup_error(WorkingMemoryError::Overflow))?,
        )
        .map_err(Error::WorkspacePlanning)?;
    let positions = {
        let descriptor = tokens.as_array().try_descriptor().map_err(|cause|sources.retain_startup_error(cause))?;
        match descriptor.shape() {
            [1, n, ..] if *n > 0 => u64::try_from(*n).map_err(|e| sources.retain_startup_error(e))?,
            _ => return Err(sources.retain_startup_error(WorkingMemoryError::IdentityMismatch)),
        }
    };
    let frontier = if request.is_some() { session
        .inspect_runtime_execution_fixed(|m, s, _| m.original_prefill_state_frontier(s))
        .map_err(|e| sources.retain_startup_error(e))?
        .map_err(|e| sources.retain_startup_error(e))?
        .ok_or_else(|| sources.retain_startup_error(WorkingMemoryError::IdentityMismatch))? } else {0};
    let geometry = InferenceGeometry {
        batch_size: 1,
        cached_positions: frontier,
        input_positions: positions,
        max_output_tokens: 0,
        prefill_chunk_positions: positions,
        output: demand,
    };
    let claim = selected.claim(kind, geometry, span, origin)?;
    let invocation = claim.invocation();
    let host = super::prediction::host(funding)?;
    let parts = ExternalTargetEquationQuote::inspect(
        session,
        tokens,
        invocation,
        request,
        sources,
        environment,
        funding,
        |quote, storage| {
            crate::composition::mlx::speculative::validate_registered_tensor_inputs_at(
                context, storage[1], SamplingPlacement::Target, funding,
            ).map_err(|cause| sources.retain_error(cause))?;
            let prior = prior_sources(state_prior.as_ref(), context, funding)?;
            SourceBindings::prepare(quote, storage, &prior, environment, funding, host)
        },
    )?;
    run_quoted(session, request, context, state_prior, execute, sources, claim, parts)
}

// The same quote and move-only occurrence cross into the existing native
// worker only after cold family construction has returned.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn run_quoted<A, D, F>(
    session: &mut Session<A, D>,
    request: Option<&ExternalPredictionCaptureRequest>,
    context: SpeculativeExecutionStreams<'_>,
    state_prior: &mut Option<CompletedResidentSource>,
    execute: F,
    sources: &super::controls::PhaseSource<'_>,
    claim: eredu_runtime::speculative::external_occurrence::ExternalOccurrenceClaim<'_>,
    mut parts: ExternalTargetEquationQuote,
) -> Result<Output, Error>
where
    A: CompositeArchitecture<MlxNeuralBackend, MlxHybridState, Error = eredu_nn::Error> + 'static,
    A::InputPartPlan: 'static,
    A::StaticModules: Parameterized<MlxTensor>,
    A::Unit: Parameterized<MlxTensor> + 'static,
    D: ReplicatedTextExecutionStrategy<
        Architecture<A>,
        MlxNeuralBackend,
        MlxHybridState,
        MlxArchitectureLayerwisePolicy<Architecture<A>, MlxHybridState>,
        MlxArchitectureLayerwisePolicy<Architecture<A>, MlxHybridState>,
    >,
    F: FnOnce(&mut Session<A, D>, Option<&Capture>, Option<&WorkspaceContext>) -> Result<Output, Error>,
{
    let (_, environment) = context.original_numerical()
        .ok_or_else(|| sources.retain_startup_error(WorkingMemoryError::IdentityMismatch))?;
    let phase_funding = parts.funding.clone();
    let funding = &phase_funding;
    let invocation = claim.invocation();
    let completion_roots = parts.completion_roots;
    let validation_roots = parts
        .recipe
        .record()
        .validation_roots()
        .ok_or_else(|| {
            if std::env::var_os("EREDU_EXTERNAL_FLOW_TRACE").is_some() {
                eprintln!("EXTERNAL_TARGET_MISSING operation={:?} kind={:?} detail={:?}",
                    parts.recipe.record().first_missing_operation(), invocation.kind(),
                    parts.recipe.record().missing_operation_detail());
            }
            sources.retain_startup_error(WorkingMemoryError::UnknownBound)
        })?;
    if request.is_some() {
    let mut distribution = funding.metadata_vec(1)?;
    distribution.push(
        completion_roots
            .checked_add(validation_roots)
            .filter(|n| *n > 0)
            .ok_or_else(|| sources.retain_startup_error(WorkingMemoryError::UnknownBound))?,
    );
    parts
        .recipe
        .with_native_recipe(|r| r.bind_external_target_completion(invocation, distribution)).map_err(|cause|sources.retain_error(cause))?;
    }
    let completion_stream =
        safemlx::StreamCopyPlan::<WorkspaceMetadataFunding>::capture(environment.stream())
            .map_err(|e| sources.retain_startup_error(e))?;
    let completion_controls = if request.is_some() { NestedCompletionOwner::control_bytes(
        completion_roots,
        validation_roots,
        &completion_stream,
    )
    .and_then(|n| u64::try_from(n).ok())
    .ok_or_else(|| sources.retain_startup_error(WorkingMemoryError::Overflow))? } else {0};
    let (bank_controls, source_facts) = parts.recipe.with_native_recipe(|r| {
        Mechanisms::<A>::bind_session_external_neural_recipe(
            session,
            parts.layerwise.as_ref(),
            environment.pool(),
            funding,
            r,
        )
    }).map_err(|cause|sources.retain_error(cause))?;
    let (addressable,source_facts,addressable_controls)=AddressableModelSource::prepare(
        parts.recipe.native_recipe().records(),source_facts,funding)?;
    let layout = EmbeddedNativeLayout::inspect(&parts.recipe, environment, 0, 0)
        .map_err(|e| sources.retain_startup_error(e))?;
    let capture = request.map(|r|Capture::prepare::<A>(r, Some(&parts.context))).transpose().map_err(|cause|sources.retain_startup_error(cause))?;
    let mut prior = funding.metadata_vec(context.embedded_tensor_sources().len())?;
    prior.extend(
        context
            .embedded_tensor_sources()
            .iter()
            .map(|e| (*e).clone()),
    );
    let ExternalTargetEquationQuote {
        report,
        recipe,
        completion_roots: _,
        state,
        inputs,
        layerwise,
        bindings,
        context: quote,
        funding: quote_funding,
    } = parts;
    let mut payload = Payload {
        addressable,
        completion: None,
        capture,
        _report: report,
        _state: state,
        _inputs: inputs,
        layerwise,
        context: quote,
        _bindings: bindings,
        _funding: quote_funding,
        _prior: prior,
        _state_prior: state_prior.as_ref().map(|source| source.try_clone_for_source(environment.stream(),funding)).transpose().map_err(|cause|sources.retain_error(cause))?,
    };
    let mut work = (session, Some(execute), None, state_prior, None);
    let execute_handler = |work: &mut Work<'_, A, D, F>,
         q: &Payload,
         active: &ActiveEmbeddedNativeInvocation<'_, OriginalExternalSpeculativeRole>| {
            active.begin_equation()?;
            let activation=if let Some(completion)=q.completion.as_ref(){
                let (projection,activation)=completion.activate_external(
                    active.role(),active.observer(),environment.stream())?;
                Mechanisms::<A>::install_session_nested_completion(work.0,projection)?;
                Some(activation)
            }else{None};
            let execute=work.1.take().ok_or(Error::PrefillScopeReentrant)?;
            let output=execute(work.0,q.capture.as_ref(),Some(&q.context));
            drop(activation);
            let output=output?;
            if let Some(completion)=q.completion.as_ref(){completion.validate_complete()?;}
            active.complete(|visit| {
                super::prediction::visit_target(work.0, visit)?;
                output.visit(visit);
                Ok(())
            })?;
            Ok(output)
        };
    let addressable_callbacks=AddressableModelSource::callback_controls::<_,Payload,_,Output,_>(&execute_handler)
        .ok_or_else(||sources.retain_startup_error(WorkingMemoryError::Overflow))?;
    let handlers = (
        |work: &mut Work<'_, A, D, F>, q: &Payload, scope: &SubmissionScope| {
            let mut partition=|root|q.addressable.as_ref().ok_or(Error::PrefillScopeUnavailable)?.accept(root);
            let bank=Mechanisms::<A>::prepare_session_external_neural_bank(
                work.0,
                q.layerwise.as_ref(),
                recipe.native_recipe(),
                invocation,
                work.2
                    .as_ref()
                    .ok_or(Error::PrefillScopeUnavailable)?
                    .clone(),
                scope,q.addressable.is_some().then_some(&mut partition),
            )?;
            if let Some(source)=q.addressable.as_ref(){source.bind(bank.as_ref().ok_or(Error::PrefillScopeUnavailable)?)?;}
            Ok(bank)
        },
        |work: &mut Work<'_, A, D, F>,q:&Payload,active:&ActiveEmbeddedNativeInvocation<'_,OriginalExternalSpeculativeRole>| {
            match q.addressable.as_ref(){
                Some(source)=>source.run(work,q,active,execute_handler),
                None=>execute_handler(work,q,active),
            }
        },
        |work: &mut Work<'_, A, D, F>,
         output: &Output,
         budget: &OriginalBufferBudget,
         custody: &OriginalSpeculativeBudgetCustody| {
            let prior = prior_sources(work.3.as_ref(), context, funding)?;
            let mut error = None;
            let completed = CompletedResidentSource::capture_array_sources_with_priors(
                |visit| {
                    error = super::prediction::visit_target(work.0, visit).err();
                    output.visit(visit);
                },
                &prior,
                budget,
                custody,
                funding,
            )?
            .with_completed_stream(environment.stream())?;
            drop(prior);
            if let Some(error) = error {
                return Err(error);
            }
            let mut publication_error = None;
            let evidence = retain_external_evidence_for_placement(
                completed.try_clone_for_source(environment.stream(),funding)?,
                |visit| {
                    publication_error = super::prediction::visit_target(work.0, visit).err();
                    output.visit(visit);
                },
                context.embedded_tensor_sources(),
                context,
                SamplingPlacement::Target,
                funding,
            )?;
            if let Some(error) = publication_error {
                return Err(error);
            }
            *work.3 = Some(completed);
            work.4 = Some(evidence);
            Ok(())
        },
    );
    let requirements = SpeculativeInvocationRequirements::new(
        recipe.plan(),
        layout.physical_bytes(),
        Some(layout.graph_bytes()),
        Some(layout.record_bytes()),
        controls(&layout, &work, &handlers)
            .and_then(|n| n.checked_add(bank_controls))
            .and_then(|n| n.checked_add(completion_controls))
            .and_then(|n|n.checked_add(addressable_controls))
            .and_then(|n|n.checked_add(addressable_callbacks)),
    )
    .map_err(|e| sources.retain_startup_error(e))?;
    let requirements = match source_facts {
        Some(facts) => requirements
            .with_host_source_constructions(facts)
            .map_err(|e| sources.retain_startup_error(e))?,
        None => requirements,
    };
    let role = sources
        .request()
        .reserve_external_role(claim, requirements)
        .map_err(|e| sources.retain_admission_error(e))?;
    if request.is_some() { payload.completion = Some(NestedCompletionOwner::prepare_external(
        completion_roots,
        validation_roots,
        &role,
        funding,
        completion_stream,
    ).map_err(|cause|sources.retain_error(cause))?); }
    work.2 = Some(role.clone());
    let (roots, _) = sources.numerical_prerequisites();
    let mut output = layout.run(
        &recipe,
        environment,
        roots,
        role,
        payload,
        funding.clone(),
        &mut work,
        handlers,
    ).map_err(|cause|sources.retain_error(cause))?;
    output.evidence = Some(
        work.4
            .take()
            .ok_or(Error::PrefillControl(WorkingMemoryError::IdentityMismatch))?,
    );
    Ok(output)
}
