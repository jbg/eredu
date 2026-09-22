//! The selected shared session transaction with a lexical prediction-state loan.
use super::*;
use crate::composition::mlx::speculative::autoregressive::ActiveSpeculativeInvocation;
use eredu_nn::workspace::{HostMetadataFunding, WorkspaceContext, WorkspaceMetadataAllocation};
use std::mem::{size_of, size_of_val};

type Session<A, S, D> =
    ReplicatedTextSession<A, MlxNeuralBackend, MlxReplicatedTextMechanisms<A, S>, D>;
type SessionError = eredu_runtime::ReplicatedTextSessionError<eredu_nn::Error, Error, Error>;
type LoanError = eredu_runtime::replicated_session::PredictionStateLoanError<SessionError>;

struct Arguments<'a, 'c, 'o, A, S>
where
    S: MlxStateMechanisms,
    A: eredu_runtime::LayeredArchitecture<MlxNeuralBackend, S, Error = eredu_nn::Error> + 'a,
{
    input: A::Input<'a>,
    prefill: bool,
    demand: eredu_core::OutputDemand,
    stream: &'a Stream,
    completion: Option<&'a mut (dyn AutoregressiveSequenceCompletion + 'c)>,
    checkpoint: Option<StateCheckpoint<S>>,
    funding: Option<&'a HostMetadataFunding>,
    observer: &'a mut (dyn eredu_runtime::ActivationObserver<MlxTensor, eredu_nn::Error> + 'o),
    active: Option<ActiveSpeculativeInvocation>,
}
fn retained<E: std::error::Error + Send + Sync + 'static>(
    cause: E,
    funding: Option<&HostMetadataFunding>,
) -> Error {
    match funding {
        Some(funding) => Error::Neural(funding.metadata_source(cause)),
        None => Error::Other(Box::new(cause)),
    }
}

fn forward<'a, 'c, 'o, A, S, D>(
    args: Arguments<'a, 'c, 'o, A, S>,
) -> impl FnOnce(&mut Session<A, S, D>) -> Result<Option<MlxTensor>, Error> + use<'a, 'c, 'o, A, S, D>
where
    S: MlxStateMechanisms + 'a,
    A: eredu_runtime::LayeredArchitecture<MlxNeuralBackend, S, Error = eredu_nn::Error> + 'a,
    A::Unit: 'static,
    D: eredu_runtime::ReplicatedTextExecutionStrategy<
            A,
            MlxNeuralBackend,
            S,
            MlxArchitectureLayerwisePolicy<A, S>,
            MlxArchitectureLayerwisePolicy<A, S>,
        >,
{
    move |session| {
        let result = match args.completion {
            Some(completion) if args.prefill => session
                .prefill_span_with_checkpoint_completion_and_observer(
                    args.input,
                    args.demand,
                    args.stream,
                    args.checkpoint.expect("completion owns checkpoint"),
                    args.funding.expect("completion owns funding"),
                    |output, state, stream| {
                        completion.complete(output.map(MlxTensor::as_array), state, stream)
                    },
                    args.observer,
                ),
            Some(completion) => session
                .sequence_logits_with_checkpoint_completion_and_observer(
                    args.input,
                    eredu_runtime::ExpertPass::Decode,
                    args.stream,
                    args.checkpoint.expect("completion owns checkpoint"),
                    |output, state, stream| {
                        completion.complete(Some(output.as_array()), state, stream)
                    },
                    args.observer,
                )
                .map(Some),
            None => session
                .sequence_logits(
                    args.input,
                    if args.prefill {
                        eredu_runtime::ExpertPass::Prefill
                    } else {
                        eredu_runtime::ExpertPass::Decode
                    },
                    args.stream,
                )
                .map(Some),
        }
        .map_err(|cause| retained(cause, args.funding));
        let finished = args
            .active
            .as_ref()
            .map_or(Ok(()), |active| active.finish_transaction(result.is_ok()));
        finish_prediction_state_operation_with_metadata(result, finished, args.funding)
    }
}
fn loan<'a, 'c, 'o, A, S, D>(
    args: Arguments<'a, 'c, 'o, A, S>,
    cache: &'a mut MlxPredictionTargetState,
) -> impl FnOnce(&mut Session<A, S, D>) -> Result<Option<MlxTensor>, Error> + use<'a, 'c, 'o, A, S, D>
where
    S: MlxStateMechanisms + 'static,
    A: eredu_runtime::LayeredArchitecture<MlxNeuralBackend, S, Error = eredu_nn::Error> + 'a,
    A::Unit: 'static,
    D: eredu_runtime::ReplicatedTextExecutionStrategy<
            A,
            MlxNeuralBackend,
            S,
            MlxArchitectureLayerwisePolicy<A, S>,
            MlxArchitectureLayerwisePolicy<A, S>,
        >,
{
    move |session| {
        let funding = args.funding;
        let active = args.active.clone();
        let stream = args.stream;
        let result = (|| {
            let lane = cache
                .state_mut::<S>()
                .ok_or(Error::PrefillScopeUnavailable)?;
            let (result, returned) = session
                .with_prediction_target_state(lane, stream, funding, forward::<A, S, D>(args))
                .map_err(|cause| retained(cause, funding))?;
            finish_prediction_state_operation_with_metadata(
                result,
                returned.map_err(|cause| retained(cause, funding)),
                funding,
            )
        })();
        let finished = active.as_ref().map_or(Ok(()), |active| {
            active.finish_prediction_loan(result.is_ok())
        });
        finish_prediction_state_operation_with_metadata(result, finished, funding)
    }
}
fn install<A, S>(
    active: ActiveSpeculativeInvocation,
) -> impl FnOnce(&mut MlxReplicatedTextMechanisms<A, S>) -> Result<(), Error>
where
    S: MlxStateMechanisms,
    A: eredu_runtime::LayeredArchitecture<MlxNeuralBackend, S, Error = eredu_nn::Error>,
{
    move |mechanisms| mechanisms.install_autoregressive_invocation(active)
}

pub(super) fn run<'a, 'c, 'o, A, S, D>(
    session: &mut Session<A, S, D>,
    cache: &'a mut MlxPredictionTargetState,
    input: A::Input<'a>,
    prefill: bool,
    demand: eredu_core::OutputDemand,
    stream: &'a Stream,
    completion: Option<&'a mut (dyn AutoregressiveSequenceCompletion + 'c)>,
    checkpoint: Option<StateCheckpoint<S>>,
    funding: Option<&'a HostMetadataFunding>,
    observer: &'a mut (dyn eredu_runtime::ActivationObserver<MlxTensor, eredu_nn::Error> + 'o),
) -> Result<Option<MlxTensor>, Error>
where
    S: MlxStateMechanisms + 'static,
    A: eredu_runtime::LayeredArchitecture<MlxNeuralBackend, S, Error = eredu_nn::Error> + 'a,
    A::Unit: 'static,
    D: eredu_runtime::ReplicatedTextExecutionStrategy<
            A,
            MlxNeuralBackend,
            S,
            MlxArchitectureLayerwisePolicy<A, S>,
            MlxArchitectureLayerwisePolicy<A, S>,
        >,
{
    if let Some(funding) = funding {
        funding
            .reserve_metadata(frames::<A, S, D>().ok_or(Error::PrefillScopeUnavailable)?)
            .map_err(Error::WorkspacePlanning)?;
    }
    let active = completion
        .as_ref()
        .map(|completion| completion.active_invocation());
    let args = Arguments {
        input,
        prefill,
        demand,
        stream,
        completion,
        checkpoint,
        funding,
        observer,
        active: active.clone(),
    };
    let operation = loan::<A, S, D>(args, cache);
    match active {
        Some(active) => session
            .with_execution_mechanism_binding(
                funding.ok_or(Error::PrefillScopeUnavailable)?,
                install::<A, S>(active),
                MlxReplicatedTextMechanisms::retire_autoregressive_invocation,
                operation,
            )
            .map_err(|cause| retained(cause, funding))
            .and_then(|value| value)
            .and_then(|value| value),
        None => operation(session),
    }
}

fn frames<A, S, D>() -> Option<usize>
where
    S: MlxStateMechanisms + 'static,
    A: eredu_runtime::LayeredArchitecture<MlxNeuralBackend, S, Error = eredu_nn::Error>,
    A::Unit: 'static,
    D: eredu_runtime::ReplicatedTextExecutionStrategy<
            A,
            MlxNeuralBackend,
            S,
            MlxArchitectureLayerwisePolicy<A, S>,
            MlxArchitectureLayerwisePolicy<A, S>,
        >,
{
    let parts = [
        size_of::<Arguments<'_, '_, '_, A, S>>(),
        size_of::<Option<ActiveSpeculativeInvocation>>(),
        size_of::<Result<Option<MlxTensor>, Error>>(),
        size_of::<Result<(), Error>>(),
        size_of::<(
            &mut Session<A, S, D>,
            &mut MlxPredictionTargetState,
            &Stream,
        )>(),
    ];
    parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
}

fn error_controls() -> Option<usize> {
    let parts = [
        WorkspaceContext::metadata_source_bytes::<SessionError>()?,
        WorkspaceContext::metadata_source_bytes::<LoanError>()?.checked_mul(2)?,
        WorkspaceContext::metadata_source_bytes::<
            eredu_runtime::replicated_session::ExecutionMechanismBindingCause,
        >()?,
        WorkspaceContext::metadata_source_bytes::<PredictionStateRecoveryFailure>()?
            .checked_mul(3)?,
    ];
    parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
}

pub(super) fn control_bytes<'a, A, S, D>(session: &'a Session<A, S, D>) -> Option<usize>
where
    S: MlxStateMechanisms + 'static,
    A: eredu_runtime::LayeredArchitecture<MlxNeuralBackend, S, Error = eredu_nn::Error> + 'a,
    A::Unit: 'static,
    D: eredu_runtime::ReplicatedTextExecutionStrategy<
            A,
            MlxNeuralBackend,
            S,
            MlxArchitectureLayerwisePolicy<A, S>,
            MlxArchitectureLayerwisePolicy<A, S>,
        >,
{
    fn forward_size<'a, A, S, D, F>(
        session: &Session<A, S, D>,
        _: impl FnOnce(Arguments<'a, 'a, 'a, A, S>) -> F,
    ) -> Option<usize>
    where
        S: MlxStateMechanisms + 'static,
        A: eredu_runtime::LayeredArchitecture<MlxNeuralBackend, S, Error = eredu_nn::Error> + 'a,
        A::Unit: 'static,
        D: eredu_runtime::ReplicatedTextExecutionStrategy<
                A,
                MlxNeuralBackend,
                S,
                MlxArchitectureLayerwisePolicy<A, S>,
                MlxArchitectureLayerwisePolicy<A, S>,
            >,
        F: FnOnce(&mut Session<A, S, D>) -> Result<Option<MlxTensor>, Error>,
    {
        session.prediction_state_loan_control_bytes::<Option<MlxTensor>, Error, F>()
    }
    fn loan_size<'a, A, S, D, F, I>(
        session: &Session<A, S, D>,
        _: impl FnOnce(Arguments<'a, 'a, 'a, A, S>, &'a mut MlxPredictionTargetState) -> F,
        _: impl FnOnce(ActiveSpeculativeInvocation) -> I,
    ) -> Option<usize>
    where
        S: MlxStateMechanisms + 'static,
        A: eredu_runtime::LayeredArchitecture<MlxNeuralBackend, S, Error = eredu_nn::Error> + 'a,
        A::Unit: 'static,
        D: eredu_runtime::ReplicatedTextExecutionStrategy<
                A,
                MlxNeuralBackend,
                S,
                MlxArchitectureLayerwisePolicy<A, S>,
                MlxArchitectureLayerwisePolicy<A, S>,
            >,
        F: FnOnce(&mut Session<A, S, D>) -> Result<Option<MlxTensor>, Error>,
        I: FnOnce(&mut MlxReplicatedTextMechanisms<A, S>) -> Result<(), Error>,
    {
        session.execution_mechanism_binding_control_bytes::<(),Result<Option<MlxTensor>,Error>,Error,I,F>()
    }
    frames::<A, S, D>()?
        .checked_add(error_controls()?)?
        .checked_add(forward_size(session, forward::<A, S, D>)?)?
        .checked_add(loan_size(session, loan::<A, S, D>, install::<A, S>)?)
}
