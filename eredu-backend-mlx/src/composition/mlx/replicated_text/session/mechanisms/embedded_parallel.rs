//! Lexical source projection around the existing embedded model callback.
use super::*;
use crate::composition::mlx::speculative::embedded_native::EmbeddedParallelInvocation;
use eredu_nn::workspace::{HostMetadataFunding, WorkspaceContext, WorkspaceMetadataAllocation};
use std::mem::{size_of, size_of_val};

type Session<A, S, D> =
    ReplicatedTextSession<A, MlxNeuralBackend, MlxReplicatedTextMechanisms<A, S>, D>;

fn install<A, S>(
    active: EmbeddedParallelInvocation,
) -> impl FnOnce(&mut MlxReplicatedTextMechanisms<A, S>) -> Result<(), Error>
where
    S: MlxStateMechanisms,
    A: eredu_runtime::LayeredArchitecture<MlxNeuralBackend, S, Error = eredu_nn::Error>,
{
    move |mechanisms| mechanisms.install_embedded_invocation(active)
}
fn operation<A, S, D, C, T>(
    args: C,
    call: fn(&mut Session<A, S, D>, C) -> Result<T, Error>,
) -> impl FnOnce(&mut Session<A, S, D>) -> Result<T, Error>
where
    S: MlxStateMechanisms,
    A: eredu_runtime::LayeredArchitecture<MlxNeuralBackend, S, Error = eredu_nn::Error>,
    D: eredu_runtime::ReplicatedTextExecutionStrategy<
            A,
            MlxNeuralBackend,
            S,
            MlxArchitectureLayerwisePolicy<A, S>,
            MlxArchitectureLayerwisePolicy<A, S>,
        >,
    A::Unit: 'static,
{
    move |session| call(session, args)
}
fn frames<A, S, D, C, T>() -> Option<usize>
where
    S: MlxStateMechanisms,
    A: eredu_runtime::LayeredArchitecture<MlxNeuralBackend, S, Error = eredu_nn::Error>,
    D: eredu_runtime::ReplicatedTextExecutionStrategy<
            A,
            MlxNeuralBackend,
            S,
            MlxArchitectureLayerwisePolicy<A, S>,
            MlxArchitectureLayerwisePolicy<A, S>,
        >,
    A::Unit: 'static,
{
    let parts = [
        size_of::<(
            &mut Session<A, S, D>,
            EmbeddedParallelInvocation,
            &HostMetadataFunding,
            C,
            fn(&mut Session<A, S, D>, C) -> Result<T, Error>,
        )>(),
        size_of::<Result<T, Error>>(),
    ];
    parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
}
impl<A, S> MlxReplicatedTextMechanisms<A, S>
where
    S: MlxStateMechanisms,
    A: eredu_runtime::LayeredArchitecture<MlxNeuralBackend, S, Error = eredu_nn::Error>,
    A::Unit: 'static,
{
    pub(crate) fn session_embedded_parallel_control_bytes<D, C, T>(
        session: &Session<A, S, D>,
    ) -> Option<usize>
    where
        D: eredu_runtime::ReplicatedTextExecutionStrategy<
                A,
                MlxNeuralBackend,
                S,
                MlxArchitectureLayerwisePolicy<A, S>,
                MlxArchitectureLayerwisePolicy<A, S>,
            >,
    {
        fn query<A, S, D, C, T, I, F>(
            session: &Session<A, S, D>,
            _: impl FnOnce(EmbeddedParallelInvocation) -> I,
            _: impl FnOnce(C, fn(&mut Session<A, S, D>, C) -> Result<T, Error>) -> F,
        ) -> Option<usize>
        where
            S: MlxStateMechanisms,
            A: eredu_runtime::LayeredArchitecture<MlxNeuralBackend, S, Error = eredu_nn::Error>,
            A::Unit: 'static,
            D: eredu_runtime::ReplicatedTextExecutionStrategy<
                    A,
                    MlxNeuralBackend,
                    S,
                    MlxArchitectureLayerwisePolicy<A, S>,
                    MlxArchitectureLayerwisePolicy<A, S>,
                >,
            I: FnOnce(&mut MlxReplicatedTextMechanisms<A, S>) -> Result<(), Error>,
            F: FnOnce(&mut Session<A, S, D>) -> Result<T, Error>,
        {
            session.execution_mechanism_binding_control_bytes::<(), Result<T, Error>, Error, I, F>()
        }
        frames::<A, S, D, C, T>()?
            .checked_add(WorkspaceContext::metadata_source_bytes::<
                eredu_runtime::replicated_session::ExecutionMechanismBindingCause,
            >()?)?
            .checked_add(query(session, install::<A, S>, operation::<A, S, D, C, T>)?)
    }
    pub(crate) fn with_session_embedded_parallel<D, C, T>(
        session: &mut Session<A, S, D>,
        active: EmbeddedParallelInvocation,
        funding: &HostMetadataFunding,
        args: C,
        call: fn(&mut Session<A, S, D>, C) -> Result<T, Error>,
    ) -> Result<T, Error>
    where
        D: eredu_runtime::ReplicatedTextExecutionStrategy<
                A,
                MlxNeuralBackend,
                S,
                MlxArchitectureLayerwisePolicy<A, S>,
                MlxArchitectureLayerwisePolicy<A, S>,
            >,
    {
        funding.reserve_metadata(
            frames::<A, S, D, C, T>()
                .ok_or(eredu_nn::workspace::HostMetadataFundingError::Overflow)?,
        )?;
        active.validate_execution(session.inference_execution_identity())?;
        session
            .with_execution_mechanism_binding(
                funding,
                install::<A, S>(active),
                Self::retire_embedded_invocation,
                operation(args, call),
            )
            .map_err(|cause| Error::Neural(funding.metadata_source(cause)))?
            .and_then(|result| result)
    }
}
