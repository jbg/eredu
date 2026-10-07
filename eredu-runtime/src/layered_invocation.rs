//! Common invocation access without changing layered traversal or residency policy.

use eredu_nn::{NeuralBackend, Tensor};

use crate::{
    LayeredArchitecture, LayerwisePolicy, LayerwiseRuntime, LayerwiseRuntimeError,
    ParallelLayeredArchitecture, ResidentRuntime, RuntimeState, SubmissionBackend,
};

/// Invokes an existing runtime while allowing architecture-owned ingress and readout.
///
/// The caller supplies unpublished mutable state and retains completion and commit
/// ownership. Implementations retain their ordinary unit providers and traversal.
pub trait LayeredInvocation<B: NeuralBackend, S: RuntimeState<B>> {
    /// Architecture owning ingress, units and readout.
    type Architecture: LayeredArchitecture<B, S>;
    /// The unchanged runtime failure, including residency-policy failures.
    type Error;

    /// Borrows architecture-owned modules between invocations.
    fn architecture_mut(&mut self) -> &mut Self::Architecture;

    /// Runs one ordinary invocation over the supplied candidate state.
    fn invoke<'a>(
        &mut self,
        input: <Self::Architecture as LayeredArchitecture<B, S>>::Input<'a>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        self.invoke_with_context(input, state, context)
            .map(|(output, _)| output)
    }

    /// Returns architecture-owned captures from the same traversal as its output.
    /// Captures remain request-local and never require a second forward pass.
    fn invoke_with_context<'a>(
        &mut self,
        input: <Self::Architecture as LayeredArchitecture<B, S>>::Input<'a>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<
        (
            B::Tensor,
            <Self::Architecture as LayeredArchitecture<B, S>>::ForwardContext,
        ),
        Self::Error,
    >;
}

/// Invokes the runtime's existing collective lifecycle on the same candidate state.
pub trait ParallelLayeredInvocation<B: NeuralBackend, S: RuntimeState<B>>:
    LayeredInvocation<B, S>
{
    /// Runs one rank-local invocation with the caller's collective context.
    fn invoke_parallel<'a>(
        &mut self,
        input: <Self::Architecture as LayeredArchitecture<B, S>>::Input<'a>,
        state: &mut S,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error>;
}

impl<A, B, S> LayeredInvocation<B, S> for ResidentRuntime<A, B, S>
where
    B: NeuralBackend,
    S: RuntimeState<B>,
    A: LayeredArchitecture<B, S>,
{
    type Architecture = A;
    type Error = A::Error;
    fn architecture_mut(&mut self) -> &mut A {
        ResidentRuntime::architecture_mut(self)
    }
    fn invoke_with_context<'a>(
        &mut self,
        input: A::Input<'a>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(B::Tensor, A::ForwardContext), Self::Error> {
        self.forward_with_context(input, state, context)
    }
}

impl<A, B, S, P> LayeredInvocation<B, S> for LayerwiseRuntime<A, B, S, P>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    S: RuntimeState<B>,
    A: LayeredArchitecture<B, S>,
    P: LayerwisePolicy<B, A::Unit>,
    A::Error: std::fmt::Display,
    P::Error: std::fmt::Display,
{
    type Architecture = A;
    type Error = LayerwiseRuntimeError<A::Error, P::Error>;
    fn architecture_mut(&mut self) -> &mut A {
        LayerwiseRuntime::architecture_mut(self)
    }
    fn invoke_with_context<'a>(
        &mut self,
        input: A::Input<'a>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(B::Tensor, A::ForwardContext), Self::Error> {
        self.forward_with_context_hook(input, state, context, |_, _, _| Ok(()))
    }
}

impl<A, B, S, P> ParallelLayeredInvocation<B, S> for LayerwiseRuntime<A, B, S, P>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    S: RuntimeState<B>,
    A: ParallelLayeredArchitecture<B, S>,
    P: LayerwisePolicy<B, A::Unit>,
    A::Error: std::fmt::Display,
    P::Error: std::fmt::Display,
{
    fn invoke_parallel<'a>(
        &mut self,
        input: A::Input<'a>,
        state: &mut S,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        self.forward_parallel(input, state, parallel, context)
    }
}
