//! Entry and dependency state for the existing group traversals.
//!
//! Ordinary calls and authenticated retained ingress share the same unit loops.
//! This is crate-private: it is not an arbitrary callback execution surface.

use super::*;

pub(crate) trait LayeredInvocation<A, B, S>
where
    B: NeuralBackend,
    S: RuntimeState<B>,
    A: LayeredArchitecture<B, S>,
{
    fn begin<H: LayeredTraversalHook<B, A::ForwardContext, A::Error> + ?Sized>(
        &mut self,
        architecture: &mut A,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
        hook: &mut H,
    ) -> Result<LayeredForwardState<B::Tensor, A::ForwardContext>, A::Error>;

    fn begin_parallel<H: LayeredTraversalHook<B, A::ForwardContext, A::Error> + ?Sized>(
        &mut self,
        architecture: &mut A,
        state: &mut S,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        hook: &mut H,
    ) -> Result<LayeredForwardState<B::Tensor, A::ForwardContext>, A::Error>
    where
        A: ParallelLayeredArchitecture<B, S>;

    // Some(None) is an already completed ancestor whose value is no longer a
    // live cut dependency. No callback, model call, or new completion is emitted.
    fn retained_group(&self, _group: usize) -> Option<Option<B::Tensor>> {
        None
    }

    fn is_retained_group(&self, _group: usize) -> bool {
        false
    }

    fn before_group(
        &mut self,
        _architecture: &mut A,
        _group: usize,
        _initial: &mut B::Tensor,
        _forward: &mut A::ForwardContext,
        _state: &mut S,
        _parallel: Option<&B::ParallelContext>,
        _context: &<B::Tensor as Tensor>::Context,
    ) -> Result<bool, A::Error> {
        Ok(false)
    }

    fn before_group_with_hook<H: LayeredTraversalHook<B, A::ForwardContext, A::Error> + ?Sized>(
        &mut self,
        architecture: &mut A,
        group: usize,
        initial: &mut B::Tensor,
        forward: &mut A::ForwardContext,
        state: &mut S,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        _hook: &mut H,
    ) -> Result<bool, A::Error> {
        self.before_group(
            architecture,
            group,
            initial,
            forward,
            state,
            parallel,
            context,
        )
    }

    // A real completed traversal observed the architecture's inactive decision.
    // No tensor is manufactured or retained for that dependency.
    fn after_inactive_group(&mut self, _group: usize) {}
    fn is_inactive_dependency(&self, _group: usize) -> bool {
        false
    }
    fn after_group(&mut self, _group: usize, _value: &B::Tensor) {}
}

pub(crate) struct OrdinaryLayeredInput<'a, A, B, S>
where
    B: NeuralBackend,
    S: RuntimeState<B>,
    A: LayeredArchitecture<B, S> + 'a,
{
    input: Option<A::Input<'a>>,
    marker: std::marker::PhantomData<fn() -> (B, S)>,
}

impl<'a, A, B, S> OrdinaryLayeredInput<'a, A, B, S>
where
    B: NeuralBackend,
    S: RuntimeState<B>,
    A: LayeredArchitecture<B, S> + 'a,
{
    pub(crate) fn new(input: A::Input<'a>) -> Self {
        Self {
            input: Some(input),
            marker: std::marker::PhantomData,
        }
    }
}

impl<A, B, S> LayeredInvocation<A, B, S> for OrdinaryLayeredInput<'_, A, B, S>
where
    B: NeuralBackend,
    S: RuntimeState<B>,
    A: LayeredArchitecture<B, S>,
{
    fn begin<H: LayeredTraversalHook<B, A::ForwardContext, A::Error> + ?Sized>(
        &mut self,
        architecture: &mut A,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
        hook: &mut H,
    ) -> Result<LayeredForwardState<B::Tensor, A::ForwardContext>, A::Error> {
        let input = self.input.take().expect("one begin per layered invocation");
        if hook.observes_activations() {
            architecture.begin_forward_observed(
                input,
                state,
                context,
                &mut TraversalActivationObserver {
                    hook,
                    types: std::marker::PhantomData,
                },
            )
        } else {
            architecture.begin_forward(input, state, context)
        }
    }

    fn begin_parallel<H: LayeredTraversalHook<B, A::ForwardContext, A::Error> + ?Sized>(
        &mut self,
        architecture: &mut A,
        state: &mut S,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        hook: &mut H,
    ) -> Result<LayeredForwardState<B::Tensor, A::ForwardContext>, A::Error>
    where
        A: ParallelLayeredArchitecture<B, S>,
    {
        let input = self.input.take().expect("one begin per layered invocation");
        if hook.observes_activations() {
            architecture.begin_forward_parallel_observed(
                input,
                state,
                parallel,
                context,
                &mut TraversalActivationObserver {
                    hook,
                    types: std::marker::PhantomData,
                },
            )
        } else {
            architecture.begin_forward_parallel(input, state, parallel, context)
        }
    }
}
