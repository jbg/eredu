//! Portable invocation of a local architecture using an installed unit policy.

use eredu_core::backend::LocalComponentStorage;
use std::borrow::Borrow;

use eredu_core::{
    residency::{ParameterConversionRetentionTrimReport, ParameterConversionTrimError},
    BackendFailure, SubmissionBackend,
};
use eredu_nn::{NeuralBackend, Tensor};

use crate::{
    ActivationObserver, BoundLocalComponent, DenseDiskStreamReport, LayeredArchitecture,
    LayerwisePolicy, LayerwiseRuntime, ResidencyReport, RuntimeState, SelectedStateRealization,
    StateLayout, UnitPolicyResidency,
};

type Context<B> = <<B as NeuralBackend>::Tensor as Tensor>::Context;

/// Conversion retention mechanism of the installed local parameter policy.
pub trait LocalParameterRetention {
    /// Releases optional converted parameters at an idle boundary.
    fn trim_local_conversions(
        &self,
    ) -> Result<Vec<ParameterConversionRetentionTrimReport>, ParameterConversionTrimError>;
}

/// Owns local layerwise invocation, its selected unit policy, and state placement.
pub struct LocalComponentDriver<A, B, S, P>
where
    B: SubmissionBackend<Executor = Context<B>>,
    S: RuntimeState<B>,
    A: LayeredArchitecture<B, S>,
    P: LayerwisePolicy<B, A::Unit>,
{
    runtime: LayerwiseRuntime<A, B, S, P>,
    state_layout: StateLayout,
    selected_state: Option<SelectedStateRealization>,
    executor: B::OwnedExecutor,
}

impl<A, B, S, P> LocalComponentDriver<A, B, S, P>
where
    B: SubmissionBackend<Executor = Context<B>>,
    S: RuntimeState<B>,
    A: LayeredArchitecture<B, S>,
    A::Error: std::fmt::Display,
    P: LayerwisePolicy<B, A::Unit>,
    P::Error: std::fmt::Display,
{
    /// Installs the architecture with its fully bound parameters and exact state placement.
    pub fn new(
        architecture: A,
        policy: P,
        state_layout: StateLayout,
        selected_state: Option<SelectedStateRealization>,
        executor: B::OwnedExecutor,
    ) -> Self {
        Self {
            runtime: LayerwiseRuntime::new(architecture, policy),
            state_layout,
            selected_state,
            executor,
        }
    }
}

impl<A, B, S, P> BoundLocalComponent<A, B, S> for LocalComponentDriver<A, B, S, P>
where
    B: SubmissionBackend<Executor = Context<B>> + LocalComponentStorage,
    S: crate::state::PortableTextState<B, B::CacheStorage>,
    P: LayerwisePolicy<B, A::Unit> + UnitPolicyResidency + LocalParameterRetention,
    P::Error: std::error::Error + Send + Sync + 'static,
    A: LayeredArchitecture<B, S, Error = eredu_nn::Error> + 'static,
    A::Unit: 'static,
{
    fn execute(
        &mut self,
        input: A::Input<'_>,
        state: &mut S,
        observer: &mut dyn ActivationObserver<B::Tensor, eredu_nn::Error>,
    ) -> Result<B::Tensor, eredu_nn::Error> {
        self.runtime
            .forward_with_observer(input, state, self.executor.borrow(), observer)
            .map_err(eredu_nn::Error::backend_source)
    }

    fn execute_parallel(
        &mut self,
        input: A::Input<'_>,
        state: &mut S,
        parallel: &B::ParallelContext,
    ) -> Result<B::Tensor, eredu_nn::Error>
    where
        A: crate::ParallelLayeredArchitecture<B, S>,
    {
        self.runtime
            .forward_parallel(input, state, parallel, self.executor.borrow())
            .map_err(eredu_nn::Error::backend_source)
    }

    fn state_layout(&self) -> &StateLayout {
        &self.state_layout
    }

    fn readout(
        &mut self,
        project: impl FnOnce(&mut A, &Context<B>) -> Result<B::Tensor, eredu_nn::Error>,
    ) -> Result<B::Tensor, eredu_nn::Error> {
        project(self.runtime.architecture_mut(), self.executor.borrow())
    }

    fn settle<'a>(values: impl Iterator<Item = &'a B::Tensor>) -> Result<(), BackendFailure> {
        B::settle_local_values(values)
    }

    fn fresh_state(&self) -> Result<S, BackendFailure> {
        if let Some(selected) = &self.selected_state {
            crate::state::allocate_selected_text_state::<B, B::CacheStorage, S>(selected)
        } else {
            S::device(self.state_layout.clone())
        }
    }

    fn residency(
        &self,
    ) -> Result<(ResidencyReport, Option<DenseDiskStreamReport>), BackendFailure> {
        let policy = self.runtime.policy();
        Ok((policy.residency()?, policy.dense_stream_report()?))
    }

    fn trim_conversions(
        &self,
    ) -> Result<Vec<ParameterConversionRetentionTrimReport>, ParameterConversionTrimError> {
        self.runtime.policy().trim_local_conversions()
    }
}

impl<A, B, S, P> crate::LayeredInvocation<B, S> for LocalComponentDriver<A, B, S, P>
where
    B: SubmissionBackend<Executor = Context<B>>,
    S: RuntimeState<B>,
    P: LayerwisePolicy<B, A::Unit>,
    P::Error: std::fmt::Display,
    A: LayeredArchitecture<B, S, Error = eredu_nn::Error>,
    A::Unit: 'static,
{
    type Architecture = A;
    type Error = crate::LayerwiseRuntimeError<eredu_nn::Error, P::Error>;

    fn architecture_mut(&mut self) -> &mut A {
        self.runtime.architecture_mut()
    }

    fn invoke_with_context<'a>(
        &mut self,
        input: A::Input<'a>,
        state: &mut S,
        context: &Context<B>,
    ) -> Result<(B::Tensor, A::ForwardContext), Self::Error> {
        self.runtime.invoke_with_context(input, state, context)
    }
}

impl<A, B, S, P> crate::ParallelLayeredInvocation<B, S> for LocalComponentDriver<A, B, S, P>
where
    B: SubmissionBackend<Executor = Context<B>>,
    S: RuntimeState<B>,
    P: LayerwisePolicy<B, A::Unit>,
    P::Error: std::fmt::Display,
    A: crate::ParallelLayeredArchitecture<B, S, Error = eredu_nn::Error>,
    A::Unit: 'static,
{
    fn invoke_parallel<'a>(
        &mut self,
        input: A::Input<'a>,
        state: &mut S,
        parallel: &B::ParallelContext,
        context: &Context<B>,
    ) -> Result<B::Tensor, Self::Error> {
        self.runtime
            .forward_parallel(input, state, parallel, context)
    }
}
