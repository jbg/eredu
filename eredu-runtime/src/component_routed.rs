//! Provider-aware equation capabilities for shared component execution.
#![allow(clippy::too_many_arguments)]
use crate::*;
use eredu_nn::Tensor;
use std::borrow::Borrow;

/// Explicit provider-aware component equations.
pub trait RoutedComponentKernel<B, S>: ComponentEquations<B, S>
where
    B: eredu_nn::GroupedNeuralBackend,
    S: RuntimeState<B>,
{
    /// Classifies the semantic provider pass for one validated invocation.
    fn expert_pass(
        &self,
        _address: ExecutionUnitAddress,
        hidden: &B::Tensor,
        _forward: &Self::ForwardContext,
    ) -> ExpertPass {
        let sequence = hidden
            .shape()
            .get(hidden.shape().len().saturating_sub(2))
            .copied()
            .unwrap_or(1);
        if sequence > 1 {
            ExpertPass::Prefill
        } else {
            ExpertPass::Decode
        }
    }

    /// Declares observation points for one validated routed invocation.
    fn routing_points(
        &self,
        _address: ExecutionUnitAddress,
    ) -> Result<Option<RoutedObservationPoints>, Self::Error> {
        Ok(None)
    }

    /// Reports internal component hooks on the ordinary provider call.
    fn routed_unit_observations(&self) -> bool {
        false
    }

    /// Reports sparse expert hooks on the ordinary provider call.
    fn routed_sparse_observations(&self) -> bool {
        false
    }

    /// Executes one ordinary unit through its admitted expert provider.
    fn execute_routed<P>(
        &mut self,
        address: ExecutionUnitAddress,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        pass: ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error>
    where
        P: ParameterProvider<B>,
        P::Error: std::fmt::Display;

    /// Executes the same provider-backed unit with internal instrumentation.
    fn execute_routed_observed<P, O>(
        &mut self,
        address: ExecutionUnitAddress,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        pass: ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Self::Error>
    where
        P: ParameterProvider<B>,
        P::Error: std::fmt::Display,
        O: ActivationObserver<B::Tensor, Self::Error> + ?Sized,
        Self::Error: std::fmt::Display,
    {
        match self.routing_points(address)? {
            Some(point) => {
                let mut observed = ObservedExpertProvider::new(provider, observer, point);
                self.execute_routed(
                    address,
                    unit,
                    hidden,
                    state,
                    forward,
                    pass,
                    &mut observed,
                    context,
                )
            }
            None => self.execute_routed(
                address, unit, hidden, state, forward, pass, provider, context,
            ),
        }
    }
}

impl<B, D, K, S, R> RoutedLayeredArchitecture<B, S> for ComponentAdapter<D, K, R>
where
    B: eredu_nn::GroupedNeuralBackend,
    D: Borrow<ComponentDefinition> + 'static,
    S: RuntimeState<B>,
    R: ComponentResourceAccess<B, K> + 'static,
    K: RoutedComponentKernel<B, S> + 'static,
{
    fn expert_pass_for_unit(
        &self,
        group: usize,
        index: usize,
        hidden: &B::Tensor,
        forward: &Self::ForwardContext,
    ) -> ExpertPass {
        self.kernel().expert_pass(
            self.address(group, index)
                .expect("validated provider traversal"),
            hidden,
            forward,
        )
    }

    fn routed_observation_points(
        &self,
        group: usize,
        index: usize,
    ) -> Result<Option<RoutedObservationPoints>, Self::Error> {
        self.kernel().routing_points(self.address(group, index)?)
    }

    fn routed_unit_observations(&self) -> bool {
        self.kernel().routed_unit_observations()
    }

    fn routed_sparse_observations(&self) -> bool {
        self.kernel().routed_sparse_observations()
    }

    fn forward_unit_with_provider<P>(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        pass: ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error>
    where
        P: ParameterProvider<B>,
        P::Error: std::fmt::Display,
    {
        let address = self.address(group, index)?;
        self.kernel_mut().execute_routed(
            address, unit, hidden, state, forward, pass, provider, context,
        )
    }

    fn forward_unit_observed_with_provider<P, O>(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        pass: ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Self::Error>
    where
        P: ParameterProvider<B>,
        P::Error: std::fmt::Display,
        O: ActivationObserver<B::Tensor, Self::Error> + ?Sized,
        Self::Error: std::fmt::Display,
    {
        let address = self.address(group, index)?;
        self.kernel_mut().execute_routed_observed(
            address, unit, hidden, state, forward, pass, provider, context, observer,
        )
    }
}

/// Explicit parallel provider-aware component equations.
pub trait ParallelRoutedComponentKernel<B, S>:
    RoutedComponentKernel<B, S> + ParallelComponentKernel<B, S>
where
    B: eredu_nn::GroupedNeuralBackend,
    S: RuntimeState<B>,
{
    /// Reports internal component hooks on the parallel provider call.
    fn parallel_routed_unit_observations(&self) -> bool {
        false
    }

    /// Reports sparse expert hooks on the parallel provider call.
    fn parallel_routed_sparse_observations(&self) -> bool {
        false
    }

    /// Executes one parallel unit through its admitted expert provider.
    fn execute_parallel_routed<P>(
        &mut self,
        address: ExecutionUnitAddress,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        pass: ExpertPass,
        provider: &mut P,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error>
    where
        P: crate::TensorParallelParameterProvider<B>,
        P::Error: std::fmt::Display;

    /// Executes the same parallel provider-backed unit with internal instrumentation.
    fn execute_parallel_routed_observed<P, O>(
        &mut self,
        address: ExecutionUnitAddress,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        pass: ExpertPass,
        provider: &mut P,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Self::Error>
    where
        P: crate::TensorParallelParameterProvider<B>,
        P::Error: std::fmt::Display,
        O: ActivationObserver<B::Tensor, Self::Error> + ?Sized,
        Self::Error: std::fmt::Display,
    {
        match self.routing_points(address)? {
            Some(point) => {
                let mut observed = ObservedExpertProvider::new(provider, observer, point);
                self.execute_parallel_routed(
                    address,
                    unit,
                    hidden,
                    state,
                    forward,
                    pass,
                    &mut observed,
                    parallel,
                    context,
                )
            }
            None => self.execute_parallel_routed(
                address, unit, hidden, state, forward, pass, provider, parallel, context,
            ),
        }
    }
}

impl<B, D, K, S, R> ParallelRoutedLayeredArchitecture<B, S> for ComponentAdapter<D, K, R>
where
    B: eredu_nn::GroupedNeuralBackend,
    D: Borrow<ComponentDefinition> + 'static,
    S: RuntimeState<B>,
    R: ComponentResourceAccess<B, K> + 'static,
    K: ParallelRoutedComponentKernel<B, S> + 'static,
{
    fn parallel_routed_unit_observations(&self) -> bool {
        self.kernel().parallel_routed_unit_observations()
    }

    fn parallel_routed_sparse_observations(&self) -> bool {
        self.kernel().parallel_routed_sparse_observations()
    }

    fn forward_unit_parallel_with_provider<P>(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        pass: ExpertPass,
        provider: &mut P,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error>
    where
        P: crate::TensorParallelParameterProvider<B>,
        P::Error: std::fmt::Display,
    {
        let address = self.address(group, index)?;
        self.kernel_mut().execute_parallel_routed(
            address, unit, hidden, state, forward, pass, provider, parallel, context,
        )
    }

    fn forward_unit_parallel_observed_with_provider<P, O>(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        pass: ExpertPass,
        provider: &mut P,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Self::Error>
    where
        P: crate::TensorParallelParameterProvider<B>,
        P::Error: std::fmt::Display,
        O: ActivationObserver<B::Tensor, Self::Error> + ?Sized,
        Self::Error: std::fmt::Display,
    {
        let address = self.address(group, index)?;
        self.kernel_mut().execute_parallel_routed_observed(
            address, unit, hidden, state, forward, pass, provider, parallel, context, observer,
        )
    }
}
