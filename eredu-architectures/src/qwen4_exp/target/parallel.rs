//! Rank-local target equations using the ordinary residual, request and state lifecycle.
use super::*;
use eredu_runtime::{
    ComponentEquations, ExpertPass, LayerRuntimeState, LayeredForwardState,
    ParallelComponentKernel, ParallelRoutedComponentKernel, ResidentExpertProvider,
    TensorParallelParameterProvider,
};

impl<B: GroupedNeuralBackend + DistributedNeuralBackend> TargetKernel<B> {
    pub(super) fn validate_parallel(&self, parallel: &B::ParallelContext) -> Result<(), Error> {
        let plan = self.tensor_partition.as_ref().ok_or_else(|| {
            Error::backend("parallel target requires a retained tensor partition")
        })?;
        if B::parallel_size(parallel) != plan.ranks() || B::parallel_rank(parallel) != plan.rank() {
            return Err(Error::backend(
                "target tensor partition differs from collective rank or size",
            ));
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn run_parallel_equation<P, S>(
        &self,
        index: usize,
        unit: &mut Unit<B>,
        hidden: &B::Tensor,
        state: &mut S,
        request: &RequestContext<B::Tensor>,
        provider: &mut P,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error>
    where
        P: TensorParallelParameterProvider<B>,
        P::Error: std::fmt::Display,
        S: AttentionCache<B::Tensor> + RuntimeStateComponents<B> + RuntimeAppendStreams<B>,
    {
        self.validate_parallel(parallel)?;
        self.execute_with(
            index,
            unit,
            hidden,
            state,
            request,
            provider,
            Some(parallel),
            context,
            instrumentation,
            |feed_forward, hidden, points, provider, instrumentation| {
                feed_forward.forward_parallel(
                    hidden,
                    points,
                    provider,
                    parallel,
                    context,
                    instrumentation,
                )
            },
        )
    }
}

impl<B, S> ParallelComponentKernel<B, S> for TargetKernel<B>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend + DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B> + RuntimeAppendStreams<B>,
{
    fn parallel_observation_hooks(&self) -> eredu_runtime::inspection::ObservationHookSupport {
        eredu_runtime::inspection::ObservationHookSupport::internal(true, true, true)
    }
    fn begin_parallel<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut S,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Error> {
        self.validate_parallel(parallel)?;
        // Vocabulary and the small residual ingress are explicitly replicated.
        self.begin(input, state, context)
    }
    fn begin_parallel_observed<'a, O>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut S,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        self.validate_parallel(parallel)?;
        self.begin_observed(input, state, context, observer)
    }
    fn execute_parallel(
        &mut self,
        address: eredu_runtime::ExecutionUnitAddress,
        unit: &mut Unit<B>,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.execute_parallel_routed(
            address,
            unit,
            hidden,
            state,
            forward,
            ExpertPass::Prefill,
            &mut ResidentExpertProvider,
            parallel,
            context,
        )
    }
    fn execute_parallel_observed<O>(
        &mut self,
        address: eredu_runtime::ExecutionUnitAddress,
        unit: &mut Unit<B>,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        self.execute_parallel_routed_observed(
            address,
            unit,
            hidden,
            state,
            forward,
            ExpertPass::Prefill,
            &mut ResidentExpertProvider,
            parallel,
            context,
            observer,
        )
    }
    fn finish_parallel(
        &mut self,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.validate_parallel(parallel)?;
        self.finish(hidden, state, forward, context)
    }
    fn finish_parallel_observed<O>(
        &mut self,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        self.validate_parallel(parallel)?;
        self.finish_observed(hidden, state, forward, context, observer)
    }
}
impl<B, S> ParallelRoutedComponentKernel<B, S> for TargetKernel<B>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend + DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B> + RuntimeAppendStreams<B>,
{
    fn parallel_routed_unit_observations(&self) -> bool {
        true
    }
    fn parallel_routed_sparse_observations(&self) -> bool {
        true
    }
    fn execute_parallel_routed<P>(
        &mut self,
        address: eredu_runtime::ExecutionUnitAddress,
        unit: &mut Unit<B>,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        _: ExpertPass,
        provider: &mut P,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        P: TensorParallelParameterProvider<B>,
        P::Error: std::fmt::Display,
    {
        let index = address.index();
        self.run_parallel_equation(
            index,
            unit,
            hidden,
            state
                .layer(self.state_index(index)?)
                .map_err(Error::backend)?,
            &forward.request,
            provider,
            parallel,
            context,
            &mut ComponentInstrumentation::disabled(),
        )
    }
    fn execute_parallel_routed_observed<P, O>(
        &mut self,
        address: eredu_runtime::ExecutionUnitAddress,
        unit: &mut Unit<B>,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        _: ExpertPass,
        provider: &mut P,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Error>
    where
        P: TensorParallelParameterProvider<B>,
        P::Error: std::fmt::Display,
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        let index = address.index();
        let mut borrowed = eredu_runtime::BorrowedActivationObserver(observer);
        let path = format!("model.layers.{}", self.spec.units[index].layer());
        self.run_parallel_equation(
            index,
            unit,
            hidden,
            state
                .layer(self.state_index(index)?)
                .map_err(Error::backend)?,
            &forward.request,
            provider,
            parallel,
            context,
            &mut ComponentInstrumentation::new(&path, &mut borrowed),
        )
    }
}
