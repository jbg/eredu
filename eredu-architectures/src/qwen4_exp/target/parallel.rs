//! Rank-local target equations using the ordinary residual, request and state lifecycle.
use super::*;
use eredu_runtime::{
    ExpertPass, LayerRuntimeState, LayeredArchitecture, LayeredForwardState,
    ParallelLayeredArchitecture, ParallelRoutedLayeredArchitecture, ResidentExpertProvider,
    TensorParallelParameterProvider,
};

impl<B: GroupedNeuralBackend + DistributedNeuralBackend> TargetModel<B> {
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
    fn execute_parallel<P, S>(
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

impl<B, S> ParallelLayeredArchitecture<B, S> for TargetModel<B>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend + DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B> + RuntimeAppendStreams<B>,
{
    fn parallel_observation_hooks(&self) -> eredu_runtime::inspection::ObservationHookSupport {
        eredu_runtime::inspection::ObservationHookSupport::internal(true, true, true)
    }

    fn begin_forward_parallel<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut S,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Error> {
        self.validate_parallel(parallel)?;
        // Vocabulary and the small residual ingress are explicitly replicated.
        self.begin_forward(input, state, context)
    }

    fn begin_forward_parallel_observed<'a, O>(
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
        self.begin_forward_observed(input, state, context, observer)
    }

    fn forward_unit_parallel(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Unit<B>,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.forward_unit_parallel_with_provider(
            group,
            index,
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

    fn forward_unit_parallel_observed<O>(
        &mut self,
        group: usize,
        index: usize,
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
        self.forward_unit_parallel_observed_with_provider(
            group,
            index,
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

    fn finish_forward_parallel(
        &mut self,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.validate_parallel(parallel)?;
        self.finish_forward(hidden, state, forward, context)
    }

    fn finish_forward_parallel_observed<O>(
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
        self.finish_forward_observed(hidden, state, forward, context, observer)
    }
}

impl<B, S> ParallelRoutedLayeredArchitecture<B, S> for TargetModel<B>
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

    fn forward_unit_parallel_with_provider<P>(
        &mut self,
        group: usize,
        index: usize,
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
        self.decoder.unit_path(group, index)?;
        self.execute_parallel(
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

    fn forward_unit_parallel_observed_with_provider<P, O>(
        &mut self,
        group: usize,
        index: usize,
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
        self.decoder.unit_path(group, index)?;
        let mut borrowed = eredu_runtime::BorrowedActivationObserver(observer);
        let path = format!("model.layers.{}", self.spec.units[index].layer());
        self.execute_parallel(
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
