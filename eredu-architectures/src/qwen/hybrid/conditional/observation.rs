//! Target component seams over the ordinary conditional block and media drivers.
use super::*;

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> ConditionalLayeredModel<B> {
    fn observed_target_entry<O>(
        &self,
        group: usize,
        index: usize,
        hidden: &B::Tensor,
        forward: &mut ConditionalForwardContext<B::Tensor>,
        observer: &mut O,
    ) -> Result<Option<B::Tensor>, Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        if group != 1 || index != 0 {
            return Ok(None);
        }
        // Group entry has already assembled text and media. Observe the value
        // consumed by the first target block, including on a later PP rank.
        let mut borrowed = eredu_runtime::BorrowedActivationObserver(observer);
        let mut instrumentation =
            crate::decoder::ComponentInstrumentation::new("readout", &mut borrowed);
        let hidden = instrumentation.apply("embedding", hidden.clone())?;
        forward.embedded = Some(hidden.clone());
        Ok(Some(hidden))
    }

    pub(super) fn forward_target_observed<S, P, O>(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut ConditionalUnit<B>,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut ConditionalForwardContext<B::Tensor>,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        if group == 0 {
            return <Self as LayeredArchitecture<B, S>>::forward_unit(
                self, group, index, unit, hidden, state, forward, context,
            );
        }
        if group != 1 {
            return self.forward_unit_with_provider(
                group, index, unit, hidden, state, forward, provider, context,
            );
        }
        let entry = self.observed_target_entry(group, index, hidden, forward, observer)?;
        let hidden = entry.as_ref().unwrap_or(hidden);
        let ConditionalUnit::Target(block) = unit else {
            return Err(Error::backend(
                "conditional observed target unit/group mismatch",
            ));
        };
        let path = format!("model.layers.{index}");
        let points = eredu_runtime::RoutedObservationPoints::new(
            eredu_runtime::RoutedBankId::new(0),
            format_args!("{path}.mlp"),
            self.parsed.text.num_experts, None)?;
        let lane = state
            .layer(self.state_index(group, index)?)
            .map_err(Error::backend)?;
        let output = block.forward_components_with_provider(
            &path,
            points,
            hidden,
            forward.mask.as_ref(),
            lane,
            context,
            provider,
            observer,
        )?;
        let mut borrowed = eredu_runtime::BorrowedActivationObserver(observer);
        self.add_deepstack(
            group,
            index,
            output,
            forward,
            context,
            &mut crate::decoder::ComponentInstrumentation::new(&path, &mut borrowed),
        )
    }

    pub(super) fn finish_target_observed<O>(
        &mut self,
        hidden: &B::Tensor,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        let mut borrowed = eredu_runtime::BorrowedActivationObserver(observer);
        let mut instrumentation =
            crate::decoder::ComponentInstrumentation::new("readout", &mut borrowed);
        match parallel {
            Some(parallel) => self.static_modules.text.finish_parallel_instrumented(
                hidden,
                parallel,
                context,
                &mut instrumentation,
            ),
            None => {
                self.static_modules
                    .text
                    .finish_instrumented(hidden, context, &mut instrumentation)
            }
        }
    }
}

impl<B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend>
    ConditionalLayeredModel<B>
{
    pub(super) fn forward_target_parallel_observed<S, P, O>(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut ConditionalUnit<B>,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut ConditionalForwardContext<B::Tensor>,
        provider: &mut P,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
        P: eredu_runtime::TensorParallelRoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        if group == 0 {
            return <Self as ParallelLayeredArchitecture<B, S>>::forward_unit_parallel(
                self, group, index, unit, hidden, state, forward, parallel, context,
            );
        }
        if group != 1 {
            return self.forward_unit_with_provider_parallel(
                group, index, unit, hidden, state, forward, provider, parallel, context,
            );
        }
        let entry = self.observed_target_entry(group, index, hidden, forward, observer)?;
        let hidden = entry.as_ref().unwrap_or(hidden);
        let ConditionalUnit::Target(block) = unit else {
            return Err(Error::backend(
                "conditional observed parallel target unit/group mismatch",
            ));
        };
        let path = format!("model.layers.{index}");
        let points = eredu_runtime::RoutedObservationPoints::new(
            eredu_runtime::RoutedBankId::new(0),
            format_args!("{path}.mlp"),
            self.parsed.text.num_experts, None)?;
        let lane = state
            .layer(self.state_index(group, index)?)
            .map_err(Error::backend)?;
        let output = block.forward_components_parallel_with_provider(
            &path,
            points,
            hidden,
            forward.mask.as_ref(),
            lane,
            parallel,
            context,
            provider,
            observer,
        )?;
        let mut borrowed = eredu_runtime::BorrowedActivationObserver(observer);
        self.add_deepstack(
            group,
            index,
            output,
            forward,
            context,
            &mut crate::decoder::ComponentInstrumentation::new(&path, &mut borrowed),
        )
    }
}
