//! Component hooks share the ordinary text, media assembly, and readout drivers.
use super::*;
use crate::decoder::ComponentInstrumentation;

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> LayeredModel<B> {
    fn observed_text_entry<O>(
        &self,
        index: usize,
        hidden: &B::Tensor,
        observer: &mut O,
    ) -> Result<Option<B::Tensor>, Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        if index != 0 {
            return Ok(None);
        }
        // The first text unit consumes the completed text/media assembly. Later
        // pipeline stages must not report their received residual as an embedding.
        let mut borrowed = eredu_runtime::BorrowedActivationObserver(observer);
        ComponentInstrumentation::new("readout", &mut borrowed)
            .apply("embedding", hidden.clone())
            .map(Some)
    }

    pub(super) fn finish_text_observed<O>(
        &mut self,
        hidden: &B::Tensor,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        if parallel.is_some() && self.parallel_geometry.is_none() {
            return Err(Error::backend(
                "Muse-Glimmer model was not built with rank-local geometry",
            ));
        }
        let mut borrowed = eredu_runtime::BorrowedActivationObserver(observer);
        self.static_modules.text.logits_instrumented(
            hidden,
            parallel,
            context,
            &mut ComponentInstrumentation::new("readout", &mut borrowed),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward_text_observed<S, P, O>(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Unit<B>,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut ForwardContext<B::Tensor>,
        pass: ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor>,
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        if group != 1 {
            return <Self as LayeredArchitecture<B, S>>::forward_unit(
                self, group, index, unit, hidden, state, forward, context,
            );
        }
        let Unit::Text(unit) = unit else {
            return Err(Error::backend("Muse-Glimmer observed unit/group mismatch"));
        };
        let entry = self.observed_text_entry(index, hidden, observer)?;
        let hidden = entry.as_ref().unwrap_or(hidden);
        let path = format!("model.layers.{index}");
        let mut borrowed = eredu_runtime::BorrowedActivationObserver(observer);
        unit.forward_with_provider_observed(
            hidden,
            forward.mask.as_ref(),
            Some(
                state
                    .layer(self.local_state_ordinal(index)?)
                    .map_err(Error::backend)?,
            ),
            pass,
            provider,
            context,
            &mut ComponentInstrumentation::new(&path, &mut borrowed),
        )
    }
}

impl<B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend>
    LayeredModel<B>
{
    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward_text_parallel_observed<S, P, O>(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Unit<B>,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut ForwardContext<B::Tensor>,
        pass: ExpertPass,
        provider: &mut P,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor>,
        P: eredu_runtime::TensorParallelRoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        if group != 1 {
            return <Self as ParallelLayeredArchitecture<B, S>>::forward_unit_parallel(
                self, group, index, unit, hidden, state, forward, parallel, context,
            );
        }
        let Unit::Text(unit) = unit else {
            return Err(Error::backend("Muse-Glimmer observed unit/group mismatch"));
        };
        let entry = self.observed_text_entry(index, hidden, observer)?;
        let hidden = entry.as_ref().unwrap_or(hidden);
        let path = format!("model.layers.{index}");
        let mut borrowed = eredu_runtime::BorrowedActivationObserver(observer);
        unit.forward_parallel_with_provider_observed(
            hidden,
            forward.mask.as_ref(),
            Some(
                state
                    .layer(self.local_state_ordinal(index)?)
                    .map_err(Error::backend)?,
            ),
            pass,
            provider,
            parallel,
            context,
            &mut ComponentInstrumentation::new(&path, &mut borrowed),
        )
    }
}
