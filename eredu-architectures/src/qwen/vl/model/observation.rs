//! Component seams over the ordinary Qwen decoder and prepared vision additions.
use super::*;
use crate::decoder::ComponentInstrumentation;

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> LayeredModel<B> {
    fn component_text_entry<O>(
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
        let mut borrowed = eredu_runtime::BorrowedActivationObserver(observer);
        ComponentInstrumentation::new("readout", &mut borrowed)
            .apply("embedding", hidden.clone())
            .map(Some)
    }

    pub(super) fn finish_components<O>(
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
        let mut instrumentation = ComponentInstrumentation::new("readout", &mut borrowed);
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

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> LayeredModel<B> {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward_components<S, P, O>(
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
        S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        if group != 1 {
            return self.forward_unit_with_provider(
                group, index, unit, hidden, state, forward, provider, context,
            );
        }
        let entry = self.component_text_entry(index, hidden, observer)?;
        let hidden = entry.as_ref().unwrap_or(hidden);
        let Unit::Text(block) = unit else {
            return Err(Error::backend("Qwen3-VL observed text unit/group mismatch"));
        };
        if forward
            .deepstack
            .get(index)
            .is_some_and(|features| features.shape() != hidden.shape())
        {
            self.ensure_visual_mask(forward, context)?;
        }
        let path = format!("{}.layers.{index}", self.args.text.parameter_root);
        let points = self.args.text.routed_observation_points(&path, index);
        let state_ordinal = self.text_state_ordinal(index)?;
        let mut borrowed = eredu_runtime::BorrowedActivationObserver(observer);
        let mut instrumentation = ComponentInstrumentation::new(&path, &mut borrowed);
        let output = block.forward_routed_observed(
            index,
            AttentionInput {
                hidden,
                mask: forward.mask.as_ref(),
                cache: Some(state.layer(state_ordinal).map_err(Error::backend)?),
                allow_sliding_prefill: true,
                rotary_position: Some(RotaryPosition::Embeddings {
                    cosine: &forward.rotary()?.0,
                    sine: &forward.rotary()?.1,
                }),
            },
            pass,
            provider,
            context,
            &mut instrumentation,
            points,
        )?;
        self.add_deepstack(index, output, forward, context, &mut instrumentation)
    }
}

impl<B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend>
    LayeredModel<B>
{
    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward_components_parallel<S, P, O>(
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
        S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
        P: eredu_runtime::TensorParallelRoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        if group != 1 {
            return self.forward_unit_with_provider_parallel(
                group, index, unit, hidden, state, forward, provider, parallel, context,
            );
        }
        let entry = self.component_text_entry(index, hidden, observer)?;
        let hidden = entry.as_ref().unwrap_or(hidden);
        let Unit::Text(block) = unit else {
            return Err(Error::backend("Qwen3-VL observed text unit/group mismatch"));
        };
        if forward
            .deepstack
            .get(index)
            .is_some_and(|features| features.shape() != hidden.shape())
        {
            self.ensure_visual_mask(forward, context)?;
        }
        let path = format!("{}.layers.{index}", self.args.text.parameter_root);
        let points = self.args.text.routed_observation_points(&path, index);
        let state_ordinal = self.text_state_ordinal(index)?;
        let mut borrowed = eredu_runtime::BorrowedActivationObserver(observer);
        let mut instrumentation = ComponentInstrumentation::new(&path, &mut borrowed);
        let output = block.forward_routed_parallel_observed(
            index,
            AttentionInput {
                hidden,
                mask: forward.mask.as_ref(),
                cache: Some(state.layer(state_ordinal).map_err(Error::backend)?),
                allow_sliding_prefill: true,
                rotary_position: Some(RotaryPosition::Embeddings {
                    cosine: &forward.rotary()?.0,
                    sine: &forward.rotary()?.1,
                }),
            },
            pass,
            provider,
            parallel,
            context,
            &mut instrumentation,
            points,
        )?;
        self.add_deepstack(index, output, forward, context, &mut instrumentation)
    }
}
