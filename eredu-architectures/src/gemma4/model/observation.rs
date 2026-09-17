//! Component seams over Gemma's shared block equations and assembled text stream.
use super::*;
use crate::decoder::ComponentInstrumentation;

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> LayeredModel<B> {
    // All ordinary and observed entry points must share this preparation.
    pub(super) fn text_block_inputs(
        &self,
        index: usize,
        hidden: &B::Tensor,
        forward: &ForwardContext<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(Option<B::Tensor>, Option<B::Tensor>), Error> {
        let policy = self
            .args
            .text
            .layer_policy(index)
            .ok_or_else(|| Error::backend("missing Gemma 4 layer policy"))?;
        // Full attention uses the absolute prefix. Sliding attention instead
        // consumes its actual visible cache history in Attention::attend; a
        // prefix-wide mask cannot broadcast after that history was truncated.
        let mask =
            if forward.mask.is_none() && hidden.dim(1) > 1 && policy.attention.window().is_none() {
                Some(B::causal_mask(
                    hidden.dim(1),
                    forward.position_offset,
                    None,
                    context,
                )?)
            } else {
                None
            };
        let per_layer_input = forward
            .per_layer_inputs
            .as_ref()
            .map(|inputs| {
                inputs.index(
                    &[
                        Index::Full,
                        Index::Full,
                        Index::At(index as i32),
                        Index::Full,
                    ],
                    context,
                )
            })
            .transpose()?;
        Ok((mask, per_layer_input))
    }

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
        self.static_modules.text.project_logits_instrumented(
            hidden,
            self.args.text.final_logit_softcapping,
            parallel,
            context,
            &mut ComponentInstrumentation::new("readout", &mut borrowed),
        )
    }

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
        S::LayerState: AttentionCache<B::Tensor>,
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        if group != 2 {
            return <Self as RoutedLayeredArchitecture<B, S>>::forward_unit_with_provider(
                self, group, index, unit, hidden, state, forward, pass, provider, context,
            );
        }
        let entry = self.component_text_entry(index, hidden, observer)?;
        let hidden = entry.as_ref().unwrap_or(hidden);
        let Unit::Text(block) = unit else {
            return Err(Error::backend("Gemma 4 observed text unit/group mismatch"));
        };
        let (generated_mask, per_layer_input) =
            self.text_block_inputs(index, hidden, forward, context)?;
        let state_ordinal = self.attention_state_ordinal(index, state.layout().len())?;
        let path = format!("model.language_model.layers.{index}");
        let mut borrowed = eredu_runtime::BorrowedActivationObserver(observer);
        block.forward_with_provider_observed(
            BlockInput {
                hidden,
                mask: forward.mask.as_ref().or(generated_mask.as_ref()),
                cache: Some(state.layer(state_ordinal).map_err(Error::backend)?),
                shared: &mut forward.shared,
                per_layer_input: per_layer_input.as_ref(),
                rotary_position: Some(RotaryPosition::Offset(forward.position_offset)),
            },
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
        S::LayerState: AttentionCache<B::Tensor>,
        P: eredu_runtime::TensorParallelRoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        if group != 2 {
            return <Self as ParallelRoutedLayeredArchitecture<B, S>>::forward_unit_parallel_with_provider(
                self, group, index, unit, hidden, state, forward, pass, provider, parallel, context,
            );
        }
        let entry = self.component_text_entry(index, hidden, observer)?;
        let hidden = entry.as_ref().unwrap_or(hidden);
        let Unit::Text(block) = unit else {
            return Err(Error::backend("Gemma 4 observed text unit/group mismatch"));
        };
        let (generated_mask, per_layer_input) =
            self.text_block_inputs(index, hidden, forward, context)?;
        let state_ordinal = self.attention_state_ordinal(index, state.layout().len())?;
        let path = format!("model.language_model.layers.{index}");
        let mut borrowed = eredu_runtime::BorrowedActivationObserver(observer);
        block.forward_parallel_with_provider_observed(
            BlockInput {
                hidden,
                mask: forward.mask.as_ref().or(generated_mask.as_ref()),
                cache: Some(state.layer(state_ordinal).map_err(Error::backend)?),
                shared: &mut forward.shared,
                per_layer_input: per_layer_input.as_ref(),
                rotary_position: Some(RotaryPosition::Offset(forward.position_offset)),
            },
            pass,
            provider,
            parallel,
            context,
            &mut ComponentInstrumentation::new(&path, &mut borrowed),
        )
    }
}
