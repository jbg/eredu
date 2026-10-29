//! Supplied embeddings and typed transformer boundaries in the shared distributed driver.
use super::*;
use eredu_runtime::{ArchitectureBoundary, LayeredPartitionInput};

impl<B, C, D> eredu_runtime::PipelineComponentKernel<B, TransformerRuntimeState<'_, B, C, D>>
    for TransformerKernel<B>
where
    B: NeuralBackend,
    C: AttentionCache<B::Tensor> + RuntimeLayerState<B>,
    D: DerefMut<Target = [C]>,
{
    fn input_dimensions(&self, input: &Self::Input<'_>) -> Result<(i32, i32), Self::Error> {
        if input.embedded.shape().len() != 3 {
            return Err(TransformerError::Configuration(
                "TTS pipeline ingress must be batch, sequence and hidden features",
            ));
        }
        Ok((input.embedded.dim(0), input.embedded.dim(1)))
    }

    fn begin_pipeline<'a, O>(
        &mut self,
        input: Self::Input<'a>,
        incoming: Option<(
            B::Tensor,
            <Self::Boundary as ArchitectureBoundary>::Boundary<B::Tensor>,
        )>,
        state: &mut TransformerRuntimeState<'_, B, C, D>,
        selected: eredu_runtime::ComponentPartitionContext<'_, B>,
        _observer: &mut O,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Self::Error> + ?Sized,
    {
        if selected.first_state_ordinal != 0 {
            return Err(TransformerError::Configuration(
                "TTS pipeline state must use the selected local cache origin",
            ));
        }
        let (ingress, keys) = match incoming {
            Some((hidden, auxiliary)) => {
                (LayeredPartitionInput::Hidden { hidden, auxiliary }, None)
            }
            None => (
                LayeredPartitionInput::Tokens(input.embedded),
                input.key_mask,
            ),
        };
        self.begin_prepared_partition(
            ingress,
            keys,
            state,
            selected.expected,
            selected.parallel,
            selected.native,
        )
    }

    fn output_width(&self) -> i32 {
        self.config.hidden_size()
    }

    fn tensor_vocabulary_sharded(&self) -> bool {
        false
    }
}
