//! Supplied embeddings and typed transformer boundaries in the shared distributed driver.
use super::*;
use eredu_runtime::PipelinePartitionArchitecture;
use eredu_runtime::{ArchitectureBoundary, LayeredPartitionInput};

impl<B, C, D> PipelinePartitionArchitecture<B, TransformerRuntimeState<'_, B, C, D>>
    for TransformerArchitecture<B>
where
    B: NeuralBackend,
    C: AttentionCache<B::Tensor> + RuntimeLayerState<B>,
    D: DerefMut<Target = [C]>,
{
    fn partition_input_dimensions(
        &self,
        input: &Self::Input<'_>,
    ) -> Result<(i32, i32), Self::Error> {
        if input.embedded.shape().len() != 3 {
            return Err(TransformerError::Configuration(
                "TTS pipeline ingress must be batch, sequence and hidden features",
            ));
        }
        Ok((input.embedded.dim(0), input.embedded.dim(1)))
    }

    fn begin_pipeline_partition<'a, O>(
        &mut self,
        input: Self::Input<'a>,
        incoming: Option<(
            B::Tensor,
            <Self::Boundary as ArchitectureBoundary>::Boundary<B::Tensor>,
        )>,
        state: &mut TransformerRuntimeState<'_, B, C, D>,
        expected: Option<&StateLayout>,
        first_state_ordinal: usize,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        _observer: &mut O,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Self::Error> + ?Sized,
    {
        if first_state_ordinal != 0 {
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
        self.begin_prepared_partition(ingress, keys, state, expected, parallel, context)
    }

    fn partition_output_width(&self) -> i32 {
        self.config.hidden_size()
    }

    fn partition_retained_state_ordinals(
        &self,
        group: usize,
        index: usize,
        local_ordinal: usize,
    ) -> std::ops::Range<usize> {
        <Self as LayeredArchitecture<B, TransformerRuntimeState<'_, B, C, D>>>::retained_state_ordinals(
            self, group, index, local_ordinal,
        )
    }

    fn partition_tensor_vocabulary_sharded(&self) -> bool {
        false
    }
}
