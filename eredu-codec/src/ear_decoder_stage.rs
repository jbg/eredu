//! Portable execution of a bound pipeline stage.

use crate::{
    ear_decoder_layered::{DecoderKernel, DecoderRuntimeState},
    ear_pipeline::DecoderPipelineExecution,
};
use eredu_core::{BackendFailure, DistributedCommitEpoch, DistributedCommitOutcome};
use eredu_nn::{NeuralBackend, Tensor};
use eredu_runtime::StateLayout;
use eredu_runtime::{BoundComponentPipeline, ComponentAdapter, ComponentDefinition};

/// Typed architecture retained by a selected stage.
pub type EarDecoderPipelineArchitecture<B> =
    ComponentAdapter<ComponentDefinition, DecoderKernel<B>>;

/// One codec-selected streaming decoder stage. The shared distributed driver owns
/// parameter leases, transport, exact completion and commit agreement.
pub struct EarDecoderPipeline<B: NeuralBackend, P> {
    execution: P,
    state_layout: StateLayout,
    initial: DecoderRuntimeState<B::Tensor>,
}
impl<B, P> EarDecoderPipeline<B, P>
where
    B: NeuralBackend,
{
    /// Wraps a bound component and its retained stage facts.
    pub fn new(
        execution: P,
        state_layout: StateLayout,
        initial: DecoderRuntimeState<B::Tensor>,
    ) -> Self {
        Self {
            execution,
            state_layout,
            initial,
        }
    }

    /// Borrows the bound execution for residency and disk telemetry.
    pub fn pipeline(&self) -> &P {
        &self.execution
    }

    /// Exact stage-local convolution-state geometry, including stateless units.
    pub fn state_layout(&self) -> &StateLayout {
        &self.state_layout
    }
}

impl<B, P> DecoderPipelineExecution<B> for EarDecoderPipeline<B, P>
where
    B: eredu_nn::DistributedNeuralBackend,
    P: BoundComponentPipeline<EarDecoderPipelineArchitecture<B>, B, DecoderRuntimeState<B::Tensor>>,
{
    type State = DecoderRuntimeState<B::Tensor>;

    fn config(&self) -> &crate::ear_decoder::DecoderConfig {
        self.execution.architecture().kernel().config()
    }

    fn initial_state(&self) -> Result<Self::State, crate::Error> {
        Ok(self.initial.clone())
    }

    fn validate_state(
        &self,
        parent: &Self::State,
        batch: i32,
        finished: bool,
    ) -> Result<(), crate::Error> {
        parent.validate_stream_partition(
            self.execution.architecture().kernel(),
            batch,
            if finished {
                crate::streaming::StreamPhase::Finished
            } else {
                crate::streaming::StreamPhase::AcceptingInput
            },
        )
    }

    fn finish(&self, parent: &Self::State) -> Result<Self::State, crate::Error> {
        parent.finish_partition(self.execution.architecture().kernel())
    }

    fn forward(
        &mut self,
        latent: &B::Tensor,
        state: &mut Self::State,
        descriptor: [u64; 8],
        epoch: DistributedCommitEpoch,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(B::Tensor, DistributedCommitOutcome), BackendFailure> {
        self.execution.execute(
            latent,
            vec![latent.clone()],
            state,
            descriptor,
            epoch,
            context,
        )
    }
}
