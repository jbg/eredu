//! Portable execution of a bound pipeline stage.

use super::{
    rnnt::{RnntDimensions, RnntError, RnntPredictionState},
    rnnt_pipeline::RnntPredictorExecution,
    rnnt_predictor::{RnntPredictorInput, RnntPredictorKernel, RnntPredictorRuntimeState},
};
use eredu_core::{BackendFailure, DistributedCommitEpoch, DistributedCommitOutcome};
use eredu_nn::Tensor;
use eredu_runtime::StateLayout;
use eredu_runtime::{BoundComponentPipeline, ComponentAdapter, ComponentDefinition};

/// Typed architecture retained by a selected stage.
pub type RnntPredictorPipelineArchitecture<B> =
    ComponentAdapter<ComponentDefinition, RnntPredictorKernel<B>>;

/// Selected recurrent stage with caller-owned, unpublished local LSTM history.
pub struct RnntPredictorPipeline<P> {
    execution: P,
    state_layout: StateLayout,
}

impl<B, P> RnntPredictorExecution<B> for RnntPredictorPipeline<P>
where
    B: eredu_nn::DistributedNeuralBackend,
    P: BoundComponentPipeline<
        RnntPredictorPipelineArchitecture<B>,
        B,
        RnntPredictorRuntimeState<B::Tensor>,
    >,
{
    fn dimensions(&self) -> RnntDimensions {
        self.execution.architecture().kernel().dimensions()
    }
    fn predictor_ownership(&self) -> std::ops::Range<usize> {
        self.execution.architecture().kernel().predictor_ownership()
    }
    fn predictor_step(
        &mut self,
        input: &B::Tensor,
        parent: Option<&RnntPredictionState<B::Tensor>>,
        step: u64,
        descriptor: [u64; 7],
        epoch: DistributedCommitEpoch,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(RnntPredictionState<B::Tensor>, DistributedCommitOutcome), RnntError> {
        self.forward(input, parent, step, descriptor, epoch, context)
            .map_err(RnntError::from)
    }
}

impl<P> RnntPredictorPipeline<P> {
    /// Wraps a bound component and its retained stage facts.
    pub fn new(execution: P, state_layout: StateLayout) -> Self {
        Self {
            execution,
            state_layout,
        }
    }

    /// Borrows the bound execution for residency and disk telemetry.
    pub fn pipeline(&self) -> &P {
        &self.execution
    }

    /// Phase-local recurrent carry geometry.
    pub fn state_layout(&self) -> &StateLayout {
        &self.state_layout
    }

    /// Advances one explicit recurrent substep from an immutable parent branch.
    /// The returned prediction retains only this stage's carry and the completed
    /// final-layer readout; the caller adopts it only for a committed outcome.
    /// Initial agreement includes both the seven-word caller identity and `step`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward<B>(
        &mut self,
        input: &B::Tensor,
        parent: Option<&RnntPredictionState<B::Tensor>>,
        step: u64,
        descriptor: [u64; 7],
        epoch: DistributedCommitEpoch,
        stream: &<B::Tensor as Tensor>::Context,
    ) -> Result<(RnntPredictionState<B::Tensor>, DistributedCommitOutcome), BackendFailure>
    where
        B: eredu_nn::DistributedNeuralBackend,
        P: BoundComponentPipeline<
            RnntPredictorPipelineArchitecture<B>,
            B,
            RnntPredictorRuntimeState<B::Tensor>,
        >,
    {
        let mut state = RnntPredictorRuntimeState::for_partition(
            self.execution.architecture().kernel(),
            parent,
        )
        .map_err(BackendFailure::from_error)?;
        let mut request = [0; 8];
        request[..7].copy_from_slice(&descriptor);
        request[7] = step;
        let (output, outcome) = self.execution.execute(
            RnntPredictorInput { input, step },
            vec![input.clone()],
            &mut state,
            request,
            epoch,
            stream,
        )?;
        let prediction = state
            .into_prediction(output)
            .map_err(BackendFailure::from_error)?;
        Ok((prediction, outcome))
    }
}
