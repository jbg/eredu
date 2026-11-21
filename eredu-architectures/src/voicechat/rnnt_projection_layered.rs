//! Bound RNNT execution through shared parameter and traversal mechanisms.

use eredu_core::SubmissionBackend;

use super::{
    rnnt::{
        RnntDimensions, RnntError, RnntExecution, RnntPredictionState, RnntProjectionExecution,
    },
    rnnt_layered::{runtime_error, RnntInput, RnntKernel, RnntUnit},
};
use eredu_nn::{DistributedNeuralBackend, NeuralBackend, Tensor};
use eredu_runtime::{
    BoundStatelessComponent, ComponentAdapter, ComponentDefinition, LayeredTensorCohort,
    LayerwisePolicy,
};

/// Bound RNNT phases with optional recurrent parameter ownership.
pub type BoundRnnt<B, P, C, const RECURRENT: bool = true> = BoundStatelessComponent<
    ComponentAdapter<ComponentDefinition, RnntKernel<B, RECURRENT>>,
    B,
    P,
    C,
>;

/// Bound projection-only RNNT phases.
pub type RnntProjection<B, P, C> = BoundRnnt<B, P, C, false>;

impl<B, P, C, const RECURRENT: bool> RnntProjectionExecution<B> for BoundRnnt<B, P, C, RECURRENT>
where
    B: DistributedNeuralBackend
        + SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    P: LayerwisePolicy<B, RnntUnit<B>>,
    P::Error: std::error::Error + Send + Sync + 'static,
    C: LayeredTensorCohort<B>,
{
    fn dimensions(&self) -> RnntDimensions {
        self.architecture().kernel().dimensions()
    }

    fn embed_token(
        &mut self,
        token: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, RnntError> {
        self.invoke(RnntInput::Embed(token), context)
            .map(|(output, _)| output)
            .map_err(runtime_error)
    }

    fn encode_frame(
        &mut self,
        encoder: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, RnntError> {
        self.invoke(RnntInput::Encode(encoder), context)
            .map(|(output, _)| output)
            .map_err(runtime_error)
    }

    fn project_prediction(
        &mut self,
        prediction: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, RnntError> {
        self.invoke(RnntInput::Prediction(prediction), context)
            .map(|(output, _)| output)
            .map_err(runtime_error)
    }

    fn project_output(
        &mut self,
        hidden: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, RnntError> {
        self.invoke(RnntInput::Output(hidden), context)
            .map(|(output, _)| output)
            .map_err(runtime_error)
    }
}

impl<B, P, C> RnntExecution<B> for BoundRnnt<B, P, C>
where
    B: DistributedNeuralBackend
        + SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    P: LayerwisePolicy<B, RnntUnit<B>>,
    P::Error: std::error::Error + Send + Sync + 'static,
    C: LayeredTensorCohort<B>,
{
    fn dimensions(&self) -> RnntDimensions {
        RnntProjectionExecution::dimensions(self)
    }

    fn embed_token(
        &mut self,
        token: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, RnntError> {
        RnntProjectionExecution::embed_token(self, token, context)
    }

    fn predictor_step(
        &mut self,
        input: &B::Tensor,
        previous: Option<&RnntPredictionState<B::Tensor>>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<RnntPredictionState<B::Tensor>, RnntError> {
        self.invoke(RnntInput::Predict { input, previous }, context)
            .map_err(runtime_error)?
            .1
            .into_predictor()
    }

    fn encode_frame(
        &mut self,
        encoder: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, RnntError> {
        RnntProjectionExecution::encode_frame(self, encoder, context)
    }

    fn project_prediction(
        &mut self,
        prediction: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, RnntError> {
        RnntProjectionExecution::project_prediction(self, prediction, context)
    }

    fn project_output(
        &mut self,
        hidden: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, RnntError> {
        RnntProjectionExecution::project_output(self, hidden, context)
    }
}
