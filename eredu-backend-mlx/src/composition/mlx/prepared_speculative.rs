//! Whole-session MLX speculative generation capability.

use eredu_core::speculative::embedded::EmbeddedPredictionOperators;
use safemlx::{error::Exception, transforms::async_eval_with_event, Array, Stream};

use super::{
    speculative::{MlxAssistantBinder, MlxDraftStorage, MlxSpeculativeSamplingPrimitives},
    MlxBackend,
};
use crate::backend::nn::tensor::TokenValidationScope;
use crate::MlxTensor;

impl<Dispatch> eredu_core::session_authority::SpeculativeTargetNative for MlxBackend<'_, Dispatch> {
    type TargetOperators = super::speculative::autoregressive::MlxAutoregressiveMechanisms;
}

impl<Dispatch: 'static> eredu_core::session_authority::DraftNativeResources
    for MlxBackend<'_, Dispatch>
{
    type DraftQueue = Stream;
    type AssistantBinder = MlxAssistantBinder;
    type AssistantDispatch = Dispatch;
    type AssistantOperators = super::speculative::external::MlxExternalAssistantMechanisms;
    type DraftStorage<P: 'static> = MlxDraftStorage<P>;
}

impl<'world, Dispatch>
    eredu_core::speculative::installed::ResidentSpeculation<MlxBackend<'world, Dispatch>>
    for super::speculative::autoregressive::MlxAutoregressiveMechanisms
{
    type Session = super::session::model_session::MlxSessionMechanisms;
    type Sampling = MlxSpeculativeSamplingPrimitives;

    fn target_queue(backend: &MlxBackend<'world, Dispatch>) -> Stream {
        backend.stream().clone()
    }
}

#[doc(hidden)]
pub struct MlxEmbeddedPredictionMechanisms;

impl EmbeddedPredictionOperators for MlxEmbeddedPredictionMechanisms {
    type Tensor = MlxTensor;
    type Logits = Array;
    type Completion = super::speculative::MlxSpeculativeCompletion;
    type Error = Exception;
    type Capture =
        crate::composition::mlx::session::bounded_capture::MlxSpeculativeCaptureMechanisms;

    fn token(token: u32, _context: &Stream) -> Result<MlxTensor, Exception> {
        let token = i32::try_from(token)
            .map_err(|_| Exception::custom("prediction token exceeds int32 storage"))?;
        Ok(MlxTensor::from_array(Array::from_slice(&[token], &[1, 1])))
    }

    fn validate<T>(operation: impl FnOnce() -> Result<T, Exception>) -> Result<T, Exception> {
        let scope = TokenValidationScope::begin()?;
        let output = operation()?;
        let validations = scope.finish();
        if !validations.is_empty() {
            async_eval_with_event(validations.arrays())?.synchronize()?;
            validations.validate_completed()?;
        }
        Ok(output)
    }

    fn invalid(message: impl Into<String>) -> Exception {
        Exception::custom(message.into())
    }

    fn failure(error: eredu_core::BackendFailure) -> Exception {
        Exception::from_source(error)
    }

    fn logits_estimate(logits: &Array) -> Option<eredu_core::execution_control::SnapshotEstimate> {
        super::speculative::state_snapshot::estimate_array(logits)
    }

    fn tensor_estimate(
        value: &MlxTensor,
    ) -> Option<eredu_core::execution_control::SnapshotEstimate> {
        super::speculative::state_snapshot::estimate_array(value.as_array())
    }

    fn copy_tensor(value: &MlxTensor, queue: &Stream) -> Result<MlxTensor, Exception> {
        super::speculative::state_snapshot::copy_array(value.as_array(), queue)
            .map(MlxTensor::from_array)
    }

    fn logits_row(
        value: &Self::Tensor,
        row: usize,
        queue: &Stream,
    ) -> Result<Self::Logits, Self::Error> {
        eredu_nn::sequence::row(value, row, false, queue)
            .map(MlxTensor::into_array)
            .map_err(Exception::from_source)
    }

    fn target_tokens(
        tokens: &[u32],
        destination: &Stream,
        crosses_devices: bool,
    ) -> Result<Self::Tensor, Self::Error> {
        let width = i32::try_from(tokens.len())
            .map_err(|_| Exception::custom("prediction token count exceeds i32"))?;
        let mut value = Array::from_slice(tokens, &[1, width]);
        if crosses_devices {
            value = value.copy(destination)?;
        }
        Ok(MlxTensor::from_array(value))
    }

    fn submit_verification(roots: &[&MlxTensor]) -> Result<Self::Completion, Exception> {
        super::speculative::MlxSpeculativeCompletion::submit(
            roots.iter().map(|value| value.as_array()),
        )
    }
}
