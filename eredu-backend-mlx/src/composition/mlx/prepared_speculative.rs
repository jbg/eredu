//! Whole-session MLX speculative generation capability.

use eredu_core::speculative::embedded::EmbeddedPredictionOperators;
use safemlx::{
    error::Exception, ops::indexing::TryIndexOp, transforms::async_eval_with_event, Array, Stream,
};

use super::{
    speculative::{
        scheduler::component_timing_enabled, MlxAssistantBinder, MlxDraftStorage,
        MlxSpeculativeSamplingPrimitives,
    },
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

    fn collect_component_timings() -> bool {
        component_timing_enabled()
    }
}

#[doc(hidden)]
pub struct MlxEmbeddedPredictionMechanisms;

impl EmbeddedPredictionOperators for MlxEmbeddedPredictionMechanisms {
    type Tensor = MlxTensor;
    type Logits = Array;
    type Completion = super::speculative::MlxSpeculativeCompletion;
    type Error = Exception;
    type Telemetry = super::speculative::scheduler::SpeculativeComponentTimings;
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

    fn take_telemetry() -> Result<Self::Telemetry, Exception> {
        Ok(Self::Telemetry::default())
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

    fn sequence_len(value: &Self::Tensor) -> Result<usize, Self::Error> {
        usize::try_from(value.as_array().dim(1))
            .map_err(|_| Exception::custom("prediction sequence length exceeds usize"))
    }

    fn logits_row(
        value: &Self::Tensor,
        row: usize,
        queue: &Stream,
    ) -> Result<Self::Logits, Self::Error> {
        let row =
            i32::try_from(row).map_err(|_| Exception::custom("prediction row exceeds i32"))?;
        value.as_array().try_index_device((.., row, ..), queue)
    }

    fn tensor_row(
        value: &Self::Tensor,
        row: usize,
        queue: &Stream,
    ) -> Result<Self::Tensor, Self::Error> {
        let row =
            i32::try_from(row).map_err(|_| Exception::custom("prediction row exceeds i32"))?;
        value
            .as_array()
            .try_index_device((.., row..row + 1, ..), queue)
            .map(MlxTensor::from_array)
    }

    fn tensor_prefix(
        value: &Self::Tensor,
        end: usize,
        queue: &Stream,
    ) -> Result<Self::Tensor, Self::Error> {
        let end =
            i32::try_from(end).map_err(|_| Exception::custom("prediction prefix exceeds i32"))?;
        value
            .as_array()
            .try_index_device((.., ..end, ..), queue)
            .map(MlxTensor::from_array)
    }

    fn token_range(
        value: &Self::Tensor,
        start: usize,
        end: usize,
        queue: &Stream,
    ) -> Result<Self::Tensor, Self::Error> {
        let start = i32::try_from(start)
            .map_err(|_| Exception::custom("prediction token start exceeds i32"))?;
        let end = i32::try_from(end)
            .map_err(|_| Exception::custom("prediction token end exceeds i32"))?;
        value
            .as_array()
            .try_index_device((.., start..end), queue)
            .map(MlxTensor::from_array)
    }

    fn target_tokens(
        tokens: &[u32],
        destination: Option<&Stream>,
    ) -> Result<Self::Tensor, Self::Error> {
        let width = i32::try_from(tokens.len())
            .map_err(|_| Exception::custom("prediction token count exceeds i32"))?;
        let mut value = Array::from_slice(tokens, &[1, width]);
        if let Some(queue) = destination {
            value = value.copy(queue)?;
        }
        Ok(MlxTensor::from_array(value))
    }

    fn submit_verification(roots: &[&MlxTensor]) -> Result<Self::Completion, Exception> {
        super::speculative::MlxSpeculativeCompletion::submit(
            roots.iter().map(|value| value.as_array()),
        )
    }
}
