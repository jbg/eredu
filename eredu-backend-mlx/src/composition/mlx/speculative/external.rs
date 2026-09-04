//! Family-neutral MLX mechanisms for architecture-owned external assistants.

use eredu_architectures::{
    composite_execution::{
        ExternalPredictionCaptureRequest, ExternalPredictionTargetCapture,
        ExternalPredictionTargetOperation,
    },
    external_assistant::{
        ExternalAssistantCache, ExternalAssistantExecutionMechanisms,
        ExternalAssistantTensorPlacement, ExternalAssistantTransfer,
    },
    ExternalAssistantArchitecture,
};
use safemlx::{
    error::Exception, ops::indexing::TryIndexOp, transforms::async_eval_with_event, Array, Stream,
};

use crate::{
    backend::{
        nn::shared::MlxNeuralBackend,
        runtime::{cache::kv::ConcatKeyValueCache, media::input::ModelInput},
    },
    composition::mlx::{
        replicated_text::{ErasedExternalPredictionExecutable, MlxPredictionTargetState},
        speculative::{
            scheduler::SpeculativeComponentTimings, MlxExternalAssistant, MlxSpeculativeCompletion,
            SpeculativeExecutionStreams,
        },
        MlxModelInput,
    },
    MlxTensor,
};

/// The single MLX mechanism adapter used by every architecture-owned external lifecycle.
pub(crate) struct MlxExternalAssistantMechanisms;

/// Architecture-owned semantic envelope over opaque native target storage.
pub(crate) type MlxExternalPredictionCache = ExternalAssistantCache<MlxPredictionTargetState>;

impl<A> ExternalAssistantExecutionMechanisms<A> for MlxExternalAssistantMechanisms
where
    A: ExternalAssistantArchitecture,
{
    type NeuralBackend = MlxNeuralBackend;
    type AttentionCache = ConcatKeyValueCache;
    type Target = dyn ErasedExternalPredictionExecutable;
    type Assistant = MlxExternalAssistant<A>;
    type Input = MlxModelInput;
    type NativeCache = MlxPredictionTargetState;
    type NativeCacheCheckpoint = MlxPredictionTargetState;
    type Tensor = MlxTensor;
    type Logits = Array;
    type Context<'a> = SpeculativeExecutionStreams<'a>;
    type Completion = MlxSpeculativeCompletion;
    type Telemetry = SpeculativeComponentTimings;
    type Error = Exception;

    fn config(assistant: &Self::Assistant) -> &A::Config {
        &assistant.config
    }

    fn module(assistant: &mut Self::Assistant) -> &mut A::Module<Self::NeuralBackend> {
        &mut assistant.module.inner
    }

    fn neural_error(error: eredu_nn::Error) -> Self::Error {
        Exception::custom(error.to_string())
    }

    fn error(message: String) -> Self::Error {
        Exception::custom(message)
    }

    fn prepared_input_cache_identity(
        input: &Self::Input,
    ) -> Result<eredu_runtime::PreparedInputCacheIdentity, Self::Error> {
        input.cache_identity().cloned().ok_or_else(|| {
            Exception::custom(
                "external speculative input is missing its prepared-input cache identity",
            )
        })
    }

    fn tensor_shape(value: &Self::Tensor) -> Result<Vec<usize>, Self::Error> {
        value
            .as_array()
            .shape()
            .iter()
            .map(|extent| {
                usize::try_from(*extent).map_err(|_| {
                    Exception::custom("external target capture has a negative tensor extent")
                })
            })
            .collect()
    }

    fn prefill_target_native<'a>(
        target: &mut Self::Target,
        request: &ExternalPredictionCaptureRequest,
        input: Self::Input,
        cache: &mut Self::NativeCache,
        _context: Self::Context<'a>,
    ) -> Result<(Self::Tensor, ExternalPredictionTargetCapture<Self::Tensor>), Self::Error> {
        input.with_borrowed(|input: ModelInput<'_>| {
            target
                .prefill_external_prediction_target(input, request, cache)
                .map_err(|error| Exception::custom(error.to_string()))
        })
    }

    fn verify_target_native<'a>(
        target: &mut Self::Target,
        request: &ExternalPredictionCaptureRequest,
        tokens: &Self::Tensor,
        cache: &mut Self::NativeCache,
        _context: Self::Context<'a>,
    ) -> Result<(Self::Tensor, ExternalPredictionTargetCapture<Self::Tensor>), Self::Error> {
        target
            .verify_external_prediction_target(tokens, request, cache)
            .map_err(|error| Exception::custom(error.to_string()))
    }

    fn checkpoint_native(
        cache: &Self::NativeCache,
    ) -> Result<Self::NativeCacheCheckpoint, Self::Error> {
        cache.deep_clone()
    }

    fn restore_checkpoint_native<'a>(
        cache: &mut Self::NativeCache,
        checkpoint: &Self::NativeCacheCheckpoint,
        context: Self::Context<'a>,
    ) -> Result<(), Self::Error> {
        cache.restore(checkpoint, context.target())
    }

    fn native_cache_len(cache: &Self::NativeCache) -> Result<i32, Self::Error> {
        i32::try_from(
            cache
                .generation()
                .map_err(|error| Exception::custom(error.to_string()))?,
        )
        .map_err(|_| Exception::custom("external target cache frontier exceeds i32"))
    }

    fn observe_tensor(
        assistant: &mut Self::Assistant,
        path: &str,
        value: Self::Tensor,
    ) -> Result<Self::Tensor, Self::Error> {
        assistant.observers.observe_tensor(path, &value)
    }

    fn observe_logits(
        assistant: &mut Self::Assistant,
        path: &str,
        value: Self::Logits,
    ) -> Result<Self::Logits, Self::Error> {
        assistant.observers.observe_logits(path, &value)
    }

    fn sequence_len(value: &Self::Tensor) -> Result<usize, Self::Error> {
        usize::try_from(value.as_array().dim(1))
            .map_err(|_| Exception::custom("external assistant sequence length exceeds usize"))
    }

    fn sequence_row<'a>(
        value: &Self::Tensor,
        row: usize,
        retain_dimension: bool,
        placement: ExternalAssistantTensorPlacement,
        context: Self::Context<'a>,
    ) -> Result<Self::Tensor, Self::Error> {
        let row = i32::try_from(row)
            .map_err(|_| Exception::custom("external assistant row exceeds i32"))?;
        let stream = match placement {
            ExternalAssistantTensorPlacement::Target => context.target(),
            ExternalAssistantTensorPlacement::Draft => context.draft(),
        };
        let array = if retain_dimension {
            value
                .as_array()
                .try_index_device((.., row..row + 1, ..), stream)?
        } else {
            value.as_array().try_index_device((.., row, ..), stream)?
        };
        Ok(MlxTensor::from_array(array))
    }

    fn into_logits(value: Self::Tensor) -> Self::Logits {
        value.into_array()
    }

    fn sequence_suffix<'a>(
        value: &Self::Tensor,
        maximum: i32,
        context: Self::Context<'a>,
    ) -> Result<Self::Tensor, Self::Error> {
        let length = value.as_array().dim(1);
        if length <= maximum {
            Ok(value.clone())
        } else {
            value
                .as_array()
                .try_index_device((.., length - maximum.., ..), context.target())
                .map(MlxTensor::from_array)
        }
    }

    fn shared_prefix<'a>(
        value: &Self::Tensor,
        cache_len: i32,
        context: Self::Context<'a>,
    ) -> Result<Self::Tensor, Self::Error> {
        let retained = value.as_array().dim(-2).min(cache_len);
        value
            .as_array()
            .try_index_device((.., .., ..retained, ..), context.target())
            .map(MlxTensor::from_array)
    }

    fn token_prefix<'a>(
        value: &Self::Tensor,
        end: usize,
        context: Self::Context<'a>,
    ) -> Result<Self::Tensor, Self::Error> {
        let end = i32::try_from(end)
            .map_err(|_| Exception::custom("external assistant prefix exceeds i32"))?;
        value
            .as_array()
            .try_index_device((.., ..end), context.target())
            .map(MlxTensor::from_array)
    }

    fn target_tokens<'a>(
        tokens: &[u32],
        context: Self::Context<'a>,
    ) -> Result<Self::Tensor, Self::Error> {
        let width = i32::try_from(tokens.len())
            .map_err(|_| Exception::custom("external assistant token count exceeds i32"))?;
        let mut value = Array::from_slice(tokens, &[1, width]);
        if context.crosses_devices() {
            value = value.copy(context.target())?;
        }
        Ok(MlxTensor::from_array(value))
    }

    fn transfer<'a>(
        value: &Self::Tensor,
        direction: ExternalAssistantTransfer,
        context: Self::Context<'a>,
    ) -> Result<Self::Tensor, Self::Error> {
        if !context.is_split() {
            return Ok(value.clone());
        }
        if !context.crosses_devices() {
            match direction {
                ExternalAssistantTransfer::TargetToDraft => {
                    let _completion = context.wait_for_target_outputs([value.as_array()])?;
                }
                ExternalAssistantTransfer::DraftToTarget => {
                    let _completion = context.wait_for_draft_outputs([value.as_array()])?;
                }
            }
            return Ok(value.clone());
        }
        async_eval_with_event([value.as_array()])?.synchronize()?;
        let destination = match direction {
            ExternalAssistantTransfer::TargetToDraft => context.draft(),
            ExternalAssistantTransfer::DraftToTarget => context.target(),
        };
        let copied = value.as_array().copy(destination)?;
        async_eval_with_event([&copied])?.synchronize()?;
        Ok(MlxTensor::from_array(copied))
    }

    fn target_operation<'a>(
        target: &mut Self::Target,
        operation: ExternalPredictionTargetOperation<'_, Self::Tensor>,
        _context: Self::Context<'a>,
    ) -> Result<Self::Tensor, Self::Error> {
        target
            .apply_external_prediction_target_operation(operation)
            .map_err(|error| Exception::custom(error.to_string()))
    }

    fn neural_context<'a>(
        context: Self::Context<'a>,
        placement: ExternalAssistantTensorPlacement,
    ) -> &'a Stream {
        match placement {
            ExternalAssistantTensorPlacement::Target => context.target(),
            ExternalAssistantTensorPlacement::Draft => context.draft(),
        }
    }

    fn submit_completion<'a>(
        values: impl IntoIterator<Item = &'a Self::Tensor>,
    ) -> Result<Self::Completion, Self::Error>
    where
        Self::Tensor: 'a,
    {
        MlxSpeculativeCompletion::submit(values.into_iter().map(MlxTensor::as_array))
    }
}
