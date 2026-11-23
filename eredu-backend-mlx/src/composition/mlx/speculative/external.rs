//! MLX tensor primitives for externally conditioned prediction.

use super::{assistant::MlxAssistantBinder, MlxSpeculativeCompletion};
use crate::{
    backend::{nn::shared::MlxNeuralBackend, runtime::cache::MlxCacheBackend},
    composition::mlx::replicated_text::MlxTextValues,
    MlxTensor,
};
use eredu_core::speculative::assistant::ExternalAssistantOperators;
use safemlx::{error::Exception, ops::indexing::TryIndexOp, Array, Stream};

/// Native resource binding shared by all conditioned assistants.
pub struct MlxExternalAssistantMechanisms;

impl ExternalAssistantOperators for MlxExternalAssistantMechanisms {
    fn control_logits_estimate(
        logits: &Self::Logits,
    ) -> Option<eredu_core::execution_control::SnapshotEstimate> {
        super::state_snapshot::estimate_array(logits)
    }
    type Capture =
        crate::composition::mlx::session::bounded_capture::MlxSpeculativeCaptureMechanisms;
    type NeuralBackend = MlxNeuralBackend;
    type Binder = MlxAssistantBinder;
    fn placement_memory(
        queue: &safemlx::Stream,
    ) -> Result<eredu_core::capability::ParameterMemoryFacts, eredu_core::BackendFailure> {
        crate::composition::mlx::capability::placement_memory(queue)
    }
    type Cache = MlxCacheBackend;
    type Text = MlxTextValues;
    type Tensor = MlxTensor;
    type Logits = Array;
    type Completion = MlxSpeculativeCompletion;
    type Error = Exception;

    fn target_input(tokens: &MlxTensor) -> Array {
        tokens.as_array().clone()
    }

    fn tensor_backing_bytes(
        tensors: &[&Self::Tensor],
    ) -> Result<Option<u64>, eredu_core::BackendFailure> {
        super::memory_observation::tensor_backing(tensors)
    }

    fn bound_assistant_group<'a>(
        transport: &'a crate::backend::distributed::MlxDistributedSession,
        group: eredu_core::CollectiveGroupId,
        manifest: &eredu_core::communication::CommunicationManifest,
    ) -> Result<&'a crate::backend::runtime::distributed::Group, Self::Error> {
        transport
            .validate_selected_manifest(manifest)
            .map_err(Exception::from_source)?;
        transport.selected_group(group).ok_or_else(|| {
            Exception::custom("retained target tensor communicator has no native binding")
        })
    }

    fn failure(error: impl std::error::Error + Send + Sync + 'static) -> Exception {
        Exception::from_source(error)
    }

    fn invalid(message: String) -> Exception {
        Exception::custom(message)
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

    fn control_tensor_bytes(tensor: &Self::Tensor) -> Option<u64> {
        (tensor.as_array().nbytes() as u64)
            .checked_mul(2)?
            .checked_add(4096)
    }
    fn sequence_len(value: &Self::Tensor) -> Result<usize, Self::Error> {
        eredu_nn::sequence::length(value).map_err(Exception::from_source)
    }

    fn sequence_row(
        value: &Self::Tensor,
        row: usize,
        retain_dimension: bool,
        stream: &Stream,
    ) -> Result<Self::Tensor, Self::Error> {
        eredu_nn::sequence::row(value, row, retain_dimension, stream)
            .map_err(Exception::from_source)
    }

    fn into_logits(value: Self::Tensor) -> Self::Logits {
        value.into_array()
    }

    fn sequence_suffix(
        value: &Self::Tensor,
        maximum: i32,
        stream: &Stream,
    ) -> Result<Self::Tensor, Self::Error> {
        let length = value.as_array().dim(1);
        if length <= maximum {
            Ok(value.clone())
        } else {
            value
                .as_array()
                .try_index_device((.., length - maximum.., ..), stream)
                .map(MlxTensor::from_array)
        }
    }

    fn shared_prefix(
        value: &Self::Tensor,
        cache_len: i32,
        stream: &Stream,
    ) -> Result<Self::Tensor, Self::Error> {
        let retained = value.as_array().dim(-2).min(cache_len);
        value
            .as_array()
            .try_index_device((.., .., ..retained, ..), stream)
            .map(MlxTensor::from_array)
    }

    fn token_prefix(
        value: &Self::Tensor,
        end: usize,
        stream: &Stream,
    ) -> Result<Self::Tensor, Self::Error> {
        eredu_nn::sequence::span(value, 0..end, stream).map_err(Exception::from_source)
    }

    fn target_tokens(
        tokens: &[u32],
        destination: &Stream,
        crosses_devices: bool,
    ) -> Result<Self::Tensor, Self::Error> {
        let width = i32::try_from(tokens.len())
            .map_err(|_| Exception::custom("external assistant token count exceeds i32"))?;
        let mut value = Array::from_slice(tokens, &[1, width]);
        if crosses_devices {
            value = value.copy(destination)?;
        }
        Ok(MlxTensor::from_array(value))
    }

    fn tensor_root(value: &MlxTensor) -> &Array {
        value.as_array()
    }

    fn can_share_tensor(_: &MlxTensor, _: &Stream) -> bool {
        true
    }

    fn copy_tensor(value: &MlxTensor, destination: &Stream) -> Result<MlxTensor, Exception> {
        super::state_snapshot::copy_array(value.as_array(), destination).map(MlxTensor::from_array)
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
