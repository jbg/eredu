//! Native tensor representations and conversions used by portable text execution.

use crate::{
    backend::{
        error::Error,
        nn::shared::MlxNeuralBackend,
        runtime::{cache::MlxCacheBackend, media::input},
        MlxDistributedSession,
    },
    MlxTensor,
};
use eredu_core::BackendFailure;
use safemlx::{error::Exception, Array, Stream};

/// Concrete representations supplied by MLX for the portable executable.
#[doc(hidden)]
pub struct MlxTextValues;

impl eredu_core::backend::NativeTextValues for MlxTextValues {
    type InputTensor = Array;
    type Decode = Array;
    type Scores = Array;
    type Tensor = MlxTensor;
    type InvocationTensor = Array;
    type ObserverMapping = crate::composition::observer_mapping::ArrayToTensor;
    type Context = Stream;
    type Error = Error;
    type StateError = Exception;
    type Group = crate::backend::runtime::distributed::Group;
    type InputInspector = input::MlxInputInspector;

    fn snapshot_retirement() -> Option<Box<dyn std::any::Any>> {
        Some(Box::new(crate::backend::ordinary_retirement::ReclaimOnDrop))
    }

    fn input_inspector() -> Self::InputInspector {
        input::MlxInputInspector
    }

    fn read_token_ids(tokens: &Array, _: &Stream) -> Result<Vec<u32>, Error> {
        input::host_token_ids(tokens)
    }

    type Neural = MlxNeuralBackend;
    type Cache = MlxCacheBackend;
    type Communication = MlxDistributedSession;
    type TensorInputInspector = input::MlxTensorInputInspector;
    type TensorObserver = crate::composition::observer_mapping::TensorToArray;

    fn sampling_group(
        communication: &MlxDistributedSession,
        group: eredu_core::CollectiveGroupId,
    ) -> Option<&Self::Group> {
        communication.selected_group(group)
    }

    fn authority(
        communication: &MlxDistributedSession,
    ) -> &eredu_core::communication::CommunicationAuthority {
        communication.authority()
    }
    fn input_tensor(value: &Array) -> MlxTensor {
        MlxTensor::from_array(value.clone())
    }
    fn tensor_input_inspector() -> Self::TensorInputInspector {
        input::MlxTensorInputInspector
    }
    fn decode_tensor(tokens: &Array) -> MlxTensor {
        Self::input_tensor(tokens)
    }
    fn scores(value: MlxTensor) -> Array {
        value.into_array()
    }
    fn concatenate_token_parts(parts: &[&Array], stream: &Stream) -> Result<MlxTensor, Error> {
        input::concatenate_token_parts(parts, stream)
            .map(MlxTensor::from_array)
            .map_err(Into::into)
    }
    fn prompt_tokens(value: &Array) -> Array {
        value.clone()
    }
    fn decode_prompt_tokens(value: &Array) -> Array {
        value.clone()
    }
    fn state_error(error: BackendFailure) -> Exception {
        Exception::from_source(error)
    }
    fn tensor_bytes(value: &MlxTensor) -> u64 {
        value.as_array().nbytes() as u64
    }
    fn decoder_token_ids(value: &MlxTensor) -> Result<Vec<u32>, Error> {
        input::host_token_ids(value.as_array())
    }

    type Materializer = super::MlxEmbeddedPredictionMaterializer;
    type Embedded = super::prepared_speculative::MlxEmbeddedPredictionMechanisms;

    fn settle_input(value: &MlxTensor) -> Result<(), Exception> {
        value.as_array().evaluated().map(|_| ())
    }

    fn input_failure(error: BackendFailure) -> Exception {
        Exception::from_source(error)
    }
}
